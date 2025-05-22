import transformers, torch, os, datasets, random
import torch.nn.functional as F, torch, torch.nn as nn
import geoopt
from quant.cayley_opt import SGDG
from quant.ost_model_utils import SmoothModule, RotateModule

import torch.distributed.fsdp as fsdp
from transformers.trainer_utils import IntervalStrategy
from transformers.utils import logging
logger = logging.get_logger(__name__)

fsdp.FullyShardedDataParallel


class MyTrainer(transformers.Trainer):
    def __init__(self, pretrained_model=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (
            hasattr(self.accelerator.state, "fsdp_plugin")
            and self.accelerator.state.fsdp_plugin is not None
        ):
            model: nn.Module = self.model
            ignored_modules = list()
            for m in model.modules():
                if isinstance(m, (RotateModule, SmoothModule)):
                    ignored_modules.append(m)
            self.accelerator.state.fsdp_plugin.ignored_modules = ignored_modules
            self.accelerator.state.fsdp_plugin.use_orig_params = True
        
        self.pretrained_model = None
        if pretrained_model is not None:
            self.pretrained_model = pretrained_model.to(self.args.device).eval()
            for param in self.pretrained_model.parameters():
                param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False):
        torch.cuda.empty_cache()

        args = self.args
        loss_type = args.loss_type
        contrastive_loss_weight = args.contrastive_loss_weight

        if loss_type == "origin":
            return super().compute_loss(model, inputs, return_outputs)

        if loss_type == "contrastive_kl":
            # if return_outputs:
            #     breakpoint()
                
            labels = inputs.pop("labels", None)
            # Get logits from original model
            ori_logits = self.get_ori_outputs(model, inputs).logits.detach()
            
            # Get logits from current model
            outputs = model(**inputs)
            logits = outputs.logits

            torch.cuda.empty_cache()

            # Calculate KL divergence between original and current model
            kl_loss = F.kl_div(
                F.log_softmax(logits.flatten(0, -2), dim=-1),
                F.softmax(ori_logits, dim=-1).flatten(0, -2),
                reduction="batchmean",
            )
            # kl_loss = F.kl_div(
            #     F.log_softmax(logits.flatten(0, -2) / self.contrastive_temp, dim=-1),
            #     F.softmax(ori_logits.flatten(0, -2) / self.contrastive_temp, dim=-1),
            #     reduction="batchmean",
            # )
            # Clear original logits from memory
            del ori_logits

            # Get logits from additional pre-trained model
            if self.pretrained_model is not None:
                with torch.no_grad():
                    # print(inputs)
                    # breakpoint()
                    
                    pretrain_logits = (self.pretrained_model(**inputs)).logits.detach()
            else:
                pretrain_logits = None

            # Calculate negative contrastive loss using pretrained model
            # Negative KL divergence pushes the distributions apart
            contrastive_loss = F.kl_div(
                F.log_softmax(logits.flatten(0, -2), dim=-1),
                F.softmax(pretrain_logits, dim=-1).flatten(0, -2),
                reduction="batchmean",
            )
            # Clear pretrained logits from memory
            del pretrain_logits

            # Combine losses with a weighting factor (can be adjusted)
            # Positive KL loss pulls towards original model
            # Negative contrastive loss pushes away from pretrained model
            loss = kl_loss - contrastive_loss_weight * contrastive_loss
            
            torch.cuda.empty_cache()

            return (loss, outputs) if return_outputs else loss

        if "contrastive_kl_top" in loss_type:
            labels = inputs.pop("labels", None)
            k = 1000
            if loss_type == "contrastive_kl_top":
                k_contrastive = 1000 
            else:
                k_contrastive = int(loss_type.split("_")[-1])
            # Get logits from original model
            ori_logits = self.get_ori_outputs(model, inputs).logits.detach()
            
            # Get logits from current model
            outputs = model(**inputs)
            logits = outputs.logits

            torch.cuda.empty_cache()

            top_ori_logits, indices = ori_logits.topk(k, dim=-1, sorted=False)
            if args.post_attn:
                ref = F.softmax(ori_logits,dim=-1).gather(-1,indices).flatten(0,-2)
                can = F.log_softmax(logits,dim=-1).gather(-1,indices).flatten(0,-2)
                kl_loss = F.kl_div(can,ref,reduction="batchmean")
            else:
                top_logits = logits.gather(-1, indices)
                kl_loss = F.kl_div(
                    F.log_softmax(top_logits, dim=-1).flatten(0, -2),
                    F.softmax(top_ori_logits, dim=-1).flatten(0, -2),
                    reduction="batchmean",
                )

            del ori_logits

            if self.pretrained_model is not None:
                with torch.no_grad():
                    pretrain_logits = (self.pretrained_model(**inputs)).logits.detach()
            else:
                pretrain_logits = None

            top_pretrain_logits, indices = pretrain_logits.topk(k_contrastive, dim=-1, sorted=False)
            if args.post_attn:
                ref = F.softmax(pretrain_logits,dim=-1).gather(-1,indices).flatten(0,-2)
                can = F.log_softmax(logits,dim=-1).gather(-1,indices).flatten(0,-2)
                contrastive_loss = F.kl_div(can,ref,reduction="batchmean")
            else:
                top_logits = logits.gather(-1, indices)
                contrastive_loss = F.kl_div(
                    F.log_softmax(top_logits, dim=-1).flatten(0, -2),
                    F.softmax(top_pretrain_logits, dim=-1).flatten(0, -2),
                    reduction="batchmean",
                )

            del pretrain_logits

            # Combine losses with a weighting factor (can be adjusted)
            # Positive KL loss pulls towards original model
            # Negative contrastive loss pushes away from pretrained model
            loss = kl_loss - contrastive_loss_weight * contrastive_loss
            
            torch.cuda.empty_cache()

            return (loss, outputs) if return_outputs else loss


        if "contrastive_kl_exp0" in loss_type:
            labels = inputs.pop("labels", None)
            k = 1000 
            if loss_type == "contrastive_kl_exp0":
                k_contrastive = 1000
            else:
                k_contrastive = int(loss_type.split("_")[-1])
            # Get logits from original model
            ori_logits = self.get_ori_outputs(model, inputs).logits.detach()
            
            # Get logits from current model
            outputs = model(**inputs)
            logits = outputs.logits

            torch.cuda.empty_cache()

            top_ori_logits, indices = ori_logits.topk(k, dim=-1, sorted=False)
            if args.post_attn:
                ref = F.softmax(ori_logits,dim=-1).gather(-1,indices).flatten(0,-2)
                can = F.log_softmax(logits,dim=-1).gather(-1,indices).flatten(0,-2)
                kl_loss = F.kl_div(can,ref,reduction="batchmean")
            else:
                top_logits = logits.gather(-1, indices)
                kl_loss = F.kl_div(
                    F.log_softmax(top_logits, dim=-1).flatten(0, -2),
                    F.softmax(top_ori_logits, dim=-1).flatten(0, -2),
                    reduction="batchmean",
                )


            if self.pretrained_model is not None:
                with torch.no_grad():
                    pretrain_logits = (self.pretrained_model(**inputs)).logits.detach()
            else:
                pretrain_logits = None
            
            # Compute absolute difference between original and pretrained logits
            diff = (ori_logits - pretrain_logits).abs()
            _, indices = diff.topk(k_contrastive, dim=-1, largest=True, sorted=False)

            if args.post_attn:
                ref = F.softmax(pretrain_logits,dim=-1).gather(-1,indices).flatten(0,-2)
                can = F.log_softmax(logits,dim=-1).gather(-1,indices).flatten(0,-2)
                contrastive_loss = F.kl_div(can,ref,reduction="batchmean")
            else:
                top_logits = logits.gather(-1, indices)
                top_pretrain_logits = pretrain_logits.gather(-1, indices)
                contrastive_loss = F.kl_div(
                    F.log_softmax(top_logits, dim=-1).flatten(0, -2),
                    F.softmax(top_pretrain_logits, dim=-1).flatten(0, -2),
                    reduction="batchmean",
                )

            del ori_logits, pretrain_logits

            # Combine losses with a weighting factor (can be adjusted)
            # Positive KL loss pulls towards original model
            # Negative contrastive loss pushes away from pretrained model
            loss = kl_loss - contrastive_loss_weight * contrastive_loss
            
            torch.cuda.empty_cache()

            return (loss, outputs) if return_outputs else loss


        if loss_type == "rkl":
            labels = inputs.pop("labels", None)
            ori_logits = self.get_ori_outputs(model, inputs).logits
            outputs = model(**inputs)
            logits = outputs.logits
            loss = F.kl_div(
                F.log_softmax(ori_logits.flatten(0, -2), dim=-1),
                F.softmax(logits, dim=-1).flatten(0, -2),
                reduction="batchmean",
            )
            return (loss, outputs) if return_outputs else loss
        if loss_type == "kl":
            labels = inputs.pop("labels", None)
            ori_logits = self.get_ori_outputs(model, inputs).logits
            outputs = model(**inputs)
            logits = outputs.logits
            loss = F.kl_div(
                F.log_softmax(logits.flatten(0, -2), dim=-1),
                F.softmax(ori_logits, dim=-1).flatten(0, -2),
                reduction="batchmean",
            )
            return (loss, outputs) if return_outputs else loss

        if (
            "r_kl_top" in loss_type
        ):  
            labels = inputs.pop("labels", None)
            if loss_type == "k_top":
                k = 1000
            else:
                k = int(loss_type.split("_")[-1])
            ori_logits = self.get_ori_outputs(model, inputs).logits
            outputs = model(**inputs)
            logits = outputs.logits
            top_logits, indices = logits.topk(k, dim=-1, sorted=False)
            top_ori_logits = ori_logits.gather(-1, indices)
            loss = F.kl_div(
                F.log_softmax(top_ori_logits.flatten(0, -2), dim=-1),
                F.softmax(top_logits.flatten(0, -2), dim=-1),
                reduction="batchmean",
            )
            return (loss, outputs) if return_outputs else loss

        if "kl_top" in loss_type:
            labels = inputs.pop("labels", None)
            if loss_type == "kl_top":
                k = 1000 
            else:
                k = int(loss_type.split("_")[-1])
            ori_logits = self.get_ori_outputs(model, inputs).logits
            outputs = model(**inputs)
            logits = outputs.logits
            top_ori_logits, indices = ori_logits.topk(k, dim=-1, sorted=False)
            if args.post_attn:
                ref = F.softmax(ori_logits,dim=-1).gather(-1,indices).flatten(0,-2)
                can = F.log_softmax(logits,dim=-1).gather(-1,indices).flatten(0,-2)
                loss = F.kl_div(can,ref,reduction="batchmean")
            else:
                top_logits = logits.gather(-1, indices)
                loss = F.kl_div(
                    F.log_softmax(top_logits, dim=-1).flatten(0, -2),
                    F.softmax(top_ori_logits, dim=-1).flatten(0, -2),
                    reduction="batchmean",
                )
            return (loss, outputs) if return_outputs else loss

        # if loss_type == "mse":
        #     outputs = model(**inputs)
        #     logits = outputs["logits"]
        #     predict = torch.gather(logits, -1, indices.squeeze(1))
        #     loss = F.mse_loss(predict, values.squeeze(1))
        #     return (loss, outputs) if return_outputs else loss
        # elif loss_type == "kl":
        #     outputs = model(**inputs)
        #     logits = outputs["logits"]
        #     predict = torch.gather(logits, -1, indices.squeeze(1))
        #     predict = F.softmax(predict, dim=-1)
        #     values = F.softmax(values.squeeze(1), dim=-1)

        #     predict_dist = torch.distributions.Categorical(probs=predict)
        #     values_dist = torch.distributions.Categorical(probs=values)

        #     loss = torch.distributions.kl_divergence(predict_dist, values_dist).mean()
        #     return (loss, outputs) if return_outputs else loss
        # elif loss_type == "rkl":
        #     ori_logits = self.get_ori_outputs(model, inputs).logits
        #     outputs = model(**inputs)
        #     logits = outputs.logits

        #     loss = F.kl_div(
        #         F.log_softmax(ori_logits.flatten(0, -2), dim=-1),
        #         F.softmax(logits, dim=-1).flatten(0, -2),
        #         reduction="batchmean",
        #     )
        #     return (loss, outputs) if return_outputs else loss

        if loss_type == "mse":
            labels = inputs.pop("labels", None)
            ori_logits = self.get_ori_outputs(model, inputs).logits
            outputs = model(**inputs)
            logits = outputs.logits
            loss = F.mse_loss(logits, ori_logits)
            return (loss, outputs) if return_outputs else loss
        if loss_type == "kd":
            ori_logits = self.get_ori_outputs(model, inputs).logits
            outputs = model(**inputs)
            logits = outputs.logits
            T, alpha = self.temperature, self.loss_alpha
            ori_loss = outputs["loss"]
            logits = logits.view(-1, logits.size(-1))
            ori_logits = ori_logits.view(-1, ori_logits.size(-1))
            distill_loss = F.kl_div(
                F.log_softmax(logits / T, dim=-1).flatten(0, -2),
                F.softmax(ori_logits / T, dim=-1).flatten(0, -2),
                reduction="batchmean",
            )
            loss = ori_loss * (1 - alpha) + distill_loss * (alpha * T * T)
            return (loss, outputs) if return_outputs else loss

    @torch.no_grad()
    def get_ori_outputs(self, model, inputs):
        args = self.args
        # Create a new dictionary with only the keys we want to pass to the model
        model_inputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }
        acc = self.accelerator

        
        def set_temporary(model, temporary=True):
            model.temporary = temporary
            model.model.temporary = temporary
            model.model.embed_tokens.temporary = temporary
            model.lm_head.temporary = temporary
            for layer in model.model.layers:
                layer.set_temporary(temporary)

        def set_quant_state(
            model,
            use_weight_quant: bool = False,
            use_act_quant: bool = False,
            use_fully_quant: bool = False,
        ):
            model.model.norm.use_act_quant = (
                use_fully_quant
            )
            model.model.embed_tokens.use_act_quant = (
                use_fully_quant
            )
            for layer in model.model.layers:
                layer.set_quant_state(use_weight_quant, use_act_quant, use_fully_quant)

        set_temporary(acc.unwrap_model(model), False)
        set_quant_state(acc.unwrap_model(model), False, False, False)
        outputs = model(**model_inputs, output_hidden_states=True)
        set_temporary(acc.unwrap_model(model), True)
        set_quant_state(
            acc.unwrap_model(model), args.train_enable_wquant, True, args.fully_quant
        )
        return outputs

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        
        
        args = self.args
        params_rotate = []
        params_smooth = []
        for param in self.model.parameters():
            param: torch.nn.Parameter
            if param.requires_grad:
                if len(param.size()) == 2:
                    params_rotate.append(param)
                else:
                    params_smooth.append(param)
        dict_rotate = {
            "params": params_rotate,
            "lr": args.rotate_lr,
            "momentum": args.rotate_momentom,
            "stiefel": True,
            "grassmann": True,
            "omega": 0.1,
        }
        dict_smooth = {
            "params": params_smooth,
            "lr": args.smooth_lr,
            "momentum": args.smooth_momentom,
            "stiefel": False,
            "nesterov": False,
        }
        if args.opt_type == "SGDG":
            optimizer = SGDG(
                [dict_rotate, dict_smooth], weight_decay=0
            )  
        elif args.opt_type == "RSGD":
            optimizer = geoopt.optim.RiemannianSGD(
                [dict_rotate, dict_smooth], weight_decay=0, lr=args.rotate_lr,stabilize=10,
            )
        elif args.opt_type == "RAdam":
            optimizer = geoopt.optim.RiemannianAdam(
                [dict_rotate, dict_smooth], weight_decay=0, lr=args.rotate_lr,stabilize=10
            )
        self.optimizer = optimizer
        
        self.create_scheduler(
            num_training_steps=num_training_steps,
            optimizer=optimizer,
        )
