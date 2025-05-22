import transformers, torch, os, datasets, random,utils
import torch.nn.functional as F, torch, torch.nn as nn
import utils.data_utils as data_utils
import geoopt
from quant.ost_model_utils import LM
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainerCallback,
    default_data_collator,
)
from accelerate.hooks import remove_hook_from_module
from utils.data_utils import CustomJsonDataset, group_texts
from datasets import Dataset, IterableDataset
from quant.cayley_opt import SGDG
from torch.optim import lr_scheduler
from accelerate import DistributedType
from quant.trainer import MyTrainer
from loguru import logger
from utils import distribute_model


def rotate_smooth_train(args, lm: LM):

    logger.info("train rotate model")
    if args.smooth_up_down:
        logger.info("train smooth up down")
    if args.smooth_up_gate:
        logger.info("train smooth up gate")
    if args.smooth_qk:
        logger.info("train smooth qk")
    if args.smooth_ov:
        logger.info("train smooth ov")
    if args.smooth_norm_linear:
        logger.info("smooth norm linear")

    lm.model.config.use_cache = False
    train_dataset, eval_dataset = get_train_eval_dataset(args, lm.tokenizer)
    if lm.tokenizer.pad_token is None:
        lm.tokenizer.pad_token = lm.tokenizer.eos_token
    param_keys = get_param_keys(lm.model)
    utils.cleanup_memory()
    if args.train_distribute:
        distribute_model(lm.model)
    trainer = MyTrainer(
        model=lm.model,
        tokenizer=lm.tokenizer,
        pretrained_model=lm.pretrained_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        
        args=args,
    )
    trainer.train()
    acc = trainer.accelerator
    st = {k: v for k, v in (acc.get_state_dict(trainer.model)).items() if k in param_keys}
    acc.wait_for_everyone()
    if acc.is_main_process:
        torch.save(st, f"{args.output_dir}/model.bin")
    else:
        print(f"sub process{acc.process_index} exit")
        exit(0)
    if acc.distributed_type == DistributedType.FSDP:
        print("reloading lm")
        new_lm = LM(args)
        new_lm.fuse_layer_norms()
        new_lm.generate_rotate_parameters(args)
        new_lm.model.load_state_dict(st, strict=False)
        return new_lm
    else:
        lm.model = acc.unwrap_model(trainer.model)
        if args.train_distribute:
            remove_hook_from_module(lm.model)
        return lm


def get_train_eval_dataset(args, tokenizer):
    cache_dir = "./cache/" + args.model.split("/")[-1] + "_".join(["tokenized", args.train_dataset])
    
    
    
    
    
    
    
    
    
    if os.path.exists(cache_dir):
        tokenized_datasets = datasets.load_from_disk(cache_dir)
    else:
        if args.train_dataset == "wikitext2":
            train_dataset = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")

        def tokenize_function(examples):
            return tokenizer(examples["text"])

        tokenized_datasets = train_dataset.map(
            tokenize_function,
            batched=True,
        )
        grouped_datasets = group_texts(2048, tokenized_datasets)
        tokenized_datasets = Dataset.from_dict(grouped_datasets)
        tokenized_datasets.save_to_disk(cache_dir)
    test_loader = data_utils.get_loaders(
        args.eval_dataset, seed=args.seed, model=args.model, seqlen=2048, eval_mode=True
    )
    nsample = test_loader["input_ids"].numel() // 2048
    input_ids = test_loader["input_ids"].reshape(-1)[: nsample * 2048]
    eval_dataset = Dataset.from_dict(dict(input_ids=input_ids.split(2048, dim=-1)))

    def f(examples):
        examples["labels"] = examples["input_ids"]
        return examples

    eval_dataset = eval_dataset.map(f)
    return tokenized_datasets, eval_dataset


def get_param_keys(model):
    keys = list()
    for k, v in model.named_parameters():
        if v.requires_grad:
            keys.append(k)
    return keys
