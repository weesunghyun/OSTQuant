#!/usr/bin/env sh
export HF_DATASETS_TRUST_REMOTE_CODE=1
export HF_HOME="/app/data/.cache/huggingface/hub/"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

model_pairs=(
    # Base and instruct pairs
    # "Meta-Llama-3-8B-Instruct Meta-Llama-3-8B"
    "Llama-3.1-8B-Instruct Meta-Llama-3-8B"
    # "Llama-3.1-8B-Instruct Meta-Llama-3.1-8B"
    # "Llama-2-7b-chat-hf Llama-2-7b-hf"
    # "Llama-2-13b-chat-hf Llama-2-13b-hf"
    # "Llama-2-70b-chat-hf Llama-2-70b-hf"
)

# adapters=("nvidia/llama-3.1-nemoguard-8b-content-safety")

combinations=(
    "4 4 4"
    # "8 8 8"
    # "4 16 16"
)

losses=(
    # "reference"
    # "rtn"
    # "gptq"
    # "origin"
    # "mse"
    # "kl"
    # "kl_top"
    # "contrastive_kl"
    "contrastive_kl_top"
    # "contrastive_kl_exp0"
)

contrastive_loss_weights=(
    # 0.0
    # 0.01
    # 0.05
    # 0.1
    # 0.2
    0.5
    # 0.75
    # 1.0
)

max_steps=100
gpu_id=7
output_dir_base="output"
# load_qmodel_path=output_0507/Llama-3.1-8B-Instruct_w4a4kv4_contrastive_kl_top_0.0/checkpoint_quantized.pth

for model_pair in "${model_pairs[@]}"; do
    read -r target_model base_model <<< "$model_pair"
    
    input_model="meta-llama/$target_model"
    pretrained_model="meta-llama/$base_model"

    for combo in "${combinations[@]}"; do
        read -r w_bit a_bit kv_bit <<< "$combo"

        for loss in "${losses[@]}"; do

            if [[ $loss == *"contrastive"* ]]; then
                for contrastive_loss_weight in "${contrastive_loss_weights[@]}"; do
                    output_dir="${output_dir_base}/${target_model}_w${w_bit}a${a_bit}kv${kv_bit}_${loss}_${contrastive_loss_weight}"
                    echo $output_dir

                    resume_path="${output_dir}/model.bin"

                    CUDA_VISIBLE_DEVICES=${gpu_id} torchrun --nproc_per_node=1 --master_port=8902 main.py \
                    --output_dir "${output_dir}" --model "${input_model}" \
                    --train_rotate=False --resume_path="${resume_path}" \
                    --loss_type="${loss}" --contrastive_loss_weight="${contrastive_loss_weight}" --post_attn=True \
                    --rotate_ov=True --rotate_post_rope=False --online_qk_hadamard=True --smooth_qk=True --smooth_ov=True --smooth_up_down=True --smooth_norm_linear=True \
                    --bf16=True --test_static=False --lm_eval=True --per_device_train_batch_size=4 --per_device_eval_batch_size=4 \
                    --max_steps=${max_steps} --w_bits="${w_bit}" --a_bits="${a_bit}" --v_bits="${kv_bit}" --k_bits="${kv_bit}" --down_bits="${a_bit}" \
                    --train_enable_wquant=False --sub_mean False  --distribute=True --use_klt

                done
            elif [[ $loss == *"reference"* ]]; then
                output_dir="${output_dir_base}/${target_model}_w${w_bit}a${a_bit}kv${kv_bit}_${loss}"
                echo $output_dir

                CUDA_VISIBLE_DEVICES=${gpu_id} torchrun --nproc_per_node=1 --master_port=8902 main.py \
                --output_dir "${output_dir}" --model "${input_model}" \
                --pre_eval=True \
                --rotate False --w_gptq=False --force_clip=False \
                --bf16=True --test_static=False --lm_eval=True --per_device_train_batch_size=4 --per_device_eval_batch_size=4 \
                --max_steps=${max_steps} --w_bits="${w_bit}" --a_bits="${a_bit}" --v_bits="${kv_bit}" --k_bits="${kv_bit}" --down_bits="${a_bit}" \
                --train_enable_wquant=False --sub_mean False  --distribute=True --use_klt  
                
            elif [[ $loss == *"rtn"* ]]; then
                output_dir="${output_dir_base}/${target_model}_w${w_bit}a${a_bit}kv${kv_bit}_${loss}"
                echo $output_dir

                CUDA_VISIBLE_DEVICES=${gpu_id} torchrun --nproc_per_node=1 --master_port=8902 main.py \
                --output_dir "${output_dir}" --model "${input_model}" \
                --rotate False --w_gptq=False --force_clip=False \
                --bf16=True --test_static=False --lm_eval=True --per_device_train_batch_size=4 --per_device_eval_batch_size=4 \
                --max_steps=${max_steps} --w_bits="${w_bit}" --a_bits="${a_bit}" --v_bits="${kv_bit}" --k_bits="${kv_bit}" --down_bits="${a_bit}" \
                --train_enable_wquant=False --sub_mean False  --distribute=True --use_klt  
                
            elif [[ $loss == *"gptq"* ]]; then 
                output_dir="${output_dir_base}/${target_model}_w${w_bit}a${a_bit}kv${kv_bit}_${loss}"
                echo $output_dir

                CUDA_VISIBLE_DEVICES=${gpu_id} torchrun --nproc_per_node=1 --master_port=8902 main.py \
                --output_dir "${output_dir}" --model "${input_model}" \
                --rotate False \
                --bf16=True --test_static=False --lm_eval=True --per_device_train_batch_size=4 --per_device_eval_batch_size=4 \
                --max_steps=${max_steps} --w_bits="${w_bit}" --a_bits="${a_bit}" --v_bits="${kv_bit}" --k_bits="${kv_bit}" --down_bits="${a_bit}" \
                --train_enable_wquant=False --sub_mean False  --distribute=True --use_klt  
                
            else
                output_dir="${output_dir_base}/${target_model}_w${w_bit}a${a_bit}kv${kv_bit}_${loss}"
                echo $output_dir

                resume_path="${output_dir}/model.bin"

                CUDA_VISIBLE_DEVICES=${gpu_id} torchrun --nproc_per_node=1 --master_port=8902 main.py \
                --output_dir "${output_dir}" --model "${input_model}" \
                --train_rotate=False --resume_path="${resume_path}" \
                --loss_type="${loss}" --post_attn=True \
                --rotate_ov=True --rotate_post_rope=False --online_qk_hadamard=True --smooth_qk=True --smooth_ov=True --smooth_up_down=True --smooth_norm_linear=True \
                --bf16=True --test_static=False --lm_eval=True --per_device_train_batch_size=4 --per_device_eval_batch_size=4 \
                --max_steps=${max_steps} --w_bits="${w_bit}" --a_bits="${a_bit}" --v_bits="${kv_bit}" --k_bits="${kv_bit}" --down_bits="${a_bit}" \
                --train_enable_wquant=False --sub_mean False  --distribute=True --use_klt
            fi
        done
    done
done