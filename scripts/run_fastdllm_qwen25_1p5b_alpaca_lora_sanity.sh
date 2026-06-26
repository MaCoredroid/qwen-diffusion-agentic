#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mark/qwen_diffusion"
V2="${ROOT}/fast-dllm/v2"
ENV_PY="${ROOT}/.venv-fastdllm/bin/python"
CUDA_ROOT="${ROOT}/.venv-fastdllm/lib/python3.10/site-packages/nvidia/cu13"

export CUDA_HOME="${CUDA_ROOT}"
export PATH="${CUDA_HOME}/bin:${ROOT}/.venv-fastdllm/bin:${PATH}"
export LIBRARY_PATH="${CUDA_HOME}/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DS_SKIP_CUDA_CHECK=1

cd "${V2}"

exec "${ENV_PY}" train_scripts/finetune.py \
    --model_name_or_path "${ROOT}/models/qwen2.5-1.5b-fastdllm-init" \
    --trust_remote_code 1 \
    --mdm 1 \
    --use_lora 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj \
    --dataset_path "${V2}/data/alpaca/train_conversation" \
    --output_dir "${ROOT}/runs/fastdllm_qwen25_1p5b_alpaca_lora_sanity" \
    --overwrite_output_dir \
    --conversation_template fast_dllm_v2 \
    --num_train_epochs 1 \
    --max_steps 1 \
    --max_train_samples 8 \
    --learning_rate 2e-4 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_ratio 0.03 \
    --disable_group_texts 0 \
    --block_size 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --bf16 \
    --run_name fastdllm_qwen25_1p5b_alpaca_lora_sanity \
    --validation_split_percentage 0 \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 1 \
    --dataloader_num_workers 0 \
    --preprocessing_num_workers 1 \
    --save_total_limit 1 \
    --gradient_checkpointing 1 \
    --report_to none
