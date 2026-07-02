#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mark/qwen_diffusion"

export MODEL_PATH="${ROOT}/models/qwen3.5-9b-fastdllm-init"
export DATASET_DIR="${ROOT}/data/flare_redesign_run1_copy_retention_mix"
export OUTPUT_DIR="${ROOT}/runs/s1c_run1_envelope_2000_qwen35_9b"

# Keep the Run-1 data bytes fixed for the single-variable step-budget probe.
export SKIP_DATASET_BUILD=1

export MAX_STEPS=2000
export MAX_TRAIN_SAMPLES=5055
export BLOCK_SIZE=512
export TRAIN_BD_SIZE=32
export GRAD_ACCUM=1
export LEARNING_RATE=1e-5
export LR_SCHEDULER_TYPE=cosine
export WARMUP_RATIO=0.03
export SAVE_STEPS=200
export SAVE_TOTAL_LIMIT=20
export LOGGING_STEPS=5

export LORA_R=16
export LORA_ALPHA=32
export LORA_DROPOUT=0.05
export LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj"

export OVERWRITE_CACHE=1
export DISABLE_GROUP_TEXTS=0
export TRUNCATION_SIDE=left
export CONVERSATION_TEMPLATE=fast_dllm_v2_native
export VALUE_SPAN_LOSS_WEIGHT=2.0
export VALUE_SPAN_MASK_PROB=1.0
export SEED=71101
export DATA_SEED=71101

export FASTDLLM_FLARE_TWO_STREAM=1
export FLARE_TWO_STREAM=1
export FASTDLLM_FLARE_GDN_ROUTE=route_i
export FASTDLLM_FLARE_MASK_RATE_MIN=0.3
export FASTDLLM_FLARE_MASK_RATE_MAX=0.8
export FASTDLLM_FLARE_ADAPTIVE_COPY_SCHEDULE=1
export FASTDLLM_FLARE_HIGH_ENTROPY_MASK_RATE_MIN=0.02
export FASTDLLM_FLARE_HIGH_ENTROPY_MASK_RATE_MAX=0.12
export FASTDLLM_BATCH_FLARE_NOISY_GDN=1
export FASTDLLM_OPTIMIZE_FLARE_CLEAN_GDN=1
export FASTDLLM_GDN_KERNEL=fla
export FASTDLLM_FLARE_DEBUG=2
export FASTDLLM_DEBUG_ARGUMENT_SPAN_LOSS=2

unset LORA_MODEL_PATH
unset LR_SCHEDULER_KWARGS
unset WARMUP_STEPS
unset FASTDLLM_TRAIN_GPU_METRICS_JSON

exec "${ROOT}/scripts/run_flare_redesign_run1.sh"
