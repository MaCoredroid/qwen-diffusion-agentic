#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mark/qwen_diffusion"
OUT="${ROOT}/runs/flare_agentic_phase1/two_stream_agentic_v1_from_B1000_s1024_step1000"

mkdir -p "${OUT}"
cat > "${OUT}/launch_env.txt" <<EOF
unit=qwen-flare-agentic-v1-twostream-s1024-step1000-v3
start_time=$(date -Is)
continue_from=${ROOT}/runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000
data=${ROOT}/data/flare_agentic_mix_v1_train_only
max_steps=1000
max_train_samples=1024
block_size=1024
train_bd_size=32
mode=two_stream
route=route_i
gradient_checkpointing=0
seed=20260701
data_seed=20260701
EOF

exec env \
  DATASET_DIR="${ROOT}/data/flare_agentic_mix_v1_train_only" \
  LORA_MODEL_PATH="${ROOT}/runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000" \
  MAX_STEPS=1000 \
  MAX_TRAIN_SAMPLES=1024 \
  BLOCK_SIZE=1024 \
  LEARNING_RATE=1e-5 \
  GRAD_ACCUM=1 \
  SAVE_STEPS=1000 \
  SAVE_TOTAL_LIMIT=1 \
  LORA_R=8 \
  LORA_ALPHA=16 \
  LORA_DROPOUT=0.05 \
  LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj \
  TRAIN_BD_SIZE=32 \
  SEED=20260701 \
  DATA_SEED=20260701 \
  GRADIENT_CHECKPOINTING=0 \
  LOGGING_STEPS=10 \
  FASTDLLM_FLARE_DEBUG=1000 \
  FASTDLLM_FLARE_GDN_ROUTE=route_i \
  "${ROOT}/scripts/run_flare_stage1_ab_pilot_job.sh" \
  two_stream \
  "${OUT}"
