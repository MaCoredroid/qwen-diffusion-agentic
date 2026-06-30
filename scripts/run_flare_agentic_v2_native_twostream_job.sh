#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mark/qwen_diffusion"
OUT="${ROOT}/runs/flare_agentic_phase2/two_stream_native_mix_v2_from_init_s1024_step1000"
DATA_FULL="${ROOT}/data/flare_agentic_mix_v2_native"
DATA="${ROOT}/data/flare_agentic_mix_v2_native_train_only"
UNIT="qwen-flare-agentic-v2-native-twostream-s1024-step1000"

mkdir -p "${OUT}"
rm -rf "${DATA}"
mkdir -p "${DATA}"
ln -s "../flare_agentic_mix_v2_native/train_agentic_mix.json" "${DATA}/train_agentic_mix.json"

cat > "${OUT}/launch_env.txt" <<EOF
unit=${UNIT}
start_time=$(date -Is)
start_point=fresh_from_init
base_model=${ROOT}/models/qwen3.5-9b-fastdllm-init
lora_model_path=
data=${DATA}
data_source=${DATA_FULL}
data_manifest=${DATA_FULL}/manifest.json
max_steps=1000
max_train_samples=1024
block_size=1024
train_bd_size=32
mode=two_stream
route=route_i
gradient_checkpointing=0
seed=20260704
data_seed=20260704
native_format=qwen_native_function_parameter
EOF

exec env \
  DATASET_DIR="${DATA}" \
  LORA_MODEL_PATH="" \
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
  SEED=20260704 \
  DATA_SEED=20260704 \
  GRADIENT_CHECKPOINTING=0 \
  LOGGING_STEPS=10 \
  FASTDLLM_FLARE_DEBUG=1000 \
  FASTDLLM_FLARE_GDN_ROUTE=route_i \
  "${ROOT}/scripts/run_flare_stage1_ab_pilot_job.sh" \
  two_stream \
  "${OUT}"
