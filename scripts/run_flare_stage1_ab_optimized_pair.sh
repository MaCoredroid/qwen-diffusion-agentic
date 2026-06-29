#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mark/qwen_diffusion"
cd "${ROOT}"

export MAX_STEPS="${MAX_STEPS:-200}"
export SAVE_STEPS="${SAVE_STEPS:-${MAX_STEPS}}"
export BLOCK_SIZE="${BLOCK_SIZE:-1024}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
export FASTDLLM_FLARE_DEBUG="${FASTDLLM_FLARE_DEBUG:-0}"
export LOGGING_STEPS="${LOGGING_STEPS:-10}"
export FASTDLLM_PROFILE_TRAINING_STEP="${FASTDLLM_PROFILE_TRAINING_STEP:-0}"
export FASTDLLM_PROFILE_FLARE_SECTIONS="${FASTDLLM_PROFILE_FLARE_SECTIONS:-0}"
export FASTDLLM_PROFILE_GDN_SCAN="${FASTDLLM_PROFILE_GDN_SCAN:-0}"
export FASTDLLM_COMPILE_GDN_SCAN="${FASTDLLM_COMPILE_GDN_SCAN:-0}"

echo "[$(date -Is)] START diffusion_only_A_s1024_step${MAX_STEPS}"
scripts/run_flare_stage1_ab_pilot_job.sh \
    diffusion \
    "${ROOT}/runs/flare_stage1_ab_pilot/diffusion_only_A_s1024_step${MAX_STEPS}"
echo "[$(date -Is)] DONE diffusion_only_A_s1024_step${MAX_STEPS}"

echo "[$(date -Is)] START two_stream_B_s1024_step${MAX_STEPS}"
FASTDLLM_FLARE_GDN_ROUTE=route_i \
scripts/run_flare_stage1_ab_pilot_job.sh \
    two_stream \
    "${ROOT}/runs/flare_stage1_ab_pilot/two_stream_B_s1024_step${MAX_STEPS}"
echo "[$(date -Is)] DONE two_stream_B_s1024_step${MAX_STEPS}"
