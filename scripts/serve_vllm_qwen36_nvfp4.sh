#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"

MODEL="${MODEL:-sakamakismile/Qwen3.6-27B-NVFP4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

exec .venv-vllm/bin/vllm serve "${MODEL}" \
  --trust-remote-code \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --host "${HOST}" \
  --port "${PORT}"
