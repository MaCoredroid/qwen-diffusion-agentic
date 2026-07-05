#!/usr/bin/env bash
# Stage-A smoke: serve the STOCK AR Qwen3.5-9B (@c202236 HF snapshot) on stock
# vLLM 0.23 (.venv-vllm) as the AR comparison arm (:9951), INSIDE the RAM cage.
# ONE heavy process at a time.
#
# Standard `vllm serve` with cudagraph (NOT enforce-eager) per the Stage-A spec
# AR-arm directive. Native qwen3_xml tool parser + the SAME codex chat template
# as the diffusion arm ([[native-function-format-rule]]) so both arms drive
# qwen-code identically. FR13 align-class APC on. Set AR_ENFORCE_EAGER=1 to fall
# back to eager if cudagraph capture fails to boot on this GDN-hybrid / sm_120.
set -euo pipefail
cd /home/mark/qwen_diffusion
SNAP=${SNAP:-/home/mark/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}
VLLM_BIN=${VLLM_BIN:-/home/mark/qwen_diffusion/.venv-vllm/bin/vllm}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-/home/mark/shared/lumoFlyWheel_codex_fork/docker/chat_templates/qwen3-openai-codex.jinja}
PORT=${PORT:-9951}
GPU_UTIL=${GPU_UTIL:-0.80}

export CUDA_HOME=${CUDA_HOME:-/home/mark/qwen_diffusion/.venv-vllm/lib/python3.12/site-packages/nvidia/cu13}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export VLLM_NO_USAGE_STATS=${VLLM_NO_USAGE_STATS:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

EAGER_ARGS=()
if [[ "${AR_ENFORCE_EAGER:-0}" == "1" ]]; then EAGER_ARGS=(--enforce-eager); fi

echo "[ar-stock-serve] snapshot=$SNAP port=$PORT gpu_util=$GPU_UTIL eager=${AR_ENFORCE_EAGER:-0}" >&2

exec "$VLLM_BIN" serve "$SNAP" \
  --served-model-name qwen3.5-9b-ar \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype bfloat16 \
  --kv-cache-dtype auto \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len 8192 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 1 \
  --gdn-prefill-backend triton \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --mamba-block-size 1024 \
  --mamba-ssm-cache-dtype float32 \
  --no-enable-flashinfer-autotune \
  "${EAGER_ARGS[@]}" \
  --chat-template "$CHAT_TEMPLATE" \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3
