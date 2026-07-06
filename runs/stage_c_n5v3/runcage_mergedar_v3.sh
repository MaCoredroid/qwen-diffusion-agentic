#!/usr/bin/env bash
# v3 merged-AR runcage: BYTE-IDENTICAL to runs/stage_c_driver/runcage_mergedar.sh
# EXCEPT the `--override-generation-config '{"temperature": 0.0}'` line is REMOVED.
# In v3 every arm runs at the reference envelope (temp 0.6 / top_p 0.95 / top_k 20
# / seeded) which is FORCED per-request by the proxy (LUMO_PROXY_FORCE_*). Dropping
# the server-side temp-0 default makes the proxy envelope the SOLE sampling control
# on this arm -- uniform with the stock-AR arm (runcage_ar.sh, which also has no
# override) and free of any request-vs-generation_config precedence ambiguity.
# Same rlv2 weights served as PLAIN AR on stock vLLM 0.23 (.venv-vllm).
set -euo pipefail
cd /home/mark/qwen_diffusion
SNAP=${SNAP:-/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16}
VLLM_BIN=${VLLM_BIN:-/home/mark/qwen_diffusion/.venv-vllm/bin/vllm}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-/home/mark/shared/lumoFlyWheel_codex_fork/docker/chat_templates/qwen3-openai-codex.jinja}
PORT=${PORT:-9953}
GPU_UTIL=${GPU_UTIL:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}
SERVED_NAME=${SERVED_NAME:-qwen3.5-9b-mergedar}

export CUDA_HOME=${CUDA_HOME:-/home/mark/qwen_diffusion/.venv-vllm/lib/python3.12/site-packages/nvidia/cu13}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export VLLM_NO_USAGE_STATS=${VLLM_NO_USAGE_STATS:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

EAGER_ARGS=()
if [[ "${AR_ENFORCE_EAGER:-0}" == "1" ]]; then EAGER_ARGS=(--enforce-eager); fi

echo "[stage-c n5v3 mergedar-serve] snapshot=$SNAP port=$PORT gpu_util=$GPU_UTIL max_model_len=$MAX_MODEL_LEN NO-temp-override(envelope via proxy) served=$SERVED_NAME eager=${AR_ENFORCE_EAGER:-0}" >&2

exec "$VLLM_BIN" serve "$SNAP" \
  --served-model-name "$SERVED_NAME" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype bfloat16 \
  --kv-cache-dtype auto \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
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
