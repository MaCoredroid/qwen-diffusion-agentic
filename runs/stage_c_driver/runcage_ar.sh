#!/usr/bin/env bash
# Stage-C AR arm launcher: stock AR Qwen3.5-9B (@c202236) on stock vLLM 0.23
# (.venv-vllm), cudagraph, native qwen3_xml tools, INSIDE the RAM cage. ONE heavy
# process at a time.
#
# DIFFERENCE vs runs/stage_a_smoke/runcage_ar.sh: --max-model-len is raised
# 8192 -> 32768 (override via MAX_MODEL_LEN) because a REAL SWE-Bench qwen-code
# episode needs a much larger context than the toy smoke: the first turn alone is
# ~6.1k input tokens (system prompt + tool schemas + AGENTS.md problem statement),
# and tool results accumulate across turns. The 8192 byte-cert regime 400s on
# turn 1 (input+output > 8192). Everything else identical to the Stage-A AR arm.
set -euo pipefail
cd /home/mark/qwen_diffusion
SNAP=${SNAP:-/home/mark/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}
VLLM_BIN=${VLLM_BIN:-/home/mark/qwen_diffusion/.venv-vllm/bin/vllm}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-/home/mark/shared/lumoFlyWheel_codex_fork/docker/chat_templates/qwen3-openai-codex.jinja}
PORT=${PORT:-9951}
GPU_UTIL=${GPU_UTIL:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}
# Concurrency for the batched N=50 run (continuous batching). Default 1 keeps the
# byte-cert / b=1 gate regime unchanged; the frozen N-run sets MAX_NUM_SEQS=4.
# The AR arm has NO mamba checkpoint cache and a flat ~22GB footprint at gmu 0.85,
# so 4-way KV fits with wide headroom (unlike the diffusion arm — see
# runs/loop_halt_polish/report.md, boot-probe section).
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}

export CUDA_HOME=${CUDA_HOME:-/home/mark/qwen_diffusion/.venv-vllm/lib/python3.12/site-packages/nvidia/cu13}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export VLLM_NO_USAGE_STATS=${VLLM_NO_USAGE_STATS:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

EAGER_ARGS=()
if [[ "${AR_ENFORCE_EAGER:-0}" == "1" ]]; then EAGER_ARGS=(--enforce-eager); fi

echo "[stage-c ar-serve] snapshot=$SNAP port=$PORT gpu_util=$GPU_UTIL max_model_len=$MAX_MODEL_LEN eager=${AR_ENFORCE_EAGER:-0}" >&2

exec "$VLLM_BIN" serve "$SNAP" \
  --served-model-name qwen3.5-9b-ar \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype bfloat16 \
  --kv-cache-dtype auto \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" \
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
