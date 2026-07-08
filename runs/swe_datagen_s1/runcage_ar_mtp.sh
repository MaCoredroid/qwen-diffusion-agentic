#!/usr/bin/env bash
# Stage-C AR arm launcher, MTP SPECULATIVE-DECODE variant — STAGED, NOT ACTIVE.
# Byte-for-byte copy of runcage_ar_probe.sh (same snapshot @c202236, same vLLM 0.23
# in .venv-vllm, same cage/env conventions) PLUS the official Qwen3.5 MTP self-
# speculative-decode head enabled via --speculative-config. Nothing else changes.
#
# WHY THIS IS SAFE / LOSSLESS:
#   The MTP head (mtp.* — 15 tensors, one full-attention draft decoder layer, 464 MiB)
#   ships INSIDE the pinned snapshot; no separate draft checkpoint. n_predict =
#   mtp_num_hidden_layers = 1. Draft weights auto-load from the same model
#   (vllm/config/speculative.py:561 sets self.model = target model). Default
#   rejection_sample_method="standard" is probabilistic => DISTRIBUTION-PRESERVING:
#   the accepted-token stream is identical in law to plain AR decode, so teacher
#   quality is unchanged in principle. Speculative decode only trades compute for
#   wall-clock; it does not change WHAT the teacher emits.
#
# COMPOSES WITH THE GDN HYBRID:
#   Qwen3.5 is IsHybrid + HasInnerState and does NOT declare SupportsMambaPrefixCaching,
#   so with prefix-caching ON vLLM 0.23 auto-resolves mamba_cache_mode to "align"
#   (model_executor/models/config.py). The MTP draft layer is FULL-ATTENTION (adds
#   attention-KV only, no extra mamba/GDN state). We pass --mamba-cache-mode align
#   explicitly as belt-and-suspenders. Do NOT export VLLM_USE_V2_MODEL_RUNNER=1.
#   qwen3_5 is NOT special-cased to enforce_eager (only deepseek_v32 is) => cudagraph stays on.
#
# TUNING:
#   NUM_SPEC_TOKENS (default 1) = number of speculative tokens. Default==n_predict==1.
#   Any int >=1 is legal (must be divisible by n_predict=1). >1 loops the SINGLE MTP
#   layer autoregressively and vLLM logs a lower-acceptance warning — start at 1,
#   optionally probe 2 at the gate.
#
# ROLLBACK: the orchestrator selects the runcage via RUNCAGE_SCRIPT (datagen_gen.sh);
#   point it back at runcage_ar_probe.sh (the default) to revert instantly.
#
# Everything below is IDENTICAL to runcage_ar_probe.sh except the two spec-decode
# additions marked [MTP].
set -euo pipefail
cd /home/mark/qwen_diffusion
SNAP=${SNAP:-/home/mark/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}
VLLM_BIN=${VLLM_BIN:-/home/mark/qwen_diffusion/.venv-vllm/bin/vllm}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-/home/mark/shared/lumoFlyWheel_codex_fork/docker/chat_templates/qwen3-openai-codex.jinja}
PORT=${PORT:-9951}
GPU_UTIL=${GPU_UTIL:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}
# [MTP] number of speculative tokens per step (default = n_predict = 1)
NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS:-1}

export CUDA_HOME=${CUDA_HOME:-/home/mark/qwen_diffusion/.venv-vllm/lib/python3.12/site-packages/nvidia/cu13}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export VLLM_NO_USAGE_STATS=${VLLM_NO_USAGE_STATS:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

EAGER_ARGS=()
if [[ "${AR_ENFORCE_EAGER:-0}" == "1" ]]; then EAGER_ARGS=(--enforce-eager); fi

# [MTP] self-speculative-decode config. method qwen3_5_mtp auto-normalizes to "mtp"
# and pulls the draft layer from the SAME snapshot. standard (probabilistic)
# rejection sampling is the vLLM default => lossless; do NOT set "synthetic".
SPEC_CONFIG="{\"method\": \"qwen3_5_mtp\", \"num_speculative_tokens\": ${NUM_SPEC_TOKENS}}"

echo "[probe ar-mtp] snapshot=$SNAP port=$PORT gpu_util=$GPU_UTIL max_model_len=$MAX_MODEL_LEN eager=${AR_ENFORCE_EAGER:-0} spec_tokens=$NUM_SPEC_TOKENS" >&2

exec "$VLLM_BIN" serve "$SNAP" \
  --served-model-name qwen3.5-9b-ar \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype bfloat16 \
  --kv-cache-dtype auto \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs "${MAX_NUM_SEQS:-1}" \
  --gdn-prefill-backend triton \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --mamba-block-size 1024 \
  --mamba-ssm-cache-dtype float32 \
  --mamba-cache-mode align \
  --speculative-config "$SPEC_CONFIG" \
  --no-enable-flashinfer-autotune \
  "${EAGER_ARGS[@]}" \
  --chat-template "$CHAT_TEMPLATE" \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3
