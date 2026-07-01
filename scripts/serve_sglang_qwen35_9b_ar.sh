#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Matched AR-9B baseline for the FLARE agentic eval.  Defaults intentionally
# mirror the current diffusion HF serving validation: bf16 weights, native Qwen
# tool parser, thinking disabled by callers, no teacher/checkpoint adapter.
AR_PROFILE="${PROFILE:-bf16}"

case "${AR_PROFILE}" in
  bf16)
    export PROFILE=custom
    export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-9B}"
    export QUANTIZATION="${QUANTIZATION:-}"
    export LOAD_FORMAT="${LOAD_FORMAT:-}"
    export DTYPE="${DTYPE:-bfloat16}"
    ;;
  bnb4|nf4)
    # Optional fairness profile if the diffusion side is later served through a
    # runtime 4-bit path instead of the current bf16 HF route.
    export PROFILE=custom
    export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-9B}"
    export QUANTIZATION="${QUANTIZATION:-bitsandbytes}"
    export LOAD_FORMAT="${LOAD_FORMAT:-bitsandbytes}"
    export DTYPE="${DTYPE:-bfloat16}"
    ;;
  custom)
    export PROFILE=custom
    : "${MODEL_PATH:?Set MODEL_PATH when PROFILE=custom}"
    export QUANTIZATION="${QUANTIZATION:-}"
    export LOAD_FORMAT="${LOAD_FORMAT:-}"
    export DTYPE="${DTYPE:-bfloat16}"
    ;;
  *)
    echo "Unknown PROFILE=${AR_PROFILE}; use bf16, bnb4, nf4, or custom." >&2
    exit 2
    ;;
esac

export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-30000}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-8192}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-4}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-4096}"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen}"
export REASONING_PARSER="${REASONING_PARSER:-qwen3}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-9b-ar}"
export ENABLE_MTP="${ENABLE_MTP:-0}"

exec scripts/serve_sglang_qwen36_teacher.sh
