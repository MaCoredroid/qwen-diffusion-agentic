#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

SGLANG_PYTHON="${SGLANG_PYTHON:-.venv-sglang/bin/python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
PROFILE="${PROFILE:-nvfp4}"
SGLANG_MIN_VERSION="${SGLANG_MIN_VERSION:-0.5.10}"
SKIP_VERSION_CHECK="${SKIP_VERSION_CHECK:-0}"

case "${PROFILE}" in
  fp8)
    MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.6-27B-FP8}"
    QUANTIZATION="${QUANTIZATION:-fp8}"
    KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
    ;;
  nvfp4|fp4|q4)
    MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.6-27B-NVFP4}"
    QUANTIZATION="${QUANTIZATION:-petit_nvfp4}"
    KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
    ;;
  custom)
    MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH for PROFILE=custom}"
    QUANTIZATION="${QUANTIZATION:-}"
    KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
    ;;
  *)
    echo "Unknown PROFILE=${PROFILE}; use fp8, nvfp4, or custom." >&2
    exit 2
    ;;
esac

CONTEXT_LENGTH="${CONTEXT_LENGTH:-8192}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.84}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-4}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-4096}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-fa3}"
PREFILL_ATTENTION_BACKEND="${PREFILL_ATTENTION_BACKEND:-${ATTENTION_BACKEND}}"
DECODE_ATTENTION_BACKEND="${DECODE_ATTENTION_BACKEND:-${ATTENTION_BACKEND}}"
FP8_GEMM_BACKEND="${FP8_GEMM_BACKEND:-auto}"
FP4_GEMM_BACKEND="${FP4_GEMM_BACKEND:-auto}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.6-27b-teacher}"
DTYPE="${DTYPE:-auto}"

# Qwen3.6 uses MTP in the model family. SGLang exposes this through speculative
# decoding flags when the backend/model path supports it. Enable it after the
# base profile loads cleanly; keep the toggle explicit because invalid speculative
# settings can prevent the server from starting.
ENABLE_MTP="${ENABLE_MTP:-0}"
SPECULATIVE_ALGORITHM="${SPECULATIVE_ALGORITHM:-NEXTN}"
SPECULATIVE_NUM_STEPS="${SPECULATIVE_NUM_STEPS:-1}"
SPECULATIVE_NUM_DRAFT_TOKENS="${SPECULATIVE_NUM_DRAFT_TOKENS:-4}"

if [[ ! -x "${SGLANG_PYTHON}" ]]; then
  echo "Missing SGLang Python: ${SGLANG_PYTHON}" >&2
  exit 1
fi

if [[ "${SKIP_VERSION_CHECK}" != "1" ]]; then
  "${SGLANG_PYTHON}" - "${SGLANG_MIN_VERSION}" <<'PY'
import sys
from packaging.version import Version
import sglang

required = Version(sys.argv[1])
current = Version(sglang.__version__)
if current < required:
    raise SystemExit(
        f"SGLang {current} found, but Qwen3.6 teacher serving should use >= {required}. "
        "Upgrade .venv-sglang or rerun with SKIP_VERSION_CHECK=1 for a smoke test."
    )
print(f"SGLang {current} ok")
PY
fi

cmd=(
  "${SGLANG_PYTHON}" -m sglang.launch_server
  --model-path "${MODEL_PATH}"
  --trust-remote-code
  --host "${HOST}"
  --port "${PORT}"
  --dtype "${DTYPE}"
  --context-length "${CONTEXT_LENGTH}"
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
  --attention-backend "${ATTENTION_BACKEND}"
  --prefill-attention-backend "${PREFILL_ATTENTION_BACKEND}"
  --decode-attention-backend "${DECODE_ATTENTION_BACKEND}"
  --fp8-gemm-backend "${FP8_GEMM_BACKEND}"
  --fp4-gemm-backend "${FP4_GEMM_BACKEND}"
  --tool-call-parser "${TOOL_CALL_PARSER}"
  --reasoning-parser "${REASONING_PARSER}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --show-time-cost
)

if [[ -n "${QUANTIZATION}" ]]; then
  cmd+=(--quantization "${QUANTIZATION}")
fi

if [[ -n "${KV_CACHE_DTYPE}" && "${KV_CACHE_DTYPE}" != "auto" ]]; then
  cmd+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
fi

if [[ -n "${MAX_TOTAL_TOKENS}" ]]; then
  cmd+=(--max-total-tokens "${MAX_TOTAL_TOKENS}")
fi

if [[ "${ENABLE_MTP}" == "1" ]]; then
  cmd+=(
    --speculative-algorithm "${SPECULATIVE_ALGORITHM}"
    --speculative-num-steps "${SPECULATIVE_NUM_STEPS}"
    --speculative-num-draft-tokens "${SPECULATIVE_NUM_DRAFT_TOKENS}"
  )
fi

echo "Launching SGLang teacher:"
printf ' %q' "${cmd[@]}"
echo
exec "${cmd[@]}"
