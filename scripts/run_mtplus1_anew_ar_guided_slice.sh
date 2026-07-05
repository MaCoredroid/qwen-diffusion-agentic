#!/usr/bin/env bash
# Single-slice AR-guided eval for merged+A_new exported vLLM model (convert-after-RL #29, STEP 3 eval b).
# Mirrors run_stock_qwen35_ar_guided_controls.sh::run_arm bf16 serve args + eval invocation EXACTLY,
# but boots+evals+kills ONE server for ONE slice so each call is self-contained (turn discipline).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv-fastdllm/bin/python}"
VLLM_BIN="${VLLM_BIN:-$ROOT/.venv-vllm/bin/vllm}"
MODEL="${MODEL:?set MODEL to exported vllm model dir}"
SLICE_JSONL="${SLICE_JSONL:?set SLICE_JSONL}"
OUT_DIR="${OUT_DIR:?set OUT_DIR}"
EPISODE_LIMIT="${EPISODE_LIMIT:?}"
MIN_TURNS="${MIN_TURNS:?}"
MAX_TURNS="${MAX_TURNS:?}"
SERVED_NAME="${SERVED_NAME:-diffusion-mtplus1-Anew}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-9971}"
BASE_URL="http://${HOST}:${PORT}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.66}"
MAMBA_BLOCK_SIZE="${MAMBA_BLOCK_SIZE:-1024}"

export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

mkdir -p "$OUT_DIR"
LOG_PATH="$OUT_DIR/vllm_server.log"
SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    for _ in $(seq 1 20); do kill -0 "$SERVER_PID" >/dev/null 2>&1 || break; sleep 1; done
    kill -9 "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

"$VLLM_BIN" serve "$MODEL" \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$SERVED_NAME" \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --mamba-block-size "$MAMBA_BLOCK_SIZE" \
  --gdn-prefill-backend triton \
  --enforce-eager >"$LOG_PATH" 2>&1 &
SERVER_PID=$!

ready=0
for _ in $(seq 1 150); do
  if curl -fsS "$BASE_URL/v1/models" >/dev/null 2>&1; then ready=1; break; fi
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "vLLM server exited before readiness; tail:" >&2; tail -80 "$LOG_PATH" >&2 || true; exit 1
  fi
  sleep 2
done
if [[ "$ready" != "1" ]]; then
  echo "vLLM server not ready in time; tail:" >&2; tail -80 "$LOG_PATH" >&2 || true; exit 1
fi
curl -fsS "$BASE_URL/v1/models" >"$OUT_DIR/server_models.json"
echo "SERVER READY at $(date +%T)"

"$PYTHON" scripts/eval_flare_northstar_matched.py \
  --backend ar-vllm-guided \
  --input-jsonl "$SLICE_JSONL" \
  --out-dir "$OUT_DIR" \
  --episode-limit "$EPISODE_LIMIT" \
  --min-turns "$MIN_TURNS" \
  --max-turns "$MAX_TURNS" \
  --prompt-tokenizer-path "$MODEL" \
  --ar-model-path "$MODEL" \
  --ar-base-url "$BASE_URL" \
  --ar-served-model "$SERVED_NAME" \
  --timeout 120

echo "EVAL DONE at $(date +%T)"
