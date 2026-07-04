#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv-fastdllm/bin/python}"
VLLM_BIN="${VLLM_BIN:-$ROOT/.venv-vllm/bin/vllm}"
STOCK_MODEL="${STOCK_MODEL:-/home/mark/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-9b-stock-bf16}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-9963}"
BASE_URL="http://${HOST}:${PORT}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/endgame_stock_qwen35_ar_guided}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.68}"
MAMBA_BLOCK_SIZE="${MAMBA_BLOCK_SIZE:-1024}"

mkdir -p "$OUT_ROOT/logs"

export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

"$VLLM_BIN" serve "$STOCK_MODEL" \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --mamba-block-size "$MAMBA_BLOCK_SIZE" \
  --gdn-prefill-backend triton \
  >"$OUT_ROOT/logs/vllm_stock_server.log" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

for _ in $(seq 1 240); do
  if curl -fsS "$BASE_URL/v1/models" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "vLLM stock server exited before readiness; tail follows" >&2
    tail -200 "$OUT_ROOT/logs/vllm_stock_server.log" >&2 || true
    exit 1
  fi
  sleep 2
done

curl -fsS "$BASE_URL/v1/models" >"$OUT_ROOT/server_models.json"
cat >"$OUT_ROOT/server_launch.json" <<EOF
{
  "model": "$STOCK_MODEL",
  "served_model_name": "$SERVED_MODEL_NAME",
  "base_url": "$BASE_URL",
  "dtype": "bfloat16",
  "quantization": "none",
  "vllm_bin": "$VLLM_BIN",
  "max_model_len": $MAX_MODEL_LEN,
  "gpu_memory_utilization": $GPU_MEMORY_UTILIZATION,
  "enable_prefix_caching": true,
  "mamba_cache_mode": "align",
  "mamba_block_size": $MAMBA_BLOCK_SIZE,
  "gdn_prefill_backend": "triton",
  "note": "Stock Qwen/Qwen3.5-9B cached snapshot; not NVFP4."
}
EOF

"$PYTHON" scripts/eval_flare_northstar_matched.py \
  --backend ar-vllm-guided \
  --input-jsonl "$ROOT/data/toolcall_eval_native/flare_scaleup_native_58.jsonl" \
  --out-dir "$OUT_ROOT/matched20" \
  --episode-limit 20 \
  --min-turns 3 \
  --max-turns 6 \
  --prompt-tokenizer-path "$STOCK_MODEL" \
  --ar-model-path "$STOCK_MODEL" \
  --ar-base-url "$BASE_URL" \
  --ar-served-model "$SERVED_MODEL_NAME" \
  --timeout 120

"$PYTHON" scripts/eval_flare_northstar_matched.py \
  --backend ar-vllm-guided \
  --input-jsonl "$ROOT/data/toolcall_eval_native/flare_nevertrain_bfcl_apibank.jsonl" \
  --out-dir "$OUT_ROOT/nevertrain_bfcl_apibank60" \
  --episode-limit 60 \
  --min-turns 1 \
  --max-turns 6 \
  --prompt-tokenizer-path "$STOCK_MODEL" \
  --ar-model-path "$STOCK_MODEL" \
  --ar-base-url "$BASE_URL" \
  --ar-served-model "$SERVED_MODEL_NAME" \
  --timeout 120

echo "stock guided controls written under $OUT_ROOT"
