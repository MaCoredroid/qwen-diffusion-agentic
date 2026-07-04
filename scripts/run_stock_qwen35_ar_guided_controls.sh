#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv-fastdllm/bin/python}"
VLLM_BIN="${VLLM_BIN:-$ROOT/.venv-vllm/bin/vllm}"
STOCK_MODEL="${STOCK_MODEL:-/home/mark/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
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

SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

run_arm() {
  local arm="$1"
  local dtype="$2"
  local quantization="$3"
  local served_model_name="$4"
  local arm_root="$OUT_ROOT/$arm"
  local log_path="$OUT_ROOT/logs/vllm_stock_${arm}_server.log"

  mkdir -p "$arm_root"
  rm -f "$arm_root/server_models.json" "$arm_root/server_launch.json"

  local serve_args=(
    serve "$STOCK_MODEL"
    --trust-remote-code
    --dtype "$dtype"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --host "$HOST"
    --port "$PORT"
    --served-model-name "$served_model_name"
    --enable-prefix-caching
    --mamba-cache-mode align
    --mamba-block-size "$MAMBA_BLOCK_SIZE"
    --gdn-prefill-backend triton
  )
  if [[ "$quantization" != "none" ]]; then
    serve_args+=(--quantization "$quantization")
  fi

  "$VLLM_BIN" "${serve_args[@]}" >"$log_path" 2>&1 &
  SERVER_PID=$!

  local ready=0
  for _ in $(seq 1 240); do
    if curl -fsS "$BASE_URL/v1/models" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
      echo "vLLM stock ${arm} server exited before readiness; tail follows" >&2
      tail -200 "$log_path" >&2 || true
      exit 1
    fi
    sleep 2
  done
  if [[ "$ready" != "1" ]]; then
    echo "vLLM stock ${arm} server did not become ready; tail follows" >&2
    tail -200 "$log_path" >&2 || true
    exit 1
  fi

  curl -fsS "$BASE_URL/v1/models" >"$arm_root/server_models.json"
  cat >"$arm_root/server_launch.json" <<EOF
{
  "model": "$STOCK_MODEL",
  "served_model_name": "$served_model_name",
  "base_url": "$BASE_URL",
  "dtype": "$dtype",
  "quantization": "$quantization",
  "vllm_bin": "$VLLM_BIN",
  "max_model_len": $MAX_MODEL_LEN,
  "gpu_memory_utilization": $GPU_MEMORY_UTILIZATION,
  "enable_prefix_caching": true,
  "mamba_cache_mode": "align",
  "mamba_block_size": $MAMBA_BLOCK_SIZE,
  "gdn_prefill_backend": "triton",
  "note": "Stock Qwen/Qwen3.5-9B cached snapshot. bf16 arm is not NVFP4; FP8 arm uses vLLM --quantization fp8."
}
EOF

  "$PYTHON" scripts/eval_flare_northstar_matched.py \
    --backend ar-vllm-guided \
    --input-jsonl "$ROOT/data/toolcall_eval_native/flare_scaleup_native_58.jsonl" \
    --out-dir "$arm_root/matched20" \
    --episode-limit 20 \
    --min-turns 3 \
    --max-turns 6 \
    --prompt-tokenizer-path "$STOCK_MODEL" \
    --ar-model-path "$STOCK_MODEL" \
    --ar-base-url "$BASE_URL" \
    --ar-served-model "$served_model_name" \
    --timeout 120

  "$PYTHON" scripts/eval_flare_northstar_matched.py \
    --backend ar-vllm-guided \
    --input-jsonl "$ROOT/data/toolcall_eval_native/flare_nevertrain_bfcl_apibank.jsonl" \
    --out-dir "$arm_root/nevertrain_bfcl_apibank60" \
    --episode-limit 60 \
    --min-turns 1 \
    --max-turns 6 \
    --prompt-tokenizer-path "$STOCK_MODEL" \
    --ar-model-path "$STOCK_MODEL" \
    --ar-base-url "$BASE_URL" \
    --ar-served-model "$served_model_name" \
    --timeout 120

  cleanup
  SERVER_PID=""
  sleep 5
}

run_arm bf16 bfloat16 none "${SERVED_MODEL_NAME_BF16:-qwen3.5-9b-stock-bf16}"
run_arm fp8 bfloat16 fp8 "${SERVED_MODEL_NAME_FP8:-qwen3.5-9b-stock-fp8}"

echo "stock bf16/fp8 guided controls written under $OUT_ROOT"
