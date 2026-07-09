#!/usr/bin/env bash
# ARM-2 KILL-T1 anchor gate: serve ONE exported vllm-bf16 model (arm-1's exact
# serving config) and run the AR-guided matched-20 tool-call eval, then tear the
# server down. One GPU tenant; nothing outlives the call (EXIT trap kills server).
#
# Usage: swe_sft_arm2_gate_driver.sh <MODEL_DIR> <SERVED_NAME> <PORT> <OUT_DIR>
set -euo pipefail
cd /home/mark/qwen_diffusion

MODEL_DIR="$1"; SERVED_NAME="$2"; PORT="$3"; OUT_DIR="$4"
HOST=127.0.0.1
BASE_URL="http://${HOST}:${PORT}"
VLLM_BIN="$PWD/.venv-vllm/bin/vllm"
PY="$PWD/.venv-fastdllm/bin/python"
LOG="$OUT_DIR/vllm_server.log"
mkdir -p "$OUT_DIR"

export VLLM_USE_V1=1 VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false

SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    for _ in $(seq 1 30); do kill -0 "$SERVER_PID" >/dev/null 2>&1 || break; sleep 1; done
    kill -9 "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

echo "[gate] boot vllm $MODEL_DIR served=$SERVED_NAME port=$PORT"
"$VLLM_BIN" serve "$MODEL_DIR" \
  --trust-remote-code --dtype bfloat16 --max-model-len 4096 \
  --gpu-memory-utilization 0.66 --enforce-eager --enable-prefix-caching \
  --mamba-cache-mode align --mamba-block-size 1024 --gdn-prefill-backend triton \
  --host "$HOST" --port "$PORT" --served-model-name "$SERVED_NAME" \
  >"$LOG" 2>&1 &
SERVER_PID=$!

ready=0
for _ in $(seq 1 180); do
  if curl -fsS "$BASE_URL/v1/models" >/dev/null 2>&1; then ready=1; break; fi
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then echo "[gate] server died pre-ready"; tail -40 "$LOG"; exit 1; fi
  sleep 2
done
[[ "$ready" == 1 ]] || { echo "[gate] server not ready"; tail -40 "$LOG"; exit 1; }
curl -fsS "$BASE_URL/v1/models" >"$OUT_DIR/server_models.json"
echo "[gate] server ready; running matched-20 AR-guided eval -> $OUT_DIR"

"$PY" scripts/eval_flare_northstar_matched.py \
  --backend ar-vllm-guided \
  --input-jsonl "$PWD/data/toolcall_eval_native/flare_scaleup_native_58.jsonl" \
  --out-dir "$OUT_DIR" \
  --episode-limit 20 --min-turns 3 --max-turns 6 \
  --prompt-tokenizer-path "$MODEL_DIR" \
  --ar-model-path "$MODEL_DIR" \
  --ar-base-url "$BASE_URL" \
  --ar-served-model "$SERVED_NAME" \
  --timeout 120

echo "[gate] eval done for $SERVED_NAME"
cleanup
SERVER_PID=""
echo "[gate] server torn down for $SERVED_NAME"
