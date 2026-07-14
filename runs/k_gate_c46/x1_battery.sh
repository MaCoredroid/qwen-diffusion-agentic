#!/usr/bin/env bash
# X.1 N=64 read-arg replay battery: boot ONE FLARE server on a given export+policy, run the
# 5-divergence-prompt x N replay (P(limit|read_file) grounding rate), tear the server down.
#   usage: x1_battery.sh <MODEL_DIR> <hybrid_clean|careful_live_grammar> <TAG> <N> <OUT.json>
# server DOWN + GPU idle on exit (trap).
set -uo pipefail
cd /home/mark/qwen_diffusion
ROOT=/home/mark/qwen_diffusion
MODEL_DIR="${1:?MODEL_DIR}"; POLICY="${2:?policy}"; TAG="${3:?tag}"; N="${4:-64}"; OUT="${5:?out.json}"
SERVE=/home/mark/shared/lumoFlyWheel_codex_fork/scripts/qwen35_9b_flare_hybrid_serve.sh
PY="$ROOT/.venv/bin/python"
PORT=9952; SCOPE="x1_batt_${TAG}"; BOOT_DL=1000; GPU_CEIL=8000
LOGDIR="$ROOT/runs/k_gate_c46/x1_battery_logs"; mkdir -p "$LOGDIR"
slog="$LOGDIR/${TAG}_server.log"

gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
cleanup(){
  systemctl --user stop "${SCOPE}.scope" 2>/dev/null || true
  pkill -TERM -f 'qwen35_9b_flare_hybrid_serve|EngineCore|vllm' 2>/dev/null || true
  local dl=$((SECONDS+120))
  while :; do local u; u=$(gpu_used); [[ "$u" -lt "$GPU_CEIL" ]] && { echo "[cleanup] GPU settled ${u}MiB" >&2; break; }
    [[ $SECONDS -gt $dl ]] && { pkill -KILL -f 'qwen35_9b_flare_hybrid_serve|EngineCore|vllm' 2>/dev/null||true; break; }; sleep 5; done
}
trap cleanup EXIT

# preflight: GPU must be free
u=$(gpu_used); [[ "$u" -lt 3000 ]] || { echo "[x1-batt] GPU busy ${u}MiB — refuse" >&2; exit 1; }
echo "[x1-batt] boot scope=$SCOPE model=$MODEL_DIR policy=$POLICY N=$N" >&2

systemd-run --user --scope --unit "$SCOPE" -- \
  env MODEL_DIR="$MODEL_DIR" MASK_TOKEN_ID=248077 MAX_MODEL_LEN=32768 GPU_UTIL=0.74 MAX_NUM_SEQS=4 \
      PORT="$PORT" HF_HUB_OFFLINE=1 DECODE_POLICY="$POLICY" \
      bash "$SERVE" >"$slog" 2>&1 &
SCOPEPID=$!

# wait ready
dl=$((SECONDS+BOOT_DL))
while :; do
  curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && { echo "[x1-batt] server ready" >&2; break; }
  [[ $SECONDS -gt $dl ]] && { echo "[x1-batt] BOOT TIMEOUT; tail server log:" >&2; tail -20 "$slog" >&2; exit 1; }
  sleep 5
done

echo "[x1-batt] running replay N=$N tag=$TAG -> $OUT" >&2
"$PY" "$ROOT/runs/k_gate_c46/x1_repro_replay.py" --tag "$TAG" --n "$N" --out "$OUT" --workers 8 --max-tokens 256
RC=$?
echo "[x1-batt] replay rc=$RC" >&2
exit $RC
