#!/usr/bin/env bash
# KILL-T1 SPOT launcher: boot ONE FLARE server on the X.1 twin (hybrid_clean), run the
# 3-matched-turn exact_args spot, tear the server down. server DOWN + GPU idle on exit (trap).
#   usage: x1_killt1_launch.sh <MODEL_DIR> <POLICY> <OUT.json>
set -uo pipefail
cd /home/mark/qwen_diffusion
ROOT=/home/mark/qwen_diffusion
MODEL_DIR="${1:?MODEL_DIR}"; POLICY="${2:-hybrid_clean}"; OUT="${3:?out.json}"
SERVE=/home/mark/shared/lumoFlyWheel_codex_fork/scripts/qwen35_9b_flare_hybrid_serve.sh
PY="$ROOT/.venv/bin/python"
PORT=9952; SCOPE="x1_killt1"; BOOT_DL=1000; GPU_CEIL=8000
LOGDIR="$ROOT/runs/k_gate_c46/x1_battery_logs"; mkdir -p "$LOGDIR"
slog="$LOGDIR/killt1_server.log"

gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
cleanup(){
  systemctl --user stop "${SCOPE}.scope" 2>/dev/null || true
  pkill -TERM -f 'qwen35_9b_flare_hybrid_serve|EngineCore|vllm' 2>/dev/null || true
  local dl=$((SECONDS+120))
  while :; do local u; u=$(gpu_used); [[ "$u" -lt "$GPU_CEIL" ]] && { echo "[cleanup] GPU settled ${u}MiB" >&2; break; }
    [[ $SECONDS -gt $dl ]] && { pkill -KILL -f 'qwen35_9b_flare_hybrid_serve|EngineCore|vllm' 2>/dev/null||true; break; }; sleep 5; done
}
trap cleanup EXIT

u=$(gpu_used); [[ "$u" -lt 3000 ]] || { echo "[killt1] GPU busy ${u}MiB — refuse" >&2; exit 1; }
echo "[killt1] boot scope=$SCOPE model=$MODEL_DIR policy=$POLICY" >&2

systemd-run --user --scope --unit "$SCOPE" -- \
  env MODEL_DIR="$MODEL_DIR" MASK_TOKEN_ID=248077 MAX_MODEL_LEN=32768 GPU_UTIL=0.74 MAX_NUM_SEQS=4 \
      PORT="$PORT" HF_HUB_OFFLINE=1 DECODE_POLICY="$POLICY" \
      bash "$SERVE" >"$slog" 2>&1 &

dl=$((SECONDS+BOOT_DL))
while :; do
  curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && { echo "[killt1] server ready" >&2; break; }
  [[ $SECONDS -gt $dl ]] && { echo "[killt1] BOOT TIMEOUT; tail server log:" >&2; tail -20 "$slog" >&2; exit 1; }
  sleep 5
done

echo "[killt1] running spot -> $OUT" >&2
"$PY" "$ROOT/runs/k_gate_c46/x1_killt1_spot.py" --out "$OUT"
RC=$?
echo "[killt1] spot rc=$RC" >&2
exit $RC
