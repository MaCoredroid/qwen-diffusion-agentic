#!/usr/bin/env bash
# X.2 PHASE DATA — boot the SAME-WEIGHTS AR teacher ONCE (mswe-S-iter2 vllm-bf16,
# careful_live_grammar = native AR path = the 12/48 policy), generate the deterministic
# read-arg distillation targets over the KEEPER read-phase states, tear the server down.
# Server DOWN + GPU idle on exit (trap). Foreground (bounded ~30 min); the DATA phase.
set -uo pipefail
cd /home/mark/qwen_diffusion
ROOT=/home/mark/qwen_diffusion
SERVE=/home/mark/shared/lumoFlyWheel_codex_fork/scripts/qwen35_9b_flare_hybrid_serve.sh
AR=$ROOT/models/qwen3.5-9b-fastdllm-mswe-S-iter2-vllm-bf16
PY=$ROOT/.venv/bin/python
OUTRUN=$ROOT/runs/kraise_reconvert_iter2_x2
PREFIXES=$OUTRUN/gen_prefixes.jsonl
OUT=$OUTRUN/ar_targets.jsonl
PORT=9953; SCOPE="x2_argen"; BOOT_DL=1200; GPU_CEIL=8000
slog="$OUTRUN/ar_gen_server.log"
mkdir -p "$OUTRUN"

gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
cleanup(){
  echo "[x2-argen] teardown" >&2
  systemctl --user stop "${SCOPE}.scope" 2>/dev/null || true
  pkill -TERM -f 'qwen35_9b_flare_hybrid_serve|EngineCore|vllm' 2>/dev/null || true
  local dl=$((SECONDS+120))
  while :; do local u; u=$(gpu_used); [[ "$u" -lt "$GPU_CEIL" ]] && { echo "[x2-argen] GPU settled ${u}MiB" >&2; break; }
    [[ $SECONDS -gt $dl ]] && { pkill -KILL -f 'qwen35_9b_flare_hybrid_serve|EngineCore|vllm' 2>/dev/null||true; break; }; sleep 5; done
}
trap cleanup EXIT

[[ -f "$PREFIXES" ]] || { echo "[x2-argen] missing $PREFIXES (run: x2_ar_self_distill.py requests)" >&2; exit 1; }
[[ -d "$AR" ]] || { echo "[x2-argen] missing AR export $AR" >&2; exit 1; }
u=$(gpu_used); [[ "$u" -lt 3000 ]] || { echo "[x2-argen] GPU busy ${u}MiB — refuse" >&2; exit 1; }

echo "[x2-argen] boot scope=$SCOPE model=$AR policy=careful_live_grammar port=$PORT" >&2
systemd-run --user --scope --unit "$SCOPE" -- \
  env MODEL_DIR="$AR" MAX_MODEL_LEN=8192 GPU_UTIL=0.74 MAX_NUM_SEQS=8 \
      PORT="$PORT" HF_HUB_OFFLINE=1 DECODE_POLICY=careful_live_grammar \
      bash "$SERVE" >"$slog" 2>&1 &

dl=$((SECONDS+BOOT_DL))
while :; do
  curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && { echo "[x2-argen] server ready" >&2; break; }
  [[ $SECONDS -gt $dl ]] && { echo "[x2-argen] BOOT TIMEOUT; server log tail:" >&2; tail -30 "$slog" >&2; exit 1; }
  sleep 5
done

echo "[x2-argen] generating -> $OUT" >&2
"$PY" "$ROOT/scripts/x2_gen_client.py" \
  --prefixes "$PREFIXES" --out "$OUT" \
  --url "http://127.0.0.1:${PORT}/v1/completions" --workers 8 --max-tokens 24
RC=$?
echo "[x2-argen] client rc=$RC  targets=$(wc -l < "$OUT" 2>/dev/null || echo 0)" >&2
exit $RC
