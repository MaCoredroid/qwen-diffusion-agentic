#!/usr/bin/env bash
set -uo pipefail
cd /home/mark/qwen_diffusion
SLOG="${1:?logfile}"; SCOPE="${2:?scope}"
PORT=9952; BOOT_DL=900
LAUNCH="VLLM_FASTDLLM_W1_DRAFT_VERIFY=1 VLLM_W1_TRACE=1 \
MODEL_DIR=/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-mswe2-S-twinK1-vllm-bf16 \
MASK_TOKEN_ID=248077 MAX_MODEL_LEN=32768 GPU_UTIL=0.74 MAX_NUM_SEQS=4 PORT=9952 HF_HUB_OFFLINE=1 \
bash /home/mark/shared/lumoFlyWheel_codex_fork/scripts/qwen35_9b_flare_hybrid_serve.sh"
echo "[boot] launching scope=$SCOPE gate=1 trace=1 $(date -u +%FT%TZ)" >&2
systemd-run --user --scope --unit="$SCOPE" -p MemoryMax=24G -p MemorySwapMax=6G \
  bash -c "$LAUNCH" > "$SLOG" 2>&1 &
deadline=$((SECONDS+BOOT_DL)); grace=$((SECONDS+45)); seen=0
while :; do
  curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && { echo "[boot] :$PORT UP $(date -u +%FT%TZ)" >&2; exit 0; }
  st=$(systemctl --user is-active "${SCOPE}.scope" 2>/dev/null || true)
  if [[ "$st" == active || "$st" == activating ]]; then seen=1
  elif [[ $seen -eq 1 ]]; then echo "[boot] scope died state=$st" >&2; exit 2
  elif [[ $SECONDS -gt $grace ]]; then echo "[boot] scope never active state=$st" >&2; exit 2; fi
  [[ $SECONDS -gt $deadline ]] && { echo "[boot] BOOT TIMEOUT" >&2; exit 1; }
  sleep 5; done
