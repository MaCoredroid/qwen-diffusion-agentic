#!/usr/bin/env bash
# W-1c LiveCert: boot ONE FLARE hybrid_clean twin@plain server, gate parameterized.
#   usage: boot_server.sh <gate 0|1> <logfile> <scope>
# Frozen envelope IDENTICAL to the C46-iter2 twin arm (run_arm_twin.sh):
#   twin@plain twinK1 export, mask 248077, max_model_len 32768, gmu 0.74,
#   max_num_seqs 4, VLLM_FLARE_BIDIR_PROBE=1 (the certified full-reveal envelope).
# The ONLY delta vs the banked gate-OFF arm is VLLM_FASTDLLM_W1_DRAFT_VERIFY=$GATE.
# Leaves the server UP (caller drives + tears down via teardown_server.sh).
set -uo pipefail
cd /home/mark/qwen_diffusion
GATE="${1:?gate 0|1}"; SLOG="${2:?logfile}"; SCOPE="${3:?scope}"
PORT=9952; BOOT_DL=900; GPU_CEIL=8000
gpu_used()  { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
gpu_capps() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . ; }

# preflight: GPU clear (gnome baseline tolerated)
dl=$((SECONDS+300))
while :; do u=$(gpu_used)
  [[ "$u" -lt "$GPU_CEIL" ]] && { echo "[boot] preflight clear ${u}MiB" >&2; break; }
  [[ $SECONDS -gt $dl ]] && { echo "[boot] preflight TIMEOUT ${u}MiB" >&2; exit 1; }
  echo "[boot] preflight busy ${u}MiB" >&2; sleep 8; done

LAUNCH="VLLM_FASTDLLM_W1_DRAFT_VERIFY=${GATE} \
MODEL_DIR=/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-mswe2-S-twinK1-vllm-bf16 \
MASK_TOKEN_ID=248077 MAX_MODEL_LEN=32768 GPU_UTIL=0.74 MAX_NUM_SEQS=4 PORT=9952 HF_HUB_OFFLINE=1 \
bash /home/mark/shared/lumoFlyWheel_codex_fork/scripts/qwen35_9b_flare_hybrid_serve.sh"

echo "[boot] launching scope=$SCOPE gate=$GATE $(date -u +%FT%TZ)" >&2
systemd-run --user --scope --unit="$SCOPE" \
  -p MemoryMax=24G -p MemorySwapMax=6G \
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
