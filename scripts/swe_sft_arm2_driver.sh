#!/usr/bin/env bash
# SWE-SFT arm-2 (M_swe_T) -- STOCK-base twin of arm-1. Design arm T (control).
# IDENTICAL config/pool/steps as arm-1 (block 12288, 334-keeper pool, horizon 400,
# seed 71101, r16/a32, targets q,k,v,o+GDN+MLP gate_up/down, LR 1e-5 cosine warmup 0.03),
# the ONLY difference = base model: stock init (models/qwen3.5-9b-fastdllm-init,
# NO RL-v2 merge) instead of arm-1's merged-RL-v2 (mtplus1-merged).
# Launched ONLY after arm-1 PASSES the KILL-T1 anchor gate.
set -euo pipefail
cd /home/mark/qwen_diffusion

HORIZON="${HORIZON:-400}"
SEED="${SEED:-71101}"
BLOCK="${BLOCK:-12288}"                     # measured in arm-1; no re-probe
LOGITS_CHUNK="${LOGITS_CHUNK:-2048}"
BASE_MODEL="${BASE_MODEL:-$PWD/models/qwen3.5-9b-fastdllm-init}"   # design arm T = stock
ARM_DIR="$PWD/runs/swe_sft_arm2"
OUT="${OUT:-$ARM_DIR/Aswe_T_step${HORIZON}_seed${SEED}}"
DATA_FULL="$PWD/data/swe_sft_pool/lmflow_pretok"   # identical pool arm-1 trained on (block 12288)
PIDFILE="$ARM_DIR/train.pid"
METRICS="$ARM_DIR/metrics.jsonl"
TRAINLOG="$OUT/train.log"
PY="$PWD/.venv-fastdllm/bin/python"
TRAINER="scripts/swe_sft_arm1_qlora_train.py"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FASTDLLM_GDN_KERNEL=fla
mkdir -p "$ARM_DIR" "$OUT"

ts() { date -Iseconds; }
say() { echo "[arm2-driver $(ts)] $*"; }

# ---------- 1. PREFLIGHT (GPU must be free: one tenant at a time) ----------
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [[ "$USED" -ge "${GPU_FREE_MIB:-3000}" ]]; then
  say "ABORT: GPU busy (${USED} MiB). Not launching arm-2."; exit 9
fi
FREEG=$(free -g | awk '/Mem/{print $7}')
if [[ "${FREEG:-0}" -lt 8 ]]; then say "ABORT: host RAM headroom ${FREEG}G < 8G."; exit 8; fi
if [[ ! -f "$DATA_FULL/swe_sft_train.json" ]]; then say "ABORT: pool $DATA_FULL/swe_sft_train.json missing."; exit 6; fi
say "preflight OK: gpu_used=${USED}MiB host_free=${FREEG}G base=$BASE_MODEL block=$BLOCK pool=$DATA_FULL"

# ---------- 2. DETACHED CAGED LAUNCH (stock base) ----------
say "launch arm-2: base=stock-init HORIZON=$HORIZON block=$BLOCK out=$OUT (auto-resume)"
: > "$METRICS"
setsid bash -c "
  echo \$\$ > '$PIDFILE'
  trap 'rm -f \"$PIDFILE\"' EXIT
  exec systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=4G -- \
    env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FASTDLLM_GDN_KERNEL=fla \
    '$PY' '$TRAINER' \
      --model '$BASE_MODEL' \
      --dataset '$DATA_FULL/swe_sft_train.json' --output-dir '$OUT' \
      --block-size '$BLOCK' --horizon '$HORIZON' --stop-at-step '$HORIZON' \
      --save-steps 100 --save-total-limit 6 --logging-steps 5 --logits-chunk '$LOGITS_CHUNK' \
      --seed '$SEED' --resume auto --metrics '$METRICS'
" > "$TRAINLOG" 2>&1 &

sleep 3
say "DETACHED. pidfile=$PIDFILE pid=$(cat "$PIDFILE" 2>/dev/null) log=$TRAINLOG metrics=$METRICS"
