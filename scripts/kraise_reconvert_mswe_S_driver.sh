#!/usr/bin/env bash
# K-track re-conversion of M_swe_S -> twin@K1 (SELF-VERIFYING DETACHED LAUNCHER).
#
# Object: CONVERSION_READY.md PRIMARY = M_swe_S. This runs the CERTIFIED #29 two-stream
# FLARE conversion recipe (convert_after_rl_design.md sec 4; convert_after_rl_result.md:
# McNemar zero net-loss, 2 seeds) VERBATIM on all pinned hyperparameters, changing ONLY
#   MODEL_PATH  -> models/qwen3.5-9b-fastdllm-mswe-S-merged  (the SWE-SFT merged base)
#   OUTPUT_DIR  -> runs/kraise_reconvert/mswe_S_twinK1_run1recipe_step400_seed81101
#   SEED/DATA_SEED -> 81101 (fresh, distinct from the RL-v2 A_new 80101/80102)
#
# The conversion trains at ITS OWN feasible block size (BLOCK_SIZE=512 / TRAIN_BD_SIZE=32),
# NOT the SFT block 12288 (per the directive: SWE capability lives in the merged base
# weights; the re-conversion is NOT trained on the SWE pool -- leakage firewall).
#
# Runs as a SINGLE CONTINUOUS 400-step cosine (the gold reference the #29 4-chunk resume
# reproduced bit-for-bit): FASTDLLM_STOP_AT_STEP is left UNSET so the HF Trainer runs to
# MAX_STEPS=400 in one process; SAVE_STEPS=100 emits {100,200,300,400} checkpoints; the
# final adapter lands at OUTPUT_DIR root (adapter_model.safetensors).
#
# DETACHED: setsid + systemd-run --scope MemoryMax=22G, pidfile, train.log; GPU pre-flight
# refuses to fight another tenant. Poll: tail train.log ; ls OUTPUT_DIR/checkpoint-*.
set -euo pipefail
cd /home/mark/qwen_diffusion

SEED="${SEED:-81101}"
GPU_FREE_MIB="${GPU_FREE_MIB:-3000}"
OUT="$PWD/runs/kraise_reconvert/mswe_S_twinK1_run1recipe_step400_seed${SEED}"
PIDFILE="$OUT/train.pid"
TRAINLOG="$OUT/train.log"
BASE="$PWD/models/qwen3.5-9b-fastdllm-mswe-S-merged"
DATA="$PWD/data/flare_redesign_run1_copy_retention_mix"
mkdir -p "$OUT"

ts() { date -Iseconds; }
say() { echo "[reconv-mswe-S $(ts)] $*"; }

# ---------- PREFLIGHT (one GPU tenant; never fight another run) ----------
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [[ "$USED" -ge "$GPU_FREE_MIB" ]]; then
  say "ABORT: GPU busy (${USED} MiB >= ${GPU_FREE_MIB}); not launching."; exit 9
fi
FREEG=$(free -g | awk '/Mem/{print $7}')
if [[ "${FREEG:-0}" -lt 8 ]]; then say "ABORT: host RAM headroom ${FREEG}G < 8G."; exit 8; fi
[[ -d "$BASE" ]] || { say "ABORT: merged base missing: $BASE"; exit 7; }
[[ -f "$DATA/train_agentic_mix.json" ]] || { say "ABORT: conversion mix missing: $DATA"; exit 6; }
say "preflight OK: gpu_used=${USED}MiB host_free=${FREEG}G base=$BASE seed=$SEED"

# ---------- DETACHED LAUNCH (continuous 400-step cosine, caged) ----------
say "launch: continuous 400-step two-stream conversion (Run-1 recipe) -> $OUT"
setsid bash -c "
  echo \$\$ > '$PIDFILE'
  trap 'rm -f \"$PIDFILE\"' EXIT
  exec systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=4G -- \
    env \
      ENV_PY='$PWD/.venv-fastdllm/bin/python' \
      DATASET_DIR='$DATA' \
      OUTPUT_DIR='$OUT' \
      MODEL_PATH='$BASE' \
      MAX_STEPS=400 MAX_TRAIN_SAMPLES=5055 BLOCK_SIZE=512 TRAIN_BD_SIZE=32 GRAD_ACCUM=1 \
      LEARNING_RATE=1e-5 SAVE_STEPS=100 SAVE_TOTAL_LIMIT=6 LOGGING_STEPS=5 \
      LORA_R=16 LORA_ALPHA=32 LORA_DROPOUT=0.05 \
      LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj \
      DISABLE_GROUP_TEXTS=0 TRUNCATION_SIDE=left CONVERSATION_TEMPLATE=fast_dllm_v2_native \
      VALUE_SPAN_LOSS_WEIGHT=2.0 VALUE_SPAN_MASK_PROB=1.0 \
      SEED='$SEED' DATA_SEED='$SEED' \
      SKIP_DATASET_BUILD=1 OVERWRITE_CACHE=1 MAX_JOBS=4 \
      bash '$PWD/scripts/run_flare_redesign_run1.sh'
" > "$TRAINLOG" 2>&1 &

sleep 3
say "DETACHED. pidfile=$PIDFILE pid=$(cat "$PIDFILE" 2>/dev/null) log=$TRAINLOG"
say "monitor: tail -f $TRAINLOG ; ls $OUT/checkpoint-*"
