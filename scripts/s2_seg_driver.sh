#!/usr/bin/env bash
# S2 pilot — drive ONE resumable training sub-segment (<=560s foreground turn).
# Auto-resumes from the latest checkpoint, applies the three checkpoint-trap
# mitigations, runs to STOP_AT_STEP under the memory scope, prints a summary.
#
#   scripts/s2_seg_driver.sh <STOP_AT_STEP>
set -euo pipefail
cd /home/mark/qwen_diffusion

STOP="${1:?usage: s2_seg_driver.sh <STOP_AT_STEP>}"
OUT="$PWD/runs/s2_pilot/Apilot_step400_seed90101"
LOGDIR="$OUT/seg_logs"; mkdir -p "$LOGDIR"
SEGLOG="$LOGDIR/seg_stop${STOP}.log"

# --- locate latest checkpoint + mitigation (2): delete stray adapter_model/ subdir ---
RESUME=""
FRESH=1
if [[ -d "$OUT" ]]; then
  LATEST=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sed 's#.*checkpoint-##' | sort -n | tail -1 || true)
  if [[ -n "${LATEST:-}" ]]; then
    RESUME="$OUT/checkpoint-$LATEST"
    FRESH=0
    if [[ -d "$RESUME/adapter_model" ]]; then
      echo "[driver] mitigation(2): removing stray $RESUME/adapter_model"
      rm -rf "$RESUME/adapter_model"
    fi
  fi
fi
echo "[driver] STOP=$STOP RESUME=${RESUME:-<fresh>} FRESH=$FRESH  $(date -Iseconds)"

# --- GPU pre-flight (idle baseline ~2.2GB; require <3000 MiB) ---
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [[ "$USED" -ge 3000 ]]; then
  echo "[driver] ABORT: GPU busy ($USED MiB >= 3000)"; exit 9
fi

# --- run the segment under the mandated memory scope ---
set +e
systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=4G -- \
  env SAVE_STEPS=30 SEG_SEED=90101 STOP_AT_STEP="$STOP" RESUME_FROM_CHECKPOINT="$RESUME" \
  bash scripts/s2_train_segment.sh > "$SEGLOG" 2>&1
RC=$?
set -e

echo "[driver] segment exit=$RC"
echo "==== flare per-forward loss (last 6) ===="
grep -E 'fastdllm-qwen35-flare\]' "$SEGLOG" | tail -6 || true
echo "==== HF loss log (last 6) ===="
grep -E "'loss':" "$SEGLOG" | tail -6 || true
echo "==== train metrics / errors ===="
grep -E 'train_runtime|train_loss|Error|Traceback|CUDA out of memory|KILL' "$SEGLOG" | tail -8 || true

# --- new latest checkpoint + mitigation (1): verify resume-state after a segment ---
NEW=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sed 's#.*checkpoint-##' | sort -n | tail -1 || true)
CKPT="$OUT/checkpoint-$NEW"
echo "[driver] latest checkpoint now: checkpoint-$NEW"
if [[ -n "${NEW:-}" ]]; then
  echo "==== resume-state files in checkpoint-$NEW ===="
  MISS=0
  for f in optimizer.pt scheduler.pt trainer_state.json; do
    if [[ -f "$CKPT/$f" ]]; then echo "  OK  $f"; else echo "  MISSING $f"; MISS=1; fi
  done
  if ls "$CKPT"/rng_state*.pth >/dev/null 2>&1; then echo "  OK  rng_state"; else echo "  MISSING rng_state"; MISS=1; fi
  if [[ -f "$CKPT/adapter_model.safetensors" ]]; then echo "  OK  adapter_model.safetensors"; else echo "  MISSING adapter_model.safetensors"; MISS=1; fi
  if [[ -d "$CKPT/adapter_model" ]]; then echo "  WARN stray adapter_model/ subdir PRESENT (trap 2)"; fi
  echo "[driver] RESUME_STATE=$([[ $MISS -eq 0 ]] && echo OK || echo INCOMPLETE)"
  # LR continuity (mitigation 3): last LR logged this segment
  grep -oE "'learning_rate': [0-9.e+-]+" "$SEGLOG" | tail -1 || true
fi
exit $RC
