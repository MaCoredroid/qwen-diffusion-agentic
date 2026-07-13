#!/usr/bin/env bash
# ============================================================================
# ITERATION-2 TWO-ARM SWE-SFT SEQUENTIAL RUNNER  (#127)
# ----------------------------------------------------------------------------
# Retrain-freely: MIRRORS the iteration-1 arm-1/arm-2 launch EXACTLY except the
# dataset. iteration-1 provenance:
#   arm-1 = scripts/swe_sft_arm1_driver.sh  (merged-RL-v2 base, block 12288)
#   arm-2 = scripts/swe_sft_arm2_driver.sh  (stock-init base, identical config)
# Both drove scripts/swe_sft_arm1_qlora_train.py (AR single-stream causal QLoRA,
# SDPA attn, chunked-CE, 4-bit NF4). This runner drives that SAME trainer with
# the SAME flags; the ONLY change vs iteration-1 is the DATASET.
#
# CHANGES FOR ITERATION-2 (and ONLY these):
#   (a) dataset = data/swe_sft_pool/lmflow_pretok_iter2/swe_sft_train.json
#       (built by build_swe_sft_lmflow_pretok.py --tokenized
#        data/swe_sft_pool/train_swe_sft_windowed.tokenized.jsonl --max-len 12288;
#        987 windows, 383 episodes, 100% assistant-label retention, 0 truncated).
#   (b) output dirs runs/swe_sft_arm1_iter2/ + runs/swe_sft_arm2_iter2/.
#   (c) step count = 400 (UNCHANGED). iteration-1's HORIZON=400 is a FIXED cosine
#       horizon / design constant (defines the {100,200,300,400} erosion-sweep
#       checkpoints + the anchor-gate cadence); it is NOT derived from dataset row
#       count anywhere in the iter-1 config. Per the mirror-the-LOGIC rule (cap
#       600 applies ONLY if steps were dataset-size-derived, which they were not),
#       keep 400.  [obs: 400 steps over 987 windows ~= 0.41 pass; iter-1 was ~1.2
#        passes over 334 episode-rows. Retrain-freely: escalate later if it
#        undertrains. Kept 400 to preserve an exact same-hyperparameter mirror.]
#
# EVERYTHING ELSE IDENTICAL: arm-1 base merged-RL-v2, arm-2 base stock, QLoRA
# r16/alpha32 same 11 targets, AR single-stream, block 12288, chunked-CE
# (logits-chunk 2048), seed 71101, LR 1e-5 cosine warmup 0.03, save-steps 100,
# save-total-limit 6, logging-steps 5, resume auto.
#
# CAGE (mirrors #110-era pattern): setsid (this script is launched under setsid) +
# per-arm systemd-run --user --scope MemoryMax=22G MemorySwapMax=4G + reset-failed
# pre-boot + pidfile (this runner's pid) + EXIT cleanup trap.
#
# ORCHESTRATION CONTRACT:
#   * SEQUENTIAL: arm-1 to completion -> arm-2.
#   * [state] lines per arm: step / loss / ETA (parsed from the arm metrics.jsonl).
#   * pidfile = runs/swe_sft_iter2.pid (this runner's pid; removed on EXIT).
#   * STOP-file (runs/swe_sft_iter2.STOP) checked BETWEEN arms -> graceful stop.
#   * On arm crash (nonzero rc or no final checkpoint): log '[state] ARM_FAILED'
#     and STOP. NO auto-retry.
# ============================================================================
set -uo pipefail
cd /home/mark/qwen_diffusion

# ---- frozen config (mirror iteration-1) ----
HORIZON=400
SEED=71101
BLOCK=12288
LOGITS_CHUNK=2048
SAVE_STEPS=100
SAVE_TOTAL_LIMIT=6
LOGGING_STEPS=5
PY="$PWD/.venv-fastdllm/bin/python"
TRAINER="scripts/swe_sft_arm1_qlora_train.py"
DATA="$PWD/data/swe_sft_pool/lmflow_pretok_iter2/swe_sft_train.json"

ARM1_BASE="$PWD/models/qwen3.5-9b-fastdllm-mtplus1-merged"   # design arm S = merged-RL-v2
ARM2_BASE="$PWD/models/qwen3.5-9b-fastdllm-init"             # design arm T = stock
ARM1_DIR="$PWD/runs/swe_sft_arm1_iter2"
ARM2_DIR="$PWD/runs/swe_sft_arm2_iter2"
ARM1_OUT="$ARM1_DIR/Aswe_S_step${HORIZON}_seed${SEED}"
ARM2_OUT="$ARM2_DIR/Aswe_T_step${HORIZON}_seed${SEED}"

PIDFILE="$PWD/runs/swe_sft_iter2.pid"
STOPFILE="$PWD/runs/swe_sft_iter2.STOP"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FASTDLLM_GDN_KERNEL=fla

mkdir -p "$ARM1_DIR" "$ARM2_DIR" "$ARM1_OUT" "$ARM2_OUT"

echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

ts() { date -Iseconds; }
state() { echo "[state] $(ts) $*"; }
say()   { echo "[iter2-runner $(ts)] $*"; }

# ---- GPU free gate (one tenant at a time; <1GB used) ----
gpu_used_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
wait_gpu_free() {
  # up to ~180s for the previous arm's VRAM to release
  local tries=0 used
  while :; do
    used=$(gpu_used_mib)
    if [[ "${used:-99999}" -lt 1000 ]]; then return 0; fi
    tries=$((tries+1))
    if [[ $tries -ge 36 ]]; then say "GPU still busy (${used} MiB) after wait"; return 1; fi
    sleep 5
  done
}

# ---- emit a [state] line from the tail of an arm's metrics.jsonl ----
emit_progress() {
  local name="$1" metrics="$2"
  "$PY" - "$name" "$metrics" "$HORIZON" <<'PY' 2>/dev/null || true
import json, sys
name, path, horizon = sys.argv[1], sys.argv[2], int(sys.argv[3])
try:
    last = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                last = line
    if not last:
        print(f"[state-nostep] arm={name} (no metrics rows yet)"); sys.exit(0)
    o = json.loads(last)
    step = int(o.get("step", 0)); loss = o.get("loss"); el = float(o.get("elapsed_s", 0.0))
    sps = (el/step) if step > 0 else 0.0
    rem = max(0, horizon - step)
    eta_min = (rem*sps)/60.0
    lf = f"{loss:.4f}" if isinstance(loss,(int,float)) else "na"
    print(f"[state] arm={name} step={step}/{horizon} loss={lf} s_per_step={sps:.2f} eta_min={eta_min:.1f}")
except Exception as e:
    print(f"[state-parseerr] arm={name} {e}")
PY
}

# ---- run one arm to completion inside a caged, detached-friendly trainer ----
# returns 0 on success (rc 0 AND checkpoint-HORIZON present), nonzero on failure.
run_arm() {
  local name="$1" base="$2" out="$3"
  local metrics="$out/metrics.jsonl"
  local log="$out/train.log"
  : > "$metrics"

  if ! wait_gpu_free; then
    state "ARM_FAILED arm=$name reason=gpu_busy"
    return 91
  fi

  systemctl --user reset-failed >/dev/null 2>&1 || true
  say "launch arm=$name base=$base out=$out block=$BLOCK horizon=$HORIZON seed=$SEED"
  state "arm=$name step=0/$HORIZON loss=na eta_min=? (starting; base=$(basename "$base"))"

  systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=4G -- \
    env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FASTDLLM_GDN_KERNEL=fla \
    "$PY" "$TRAINER" \
      --model "$base" \
      --dataset "$DATA" --output-dir "$out" \
      --block-size "$BLOCK" --horizon "$HORIZON" --stop-at-step "$HORIZON" \
      --save-steps "$SAVE_STEPS" --save-total-limit "$SAVE_TOTAL_LIMIT" \
      --logging-steps "$LOGGING_STEPS" --logits-chunk "$LOGITS_CHUNK" \
      --seed "$SEED" --resume auto --metrics "$metrics" \
      > "$log" 2>&1 &
  local cpid=$!

  # monitor: emit [state] every 45s until the trainer exits
  while kill -0 "$cpid" 2>/dev/null; do
    sleep 45
    emit_progress "$name" "$metrics"
  done
  wait "$cpid"; local rc=$?

  emit_progress "$name" "$metrics"
  if [[ $rc -ne 0 ]]; then
    state "ARM_FAILED arm=$name reason=nonzero_rc rc=$rc (see $log)"
    return $rc
  fi
  if [[ ! -d "$out/checkpoint-$HORIZON" ]]; then
    state "ARM_FAILED arm=$name reason=no_final_checkpoint (rc=0 but checkpoint-$HORIZON absent; see $log)"
    return 90
  fi
  state "ARM_DONE arm=$name rc=0 final=$out/checkpoint-$HORIZON"
  return 0
}

# ============================== SEQUENCE ====================================
say "ITER-2 two-arm runner START pid=$$ dataset=$DATA"

# ---- ARM 1 (merged-RL-v2) ----
if ! run_arm "arm1_iter2_S" "$ARM1_BASE" "$ARM1_OUT"; then
  say "arm-1 did not complete; STOPPING (no auto-retry)."
  exit 1
fi

# ---- STOP-file check BETWEEN arms ----
if [[ -f "$STOPFILE" ]]; then
  state "STOP_REQUESTED stopfile=$STOPFILE -> arm-1 complete, arm-2 NOT launched."
  say "STOP file present between arms; exiting cleanly after arm-1."
  exit 0
fi

# ---- ARM 2 (stock init) ----
if ! run_arm "arm2_iter2_T" "$ARM2_BASE" "$ARM2_OUT"; then
  say "arm-2 did not complete; STOPPING (no auto-retry)."
  exit 1
fi

state "ALL_DONE arm1=$ARM1_OUT/checkpoint-$HORIZON arm2=$ARM2_OUT/checkpoint-$HORIZON"
say "ITER-2 two-arm runner COMPLETE."
