#!/usr/bin/env bash
# ITERATION-2 DOUBLE CONVERSION RUNNER (#128 part 1) — SEQUENTIAL, DETACHED, CAGED.
#
#   ARM A (twin@plain) : #29 convert-after-RL protocol VERBATIM as iteration-1 ran it (the
#                        kraise_reconvert_mswe_S_driver.sh recipe), on the ITER-2 M_swe_S merged base.
#                        NO V1 code enters this path -> byte-reproducible shipping candidate.
#   ARM B (twin@V1)    : IDENTICAL conversion + SECTION-V V1 copy-span joint-infill (FASTDLLM_V1_COPY_SPAN=1;
#                        scripts/v1_copy_span_infill.py collator injects flare_mask_indices).
#
# Runs plain -> to completion -> V1. [state] lines: arm/step/loss/ETA. pidfile runs/iter2_conversion.pid.
# STOP-file runs/iter2_conversion.STOP aborts between/within arms. On any crash: [state] ARM_FAILED + STOP.
#
# Shared STEP 0: merge iter-2 S adapter into mtplus1 base (W += 2.0*B@A, maxabs gate) -> the re-conversion base.
set -u
cd /home/mark/qwen_diffusion
ROOT=/home/mark/qwen_diffusion
export QWEN_DIFFUSION_ROOT="$ROOT"

ENV_PY="$ROOT/.venv-fastdllm/bin/python"
SNAP="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
EXPORT_PY="/home/mark/shared/lumoFlyWheel_codex_fork/scripts/export_qwen35_9b_fastdllm_vllm.py"

SEED="${SEED:-81101}"                     # same seed convention as iteration-1 mswe_S re-conversion
GPU_FREE_MIB="${GPU_FREE_MIB:-3000}"
OUTROOT="$ROOT/runs/kraise_reconvert_iter2"
PIDFILE="$ROOT/runs/iter2_conversion.pid"
STOPFILE="$ROOT/runs/iter2_conversion.STOP"
RUNLOG="$OUTROOT/runner.log"

BASE_ADAPTER="$ROOT/runs/swe_sft_arm1_iter2/Aswe_S_step400_seed71101/checkpoint-400"
INIT_BASE="$ROOT/models/qwen3.5-9b-fastdllm-mtplus1-merged"
MERGED="$ROOT/models/qwen3.5-9b-fastdllm-mswe-S-iter2-merged"
GATE_JSON="$OUTROOT/step1_merge/merge_sanity_gate.json"
DATA="$ROOT/data/flare_redesign_run1_copy_retention_mix"

A_OUT="$OUTROOT/mswe2_S_twinK1_run1recipe_step400_seed${SEED}"
B_OUT="$OUTROOT/mswe2_S_twinV1_run1recipe_step400_seed${SEED}"
A_EXPORT="$ROOT/models/qwen3.5-9b-fastdllm-mswe2-S-twinK1-vllm-bf16"
B_EXPORT="$ROOT/models/qwen3.5-9b-fastdllm-mswe2-S-twinV1-vllm-bf16"
V1_MANIFEST="$B_OUT/v1_copy_span_manifest.json"

mkdir -p "$OUTROOT/step1_merge"
ts() { date -Iseconds; }
log() { echo "[iter2-conv $(ts)] $*" | tee -a "$RUNLOG"; }
state() { echo "[state] $*" | tee -a "$RUNLOG"; }

cleanup() { rm -f "$PIDFILE"; }
fail_arm() { local arm="$1"; local msg="$2"; state "ARM_FAILED arm=$arm reason=\"$msg\""; touch "$STOPFILE"; cleanup; exit 1; }

echo $$ > "$PIDFILE"
trap 'cleanup' EXIT
trap 'fail_arm SIGNAL "runner received SIGTERM/SIGINT"' TERM INT

stop_requested() { [[ -f "$STOPFILE" ]]; }

# ---------- PREFLIGHT ----------
log "PREFLIGHT begin seed=$SEED"
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[[ "${USED:-99999}" -lt "$GPU_FREE_MIB" ]] || fail_arm PREFLIGHT "GPU busy (${USED}MiB >= ${GPU_FREE_MIB})"
FREEG=$(free -g | awk '/Mem/{print $7}')
[[ "${FREEG:-0}" -ge 8 ]] || fail_arm PREFLIGHT "host RAM headroom ${FREEG}G < 8G"
[[ -d "$BASE_ADAPTER" ]] || fail_arm PREFLIGHT "iter2 S adapter missing: $BASE_ADAPTER"
[[ -d "$INIT_BASE" ]] || fail_arm PREFLIGHT "init base missing: $INIT_BASE"
[[ -f "$DATA/train_agentic_mix.json" ]] || fail_arm PREFLIGHT "conversion mix missing: $DATA"
[[ -d "$SNAP" ]] || fail_arm PREFLIGHT "HF snapshot missing: $SNAP"
[[ -f "$EXPORT_PY" ]] || fail_arm PREFLIGHT "export script missing: $EXPORT_PY"
log "PREFLIGHT OK gpu_used=${USED}MiB host_free=${FREEG}G"

# ---------- STEP 0: merge (shared re-conversion base, W += 2.0*B@A, maxabs gate) ----------
if [[ -f "$MERGED/model.safetensors.index.json" && -f "$GATE_JSON" ]] && grep -q '"gate_pass": *true' "$GATE_JSON" 2>/dev/null; then
  log "STEP0 merge: reuse existing $MERGED (gate PASS)"
else
  state "arm=STEP0_merge step=- loss=- eta=merging"
  log "STEP0 merge -> $MERGED"
  if ! systemd-run --user --scope -p MemoryMax=26G -p MemorySwapMax=8G -- \
        "$ENV_PY" "$ROOT/scripts/merge_adapter_into_fastdllm_candidate.py" \
          --init "$INIT_BASE" --adapter "$BASE_ADAPTER" --out "$MERGED" \
          --gate-out "$GATE_JSON" --device cpu >>"$OUTROOT/step0_merge.log" 2>&1; then
    fail_arm STEP0_merge "merge script nonzero exit (KILL-1); see $OUTROOT/step0_merge.log"
  fi
  grep -q '"gate_pass": *true' "$GATE_JSON" 2>/dev/null || fail_arm STEP0_merge "merge_sanity_gate not PASS ($GATE_JSON)"
  log "STEP0 merge PASS ($GATE_JSON)"
fi

# ---------- helper: launch one 400-step re-conversion caged, poll [state], block until done ----------
# args: <arm-label> <output-dir> <extra-env-file-or-empty>
run_conversion() {
  local ARM="$1"; local OUT="$2"; local EXTRA="$3"
  mkdir -p "$OUT"
  local TLOG="$OUT/train.log"
  local CPID="$OUT/train.child.pid"
  : > "$TLOG"
  log "ARM $ARM re-conversion START -> $OUT"

  # the iteration-1 kraise_reconvert recipe (env -> run_flare_redesign_run1.sh), verbatim hyperparameters.
  ( systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=4G -- \
      env \
        ENV_PY="$ENV_PY" \
        QWEN_DIFFUSION_ROOT="$ROOT" \
        DATASET_DIR="$DATA" \
        OUTPUT_DIR="$OUT" \
        MODEL_PATH="$MERGED" \
        MAX_STEPS=400 MAX_TRAIN_SAMPLES=5055 BLOCK_SIZE=512 TRAIN_BD_SIZE=32 GRAD_ACCUM=1 \
        LEARNING_RATE=1e-5 SAVE_STEPS=100 SAVE_TOTAL_LIMIT=6 LOGGING_STEPS=5 \
        LORA_R=16 LORA_ALPHA=32 LORA_DROPOUT=0.05 \
        LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj \
        DISABLE_GROUP_TEXTS=0 TRUNCATION_SIDE=left CONVERSATION_TEMPLATE=fast_dllm_v2_native \
        VALUE_SPAN_LOSS_WEIGHT=2.0 ${EXTRA} \
        SEED="$SEED" DATA_SEED="$SEED" \
        SKIP_DATASET_BUILD=1 OVERWRITE_CACHE=1 MAX_JOBS=4 \
        bash "$ROOT/scripts/run_flare_redesign_run1.sh" >>"$TLOG" 2>&1 ) &
  local WPID=$!
  echo "$WPID" > "$CPID"
  log "ARM $ARM child pid=$WPID log=$TLOG"

  # poll loop: emit [state] step/loss/ETA until child exits
  local first_seen=0
  while kill -0 "$WPID" 2>/dev/null; do
    if stop_requested; then
      log "ARM $ARM: STOP-file present -> killing child $WPID"
      kill "$WPID" 2>/dev/null; sleep 3; kill -9 "$WPID" 2>/dev/null
      fail_arm "$ARM" "STOP-file requested"
    fi
    # last "NN/400" progress + last loss + tqdm ETA remaining
    local prog loss eta
    prog=$(grep -oE '[0-9]+/400' "$TLOG" 2>/dev/null | tail -1)
    loss=$(grep -oE "'loss': [0-9.]+" "$TLOG" 2>/dev/null | tail -1 | grep -oE '[0-9.]+$')
    eta=$(grep -oE '<[0-9:]+,' "$TLOG" 2>/dev/null | tail -1 | tr -d '<,')
    [[ -n "$prog" && "$first_seen" -eq 0 ]] && first_seen=1
    state "arm=$ARM step=${prog:-0/400} loss=${loss:-NA} eta=${eta:-NA}"
    sleep 60
  done
  wait "$WPID"; local rc=$?
  if [[ $rc -ne 0 ]]; then fail_arm "$ARM" "training nonzero exit rc=$rc; see $TLOG"; fi
  [[ -f "$OUT/adapter_model.safetensors" ]] || fail_arm "$ARM" "no final adapter at $OUT"
  local floss
  floss=$(grep -oE "'loss': [0-9.]+" "$TLOG" 2>/dev/null | tail -1 | grep -oE '[0-9.]+$')
  state "arm=$ARM step=400/400 loss=${floss:-NA} eta=done"
  log "ARM $ARM re-conversion DONE rc=0 final_loss=${floss:-NA}"
}

# ---------- helper: export one arm's clean stream to vLLM AR (the HF merged form) ----------
run_export() {
  local ARM="$1"; local ADAPTER="$2"; local OUTMODEL="$3"
  log "ARM $ARM export -> $OUTMODEL"
  if ! systemd-run --user --scope -p MemoryMax=26G -p MemorySwapMax=8G -- \
        "$ENV_PY" "$EXPORT_PY" \
          --official-model "$SNAP" \
          --converted-model "$MERGED" \
          --adapter "$ADAPTER" \
          --output "$OUTMODEL" \
          --overwrite >>"$OUTROOT/${ARM}_export.log" 2>&1; then
    fail_arm "${ARM}_export" "export nonzero exit; see $OUTROOT/${ARM}_export.log"
  fi
  log "ARM $ARM export DONE -> $OUTMODEL"
}

# ============================ ARM A — twin@plain (shipping candidate) ============================
stop_requested && fail_arm ARM_A "STOP-file present before ARM A"
run_conversion "A" "$A_OUT" ""
run_export "A" "$A_OUT" "$A_EXPORT"

# ============================ ARM B — twin@V1 (piggyback, DIRECTIVE-3) ===========================
stop_requested && fail_arm ARM_B "STOP-file present before ARM B"
# V1 extra env: activate the copy-span joint-infill collator; hand value-span masking to the collator
# (VALUE_SPAN_MASK_PROB=0 so the model's blanket forced-value mask is OFF and the curriculum controls copy
# spans while derived values stay forced by the collator). VALUE_SPAN_LOSS_WEIGHT stays 2.0 (loss emphasis).
V1_EXTRA="FASTDLLM_V1_COPY_SPAN=1 VALUE_SPAN_MASK_PROB=0.0 V1_MANIFEST_PATH=$V1_MANIFEST"
run_conversion "B" "$B_OUT" "$V1_EXTRA"
run_export "B" "$B_OUT" "$B_EXPORT"

state "arm=ALL step=DONE loss=- eta=complete"
log "DOUBLE CONVERSION COMPLETE: A=$A_EXPORT  B=$B_EXPORT  v1_manifest=$V1_MANIFEST"
cleanup
