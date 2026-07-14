#!/usr/bin/env bash
# X.2 AR-SELF-DISTILLATION conversion runner (SECTION X.2) — SEGMENTED, DETACHED, CAGED,
# KL-INSTRUMENTED. This is the X.1 lesson made binding: X.1 KILLED at the C46 gate because
# a NARROW 5x read-arg weight patched grounding but distorted the broad policy (loop-halts
# 14->34) with KL UNINSTRUMENTED. X.2 fixes BOTH:
#   (1) ON-POLICY target, CONSERVATIVE weight: the read_file limit/offset value slots carry
#       the SAME-WEIGHTS AR teacher's greedy value (deterministic, the 12/48 policy's own
#       conditional), trained at the STANDARD O2 derived-value weight (RWW=2.0). NOT 5x.
#   (2) MANDATORY KL INSTRUMENTATION: between every training segment (GPU released) the
#       s2-kit KL-to-base probe runs on the just-saved checkpoint over a HELD non-read probe
#       set; the KL line is logged to train.log; HARD TRIP at KL>0.05 -> STOP at that
#       checkpoint (the S2 kit rule; the reason A_S2 halted at step 120).
#
# Single-GPU KL is realized the PROVEN s2_pilot way: bit-exact segmented training
# (FASTDLLM_STOP_AT_STEP fixed-HORIZON + RESUME_FROM_CHECKPOINT) so the GPU is FREE between
# segments for the probe. STOP-file aborts. pidfile. You exit while it runs.
set -u
ROOT=/home/mark/qwen_diffusion
cd "$ROOT"
export QWEN_DIFFUSION_ROOT="$ROOT"
ENV_PY="$ROOT/.venv-fastdllm/bin/python"

SEED="${SEED:-81102}"
HORIZON="${HORIZON:-800}"
SEG="${SEG:-100}"
BLOCK="${BLOCK:-2048}"           # 2048 = what X.1 ACTUALLY trained on (.meta: instance_len_max 2048);
                                 # at 4096 the FLARE 2-stream softmax (2L=8192) OOMs the 31GiB card.
RWW="${RWW:-2.0}"                 # X.2 CONSERVATIVE read-window weight (X.1 used 5.0 -> KILLED)
KL_CAP="${KL_CAP:-0.05}"
MERGED="$ROOT/models/qwen3.5-9b-fastdllm-mswe-S-iter2-merged"
DATA="$ROOT/data/swe_x2_ar_distill_mix"
KL_PROBE="$ROOT/runs/kraise_reconvert_iter2_x2/x2_kl_probe.json"
OUT="$ROOT/runs/kraise_reconvert_iter2_x2/mswe2_S_x2_ardistill_h${HORIZON}_seed${SEED}"
PIDFILE="$ROOT/runs/x2_conversion.pid"
STOPFILE="$ROOT/runs/x2_conversion.STOP"
TLOG="$OUT/train.log"
KLLOG="$OUT/kl_to_base.jsonl"

mkdir -p "$OUT"
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT
: > "$TLOG"

ts() { date -Iseconds; }
log() { echo "[x2-conv $(ts)] $*" | tee -a "$TLOG"; }

gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
wait_gpu_idle(){ local dl=$((SECONDS+240)); while :; do local u; u=$(gpu_used); [[ "$u" -lt 3000 ]] && return 0; [[ $SECONDS -gt $dl ]] && { log "WARN gpu not idle (${u}MiB) after wait"; return 0; }; sleep 5; done; }

# ---------- preflight ----------
[[ -d "$MERGED" ]] || { log "FAIL merged base missing $MERGED"; exit 1; }
[[ -f "$DATA/x2_train.json" ]] || { log "FAIL dataset missing $DATA/x2_train.json"; exit 1; }
[[ -f "$KL_PROBE" ]] || { log "FAIL kl probe missing $KL_PROBE"; exit 1; }
U=$(gpu_used); [[ "${U:-99999}" -lt 3000 ]] || { log "FAIL GPU busy ${U}MiB"; exit 1; }
log "PREFLIGHT ok gpu=${U}MiB HORIZON=$HORIZON SEG=$SEG BLOCK=$BLOCK RWW=$RWW KL_CAP=$KL_CAP seed=$SEED out=$OUT"
log "X.1-LESSON: conservative on-policy distill (RWW=$RWW, NOT 5.0) + mandatory KL trip at >$KL_CAP"

# ---------- KL probe helper (s2 kit; GPU exclusive; appends [kl] line; returns 3 on trip) ----------
run_kl() {
  local step="$1"; local adapter="$2"
  wait_gpu_idle
  log "[kl] probe step=$step adapter=${adapter:-<base>}"
  set +e
  systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=8G -- \
    "$ENV_PY" "$ROOT/scripts/s2_kl_probe.py" \
      --base "$MERGED" --adapter "$adapter" \
      --probe "$KL_PROBE" --step "$step" --kl-cap "$KL_CAP" --out "$KLLOG" >>"$TLOG" 2>&1
  local rc=$?
  # (no global set -e; rc captured above)
  local line; line=$(tail -1 "$KLLOG" 2>/dev/null)
  log "[kl] step=$step rc=$rc $line"
  return $rc
}

# ---------- segment helper: bit-exact resumable train to STOP_AT_STEP ----------
latest_ckpt() { ls -d "$OUT"/checkpoint-* 2>/dev/null | sed 's#.*checkpoint-##' | sort -n | tail -1; }

run_segment() {
  local stop="$1"; local resume="$2"; local ovr="$3"
  wait_gpu_idle
  log "SEGMENT -> stop=$stop resume=${resume:-<fresh>} ovr_cache=$ovr"
  set +e
  systemd-run --user --scope -p MemoryMax=26G -p MemorySwapMax=8G -- \
    env \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      ENV_PY="$ENV_PY" \
      QWEN_DIFFUSION_ROOT="$ROOT" \
      DATASET_DIR="$DATA" \
      OUTPUT_DIR="$OUT" \
      MODEL_PATH="$MERGED" \
      MAX_STEPS="$HORIZON" MAX_TRAIN_SAMPLES=6000 BLOCK_SIZE="$BLOCK" TRAIN_BD_SIZE=32 GRAD_ACCUM=1 \
      LEARNING_RATE=1e-5 LR_SCHEDULER_TYPE=cosine WARMUP_RATIO=0.03 \
      SAVE_STEPS="$SEG" SAVE_TOTAL_LIMIT=10 LOGGING_STEPS=5 \
      LORA_R=16 LORA_ALPHA=32 LORA_DROPOUT=0.05 \
      LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj \
      DISABLE_GROUP_TEXTS=1 TRUNCATION_SIDE=left CONVERSATION_TEMPLATE=fast_dllm_v2_native \
      VALUE_SPAN_LOSS_WEIGHT=2.0 VALUE_SPAN_MASK_PROB=0.0 \
      GRADIENT_CHECKPOINTING=1 \
      SEED="$SEED" DATA_SEED="$SEED" \
      SKIP_DATASET_BUILD=1 OVERWRITE_CACHE="$ovr" MAX_JOBS=4 \
      ENTRY_SCRIPT="train_scripts/train_s2_finetune.py" FASTDLLM_S2_PRETOK=1 \
      FASTDLLM_RESUMABLE_CKPT=1 FASTDLLM_STOP_AT_STEP="$stop" RESUME_FROM_CHECKPOINT="$resume" \
      FASTDLLM_READ_WINDOW_ARG_LOSS_WEIGHT="$RWW" \
      FASTDLLM_DEBUG_ARGUMENT_SPAN_LOSS="${FASTDLLM_DEBUG_ARGUMENT_SPAN_LOSS:-8}" \
      bash "$ROOT/scripts/run_flare_redesign_run1.sh" >>"$TLOG" 2>&1
  local rc=$?
  # (no global set -e; rc captured above)
  log "SEGMENT stop=$stop exited rc=$rc"
  return $rc
}

# ================= STEP-0 KL SANITY (proves the instrument is wired; base==base -> KL~0) =========
[[ -f "$STOPFILE" ]] && { log "STOP-file present at start; abort"; exit 1; }
run_kl 0 ""    # adapter="" => base vs base => KL must be ~0; verifiable BEFORE any training

# ================= SEGMENTED TRAIN LOOP with KL early-stop =================
prev=0
for (( stop=SEG; stop<=HORIZON; stop+=SEG )); do
  [[ -f "$STOPFILE" ]] && { log "STOP-file present -> halt before stop=$stop"; break; }
  if [[ "$prev" -eq 0 ]]; then resume=""; ovr=1; else resume="$OUT/checkpoint-$prev"; ovr=0; fi
  run_segment "$stop" "$resume" "$ovr"; src=$?
  if [[ $src -ne 0 ]]; then log "FAIL segment rc=$src at stop=$stop; halting"; touch "$STOPFILE"; break; fi
  now=$(latest_ckpt)
  [[ -z "$now" ]] && { log "FAIL no checkpoint after stop=$stop"; touch "$STOPFILE"; break; }
  ck="$OUT/checkpoint-$now"
  # MANDATORY KL trip
  run_kl "$now" "$ck"; klrc=$?
  if [[ $klrc -eq 3 ]]; then
    log "HARD TRIP: KL>$KL_CAP at checkpoint-$now -> STOP training at this checkpoint (S2 kit rule)"
    log "HALT-STATE checkpoint = $ck (analogue of A_S2 step-120 early-stop)"
    touch "$STOPFILE"
    break
  fi
  prev="$now"
done

log "X.2 conversion loop DONE (last checkpoint = checkpoint-$(latest_ckpt)); GPU idle on exit"
wait_gpu_idle
exit 0
