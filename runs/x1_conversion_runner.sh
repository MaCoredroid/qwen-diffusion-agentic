#!/usr/bin/env bash
# X.1 P0 re-conversion runner (SECTION X.1(a)+(b)) — SINGLE ARM, detached, caged.
#
#   twin@X1 : the iter-2 primary (mswe-S-iter2-merged) re-converted with
#     (a) READ_WINDOW_ARG read-grounding-weighted L_diff (FASTDLLM_READ_WINDOW_ARG_LOSS_WEIGHT),
#     (b) high-context read-arg curriculum (data/swe_x1_read_grounding_mix, block 4096),
#     (c) 800 steps (double the plain 400).
#   Pretok FLARE two-stream ingestion (train_s2_finetune.py + FASTDLLM_S2_PRETOK=1).
#   NEW output dir; existing twins untouched. STOP-file aborts. pidfile.
set -u
ROOT=/home/mark/qwen_diffusion
cd "$ROOT"
export QWEN_DIFFUSION_ROOT="$ROOT"
ENV_PY="$ROOT/.venv-fastdllm/bin/python"

SEED="${SEED:-81101}"
STEPS="${STEPS:-800}"
BLOCK="${BLOCK:-4096}"
RWW="${RWW:-5.0}"
MERGED="$ROOT/models/qwen3.5-9b-fastdllm-mswe-S-iter2-merged"
DATA="$ROOT/data/swe_x1_read_grounding_mix"
OUT="$ROOT/runs/kraise_reconvert_iter2_x1/mswe2_S_x1_readground_step${STEPS}_seed${SEED}"
PIDFILE="$ROOT/runs/x1_conversion.pid"
STOPFILE="$ROOT/runs/x1_conversion.STOP"
TLOG="$OUT/train.log"

mkdir -p "$OUT"
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT
: > "$TLOG"

ts() { date -Iseconds; }
log() { echo "[x1-conv $(ts)] $*" | tee -a "$TLOG"; }

# preflight
[[ -d "$MERGED" ]] || { log "FAIL merged base missing"; exit 1; }
[[ -f "$DATA/x1_train.json" ]] || { log "FAIL dataset missing"; exit 1; }
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[[ "${USED:-99999}" -lt 3000 ]] || { log "FAIL GPU busy ${USED}MiB"; exit 1; }
log "PREFLIGHT ok gpu_used=${USED}MiB steps=$STEPS block=$BLOCK rww=$RWW seed=$SEED out=$OUT"

# X.1 envs inherited by the pilot/finetuner through run_flare_redesign_run1.sh
export ENTRY_SCRIPT="train_scripts/train_s2_finetune.py"
export FASTDLLM_S2_PRETOK=1
export FASTDLLM_READ_WINDOW_ARG_LOSS_WEIGHT="$RWW"   # X.1(a) read-grounding-weighted L_diff
export FASTDLLM_DEBUG_ARGUMENT_SPAN_LOSS="${FASTDLLM_DEBUG_ARGUMENT_SPAN_LOSS:-8}"

systemd-run --user --scope -p MemoryMax=26G -p MemorySwapMax=8G -- \
  env \
    ENV_PY="$ENV_PY" \
    QWEN_DIFFUSION_ROOT="$ROOT" \
    DATASET_DIR="$DATA" \
    OUTPUT_DIR="$OUT" \
    MODEL_PATH="$MERGED" \
    MAX_STEPS="$STEPS" MAX_TRAIN_SAMPLES=6000 BLOCK_SIZE="$BLOCK" TRAIN_BD_SIZE=32 GRAD_ACCUM=1 \
    LEARNING_RATE=1e-5 SAVE_STEPS=200 SAVE_TOTAL_LIMIT=6 LOGGING_STEPS=5 \
    LORA_R=16 LORA_ALPHA=32 LORA_DROPOUT=0.05 \
    LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj \
    DISABLE_GROUP_TEXTS=1 TRUNCATION_SIDE=left CONVERSATION_TEMPLATE=fast_dllm_v2_native \
    VALUE_SPAN_LOSS_WEIGHT=2.0 VALUE_SPAN_MASK_PROB=0.0 \
    GRADIENT_CHECKPOINTING=1 \
    SEED="$SEED" DATA_SEED="$SEED" \
    SKIP_DATASET_BUILD=1 OVERWRITE_CACHE=1 MAX_JOBS=4 \
    ENTRY_SCRIPT="$ENTRY_SCRIPT" FASTDLLM_S2_PRETOK=1 \
    FASTDLLM_READ_WINDOW_ARG_LOSS_WEIGHT="$RWW" \
    FASTDLLM_DEBUG_ARGUMENT_SPAN_LOSS="$FASTDLLM_DEBUG_ARGUMENT_SPAN_LOSS" \
    bash "$ROOT/scripts/run_flare_redesign_run1.sh" >>"$TLOG" 2>&1

RC=$?
log "conversion exited rc=$RC"
exit $RC
