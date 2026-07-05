#!/usr/bin/env bash
# Convert-after-RL STEP 2 — one training SEGMENT of the A_new fresh two-stream conversion.
#
# Faithful-chunking contract:
#   * MAX_STEPS is ALWAYS 400 so the cosine LR scheduler horizon (and 0.03 warmup) is
#     identical to a single 400-step run. Segments stop early via FASTDLLM_STOP_AT_STEP
#     (env-gated StopAtStepCallback) and resume via RESUME_FROM_CHECKPOINT. HF Trainer
#     restores optimizer/scheduler/RNG/data-order state on resume, so the segmented run
#     reproduces the single-run trajectory (this is why cumulative-max_steps chunking is
#     NOT used — it would rebuild the cosine over the wrong horizon and decay to 0 early).
#   * All design-pinned hyperparameters below are verbatim from convert_after_rl_design.md
#     section 4. Only chunk/resume/cache controls are added.
#
# Required env: STOP_AT_STEP   (absolute global_step at which this segment stops)
# Optional env: RESUME_FROM_CHECKPOINT (absolute path to checkpoint-N to resume from)
#               SEG_SEED (default 80101)
set -euo pipefail
cd /home/mark/qwen_diffusion

: "${STOP_AT_STEP:?set STOP_AT_STEP}"
SEG_SEED="${SEG_SEED:-80101}"
OUTPUT_DIR="$PWD/runs/convert_after_rl/Anew_run1recipe_step400_seed${SEG_SEED}"

# Fresh segment (no resume) tokenizes + writes cache; resume segments reuse it.
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  OVR_CACHE=0
else
  OVR_CACHE=1
fi

export FASTDLLM_STOP_AT_STEP="${STOP_AT_STEP}"
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
# Write full resumable checkpoints (adapter@root + optimizer + scheduler + rng + trainer_state).
export FASTDLLM_RESUMABLE_CKPT=1

ENV_PY="$PWD/.venv-fastdllm/bin/python" \
DATASET_DIR="$PWD/data/flare_redesign_run1_copy_retention_mix" \
OUTPUT_DIR="${OUTPUT_DIR}" \
MODEL_PATH="$PWD/models/qwen3.5-9b-fastdllm-mtplus1-merged" \
MAX_STEPS=400 MAX_TRAIN_SAMPLES=5055 BLOCK_SIZE=512 TRAIN_BD_SIZE=32 GRAD_ACCUM=1 \
LEARNING_RATE=1e-5 SAVE_STEPS=100 SAVE_TOTAL_LIMIT=4 LOGGING_STEPS=5 \
LORA_R=16 LORA_ALPHA=32 LORA_DROPOUT=0.05 \
LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj \
DISABLE_GROUP_TEXTS=0 TRUNCATION_SIDE=left CONVERSATION_TEMPLATE=fast_dllm_v2_native \
VALUE_SPAN_LOSS_WEIGHT=2.0 VALUE_SPAN_MASK_PROB=1.0 \
SEED="${SEG_SEED}" DATA_SEED="${SEG_SEED}" \
SKIP_DATASET_BUILD=1 OVERWRITE_CACHE="${OVR_CACHE}" \
MAX_JOBS=4 \
scripts/run_flare_redesign_run1.sh
