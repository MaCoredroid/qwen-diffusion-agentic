#!/usr/bin/env bash
# S2 pilot — one training SEGMENT of the A_S2 trajectory-consistency LoRA.
#
# Objective (spec s2_pilot_design.md sec.2): two-stream FLARE  L = L_AR + L_diff
# on the self-trajectory corpus, mask rate U(0.50,0.90), UNIFORM CE (value-span
# OFF), LoRA r16/a32/drop0.05 on the 9 attn+GDN targets, base M_{t+1}-merged.
# This is the SAME certified run_flare_redesign_run1 chain as convert-after-RL
# step-2, changed ONLY by env: pre-tokenized ingestion + mask rate + value-span
# off + WSD LR + grad-accum 2 + block_size 1152 + one-example-per-sample.
#
# Faithful-chunking contract (proven bit-exact in step-2): MAX_STEPS is ALWAYS
# 400 so the WSD horizon is identical to a single run; segments stop early via
# FASTDLLM_STOP_AT_STEP and resume via RESUME_FROM_CHECKPOINT (HF restores
# optimizer/scheduler/RNG/data-order). NEVER chunk by raising max_steps.
#
# Required env: STOP_AT_STEP           (absolute global_step to stop this segment)
# Optional env: RESUME_FROM_CHECKPOINT (abs path to checkpoint-N to resume from)
set -euo pipefail
cd /home/mark/qwen_diffusion

: "${STOP_AT_STEP:?set STOP_AT_STEP}"
SEG_SEED="${SEG_SEED:-90101}"
OUTPUT_DIR="$PWD/runs/s2_pilot/Apilot_step400_seed${SEG_SEED}"
DATASET_DIR="$PWD/runs/s2_pilot/s2_flare_dataset"

# Fresh segment (no resume) writes the tokenizer cache; resume segments reuse it.
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  OVR_CACHE=0
else
  OVR_CACHE=1
fi

# --- checkpoint-trap mitigations (env-gated, backward compatible) ---
export FASTDLLM_STOP_AT_STEP="${STOP_AT_STEP}"
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
export FASTDLLM_RESUMABLE_CKPT=1

# --- S2 pre-tokenized ingestion (no re-tokenization of cached ids) ---
export ENTRY_SCRIPT="train_scripts/train_s2_finetune.py"
export FASTDLLM_S2_PRETOK=1

# --- S2 objective knobs (vs step-2) ---
export FASTDLLM_FLARE_MASK_RATE_MIN=0.50
export FASTDLLM_FLARE_MASK_RATE_MAX=0.90
# WSD peak 1e-5: warmup 0-40, stable 40-340, decay 340-400 -> 1e-6
export LR_SCHEDULER_TYPE=warmup_stable_decay
export LR_SCHEDULER_KWARGS='{"num_stable_steps":300,"num_decay_steps":60,"min_lr_ratio":0.1}'
export WARMUP_STEPS=40

ENV_PY="$PWD/.venv-fastdllm/bin/python" \
DATASET_DIR="${DATASET_DIR}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
MODEL_PATH="$PWD/models/qwen3.5-9b-fastdllm-mtplus1-merged" \
MAX_STEPS=400 MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-1000}" BLOCK_SIZE=1152 TRAIN_BD_SIZE=32 GRAD_ACCUM=2 \
LEARNING_RATE=1e-5 SAVE_STEPS="${SAVE_STEPS:-100}" SAVE_TOTAL_LIMIT=6 LOGGING_STEPS="${LOGGING_STEPS:-5}" \
LORA_R=16 LORA_ALPHA=32 LORA_DROPOUT=0.05 \
LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj \
DISABLE_GROUP_TEXTS=1 TRUNCATION_SIDE=left CONVERSATION_TEMPLATE=fast_dllm_v2_native \
VALUE_SPAN_LOSS_WEIGHT=1.0 VALUE_SPAN_MASK_PROB=0.0 \
SEED="${SEG_SEED}" DATA_SEED="${SEG_SEED}" \
SKIP_DATASET_BUILD=1 OVERWRITE_CACHE="${OVR_CACHE}" \
MAX_JOBS=4 \
scripts/run_flare_redesign_run1.sh
