#!/usr/bin/env bash
# SWE-SFT arm-1 (M_swe_S) — one training SEGMENT of the merged-RL-v2 + SWE-SFT
# LoRA, per swe_tuning_campaign_design.md sec.2.3/2.4.
#
# Objective: the certified two-stream FLARE trainer (run_flare_redesign_run1.sh),
# the ONLY proven chunked-resume SFT path on this stack (used by multiturn-SFT
# warmstart, convert-after-RL step-2, and the S2 pilot). Its L_AR component is the
# autoregressive SWE-SFT objective the design's "AR-side SFT" calls for; the
# L_diff component additionally exercises the diffusion stream (harmless; the §3.1
# fresh conversion still excludes the SWE pool, so the sharp preservation test is
# intact). Owner note: pure-AR (mdm 0) is NOT wired as a proven chunked-resume
# path; two-stream is the lowest-regret realization. Retrain-freely applies.
#
# Ingestion: S2 pre-tokenized passthrough (train_s2_finetune.py + FASTDLLM_S2_PRETOK=1)
# consumes serve-exact input_ids+labels -> zero re-tokenization -> native qwen3_xml
# format guaranteed (bypasses the fast_dllm_v2_native preset whitespace divergence
# flagged in the dataset manifest).
#
# Faithful-chunking contract (bit-exact, proven in convert-after-RL step-2):
# MAX_STEPS is ALWAYS the fixed horizon so the cosine LR schedule is identical to a
# single continuous run; segments stop early via FASTDLLM_STOP_AT_STEP and resume
# via RESUME_FROM_CHECKPOINT (HF restores optimizer/scheduler/RNG/data-order).
# NEVER chunk by raising max_steps. SAVE_STEPS=100 also emits {100,200,300,400}
# checkpoints for the design's step-count erosion sweep.
#
# Required env: STOP_AT_STEP           (absolute global_step to stop this segment)
# Optional env: RESUME_FROM_CHECKPOINT (abs path to checkpoint-N to resume from)
#               HORIZON (default 400), BLOCK_SIZE (default 16384), SEG_SEED (71101),
#               DATASET_DIR, OUTPUT_DIR, MAX_TRAIN_SAMPLES
set -euo pipefail
cd /home/mark/qwen_diffusion

: "${STOP_AT_STEP:?set STOP_AT_STEP}"
SEG_SEED="${SEG_SEED:-71101}"
HORIZON="${HORIZON:-400}"
BLOCK_SIZE="${BLOCK_SIZE:-16384}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/runs/swe_sft_arm1/Aswe_S_step${HORIZON}_seed${SEG_SEED}}"
DATASET_DIR="${DATASET_DIR:-$PWD/data/swe_sft_pool/lmflow_pretok}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-323}"

# Fresh segment (no resume) writes the tokenizer/pad cache; resume segments reuse it.
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  OVR_CACHE=0
else
  OVR_CACHE=1
fi

# --- checkpoint-trap mitigations (env-gated, backward compatible) ---
export FASTDLLM_STOP_AT_STEP="${STOP_AT_STEP}"
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
export FASTDLLM_RESUMABLE_CKPT=1

# --- pre-tokenized ingestion (no re-tokenization of the serve-exact ids) ---
export ENTRY_SCRIPT="train_scripts/train_s2_finetune.py"
export FASTDLLM_S2_PRETOK=1

ENV_PY="$PWD/.venv-fastdllm/bin/python" \
DATASET_DIR="${DATASET_DIR}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
MODEL_PATH="$PWD/models/qwen3.5-9b-fastdllm-mtplus1-merged" \
MAX_STEPS="${HORIZON}" MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES}" BLOCK_SIZE="${BLOCK_SIZE}" TRAIN_BD_SIZE=32 GRAD_ACCUM=1 \
LEARNING_RATE=1e-5 LR_SCHEDULER_TYPE=cosine WARMUP_RATIO=0.03 \
SAVE_STEPS="${SAVE_STEPS:-100}" SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-6}" LOGGING_STEPS="${LOGGING_STEPS:-5}" \
LORA_R=16 LORA_ALPHA=32 LORA_DROPOUT=0.05 \
LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj,gate_up_proj,down_proj \
DISABLE_GROUP_TEXTS=1 TRUNCATION_SIDE=left CONVERSATION_TEMPLATE=fast_dllm_v2_native \
VALUE_SPAN_LOSS_WEIGHT=1.0 VALUE_SPAN_MASK_PROB=0.0 \
SEED="${SEG_SEED}" DATA_SEED="${SEG_SEED}" \
SKIP_DATASET_BUILD=1 OVERWRITE_CACHE="${OVR_CACHE}" \
MAX_JOBS=4 \
scripts/run_flare_redesign_run1.sh
