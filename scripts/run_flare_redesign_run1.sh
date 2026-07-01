#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mark/qwen_diffusion"
ENV_PY="${ENV_PY:-${ROOT}/.venv-fastdllm/bin/python}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/data/flare_redesign_run1_copy_retention_mix}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/runs/flare_redesign_run1_copy_grounded_qwen35_9b}"
MODEL_PATH="${MODEL_PATH:-${ROOT}/models/qwen3.5-9b-fastdllm-init}"

mkdir -p "${OUTPUT_DIR}"

if [[ "${SKIP_DATASET_BUILD:-0}" != "1" ]]; then
    "${ENV_PY}" "${ROOT}/scripts/build_flare_redesign_run1_copy_mix.py" \
        --model "${MODEL_PATH}" \
        --out-dir "${DATASET_DIR}" \
        --block-size "${BLOCK_SIZE:-512}" \
        --truncation-side left
fi

ARGUMENT_SPAN_TOKEN_LINE="$("${ENV_PY}" "${ROOT}/scripts/fastdllm_argument_span_token_ids.py" \
    --tokenizer "${MODEL_PATH}" \
    --start-fragment "<parameter=" \
    --end-fragment "</parameter>" \
    --json-out "${OUTPUT_DIR}/native_argument_span_token_ids.json")"
ARGUMENT_SPAN_START_TOKEN_IDS="$(printf "%s" "${ARGUMENT_SPAN_TOKEN_LINE}" | cut -f1)"
ARGUMENT_SPAN_END_TOKEN_IDS="$(printf "%s" "${ARGUMENT_SPAN_TOKEN_LINE}" | cut -f2)"
export ARGUMENT_SPAN_START_TOKEN_IDS
export ARGUMENT_SPAN_END_TOKEN_IDS

export FASTDLLM_FLARE_TWO_STREAM=1
export FLARE_TWO_STREAM=1
export FASTDLLM_FLARE_GDN_ROUTE="${FASTDLLM_FLARE_GDN_ROUTE:-route_i}"
export FASTDLLM_FLARE_MASK_RATE_MIN="${FASTDLLM_FLARE_MASK_RATE_MIN:-0.3}"
export FASTDLLM_FLARE_MASK_RATE_MAX="${FASTDLLM_FLARE_MASK_RATE_MAX:-0.8}"
export FASTDLLM_FLARE_ADAPTIVE_COPY_SCHEDULE="${FASTDLLM_FLARE_ADAPTIVE_COPY_SCHEDULE:-1}"
export FASTDLLM_FLARE_HIGH_ENTROPY_MASK_RATE_MIN="${FASTDLLM_FLARE_HIGH_ENTROPY_MASK_RATE_MIN:-0.02}"
export FASTDLLM_FLARE_HIGH_ENTROPY_MASK_RATE_MAX="${FASTDLLM_FLARE_HIGH_ENTROPY_MASK_RATE_MAX:-0.12}"
export FASTDLLM_BATCH_FLARE_NOISY_GDN="${FASTDLLM_BATCH_FLARE_NOISY_GDN:-1}"
export FASTDLLM_OPTIMIZE_FLARE_CLEAN_GDN="${FASTDLLM_OPTIMIZE_FLARE_CLEAN_GDN:-1}"
export FASTDLLM_GDN_KERNEL="${FASTDLLM_GDN_KERNEL:-fla}"
export FASTDLLM_FLARE_DEBUG="${FASTDLLM_FLARE_DEBUG:-2}"
export FASTDLLM_DEBUG_ARGUMENT_SPAN_LOSS="${FASTDLLM_DEBUG_ARGUMENT_SPAN_LOSS:-2}"

exec env \
    MODEL_PATH="${MODEL_PATH}" \
    DATASET_DIR="${DATASET_DIR}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    BUILD_CURRICULUM=0 \
    OVERWRITE_CACHE="${OVERWRITE_CACHE:-1}" \
    MAX_STEPS="${MAX_STEPS:-400}" \
    MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-5055}" \
    BLOCK_SIZE="${BLOCK_SIZE:-512}" \
    TRAIN_BD_SIZE="${TRAIN_BD_SIZE:-32}" \
    GRAD_ACCUM="${GRAD_ACCUM:-1}" \
    LEARNING_RATE="${LEARNING_RATE:-1e-5}" \
    SAVE_STEPS="${SAVE_STEPS:-100}" \
    SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-4}" \
    LOGGING_STEPS="${LOGGING_STEPS:-5}" \
    LORA_R="${LORA_R:-16}" \
    LORA_ALPHA="${LORA_ALPHA:-32}" \
    LORA_DROPOUT="${LORA_DROPOUT:-0.05}" \
    LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj}" \
    DISABLE_GROUP_TEXTS="${DISABLE_GROUP_TEXTS:-0}" \
    TRUNCATION_SIDE="${TRUNCATION_SIDE:-left}" \
    CONVERSATION_TEMPLATE="${CONVERSATION_TEMPLATE:-fast_dllm_v2_native}" \
    VALUE_SPAN_LOSS_WEIGHT="${VALUE_SPAN_LOSS_WEIGHT:-2.0}" \
    VALUE_SPAN_MASK_PROB="${VALUE_SPAN_MASK_PROB:-1.0}" \
    VALUE_SPAN_TOKEN_MANIFEST="${VALUE_SPAN_TOKEN_MANIFEST:-${OUTPUT_DIR}/value_span_token_ids.json}" \
    ARGUMENT_SPAN_TOKEN_MANIFEST="${ARGUMENT_SPAN_TOKEN_MANIFEST:-${OUTPUT_DIR}/argument_span_token_ids.json}" \
    SEED="${SEED:-71101}" \
    DATA_SEED="${DATA_SEED:-71101}" \
    "${ROOT}/scripts/run_fastdllm_qwen35_9b_agentic_qlora_pilot.sh"
