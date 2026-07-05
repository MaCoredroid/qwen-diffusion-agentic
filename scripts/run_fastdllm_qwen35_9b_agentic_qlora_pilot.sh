#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mark/qwen_diffusion"
V2="${ROOT}/fast-dllm/v2"
ENV_PY="${ENV_PY:-${ROOT}/.venv-fastdllm/bin/python}"
META_PY="${META_PY:-${ROOT}/.venv/bin/python}"
CUDA_ROOT="${ROOT}/.venv-fastdllm/lib/python3.10/site-packages/nvidia/cu13"

HF_MODEL="${HF_MODEL:-Qwen/Qwen3.5-9B}"
MODEL_PATH="${MODEL_PATH:-${ROOT}/models/qwen3.5-9b-fastdllm-init}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/data/qwen35_9b_diffusion_curriculum}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/runs/fastdllm_qwen35_9b_agentic_qlora_pilot}"
CONVERSATION_TEMPLATE="${CONVERSATION_TEMPLATE:-fast_dllm_v2}"

MAX_STEPS="${MAX_STEPS:-50}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-64}"
BLOCK_SIZE="${BLOCK_SIZE:-512}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
LR_SCHEDULER_KWARGS="${LR_SCHEDULER_KWARGS:-}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
WARMUP_STEPS="${WARMUP_STEPS:-}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
SAVE_STEPS="${SAVE_STEPS:-25}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}"
LORA_MODEL_PATH="${LORA_MODEL_PATH:-}"
ALLOW_UNSUPPORTED_QWEN35_DIFFUSION="${ALLOW_UNSUPPORTED_QWEN35_DIFFUSION:-0}"
BUILD_CURRICULUM="${BUILD_CURRICULUM:-1}"
DISABLE_GROUP_TEXTS="${DISABLE_GROUP_TEXTS:-0}"
TRUNCATION_SIDE="${TRUNCATION_SIDE:-}"
OVERWRITE_CACHE="${OVERWRITE_CACHE:-0}"
STRUCTURAL_LOSS_WEIGHT="${STRUCTURAL_LOSS_WEIGHT:-1.0}"
STRUCTURAL_TOKEN_IDS="${STRUCTURAL_TOKEN_IDS:-}"
STRUCTURAL_TOKEN_MANIFEST="${STRUCTURAL_TOKEN_MANIFEST:-${OUTPUT_DIR}/structural_token_ids.json}"
ARGUMENT_SPAN_LOSS_WEIGHT="${ARGUMENT_SPAN_LOSS_WEIGHT:-1.0}"
ARGUMENT_SPAN_MASK_PROB="${ARGUMENT_SPAN_MASK_PROB:-0.0}"
ARGUMENT_SPAN_START_TOKEN_IDS="${ARGUMENT_SPAN_START_TOKEN_IDS:-}"
ARGUMENT_SPAN_END_TOKEN_IDS="${ARGUMENT_SPAN_END_TOKEN_IDS:-}"
ARGUMENT_SPAN_TOKEN_MANIFEST="${ARGUMENT_SPAN_TOKEN_MANIFEST:-${OUTPUT_DIR}/argument_span_token_ids.json}"
VALUE_COPY_LOSS_WEIGHT="${VALUE_COPY_LOSS_WEIGHT:-1.0}"
VALUE_COPY_TOKEN_IDS="${VALUE_COPY_TOKEN_IDS:-}"
VALUE_COPY_TOKEN_MANIFEST="${VALUE_COPY_TOKEN_MANIFEST:-${OUTPUT_DIR}/value_copy_token_ids.json}"
VALUE_SPAN_LOSS_WEIGHT="${VALUE_SPAN_LOSS_WEIGHT:-1.0}"
VALUE_SPAN_MASK_PROB="${VALUE_SPAN_MASK_PROB:-0.0}"
VALUE_SPAN_LABEL_ONLY="${VALUE_SPAN_LABEL_ONLY:-0}"
VALUE_SPAN_TOKEN_IDS="${VALUE_SPAN_TOKEN_IDS:-}"
VALUE_SPAN_TOKEN_MANIFEST="${VALUE_SPAN_TOKEN_MANIFEST:-${OUTPUT_DIR}/value_span_token_ids.json}"
TRAIN_BD_SIZE="${TRAIN_BD_SIZE:-}"
TRAIN_BD_SIZE_CHOICES="${TRAIN_BD_SIZE_CHOICES:-}"
SEED="${SEED:-42}"
DATA_SEED="${DATA_SEED:-${SEED}}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
GRADIENT_CHECKPOINTING_KWARGS="${GRADIENT_CHECKPOINTING_KWARGS:-{\"use_reentrant\":false}}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
# Entry script is overridable so alternate pre-tokenized ingestion (S2 pilot) can
# reuse this whole runner unchanged. Default is byte-identical to prior behavior.
ENTRY_SCRIPT="${ENTRY_SCRIPT:-train_scripts/finetune.py}"

export CUDA_HOME="${CUDA_ROOT}"
export PATH="${CUDA_HOME}/bin:${ROOT}/.venv-fastdllm/bin:${PATH}"
export LIBRARY_PATH="${CUDA_HOME}/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DS_SKIP_CUDA_CHECK=1
if [[ -f "${MODEL_PATH}/modeling.py" ]]; then
    MODEL_CODE_HASH="$(
        (sha256sum "${MODEL_PATH}/modeling.py" "${MODEL_PATH}/configuration.py" 2>/dev/null || true) | sha256sum | cut -d' ' -f1
    )"
    export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${ROOT}/.hf_modules_cache/${MODEL_CODE_HASH}}"
    mkdir -p "${HF_MODULES_CACHE}"
fi
export FASTDLLM_STRUCTURAL_LOSS_WEIGHT="${STRUCTURAL_LOSS_WEIGHT}"
export FASTDLLM_ARGUMENT_SPAN_LOSS_WEIGHT="${ARGUMENT_SPAN_LOSS_WEIGHT}"
export FASTDLLM_ARGUMENT_SPAN_MASK_PROB="${ARGUMENT_SPAN_MASK_PROB}"
export FASTDLLM_VALUE_COPY_LOSS_WEIGHT="${VALUE_COPY_LOSS_WEIGHT}"
export FASTDLLM_VALUE_SPAN_LOSS_WEIGHT="${VALUE_SPAN_LOSS_WEIGHT}"
export FASTDLLM_VALUE_SPAN_MASK_PROB="${VALUE_SPAN_MASK_PROB}"
export FASTDLLM_VALUE_SPAN_LABEL_ONLY="${VALUE_SPAN_LABEL_ONLY}"
if [[ -n "${TRAIN_BD_SIZE}" ]]; then
    export FASTDLLM_TRAIN_BD_SIZE="${TRAIN_BD_SIZE}"
fi
if [[ -n "${TRAIN_BD_SIZE_CHOICES}" ]]; then
    export FASTDLLM_TRAIN_BD_SIZE_CHOICES="${TRAIN_BD_SIZE_CHOICES}"
fi

mkdir -p "${ROOT}/runs" "${ROOT}/logs" "${OUTPUT_DIR}"

if [[ "${BUILD_CURRICULUM}" == "1" ]]; then
    python3 "${ROOT}/scripts/build_agentic_diffusion_curriculum.py" \
        --out-dir "${DATASET_DIR}"
fi

READINESS_JSON="${OUTPUT_DIR}/readiness.json"
python3 "${ROOT}/scripts/check_qwen35_diffusion_readiness.py" \
    --root "${ROOT}" \
    --model "${HF_MODEL}" \
    --candidate-model-path "${MODEL_PATH}" \
    --training-python "${ENV_PY}" \
    --metadata-python "${META_PY}" \
    --json-out "${READINESS_JSON}" \
    > "${OUTPUT_DIR}/readiness.stdout.json"

READY="$(python3 - "${READINESS_JSON}" <<'PY'
import json
import sys
payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
print("1" if payload.get("ready") else "0")
PY
)"

if [[ "${READY}" != "1" && "${ALLOW_UNSUPPORTED_QWEN35_DIFFUSION}" != "1" ]]; then
    python3 - "${READINESS_JSON}" <<'PY'
import json
import sys
payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
print("Qwen3.5-9B diffusion pilot is not ready. Blockers:")
for item in payload.get("blockers", []):
    print(f"- {item}")
print(f"\nReadiness details: {sys.argv[1]}")
PY
    exit 2
fi

if [[ -n "${STRUCTURAL_TOKEN_IDS}" ]]; then
    export FASTDLLM_STRUCTURAL_TOKEN_IDS="${STRUCTURAL_TOKEN_IDS}"
elif [[ "${STRUCTURAL_LOSS_WEIGHT}" != "1" && "${STRUCTURAL_LOSS_WEIGHT}" != "1.0" ]]; then
    STRUCTURAL_TOKEN_IDS="$("${ENV_PY}" "${ROOT}/scripts/fastdllm_structural_token_ids.py" \
        --tokenizer "${MODEL_PATH}" \
        --json-out "${STRUCTURAL_TOKEN_MANIFEST}")"
    export FASTDLLM_STRUCTURAL_TOKEN_IDS="${STRUCTURAL_TOKEN_IDS}"
fi

NEEDS_ARGUMENT_SPAN_IDS=0
if [[ "${ARGUMENT_SPAN_LOSS_WEIGHT}" != "1" && "${ARGUMENT_SPAN_LOSS_WEIGHT}" != "1.0" ]]; then
    NEEDS_ARGUMENT_SPAN_IDS=1
fi
if [[ "${ARGUMENT_SPAN_MASK_PROB}" != "0" && "${ARGUMENT_SPAN_MASK_PROB}" != "0.0" ]]; then
    NEEDS_ARGUMENT_SPAN_IDS=1
fi
if [[ "${VALUE_SPAN_LOSS_WEIGHT}" != "1" && "${VALUE_SPAN_LOSS_WEIGHT}" != "1.0" ]]; then
    NEEDS_ARGUMENT_SPAN_IDS=1
fi
if [[ "${VALUE_SPAN_MASK_PROB}" != "0" && "${VALUE_SPAN_MASK_PROB}" != "0.0" ]]; then
    NEEDS_ARGUMENT_SPAN_IDS=1
fi
if [[ "${VALUE_SPAN_LABEL_ONLY}" != "0" && "${VALUE_SPAN_LABEL_ONLY}" != "0.0" ]]; then
    NEEDS_ARGUMENT_SPAN_IDS=1
fi

if [[ -n "${ARGUMENT_SPAN_START_TOKEN_IDS}" || -n "${ARGUMENT_SPAN_END_TOKEN_IDS}" ]]; then
    export FASTDLLM_ARGUMENT_SPAN_START_TOKEN_IDS="${ARGUMENT_SPAN_START_TOKEN_IDS}"
    export FASTDLLM_ARGUMENT_SPAN_END_TOKEN_IDS="${ARGUMENT_SPAN_END_TOKEN_IDS}"
elif [[ "${NEEDS_ARGUMENT_SPAN_IDS}" == "1" ]]; then
    ARGUMENT_SPAN_TOKEN_LINE="$("${ENV_PY}" "${ROOT}/scripts/fastdllm_argument_span_token_ids.py" \
        --tokenizer "${MODEL_PATH}" \
        --json-out "${ARGUMENT_SPAN_TOKEN_MANIFEST}")"
    ARGUMENT_SPAN_START_TOKEN_IDS="$(printf "%s" "${ARGUMENT_SPAN_TOKEN_LINE}" | cut -f1)"
    ARGUMENT_SPAN_END_TOKEN_IDS="$(printf "%s" "${ARGUMENT_SPAN_TOKEN_LINE}" | cut -f2)"
    export FASTDLLM_ARGUMENT_SPAN_START_TOKEN_IDS="${ARGUMENT_SPAN_START_TOKEN_IDS}"
    export FASTDLLM_ARGUMENT_SPAN_END_TOKEN_IDS="${ARGUMENT_SPAN_END_TOKEN_IDS}"
fi

if [[ -n "${VALUE_COPY_TOKEN_IDS}" ]]; then
    export FASTDLLM_VALUE_COPY_TOKEN_IDS="${VALUE_COPY_TOKEN_IDS}"
elif [[ "${VALUE_COPY_LOSS_WEIGHT}" != "1" && "${VALUE_COPY_LOSS_WEIGHT}" != "1.0" ]]; then
    VALUE_COPY_TOKEN_IDS="$("${ENV_PY}" "${ROOT}/scripts/fastdllm_value_copy_token_ids.py" \
        --tokenizer "${MODEL_PATH}" \
        --dataset "${DATASET_DIR}" \
        --json-out "${VALUE_COPY_TOKEN_MANIFEST}")"
    export FASTDLLM_VALUE_COPY_TOKEN_IDS="${VALUE_COPY_TOKEN_IDS}"
fi

if [[ -n "${VALUE_SPAN_TOKEN_IDS}" ]]; then
    export FASTDLLM_VALUE_SPAN_TOKEN_IDS="${VALUE_SPAN_TOKEN_IDS}"
elif [[ "${VALUE_SPAN_LOSS_WEIGHT}" != "1" && "${VALUE_SPAN_LOSS_WEIGHT}" != "1.0" ]] || [[ "${VALUE_SPAN_MASK_PROB}" != "0" && "${VALUE_SPAN_MASK_PROB}" != "0.0" ]] || [[ "${VALUE_SPAN_LABEL_ONLY}" != "0" && "${VALUE_SPAN_LABEL_ONLY}" != "0.0" ]]; then
    VALUE_SPAN_TOKEN_IDS="$("${ENV_PY}" "${ROOT}/scripts/fastdllm_value_copy_token_ids.py" \
        --tokenizer "${MODEL_PATH}" \
        --dataset "${DATASET_DIR}" \
        --json-out "${VALUE_SPAN_TOKEN_MANIFEST}")"
    export FASTDLLM_VALUE_SPAN_TOKEN_IDS="${VALUE_SPAN_TOKEN_IDS}"
fi

cd "${V2}"

EXTRA_ARGS=(
    --disable_group_texts "${DISABLE_GROUP_TEXTS}"
)
if [[ -n "${TRUNCATION_SIDE}" ]]; then
    EXTRA_ARGS+=(--truncation_side "${TRUNCATION_SIDE}")
fi
if [[ -n "${LORA_MODEL_PATH}" ]]; then
    EXTRA_ARGS+=(--lora_model_path "${LORA_MODEL_PATH}")
fi
# Resume support for chunked/segmented training (no-op unless RESUME_FROM_CHECKPOINT is set).
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    EXTRA_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
if [[ "${GRADIENT_CHECKPOINTING}" == "1" || "${GRADIENT_CHECKPOINTING}" == "true" || "${GRADIENT_CHECKPOINTING}" == "True" ]]; then
    EXTRA_ARGS+=(--gradient_checkpointing_kwargs "${GRADIENT_CHECKPOINTING_KWARGS}")
fi
if [[ -n "${WARMUP_STEPS}" ]]; then
    EXTRA_ARGS+=(--warmup_steps "${WARMUP_STEPS}")
else
    EXTRA_ARGS+=(--warmup_ratio "${WARMUP_RATIO}")
fi
if [[ -n "${LR_SCHEDULER_KWARGS}" ]]; then
    EXTRA_ARGS+=(--lr_scheduler_kwargs "${LR_SCHEDULER_KWARGS}")
fi

exec "${ENV_PY}" "${ENTRY_SCRIPT}" \
    --model_name_or_path "${MODEL_PATH}" \
    --trust_remote_code 1 \
    --mdm 1 \
    --use_lora 1 \
    --use_qlora 1 \
    --bits 4 \
    --quant_type nf4 \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --lora_target_modules "${LORA_TARGET_MODULES}" \
    --dataset_path "${DATASET_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --overwrite_output_dir \
    --conversation_template "${CONVERSATION_TEMPLATE}" \
    --num_train_epochs 1 \
    --max_steps "${MAX_STEPS}" \
    --max_train_samples "${MAX_TRAIN_SAMPLES}" \
    --overwrite_cache "${OVERWRITE_CACHE}" \
    --learning_rate "${LEARNING_RATE}" \
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
    --seed "${SEED}" \
    --data_seed "${DATA_SEED}" \
    "${EXTRA_ARGS[@]}" \
    --block_size "${BLOCK_SIZE}" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --bf16 \
    --run_name fastdllm_qwen35_9b_agentic_qlora_pilot \
    --validation_split_percentage 0 \
    --logging_steps "${LOGGING_STEPS}" \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps "${SAVE_STEPS}" \
    --dataloader_num_workers 0 \
    --preprocessing_num_workers 1 \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
    --report_to none
