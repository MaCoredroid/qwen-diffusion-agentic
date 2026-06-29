#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/mark/qwen_diffusion}"
PYTHON="${PYTHON:-${ROOT}/.venv-fastdllm/bin/python}"
BASE_MODEL="${BASE_MODEL:-${ROOT}/models/qwen3.5-9b-fastdllm-init}"
RUN_DIR="${RUN_DIR:?Set RUN_DIR to a Fast-DLLM adapter run directory.}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${RUN_DIR}}"
OUT_ROOT="${OUT_ROOT:-${RUN_DIR}_checkpoint_sweep_eval96_modelrepair_max1}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
CHECKPOINTS="${CHECKPOINTS:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
MODEL_REPAIR_MAX_NEW_TOKENS="${MODEL_REPAIR_MAX_NEW_TOKENS:-${MAX_NEW_TOKENS}}"
INCLUDE_AGENTIC_SLICES="${INCLUDE_AGENTIC_SLICES:-1}"
INCLUDE_CONTEXTUAL_PROJECTION="${INCLUDE_CONTEXTUAL_PROJECTION:-1}"
INCLUDE_SEQUENCE_PLANNER="${INCLUDE_SEQUENCE_PLANNER:-1}"
MULTICALL_MAX_NEW_TOKENS="${MULTICALL_MAX_NEW_TOKENS:-384}"
MULTICALL_MODEL_REPAIR_MAX_NEW_TOKENS="${MULTICALL_MODEL_REPAIR_MAX_NEW_TOKENS:-${MULTICALL_MAX_NEW_TOKENS}}"
TOOLRESULT_MAX_NEW_TOKENS="${TOOLRESULT_MAX_NEW_TOKENS:-160}"
TOOLRESULT_MODEL_REPAIR_MAX_NEW_TOKENS="${TOOLRESULT_MODEL_REPAIR_MAX_NEW_TOKENS:-${TOOLRESULT_MAX_NEW_TOKENS}}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
SMALL_BLOCK_SIZE="${SMALL_BLOCK_SIZE:-8}"

cd "${ROOT}"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

if [[ -z "${CHECKPOINTS}" ]]; then
    mapfile -t discovered < <(
        find "${RUN_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' |
            sed 's/^checkpoint-//' |
            sort -n
    )
    CHECKPOINTS="${discovered[*]}"
fi

if [[ -z "${CHECKPOINTS// }" ]]; then
    echo "No checkpoints found under ${RUN_DIR}; set CHECKPOINTS explicitly." >&2
    exit 2
fi

for step in ${CHECKPOINTS}; do
    adapter="${RUN_DIR}/checkpoint-${step}/adapter_model"
    if [[ ! -f "${adapter}/adapter_config.json" ]]; then
        echo "Skipping checkpoint-${step}: missing ${adapter}/adapter_config.json" >&2
        continue
    fi

    out_dir="${OUT_ROOT}/checkpoint-${step}"
    onecall_log_path="${LOG_DIR}/$(basename "${OUT_ROOT}")_checkpoint-${step}_onecall.log"
    mkdir -p "${out_dir}"

    echo "Evaluating checkpoint-${step}: one-call slices" >&2
    PYTHONPATH="${ROOT}/scripts:${ROOT}/fast-dllm/third_party" \
    "${PYTHON}" scripts/eval_fastdllm_toolcall_cases.py \
        --base-model "${BASE_MODEL}" \
        --adapter "${adapter}" \
        --tokenizer-path "${TOKENIZER_PATH}" \
        --no-merge-adapter \
        --conversation-template fast_dllm_v2 \
        --block-size "${BLOCK_SIZE}" \
        --small-block-size "${SMALL_BLOCK_SIZE}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --full-context-sampling \
        --repair-mode schema \
        --constrained-tool-decoding \
        --constrained-max-calls 1 \
        --model-repair-pass \
        --model-repair-max-new-tokens "${MODEL_REPAIR_MAX_NEW_TOKENS}" \
        --eval public_onecall_8:data/toolcall_eval/public_onecall_hermes_smoke.jsonl:"${out_dir}/public_onecall_8.jsonl":8 \
        --eval teacher_train_labelaware_12:data/toolcall_eval/public_onecall_teacher_train_labelaware_smoke.jsonl:"${out_dir}/teacher_train_labelaware_12.jsonl":12 \
        --eval teacher_heldout_labelaware_8:data/toolcall_eval/public_onecall_teacher_heldout_labelaware_smoke.jsonl:"${out_dir}/teacher_heldout_labelaware_8.jsonl":8 \
        > "${onecall_log_path}" 2>&1

    if [[ "${INCLUDE_AGENTIC_SLICES}" == "1" ]]; then
        multicall_log_path="${LOG_DIR}/$(basename "${OUT_ROOT}")_checkpoint-${step}_multicall.log"
        echo "Evaluating checkpoint-${step}: public multi-call slice" >&2
        PYTHONPATH="${ROOT}/scripts:${ROOT}/fast-dllm/third_party" \
        "${PYTHON}" scripts/eval_fastdllm_toolcall_cases.py \
            --base-model "${BASE_MODEL}" \
            --adapter "${adapter}" \
            --tokenizer-path "${TOKENIZER_PATH}" \
            --no-merge-adapter \
            --conversation-template fast_dllm_v2 \
            --block-size "${BLOCK_SIZE}" \
            --small-block-size "${SMALL_BLOCK_SIZE}" \
            --max-new-tokens "${MULTICALL_MAX_NEW_TOKENS}" \
            --full-context-sampling \
            --repair-mode schema \
            --constrained-tool-decoding \
            --model-repair-pass \
            --model-repair-max-new-tokens "${MULTICALL_MODEL_REPAIR_MAX_NEW_TOKENS}" \
            --eval public_multicall_12:data/toolcall_eval/public_multicall_hermes_smoke.jsonl:"${out_dir}/public_multicall_12.jsonl":12 \
            > "${multicall_log_path}" 2>&1

        sequence_log_path="${LOG_DIR}/$(basename "${OUT_ROOT}")_checkpoint-${step}_multicall_sequence_preserve.log"
        echo "Rescoring checkpoint-${step}: sequence-preserving public multi-call projection" >&2
        "${PYTHON}" scripts/rescore_fastdllm_toolcall_outputs.py \
            --cases-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
            --input-jsonl "${out_dir}/public_multicall_12.jsonl" \
            --out-jsonl "${out_dir}/public_multicall_12_sequence_preserve.jsonl" \
            --text-field constrained_assistant \
            --repair-mode none \
            --constrained-tool-decoding \
            --sequence-preserving-constrained \
            > "${sequence_log_path}" 2>&1

        if [[ "${INCLUDE_CONTEXTUAL_PROJECTION}" == "1" ]]; then
            contextual_log_path="${LOG_DIR}/$(basename "${OUT_ROOT}")_checkpoint-${step}_multicall_contextual_projection.log"
            echo "Rescoring checkpoint-${step}: contextual public multi-call projection" >&2
            "${PYTHON}" scripts/rescore_scalar_repair_contextual_projection.py \
                --cases-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
                --input-jsonl "${out_dir}/public_multicall_12_sequence_preserve.jsonl" \
                --text-field constrained_assistant \
                --out-jsonl "${out_dir}/public_multicall_12_contextual_projection.jsonl" \
                > "${contextual_log_path}" 2>&1
        fi

        if [[ "${INCLUDE_SEQUENCE_PLANNER}" == "1" ]]; then
            planner_input="${out_dir}/public_multicall_12_sequence_preserve.jsonl"
            planner_text_field="constrained_assistant"
            if [[ -f "${out_dir}/public_multicall_12_contextual_projection.jsonl" ]]; then
                planner_input="${out_dir}/public_multicall_12_contextual_projection.jsonl"
                planner_text_field="contextual_projection_assistant"
            fi
            planner_log_path="${LOG_DIR}/$(basename "${OUT_ROOT}")_checkpoint-${step}_multicall_sequence_planner.log"
            echo "Rescoring checkpoint-${step}: guarded public multi-call sequence planner" >&2
            "${PYTHON}" scripts/rescore_toolcall_sequence_planner_projection.py \
                --cases-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
                --input-jsonl "${planner_input}" \
                --text-field "${planner_text_field}" \
                --out-jsonl "${out_dir}/public_multicall_12_sequence_planner_projection.jsonl" \
                > "${planner_log_path}" 2>&1
        fi

        toolresult_log_path="${LOG_DIR}/$(basename "${OUT_ROOT}")_checkpoint-${step}_toolresult.log"
        echo "Evaluating checkpoint-${step}: synthetic tool-result slices" >&2
        PYTHONPATH="${ROOT}/scripts:${ROOT}/fast-dllm/third_party" \
        "${PYTHON}" scripts/eval_fastdllm_toolcall_cases.py \
            --base-model "${BASE_MODEL}" \
            --adapter "${adapter}" \
            --tokenizer-path "${TOKENIZER_PATH}" \
            --no-merge-adapter \
            --conversation-template fast_dllm_v2 \
            --block-size "${BLOCK_SIZE}" \
            --small-block-size "${SMALL_BLOCK_SIZE}" \
            --max-new-tokens "${TOOLRESULT_MAX_NEW_TOKENS}" \
            --full-context-sampling \
            --repair-mode schema \
            --constrained-tool-decoding \
            --constrained-max-calls 1 \
            --model-repair-pass \
            --model-repair-max-new-tokens "${TOOLRESULT_MODEL_REPAIR_MAX_NEW_TOKENS}" \
            --eval synthetic_toolresult_10:data/toolcall_eval/synthetic_toolresult_smoke.jsonl:"${out_dir}/synthetic_toolresult_10.jsonl":10 \
            --eval synthetic_openai_toolresult_10:data/toolcall_eval/synthetic_toolresult_openai_smoke.jsonl:"${out_dir}/synthetic_openai_toolresult_10.jsonl":10 \
            > "${toolresult_log_path}" 2>&1
    fi
done

summary_path="${OUT_ROOT}/checkpoint_sweep_summary.tsv"
{
    printf "checkpoint\teval\trecords\traw_valid\traw_seq\traw_args\tconstrained_seq\tconstrained_args\tmodel_repair_seq\tmodel_repair_args\textra_records\tmissing_records\trepeated_records\ttokens_per_second\n"
    find "${OUT_ROOT}" -mindepth 2 -maxdepth 2 -name '*.summary.json' -print |
        grep -v '_argdiff\.summary\.json$' |
        sort |
        while read -r summary; do
            checkpoint="$(basename "$(dirname "${summary}")")"
            jq -r --arg checkpoint "${checkpoint}" \
                'def summary_eval_name:
                    .eval_name // (.out_jsonl | split("/")[-1] | sub("\\.jsonl$"; ""));
                def is_projected: (.totals.projected? != null);
                def is_planned: (.totals.planned? != null);
                def is_projection: (is_projected or is_planned);
                def records: .totals.records;
                def raw_valid:
                    if is_projection then .totals.input.valid_tool_json else .totals.valid_tool_json end;
                def raw_seq:
                    if is_projection then .totals.input.exact_tool_sequence else .totals.exact_tool_sequence end;
                def raw_args:
                    if is_projection then .totals.input.exact_arguments else .totals.exact_arguments end;
                def constrained_seq:
                    if is_projected then .totals.projected.exact_tool_sequence
                    elif is_planned then .totals.planned.exact_tool_sequence
                    else .totals.constrained_exact_tool_sequence end;
                def constrained_args:
                    if is_projected then .totals.projected.exact_arguments
                    elif is_planned then .totals.planned.exact_arguments
                    else .totals.constrained_exact_arguments end;
                def extra_records:
                    if is_projected then .totals.projected.records_with_extra_calls
                    elif is_planned then .totals.planned.records_with_extra_calls
                    else .totals.records_with_extra_calls end;
                def missing_records:
                    if is_projected then .totals.projected.records_with_missing_calls
                    elif is_planned then .totals.planned.records_with_missing_calls
                    else .totals.records_with_missing_calls end;
                def repeated_records:
                    if is_projected then .totals.projected.records_with_repeated_calls
                    elif is_planned then .totals.planned.records_with_repeated_calls
                    else .totals.records_with_repeated_calls end;
                [
                    $checkpoint,
                    summary_eval_name,
                    records,
                    raw_valid,
                    raw_seq,
                    raw_args,
                    constrained_seq,
                    constrained_args,
                    (if is_projection then "" else .totals.model_repair_exact_tool_sequence end),
                    (if is_projection then "" else .totals.model_repair_exact_arguments end),
                    extra_records,
                    missing_records,
                    repeated_records,
                    (.generated_tokens_per_second // "")
                ] | @tsv' "${summary}"
        done
} | tee "${summary_path}"

echo "Wrote ${summary_path}" >&2
