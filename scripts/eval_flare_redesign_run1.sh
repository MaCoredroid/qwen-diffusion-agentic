#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mark/qwen_diffusion"
ENV_PY="${ENV_PY:-${ROOT}/.venv-fastdllm/bin/python}"
BASE_MODEL="${BASE_MODEL:-${ROOT}/models/qwen3.5-9b-fastdllm-init}"
ADAPTER="${1:-${ADAPTER:-${ROOT}/runs/flare_redesign_run1_copy_grounded_qwen35_9b}}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/runs/flare_redesign_run1_eval}"
SCHEDULE_ROOT="${OUT_ROOT}/native_schedules"

PUBLIC_NATIVE="${PUBLIC_NATIVE:-${ROOT}/data/toolcall_eval_native/public_multicall_qwen_native_smoke.jsonl}"
HELDOUT_SOURCE="${HELDOUT_SOURCE:-${ROOT}/runs/planner_decomposition/heldout_seed_multicall_policy_targets.jsonl}"
HELDOUT_NATIVE="${HELDOUT_NATIVE:-${ROOT}/data/toolcall_eval_native/heldout_seed_multicall_policy_targets_qwen_native.jsonl}"

mkdir -p "${OUT_ROOT}" "${SCHEDULE_ROOT}" "$(dirname "${HELDOUT_NATIVE}")"

if [[ ! -f "${HELDOUT_NATIVE}" || "${REBUILD_NATIVE_HELDOUT:-0}" == "1" ]]; then
    "${ENV_PY}" "${ROOT}/scripts/convert_toolcall_cases_to_qwen_native.py" \
        --input "${HELDOUT_SOURCE}" \
        --output "${HELDOUT_NATIVE}" \
        --kind eval-jsonl
fi

build_schedule() {
    local name="$1"
    local input_jsonl="$2"
    local plan_jsonl="${SCHEDULE_ROOT}/${name}_plan.jsonl"
    local schedule_jsonl="${SCHEDULE_ROOT}/${name}_schedule.jsonl"
    "${ENV_PY}" "${ROOT}/scripts/plan_native_tool_sensitive_blocks.py" \
        --input-jsonl "${input_jsonl}" \
        --out-jsonl "${plan_jsonl}" \
        --tokenizer-path "${BASE_MODEL}" \
        --include-token-ids
    "${ENV_PY}" "${ROOT}/scripts/emit_tool_sensitive_sampler_schedule.py" \
        --input-jsonl "${plan_jsonl}" \
        --out-jsonl "${schedule_jsonl}" \
        --argument-value-block-tokens "${ARGUMENT_VALUE_BLOCK_TOKENS:-8}" \
        --json-structure-block-tokens "${NATIVE_STRUCTURE_BLOCK_TOKENS:-4}" \
        --tiny-block-tokens "${TINY_BLOCK_TOKENS:-1}" \
        --include-token-ids
}

build_schedule heldout "${HELDOUT_NATIVE}"
build_schedule public "${PUBLIC_NATIVE}"

MERGED_SCHEDULE="${SCHEDULE_ROOT}/heldout_public_native_schedule.jsonl"
"${ENV_PY}" - "${SCHEDULE_ROOT}/heldout_schedule.jsonl" "${SCHEDULE_ROOT}/public_schedule.jsonl" "${MERGED_SCHEDULE}" <<'PY'
import sys
out = open(sys.argv[3], "w", encoding="utf-8")
for path in sys.argv[1:3]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                out.write(line)
out.close()
PY

export FASTDLLM_FLARE_GDN_ROUTE="${FASTDLLM_FLARE_GDN_ROUTE:-route_i}"
export FASTDLLM_GDN_KERNEL="${FASTDLLM_GDN_KERNEL:-fla}"
export FASTDLLM_BATCH_FLARE_NOISY_GDN="${FASTDLLM_BATCH_FLARE_NOISY_GDN:-1}"
export FASTDLLM_OPTIMIZE_FLARE_CLEAN_GDN="${FASTDLLM_OPTIMIZE_FLARE_CLEAN_GDN:-1}"

TAUS="${TAUS:-0.00 0.50 0.70 0.80 0.90 0.95 0.99}"
for tau in ${TAUS}; do
    tau_label="$(printf "%s" "${tau}" | tr -d .)"
    tau_dir="${OUT_ROOT}/tau_${tau_label}"
    mkdir -p "${tau_dir}"
    "${ENV_PY}" "${ROOT}/scripts/eval_fastdllm_toolcall_cases.py" \
        --base-model "${BASE_MODEL}" \
        --adapter "${ADAPTER}" \
        --tokenizer-path "${BASE_MODEL}" \
        --no-merge-adapter \
        --conversation-template fast_dllm_v2_native \
        --eval "heldout_native_12:${HELDOUT_NATIVE}:${tau_dir}/heldout_native_12.jsonl:12" \
        --eval "public_native_12:${PUBLIC_NATIVE}:${tau_dir}/public_native_12.jsonl:12" \
        --full-context-sampling \
        --denoise-logit-mode flare_shift \
        --use-block-cache \
        --block-size "${EVAL_BLOCK_SIZE:-32}" \
        --small-block-size "${EVAL_SMALL_BLOCK_SIZE:-32}" \
        --max-new-tokens "${MAX_NEW_TOKENS:-384}" \
        --threshold "${THRESHOLD:-0.9}" \
        --temperature 0.0 \
        --sampler-schedule-jsonl "${MERGED_SCHEDULE}" \
        --parallel-commit-threshold "${tau}" \
        --strip-gold-for-generation
done

"${ENV_PY}" "${ROOT}/scripts/eval_fastdllm_lora_gsm8k_mini.py" \
    --base-model "${BASE_MODEL}" \
    --adapter "${ADAPTER}" \
    --tokenizer-path "${BASE_MODEL}" \
    --no-merge-adapter \
    --out "${OUT_ROOT}/gsm8k_first20.jsonl" \
    --gsm8k-path "${ROOT}/data/phaseA_retention/gsm8k_main_test_first20.jsonl" \
    --gsm8k-fewshot-path "${ROOT}/data/phaseA_retention/gsm8k_main_train_first5.jsonl" \
    --num-examples 20 \
    --prompt-mode phasea_fewshot \
    --decode-mode fastdllm_anywhere \
    --parallel-commit-threshold "${GSM8K_PARALLEL_COMMIT_THRESHOLD:-0.9}" \
    --block-size "${GSM8K_BLOCK_SIZE:-32}" \
    --small-block-size "${GSM8K_SMALL_BLOCK_SIZE:-32}" \
    --max-new-tokens "${GSM8K_MAX_NEW_TOKENS:-256}" \
    --threshold "${GSM8K_THRESHOLD:-0.9}"

"${ENV_PY}" - "${OUT_ROOT}" <<'PY'
import glob
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("tau_*/*.summary.json")):
    data = json.loads(path.read_text())
    totals = data.get("totals", {})
    value_tokens = totals.get("sampler_parallel_commit_value_tokens") or 0
    value_forwards = totals.get("sampler_parallel_commit_value_forward_visits") or 0
    committed = totals.get("sampler_parallel_commit_committed_tokens") or 0
    forwards = totals.get("sampler_parallel_commit_denoise_forwards") or 0
    rows.append(
        {
            "tau": data.get("parallel_commit_threshold"),
            "eval_name": data.get("eval_name"),
            "records": totals.get("records"),
            "valid_tool_json": totals.get("valid_tool_json"),
            "exact_tool_sequence": totals.get("exact_tool_sequence"),
            "exact_arguments": totals.get("exact_arguments"),
            "value_tpf": value_tokens / value_forwards if value_forwards else None,
            "tokens_per_forward": committed / forwards if forwards else None,
            "value_tokens": value_tokens,
            "value_forward_visits": value_forwards,
            "summary_path": str(path),
        }
    )
gsm_summary = root / "gsm8k_first20.summary.json"
gsm = json.loads(gsm_summary.read_text()) if gsm_summary.exists() else {}
report = {
    "adapter": str(root),
    "toolcall_tau_rows": rows,
    "gsm8k": {
        "accuracy": gsm.get("accuracy"),
        "strict_accuracy": gsm.get("strict_accuracy"),
        "correct": gsm.get("correct"),
        "strict_correct": gsm.get("strict_correct"),
        "num_examples": gsm.get("num_examples"),
        "summary_path": str(gsm_summary) if gsm_summary.exists() else None,
    },
}
(root / "run1_eval_report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
PY
