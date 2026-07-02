#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_PY="${ENV_PY:-${ROOT}/.venv-fastdllm/bin/python}"
BASE_MODEL="${BASE_MODEL:-${ROOT}/models/qwen3.5-9b-fastdllm-init}"
ADAPTER="${1:-${ADAPTER:-${ROOT}/runs/s1_budget_retrain_r64_qwen35_9b}}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/runs/s1_budget_retrain_r64_qwen35_9b_gate}"
TRAIN_PATH="${TRAIN_PATH:-${ROOT}/data/flare_redesign_run1_copy_retention_mix/train_agentic_mix.json}"
HELDOUT_NLL="${HELDOUT_NLL:-${ROOT}/data/flare_stage1_ab_pilot/heldout_nll.jsonl}"
GSM8K_JSONL="${GSM8K_JSONL:-${ROOT}/data/phaseA_retention/gsm8k_main_test_first20.jsonl}"
GSM8K_FEWSHOT_JSONL="${GSM8K_FEWSHOT_JSONL:-${ROOT}/data/phaseA_retention/gsm8k_main_train_first5.jsonl}"
HELDOUT_NATIVE="${HELDOUT_NATIVE:-${ROOT}/data/toolcall_eval_native/heldout_seed_multicall_policy_targets_qwen_native.jsonl}"
PUBLIC_NATIVE="${PUBLIC_NATIVE:-${ROOT}/data/toolcall_eval_native/public_multicall_qwen_native_smoke.jsonl}"

mkdir -p "$OUT_ROOT"

export FASTDLLM_FLARE_GDN_ROUTE="${FASTDLLM_FLARE_GDN_ROUTE:-route_i}"
export FASTDLLM_GDN_KERNEL="${FASTDLLM_GDN_KERNEL:-torch}"
export FASTDLLM_BATCH_FLARE_NOISY_GDN="${FASTDLLM_BATCH_FLARE_NOISY_GDN:-1}"
export FASTDLLM_OPTIMIZE_FLARE_CLEAN_GDN="${FASTDLLM_OPTIMIZE_FLARE_CLEAN_GDN:-1}"

EMPTY_TOOLCALL="${OUT_ROOT}/empty_toolcall.jsonl"
: > "$EMPTY_TOOLCALL"

"$ENV_PY" scripts/measure_block_quality_curve.py \
  --base-model "$BASE_MODEL" \
  --adapter "$ADAPTER" \
  --gsm8k-jsonl "$GSM8K_JSONL" \
  --gsm8k-fewshot-jsonl "$GSM8K_FEWSHOT_JSONL" \
  --toolcall-jsonl "$EMPTY_TOOLCALL" \
  --out-dir "${OUT_ROOT}/block_quality_curve" \
  --run-name s1_gsm8k_k_curve \
  --block-sizes 32 \
  --k-values 32,16,8,4 \
  --gsm8k-limit 20 \
  --toolcall-limit 0 \
  --max-new-tokens 256 \
  --temperature 0.0 \
  --top-p 0.95 \
  --anchor-block-size 32 \
  --anchor-denoise-steps 32 \
  --anchor-min-strict-accuracy 0.65

"$ENV_PY" scripts/eval_flare_stage1_ab_diffusion.py \
  --base-model "$BASE_MODEL" \
  --adapter-a "$ADAPTER" \
  --adapter-b "$ADAPTER" \
  --model-names A_diffusion_only \
  --train-path "$TRAIN_PATH" \
  --heldout-nll "$HELDOUT_NLL" \
  --gsm8k-path "$GSM8K_JSONL" \
  --gsm8k-fewshot-path "$GSM8K_FEWSHOT_JSONL" \
  --out-dir "${OUT_ROOT}/gsm8k_careful_gate" \
  --skip-nll \
  --generation-tasks gsm8k \
  --generation-limit 20 \
  --generation-batch-size 1 \
  --block-size 32 \
  --small-block-size 32 \
  --max-new-tokens 256 \
  --threshold 0.9 \
  --temperature 0.0 \
  --top-p 0.95

SCHEDULE_ROOT="${OUT_ROOT}/native_schedules"
mkdir -p "$SCHEDULE_ROOT"
for name in heldout public; do
  if [[ "$name" == "heldout" ]]; then
    input_jsonl="$HELDOUT_NATIVE"
  else
    input_jsonl="$PUBLIC_NATIVE"
  fi
  "$ENV_PY" scripts/plan_native_tool_sensitive_blocks.py \
    --input-jsonl "$input_jsonl" \
    --out-jsonl "${SCHEDULE_ROOT}/${name}_plan.jsonl" \
    --tokenizer-path "$BASE_MODEL" \
    --include-token-ids
  "$ENV_PY" scripts/emit_tool_sensitive_sampler_schedule.py \
    --input-jsonl "${SCHEDULE_ROOT}/${name}_plan.jsonl" \
    --out-jsonl "${SCHEDULE_ROOT}/${name}_schedule.jsonl" \
    --argument-value-block-tokens 8 \
    --json-structure-block-tokens 4 \
    --tiny-block-tokens 1 \
    --include-token-ids
done

MERGED_SCHEDULE="${SCHEDULE_ROOT}/heldout_public_native_schedule.jsonl"
"$ENV_PY" - "$SCHEDULE_ROOT/heldout_schedule.jsonl" "$SCHEDULE_ROOT/public_schedule.jsonl" "$MERGED_SCHEDULE" <<'PY'
import sys
with open(sys.argv[3], "w", encoding="utf-8") as out:
    for path in sys.argv[1:3]:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    out.write(line)
PY

BATTERY_DIR="${OUT_ROOT}/native_tau099"
mkdir -p "$BATTERY_DIR"
"$ENV_PY" scripts/eval_fastdllm_toolcall_cases.py \
  --base-model "$BASE_MODEL" \
  --adapter "$ADAPTER" \
  --tokenizer-path "$BASE_MODEL" \
  --no-merge-adapter \
  --conversation-template fast_dllm_v2_native \
  --eval "heldout_native_12:${HELDOUT_NATIVE}:${BATTERY_DIR}/heldout_native_12.jsonl:12" \
  --eval "public_native_12:${PUBLIC_NATIVE}:${BATTERY_DIR}/public_native_12.jsonl:12" \
  --full-context-sampling \
  --denoise-logit-mode flare_shift \
  --use-block-cache \
  --block-size 32 \
  --small-block-size 32 \
  --max-new-tokens 384 \
  --threshold 0.9 \
  --temperature 0.0 \
  --sampler-schedule-jsonl "$MERGED_SCHEDULE" \
  --parallel-commit-threshold 0.99 \
  --strip-gold-for-generation \
  --record-generated-token-ids

"$ENV_PY" scripts/analyze_flare_copyspan_outputs.py \
  --cases "$HELDOUT_NATIVE" \
  --outputs "${BATTERY_DIR}/heldout_native_12.jsonl" \
  --out "${OUT_ROOT}/heldout_copyderived_tau099.json"
"$ENV_PY" scripts/analyze_flare_copyspan_outputs.py \
  --cases "$PUBLIC_NATIVE" \
  --outputs "${BATTERY_DIR}/public_native_12.jsonl" \
  --out "${OUT_ROOT}/public_copyderived_tau099.json"

"$ENV_PY" - "$OUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
block = json.loads((root / "block_quality_curve" / "s1_gsm8k_k_curve.summary.json").read_text())
gsm = json.loads((root / "gsm8k_careful_gate" / "summary.json").read_text())
heldout = json.loads((root / "heldout_copyderived_tau099.json").read_text())
public = json.loads((root / "public_copyderived_tau099.json").read_text())

def copy_group(report):
    only = next(iter(report["runs"].values()))
    return only["groups"].get("copy") or {"exact": 0, "arguments": 0, "accuracy": 0.0}

gsm_model = gsm["models"]["A_diffusion_only"]["generation"]["summary"]["gsm8k"]
anchor = block["anchor_gate"]
held_copy = copy_group(heldout)
pub_copy = copy_group(public)
summary = {
    "block_anchor": {
        "passed": bool(anchor["passed"]),
        "strict_correct": int(anchor.get("correct") or 0),
        "examples": int(anchor.get("examples") or 0),
        "strict_accuracy": float(anchor.get("accuracy") or 0.0),
        "threshold": 0.65,
    },
    "gsm8k_careful": {
        "strict_correct": int(gsm_model["strict_correct"]),
        "examples": int(gsm_model["examples"]),
        "strict_accuracy": float(gsm_model["strict_accuracy"]),
        "flex_correct": int(gsm_model["flex_correct"]),
        "flex_accuracy": float(gsm_model["flex_accuracy"]),
        "threshold": 0.70,
        "passed": float(gsm_model["strict_accuracy"]) >= 0.70,
    },
    "frozen_battery": {
        "heldout_copy_args": {
            "exact": int(held_copy["exact"]),
            "total": int(held_copy["arguments"]),
            "threshold": ">=41/52",
            "passed": int(held_copy["exact"]) >= 41 and int(held_copy["arguments"]) == 52,
        },
        "public_copy_args": {
            "exact": int(pub_copy["exact"]),
            "total": int(pub_copy["arguments"]),
            "threshold": ">=55/60",
            "passed": int(pub_copy["exact"]) >= 55 and int(pub_copy["arguments"]) == 60,
        },
    },
    "block_quality_curve_summary": str(root / "block_quality_curve" / "s1_gsm8k_k_curve.summary.json"),
    "gsm8k_careful_summary": str(root / "gsm8k_careful_gate" / "summary.json"),
    "heldout_copyderived": str(root / "heldout_copyderived_tau099.json"),
    "public_copyderived": str(root / "public_copyderived_tau099.json"),
}
summary["passed"] = (
    summary["block_anchor"]["passed"]
    and summary["gsm8k_careful"]["passed"]
    and summary["frozen_battery"]["heldout_copy_args"]["passed"]
    and summary["frozen_battery"]["public_copy_args"]["passed"]
)
(root / "s1_gate_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
lines = [
    "# S1 Gate Summary",
    "",
    f"- Block anchor: {summary['block_anchor']['strict_correct']}/{summary['block_anchor']['examples']} strict ({summary['block_anchor']['strict_accuracy']:.3f}); pass={summary['block_anchor']['passed']}",
    f"- GSM8K careful: {summary['gsm8k_careful']['strict_correct']}/{summary['gsm8k_careful']['examples']} strict ({summary['gsm8k_careful']['strict_accuracy']:.3f}), flex {summary['gsm8k_careful']['flex_correct']}/{summary['gsm8k_careful']['examples']} ({summary['gsm8k_careful']['flex_accuracy']:.3f}); pass={summary['gsm8k_careful']['passed']}",
    f"- Heldout copy args: {summary['frozen_battery']['heldout_copy_args']['exact']}/{summary['frozen_battery']['heldout_copy_args']['total']}; pass={summary['frozen_battery']['heldout_copy_args']['passed']}",
    f"- Public copy args: {summary['frozen_battery']['public_copy_args']['exact']}/{summary['frozen_battery']['public_copy_args']['total']}; pass={summary['frozen_battery']['public_copy_args']['passed']}",
    "",
    f"Overall pass: `{summary['passed']}`",
]
(root / "s1_gate_report.md").write_text("\n".join(lines) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
raise SystemExit(0 if summary["passed"] else 2)
PY
