#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv-fastdllm/bin/python}"
BASE_MODEL="${BASE_MODEL:-$ROOT/models/qwen3.5-9b-fastdllm-init}"
ADAPTER="${1:-${ADAPTER:-$ROOT/runs/rl_multiturn_grpo_v6/from_v2_hybrid_mixed35_kl005_g4_step300/adapter_model}}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/rl_multiturn_grpo_v6/from_v2_hybrid_mixed35_kl005_g4_step300_gates}"
TRAIN_PATH="${TRAIN_PATH:-$ROOT/data/flare_redesign_run1_copy_retention_mix/train_agentic_mix.json}"
HELDOUT_NLL="${HELDOUT_NLL:-$ROOT/data/flare_stage1_ab_pilot/heldout_nll.jsonl}"
GSM8K_JSONL="${GSM8K_JSONL:-$ROOT/data/phaseA_retention/gsm8k_main_test_first20.jsonl}"
GSM8K_FEWSHOT_JSONL="${GSM8K_FEWSHOT_JSONL:-$ROOT/data/phaseA_retention/gsm8k_main_train_first5.jsonl}"
MATCHED20_JSONL="${MATCHED20_JSONL:-$ROOT/data/toolcall_eval_native/flare_scaleup_native_58.jsonl}"
NEVERTRAIN_JSONL="${NEVERTRAIN_JSONL:-$ROOT/data/toolcall_eval_native/flare_nevertrain_bfcl_apibank.jsonl}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$ROOT/models/qwen3.5-9b-fastdllm-b1000-vllm-bf16}"

mkdir -p "$OUT_ROOT"

"$PYTHON" scripts/eval_flare_stage1_ab_diffusion.py \
  --base-model "$BASE_MODEL" \
  --adapter-a "$ADAPTER" \
  --adapter-b "$ADAPTER" \
  --model-names A_diffusion_only \
  --train-path "$TRAIN_PATH" \
  --heldout-nll "$HELDOUT_NLL" \
  --gsm8k-path "$GSM8K_JSONL" \
  --gsm8k-fewshot-path "$GSM8K_FEWSHOT_JSONL" \
  --out-dir "$OUT_ROOT/gsm8k_careful_gate" \
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

"$PYTHON" - "$OUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = json.loads((root / "gsm8k_careful_gate" / "summary.json").read_text())
gsm = summary["models"]["A_diffusion_only"]["generation"]["summary"]["gsm8k"]
gate = {
    "strict_correct": int(gsm["strict_correct"]),
    "flex_correct": int(gsm["flex_correct"]),
    "examples": int(gsm["examples"]),
    "strict_accuracy": float(gsm["strict_accuracy"]),
    "flex_accuracy": float(gsm["flex_accuracy"]),
    "bar": 0.65,
}
gate["passed"] = gate["strict_accuracy"] >= gate["bar"] and gate["flex_accuracy"] >= gate["bar"]
(root / "retention_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
print(json.dumps({"retention_gate": gate}, indent=2, sort_keys=True), flush=True)
if not gate["passed"]:
    lines = [
        "# RL-v6 Gates",
        "",
        "Status: retention gate failed; matched-20 and never-train hybrid evals skipped by the ordered gate rule.",
        "",
        f"- GSM8K strict: {gate['strict_correct']}/{gate['examples']} = {gate['strict_accuracy']:.3f}",
        f"- GSM8K flex: {gate['flex_correct']}/{gate['examples']} = {gate['flex_accuracy']:.3f}",
        "- Bar: `>=0.65`",
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    raise SystemExit(2)
PY

"$PYTHON" scripts/eval_flare_northstar_hybrid_clean.py \
  --input-jsonl "$MATCHED20_JSONL" \
  --out-dir "$OUT_ROOT/matched20_hybrid" \
  --adapter "$ADAPTER" \
  --episode-limit 20 \
  --min-turns 3 \
  --max-turns 6 \
  --temperature 0.0 \
  --top-p 0.95 \
  --grammar-topk 256

"$PYTHON" scripts/audit_value_projection_tokens.py \
  --rows "$OUT_ROOT/matched20_hybrid/diffusion_hybrid_forced_grammar_seq_values/turns.jsonl" \
  --tokenizer "$TOKENIZER_PATH" \
  --out-json "$OUT_ROOT/matched20_hybrid/diffusion_hybrid_forced_grammar_seq_values/projection_value_audit.json" \
  --out-jsonl "$OUT_ROOT/matched20_hybrid/diffusion_hybrid_forced_grammar_seq_values/projection_value_audit.jsonl"

"$PYTHON" scripts/eval_flare_northstar_hybrid_clean.py \
  --input-jsonl "$NEVERTRAIN_JSONL" \
  --out-dir "$OUT_ROOT/nevertrain_bfcl_apibank60_hybrid" \
  --adapter "$ADAPTER" \
  --episode-limit 60 \
  --min-turns 1 \
  --max-turns 6 \
  --temperature 0.0 \
  --top-p 0.95 \
  --grammar-topk 256

"$PYTHON" scripts/audit_value_projection_tokens.py \
  --rows "$OUT_ROOT/nevertrain_bfcl_apibank60_hybrid/diffusion_hybrid_forced_grammar_seq_values/turns.jsonl" \
  --tokenizer "$TOKENIZER_PATH" \
  --out-json "$OUT_ROOT/nevertrain_bfcl_apibank60_hybrid/diffusion_hybrid_forced_grammar_seq_values/projection_value_audit.json" \
  --out-jsonl "$OUT_ROOT/nevertrain_bfcl_apibank60_hybrid/diffusion_hybrid_forced_grammar_seq_values/projection_value_audit.jsonl"

"$PYTHON" - "$OUT_ROOT" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
ret = json.loads((root / "retention_gate.json").read_text())
matched = json.loads((root / "matched20_hybrid" / "summary.json").read_text())
never = json.loads((root / "nevertrain_bfcl_apibank60_hybrid" / "summary.json").read_text())
matched_audit = json.loads(
    (root / "matched20_hybrid" / "diffusion_hybrid_forced_grammar_seq_values" / "projection_value_audit.json").read_text()
)["totals"]
never_audit = json.loads(
    (
        root
        / "nevertrain_bfcl_apibank60_hybrid"
        / "diffusion_hybrid_forced_grammar_seq_values"
        / "projection_value_audit.json"
    ).read_text()
)["totals"]
git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

def sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

summary = {
    "git_hash": git_hash,
    "script_sha256": {
        "rl_multiturn_grpo_pilot.py": sha("scripts/rl_multiturn_grpo_pilot.py"),
        "rl_multiturn_tool_env.py": sha("scripts/rl_multiturn_tool_env.py"),
        "eval_flare_stage1_ab_diffusion.py": sha("scripts/eval_flare_stage1_ab_diffusion.py"),
        "eval_flare_northstar_hybrid_clean.py": sha("scripts/eval_flare_northstar_hybrid_clean.py"),
        "audit_value_projection_tokens.py": sha("scripts/audit_value_projection_tokens.py"),
    },
    "retention": ret,
    "matched20_hybrid": {
        "exact_args": int(matched["exact_args"]),
        "turns": int(matched["turns"]),
        "episode_exact": int(matched["episode_exact"]),
        "episodes": int(matched["episodes"]),
        "valid": int(matched["valid_tool_call"]),
        "exact_seq": int(matched["exact_tool_sequence"]),
        "sec_per_turn": float(matched["sec_per_turn"]),
        "forwards_per_turn": float(matched["forwards_per_turn"]),
        "baseline_v2_exact_args": 47,
        "promotion_bar": 50,
        "passed_bar": int(matched["exact_args"]) >= 50,
        "audit_totals": matched_audit,
    },
    "nevertrain_hybrid": {
        "exact_args": int(never["exact_args"]),
        "turns": int(never["turns"]),
        "episode_exact": int(never["episode_exact"]),
        "episodes": int(never["episodes"]),
        "valid": int(never["valid_tool_call"]),
        "exact_seq": int(never["exact_tool_sequence"]),
        "sec_per_turn": float(never["sec_per_turn"]),
        "forwards_per_turn": float(never["forwards_per_turn"]),
        "baseline_v2_exact_args": 83,
        "no_material_regression_floor": 80,
        "passed_floor": int(never["exact_args"]) >= 80 and int(never["valid_tool_call"]) == int(never["turns"]),
        "audit_totals": never_audit,
    },
}
summary["passed"] = (
    ret["passed"]
    and summary["matched20_hybrid"]["passed_bar"]
    and summary["nevertrain_hybrid"]["passed_floor"]
    and int(matched_audit.get("zero_projected_value_tokens_verified") or 0) == 1
    and int(never_audit.get("zero_projected_value_tokens_verified") or 0) == 1
)
(root / "final_eval_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
lines = [
    "# RL-v6 Gates",
    "",
    f"Status: {'PASS' if summary['passed'] else 'FAIL / not promoted'}.",
    "",
    "| eval | result | bar | pass |",
    "| --- | ---: | ---: | --- |",
    f"| GSM8K strict | {ret['strict_correct']}/{ret['examples']} = {ret['strict_accuracy']:.3f} | >=0.65 | {ret['strict_accuracy'] >= 0.65} |",
    f"| GSM8K flex | {ret['flex_correct']}/{ret['examples']} = {ret['flex_accuracy']:.3f} | >=0.65 | {ret['flex_accuracy'] >= 0.65} |",
    f"| matched-20 hybrid exact_args | {summary['matched20_hybrid']['exact_args']}/{summary['matched20_hybrid']['turns']} | >=50/63 | {summary['matched20_hybrid']['passed_bar']} |",
    f"| matched-20 vs v2 hybrid | {summary['matched20_hybrid']['exact_args'] - 47:+d}/63 | >0 and bar 50 | {summary['matched20_hybrid']['passed_bar']} |",
    f"| never-train hybrid exact_args | {summary['nevertrain_hybrid']['exact_args']}/{summary['nevertrain_hybrid']['turns']} | >=80 exact args (v2 floor) | {summary['nevertrain_hybrid']['passed_floor']} |",
    "",
    "Audit:",
    "",
    f"- matched-20 projected value tokens: `{matched_audit.get('projected_value_tokens_exact')}`; verified={matched_audit.get('zero_projected_value_tokens_verified')}",
    f"- never-train projected value tokens: `{never_audit.get('projected_value_tokens_exact')}`; verified={never_audit.get('zero_projected_value_tokens_verified')}",
    "",
    "Harness:",
    "",
    f"- Git hash: `{git_hash}`",
    "- Hybrid sampler: `scripts/eval_flare_northstar_hybrid_clean.py::sample_hybrid_clean`",
    "- GSM8K sampler: `scripts/eval_flare_stage1_ab_diffusion.py::full_context_sample_one`",
]
(root / "report.md").write_text("\n".join(lines) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
raise SystemExit(0 if summary["passed"] else 2)
PY
