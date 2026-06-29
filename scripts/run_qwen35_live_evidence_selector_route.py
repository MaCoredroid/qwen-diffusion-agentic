#!/usr/bin/env python3
import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path


ROOT = Path("/home/mark/qwen_diffusion")
PYTHON = ".venv-fastdllm/bin/python"
DEFAULT_OUT_ROOT = ROOT / "runs/tool_sensitive_block_plans/live_v5_evidence_selector_route"

BASE_MODEL = "models/qwen3.5-9b-fastdllm-init"
ADAPTER = "runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model"
CASES = "data/toolcall_eval/public_multicall_hermes_smoke.jsonl"
PLANNER_JSONL = (
    "runs/qwen35_9b_split_route_sidecar_eval/live_smoke_public_multicall_planner/"
    "public_multi_call_planner/public_multicall_12_sequence_planner_segmentargs_v5_voice_safe.jsonl"
)

ARTIFACTS = {
    "blocks": "runs/tool_sensitive_block_plans/public_multicall_live_v5_sequence_planned_blocks_tokenized_with_ids.jsonl",
    "schedule_ids": "runs/tool_sensitive_block_plans/public_multicall_live_v5_sequence_planned_sampler_schedule_with_ids.jsonl",
    "schedule_candidates": "runs/tool_sensitive_block_plans/public_multicall_live_v5_sequence_planned_sampler_schedule_with_candidates_evidence_v5.jsonl",
    "examples": "data/candidate_ranking/public_multicall_live_v5_sequence_planned_toolname_argument_ranking_evidence_v5.jsonl",
    "tournament": "runs/candidate_ranking/public_multicall_live_v5_sequence_planned_evidence_v5_ckpt275_pairwise_tournament.jsonl",
    "schedule_choices": "runs/tool_sensitive_block_plans/public_multicall_live_v5_sequence_planned_sampler_schedule_with_evidence_pairwise_choices.jsonl",
    "generation": "runs/tool_sensitive_block_plans/public_multicall_live_v5_evidence_toolargselector_structguard_ckpt275_generation.jsonl",
    "audit": "runs/tool_sensitive_block_plans/public_multicall_live_v5_evidence_toolargselector_structguard_ckpt275_candidate_miss_audit.jsonl",
}


def rooted(path_text):
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def rel(path):
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def shell_join(parts):
    return " ".join(shlex.quote(str(part)) for part in parts)


def scoped(command):
    return [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "-p",
        "MemoryMax=28G",
        "-p",
        "MemorySwapMax=4G",
        *command,
    ]


def summary_path(path_text):
    return rooted(path_text).with_suffix(".summary.json")


def load_json(path):
    return json.loads(rooted(path).read_text(encoding="utf-8"))


def build_steps():
    return [
        {
            "name": "plan_live_blocks",
            "kind": "cpu",
            "output": ARTIFACTS["blocks"],
            "command": [
                PYTHON,
                "scripts/plan_tool_sensitive_blocks.py",
                "--input-jsonl",
                PLANNER_JSONL,
                "--text-field",
                "sequence_planner_assistant",
                "--out-jsonl",
                ARTIFACTS["blocks"],
                "--tokenizer-path",
                BASE_MODEL,
                "--include-token-ids",
            ],
        },
        {
            "name": "emit_sampler_schedule",
            "kind": "cpu",
            "output": ARTIFACTS["schedule_ids"],
            "command": [
                PYTHON,
                "scripts/emit_tool_sensitive_sampler_schedule.py",
                "--input-jsonl",
                ARTIFACTS["blocks"],
                "--out-jsonl",
                ARTIFACTS["schedule_ids"],
                "--include-token-ids",
            ],
        },
        {
            "name": "augment_evidence_candidates",
            "kind": "cpu",
            "output": ARTIFACTS["schedule_candidates"],
            "command": [
                PYTHON,
                "scripts/augment_schedule_value_candidates.py",
                "--schedule-jsonl",
                ARTIFACTS["schedule_ids"],
                "--cases-jsonl",
                CASES,
                "--tokenizer-path",
                BASE_MODEL,
                "--out-jsonl",
                ARTIFACTS["schedule_candidates"],
                "--selected-candidate-mode",
                "none",
            ],
        },
        {
            "name": "build_selector_examples",
            "kind": "cpu",
            "output": ARTIFACTS["examples"],
            "command": [
                PYTHON,
                "scripts/build_candidate_ranking_examples.py",
                "--schedule-jsonl",
                ARTIFACTS["schedule_candidates"],
                "--cases-jsonl",
                CASES,
                "--out-jsonl",
                ARTIFACTS["examples"],
            ],
        },
        {
            "name": "selector_tournament",
            "kind": "gpu",
            "output": ARTIFACTS["tournament"],
            "command": [
                PYTHON,
                "scripts/eval_fastdllm_candidate_pairwise_tournament.py",
                "--examples-jsonl",
                ARTIFACTS["examples"],
                "--out-jsonl",
                ARTIFACTS["tournament"],
            ],
        },
        {
            "name": "inject_selector_choices",
            "kind": "cpu",
            "output": ARTIFACTS["schedule_choices"],
            "command": [
                PYTHON,
                "scripts/inject_pairwise_tournament_schedule_choices.py",
                "--schedule-jsonl",
                ARTIFACTS["schedule_candidates"],
                "--selector-jsonl",
                ARTIFACTS["tournament"],
                "--out-jsonl",
                ARTIFACTS["schedule_choices"],
                "--include-kinds",
                "tool_name",
                "argument_value",
            ],
        },
        {
            "name": "generate_with_evidence_selector_schedule",
            "kind": "gpu",
            "output": ARTIFACTS["generation"],
            "command": [
                PYTHON,
                "scripts/eval_fastdllm_toolcall_cases.py",
                "--base-model",
                BASE_MODEL,
                "--adapter",
                ADAPTER,
                "--tokenizer-path",
                BASE_MODEL,
                "--input-jsonl",
                CASES,
                "--out-jsonl",
                ARTIFACTS["generation"],
                "--max-new-tokens",
                "560",
                "--conversation-template",
                "fast_dllm_v2",
                "--full-context-sampling",
                "--sampler-schedule-jsonl",
                ARTIFACTS["schedule_choices"],
                "--force-schedule-token-kinds",
                "json_key,json_structure,tool_tag",
                "--force-argument-boundary-target-tokens",
                "--force-best-candidate-sequence",
                "--force-best-tool-name-sequence",
                "--ban-argument-boundary-tokens",
                "--ban-argument-newline-tokens",
                "--stop-after-schedule-tool-calls",
                "--constrained-tool-decoding",
                "--constrained-sequence-preserving",
                "--constrained-max-calls",
                "3",
                "--no-merge-adapter",
            ],
        },
        {
            "name": "audit_generation",
            "kind": "cpu",
            "output": ARTIFACTS["audit"],
            "command": [
                PYTHON,
                "scripts/analyze_toolcall_candidate_misses.py",
                "--eval-jsonl",
                ARTIFACTS["generation"],
                "--cases-jsonl",
                CASES,
                "--schedule-jsonl",
                ARTIFACTS["schedule_choices"],
                "--out-jsonl",
                ARTIFACTS["audit"],
            ],
        },
    ]


def render_shell(plan):
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "cd /home/mark/qwen_diffusion",
        "",
        "# Generated by scripts/run_qwen35_live_evidence_selector_route.py",
        "",
    ]
    for step in plan["steps"]:
        output = rooted(step["output"])
        lines.append(f"# {step['name']}")
        lines.append(f"mkdir -p {shlex.quote(str(output.parent))}")
        command = scoped(step["command"]) if step["kind"] == "gpu" else step["command"]
        lines.append(shell_join(command))
        lines.append("")
    return "\n".join(lines)


def write_plan(out_root):
    steps = build_steps()
    plan = {
        "status": "qwen35-live-evidence-selector-route-plan",
        "date": "2026-06-28",
        "route": "public_multi_call_live_v5_evidence_selector",
        "planner_jsonl": PLANNER_JSONL,
        "cases_jsonl": CASES,
        "base_model": BASE_MODEL,
        "adapter": ADAPTER,
        "artifacts": ARTIFACTS,
        "gates": {
            "candidate_argument_blocks_with_sequence_candidates": 100,
            "selector_examples_usable": 131,
            "selector_correct": 131,
            "selector_correct_argument_value": 100,
            "selector_correct_tool_name": 31,
            "injected_selectors": 131,
            "injected_candidate_missing_items": 0,
            "generation_exact_tool_sequence": 12,
            "generation_exact_arguments": 12,
            "generation_valid_tool_json": 12,
            "audit_failed_records": 0,
            "audit_mismatches": 0,
        },
        "steps": steps,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    plan_json = out_root / "route_plan.json"
    plan_sh = out_root / "route_plan.sh"
    plan_json.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plan_sh.write_text(render_shell(plan), encoding="utf-8")
    return plan, plan_json, plan_sh


def check_equal(records, name, actual, expected):
    ok = actual == expected
    records.append({"name": name, "actual": actual, "expected": expected, "pass": ok})
    return ok


def verify_existing(out_root):
    records = []
    missing = []
    for label, path_text in ARTIFACTS.items():
        path = rooted(path_text)
        if not path.exists():
            missing.append({"artifact": label, "path": path_text})
        summary = summary_path(path_text)
        if not summary.exists():
            missing.append({"artifact": f"{label}_summary", "path": rel(summary)})
    if missing:
        return {"status": "qwen35-live-evidence-selector-route-verification", "all_pass": False, "missing": missing, "records": records}

    candidate_summary = load_json(summary_path(ARTIFACTS["schedule_candidates"]))
    candidate_totals = candidate_summary["totals"]
    check_equal(records, "argument_blocks_with_sequence_candidates", candidate_totals.get("argument_blocks_with_sequence_candidates"), 100)
    check_equal(records, "tool_name_blocks_with_sequence_candidates", candidate_totals.get("tool_name_blocks_with_sequence_candidates"), 31)

    examples_summary = load_json(summary_path(ARTIFACTS["examples"]))
    example_totals = examples_summary["totals"]
    check_equal(records, "selector_examples", example_totals.get("examples"), 131)
    check_equal(records, "selector_usable_for_training", example_totals.get("usable_for_training"), 131)
    check_equal(records, "selector_target_missing", example_totals.get("target_missing_from_candidates"), 0)

    tournament_summary = load_json(summary_path(ARTIFACTS["tournament"]))
    tournament_totals = tournament_summary["totals"]
    check_equal(records, "selector_correct", tournament_totals.get("correct"), 131)
    check_equal(records, "selector_correct_argument_value", tournament_totals.get("correct:argument_value"), 100)
    check_equal(records, "selector_correct_tool_name", tournament_totals.get("correct:tool_name"), 31)

    injection_summary = load_json(summary_path(ARTIFACTS["schedule_choices"]))
    injection_totals = injection_summary["totals"]
    check_equal(records, "injected_selectors", injection_totals.get("selectors"), 131)
    check_equal(records, "injected_candidate_missing_items", injection_totals.get("candidate_missing_items"), 0)

    generation_summary = load_json(summary_path(ARTIFACTS["generation"]))
    generation_totals = generation_summary["totals"]
    check_equal(records, "generation_valid_tool_json", generation_totals.get("valid_tool_json"), 12)
    check_equal(records, "generation_exact_tool_sequence", generation_totals.get("exact_tool_sequence"), 12)
    check_equal(records, "generation_exact_arguments", generation_totals.get("exact_arguments"), 12)

    audit_summary = load_json(summary_path(ARTIFACTS["audit"]))
    audit_totals = audit_summary["totals"]
    check_equal(records, "audit_failed_records", audit_totals.get("failed_records"), 0)
    check_equal(records, "audit_mismatches", audit_totals.get("mismatches"), 0)
    check_equal(records, "audit_invalid_tool_blocks", audit_totals.get("invalid_tool_blocks"), 0)

    verification = {
        "status": "qwen35-live-evidence-selector-route-verification",
        "source_plan": rel(out_root / "route_plan.json"),
        "all_pass": all(record["pass"] for record in records),
        "missing": missing,
        "records": records,
    }
    verify_json = out_root / "route_plan_verification.json"
    verify_json.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return verification


def execute_steps(steps):
    results = []
    for step in steps:
        rooted(step["output"]).parent.mkdir(parents=True, exist_ok=True)
        command = scoped(step["command"]) if step["kind"] == "gpu" else step["command"]
        started = time.time()
        completed = subprocess.run(command, cwd=ROOT, check=False)
        elapsed = time.time() - started
        result = {
            "name": step["name"],
            "kind": step["kind"],
            "output": step["output"],
            "returncode": completed.returncode,
            "elapsed_seconds": elapsed,
        }
        results.append(result)
        if completed.returncode != 0:
            return {"ok": False, "failed_step": result, "results": results}
    return {"ok": True, "results": results}


def main():
    parser = argparse.ArgumentParser(description="Plan, verify, or execute the Qwen3.5 live evidence selector route.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    plan, plan_json, plan_sh = write_plan(args.out_root)
    print(plan_json)
    print(plan_sh)

    if args.execute:
        execution = execute_steps(plan["steps"])
        execution_json = args.out_root / "route_plan_execution.json"
        execution_json.write_text(json.dumps(execution, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(execution_json)
        if not execution["ok"]:
            print(json.dumps(execution, indent=2), flush=True)
            raise SystemExit(1)

    if args.verify_existing:
        verification = verify_existing(args.out_root)
        print(args.out_root / "route_plan_verification.json")
        print(
            json.dumps(
                {
                    "ok": verification["all_pass"],
                    "checks": len(verification["records"]),
                    "missing": len(verification["missing"]),
                    "failed": sum(1 for record in verification["records"] if not record["pass"]),
                },
                indent=2,
            ),
            flush=True,
        )
        if not verification["all_pass"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
