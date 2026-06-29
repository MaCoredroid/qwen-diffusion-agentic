#!/usr/bin/env python3
import argparse
import json
import re
import shlex
import subprocess
import time
from pathlib import Path


ROOT = Path("/home/mark/qwen_diffusion")
PYTHON = ".venv-fastdllm/bin/python"

DEFAULT_BASE_MODEL = "models/qwen3.5-9b-fastdllm-init"
DEFAULT_ADAPTER = (
    "runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/"
    "checkpoint-275/adapter_model"
)
DEFAULT_CASES = "data/toolcall_eval/synthetic_multicall_failure_analogues.jsonl"
DEFAULT_PLANNER = "runs/synthetic_multicall_failure_analogues/sequence_planner_bad_draft_safe_seqmismatch.jsonl"
DEFAULT_ROUTE = "synthetic_multicall_failure_evidence_selector"


def rooted(path_text):
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def rel(path):
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def slugify(text):
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return slug.strip("_") or "route"


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
    return json.loads(Path(path).read_text(encoding="utf-8"))


def count_jsonl(path_text):
    path = rooted(path_text)
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def artifact_map(route_name):
    route = slugify(route_name)
    route_root = f"runs/tool_sensitive_block_plans/{route}"
    return {
        "blocks": f"{route_root}/blocks_tokenized_with_ids.jsonl",
        "schedule_ids": f"{route_root}/sampler_schedule_with_ids.jsonl",
        "schedule_candidates": f"{route_root}/sampler_schedule_with_candidates_evidence.jsonl",
        "examples": f"data/candidate_ranking/{route}_toolname_argument_ranking_evidence.jsonl",
        "tournament": f"runs/candidate_ranking/{route}_ckpt275_pairwise_tournament.jsonl",
        "schedule_choices": f"{route_root}/sampler_schedule_with_evidence_pairwise_choices.jsonl",
        "generation": f"{route_root}/evidence_toolargselector_structguard_ckpt275_generation.jsonl",
        "audit": f"{route_root}/candidate_miss_audit.jsonl",
    }


def build_steps(args, artifacts):
    return [
        {
            "name": "plan_live_blocks",
            "kind": "cpu",
            "output": artifacts["blocks"],
            "command": [
                PYTHON,
                "scripts/plan_tool_sensitive_blocks.py",
                "--input-jsonl",
                args.planner_jsonl,
                "--text-field",
                args.text_field,
                "--out-jsonl",
                artifacts["blocks"],
                "--tokenizer-path",
                args.base_model,
                "--include-token-ids",
            ],
        },
        {
            "name": "emit_sampler_schedule",
            "kind": "cpu",
            "output": artifacts["schedule_ids"],
            "command": [
                PYTHON,
                "scripts/emit_tool_sensitive_sampler_schedule.py",
                "--input-jsonl",
                artifacts["blocks"],
                "--out-jsonl",
                artifacts["schedule_ids"],
                "--include-token-ids",
            ],
        },
        {
            "name": "augment_evidence_candidates",
            "kind": "cpu",
            "output": artifacts["schedule_candidates"],
            "command": [
                PYTHON,
                "scripts/augment_schedule_value_candidates.py",
                "--schedule-jsonl",
                artifacts["schedule_ids"],
                "--cases-jsonl",
                args.cases_jsonl,
                "--tokenizer-path",
                args.base_model,
                "--out-jsonl",
                artifacts["schedule_candidates"],
                "--selected-candidate-mode",
                "none",
            ],
        },
        {
            "name": "build_selector_examples",
            "kind": "cpu",
            "output": artifacts["examples"],
            "command": [
                PYTHON,
                "scripts/build_candidate_ranking_examples.py",
                "--schedule-jsonl",
                artifacts["schedule_candidates"],
                "--cases-jsonl",
                args.cases_jsonl,
                "--out-jsonl",
                artifacts["examples"],
            ],
        },
        {
            "name": "selector_tournament",
            "kind": "gpu",
            "output": artifacts["tournament"],
            "command": [
                PYTHON,
                "scripts/eval_fastdllm_candidate_pairwise_tournament.py",
                "--base-model",
                args.base_model,
                "--adapter",
                args.adapter,
                "--tokenizer-path",
                args.base_model,
                "--examples-jsonl",
                artifacts["examples"],
                "--out-jsonl",
                artifacts["tournament"],
                "--no-merge-adapter",
            ],
        },
        {
            "name": "inject_selector_choices",
            "kind": "cpu",
            "output": artifacts["schedule_choices"],
            "command": [
                PYTHON,
                "scripts/inject_pairwise_tournament_schedule_choices.py",
                "--schedule-jsonl",
                artifacts["schedule_candidates"],
                "--selector-jsonl",
                artifacts["tournament"],
                "--out-jsonl",
                artifacts["schedule_choices"],
                "--include-kinds",
                "tool_name",
                "argument_value",
            ],
        },
        {
            "name": "generate_with_evidence_selector_schedule",
            "kind": "gpu",
            "output": artifacts["generation"],
            "command": [
                PYTHON,
                "scripts/eval_fastdllm_toolcall_cases.py",
                "--base-model",
                args.base_model,
                "--adapter",
                args.adapter,
                "--tokenizer-path",
                args.base_model,
                "--input-jsonl",
                args.cases_jsonl,
                "--out-jsonl",
                artifacts["generation"],
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--conversation-template",
                "fast_dllm_v2",
                "--full-context-sampling",
                "--sampler-schedule-jsonl",
                artifacts["schedule_choices"],
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
                str(args.constrained_max_calls),
                "--no-merge-adapter",
            ],
        },
        {
            "name": "audit_generation",
            "kind": "cpu",
            "output": artifacts["audit"],
            "command": [
                PYTHON,
                "scripts/analyze_toolcall_candidate_misses.py",
                "--eval-jsonl",
                artifacts["generation"],
                "--cases-jsonl",
                args.cases_jsonl,
                "--schedule-jsonl",
                artifacts["schedule_choices"],
                "--out-jsonl",
                artifacts["audit"],
            ],
        },
    ]


def render_shell(plan):
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "cd /home/mark/qwen_diffusion",
        "",
        "# Generated by scripts/run_qwen35_evidence_selector_route.py",
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


def write_plan(args, artifacts, steps):
    out_root = rooted(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "status": "qwen35-evidence-selector-route-plan",
        "date": "2026-06-28",
        "route": slugify(args.route_name),
        "planner_jsonl": args.planner_jsonl,
        "cases_jsonl": args.cases_jsonl,
        "text_field": args.text_field,
        "base_model": args.base_model,
        "adapter": args.adapter,
        "max_new_tokens": args.max_new_tokens,
        "constrained_max_calls": args.constrained_max_calls,
        "artifacts": artifacts,
        "steps": steps,
    }
    plan_json = out_root / "route_plan.json"
    plan_sh = out_root / "route_plan.sh"
    plan_json.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plan_sh.write_text(render_shell(plan), encoding="utf-8")
    return plan, plan_json, plan_sh


def check(records, name, actual, expected):
    ok = actual == expected
    records.append({"name": name, "actual": actual, "expected": expected, "pass": ok})
    return ok


def check_true(records, name, actual):
    ok = bool(actual)
    records.append({"name": name, "actual": actual, "expected": True, "pass": ok})
    return ok


def verify_existing(plan, out_root):
    artifacts = plan["artifacts"]
    records = []
    missing = []
    for label, path_text in artifacts.items():
        path = rooted(path_text)
        if not path.exists():
            missing.append({"artifact": label, "path": path_text})
        summary = summary_path(path_text)
        if not summary.exists():
            missing.append({"artifact": f"{label}_summary", "path": rel(summary)})
    if missing:
        verification = {
            "status": "qwen35-evidence-selector-route-verification",
            "source_plan": rel(rooted(out_root) / "route_plan.json"),
            "all_pass": False,
            "missing": missing,
            "records": records,
        }
        rooted(out_root).mkdir(parents=True, exist_ok=True)
        (rooted(out_root) / "route_plan_verification.json").write_text(
            json.dumps(verification, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return verification

    case_records = count_jsonl(plan["cases_jsonl"])

    candidate_totals = load_json(summary_path(artifacts["schedule_candidates"]))["totals"]
    check(
        records,
        "candidate_records",
        candidate_totals.get("records"),
        case_records,
    )
    check(
        records,
        "argument_candidate_sequence_coverage",
        candidate_totals.get("argument_blocks_with_sequence_candidates"),
        candidate_totals.get("argument_blocks_augmented"),
    )
    check(
        records,
        "tool_name_candidate_sequence_coverage",
        candidate_totals.get("tool_name_blocks_with_sequence_candidates"),
        candidate_totals.get("tool_name_blocks_augmented"),
    )

    example_totals = load_json(summary_path(artifacts["examples"]))["totals"]
    check(records, "selector_target_missing", example_totals.get("target_missing_from_candidates"), 0)
    check(records, "selector_usable", example_totals.get("usable_for_training"), example_totals.get("examples"))
    check_true(records, "selector_examples_nonzero", example_totals.get("examples", 0) > 0)

    tournament_totals = load_json(summary_path(artifacts["tournament"]))["totals"]
    check(records, "selector_correct", tournament_totals.get("correct"), example_totals.get("examples"))
    check(
        records,
        "selector_correct_argument_value",
        tournament_totals.get("correct:argument_value", 0),
        example_totals.get("examples:argument_value", 0),
    )
    check(
        records,
        "selector_correct_tool_name",
        tournament_totals.get("correct:tool_name", 0),
        example_totals.get("examples:tool_name", 0),
    )

    injection_totals = load_json(summary_path(artifacts["schedule_choices"]))["totals"]
    check(records, "injected_selectors", injection_totals.get("selectors"), example_totals.get("examples"))
    check(records, "injected_candidate_missing_items", injection_totals.get("candidate_missing_items"), 0)

    generation_totals = load_json(summary_path(artifacts["generation"]))["totals"]
    check(records, "generation_records", generation_totals.get("records"), case_records)
    check(records, "generation_valid_tool_json", generation_totals.get("valid_tool_json"), case_records)
    check(records, "generation_exact_tool_sequence", generation_totals.get("exact_tool_sequence"), case_records)
    check(records, "generation_exact_arguments", generation_totals.get("exact_arguments"), case_records)
    check(records, "generation_schema_valid", generation_totals.get("all_schema_valid"), case_records)
    check(records, "generation_required_args", generation_totals.get("all_required_args_present"), case_records)

    audit_totals = load_json(summary_path(artifacts["audit"]))["totals"]
    check(records, "audit_failed_records", audit_totals.get("failed_records"), 0)
    check(records, "audit_mismatches", audit_totals.get("mismatches"), 0)
    check(records, "audit_invalid_tool_blocks", audit_totals.get("invalid_tool_blocks"), 0)

    verification = {
        "status": "qwen35-evidence-selector-route-verification",
        "source_plan": rel(rooted(out_root) / "route_plan.json"),
        "all_pass": all(record["pass"] for record in records),
        "missing": missing,
        "records": records,
    }
    (rooted(out_root) / "route_plan_verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
    parser = argparse.ArgumentParser(description="Plan, execute, and verify a Qwen3.5 evidence-selector route.")
    parser.add_argument("--route-name", default=DEFAULT_ROUTE)
    parser.add_argument("--out-root", default=f"runs/tool_sensitive_block_plans/{DEFAULT_ROUTE}")
    parser.add_argument("--cases-jsonl", default=DEFAULT_CASES)
    parser.add_argument("--planner-jsonl", default=DEFAULT_PLANNER)
    parser.add_argument("--text-field", default="sequence_planner_assistant")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--max-new-tokens", type=int, default=560)
    parser.add_argument("--constrained-max-calls", type=int, default=3)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()

    artifacts = artifact_map(args.route_name)
    steps = build_steps(args, artifacts)
    plan, plan_json, plan_sh = write_plan(args, artifacts, steps)
    print(plan_json)
    print(plan_sh)

    if args.execute:
        execution = execute_steps(steps)
        execution_json = rooted(args.out_root) / "route_plan_execution.json"
        execution_json.write_text(json.dumps(execution, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(execution_json)
        if not execution["ok"]:
            print(json.dumps(execution, indent=2), flush=True)
            raise SystemExit(1)

    if args.verify_existing:
        verification = verify_existing(plan, args.out_root)
        print(rooted(args.out_root) / "route_plan_verification.json")
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
