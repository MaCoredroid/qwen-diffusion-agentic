#!/usr/bin/env python3
import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_MANIFEST = ROOT / "runs/qwen35_9b_split_route_sidecar_eval/route_manifest.json"
DEFAULT_OUT_ROOT = ROOT / "runs/qwen35_9b_split_route_sidecar_eval/replay_plan"
DEFAULT_PLAN_JSON = DEFAULT_OUT_ROOT / "route_runner_plan.json"
PYTHON = ".venv-fastdllm/bin/python"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rooted(path_text):
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def rel(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def slug(text):
    return "".join(ch if ch.isalnum() else "_" for ch in text.lower()).strip("_")


def route_matches(route, selected):
    if not selected:
        return True
    labels = {item.lower() for item in selected}
    return route["slice"].lower() in labels or slug(route["slice"]) in labels


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


def optional_flag(parts, condition, flag):
    if condition:
        parts.append(flag)


def optional_value(parts, value, flag):
    if value is not None:
        parts.extend([flag, str(value)])


def eval_command(summary, out_jsonl):
    command = [
        PYTHON,
        "scripts/eval_fastdllm_toolcall_cases.py",
        "--base-model",
        summary["base_model"],
        "--adapter",
        summary["adapter"],
        "--tokenizer-path",
        summary.get("tokenizer_path") or summary["base_model"],
        "--input-jsonl",
        summary["input_jsonl"],
        "--out-jsonl",
        rel(out_jsonl),
    ]
    optional_value(command, summary.get("totals", {}).get("records"), "--limit")
    for key, flag in [
        ("block_size", "--block-size"),
        ("small_block_size", "--small-block-size"),
        ("max_new_tokens", "--max-new-tokens"),
        ("threshold", "--threshold"),
        ("temperature", "--temperature"),
        ("top_p", "--top-p"),
        ("conversation_template", "--conversation-template"),
    ]:
        optional_value(command, summary.get(key), flag)
    optional_flag(command, bool(summary.get("full_context_sampling")), "--full-context-sampling")
    optional_flag(command, bool(summary.get("use_block_cache")), "--use-block-cache")
    optional_value(command, summary.get("repair_mode"), "--repair-mode")
    optional_flag(command, bool(summary.get("constrained_tool_decoding")), "--constrained-tool-decoding")
    optional_value(command, summary.get("constrained_max_calls"), "--constrained-max-calls")
    optional_flag(command, bool(summary.get("model_repair_pass")), "--model-repair-pass")
    optional_value(command, summary.get("model_repair_max_new_tokens"), "--model-repair-max-new-tokens")
    if summary.get("merge_adapter") is False:
        command.append("--no-merge-adapter")
    return command


def rescore_command(summary, input_jsonl, out_jsonl):
    if "min_input_calls_for_plan" in summary:
        command = [
            PYTHON,
            "scripts/rescore_toolcall_sequence_planner_projection.py",
            "--cases-jsonl",
            summary["cases_jsonl"],
            "--input-jsonl",
            rel(input_jsonl),
            "--out-jsonl",
            rel(out_jsonl),
        ]
        optional_value(command, summary.get("text_field"), "--text-field")
        optional_value(command, summary.get("min_input_calls_for_plan"), "--min-input-calls-for-plan")
        if summary.get("prefer_segment_args") is False:
            command.append("--no-prefer-segment-args")
        return command, "sequence_planner_projection"

    if "replacement_counts" in summary:
        command = [
            PYTHON,
            "scripts/rescore_scalar_repair_contextual_projection.py",
            "--cases-jsonl",
            summary["cases_jsonl"],
            "--input-jsonl",
            rel(input_jsonl),
            "--out-jsonl",
            rel(out_jsonl),
        ]
        optional_value(command, summary.get("text_field"), "--text-field")
        return command, "contextual_projection"

    if "constrained_tool_decoding" in summary:
        command = [
            PYTHON,
            "scripts/rescore_fastdllm_toolcall_outputs.py",
            "--cases-jsonl",
            summary["cases_jsonl"],
            "--input-jsonl",
            rel(input_jsonl),
            "--out-jsonl",
            rel(out_jsonl),
        ]
        optional_value(command, summary.get("text_field"), "--text-field")
        optional_value(command, summary.get("repair_mode"), "--repair-mode")
        optional_flag(command, bool(summary.get("constrained_tool_decoding")), "--constrained-tool-decoding")
        optional_flag(command, bool(summary.get("sequence_preserving_constrained")), "--sequence-preserving-constrained")
        optional_value(command, summary.get("constrained_max_calls"), "--constrained-max-calls")
        return command, "toolcall_rescore"

    return None, "unknown"


def check_path(path_text, missing, kind):
    if not path_text:
        missing.append({"kind": kind, "path": None})
        return
    path = rooted(path_text)
    if not path.exists():
        missing.append({"kind": kind, "path": rel(path)})


def validate_manifest(manifest):
    missing = []
    for role, adapter in manifest.get("adapter_roles", {}).items():
        check_path(adapter, missing, f"adapter:{role}")
    for base_model in manifest.get("base_models", []):
        check_path(base_model, missing, "base_model")
    for route in manifest.get("routes", []):
        check_path(route.get("input_cases_jsonl"), missing, f"cases:{route.get('slice')}")
        check_path(route.get("generation_summary_json"), missing, f"generation_summary:{route.get('slice')}")
        check_path(route.get("routed_summary_json"), missing, f"routed_summary:{route.get('slice')}")
        for item in route.get("postprocess_chain", []):
            check_path(item.get("summary_json"), missing, f"chain_summary:{route.get('slice')}")
            check_path(item.get("out_jsonl"), missing, f"chain_output:{route.get('slice')}")
            check_path(item.get("input_jsonl"), missing, f"chain_input:{route.get('slice')}")
            if item.get("cases_jsonl"):
                check_path(item.get("cases_jsonl"), missing, f"chain_cases:{route.get('slice')}")
    return missing


def metrics_from_summary(summary):
    totals = summary["totals"]
    records = totals["records"]
    raw = totals.get("input", totals)
    if "planned" in totals:
        protected = totals["planned"]
    else:
        protected = totals

    if "constrained_exact_tool_sequence" in protected:
        protected_seq = protected["constrained_exact_tool_sequence"]
        protected_args = protected["constrained_exact_arguments"]
    else:
        protected_seq = protected.get("exact_tool_sequence", "")
        protected_args = protected.get("exact_arguments", "")

    return {
        "records": records,
        "raw_valid": raw.get("valid_tool_json", ""),
        "raw_seq": raw.get("exact_tool_sequence", ""),
        "raw_args": raw.get("exact_arguments", ""),
        "protected_seq": protected_seq,
        "protected_args": protected_args,
    }


def gate_checks(route, metrics):
    checks = []
    for metric, minimum in route["gates"].items():
        actual = metrics.get(metric)
        checks.append(
            {
                "metric": metric,
                "actual": actual,
                "minimum": minimum,
                "records": metrics["records"],
                "pass": isinstance(actual, int) and actual >= minimum,
            }
        )
    return checks


def verify_plan_outputs(plan):
    records = []
    missing = []
    for route in plan["routes"]:
        output_jsonl = rooted(route["final_planned_output_jsonl"])
        summary_json = output_jsonl.with_suffix(".summary.json")
        if not summary_json.exists():
            missing.append(
                {
                    "slice": route["slice"],
                    "summary_json": rel(summary_json),
                }
            )
            records.append(
                {
                    "slice": route["slice"],
                    "route": route["route"],
                    "summary_json": rel(summary_json),
                    "metrics": None,
                    "gate_checks": [],
                    "pass": False,
                }
            )
            continue
        summary = load_json(summary_json)
        metrics = metrics_from_summary(summary)
        checks = gate_checks(route, metrics)
        records.append(
            {
                "slice": route["slice"],
                "route": route["route"],
                "summary_json": rel(summary_json),
                "metrics": metrics,
                "gate_checks": checks,
                "pass": all(check["pass"] for check in checks),
            }
        )
    return {
        "status": "split-route-replay-output-verification",
        "source_plan": plan.get("source_plan"),
        "missing_summaries": missing,
        "records": records,
        "all_pass": not missing and all(record["pass"] for record in records),
    }


def build_replay_plan(manifest, manifest_path, out_root, historical_outputs=False, use_systemd=True, selected=None):
    steps = []
    route_plans = []
    unknown_steps = []
    for route in manifest["routes"]:
        if not route_matches(route, selected):
            continue
        route_dir = out_root / slug(route["slice"])
        chronological = list(reversed(route["postprocess_chain"]))
        previous_out = None
        route_steps = []
        for idx, chain_item in enumerate(chronological):
            source_summary = load_json(rooted(chain_item["summary_json"]))
            source_out = rooted(chain_item["out_jsonl"])
            out_jsonl = source_out if historical_outputs else route_dir / source_out.name
            if idx == 0 and source_summary.get("adapter"):
                command = eval_command(source_summary, out_jsonl)
                kind = "generation"
            else:
                command, kind = rescore_command(source_summary, previous_out, out_jsonl)
                if command is None:
                    unknown_steps.append(
                        {
                            "slice": route["slice"],
                            "summary_json": chain_item["summary_json"],
                            "out_jsonl": chain_item["out_jsonl"],
                        }
                    )
                    command = ["#", "UNKNOWN_REPLAY_STEP", chain_item["summary_json"]]
            previous_out = out_jsonl
            shell_command = shell_join(scoped(command) if use_systemd and kind == "generation" else command)
            step = {
                "slice": route["slice"],
                "route": route["route"],
                "kind": kind,
                "source_summary_json": chain_item["summary_json"],
                "source_output_jsonl": chain_item["out_jsonl"],
                "planned_output_jsonl": rel(out_jsonl),
                "command": command,
                "shell": shell_command,
            }
            steps.append(step)
            route_steps.append(step)
        route_plans.append(
            {
                "slice": route["slice"],
                "route": route["route"],
                "route_role": route["route_role"],
                "input_cases_jsonl": route["input_cases_jsonl"],
                "final_planned_output_jsonl": rel(previous_out) if previous_out else None,
                "steps": route_steps,
                "gates": route["gates"],
            }
        )
    return {
        "status": "split-route-replay-plan",
        "source_manifest": rel(manifest_path),
        "out_root": rel(out_root),
        "historical_outputs": historical_outputs,
        "uses_systemd_scope_for_generation": use_systemd,
        "all_gates_pass_in_source_manifest": manifest.get("all_gates_pass"),
        "unknown_steps": unknown_steps,
        "routes": route_plans,
        "steps": steps,
    }


def render_shell(plan):
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "cd /home/mark/qwen_diffusion",
        "",
        "# Generated by scripts/run_qwen35_split_route_sidecar_manifest.py",
        "# Generation commands are wrapped in a user systemd scope with memory limits.",
        "",
    ]
    for step in plan["steps"]:
        lines.append(f"# {step['slice']} :: {step['kind']}")
        planned_out = rooted(step["planned_output_jsonl"])
        lines.append(f"mkdir -p {shlex.quote(str(planned_out.parent))}")
        lines.append(step["shell"])
        lines.append("")
    return "\n".join(lines)


def execute_plan(plan):
    results = []
    for step in plan["steps"]:
        planned_out = rooted(step["planned_output_jsonl"])
        planned_out.parent.mkdir(parents=True, exist_ok=True)
        command = scoped(step["command"]) if plan["uses_systemd_scope_for_generation"] and step["kind"] == "generation" else step["command"]
        started = time.time()
        completed = subprocess.run(command, cwd=ROOT, check=False)
        elapsed = time.time() - started
        result = {
            "slice": step["slice"],
            "kind": step["kind"],
            "planned_output_jsonl": step["planned_output_jsonl"],
            "returncode": completed.returncode,
            "elapsed_seconds": elapsed,
        }
        results.append(result)
        if completed.returncode != 0:
            return {
                "status": "split-route-replay-execution",
                "ok": False,
                "failed_step": result,
                "results": results,
            }
    return {
        "status": "split-route-replay-execution",
        "ok": True,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate and emit a replay plan for the Qwen3.5-9B split-route sidecar manifest."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--historical-outputs",
        action="store_true",
        help="Plan commands that overwrite the source artifact paths. Default writes under --out-root.",
    )
    parser.add_argument("--no-systemd-scope", action="store_true")
    parser.add_argument("--strict-replayable", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--slice",
        dest="selected_slices",
        action="append",
        default=[],
        help="Restrict planning, execution, or verification to an exact slice label or slug. Repeatable.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the generated replay plan after writing it. Use --slice for controlled partial replays.",
    )
    parser.add_argument(
        "--verify-outputs",
        action="store_true",
        help="Verify route gates against summaries for a replay plan instead of emitting a new plan.",
    )
    parser.add_argument("--plan-json", type=Path, default=DEFAULT_PLAN_JSON)
    args = parser.parse_args()

    if args.verify_outputs:
        plan = load_json(args.plan_json)
        if args.selected_slices:
            selected_routes = [route for route in plan["routes"] if route_matches(route, args.selected_slices)]
            selected_labels = {route["slice"] for route in selected_routes}
            plan = {
                **plan,
                "routes": selected_routes,
                "steps": [step for step in plan["steps"] if step["slice"] in selected_labels],
            }
        plan["source_plan"] = rel(args.plan_json)
        verification = verify_plan_outputs(plan)
        verify_json = args.plan_json.with_name(args.plan_json.stem + "_verification.json")
        verify_json.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(verify_json)
        print(
            json.dumps(
                {
                    "ok": verification["all_pass"],
                    "records": len(verification["records"]),
                    "missing_summaries": len(verification["missing_summaries"]),
                    "failed_records": sum(1 for record in verification["records"] if not record["pass"]),
                },
                indent=2,
            ),
            flush=True,
        )
        if not verification["all_pass"]:
            raise SystemExit(1)
        return

    manifest = load_json(args.manifest)
    missing = validate_manifest(manifest)
    if missing:
        print(json.dumps({"ok": False, "missing": missing}, indent=2), flush=True)
        raise SystemExit(1)

    plan = build_replay_plan(
        manifest,
        args.manifest,
        args.out_root,
        historical_outputs=args.historical_outputs,
        use_systemd=not args.no_systemd_scope,
        selected=args.selected_slices,
    )
    if args.strict_replayable and plan["unknown_steps"]:
        print(json.dumps({"ok": False, "unknown_steps": plan["unknown_steps"]}, indent=2), flush=True)
        raise SystemExit(1)

    if not args.check_only:
        args.out_root.mkdir(parents=True, exist_ok=True)
        plan_json = args.out_root / "route_runner_plan.json"
        plan_sh = args.out_root / "route_runner_plan.sh"
        plan_json.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        plan_sh.write_text(render_shell(plan), encoding="utf-8")
        print(plan_json)
        print(plan_sh)

    if args.execute:
        execution = execute_plan(plan)
        execution_json = args.out_root / "route_runner_execution.json"
        execution_json.write_text(json.dumps(execution, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(execution_json)
        if not execution["ok"]:
            print(json.dumps(execution, indent=2), flush=True)
            raise SystemExit(1)

    print(
        json.dumps(
            {
                "ok": True,
                "routes": len(plan["routes"]),
                "steps": len(plan["steps"]),
                "unknown_steps": len(plan["unknown_steps"]),
                "all_gates_pass_in_source_manifest": plan["all_gates_pass_in_source_manifest"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
