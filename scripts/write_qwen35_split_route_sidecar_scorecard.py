#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path("/home/mark/qwen_diffusion")
OUT = ROOT / "qwen35_9b_split_route_sidecar_scorecard.md"
OUT_DIR = ROOT / "runs/qwen35_9b_split_route_sidecar_eval"
OUT_JSON = OUT_DIR / "route_scorecard.json"
OUT_TSV = OUT_DIR / "route_scorecard.tsv"
OUT_MANIFEST = OUT_DIR / "route_manifest.json"


@dataclass(frozen=True)
class Source:
    name: str
    path: Path
    projected_key: str | None = None


@dataclass(frozen=True)
class RouteRow:
    label: str
    active: Source
    staged24: Source
    route: str
    rationale: str


ACTIVE_ONECALL_DIR = (
    ROOT / "runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_eval96_modelrepair_max1"
)
ACTIVE_MULTICALL_DIR = (
    ROOT
    / "runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_multicall_eval384_modelrepair"
)
ACTIVE_TOOLRESULT_DIR = (
    ROOT
    / "runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_toolresult_eval160_modelrepair_max1"
)
ACTIVE_OPENAI_TOOLRESULT_DIR = (
    ROOT
    / "runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_openai_toolresult_eval160_modelrepair_max1"
)
STAGED24_DIR = (
    ROOT
    / "runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_staged_retention_ckpt24_broad_eval96_modelrepair_agentic"
    / "checkpoint-24"
)


ROWS = [
    RouteRow(
        "public one-call",
        Source("active checkpoint-275", ACTIVE_ONECALL_DIR / "public_onecall_8_grounded_projection_v2.summary.json"),
        Source("staged checkpoint-24", STAGED24_DIR / "public_onecall_8.summary.json"),
        "staged24_generator",
        "keeps public constrained perfect and improves raw exact sequence/arguments",
    ),
    RouteRow(
        "teacher-train one-call",
        Source("active checkpoint-275", ACTIVE_ONECALL_DIR / "teacher_train_labelaware_12_grounded_projection_v2.summary.json"),
        Source("staged checkpoint-24", STAGED24_DIR / "teacher_train_labelaware_12.summary.json"),
        "staged24_generator",
        "improves constrained sequence while preserving constrained arguments",
    ),
    RouteRow(
        "teacher-heldout one-call",
        Source("active checkpoint-275", ACTIVE_ONECALL_DIR / "teacher_heldout_labelaware_8_grounded_projection_v2.summary.json"),
        Source("staged checkpoint-24", STAGED24_DIR / "teacher_heldout_labelaware_8.summary.json"),
        "staged24_generator",
        "improves raw heldout while preserving constrained heldout",
    ),
    RouteRow(
        "public multi-call planner",
        Source("active checkpoint-275", ACTIVE_MULTICALL_DIR / "public_multicall_12_sequence_planner_segmentargs_v3.summary.json", "planned"),
        Source("staged checkpoint-24", STAGED24_DIR / "public_multicall_12_sequence_planner_projection.summary.json", "planned"),
        "active_protection_path",
        "active planner keeps one more exact-argument case",
    ),
    RouteRow(
        "synthetic text tool-result",
        Source("active checkpoint-275", ACTIVE_TOOLRESULT_DIR / "synthetic_toolresult_10_grounded_projection_v2.summary.json"),
        Source("staged checkpoint-24", STAGED24_DIR / "synthetic_toolresult_10.summary.json"),
        "staged24_generator",
        "staged checkpoint-24 improves constrained exact arguments",
    ),
    RouteRow(
        "OpenAI-style tool-result",
        Source("active checkpoint-275", ACTIVE_OPENAI_TOOLRESULT_DIR / "synthetic_openai_toolresult_10_grounded_projection_v2.summary.json"),
        Source("staged checkpoint-24", STAGED24_DIR / "synthetic_openai_toolresult_10.summary.json"),
        "active_protection_path",
        "active checkpoint-275 keeps one more exact-argument case",
    ),
]


GATES = {
    "public one-call": {
        "raw_seq": 4,
        "raw_args": 3,
        "protected_seq": 8,
        "protected_args": 8,
    },
    "teacher-train one-call": {
        "protected_seq": 11,
        "protected_args": 6,
    },
    "teacher-heldout one-call": {
        "raw_seq": 2,
        "raw_args": 1,
        "protected_seq": 8,
        "protected_args": 6,
    },
    "public multi-call planner": {
        "protected_seq": 11,
        "protected_args": 10,
    },
    "synthetic text tool-result": {
        "protected_seq": 10,
        "protected_args": 9,
    },
    "OpenAI-style tool-result": {
        "protected_seq": 10,
        "protected_args": 9,
    },
}


def read_summary(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path):
    return str(path.relative_to(ROOT))


def rel_value(value):
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        try:
            return rel(path)
        except ValueError:
            return str(path)
    return str(path)


def totals_for(source):
    totals = read_summary(source.path)["totals"]
    protected = totals
    if source.projected_key:
        protected = totals[source.projected_key]
    raw = totals.get("input", totals)
    return totals["records"], raw, protected


def metric_pair(source):
    records, raw, protected = totals_for(source)
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
        "extra": protected.get("records_with_extra_calls", ""),
        "missing": protected.get("records_with_missing_calls", ""),
        "repeated": protected.get("records_with_repeated_calls", ""),
    }


def frac(value, records):
    return f"{value}/{records}" if isinstance(value, int) else "n/a"


def protected_cell(metrics):
    return f"{frac(metrics['protected_seq'], metrics['records'])} / {frac(metrics['protected_args'], metrics['records'])}"


def raw_cell(metrics):
    return f"{frac(metrics['raw_seq'], metrics['records'])} / {frac(metrics['raw_args'], metrics['records'])}"


def route_source(row):
    if row.route == "staged24_generator":
        return row.staged24
    if row.route == "active_protection_path":
        return row.active
    raise ValueError(f"unknown route {row.route}")


def gate_checks(label, metrics):
    checks = []
    for metric, minimum in GATES[label].items():
        actual = metrics[metric]
        if not isinstance(actual, int):
            passed = False
        else:
            passed = actual >= minimum
        checks.append(
            {
                "metric": metric,
                "actual": actual,
                "minimum": minimum,
                "records": metrics["records"],
                "pass": passed,
            }
        )
    return checks


def route_record(row):
    active = metric_pair(row.active)
    staged = metric_pair(row.staged24)
    routed_source = route_source(row)
    routed = metric_pair(routed_source)
    checks = gate_checks(row.label, routed)
    return {
        "slice": row.label,
        "route": row.route,
        "rationale": row.rationale,
        "active_source": rel(row.active.path),
        "staged24_source": rel(row.staged24.path),
        "routed_source": rel(routed_source.path),
        "active": active,
        "staged24": staged,
        "routed": routed,
        "gate_checks": checks,
        "pass": all(check["pass"] for check in checks),
    }


def summary_metadata(source):
    summary = read_summary(source.path)
    keys = [
        "eval_name",
        "input_jsonl",
        "cases_jsonl",
        "out_jsonl",
        "base_model",
        "adapter",
        "block_size",
        "small_block_size",
        "max_new_tokens",
        "threshold",
        "temperature",
        "top_p",
        "repair_mode",
        "constrained_tool_decoding",
        "constrained_max_calls",
        "model_repair_pass",
        "text_field",
        "min_input_calls_for_plan",
        "prefer_segment_args",
    ]
    metadata = {key: summary.get(key) for key in keys if key in summary}
    for key in ("input_jsonl", "cases_jsonl", "out_jsonl", "base_model", "adapter"):
        if key in metadata:
            metadata[key] = rel_value(metadata[key])
    metadata["summary_json"] = rel(source.path)
    if source.projected_key:
        metadata["protected_totals_key"] = source.projected_key
    elif "constrained_exact_tool_sequence" in summary["totals"]:
        metadata["protected_totals_key"] = "constrained_*"
    else:
        metadata["protected_totals_key"] = "totals"
    return metadata


def summary_for_jsonl(path_text):
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    summary = path.with_suffix(".summary.json")
    if summary.exists():
        return summary
    return None


def trace_generation_summary(source):
    summary = read_summary(source.path)
    if summary.get("adapter"):
        return source.path
    input_jsonl = summary.get("input_jsonl")
    seen = {source.path.resolve()}
    while input_jsonl:
        candidate = summary_for_jsonl(input_jsonl)
        if candidate is None:
            return None
        resolved = candidate.resolve()
        if resolved in seen:
            return None
        seen.add(resolved)
        candidate_summary = read_summary(candidate)
        if candidate_summary.get("adapter"):
            return candidate
        input_jsonl = candidate_summary.get("input_jsonl")
    return None


def postprocess_chain(source):
    chain = []
    current = source.path
    seen = set()
    while current and current.exists():
        resolved = current.resolve()
        if resolved in seen:
            break
        seen.add(resolved)
        metadata = summary_metadata(Source(source.name, current))
        chain.append(metadata)
        if metadata.get("adapter"):
            break
        current = summary_for_jsonl(metadata.get("input_jsonl"))
    return chain


def manifest_route(row, record):
    routed_source = route_source(row)
    generation_summary = trace_generation_summary(routed_source)
    generation_metadata = summary_metadata(Source(routed_source.name, generation_summary)) if generation_summary else None
    routed_metadata = summary_metadata(routed_source)
    cases_jsonl = routed_metadata.get("input_jsonl") or routed_metadata.get("cases_jsonl")
    if generation_metadata:
        cases_jsonl = generation_metadata.get("input_jsonl") or cases_jsonl
    return {
        "slice": row.label,
        "route": row.route,
        "route_role": "generator" if row.route == "staged24_generator" else "protection_sidecar",
        "input_cases_jsonl": cases_jsonl,
        "routed_summary_json": record["routed_source"],
        "routed_output_jsonl": routed_metadata.get("out_jsonl"),
        "generation_summary_json": rel(generation_summary) if generation_summary else None,
        "generation_output_jsonl": generation_metadata.get("out_jsonl") if generation_metadata else None,
        "postprocess_chain": postprocess_chain(routed_source),
        "gates": GATES[row.label],
        "gate_checks": record["gate_checks"],
    }


def build_manifest(payload):
    routes = [manifest_route(row, record) for row, record in zip(ROWS, payload["records"], strict=True)]
    adapters = {}
    for route in routes:
        generation = route["postprocess_chain"][-1] if route["postprocess_chain"] else None
        if generation and generation.get("adapter"):
            adapters[route["route"]] = generation["adapter"]
    base_models = sorted(
        {
            route["postprocess_chain"][-1]["base_model"]
            for route in routes
            if route["postprocess_chain"] and route["postprocess_chain"][-1].get("base_model")
        }
    )
    return {
        "date": payload["date"],
        "status": "split-route-router-implementation-manifest",
        "base_models": base_models,
        "adapter_roles": {
            "staged24_generator": adapters.get("staged24_generator"),
            "active_protection_path": adapters.get("active_protection_path"),
        },
        "check_command": ".venv-fastdllm/bin/python scripts/write_qwen35_split_route_sidecar_scorecard.py --check",
        "scorecard_json": rel(OUT_JSON),
        "scorecard_tsv": rel(OUT_TSV),
        "scorecard_markdown": rel(OUT),
        "all_gates_pass": payload["all_pass"],
        "routes": routes,
        "implementation_notes": [
            "This manifest describes the first executable router/sidecar target from existing eval artifacts.",
            "It does not mean live prompt-time adapter switching is implemented yet.",
            "A live runner should load the base model once, select the adapter role from the route, then apply the recorded post-processing chain for protected lanes.",
        ],
    }


def build_payload():
    records = [route_record(row) for row in ROWS]
    return {
        "date": "2026-06-27",
        "status": "route-composition-target",
        "description": (
            "Machine-readable split-route target assembled from existing "
            "checkpoint-24 and checkpoint-275 eval summaries."
        ),
        "all_pass": all(record["pass"] for record in records),
        "records": records,
    }


def failed_checks(record):
    return [
        f"{check['metric']} {check['actual']}/{check['records']} < {check['minimum']}/{check['records']}"
        for check in record["gate_checks"]
        if not check["pass"]
    ]


def render(payload):
    lines = [
        "# Qwen3.5-9B Split-Route Sidecar Scorecard",
        "",
        f"Date: {payload['date']}",
        "",
        "## Status",
        "",
        "This is a routing/protection scorecard, not a newly promoted single adapter.",
        "It tests the immediate implication of the checkpoint-24 experiments: keep the",
        "better one-call generator behavior from staged checkpoint-24, but route known",
        "multi-call/tool-result protection lanes through the active checkpoint-275",
        "projection path where checkpoint-24 regresses.",
        "",
        "The scorecard is an upper-bound target for a future sidecar or router. It is",
        "not a claim that a deployed router has been implemented.",
        "",
        "Machine-readable output:",
        "",
        f"- JSON: `{rel(OUT_JSON)}`",
        f"- TSV: `{rel(OUT_TSV)}`",
        f"- route manifest: `{rel(OUT_MANIFEST)}`",
        f"- gate verdict: `{'PASS' if payload['all_pass'] else 'FAIL'}`",
        "",
        "Replay runner:",
        "",
        "- script: `scripts/run_qwen35_split_route_sidecar_manifest.py`",
        "- default plan JSON: `runs/qwen35_9b_split_route_sidecar_eval/replay_plan/route_runner_plan.json`",
        "- default plan shell: `runs/qwen35_9b_split_route_sidecar_eval/replay_plan/route_runner_plan.sh`",
        "- partial execution: `scripts/run_qwen35_split_route_sidecar_manifest.py --slice public_one_call --execute`",
        "- output verifier: `scripts/run_qwen35_split_route_sidecar_manifest.py --verify-outputs --plan-json <plan.json>`",
        "- historical verification: `runs/qwen35_9b_split_route_sidecar_eval/historical_verify_plan/route_runner_plan_verification.json`",
        "- live public one-call smoke: `runs/qwen35_9b_split_route_sidecar_eval/live_smoke_public_onecall/route_runner_plan_verification.json`",
        "- live OpenAI-style tool-result smoke: `runs/qwen35_9b_split_route_sidecar_eval/live_smoke_openai_toolresult/route_runner_plan_verification.json`",
        "- live public multi-call planner smoke: `runs/qwen35_9b_split_route_sidecar_eval/live_smoke_public_multicall_planner/route_runner_plan_verification.json`",
        "- live synthetic text tool-result smoke: `runs/qwen35_9b_split_route_sidecar_eval/live_smoke_synthetic_text_toolresult/route_runner_plan_verification.json`",
        "- live teacher one-call smoke: `runs/qwen35_9b_split_route_sidecar_eval/live_smoke_teacher_onecall/route_runner_plan_verification.json`",
        "- live coverage: all `6` split-route lanes have verified live replay artifacts",
        "",
        "## Route Table",
        "",
        "| Slice | Route | Active protected seq/args | Ckpt-24 protected seq/args | Routed protected seq/args | Routed raw seq/args | Rationale |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in payload["records"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    record["slice"],
                    record["route"],
                    protected_cell(record["active"]),
                    protected_cell(record["staged24"]),
                    protected_cell(record["routed"]),
                    raw_cell(record["routed"]),
                    record["rationale"],
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- The split route preserves the staged checkpoint-24 public one-call raw gain",
            "  (`4/8` sequence, `3/8` arguments) while keeping public constrained",
            "  recovery at `8/8` / `8/8`.",
            "- It keeps the active multi-call protected top line at `11/12` sequence and",
            "  `10/12` arguments by routing that lane through checkpoint-275's guarded",
            "  sequence planner.",
            "- It keeps the active OpenAI-style tool-result protected top line at `10/10`",
            "  sequence and `9/10` arguments by routing that lane through checkpoint-275.",
            "- It uses checkpoint-24 for text-compatible synthetic tool-result, where",
            "  checkpoint-24 reaches `10/10` sequence and `9/10` arguments versus active",
            "  checkpoint-275's `10/10` / `8/10`.",
            "",
            "## Gate Results",
            "",
            "| Slice | Route | Routed source | Gate | Failed checks |",
            "| --- | --- | --- | ---: | --- |",
        ]
    )
    for record in payload["records"]:
        failed = "; ".join(failed_checks(record)) or "none"
        lines.append(
            "| "
            + " | ".join(
                [
                    record["slice"],
                    record["route"],
                    f"`{record['routed_source']}`",
                    "PASS" if record["pass"] else "FAIL",
                    failed,
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Next Experiment",
            "",
            "Do not train broad anti-regression rows into the same generator adapter again.",
            "The next practical experiment should implement a runtime router or sidecar",
            "repair path with these gates:",
            "",
            "- one-call prompts route to staged checkpoint-24 and must keep public raw",
            "  `>=4/8` sequence and `>=3/8` arguments",
            "- multi-call prompts route through active checkpoint-275 planner/projection",
            "  until a sidecar matches `11/12` sequence and `10/12` arguments",
            "- OpenAI-style tool-result prompts route through active checkpoint-275 until a",
            "  sidecar matches `10/10` sequence and `9/10` arguments",
            "- text tool-result prompts may route to checkpoint-24 if the route preserves",
            "  `10/10` sequence and `9/10` arguments",
            "",
            "## Source Artifacts",
            "",
        ]
    )
    for record in payload["records"]:
        lines.append(f"- {record['slice']}, active: `{record['active_source']}`")
        lines.append(f"- {record['slice']}, checkpoint-24: `{record['staged24_source']}`")
    lines.append("")
    return "\n".join(lines)


def render_tsv(payload):
    headers = [
        "slice",
        "route",
        "routed_source",
        "records",
        "raw_seq",
        "raw_args",
        "protected_seq",
        "protected_args",
        "pass",
        "failed_checks",
        "active_source",
        "staged24_source",
    ]
    lines = ["\t".join(headers)]
    for record in payload["records"]:
        routed = record["routed"]
        values = [
            record["slice"],
            record["route"],
            record["routed_source"],
            str(routed["records"]),
            str(routed["raw_seq"]),
            str(routed["raw_args"]),
            str(routed["protected_seq"]),
            str(routed["protected_args"]),
            "1" if record["pass"] else "0",
            "; ".join(failed_checks(record)),
            record["active_source"],
            record["staged24_source"],
        ]
        lines.append("\t".join(values))
    lines.append("")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compose the Qwen3.5-9B split-route sidecar scorecard from eval summaries."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit nonzero if any route gate fails after writing artifacts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_TSV.write_text(render_tsv(payload), encoding="utf-8")
    OUT_MANIFEST.write_text(json.dumps(build_manifest(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT.write_text(render(payload), encoding="utf-8")
    print(OUT)
    print(OUT_JSON)
    print(OUT_TSV)
    print(OUT_MANIFEST)
    if args.check and not payload["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
