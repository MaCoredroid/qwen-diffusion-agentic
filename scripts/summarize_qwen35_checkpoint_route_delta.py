#!/usr/bin/env python3
import argparse
from collections import Counter
import json
from pathlib import Path

from diagnose_toolcall_argument_errors import diagnose_row, load_jsonl


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_OUT_JSON = (
    ROOT
    / "runs/fastdllm_qwen35_9b_candidate_value_span_public_train_from_ckpt275_step10"
    / "checkpoint-5/route_delta_vs_current_routed_target.json"
)
DEFAULT_OUT_MD = ROOT / "qwen35_public_train_candidate_value_span_route_delta.md"
CKPT5_DIR = ROOT / "runs/fastdllm_qwen35_9b_candidate_value_span_public_train_from_ckpt275_step10/checkpoint-5"
STAGED24_DIR = (
    ROOT
    / "runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_staged_retention_ckpt24_broad_eval96_modelrepair_agentic"
    / "checkpoint-24"
)
ACTIVE_MULTICALL_DIR = (
    ROOT
    / "runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_multicall_eval384_modelrepair"
)
ACTIVE_OPENAI_TOOLRESULT_DIR = (
    ROOT
    / "runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_openai_toolresult_eval160_modelrepair_max1"
)


LANES = [
    {
        "slice": "public one-call",
        "cases": ROOT / "data/toolcall_eval/public_onecall_hermes_smoke.jsonl",
        "current": STAGED24_DIR / "public_onecall_8.jsonl",
        "candidate": CKPT5_DIR / "public_onecall_8_nomodelrepair.jsonl",
        "prefix": "constrained",
    },
    {
        "slice": "teacher-train one-call",
        "cases": ROOT / "data/toolcall_eval/public_onecall_teacher_train_labelaware_smoke.jsonl",
        "current": STAGED24_DIR / "teacher_train_labelaware_12.jsonl",
        "candidate": CKPT5_DIR / "teacher_train_labelaware_12_nomodelrepair.jsonl",
        "prefix": "constrained",
    },
    {
        "slice": "teacher-heldout one-call",
        "cases": ROOT / "data/toolcall_eval/public_onecall_teacher_heldout_labelaware_smoke.jsonl",
        "current": STAGED24_DIR / "teacher_heldout_labelaware_8.jsonl",
        "candidate": CKPT5_DIR / "teacher_heldout_labelaware_8_nomodelrepair.jsonl",
        "prefix": "constrained",
    },
    {
        "slice": "public multi-call planner",
        "cases": ROOT / "data/toolcall_eval/public_multicall_hermes_smoke.jsonl",
        "current": ACTIVE_MULTICALL_DIR / "public_multicall_12_sequence_planner_segmentargs_v3.jsonl",
        "candidate": CKPT5_DIR / "public_multicall_12_sequence_planner_projection.jsonl",
        "prefix": "sequence_planner",
    },
    {
        "slice": "synthetic text tool-result",
        "cases": ROOT / "data/toolcall_eval/synthetic_toolresult_smoke.jsonl",
        "current": STAGED24_DIR / "synthetic_toolresult_10.jsonl",
        "candidate": CKPT5_DIR / "synthetic_toolresult_10_nomodelrepair.jsonl",
        "prefix": "constrained",
    },
    {
        "slice": "OpenAI-style tool-result",
        "cases": ROOT / "data/toolcall_eval/synthetic_toolresult_openai_smoke.jsonl",
        "current": ACTIVE_OPENAI_TOOLRESULT_DIR / "synthetic_openai_toolresult_10_grounded_projection_v2.jsonl",
        "candidate": CKPT5_DIR / "synthetic_openai_toolresult_10_grounded_projection_v2_nomodelrepair.jsonl",
        "prefix": "constrained",
    },
]


def rel(path):
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def index_rows(path):
    rows = load_jsonl(path)
    return {row.get("id") or str(idx): row for idx, row in enumerate(rows)}


def metric_key(prefix, metric):
    return f"{prefix}_{metric}"


def row_metric(row, prefix, metric):
    if prefix == "raw":
        return bool(row.get(metric))
    return bool(row.get(metric_key(prefix, metric)))


def calls(row, prefix):
    value = row.get(metric_key(prefix, "calls"))
    return value if isinstance(value, list) else []


def names(row, prefix):
    value = row.get(metric_key(prefix, "called_names"))
    return value if isinstance(value, list) else [call.get("name") for call in calls(row, prefix)]


def brief_diff(diff):
    out = {
        "call_index": diff.get("call_index"),
        "tool_name": diff.get("tool_name"),
        "path": diff.get("path"),
        "kind": diff.get("kind"),
        "value_kind": diff.get("value_kind"),
    }
    for key in ["pred", "gold", "pred_len", "gold_len"]:
        if key in diff:
            out[key] = diff[key]
    return out


def compare_lane(lane):
    cases = {row.get("id") or str(idx): row for idx, row in enumerate(load_jsonl(lane["cases"]))}
    current = index_rows(lane["current"])
    candidate = index_rows(lane["candidate"])
    ids = [key for key in candidate if key in current and key in cases]
    prefix = lane["prefix"]

    protected = {
        "current_sequence": 0,
        "current_arguments": 0,
        "candidate_sequence": 0,
        "candidate_arguments": 0,
    }
    raw = {
        "current_sequence": 0,
        "current_arguments": 0,
        "candidate_sequence": 0,
        "candidate_arguments": 0,
    }
    changes = {
        "sequence_improved": [],
        "sequence_regressed": [],
        "arguments_improved": [],
        "arguments_regressed": [],
        "raw_sequence_improved": [],
        "raw_sequence_regressed": [],
        "raw_arguments_improved": [],
        "raw_arguments_regressed": [],
    }
    failure_diagnostics = []
    diff_kind_counts = Counter()
    diff_path_counts = Counter()

    for key in ids:
        cur = current[key]
        cand = candidate[key]
        case = cases[key]

        cur_seq = row_metric(cur, prefix, "exact_tool_sequence")
        cur_args = row_metric(cur, prefix, "exact_arguments")
        cand_seq = row_metric(cand, prefix, "exact_tool_sequence")
        cand_args = row_metric(cand, prefix, "exact_arguments")
        raw_cur_seq = row_metric(cur, "raw", "exact_tool_sequence")
        raw_cur_args = row_metric(cur, "raw", "exact_arguments")
        raw_cand_seq = row_metric(cand, "raw", "exact_tool_sequence")
        raw_cand_args = row_metric(cand, "raw", "exact_arguments")

        protected["current_sequence"] += int(cur_seq)
        protected["current_arguments"] += int(cur_args)
        protected["candidate_sequence"] += int(cand_seq)
        protected["candidate_arguments"] += int(cand_args)
        raw["current_sequence"] += int(raw_cur_seq)
        raw["current_arguments"] += int(raw_cur_args)
        raw["candidate_sequence"] += int(raw_cand_seq)
        raw["candidate_arguments"] += int(raw_cand_args)

        if not cur_seq and cand_seq:
            changes["sequence_improved"].append(key)
        if cur_seq and not cand_seq:
            changes["sequence_regressed"].append(key)
        if not cur_args and cand_args:
            changes["arguments_improved"].append(key)
        if cur_args and not cand_args:
            changes["arguments_regressed"].append(key)
        if not raw_cur_seq and raw_cand_seq:
            changes["raw_sequence_improved"].append(key)
        if raw_cur_seq and not raw_cand_seq:
            changes["raw_sequence_regressed"].append(key)
        if not raw_cur_args and raw_cand_args:
            changes["raw_arguments_improved"].append(key)
        if raw_cur_args and not raw_cand_args:
            changes["raw_arguments_regressed"].append(key)

        if not cand_args:
            diagnosis = diagnose_row(cand, case, prefix)
            brief_diffs = [brief_diff(diff) for diff in diagnosis["diffs"]]
            for diff in brief_diffs:
                diff_kind_counts[diff.get("kind")] += 1
                diff_path_counts[f"{diff.get('tool_name')}:{diff.get('path')}"] += 1
            failure_diagnostics.append(
                {
                    "id": key,
                    "current_sequence": cur_seq,
                    "current_arguments": cur_args,
                    "candidate_sequence": cand_seq,
                    "candidate_arguments": cand_args,
                    "candidate_names": names(cand, prefix),
                    "gold_names": diagnosis["gold_names"],
                    "diffs": brief_diffs[:12],
                    "diff_count": diagnosis["diff_count"],
                }
            )

    return {
        "slice": lane["slice"],
        "records": len(ids),
        "cases_jsonl": rel(lane["cases"]),
        "current_jsonl": rel(lane["current"]),
        "candidate_jsonl": rel(lane["candidate"]),
        "protected_prefix": prefix,
        "protected": protected,
        "raw": raw,
        "changes": changes,
        "candidate_failure_count": len(failure_diagnostics),
        "candidate_failure_diagnostics": failure_diagnostics,
        "candidate_failure_kind_counts": dict(diff_kind_counts.most_common()),
        "candidate_failure_top_paths": dict(diff_path_counts.most_common(20)),
    }


def aggregate(lanes):
    totals = Counter()
    all_changes = Counter()
    kind_counts = Counter()
    path_counts = Counter()
    for lane in lanes:
        totals["records"] += lane["records"]
        for group in ["protected", "raw"]:
            for key, value in lane[group].items():
                totals[f"{group}_{key}"] += value
        for key, rows in lane["changes"].items():
            all_changes[key] += len(rows)
        kind_counts.update(lane["candidate_failure_kind_counts"])
        path_counts.update(lane["candidate_failure_top_paths"])
    return {
        "totals": dict(totals),
        "change_counts": dict(all_changes),
        "candidate_failure_kind_counts": dict(kind_counts.most_common()),
        "candidate_failure_top_paths": dict(path_counts.most_common(20)),
    }


def write_markdown(payload, out_md):
    lines = [
        "# Qwen3.5 Checkpoint-5 Route Delta",
        "",
        "Purpose: compare value-span checkpoint-5 against the current split-route target at row level.",
        "This is a diagnostic report, not a training manifest; eval and heldout rows must not be promoted into train data.",
        "",
        "## Summary",
        "",
        "| Slice | Current protected | Candidate protected | Protected arg delta | Candidate raw/input | Decision signal |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for lane in payload["lanes"]:
        p = lane["protected"]
        raw = lane["raw"]
        cur = f"{p['current_sequence']}/{lane['records']}, {p['current_arguments']}/{lane['records']}"
        cand = f"{p['candidate_sequence']}/{lane['records']}, {p['candidate_arguments']}/{lane['records']}"
        raw_cand = f"{raw['candidate_sequence']}/{lane['records']}, {raw['candidate_arguments']}/{lane['records']}"
        delta = p["candidate_arguments"] - p["current_arguments"]
        signal = "tie"
        if delta > 0:
            signal = "candidate improves protected args"
        elif delta < 0:
            signal = "candidate regresses protected args"
        lines.append(f"| {lane['slice']} | `{cur}` | `{cand}` | `{delta:+d}` | `{raw_cand}` | {signal} |")

    lines.extend(
        [
            "",
            "Each metric cell is `exact sequence, exact arguments`.",
            "`raw/input` means the row's unprotected metric field; for chained post-processing lanes this can be the chain input rather than original generation.",
            "",
            "## Row-Level Changes",
            "",
        ]
    )
    for lane in payload["lanes"]:
        changes = lane["changes"]
        lines.extend(
            [
                f"### {lane['slice']}",
                "",
                f"- protected sequence improved: `{len(changes['sequence_improved'])}`",
                f"- protected sequence regressed: `{len(changes['sequence_regressed'])}`",
                f"- protected arguments improved: `{len(changes['arguments_improved'])}`",
                f"- protected arguments regressed: `{len(changes['arguments_regressed'])}`",
                f"- raw arguments improved: `{len(changes['raw_arguments_improved'])}`",
                f"- raw arguments regressed: `{len(changes['raw_arguments_regressed'])}`",
            ]
        )
        if changes["arguments_regressed"]:
            lines.append(f"- argument regressions: `{', '.join(changes['arguments_regressed'])}`")
        if changes["arguments_improved"]:
            lines.append(f"- argument improvements: `{', '.join(changes['arguments_improved'])}`")
        if lane["candidate_failure_top_paths"]:
            paths = ", ".join(f"`{key}` x{value}" for key, value in list(lane["candidate_failure_top_paths"].items())[:5])
            lines.append(f"- top candidate failure paths: {paths}")
        lines.append("")

    lines.extend(
        [
            "## Training Implications",
            "",
            "- Do not train on rows from this eval/heldout delta report.",
            "- Mine train-only analogues for the repeated failure classes: missing one-call tool sequence, scalar argument grounding, and OpenAI-style tool-result argument retention.",
            "- Use checkpoint-5 as positive signal for public multi-call constrained/contextual row grounding, not as a promoted route.",
            "",
            "## Artifacts",
            "",
            f"- JSON: `{rel(payload['out_json'])}`",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    lanes = [compare_lane(lane) for lane in LANES]
    payload = {
        "candidate": "value-span checkpoint-5",
        "current_target": "current split-route target",
        "out_json": str(args.out_json),
        "lanes": lanes,
        "aggregate": aggregate(lanes),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(payload, args.out_md)
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md), "lanes": len(lanes)}, indent=2))


if __name__ == "__main__":
    main()
