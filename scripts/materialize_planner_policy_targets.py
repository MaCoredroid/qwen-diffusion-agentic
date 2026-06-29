#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_CASES = ROOT / "data/toolcall_eval/heldout_seed_multicall_2to3_clean.jsonl"
DEFAULT_ANALYSIS = ROOT / "runs/planner_decomposition/heldout_seed_multicall_policy_analysis.jsonl"
DEFAULT_TEACHER_REQUIRED = ROOT / "runs/qwen36_teacher_heldout_multicall_2to3_clean_toolchoice_required.jsonl"
DEFAULT_OUT = ROOT / "runs/planner_decomposition/heldout_seed_multicall_policy_targets.jsonl"


def load_jsonl(path):
    rows = []
    if not path or not Path(path).exists():
        return rows
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def case_key(row, fallback):
    return row.get("id") or row.get("case_id") or str(fallback)


def by_key(rows):
    return {case_key(row, idx): row for idx, row in enumerate(rows)}


def compact_call(name, arguments):
    payload = {"name": name, "arguments": arguments if isinstance(arguments, dict) else {}}
    return "<tool_call>\n" + json.dumps(payload, ensure_ascii=False) + "\n</tool_call>"


def compact_calls(calls):
    return "\n".join(compact_call(call.get("name"), call.get("arguments") or {}) for call in calls if call.get("name"))


def gold_target(case):
    calls = case.get("gold_tool_calls")
    if isinstance(calls, list) and calls:
        return compact_calls(calls)
    return str(case.get("gold_assistant") or "")


def teacher_required_target(row):
    return compact_calls(row.get("calls") or [])


def choose_target(case, analysis, teacher_required):
    policy = analysis.get("recommended_policy_target")
    tags = set(analysis.get("tags") or [])
    if policy == "adjudicate_full_request_vs_seed_gold":
        return None, "rejected_adjudication_required"
    if "teacher_required_exact" in tags and teacher_required:
        return teacher_required_target(teacher_required), "teacher_required_exact"
    if policy == "teacher_required_sequence_plus_value_sidecars":
        return gold_target(case), "gold_values_for_teacher_sequence"
    if policy in {"gold_sequence_decomposition_target", "gold_split_call_target"}:
        return gold_target(case), "gold_decomposition_policy"
    if policy == "teacher_required_or_gold":
        return gold_target(case), "gold_fallback"
    return gold_target(case), "gold_curated_fallback"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-jsonl", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--analysis-jsonl", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--teacher-required-jsonl", type=Path, default=DEFAULT_TEACHER_REQUIRED)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--rejected-jsonl", type=Path, default=None)
    args = parser.parse_args()

    cases = load_jsonl(args.cases_jsonl)
    analysis_by_key = by_key(load_jsonl(args.analysis_jsonl))
    teacher_by_key = by_key(load_jsonl(args.teacher_required_jsonl))
    rejected_path = args.rejected_jsonl or args.out_jsonl.with_suffix(".rejected.jsonl")

    accepted = []
    rejected = []
    for idx, case in enumerate(cases):
        key = case_key(case, idx)
        analysis = analysis_by_key.get(key) or {}
        teacher = teacher_by_key.get(key)
        target, source = choose_target(case, analysis, teacher)
        base = {
            "id": key,
            "source": case.get("source"),
            "task": case.get("task"),
            "category": case.get("category"),
            "policy_target_source": source,
            "policy_tags": analysis.get("tags") or [],
            "recommended_policy_target": analysis.get("recommended_policy_target"),
            "gold_tool_names": case.get("gold_tool_names") or [],
            "teacher_required_names": analysis.get("teacher_required_names") or [],
            "tools": case.get("tools") or [],
            "prompt_messages": case.get("prompt_messages") or [],
            "gold_assistant": case.get("gold_assistant"),
            "gold_tool_calls": case.get("gold_tool_calls") or [],
        }
        if target is None:
            rejected.append(base)
            continue
        accepted.append({**base, "policy_planner_assistant": target})

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for row in accepted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with Path(rejected_path).open("w", encoding="utf-8") as handle:
        for row in rejected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    source_counts = {}
    for row in accepted:
        source_counts[row["policy_target_source"]] = source_counts.get(row["policy_target_source"], 0) + 1
    summary = {
        "cases_jsonl": str(args.cases_jsonl),
        "analysis_jsonl": str(args.analysis_jsonl),
        "teacher_required_jsonl": str(args.teacher_required_jsonl),
        "out_jsonl": str(args.out_jsonl),
        "rejected_jsonl": str(rejected_path),
        "accepted_records": len(accepted),
        "rejected_records": len(rejected),
        "policy_target_source_counts": dict(sorted(source_counts.items())),
        "rejected_ids": [row["id"] for row in rejected],
    }
    args.out_jsonl.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
