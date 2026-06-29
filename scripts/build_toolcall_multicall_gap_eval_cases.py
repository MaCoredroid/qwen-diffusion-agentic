#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from build_toolcall_format_public_mix import write_jsonl
from build_toolcall_multicall_gap_curriculum import (
    complex_extraction_instance,
    missing_call_instance,
    source_family,
)
from eval_toolcall_jsonl import extract_tool_calls


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_INPUT = ROOT / "data/toolcall_eval/public_multicall_hermes_smoke.jsonl"
DEFAULT_OUT = ROOT / "data/toolcall_eval/public_multicall_gap_eval.jsonl"


def load_cases(path, limit):
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
            if limit and len(cases) >= limit:
                break
    return cases


def case_instance(case):
    messages = list(case.get("prompt_messages") or [])
    messages.append({"role": "assistant", "content": case.get("gold_assistant") or ""})
    return {
        "source": f"{case.get('source') or 'case'}:{case.get('id') or 'unknown'}",
        "tools": case.get("tools") or [],
        "messages": messages,
    }


def eval_row(candidate, case, row_idx):
    messages = list(candidate.get("messages") or [])
    if len(messages) < 2:
        raise ValueError(f"candidate {candidate.get('source')} has no user/assistant pair")
    prompt_messages = [
        {"role": "system", "content": candidate.get("system") or "You are a helpful assistant."},
        {"role": "user", "content": messages[0].get("content") or ""},
    ]
    gold = messages[1].get("content") or ""
    calls, invalid = extract_tool_calls(gold)
    names = [call.get("name") for call in calls if call.get("name")]
    source = candidate.get("source") or ""
    if ":missing_call:" in source:
        gap_kind = "missing_call"
    elif ":complex_extract:" in source:
        gap_kind = "complex_extract"
    else:
        gap_kind = source_family(candidate)
    family = f"public_multicall_gap_eval:{gap_kind}"
    return {
        "source": family,
        "id": f"{case.get('id') or row_idx}:{candidate.get('source') or row_idx}",
        "parent_id": case.get("id"),
        "parent_source": case.get("source"),
        "task": case.get("task"),
        "category": case.get("category"),
        "gap_source": candidate.get("source"),
        "gap_kind": gap_kind,
        "gap_family": family,
        "tools": candidate.get("tools") or [],
        "prompt_messages": prompt_messages,
        "gold_assistant": gold,
        "gold_tool_names": names,
        "available_tool_names": [
            ((tool.get("function") or tool).get("name"))
            for tool in candidate.get("tools") or []
            if isinstance(tool, dict) and ((tool.get("function") or tool).get("name"))
        ],
        "gold_tool_calls": calls,
        "gold_invalid_tool_json_count": invalid,
    }


def build_rows(cases, args):
    rows = []
    skipped = Counter()
    for case in cases:
        instance = case_instance(case)
        calls, invalid = extract_tool_calls(case.get("gold_assistant") or "")
        if invalid or len(calls) < 2:
            skipped["not_multicall"] += 1
            continue
        if args.include_missing_call:
            for call_index, _ in enumerate(calls):
                candidate = missing_call_instance(instance, calls, call_index, args.max_excerpt_chars)
                rows.append(eval_row(candidate, case, len(rows)))
        if args.include_complex_extract:
            for call_index, _ in enumerate(calls):
                candidate = complex_extraction_instance(instance, calls, call_index, args.max_excerpt_chars)
                if candidate is None:
                    skipped["no_complex_props"] += 1
                else:
                    rows.append(eval_row(candidate, case, len(rows)))
    return rows, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-excerpt-chars", type=int, default=1200)
    parser.add_argument("--include-missing-call", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-complex-extract", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "scripts"))
    cases = load_cases(args.input_jsonl, args.limit)
    rows, skipped = build_rows(cases, args)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_jsonl, rows)
    family_counts = Counter(row["gap_family"] for row in rows)
    kind_counts = Counter(row["gap_kind"] for row in rows)
    summary = {
        "input_jsonl": str(args.input_jsonl),
        "out_jsonl": str(args.out_jsonl),
        "source_cases": len(cases),
        "rows": len(rows),
        "kind_counts": dict(sorted(kind_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "skipped_counts": dict(sorted(skipped.items())),
        "max_excerpt_chars": args.max_excerpt_chars,
        "include_missing_call": args.include_missing_call,
        "include_complex_extract": args.include_complex_extract,
    }
    summary_path = args.out_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
