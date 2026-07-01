#!/usr/bin/env python3
"""Analyze tool-call eval outputs by copy-vs-derived argument values."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/home/mark/qwen_diffusion")
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from eval_toolcall_jsonl import qwen_native_parameter_value  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def canonical_value(value: Any) -> str:
    def normalize(item: Any) -> Any:
        if isinstance(item, str):
            stripped = item.strip()
            try:
                return normalize(json.loads(stripped))
            except Exception:
                return stripped
        if isinstance(item, list):
            return [normalize(child) for child in item]
        if isinstance(item, dict):
            return {str(key): normalize(child) for key, child in item.items()}
        return item

    normalized = normalize(value)
    if isinstance(normalized, (list, dict)):
        return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if isinstance(normalized, bool) or normalized is None or isinstance(normalized, (int, float)):
        return json.dumps(normalized, sort_keys=True)
    return str(normalized).strip()


def rendered_value(value: Any) -> str:
    return qwen_native_parameter_value(value).strip()


def prompt_text(case: dict[str, Any]) -> str:
    parts = []
    for message in case.get("prompt_messages") or []:
        parts.append(str(message.get("content") or ""))
    return "\n".join(parts)


def explicit_copy_keys(case: dict[str, Any]) -> set[tuple[int | None, str, str, str]]:
    keys = set()
    for span in case.get("copy_spans") or []:
        function = str(span.get("function") or "")
        parameter = str(span.get("parameter") or "")
        value = canonical_value(span.get("value_text") or "")
        keys.add((None, function, parameter, value))
    return keys


def label_for_value(
    case: dict[str, Any],
    call_index: int,
    function: str,
    parameter: str,
    value: Any,
    explicit_keys: set[tuple[int | None, str, str, str]],
) -> str:
    canon = canonical_value(value)
    if (None, function, parameter, canon) in explicit_keys:
        return "copy"
    text = prompt_text(case)
    rendered = rendered_value(value)
    if rendered and rendered in text:
        return "copy"
    if isinstance(value, str) and value.strip() and value.strip() in text:
        return "copy"
    return "derived"


def predicted_arg(output: dict[str, Any], call_index: int, function: str, parameter: str) -> Any:
    calls = output.get("calls") or []
    if call_index >= len(calls):
        return None
    call = calls[call_index]
    if call.get("name") != function:
        return None
    arguments = call.get("arguments") or {}
    if not isinstance(arguments, dict) or parameter not in arguments:
        return None
    return arguments[parameter]


def event_value_tpf(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tokens = 0
    forwards = 0
    structural_tokens = 0
    structural_forwards = 0
    for row in rows:
        events = row.get("sampler_schedule_events") or {}
        tokens += int(events.get("parallel_commit_value_tokens") or 0)
        forwards += int(events.get("parallel_commit_value_forward_visits") or 0)
        structural_tokens += int(events.get("parallel_commit_structural_tokens") or 0)
        structural_forwards += int(events.get("parallel_commit_structural_forward_visits") or 0)
    return {
        "value_tokens": tokens,
        "value_forward_visits": forwards,
        "value_tokens_per_forward": tokens / forwards if forwards else None,
        "structural_tokens": structural_tokens,
        "structural_forward_visits": structural_forwards,
        "structural_tokens_per_forward": structural_tokens / structural_forwards if structural_forwards else None,
    }


def analyze(cases: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {str(row.get("id")): row for row in outputs}
    totals = defaultdict(Counter)
    per_record = []
    missing_outputs = 0
    for case in cases:
        case_id = str(case.get("id"))
        output = by_id.get(case_id)
        if output is None:
            missing_outputs += 1
            continue
        explicit_keys = explicit_copy_keys(case)
        record_counts = defaultdict(Counter)
        for call_index, call in enumerate(case.get("gold_tool_calls") or []):
            function = str(call.get("name") or "")
            arguments = call.get("arguments") or {}
            if not isinstance(arguments, dict):
                continue
            for parameter, gold_value in arguments.items():
                label = label_for_value(case, call_index, function, str(parameter), gold_value, explicit_keys)
                pred_value = predicted_arg(output, call_index, function, str(parameter))
                exact = pred_value is not None and canonical_value(pred_value) == canonical_value(gold_value)
                totals[label]["arguments"] += 1
                totals[label]["exact"] += int(exact)
                record_counts[label]["arguments"] += 1
                record_counts[label]["exact"] += int(exact)
        per_record.append(
            {
                "id": case_id,
                "valid_tool_json": bool(output.get("valid_tool_json")),
                "exact_arguments_record": bool(output.get("exact_arguments")),
                "groups": {
                    label: {
                        "arguments": counts["arguments"],
                        "exact": counts["exact"],
                        "accuracy": counts["exact"] / counts["arguments"] if counts["arguments"] else None,
                    }
                    for label, counts in sorted(record_counts.items())
                },
            }
        )

    groups = {}
    for label, counts in sorted(totals.items()):
        groups[label] = {
            "arguments": counts["arguments"],
            "exact": counts["exact"],
            "accuracy": counts["exact"] / counts["arguments"] if counts["arguments"] else None,
        }
    return {
        "records": len(cases),
        "outputs": len(outputs),
        "missing_outputs": missing_outputs,
        "groups": groups,
        "record_exact_arguments": sum(int(row.get("exact_arguments")) for row in outputs),
        "record_valid_tool_json": sum(int(row.get("valid_tool_json")) for row in outputs),
        "sampler": event_value_tpf(outputs),
        "per_record": per_record,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cases = read_jsonl(args.cases)
    report = {"cases": str(args.cases), "runs": {}}
    for output_path in args.outputs:
        run_key = f"{output_path.parent.name}/{output_path.stem}"
        report["runs"][run_key] = {
            "output_jsonl": str(output_path),
            **analyze(cases, read_jsonl(output_path)),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
