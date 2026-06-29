#!/usr/bin/env python3
"""Diagnose whether generated Qwen tool-call blocks stayed JSON-prefix-completable.

The existing eval scripts score final strings and optional repair/projection
outputs. This script looks one layer lower: did a generated `<tool_call>` body
remain a prefix of valid JSON, or did it enter an unrecoverable grammar state
before the final repair pass saw it?
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from eval_toolcall_jsonl import extract_tool_calls, score_tool_calls, tool_schema_by_name  # noqa: E402


TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"


class JsonPrefixParser:
    def __init__(self, text: str):
        self.text = text
        self.n = len(text)
        self.error_pos: int | None = None
        self.error_reason = ""

    def fail(self, pos: int, reason: str) -> tuple[str, int]:
        self.error_pos = pos
        self.error_reason = reason
        return "invalid", pos

    def skip_ws(self, pos: int) -> int:
        while pos < self.n and self.text[pos] in " \t\r\n":
            pos += 1
        return pos

    def parse_string(self, pos: int) -> tuple[str, int]:
        if pos >= self.n:
            return "incomplete", pos
        if self.text[pos] != '"':
            return self.fail(pos, "expected string")
        pos += 1
        escape = False
        while pos < self.n:
            char = self.text[pos]
            if escape:
                if char in '"\\/bfnrt':
                    escape = False
                    pos += 1
                    continue
                if char == "u":
                    if pos + 4 >= self.n:
                        return "incomplete", pos
                    digits = self.text[pos + 1 : pos + 5]
                    if all(item in "0123456789abcdefABCDEF" for item in digits):
                        escape = False
                        pos += 5
                        continue
                    return self.fail(pos, "invalid unicode escape")
                return self.fail(pos, "invalid escape")
            if char == "\\":
                escape = True
                pos += 1
                continue
            if char == '"':
                return "complete", pos + 1
            if ord(char) < 0x20:
                return self.fail(pos, "control character in string")
            pos += 1
        return "incomplete", pos

    def parse_literal(self, pos: int, literal: str) -> tuple[str, int]:
        remaining = self.text[pos:]
        if literal.startswith(remaining):
            return "incomplete", self.n
        if self.text.startswith(literal, pos):
            return "complete", pos + len(literal)
        return self.fail(pos, f"expected {literal}")

    def parse_number(self, pos: int) -> tuple[str, int]:
        start = pos
        if pos < self.n and self.text[pos] == "-":
            pos += 1
            if pos >= self.n:
                return "incomplete", pos
        if pos >= self.n:
            return "incomplete", pos
        if self.text[pos] == "0":
            pos += 1
        elif self.text[pos].isdigit() and self.text[pos] != "0":
            while pos < self.n and self.text[pos].isdigit():
                pos += 1
        else:
            return self.fail(pos, "expected number")
        if pos < self.n and self.text[pos] == ".":
            pos += 1
            if pos >= self.n:
                return "incomplete", pos
            if not self.text[pos].isdigit():
                return self.fail(pos, "expected digit after decimal point")
            while pos < self.n and self.text[pos].isdigit():
                pos += 1
        if pos < self.n and self.text[pos] in "eE":
            pos += 1
            if pos < self.n and self.text[pos] in "+-":
                pos += 1
            if pos >= self.n:
                return "incomplete", pos
            if not self.text[pos].isdigit():
                return self.fail(pos, "expected exponent digit")
            while pos < self.n and self.text[pos].isdigit():
                pos += 1
        if pos == start:
            return self.fail(pos, "expected number")
        return "complete", pos

    def parse_array(self, pos: int) -> tuple[str, int]:
        pos += 1
        pos = self.skip_ws(pos)
        if pos >= self.n:
            return "incomplete", pos
        if self.text[pos] == "]":
            return "complete", pos + 1
        while True:
            status, pos = self.parse_value(pos)
            if status != "complete":
                return status, pos
            pos = self.skip_ws(pos)
            if pos >= self.n:
                return "incomplete", pos
            if self.text[pos] == ",":
                pos += 1
                pos = self.skip_ws(pos)
                if pos >= self.n:
                    return "incomplete", pos
                continue
            if self.text[pos] == "]":
                return "complete", pos + 1
            return self.fail(pos, "expected comma or array close")

    def parse_object(self, pos: int) -> tuple[str, int]:
        pos += 1
        pos = self.skip_ws(pos)
        if pos >= self.n:
            return "incomplete", pos
        if self.text[pos] == "}":
            return "complete", pos + 1
        while True:
            if pos >= self.n:
                return "incomplete", pos
            if self.text[pos] != '"':
                return self.fail(pos, "expected object key string")
            status, pos = self.parse_string(pos)
            if status != "complete":
                return status, pos
            pos = self.skip_ws(pos)
            if pos >= self.n:
                return "incomplete", pos
            if self.text[pos] != ":":
                return self.fail(pos, "expected colon after object key")
            pos += 1
            pos = self.skip_ws(pos)
            status, pos = self.parse_value(pos)
            if status != "complete":
                return status, pos
            pos = self.skip_ws(pos)
            if pos >= self.n:
                return "incomplete", pos
            if self.text[pos] == ",":
                pos += 1
                pos = self.skip_ws(pos)
                if pos >= self.n:
                    return "incomplete", pos
                continue
            if self.text[pos] == "}":
                return "complete", pos + 1
            return self.fail(pos, "expected comma or object close")

    def parse_value(self, pos: int) -> tuple[str, int]:
        pos = self.skip_ws(pos)
        if pos >= self.n:
            return "incomplete", pos
        char = self.text[pos]
        if char == "{":
            return self.parse_object(pos)
        if char == "[":
            return self.parse_array(pos)
        if char == '"':
            return self.parse_string(pos)
        if char == "t":
            return self.parse_literal(pos, "true")
        if char == "f":
            return self.parse_literal(pos, "false")
        if char == "n":
            return self.parse_literal(pos, "null")
        if char == "-" or char.isdigit():
            return self.parse_number(pos)
        return self.fail(pos, "expected JSON value")

    def parse_document(self) -> dict[str, Any]:
        pos = self.skip_ws(0)
        if pos >= self.n:
            return {"status": "empty", "pos": pos, "reason": "empty"}
        status, pos = self.parse_value(pos)
        if status != "complete":
            return {"status": status, "pos": pos, "reason": self.error_reason}
        after = self.skip_ws(pos)
        if after < self.n:
            return {
                "status": "invalid",
                "pos": after,
                "reason": "extra content after JSON value",
            }
        return {"status": "complete", "pos": after, "reason": ""}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_cases_for_output(output_path: Path, explicit_cases: Path | None) -> dict[str, dict[str, Any]]:
    cases_path = explicit_cases
    if cases_path is None:
        summary_path = output_path.with_suffix(".summary.json")
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                raw = summary.get("cases_jsonl") or summary.get("input_jsonl")
                if raw:
                    candidate = Path(raw)
                    if not candidate.is_absolute():
                        candidate = ROOT / candidate
                    if candidate.exists():
                        cases_path = candidate
            except Exception:
                cases_path = None
    if cases_path is None or not cases_path.exists():
        return {}
    return {str(row.get("id") or row.get("source") or idx): row for idx, row in enumerate(load_jsonl(cases_path))}


def tool_call_segments(text: str) -> list[dict[str, Any]]:
    segments = []
    cursor = 0
    while True:
        start = text.find(TOOL_OPEN, cursor)
        if start < 0:
            break
        body_start = start + len(TOOL_OPEN)
        end = text.find(TOOL_CLOSE, body_start)
        if end < 0:
            segments.append(
                {
                    "body": text[body_start:],
                    "complete_tag": False,
                    "start": start,
                    "end": len(text),
                }
            )
            break
        segments.append(
            {
                "body": text[body_start:end],
                "complete_tag": True,
                "start": start,
                "end": end + len(TOOL_CLOSE),
            }
        )
        cursor = end + len(TOOL_CLOSE)
    return segments


def first_json_candidate(body: str) -> tuple[str, int]:
    start = body.find("{")
    if start < 0:
        return "", -1
    return body[start:].strip(), start


def duplicate_top_keys(body: str) -> list[str]:
    snippet, _ = first_json_candidate(body)
    if not snippet:
        return []
    pairs_seen = []

    def hook(pairs):
        if not pairs_seen:
            pairs_seen.extend(key for key, _ in pairs)
        return dict(pairs)

    try:
        json.loads(snippet, object_pairs_hook=hook)
    except Exception:
        return []
    counts = Counter(pairs_seen)
    return sorted(key for key, count in counts.items() if count > 1)


def analyze_segment(segment: dict[str, Any]) -> dict[str, Any]:
    body = segment.get("body") or ""
    snippet, json_offset = first_json_candidate(body)
    if json_offset < 0:
        return {
            "complete_tag": bool(segment.get("complete_tag")),
            "json_status": "no_json",
            "prefix_completable": False,
            "error_pos": None,
            "error_reason": "no JSON object start",
            "duplicate_top_keys": [],
            "body_preview": body[:240],
        }

    result = JsonPrefixParser(snippet).parse_document()
    status = result["status"]
    prefix_completable = status in {"complete", "incomplete"}
    return {
        "complete_tag": bool(segment.get("complete_tag")),
        "json_status": status,
        "prefix_completable": prefix_completable,
        "error_pos": None if result.get("pos") is None else int(result["pos"]) + json_offset,
        "error_reason": result.get("reason") or "",
        "duplicate_top_keys": duplicate_top_keys(body) if status == "complete" else [],
        "body_preview": body[:240],
    }


def row_key(row: dict[str, Any], fallback: int) -> str:
    return str(row.get("id") or row.get("source") or fallback)


def case_context(row: dict[str, Any], cases: dict[str, dict[str, Any]], fallback: int) -> dict[str, Any]:
    key = row_key(row, fallback)
    case = cases.get(key) or {}
    return {
        "tools": case.get("tools") or [],
        "gold_assistant": case.get("gold_assistant"),
        "gold_tool_names": case.get("gold_tool_names") or row.get("gold_tool_names") or [],
    }


def exact_metric(row: dict[str, Any], field: str, metric: str) -> Any:
    if field == "assistant":
        return row.get(metric)
    prefix = field.removesuffix("_assistant")
    return row.get(f"{prefix}_{metric}")


def summarize_field(
    rows: list[dict[str, Any]],
    cases: dict[str, dict[str, Any]],
    field: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    totals: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    row_details = []

    for idx, row in enumerate(rows):
        text = row.get(field)
        if not isinstance(text, str) or not text.strip():
            continue
        totals["rows_present"] += 1
        segments = tool_call_segments(text)
        totals["tool_open_count"] += text.count(TOOL_OPEN)
        totals["tool_close_count"] += text.count(TOOL_CLOSE)
        if segments:
            totals["rows_with_tool_call"] += 1
        else:
            totals["rows_without_tool_call"] += 1

        analyses = [analyze_segment(segment) for segment in segments]
        complete_tags = sum(1 for item in analyses if item["complete_tag"])
        missing_tags = sum(1 for item in analyses if not item["complete_tag"])
        invalid_segments = [item for item in analyses if item["json_status"] in {"invalid", "empty", "no_json"}]
        incomplete_segments = [item for item in analyses if item["json_status"] == "incomplete"]
        complete_json_segments = [item for item in analyses if item["json_status"] == "complete"]
        duplicate_key_segments = [item for item in analyses if item.get("duplicate_top_keys")]

        totals["segments"] += len(segments)
        totals["complete_tag_segments"] += complete_tags
        totals["missing_close_tag_segments"] += missing_tags
        totals["json_complete_segments"] += len(complete_json_segments)
        totals["json_incomplete_but_prefix_completable_segments"] += len(incomplete_segments)
        totals["json_invalid_segments"] += len(invalid_segments)
        totals["duplicate_top_key_segments"] += len(duplicate_key_segments)

        if missing_tags:
            totals["rows_with_missing_close_tag"] += 1
        if incomplete_segments:
            totals["rows_with_incomplete_prefix_completable_json"] += 1
        if invalid_segments:
            totals["rows_with_unrecoverable_json_prefix"] += 1
            for item in invalid_segments:
                reasons[item["error_reason"] or item["json_status"]] += 1
        if complete_json_segments and not invalid_segments and not incomplete_segments:
            totals["rows_all_json_complete"] += 1
        if analyses and all(item["prefix_completable"] for item in analyses):
            totals["rows_all_segments_prefix_completable"] += 1
        if duplicate_key_segments:
            totals["rows_with_duplicate_top_keys"] += 1

        ctx = case_context(row, cases, idx)
        tools = ctx["tools"]
        if tools:
            metrics = score_tool_calls(text, tools, ctx.get("gold_assistant"))
            totals["score_valid_tool_call"] += int(bool(metrics.get("valid_tool_call")))
            totals["score_exact_tool_sequence"] += int(bool(metrics.get("exact_tool_sequence")))
            totals["score_exact_arguments"] += int(bool(metrics.get("exact_arguments")))
            totals["score_all_schema_valid"] += int(bool(metrics.get("all_schema_valid")))
            totals["score_all_required_args_present"] += int(bool(metrics.get("all_required_args_present")))
            schemas = tool_schema_by_name(tools)
            unknown_names = [name for name in metrics.get("called_names", []) if name not in schemas]
            if unknown_names:
                totals["rows_with_unknown_tool_name"] += 1
        else:
            metrics = {}

        expected_calls = len(ctx.get("gold_tool_names") or [])
        if expected_calls:
            totals["expected_tool_calls"] += expected_calls
            if len(segments) < expected_calls:
                totals["rows_with_too_few_tool_call_segments"] += 1
            if len(segments) > expected_calls:
                totals["rows_with_too_many_tool_call_segments"] += 1

        if (
            invalid_segments
            or incomplete_segments
            or missing_tags
            or not bool(exact_metric(row, field, "exact_arguments"))
        ):
            row_details.append(
                {
                    "id": row_key(row, idx),
                    "field": field,
                    "expected_calls": expected_calls,
                    "segments": len(segments),
                    "complete_tag_segments": complete_tags,
                    "missing_close_tag_segments": missing_tags,
                    "json_statuses": [item["json_status"] for item in analyses],
                    "first_error_reason": (invalid_segments or incomplete_segments or [{}])[0].get("error_reason", ""),
                    "exact_tool_sequence": exact_metric(row, field, "exact_tool_sequence"),
                    "exact_arguments": exact_metric(row, field, "exact_arguments"),
                    "called_names": metrics.get("called_names") or row.get("called_names"),
                    "preview": text[:360],
                }
            )

    summary = dict(totals)
    summary["top_unrecoverable_reasons"] = reasons.most_common(10)
    return summary, row_details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path, nargs="+", help="Eval output JSONL files to diagnose.")
    parser.add_argument("--cases-jsonl", type=Path, default=None, help="Original case JSONL. Defaults to .summary input_jsonl.")
    parser.add_argument(
        "--field",
        dest="fields",
        action="append",
        default=[],
        help="Output text field to diagnose. Can be repeated. Defaults to assistant and constrained_assistant.",
    )
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-jsonl", type=Path, default=None)
    parser.add_argument("--max-examples", type=int, default=8)
    args = parser.parse_args()

    fields = args.fields or ["assistant", "constrained_assistant"]
    report = {"inputs": [], "fields": fields}
    detail_rows = []
    for path in args.jsonl:
        rows = load_jsonl(path)
        cases = load_cases_for_output(path, args.cases_jsonl)
        input_report = {
            "path": str(path),
            "rows": len(rows),
            "cases_loaded": len(cases),
            "field_summaries": {},
        }
        for field in fields:
            summary, details = summarize_field(rows, cases, field)
            input_report["field_summaries"][field] = summary
            detail_rows.extend({"source_jsonl": str(path), **item} for item in details)
        report["inputs"].append(input_report)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.out_jsonl:
        args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.out_jsonl.open("w", encoding="utf-8") as handle:
            for item in detail_rows:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(json.dumps(report, indent=2))
    if detail_rows:
        print("\nExamples:")
        for item in detail_rows[: args.max_examples]:
            compact = {
                key: item[key]
                for key in [
                    "source_jsonl",
                    "id",
                    "field",
                    "expected_calls",
                    "segments",
                    "json_statuses",
                    "first_error_reason",
                    "exact_tool_sequence",
                    "exact_arguments",
                ]
                if key in item
            }
            print(json.dumps(compact, ensure_ascii=False))


if __name__ == "__main__":
    main()
