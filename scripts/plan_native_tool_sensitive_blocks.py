#!/usr/bin/env python3
"""Plan token-sensitive blocks for Qwen-native function/parameter tool calls."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("/home/mark/qwen_diffusion")
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from plan_tool_sensitive_blocks import (  # noqa: E402
    POLICIES,
    add_structural_gaps,
    add_token_spans,
    load_jsonl,
    load_tokenizer,
    split_chunks,
)


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
FUNCTION_RE = re.compile(r"<function=([^>\n]+)>")
PARAMETER_RE = re.compile(r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def native_sensitive_spans(body: str, base_offset: int, tool_call_index: int) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    function = FUNCTION_RE.search(body)
    if function:
        spans.append(
            {
                "start": base_offset + function.start(1),
                "end": base_offset + function.end(1),
                "kind": "tool_name",
                "tool_call_index": tool_call_index,
                "json_key": "function",
                "json_path": "function",
                "argument_path": None,
                "text": function.group(1),
            }
        )
    for parameter in PARAMETER_RE.finditer(body):
        name = parameter.group(1)
        spans.append(
            {
                "start": base_offset + parameter.start(1),
                "end": base_offset + parameter.end(1),
                "kind": "json_key",
                "tool_call_index": tool_call_index,
                "json_key": name,
                "json_path": name,
                "argument_path": None,
                "text": name,
            }
        )
        value_text = parameter.group(2)
        value_start = parameter.start(2)
        value_end = parameter.end(2)
        left_trimmed = len(value_text) - len(value_text.lstrip())
        right_trimmed = len(value_text.rstrip())
        value_start += left_trimmed
        value_end = parameter.start(2) + right_trimmed
        if value_end > value_start:
            spans.append(
                {
                    "start": base_offset + value_start,
                    "end": base_offset + value_end,
                    "kind": "argument_value",
                    "tool_call_index": tool_call_index,
                    "json_key": name,
                    "json_path": name,
                    "argument_path": name,
                    "target_text": body[value_start:value_end],
                    "text": body[value_start:value_end],
                }
            )
    return sorted(spans, key=lambda item: (item["start"], item["end"], item["kind"]))


def plan_text(text: str, max_prose_chars: int, max_structure_chars: int) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    cursor = 0
    for tool_call_index, match in enumerate(TOOL_CALL_RE.finditer(text)):
        if cursor < match.start():
            segments.extend(split_chunks(cursor, match.start(), "prose", max_prose_chars))

        body_start, body_end = match.span(1)
        segments.append(
            {
                "start": match.start(),
                "end": body_start,
                "kind": "tool_tag",
                "tool_call_index": tool_call_index,
            }
        )
        spans = native_sensitive_spans(text[body_start:body_end], body_start, tool_call_index)
        add_structural_gaps(segments, body_start, body_end, spans, max_structure_chars)
        segments.append(
            {
                "start": body_end,
                "end": match.end(),
                "kind": "tool_tag",
                "tool_call_index": tool_call_index,
            }
        )
        cursor = match.end()

    if cursor < len(text):
        segments.extend(split_chunks(cursor, len(text), "prose", max_prose_chars))

    for idx, segment in enumerate(segments):
        segment["idx"] = idx
        segment["chars"] = segment["end"] - segment["start"]
        segment["policy"] = POLICIES[segment["kind"]]
    return segments


def summarize(records: list[dict[str, Any]]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for record in records:
        totals["records"] += 1
        totals["segments"] += len(record["segments"])
        totals["tool_calls"] += int(record.get("tool_call_count") or 0)
        totals["tokens"] += int(record.get("token_count") or 0)
        totals["token_blocks"] += len(record.get("token_blocks") or [])
        for block in record.get("token_blocks") or []:
            totals[f"token_blocks:{block['kind']}"] += 1
            totals[f"block_tokens:{block['kind']}"] += int(block["token_count"])
    return dict(totals)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--text-field", default="gold_assistant")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-prose-chars", type=int, default=512)
    parser.add_argument("--max-native-structure-chars", type=int, default=96)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--include-token-ids", action="store_true")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer_path)
    records = []
    for idx, case in enumerate(load_jsonl(args.input_jsonl)):
        if args.limit and idx >= args.limit:
            break
        text = case.get(args.text_field) or ""
        record = {
            "id": case.get("id") or str(idx),
            "source": case.get("source"),
            "text_field": args.text_field,
            "text": text,
            "text_chars": len(text),
            "tool_call_count": len(TOOL_CALL_RE.findall(text)),
            "segments": plan_text(text, args.max_prose_chars, args.max_native_structure_chars),
        }
        add_token_spans(record, tokenizer, include_token_ids=args.include_token_ids)
        records.append(record)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "input_jsonl": str(args.input_jsonl),
        "out_jsonl": str(args.out_jsonl),
        "text_field": args.text_field,
        "tokenizer_path": str(args.tokenizer_path),
        "totals": summarize(records),
    }
    args.out_jsonl.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
