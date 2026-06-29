#!/usr/bin/env python3
"""Prepare merged schedules for the causal value-span decisive test.

This is an inference-only schedule transform. It does not modify model weights
or eval inputs.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


VALUE_GEOMETRY_KEYS = {
    "kind",
    "token_start",
    "token_end",
    "block_size",
    "denoise_steps",
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def value_geometry_item(item: dict, token_start: int | None = None, token_end: int | None = None) -> dict:
    start = int(item["token_start"] if token_start is None else token_start)
    end = int(item["token_end"] if token_end is None else token_end)
    out = {
        "kind": "argument_value",
        "token_start": start,
        "token_end": end,
        "block_size": end - start,
    }
    if "denoise_steps" in item:
        out["denoise_steps"] = item["denoise_steps"]
    return out


def split_argument_value_item(item: dict, max_span_tokens: int) -> list[dict]:
    token_start = int(item["token_start"])
    token_end = int(item["token_end"])
    total = token_end - token_start
    if total <= max_span_tokens:
        chunks = [value_geometry_item(item)]
    else:
        chunks = []
        for offset in range(0, total, max_span_tokens):
            length = min(max_span_tokens, total - offset)
            chunks.append(value_geometry_item(item, token_start + offset, token_start + offset + length))
    return chunks


def raw_geometry_schedule(rows: list[dict]) -> list[dict]:
    transformed = []
    for row in rows:
        out = copy.deepcopy(row)
        schedule = []
        for item in out.get("schedule") or []:
            if item.get("kind") == "argument_value":
                schedule.append(value_geometry_item(item))
            else:
                schedule.append(copy.deepcopy(item))
        out["schedule"] = schedule
        out["raw_baseline_value_schedule"] = {
            "value_intervals_geometry_only": True,
            "value_geometry_keys": sorted(VALUE_GEOMETRY_KEYS),
        }
        transformed.append(out)
    return transformed


def causal_value_schedule(rows: list[dict], max_span_tokens: int) -> list[dict]:
    transformed = []
    for row in rows:
        out = copy.deepcopy(row)
        schedule = []
        for item in out.get("schedule") or []:
            if item.get("kind") == "argument_value":
                schedule.extend(split_argument_value_item(item, max_span_tokens))
            else:
                schedule.append(copy.deepcopy(item))
        schedule.sort(key=lambda item: (int(item.get("token_start", 0)), int(item.get("token_end", 0))))
        out["schedule"] = schedule
        out["causal_value_span_schedule"] = {
            "max_argument_value_span_tokens": max_span_tokens,
            "value_intervals_geometry_only": True,
            "value_geometry_keys": sorted(VALUE_GEOMETRY_KEYS),
            "strict_left_to_right_relies_on_json_prefix_guard": True,
        }
        transformed.append(out)
    return transformed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout-schedule", type=Path, required=True)
    parser.add_argument("--public-schedule", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--causal-value-span-tokens", type=int, default=1)
    args = parser.parse_args()

    if args.causal_value_span_tokens < 1 or args.causal_value_span_tokens > 2:
        raise SystemExit("--causal-value-span-tokens must be 1 or 2 for this decisive test")

    heldout = read_jsonl(args.heldout_schedule)
    public = read_jsonl(args.public_schedule)
    merged = heldout + public

    write_jsonl(args.out_dir / "raw_baseline_merged_schedule.jsonl", raw_geometry_schedule(merged))
    write_jsonl(args.out_dir / "forced_ceiling_merged_schedule.jsonl", copy.deepcopy(merged))
    write_jsonl(
        args.out_dir / "causal_value_span_merged_schedule.jsonl",
        causal_value_schedule(merged, args.causal_value_span_tokens),
    )


if __name__ == "__main__":
    main()
