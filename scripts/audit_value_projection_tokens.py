#!/usr/bin/env python3
"""Audit projected VALUE tokens with tokenizer offsets over Qwen XML output."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


PARAMETER_RE = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.DOTALL)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def schedule_events(row: dict) -> dict:
    direct = row.get("sampler_schedule_events")
    if isinstance(direct, dict):
        return direct
    meta = row.get("backend_meta") or {}
    nested = meta.get("sampler_schedule_events")
    return nested if isinstance(nested, dict) else {}


def event_int(row: dict, key: str) -> int:
    try:
        return int(schedule_events(row).get(key) or 0)
    except Exception:
        return 0


def value_spans(text: str) -> list[dict]:
    spans = []
    for match in PARAMETER_RE.finditer(text or ""):
        start, end = match.span(2)
        if start < end and text[start] == "\n":
            start += 1
        if end > start and text[end - 1] == "\n":
            end -= 1
        if start < end:
            spans.append({"key": match.group(1), "start": start, "end": end})
    return spans


def token_offsets_from_generated_ids(tokenizer, token_ids: list[int]) -> tuple[str, list[tuple[int, int]]]:
    pieces = []
    offsets = []
    cursor = 0
    for token_id in token_ids:
        piece = tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        start = cursor
        cursor += len(piece)
        offsets.append((start, cursor))
        pieces.append(piece)
    return "".join(pieces), offsets


def row_text_and_token_offsets(tokenizer, row: dict) -> tuple[str, list[tuple[int, int]], str]:
    token_ids = row.get("generated_token_ids")
    if isinstance(token_ids, list):
        usable_ids = []
        for token_id in token_ids:
            try:
                usable_ids.append(int(token_id))
            except Exception:
                return row.get("assistant") or "", [], "invalid_generated_token_ids"
        text, offsets = token_offsets_from_generated_ids(tokenizer, usable_ids)
        return text, offsets, "generated_token_ids"
    text = row.get("assistant") or ""
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    return text, offsets, "retokenized_text"


def token_count_for_spans(offsets: list[tuple[int, int]], spans: list[dict]) -> int:
    if not spans:
        return 0
    count = 0
    for start, end in offsets:
        token_start, token_end = int(start), int(end)
        if token_end <= token_start:
            continue
        if any(token_end > span["start"] and token_start < span["end"] for span in spans):
            count += 1
    return count


def output_value_token_count(tokenizer, row: dict) -> int:
    text, offsets, _ = row_text_and_token_offsets(tokenizer, row)
    return token_count_for_spans(offsets, value_spans(text))


def projected_value_tokens_exact(tokenizer, row: dict) -> tuple[int | None, int, str]:
    records = schedule_events(row).get("two_wave_wave1_projected_token_records")
    if not isinstance(records, list):
        return None, 0, "no_projected_token_records"
    text, offsets, offset_source = row_text_and_token_offsets(tokenizer, row)
    spans = value_spans(text)
    value_positions = set()
    for idx, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        if any(end > span["start"] and start < span["end"] for span in spans):
            value_positions.add(idx)
    projected_value = 0
    projected_total = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        rel_idx = record.get("rel_idx")
        if rel_idx is None:
            continue
        try:
            rel_idx = int(rel_idx)
        except Exception:
            continue
        projected_total += 1
        projected_value += int(rel_idx in value_positions)
    return projected_value, projected_total, offset_source


def audit_rows(tokenizer, rows: list[dict]) -> tuple[dict, list[dict]]:
    totals = Counter()
    audited = []
    for idx, row in enumerate(rows):
        forwards = event_int(row, "denoise_forwards_total")
        projected = event_int(row, "two_wave_wave1_projected_tokens")
        true_value_tokens = output_value_token_count(tokenizer, row)
        model_value_tokens = max(
            event_int(row, "two_wave_wave2_value_tokens"),
            event_int(row, "parallel_commit_value_tokens"),
        ) + event_int(row, "two_wave_wave1_value_tokens")
        projected_value_lb = true_value_tokens if forwards == 0 else max(0, true_value_tokens - model_value_tokens)
        projected_value_lb = min(projected_value_lb, projected)
        projected_value_exact, projected_record_count, offset_source = projected_value_tokens_exact(tokenizer, row)
        projected_value_for_gate = (
            projected_value_exact if projected_value_exact is not None else projected_value_lb
        )
        exact = bool(row.get("exact_arguments"))
        out = {
            "row_idx": idx,
            "id": row.get("id"),
            "idx": row.get("idx"),
            "episode_id": row.get("episode_id"),
            "episode_idx": row.get("episode_idx"),
            "turn_idx": row.get("turn_idx"),
            "exact_arguments": exact,
            "denoise_forwards_total": forwards,
            "wave1_projected_tokens": projected,
            "true_xml_value_tokens": true_value_tokens,
            "reported_model_value_tokens": model_value_tokens,
            "projected_true_value_tokens_lower_bound": projected_value_lb,
            "projected_value_tokens_exact": projected_value_exact,
            "projected_token_record_count": projected_record_count,
            "offset_source": offset_source,
            "exact_depends_on_projected_values": exact and projected_value_for_gate > 0,
        }
        audited.append(out)
        totals["rows"] += 1
        totals["exact_args"] += int(exact)
        totals["zero_forward_rows"] += int(forwards == 0)
        totals["zero_forward_rows_with_values"] += int(forwards == 0 and true_value_tokens > 0)
        totals["wave1_projected_tokens"] += projected
        totals["true_xml_value_tokens"] += true_value_tokens
        totals["reported_model_value_tokens"] += model_value_tokens
        totals["projected_true_value_tokens_lower_bound"] += projected_value_lb
        totals["rows_with_projected_true_value_tokens_lower_bound"] += int(projected_value_lb > 0)
        totals["projected_value_tokens_exact"] += int(projected_value_exact or 0)
        totals["rows_with_projected_value_tokens_exact"] += int((projected_value_exact or 0) > 0)
        totals["projected_token_record_count"] += int(projected_record_count or 0)
        totals["rows_with_projected_token_records"] += int(projected_value_exact is not None)
        totals[f"offset_source:{offset_source}"] += 1
        totals["exact_rows_dependent_on_projected_values"] += int(exact and projected_value_for_gate > 0)
        totals["wave1_value_tokens_counter"] += event_int(row, "two_wave_wave1_value_tokens")
        totals["wave2_forced_tokens_counter"] += event_int(row, "two_wave_wave2_forced_tokens")
        totals["parallel_commit_forced_tokens_counter"] += event_int(row, "parallel_commit_forced_tokens")
    if totals["wave1_projected_tokens"] == 0:
        totals["zero_projected_value_tokens_verified"] = 1
        totals["verification_mode"] = "no_projection_events"
    elif totals["rows_with_projected_token_records"]:
        totals["zero_projected_value_tokens_verified"] = int(totals["projected_value_tokens_exact"] == 0)
        if totals.get("offset_source:generated_token_ids", 0):
            totals["verification_mode"] = "projected_token_records_x_generated_token_offsets"
        else:
            totals["verification_mode"] = "projected_token_records_x_retokenized_offsets"
    else:
        totals["zero_projected_value_tokens_verified"] = int(
            totals["projected_true_value_tokens_lower_bound"] == 0
            and totals["zero_forward_rows_with_values"] == 0
        )
        totals["verification_mode"] = "legacy_lower_bound"
    return dict(totals), audited


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), trust_remote_code=True)
    totals, audited = audit_rows(tokenizer, read_jsonl(args.rows))
    result = {
        "rows": str(args.rows),
        "tokenizer": str(args.tokenizer),
        "totals": totals,
        "per_row_jsonl": str(args.out_jsonl),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for row in audited:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if totals["zero_projected_value_tokens_verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
