#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def chunk_size(kind, args):
    if kind == "prose":
        return args.prose_block_tokens
    if kind == "argument_value":
        return args.argument_value_block_tokens
    if kind == "json_structure":
        return args.json_structure_block_tokens
    return args.tiny_block_tokens


def policy_for(block):
    return block.get("policy") or {}


def copy_metadata(block, item):
    for key in ["tool_call_index", "json_key", "json_path", "argument_path", "target_text", "segment_texts"]:
        if key in block:
            item[key] = block[key]


def schedule_for_record(record, args):
    schedule = []
    for block_idx, block in enumerate(record.get("token_blocks") or []):
        kind = block["kind"]
        step = max(1, int(policy_for(block).get("suggested_steps") or 1))
        constraint = policy_for(block).get("constrain") or "none"
        max_tokens = max(1, int(chunk_size(kind, args)))
        cursor = int(block["token_start"])
        token_end = int(block["token_end"])
        block_token_ids = block.get("token_ids") or []
        while cursor < token_end:
            next_cursor = min(token_end, cursor + max_tokens)
            item = {
                "token_start": cursor,
                "token_end": next_cursor,
                "token_count": next_cursor - cursor,
                "kind": kind,
                "denoise_steps": step,
                "constraint": constraint,
                "source_token_block_idx": block_idx,
                "source_block_token_start": int(block["token_start"]),
                "source_block_token_end": int(block["token_end"]),
                "source_block_token_count": int(block["token_end"]) - int(block["token_start"]),
                "source_block_token_offset": cursor - int(block["token_start"]),
            }
            copy_metadata(block, item)
            if args.include_token_ids and block_token_ids:
                offset = cursor - int(block["token_start"])
                item["target_token_ids"] = block_token_ids[offset : offset + item["token_count"]]
            schedule.append(item)
            cursor = next_cursor
    return schedule


def summarize(records):
    totals = Counter()
    for record in records:
        totals["records"] += 1
        totals["tokens"] += int(record.get("token_count") or 0)
        totals["source_token_blocks"] += int(record.get("source_token_blocks") or 0)
        totals["schedule_blocks"] += len(record.get("schedule") or [])
        for item in record.get("schedule") or []:
            totals[f"schedule_blocks:{item['kind']}"] += 1
            totals[f"scheduled_tokens:{item['kind']}"] += int(item["token_count"])
            totals[f"denoise_step_tokens:{item['denoise_steps']}"] += int(item["token_count"])
    return dict(totals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, default=None)
    parser.add_argument("--prose-block-tokens", type=int, default=128)
    parser.add_argument("--argument-value-block-tokens", type=int, default=8)
    parser.add_argument("--json-structure-block-tokens", type=int, default=4)
    parser.add_argument("--tiny-block-tokens", type=int, default=1)
    parser.add_argument(
        "--include-token-ids",
        action="store_true",
        help="Carry token ids from tokenized block plans into schedule rows as target_token_ids.",
    )
    args = parser.parse_args()

    records = []
    missing_token_blocks = 0
    for row in load_jsonl(args.input_jsonl):
        if not row.get("token_blocks"):
            missing_token_blocks += 1
        schedule = schedule_for_record(row, args)
        records.append(
            {
                "id": row.get("id"),
                "source": row.get("source"),
                "text_field": row.get("text_field"),
                "token_count": row.get("token_count"),
                "tool_call_count": row.get("tool_call_count"),
                "source_token_blocks": len(row.get("token_blocks") or []),
                "schedule": schedule,
            }
        )

    summary = {
        "input_jsonl": str(args.input_jsonl),
        "out_jsonl": str(args.out_jsonl) if args.out_jsonl else None,
        "block_token_limits": {
            "prose": args.prose_block_tokens,
            "argument_value": args.argument_value_block_tokens,
            "json_structure": args.json_structure_block_tokens,
            "tiny": args.tiny_block_tokens,
        },
        "missing_token_block_records": missing_token_blocks,
        "totals": summarize(records),
    }

    if args.out_jsonl:
        args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.out_jsonl.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        args.out_jsonl.with_suffix(".summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
