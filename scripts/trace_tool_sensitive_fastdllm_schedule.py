#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from eval_fastdllm_toolcall_cases import make_prompt, resolve_chat_template


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def case_key(row, fallback_idx):
    return row.get("id") or row.get("case_id") or str(fallback_idx)


def index_cases(cases):
    return {case_key(case, idx): case for idx, case in enumerate(cases)}


def split_by_fastdllm_boundaries(start, end, block_size, small_block_size):
    pieces = []
    cursor = start
    while cursor < end:
        block_end = ((cursor // block_size) + 1) * block_size
        small_end = ((cursor // small_block_size) + 1) * small_block_size
        piece_end = min(end, block_end, small_end)
        pieces.append(
            {
                "abs_token_start": cursor,
                "abs_token_end": piece_end,
                "token_count": piece_end - cursor,
                "fastdllm_block_idx": cursor // block_size,
                "fastdllm_small_block_idx": (cursor % block_size) // small_block_size,
                "block_local_start": cursor % block_size,
                "block_local_end": piece_end % block_size if piece_end % block_size else block_size,
            }
        )
        cursor = piece_end
    return pieces


def trace_record(schedule_record, case, tokenizer, chat_template, args):
    prompt = make_prompt(
        tokenizer,
        case,
        append_instruction=args.append_instruction,
        chat_template=chat_template,
    )
    prompt_ids = tokenizer([prompt], return_tensors=None, add_special_tokens=False)["input_ids"][0]
    prompt_len = len(prompt_ids)

    traced_schedule = []
    for idx, item in enumerate(schedule_record.get("schedule") or []):
        abs_start = prompt_len + int(item["token_start"])
        abs_end = prompt_len + int(item["token_end"])
        pieces = split_by_fastdllm_boundaries(
            abs_start,
            abs_end,
            args.block_size,
            args.small_block_size,
        )
        traced = {
            **item,
            "schedule_idx": idx,
            "abs_token_start": abs_start,
            "abs_token_end": abs_end,
            "fastdllm_block_start": abs_start // args.block_size,
            "fastdllm_block_end": (abs_end - 1) // args.block_size if abs_end > abs_start else abs_start // args.block_size,
            "fastdllm_small_block_start": (abs_start % args.block_size) // args.small_block_size,
            "fastdllm_small_block_end": ((abs_end - 1) % args.block_size) // args.small_block_size
            if abs_end > abs_start
            else (abs_start % args.block_size) // args.small_block_size,
            "crosses_fastdllm_block": bool(pieces and pieces[0]["fastdllm_block_idx"] != pieces[-1]["fastdllm_block_idx"]),
            "crosses_small_block": len(pieces) > 1,
            "fastdllm_pieces": pieces,
        }
        traced_schedule.append(traced)

    return {
        "id": schedule_record.get("id"),
        "source": schedule_record.get("source"),
        "text_field": schedule_record.get("text_field"),
        "prompt_token_count": prompt_len,
        "generated_token_count": schedule_record.get("token_count"),
        "tool_call_count": schedule_record.get("tool_call_count"),
        "source_token_blocks": schedule_record.get("source_token_blocks"),
        "schedule": traced_schedule,
    }


def summarize(records):
    totals = Counter()
    for record in records:
        totals["records"] += 1
        totals["prompt_tokens"] += int(record.get("prompt_token_count") or 0)
        totals["generated_tokens"] += int(record.get("generated_token_count") or 0)
        totals["source_token_blocks"] += int(record.get("source_token_blocks") or 0)
        totals["schedule_blocks"] += len(record.get("schedule") or [])
        for item in record.get("schedule") or []:
            totals[f"schedule_blocks:{item['kind']}"] += 1
            totals[f"scheduled_tokens:{item['kind']}"] += int(item.get("token_count") or 0)
            totals["crosses_fastdllm_block"] += int(bool(item.get("crosses_fastdllm_block")))
            totals["crosses_small_block"] += int(bool(item.get("crosses_small_block")))
            totals["fastdllm_pieces"] += len(item.get("fastdllm_pieces") or [])
            for piece in item.get("fastdllm_pieces") or []:
                totals[f"piece_tokens:{item['kind']}"] += int(piece.get("token_count") or 0)
                totals[f"pieces:{item['kind']}"] += 1
    return dict(totals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule-jsonl", type=Path, required=True)
    parser.add_argument("--cases-jsonl", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--conversation-template", default=None)
    parser.add_argument("--append-instruction", action="store_true")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--small-block-size", type=int, default=8)
    args = parser.parse_args()

    if args.block_size % args.small_block_size:
        raise SystemExit("--block-size must be divisible by --small-block-size")

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_path), trust_remote_code=True)
    chat_template = resolve_chat_template(args.conversation_template)
    cases = index_cases(load_jsonl(args.cases_jsonl))

    records = []
    missing_cases = []
    for idx, schedule_record in enumerate(load_jsonl(args.schedule_jsonl)):
        key = case_key(schedule_record, idx)
        case = cases.get(key)
        if case is None:
            missing_cases.append(key)
            continue
        records.append(trace_record(schedule_record, case, tokenizer, chat_template, args))

    summary = {
        "schedule_jsonl": str(args.schedule_jsonl),
        "cases_jsonl": str(args.cases_jsonl),
        "tokenizer_path": str(args.tokenizer_path),
        "out_jsonl": str(args.out_jsonl),
        "conversation_template": args.conversation_template,
        "append_instruction": args.append_instruction,
        "block_size": args.block_size,
        "small_block_size": args.small_block_size,
        "missing_cases": missing_cases,
        "totals": summarize(records),
    }

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
