#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path("/home/mark/qwen_diffusion")
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from audit_toolcall_eval_overlap import (  # noqa: E402
    assistant_text,
    eval_records,
    fingerprint,
    load_conversation_json,
    user_fingerprint,
    user_text,
)
from build_fastdllm_toolcall_data import make_eval_case  # noqa: E402
from eval_toolcall_jsonl import extract_tool_calls  # noqa: E402


DEFAULT_INPUT = ROOT / "data/toolcall_seed/qwen_toolcall_seed.jsonl"
DEFAULT_TRAIN_EXCLUDE = [ROOT / "data/fastdllm_toolcall_train/train_toolcall_no_public_multicall_smoke.json"]
DEFAULT_EVAL_EXCLUDE = [
    ROOT / "data/toolcall_eval/public_multicall_hermes_smoke.jsonl",
    ROOT / "data/toolcall_eval/synthetic_multicall_failure_analogues.jsonl",
]
DEFAULT_OUT = ROOT / "data/toolcall_eval/heldout_seed_multicall_2to3_clean.jsonl"


def load_seed_rows(path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def case_user_text(case):
    chunks = []
    for message in case.get("prompt_messages") or []:
        if message.get("role") == "user":
            content = str(message.get("content") or "").strip()
            if content:
                chunks.append(content)
    return "\n\n".join(chunks)


def excluded_fingerprints(train_paths, eval_paths):
    exact = set()
    users = set()
    train_count = 0
    eval_count = 0
    for path in train_paths:
        for instance in load_conversation_json(path):
            user = user_text(instance)
            assistant = assistant_text(instance)
            exact.add(fingerprint(user, assistant))
            users.add(user_fingerprint(user))
            train_count += 1
    for record in eval_records(eval_paths):
        exact.add(record["fingerprint"])
        users.add(record["user_fingerprint"])
        eval_count += 1
    return exact, users, train_count, eval_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--exclude-train-json", type=Path, nargs="*", default=DEFAULT_TRAIN_EXCLUDE)
    parser.add_argument("--exclude-eval-jsonl", type=Path, nargs="*", default=DEFAULT_EVAL_EXCLUDE)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-gold-calls", type=int, default=2)
    parser.add_argument("--max-gold-calls", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--id-prefix", default="heldout_seed_multicall")
    args = parser.parse_args()

    exact_excluded, user_excluded, train_excluded, eval_excluded = excluded_fingerprints(
        args.exclude_train_json,
        args.exclude_eval_jsonl,
    )

    rows = []
    skipped = Counter()
    call_counts = Counter()
    source_counts = Counter()
    for idx, record in enumerate(load_seed_rows(args.input_jsonl)):
        case = make_eval_case(record)
        if case is None:
            skipped["no_eval_case"] += 1
            continue
        calls, invalid = extract_tool_calls(case.get("gold_assistant") or "")
        if invalid:
            skipped["invalid_gold_tool_json"] += 1
            continue
        if len(calls) < args.min_gold_calls or len(calls) > args.max_gold_calls:
            skipped["call_count_outside_range"] += 1
            continue
        user = case_user_text(case)
        exact_fp = fingerprint(user, case.get("gold_assistant") or "")
        user_fp = user_fingerprint(user)
        if exact_fp in exact_excluded:
            skipped["excluded_exact_overlap"] += 1
            continue
        if user_fp in user_excluded:
            skipped["excluded_user_overlap"] += 1
            continue
        row = {
            **case,
            "id": f"{args.id_prefix}_{len(rows):04d}",
            "seed_id": case.get("id"),
            "seed_source": case.get("source"),
            "source": "heldout_seed_multicall_clean",
            "gold_tool_calls": calls,
            "gold_invalid_tool_json_count": invalid,
            "seed_index": idx,
        }
        rows.append(row)
        call_counts[str(len(calls))] += 1
        source_counts[str(case.get("source") or "unknown")] += 1
        if args.limit and len(rows) >= args.limit:
            break

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_jsonl": str(args.input_jsonl),
        "out_jsonl": str(args.out_jsonl),
        "exclude_train_json": [str(path) for path in args.exclude_train_json],
        "exclude_eval_jsonl": [str(path) for path in args.exclude_eval_jsonl],
        "excluded_train_records": train_excluded,
        "excluded_eval_records": eval_excluded,
        "records": len(rows),
        "min_gold_calls": args.min_gold_calls,
        "max_gold_calls": args.max_gold_calls,
        "limit": args.limit,
        "skipped": dict(sorted(skipped.items())),
        "tool_call_count_histogram": dict(sorted(call_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "no_exact_or_user_overlap": True,
        "ids": [row["id"] for row in rows],
        "seed_ids": [row.get("seed_id") for row in rows],
    }
    args.out_jsonl.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
