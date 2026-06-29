#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
from pathlib import Path


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def load_instances(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    instances = payload.get("instances")
    if payload.get("type") != "conversation" or not isinstance(instances, list):
        raise ValueError(f"{path} must be a conversation dataset with instances")
    return instances


def last_assistant_index(messages):
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "assistant":
            return idx
    return None


def materialize_instance(instance, idx, args):
    messages = instance.get("messages") or []
    assistant_idx = last_assistant_index(messages)
    if assistant_idx is None:
        return None, "missing_assistant"
    gold = str(messages[assistant_idx].get("content") or "")
    tool_call_count = len(TOOL_CALL_RE.findall(gold))
    if tool_call_count < args.min_tool_calls:
        return None, "too_few_tool_calls"
    prompt_messages = []
    system = instance.get("system")
    if system:
        prompt_messages.append({"role": "system", "content": str(system)})
    for message in messages[:assistant_idx]:
        role = message.get("role")
        content = message.get("content")
        if role and content is not None:
            prompt_messages.append({"role": role, "content": str(content)})
    if not prompt_messages:
        return None, "missing_prompt"
    source = instance.get("source") or args.source
    return (
        {
            "id": f"{args.id_prefix}_{idx:04d}",
            "source": source,
            "prompt_messages": prompt_messages,
            "tools": instance.get("tools") or [],
            "gold_assistant": gold,
            "tool_call_count": tool_call_count,
            "train_source_path": str(args.input_json),
            "train_source_index": idx,
        },
        None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--min-tool-calls", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--id-prefix", default="public_train_toolcall")
    parser.add_argument("--source", default="public_train_toolcall")
    args = parser.parse_args()

    rows = []
    skipped = Counter()
    tool_counts = Counter()
    for idx, instance in enumerate(load_instances(args.input_json)):
        row, reason = materialize_instance(instance, idx, args)
        if reason:
            skipped[reason] += 1
            continue
        rows.append(row)
        tool_counts[str(row["tool_call_count"])] += 1
        if args.limit and len(rows) >= args.limit:
            break

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "input_json": str(args.input_json),
        "out_jsonl": str(args.out_jsonl),
        "records": len(rows),
        "min_tool_calls": args.min_tool_calls,
        "limit": args.limit,
        "skipped": dict(sorted(skipped.items())),
        "tool_call_count_histogram": dict(sorted(tool_counts.items())),
        "no_eval_leakage": True,
    }
    args.out_jsonl.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
