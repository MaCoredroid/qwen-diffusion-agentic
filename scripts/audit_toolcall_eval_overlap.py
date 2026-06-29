#!/usr/bin/env python3
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_conversation_json(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    instances = payload.get("instances")
    if not isinstance(instances, list):
        raise ValueError(f"{path} does not contain an instances list")
    return instances


def normalize_text(text):
    return "\n".join(str(text or "").strip().split())


def messages_for_row(row):
    messages = row.get("prompt_messages")
    if isinstance(messages, list):
        return messages
    messages = row.get("messages")
    if isinstance(messages, list):
        return messages
    return []


def user_text(row):
    chunks = []
    for message in messages_for_row(row):
        if message.get("role") == "user":
            content = str(message.get("content") or "").strip()
            if content:
                chunks.append(content)
    return "\n\n".join(chunks)


def assistant_text(row):
    if row.get("gold_assistant") is not None:
        return str(row.get("gold_assistant") or "")
    chunks = []
    for message in messages_for_row(row):
        if message.get("role") == "assistant":
            content = str(message.get("content") or "").strip()
            if content:
                chunks.append(content)
    return "\n".join(chunks)


def fingerprint(user, assistant):
    payload = normalize_text(user) + "\n---assistant---\n" + normalize_text(assistant)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def user_fingerprint(user):
    return hashlib.sha256(normalize_text(user).encode("utf-8")).hexdigest()


def eval_records(paths):
    records = []
    for path in paths:
        for idx, row in enumerate(load_jsonl(path)):
            user = user_text(row)
            assistant = assistant_text(row)
            records.append(
                {
                    "path": str(path),
                    "idx": idx,
                    "id": row.get("id") or row.get("case_id") or str(idx),
                    "source": row.get("source"),
                    "user_text": user,
                    "assistant_text": assistant,
                    "fingerprint": fingerprint(user, assistant),
                    "user_fingerprint": user_fingerprint(user),
                }
            )
    return records


def train_records(paths):
    records = []
    for path in paths:
        for idx, row in enumerate(load_conversation_json(path)):
            user = user_text(row)
            assistant = assistant_text(row)
            records.append(
                {
                    "path": str(path),
                    "idx": idx,
                    "id": row.get("id") or row.get("case_id") or str(idx),
                    "source": row.get("source"),
                    "user_text": user,
                    "assistant_text": assistant,
                    "fingerprint": fingerprint(user, assistant),
                    "user_fingerprint": user_fingerprint(user),
                }
            )
    return records


def short(text, limit=220):
    text = " ".join(str(text or "").split())
    return text[:limit] + ("..." if len(text) > limit else "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-conversation-json", type=Path, nargs="+", required=True)
    parser.add_argument("--eval-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-overlaps-jsonl", type=Path, default=None)
    args = parser.parse_args()

    evals = eval_records(args.eval_jsonl)
    trains = train_records(args.train_conversation_json)
    eval_by_exact = {row["fingerprint"]: row for row in evals}
    evals_by_user = {}
    for row in evals:
        evals_by_user.setdefault(row["user_fingerprint"], []).append(row)

    exact_overlaps = []
    user_overlaps = []
    for train in trains:
        exact = eval_by_exact.get(train["fingerprint"])
        if exact:
            exact_overlaps.append(
                {
                    "train_path": train["path"],
                    "train_idx": train["idx"],
                    "train_id": train["id"],
                    "train_source": train["source"],
                    "eval_path": exact["path"],
                    "eval_idx": exact["idx"],
                    "eval_id": exact["id"],
                    "eval_source": exact["source"],
                    "fingerprint": train["fingerprint"],
                    "user_excerpt": short(train["user_text"]),
                }
            )
        for exact_user in evals_by_user.get(train["user_fingerprint"], []):
            user_overlaps.append(
                {
                    "train_path": train["path"],
                    "train_idx": train["idx"],
                    "train_id": train["id"],
                    "train_source": train["source"],
                    "eval_path": exact_user["path"],
                    "eval_idx": exact_user["idx"],
                    "eval_id": exact_user["id"],
                    "eval_source": exact_user["source"],
                    "user_fingerprint": train["user_fingerprint"],
                    "exact_assistant_match": train["fingerprint"] == exact_user["fingerprint"],
                    "user_excerpt": short(train["user_text"]),
                }
            )

    exact_counts = Counter(item["eval_path"] for item in exact_overlaps)
    user_counts = Counter(item["eval_path"] for item in user_overlaps)
    report = {
        "train_conversation_json": [str(path) for path in args.train_conversation_json],
        "eval_jsonl": [str(path) for path in args.eval_jsonl],
        "train_records": len(trains),
        "eval_records": len(evals),
        "exact_overlap_count": len(exact_overlaps),
        "user_overlap_count": len(user_overlaps),
        "exact_overlap_eval_path_counts": dict(sorted(exact_counts.items())),
        "user_overlap_eval_path_counts": dict(sorted(user_counts.items())),
        "exact_overlaps": exact_overlaps,
        "user_overlaps": user_overlaps,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    overlaps_path = args.out_overlaps_jsonl or args.out_json.with_suffix(".overlaps.jsonl")
    with overlaps_path.open("w", encoding="utf-8") as handle:
        for item in exact_overlaps:
            handle.write(json.dumps({"kind": "exact", **item}, ensure_ascii=False) + "\n")
        for item in user_overlaps:
            if item["exact_assistant_match"]:
                continue
            handle.write(json.dumps({"kind": "user_only", **item}, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"exact_overlaps", "user_overlaps"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
