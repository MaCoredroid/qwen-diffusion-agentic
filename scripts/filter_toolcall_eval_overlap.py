#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from audit_toolcall_eval_overlap import (
    assistant_text,
    eval_records,
    fingerprint,
    load_conversation_json,
    user_text,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-conversation-json", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, default=None)
    args = parser.parse_args()

    eval_fingerprints = {row["fingerprint"] for row in eval_records(args.eval_jsonl)}
    instances = load_conversation_json(args.train_conversation_json)
    kept = []
    removed = []
    for idx, instance in enumerate(instances):
        row_fp = fingerprint(user_text(instance), assistant_text(instance))
        if row_fp in eval_fingerprints:
            removed.append(
                {
                    "idx": idx,
                    "id": instance.get("id") or instance.get("case_id") or str(idx),
                    "source": instance.get("source"),
                    "fingerprint": row_fp,
                    "user_excerpt": " ".join(user_text(instance).split())[:220],
                }
            )
        else:
            kept.append(instance)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps({"type": "conversation", "instances": kept}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "input": str(args.train_conversation_json),
        "eval_jsonl": [str(path) for path in args.eval_jsonl],
        "output": str(args.out_json),
        "input_count": len(instances),
        "kept_count": len(kept),
        "removed_count": len(removed),
        "removed": removed,
    }
    manifest_path = args.out_manifest or args.out_json.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "removed"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
