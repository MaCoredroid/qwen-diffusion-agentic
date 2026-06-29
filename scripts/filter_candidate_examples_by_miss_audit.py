#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path("/home/mark/qwen_diffusion")
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_candidate_ranking_examples import conversation_instance, load_jsonl  # noqa: E402
from diagnose_schedule_value_candidates import normalize_for_compare  # noqa: E402


def mismatch_keys(audit_rows):
    keys = []
    for row in audit_rows:
        for mismatch in row.get("mismatches") or []:
            if mismatch.get("kind") != "argument_value":
                continue
            keys.append(
                {
                    "id": row.get("id"),
                    "kind": "argument_value",
                    "tool_call_index": mismatch.get("tool_call_index"),
                    "json_key": mismatch.get("json_key"),
                    "target": mismatch.get("gold"),
                    "path": mismatch.get("path"),
                    "generated": mismatch.get("generated"),
                }
            )
    return keys


def same_key(example, target):
    return (
        example.get("id") == target.get("id")
        and example.get("kind") == target.get("kind")
        and int(example.get("tool_call_index", -1)) == int(target.get("tool_call_index", -2))
        and example.get("json_key") == target.get("json_key")
        and normalize_for_compare(example.get("target")) == normalize_for_compare(target.get("target"))
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples-jsonl", type=Path, required=True)
    parser.add_argument("--audit-jsonl", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-train-json", type=Path, default=None)
    args = parser.parse_args()

    targets = mismatch_keys(load_jsonl(args.audit_jsonl))
    examples = list(load_jsonl(args.examples_jsonl))
    selected = []
    totals = Counter()
    for target in targets:
        matches = [example for example in examples if same_key(example, target)]
        totals["targets"] += 1
        totals["targets_with_match"] += int(bool(matches))
        for example in matches:
            row = dict(example)
            row["miss_path"] = target.get("path")
            row["miss_generated"] = target.get("generated")
            selected.append(row)
            totals["examples"] += 1
            totals[f"examples:{row.get('json_key')}"] += 1
            totals["usable_for_training"] += int(bool(row.get("usable_for_training")))

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    train_path = args.out_train_json or args.out_jsonl.with_suffix(".train.json")
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_instances = [
        conversation_instance(example)
        for example in selected
        if example.get("usable_for_training")
    ]
    train_path.write_text(
        json.dumps({"type": "conversation", "instances": train_instances}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = {
        "examples_jsonl": str(args.examples_jsonl),
        "audit_jsonl": str(args.audit_jsonl),
        "out_jsonl": str(args.out_jsonl),
        "out_train_json": str(train_path),
        "totals": dict(totals),
    }
    args.out_jsonl.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
