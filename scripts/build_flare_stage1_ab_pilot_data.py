#!/usr/bin/env python3
"""Build the frozen real-text slice for the FLARE Stage-1 A/B pilot."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def gsm8k_instance(row):
    return {
        "system": "You are a helpful assistant. Solve math problems clearly.",
        "messages": [
            {
                "role": "user",
                "content": "Solve the grade-school math problem. Show the reasoning and final answer.\n\n"
                + row["question"],
            },
            {"role": "assistant", "content": row["answer"]},
        ],
    }


def mbpp_instance(row):
    tests = "\n".join(row.get("test_list") or [])
    return {
        "system": "You are a helpful coding assistant.",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a Python function for this task. Return code only.\n\n"
                    f"Task: {row['text']}\n\nTests:\n{tests}"
                ),
            },
            {"role": "assistant", "content": row["code"]},
        ],
    }


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/flare_stage1_ab_pilot")
    parser.add_argument("--gsm8k-train", type=int, default=160)
    parser.add_argument("--mbpp-train", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260701)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    rng = random.Random(args.seed)

    gsm_train = list(load_dataset("openai/gsm8k", "main", split="train"))
    mbpp_train = list(load_dataset("google-research-datasets/mbpp", "full", split="train"))
    rng.shuffle(gsm_train)
    rng.shuffle(mbpp_train)

    train_rows = []
    for row in gsm_train[: args.gsm8k_train]:
        train_rows.append(gsm8k_instance(row) | {"source": "openai/gsm8k:main:train"})
    for row in mbpp_train[: args.mbpp_train]:
        train_rows.append(mbpp_instance(row) | {"source": "google-research-datasets/mbpp:full:train"})
    rng.shuffle(train_rows)

    train_payload = {"type": "conversation", "instances": train_rows}
    write_json(out_dir / "train_agentic_mix.json", train_payload)

    heldout_rows = []
    gsm_test = Path("data/phaseA_retention/gsm8k_main_test_first20.jsonl")
    with gsm_test.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            heldout_rows.append(gsm8k_instance(row) | {"id": f"gsm8k-{row['idx']}", "task": "gsm8k"})

    mbpp_test = Path("data/phaseA_retention/mbpp_full_test_first20.jsonl")
    with mbpp_test.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            heldout_rows.append(mbpp_instance(row) | {"id": f"mbpp-{row['task_id']}", "task": "mbpp"})
    write_jsonl(out_dir / "heldout_nll.jsonl", heldout_rows)

    manifest = {
        "train_path": str((out_dir / "train_agentic_mix.json").resolve()),
        "heldout_path": str((out_dir / "heldout_nll.jsonl").resolve()),
        "seed": args.seed,
        "train_count": len(train_rows),
        "train_sources": {
            "openai/gsm8k:main:train": args.gsm8k_train,
            "google-research-datasets/mbpp:full:train": args.mbpp_train,
        },
        "heldout_count": len(heldout_rows),
        "heldout_sources": {
            "openai/gsm8k:main:test:first20": 20,
            "google-research-datasets/mbpp:full:test:first20": 20,
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
