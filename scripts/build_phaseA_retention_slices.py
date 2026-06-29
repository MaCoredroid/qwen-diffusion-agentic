#!/usr/bin/env python3
"""Build deterministic small slices for Phase A behavior-retention evals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_OUT_DIR = ROOT / "data/phaseA_retention"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_mbpp(out_dir: Path, limit: int) -> tuple[Path, list[dict], dict]:
    ds = load_dataset("google-research-datasets/mbpp", "full", split=f"test[:{limit}]")
    rows = []
    for idx, row in enumerate(ds):
        rows.append(
            {
                "idx": idx,
                "source": "google-research-datasets/mbpp:full:test",
                "task": "mbpp",
                "task_id": row["task_id"],
                "text": row["text"],
                "code": row["code"],
                "test_list": row["test_list"],
                "test_setup_code": row["test_setup_code"],
                "challenge_test_list": row["challenge_test_list"],
            }
        )
    return (
        out_dir / f"mbpp_full_test_first{limit}.jsonl",
        rows,
        {
            "task": "mbpp",
            "dataset": "google-research-datasets/mbpp",
            "config": "full",
            "split": "test",
            "selection": f"first {limit} rows by datasets split order",
            "ids": [row["task_id"] for row in rows],
        },
    )


def build_ifeval(out_dir: Path, limit: int) -> tuple[Path, list[dict], dict]:
    ds = load_dataset("google/IFEval", split=f"train[:{limit}]")
    rows = []
    for idx, row in enumerate(ds):
        rows.append(
            {
                "idx": idx,
                "source": "google/IFEval:train",
                "task": "ifeval",
                "key": row["key"],
                "prompt": row["prompt"],
                "instruction_id_list": row["instruction_id_list"],
                "kwargs": row["kwargs"],
            }
        )
    return (
        out_dir / f"ifeval_train_first{limit}.jsonl",
        rows,
        {
            "task": "ifeval",
            "dataset": "google/IFEval",
            "config": None,
            "split": "train",
            "selection": f"first {limit} rows by datasets split order",
            "ids": [row["key"] for row in rows],
        },
    )


def build_gsm8k_split(out_dir: Path, split: str, limit: int) -> tuple[Path, list[dict], dict]:
    ds = load_dataset("openai/gsm8k", "main", split=f"{split}[:{limit}]")
    rows = []
    for idx, row in enumerate(ds):
        rows.append(
            {
                "idx": idx,
                "source": f"openai/gsm8k:main:{split}",
                "task": "gsm8k",
                "question": row["question"],
                "answer": row["answer"],
            }
        )
    return (
        out_dir / f"gsm8k_main_{split}_first{limit}.jsonl",
        rows,
        {
            "task": "gsm8k",
            "dataset": "openai/gsm8k",
            "config": "main",
            "split": split,
            "selection": f"first {limit} rows by datasets split order",
            "ids": list(range(limit)),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--mbpp-limit", type=int, default=20)
    parser.add_argument("--ifeval-limit", type=int, default=20)
    parser.add_argument("--gsm8k-limit", type=int, default=20)
    parser.add_argument("--gsm8k-fewshot-limit", type=int, default=5)
    args = parser.parse_args()

    specs = [
        build_mbpp(args.out_dir, args.mbpp_limit),
        build_ifeval(args.out_dir, args.ifeval_limit),
        build_gsm8k_split(args.out_dir, "test", args.gsm8k_limit),
        build_gsm8k_split(args.out_dir, "train", args.gsm8k_fewshot_limit),
    ]
    manifest = {"slices": []}
    for path, rows, meta in specs:
        write_jsonl(path, rows)
        meta = dict(meta)
        meta["path"] = str(path)
        meta["num_rows"] = len(rows)
        manifest["slices"].append(meta)

    manifest_path = args.out_dir / "phaseA_retention_slices.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
