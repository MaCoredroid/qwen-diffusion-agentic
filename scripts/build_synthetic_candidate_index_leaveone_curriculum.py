#!/usr/bin/env python3
import argparse
import copy
import json
import random
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path("/home/mark/qwen_diffusion")
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_synthetic_multicall_candidate_index_examples import load_jsonl  # noqa: E402
from build_toolcall_labelaware_public_mix import (  # noqa: E402
    resolve_chat_template,
    summarize_audit,
    token_stats,
)


DEFAULT_MODEL = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_EXAMPLES = ROOT / "data/candidate_ranking/synthetic_multicall_failure_analogue_index_ranking.jsonl"
DEFAULT_OUT_DIR = ROOT / "data/qwen35_9b_synthetic_candidate_index_leaveone_voice003_curriculum"
DEFAULT_HOLDOUT_ID = "synthetic_voice_command_camera_003"
INDEX_SYSTEM = "You select the correct candidate index for tool-call behavior preservation."


def strip_source(instance):
    clone = copy.deepcopy(instance)
    clone.pop("source", None)
    return clone


def instance_from_example(example, source):
    return {
        "source": source,
        "messages": [
            {"role": "system", "content": INDEX_SYSTEM},
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": str(example["target_index"])},
        ],
    }


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def audit_instances(tokenizer, chat_template, instances, args):
    accepted = []
    rejected = []
    audit_rows = []
    for instance in instances:
        stats = token_stats(tokenizer, chat_template, instance, args.block_size, args.truncation_side)
        row = {"source": instance.get("source") or "unknown", **stats}
        audit_rows.append(row)
        if stats["kept_labels"] < args.min_labels:
            rejected.append({**row, "reason": "too_few_labels"})
            continue
        if args.require_full_labels and not stats["full_labels_kept"]:
            rejected.append({**row, "reason": "partial_labels"})
            continue
        accepted.append(instance)
    return accepted, rejected, audit_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--examples-jsonl", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--holdout-id", action="append", default=[DEFAULT_HOLDOUT_ID])
    parser.add_argument("--repeat", type=int, default=12)
    parser.add_argument("--conversation-template", default="fast_dllm_v2")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--truncation-side", choices=["left", "right"], default="left")
    parser.add_argument("--min-labels", type=int, default=1)
    parser.add_argument("--require-full-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=814)
    args = parser.parse_args()

    examples = [row for row in load_jsonl(args.examples_jsonl) if row.get("usable_for_training")]
    holdout_ids = set(args.holdout_id or [])
    train_examples = [row for row in examples if row.get("id") not in holdout_ids]
    holdout_examples = [row for row in examples if row.get("id") in holdout_ids]
    if not holdout_examples:
        raise SystemExit(f"no holdout examples matched {sorted(holdout_ids)}")
    if not train_examples:
        raise SystemExit("no training examples left after holdout split")

    raw_instances = []
    source_counts = Counter()
    for example in train_examples:
        family = example.get("analogue_family") or "unknown_family"
        kind = example.get("kind") or "unknown_kind"
        for repeat_idx in range(args.repeat):
            source = f"synthetic_index_leaveone:{family}:{kind}:{example.get('id')}:repeat{repeat_idx}"
            raw_instances.append(instance_from_example(example, source))
            source_counts[f"{family}:{kind}"] += 1

    rng = random.Random(args.seed)
    rng.shuffle(raw_instances)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    chat_template = resolve_chat_template(args.conversation_template)
    accepted, rejected, audit_rows = audit_instances(tokenizer, chat_template, raw_instances, args)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train_agentic_mix.json"
    train_path.write_text(
        json.dumps(
            {"type": "conversation", "instances": [strip_source(instance) for instance in accepted]},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    audit_path = args.out_dir / "train_agentic_mix.audit.jsonl"
    write_jsonl(audit_path, audit_rows)
    rejected_path = args.out_dir / "train_agentic_mix.rejected.jsonl"
    write_jsonl(rejected_path, rejected)
    train_examples_path = args.out_dir / "train_examples.jsonl"
    write_jsonl(train_examples_path, train_examples)
    holdout_path = args.out_dir / "holdout_examples.jsonl"
    write_jsonl(holdout_path, holdout_examples)

    manifest = {
        "train_path": str(train_path),
        "audit_path": str(audit_path),
        "rejected_path": str(rejected_path),
        "train_examples_path": str(train_examples_path),
        "holdout_examples_path": str(holdout_path),
        "examples_jsonl": str(args.examples_jsonl),
        "raw_count": len(raw_instances),
        "count": len(accepted),
        "rejected_count": len(rejected),
        "train_example_count": len(train_examples),
        "holdout_example_count": len(holdout_examples),
        "holdout_ids": sorted(holdout_ids),
        "train_ids": [row.get("id") for row in train_examples],
        "source_counts": dict(sorted(source_counts.items())),
        "tokenizer_model": str(args.model),
        "conversation_template": args.conversation_template,
        "block_size": args.block_size,
        "truncation_side": args.truncation_side,
        "min_labels": args.min_labels,
        "require_full_labels": args.require_full_labels,
        "repeat": args.repeat,
        "seed": args.seed,
        "contains_holdout_in_training": bool(set(row.get("id") for row in train_examples) & holdout_ids),
        "promotion_allowed": False,
        "promotion_note": (
            "Diagnostic leave-one-out selector curriculum over synthetic analogues. "
            "Use only to test whether selector pressure can generalize to the heldout synthetic miss."
        ),
        "audit_summary": summarize_audit(audit_rows, audit_rows),
    }
    manifest_path = args.out_dir / "train_agentic_mix.manifest"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
