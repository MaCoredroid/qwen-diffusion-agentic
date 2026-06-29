#!/usr/bin/env python3
import argparse
import copy
import json
import random
import re
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
DEFAULT_PLANNER_JSON = ROOT / "data/qwen35_9b_synthetic_multicall_planner_distill_curriculum/train_agentic_mix.json"
DEFAULT_PLANNER_AUDIT = ROOT / "data/qwen35_9b_synthetic_multicall_planner_distill_curriculum/train_agentic_mix.audit.jsonl"
DEFAULT_SELECTOR_JSONL = ROOT / "data/candidate_ranking/synthetic_multicall_failure_analogue_index_ranking.jsonl"
DEFAULT_OUT_DIR = ROOT / "data/qwen35_9b_synthetic_selector_replay_mix_leaveone_voice003_curriculum"
DEFAULT_HOLDOUT_ID = "synthetic_voice_command_camera_003"
INDEX_SYSTEM = "You select the correct candidate index for tool-call behavior preservation."


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def strip_source(instance):
    clone = copy.deepcopy(instance)
    clone.pop("source", None)
    clone.pop("mix_kind", None)
    return clone


def source_id(source):
    match = re.search(r":(synthetic_[a-z_]+_\d{3}):", str(source))
    return match.group(1) if match else None


def load_planner_instances(path, audit_path):
    payload = load_json(path)
    audit_rows = list(load_jsonl(audit_path))
    instances = payload.get("instances") or []
    if len(instances) != len(audit_rows):
        raise ValueError(f"planner instance/audit mismatch: {len(instances)} vs {len(audit_rows)}")
    rows = []
    for instance, audit in zip(instances, audit_rows):
        cloned = copy.deepcopy(instance)
        cloned["source"] = audit.get("source") or "unknown"
        cloned["mix_kind"] = "planner_replay"
        rows.append(cloned)
    return rows


def selector_instance(example, repeat_idx):
    family = example.get("analogue_family") or "unknown_family"
    kind = example.get("kind") or "unknown_kind"
    return {
        "source": f"selector_index:{family}:{kind}:{example.get('id')}:repeat{repeat_idx}",
        "mix_kind": "selector_index",
        "messages": [
            {"role": "system", "content": INDEX_SYSTEM},
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": str(example["target_index"])},
        ],
    }


def audit_instances(tokenizer, chat_template, instances, args):
    accepted = []
    rejected = []
    audit_rows = []
    for instance in instances:
        stats = token_stats(tokenizer, chat_template, instance, args.block_size, args.truncation_side)
        row = {
            "source": instance.get("source") or "unknown",
            "mix_kind": instance.get("mix_kind") or "unknown",
            **stats,
        }
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
    parser.add_argument("--planner-json", type=Path, default=DEFAULT_PLANNER_JSON)
    parser.add_argument("--planner-audit-jsonl", type=Path, default=DEFAULT_PLANNER_AUDIT)
    parser.add_argument("--selector-jsonl", type=Path, default=DEFAULT_SELECTOR_JSONL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--holdout-id", action="append", default=[DEFAULT_HOLDOUT_ID])
    parser.add_argument("--planner-repeat", type=int, default=4)
    parser.add_argument("--selector-repeat", type=int, default=2)
    parser.add_argument("--conversation-template", default="fast_dllm_v2")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--truncation-side", choices=["left", "right"], default="left")
    parser.add_argument("--min-labels", type=int, default=1)
    parser.add_argument("--require-full-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=829)
    args = parser.parse_args()

    holdout_ids = set(args.holdout_id or [])
    planner_rows = load_planner_instances(args.planner_json, args.planner_audit_jsonl)
    planner_train = [row for row in planner_rows if source_id(row.get("source")) not in holdout_ids]
    planner_holdout = [row for row in planner_rows if source_id(row.get("source")) in holdout_ids]
    selector_examples = [row for row in load_jsonl(args.selector_jsonl) if row.get("usable_for_training")]
    selector_train = [row for row in selector_examples if row.get("id") not in holdout_ids]
    selector_holdout = [row for row in selector_examples if row.get("id") in holdout_ids]

    if not planner_train or not selector_train:
        raise SystemExit("empty planner or selector train split")
    if not planner_holdout or not selector_holdout:
        raise SystemExit(f"holdout split did not match all sources for {sorted(holdout_ids)}")

    instances = []
    source_counts = Counter()
    train_ids = set()
    for repeat_idx in range(args.planner_repeat):
        for row in planner_train:
            cloned = copy.deepcopy(row)
            cloned["source"] = f"{row.get('source')}:mixrepeat{repeat_idx}"
            instances.append(cloned)
            source_counts["planner_replay"] += 1
            case_id = source_id(row.get("source"))
            if case_id:
                train_ids.add(case_id)
    for example in selector_train:
        train_ids.add(example.get("id"))
        for repeat_idx in range(args.selector_repeat):
            instances.append(selector_instance(example, repeat_idx))
            source_counts["selector_index"] += 1

    rng = random.Random(args.seed)
    rng.shuffle(instances)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    chat_template = resolve_chat_template(args.conversation_template)
    accepted, rejected, audit_rows = audit_instances(tokenizer, chat_template, instances, args)

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
    rejected_path = args.out_dir / "train_agentic_mix.rejected.jsonl"
    write_jsonl(audit_path, audit_rows)
    write_jsonl(rejected_path, rejected)

    manifest = {
        "train_path": str(train_path),
        "audit_path": str(audit_path),
        "rejected_path": str(rejected_path),
        "planner_json": str(args.planner_json),
        "planner_audit_jsonl": str(args.planner_audit_jsonl),
        "selector_jsonl": str(args.selector_jsonl),
        "raw_count": len(instances),
        "count": len(accepted),
        "rejected_count": len(rejected),
        "source_counts": dict(sorted(source_counts.items())),
        "holdout_ids": sorted(holdout_ids),
        "train_ids": sorted(train_ids),
        "planner_train_base_rows": len(planner_train),
        "planner_holdout_base_rows": len(planner_holdout),
        "selector_train_examples": len(selector_train),
        "selector_holdout_examples": len(selector_holdout),
        "planner_repeat": args.planner_repeat,
        "selector_repeat": args.selector_repeat,
        "contains_holdout_in_training": bool(train_ids & holdout_ids),
        "tokenizer_model": str(args.model),
        "conversation_template": args.conversation_template,
        "block_size": args.block_size,
        "truncation_side": args.truncation_side,
        "min_labels": args.min_labels,
        "require_full_labels": args.require_full_labels,
        "seed": args.seed,
        "promotion_allowed": False,
        "promotion_note": (
            "Diagnostic synthetic replay-plus-selector mix with the active heldout case removed. "
            "Use only for gate evidence before considering larger train-only mixes."
        ),
        "audit_summary": summarize_audit(audit_rows, audit_rows),
    }
    manifest_path = args.out_dir / "train_agentic_mix.manifest"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
