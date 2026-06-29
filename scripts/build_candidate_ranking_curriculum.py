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

from build_candidate_ranking_examples import load_jsonl  # noqa: E402
from build_toolcall_labelaware_public_mix import (  # noqa: E402
    resolve_chat_template,
    summarize_audit,
    token_stats,
)


DEFAULT_MODEL = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_EXAMPLES = ROOT / "data/candidate_ranking/public_multicall_toolname_argument_ranking_v3_12.jsonl"
DEFAULT_DELTA = ROOT / "qwen35_candidate_ranking_delta_result.json"
DEFAULT_OUT_DIR = ROOT / "data/qwen35_9b_candidate_ranker_public12_diagnostic_curriculum"
INDEX_SYSTEM = "You select the correct candidate index for tool-call behavior preservation."
VALUE_SYSTEM = "You emit the exact JSON value span for tool-call behavior preservation."


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def example_key(row):
    return (
        row.get("id"),
        row.get("kind"),
        row.get("tool_call_index"),
        row.get("json_key"),
        json.dumps(row.get("target"), ensure_ascii=False),
    )


def delta_keys(delta, field):
    return {example_key(row) for row in delta.get(field) or []}


def value_span_prompt(example):
    candidate_lines = "\n".join(
        f"{idx}: {json.dumps(value, ensure_ascii=False)}"
        for idx, value in enumerate(example.get("candidate_values") or [])
    )
    parts = [
        "Choose the correct candidate for a Qwen tool-call trace.",
        "Return only the exact JSON value span for that candidate.",
        "Use JSON quotes for strings. Use exact numeric, boolean, or null spelling for non-strings.",
        "Do not return the candidate index and do not add prose.",
        "",
        "Context and tool information:",
        example.get("prompt") or "",
        "",
        "Candidates:",
        candidate_lines,
        "",
        "Exact JSON value span:",
    ]
    return "\n".join(parts).strip()


def answer_for_example(example, answer_mode):
    if answer_mode == "index":
        return str(example["target_index"])
    if answer_mode == "target_text":
        target_text = example.get("target_text")
        if target_text is not None:
            return str(target_text)
        return json.dumps(example.get("target"), ensure_ascii=False)
    raise ValueError(f"unsupported answer mode {answer_mode!r}")


def prompt_for_example(example, answer_mode):
    if answer_mode == "index":
        return example["prompt"]
    if answer_mode == "target_text":
        return value_span_prompt(example)
    raise ValueError(f"unsupported answer mode {answer_mode!r}")


def system_for_mode(answer_mode):
    if answer_mode == "index":
        return INDEX_SYSTEM
    if answer_mode == "target_text":
        return VALUE_SYSTEM
    raise ValueError(f"unsupported answer mode {answer_mode!r}")


def instance_from_example(example, source, args):
    return {
        "system": system_for_mode(args.answer_mode),
        "messages": [
            {"role": "user", "content": prompt_for_example(example, args.answer_mode)},
            {"role": "assistant", "content": answer_for_example(example, args.answer_mode)},
        ],
        "source": source,
    }


def strip_source(instance):
    clone = copy.deepcopy(instance)
    clone.pop("source", None)
    return clone


def repeat_count(example, hard_keys, regressed_keys, improved_keys, args):
    key = example_key(example)
    if key in hard_keys:
        return args.remaining_repeat
    if key in regressed_keys:
        return args.regressed_repeat
    if key in improved_keys:
        return args.improved_repeat
    if example.get("kind") == "tool_name":
        return args.tool_name_repeat
    if int(example.get("candidate_count") or 0) <= 1:
        return args.single_candidate_repeat
    return args.default_repeat


def build_instances(examples, delta, args):
    hard_keys = delta_keys(delta, "remaining_failures")
    regressed_keys = delta_keys(delta, "regressed")
    improved_keys = delta_keys(delta, "improved")
    instances = []
    skipped = Counter()
    source_counts = Counter()
    for example in examples:
        if not example.get("usable_for_training"):
            skipped["not_usable"] += 1
            continue
        if example.get("kind") not in set(args.include_kinds):
            skipped["unsupported_kind"] += 1
            continue
        repeats = repeat_count(example, hard_keys, regressed_keys, improved_keys, args)
        if repeats <= 0:
            skipped["repeat_zero"] += 1
            continue
        family = "candidate_ranker"
        key = example_key(example)
        if key in hard_keys:
            family = "candidate_ranker_remaining_failure"
        elif key in regressed_keys:
            family = "candidate_ranker_regressed"
        elif key in improved_keys:
            family = "candidate_ranker_improved"
        elif example.get("kind") == "tool_name":
            family = "candidate_ranker_tool_name"
        elif int(example.get("candidate_count") or 0) <= 1:
            family = "candidate_ranker_single_candidate"
        for repeat_idx in range(repeats):
            source = (
                f"{family}:{example.get('id')}:call{example.get('tool_call_index')}:"
                f"{example.get('json_key') or example.get('kind')}:repeat{repeat_idx}"
            )
            instances.append(instance_from_example(example, source, args))
            source_counts[family] += 1
    return instances, skipped, source_counts


def audit_instances(tokenizer, chat_template, instances, args):
    accepted = []
    rejected = []
    audit_rows = []
    for instance in instances:
        stats = token_stats(tokenizer, chat_template, instance, args.block_size, args.truncation_side)
        row = {
            "source": instance.get("source") or "unknown",
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


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--examples-jsonl", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--delta-json", type=Path, default=DEFAULT_DELTA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--conversation-template", default="fast_dllm_v2")
    parser.add_argument(
        "--answer-mode",
        choices=["index", "target_text"],
        default="index",
        help="Train the assistant to emit either the candidate index or the exact JSON target span.",
    )
    parser.add_argument(
        "--include-kinds",
        nargs="+",
        default=["tool_name", "argument_value"],
        choices=["tool_name", "argument_value"],
        help="Candidate example kinds to include.",
    )
    parser.add_argument("--default-repeat", type=int, default=1)
    parser.add_argument("--tool-name-repeat", type=int, default=1)
    parser.add_argument("--single-candidate-repeat", type=int, default=1)
    parser.add_argument("--improved-repeat", type=int, default=2)
    parser.add_argument("--regressed-repeat", type=int, default=6)
    parser.add_argument("--remaining-repeat", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--truncation-side", choices=["left", "right"], default="left")
    parser.add_argument("--min-labels", type=int, default=1)
    parser.add_argument("--require-full-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--contains-eval-slice", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diagnostic-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=811)
    args = parser.parse_args()

    examples = list(load_jsonl(args.examples_jsonl))
    delta = load_json(args.delta_json) if args.delta_json.exists() else {}
    raw_instances, skipped, source_counts = build_instances(examples, delta, args)
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

    manifest = {
        "train_path": str(train_path),
        "audit_path": str(audit_path),
        "rejected_path": str(rejected_path),
        "count": len(accepted),
        "raw_count": len(raw_instances),
        "rejected_count": len(rejected),
        "examples_jsonl": str(args.examples_jsonl),
        "delta_json": str(args.delta_json),
        "source_counts": dict(sorted(source_counts.items())),
        "skipped_counts": dict(sorted(skipped.items())),
        "tokenizer_model": str(args.model),
        "conversation_template": args.conversation_template,
        "answer_mode": args.answer_mode,
        "include_kinds": list(args.include_kinds),
        "block_size": args.block_size,
        "truncation_side": args.truncation_side,
        "min_labels": args.min_labels,
        "require_full_labels": args.require_full_labels,
        "repeats": {
            "default": args.default_repeat,
            "tool_name": args.tool_name_repeat,
            "single_candidate": args.single_candidate_repeat,
            "improved": args.improved_repeat,
            "regressed": args.regressed_repeat,
            "remaining": args.remaining_repeat,
        },
        "diagnostic_only": bool(args.diagnostic_only),
        "contains_eval_slice": bool(args.contains_eval_slice),
        "promotion_allowed": not bool(args.diagnostic_only) and not bool(args.contains_eval_slice),
        "promotion_note": (
            "Use checkpoints trained on this corpus only if separate heldout gates improve."
            if not args.contains_eval_slice
            else "Built from an eval slice. Use only for objective/debug gates; do not promote checkpoints trained on this corpus."
        ),
        "chosen_audit_summary": summarize_audit(audit_rows, audit_rows),
        "seed": args.seed,
    }
    manifest_path = args.out_dir / "train_agentic_mix.manifest"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
