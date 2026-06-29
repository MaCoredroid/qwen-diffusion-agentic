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
DEFAULT_EXAMPLES = ROOT / "data/candidate_ranking/public_multicall_targetcandidate_v5_miss_targets.jsonl"
DEFAULT_OUT_DIR = ROOT / "data/qwen35_9b_public_multicall_v5_focused_miss_pairwise_diag_curriculum"
PAIRWISE_SYSTEM = "You compare two candidate values for tool-call behavior preservation."


def context_from_ranking_prompt(prompt):
    marker = "\n\nSpan kind:"
    before_span = prompt.split(marker, 1)[0]
    context_marker = "User/tool context:\n"
    if context_marker in before_span:
        return before_span.split(context_marker, 1)[1].strip()
    return before_span.strip()


def anchor_text(value):
    if value is None:
        return ""
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if len(text) < 3:
        return ""
    return text


def path_terms(example):
    raw = " ".join(
        str(item or "")
        for item in [
            example.get("json_key"),
            example.get("json_path") or example.get("argument_path") or example.get("miss_path"),
        ]
    )
    terms = []
    cleaned = raw.replace("_", " ").replace("[", " ").replace("]", " ").replace(".", " ")
    chunks = [chunk.strip() for chunk in cleaned.split() if len(chunk.strip()) >= 4]
    if cleaned.strip():
        terms.append(cleaned.strip())
    terms.extend(chunks)
    return terms


def snippet_around(context, anchor, window=170):
    if not anchor:
        return ""
    lower_context = context.lower()
    lower_anchor = anchor.lower()
    idx = lower_context.find(lower_anchor)
    if idx < 0:
        return ""
    start = max(0, idx - window)
    end = min(len(context), idx + len(anchor) + window)
    snippet = context[start:end].replace("\n", " ")
    return " ".join(snippet.split())


def relevant_request_snippets(context, example, limit=6):
    anchors = []
    for argument in example.get("local_peer_arguments") or []:
        anchor = anchor_text(argument.get("target"))
        if anchor:
            anchors.append(anchor)
    for argument in example.get("same_call_peer_arguments") or []:
        anchor = anchor_text(argument.get("target"))
        if anchor:
            anchors.append(anchor)
    anchors.extend(path_terms(example))

    snippets = []
    seen = set()
    for anchor in anchors:
        snippet = snippet_around(context, anchor)
        if not snippet:
            continue
        key = snippet.lower()
        if key in seen:
            continue
        seen.add(key)
        snippets.append(snippet)
        if len(snippets) >= limit:
            break
    return snippets


def pairwise_prompt(example, value_a, value_b):
    json_path = example.get("miss_path") or example.get("json_path") or example.get("argument_path")
    context = context_from_ranking_prompt(example.get("prompt") or "")
    parts = [
        "Choose the correct option for a Qwen tool-call trace.",
        "Use the user request, available tools, call index, and argument key.",
        "Resolve derived values when needed: ranges may map to a midpoint scalar, equal rounded splits should preserve totals, and policy enums require applying the stated condition.",
        "Return only A or B.",
        "",
        "User/tool context:",
        context,
        "",
        f"Span kind: {example['kind']}",
        f"Tool call index: {example['tool_call_index']}",
    ]
    if example.get("json_key") is not None:
        parts.append(f"JSON key: {example['json_key']}")
    if json_path:
        parts.append(f"JSON path: {json_path}")
    if example.get("kind") == "tool_name" and example.get("same_call_arguments"):
        parts.append("Same-call argument sketch:")
        for argument in example["same_call_arguments"]:
            path = argument.get("json_path") or argument.get("argument_path") or argument.get("json_key")
            value = json.dumps(argument.get("target"), ensure_ascii=False)
            parts.append(f"- {path}: {value}")
    if example.get("kind") == "argument_value":
        if example.get("local_peer_arguments"):
            parts.append("Local peer argument sketch:")
            for argument in example["local_peer_arguments"]:
                path = argument.get("json_path") or argument.get("argument_path") or argument.get("json_key")
                value = json.dumps(argument.get("target"), ensure_ascii=False)
                parts.append(f"- {path}: {value}")
        elif example.get("same_call_peer_arguments"):
            parts.append("Same-call peer argument sketch:")
            for argument in example["same_call_peer_arguments"][:12]:
                path = argument.get("json_path") or argument.get("argument_path") or argument.get("json_key")
                value = json.dumps(argument.get("target"), ensure_ascii=False)
                parts.append(f"- {path}: {value}")
    snippets = relevant_request_snippets(context, example)
    if snippets:
        parts.append("Relevant request snippets:")
        for snippet in snippets:
            parts.append(f"- {snippet}")
    parts.extend(
        [
            "Options:",
            f"A: {json.dumps(value_a, ensure_ascii=False)}",
            f"B: {json.dumps(value_b, ensure_ascii=False)}",
        ]
    )
    return "\n".join(parts).strip()


def pair_key(example):
    return (
        example.get("id"),
        example.get("kind"),
        example.get("tool_call_index"),
        example.get("json_key"),
        json.dumps(example.get("target"), ensure_ascii=False),
    )


def build_pairwise_examples(examples, args):
    rows = []
    skipped = Counter()
    only_ids = set(args.only_ids or [])
    only_json_paths = set(args.only_json_paths or [])
    only_json_keys = set(args.only_json_keys or [])
    for example in examples:
        json_path = example.get("json_path") or example.get("argument_path")
        if only_ids and example.get("id") not in only_ids:
            skipped["filtered_id"] += 1
            continue
        if only_json_paths and json_path not in only_json_paths:
            skipped["filtered_json_path"] += 1
            continue
        if only_json_keys and example.get("json_key") not in only_json_keys:
            skipped["filtered_json_key"] += 1
            continue
        if not example.get("usable_for_training"):
            skipped["not_usable"] += 1
            continue
        if example.get("kind") not in set(args.include_kinds):
            skipped["unsupported_kind"] += 1
            continue
        candidates = example.get("candidate_values") or []
        target_idx = int(example.get("target_index", -1))
        if target_idx < 0 or target_idx >= len(candidates):
            skipped["target_missing"] += 1
            continue
        target = candidates[target_idx]
        for candidate_idx, candidate in enumerate(candidates):
            if candidate_idx == target_idx:
                continue
            orders = [(target, candidate, "A", target_idx, candidate_idx)]
            if args.both_orders:
                orders.append((candidate, target, "B", candidate_idx, target_idx))
            for order_idx, (value_a, value_b, answer, index_a, index_b) in enumerate(orders):
                for repeat_idx in range(args.repeat):
                    row = {
                        "id": example.get("id"),
                        "source": example.get("source"),
                        "kind": example.get("kind"),
                        "tool_call_index": example.get("tool_call_index"),
                        "json_key": example.get("json_key"),
                        "target": example.get("target"),
                        "target_text": example.get("target_text"),
                        "target_index": target_idx,
                        "miss_path": example.get("miss_path"),
                        "miss_generated": example.get("miss_generated"),
                        "json_path": example.get("json_path") or example.get("argument_path"),
                        "argument_path": example.get("argument_path") or example.get("json_path"),
                        "same_call_arguments": example.get("same_call_arguments") or [],
                        "same_call_peer_arguments": example.get("same_call_peer_arguments") or [],
                        "local_peer_arguments": example.get("local_peer_arguments") or [],
                        "candidate_values": candidates,
                        "candidate_a": value_a,
                        "candidate_b": value_b,
                        "candidate_a_index": index_a,
                        "candidate_b_index": index_b,
                        "answer": answer,
                        "distractor_index": candidate_idx,
                        "order_idx": order_idx,
                        "repeat_idx": repeat_idx,
                        "usable_for_training": True,
                    }
                    row["prompt"] = pairwise_prompt(example, value_a, value_b)
                    row["group_key"] = "|".join("" if item is None else str(item) for item in pair_key(example))
                    rows.append(row)
    return rows, skipped


def instance_from_pair(row):
    return {
        "system": PAIRWISE_SYSTEM,
        "messages": [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["answer"]},
        ],
        "source": (
            f"candidate_pairwise:{row.get('id')}:call{row.get('tool_call_index')}:"
            f"{row.get('json_key')}:distractor{row.get('distractor_index')}:"
            f"order{row.get('order_idx')}:repeat{row.get('repeat_idx')}"
        ),
    }


def strip_source(instance):
    clone = copy.deepcopy(instance)
    clone.pop("source", None)
    return clone


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
    parser.add_argument("--conversation-template", default="fast_dllm_v2")
    parser.add_argument("--include-kinds", nargs="+", default=["argument_value"], choices=["tool_name", "argument_value"])
    parser.add_argument("--only-ids", nargs="*", default=None)
    parser.add_argument("--only-json-paths", nargs="*", default=None)
    parser.add_argument("--only-json-keys", nargs="*", default=None)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--both-orders", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--truncation-side", choices=["left", "right"], default="left")
    parser.add_argument("--min-labels", type=int, default=1)
    parser.add_argument("--require-full-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--contains-eval-slice", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diagnostic-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=971)
    args = parser.parse_args()

    pairwise_rows, skipped = build_pairwise_examples(list(load_jsonl(args.examples_jsonl)), args)
    instances = [instance_from_pair(row) for row in pairwise_rows]
    rng = random.Random(args.seed)
    rng.shuffle(instances)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    chat_template = resolve_chat_template(args.conversation_template)
    accepted, rejected, audit_rows = audit_instances(tokenizer, chat_template, instances, args)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    examples_path = args.out_dir / "pairwise_examples.jsonl"
    write_jsonl(examples_path, pairwise_rows)
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
        "examples_path": str(examples_path),
        "audit_path": str(audit_path),
        "rejected_path": str(rejected_path),
        "count": len(accepted),
        "raw_count": len(instances),
        "pairwise_example_count": len(pairwise_rows),
        "rejected_count": len(rejected),
        "examples_jsonl": str(args.examples_jsonl),
        "skipped_counts": dict(sorted(skipped.items())),
        "tokenizer_model": str(args.model),
        "conversation_template": args.conversation_template,
        "include_kinds": list(args.include_kinds),
        "only_ids": list(args.only_ids or []),
        "only_json_paths": list(args.only_json_paths or []),
        "only_json_keys": list(args.only_json_keys or []),
        "repeat": args.repeat,
        "both_orders": args.both_orders,
        "block_size": args.block_size,
        "truncation_side": args.truncation_side,
        "min_labels": args.min_labels,
        "require_full_labels": args.require_full_labels,
        "diagnostic_only": bool(args.diagnostic_only),
        "contains_eval_slice": bool(args.contains_eval_slice),
        "promotion_allowed": not bool(args.diagnostic_only) and not bool(args.contains_eval_slice),
        "promotion_note": (
            "Built from an eval slice. Use only for objective/debug gates; do not promote checkpoints trained on this corpus."
            if args.contains_eval_slice
            else "Promotion requires separate heldout gates."
        ),
        "chosen_audit_summary": summarize_audit(audit_rows, audit_rows),
        "seed": args.seed,
    }
    manifest_path = args.out_dir / "train_agentic_mix.manifest"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
