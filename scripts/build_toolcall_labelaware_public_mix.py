#!/usr/bin/env python3
import argparse
import copy
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from build_toolcall_format_public_mix import (
    DEFAULT_FORMAT_TRAIN,
    DEFAULT_PUBLIC_EVAL,
    DEFAULT_PUBLIC_TEACHER,
    DEFAULT_PUBLIC_TRAIN,
    DEFAULT_SYSTEM,
    case_to_instance,
    compact_calls,
    dedupe,
    load_conversation,
    load_jsonl,
    normalized_instance,
    strip_source,
    teacher_exact_records,
    write_jsonl,
)
from eval_toolcall_jsonl import extract_tool_calls


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_MODEL = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_OUT_DIR = ROOT / "data/qwen35_9b_toolcall_labelaware_public_curriculum"
DEFAULT_TEACHER_TRAIN_EVAL = ROOT / "data/toolcall_eval/public_onecall_teacher_train_labelaware_smoke.jsonl"
DEFAULT_TEACHER_HELDOUT_EVAL = ROOT / "data/toolcall_eval/public_onecall_teacher_heldout_labelaware_smoke.jsonl"


def drop_none_fields(value):
    if isinstance(value, dict):
        return {key: drop_none_fields(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [drop_none_fields(item) for item in value if item is not None]
    return value


def resolve_chat_template(name):
    third_party = ROOT / "fast-dllm/third_party"
    if str(third_party) not in sys.path:
        sys.path.insert(0, str(third_party))
    from lmflow.utils.conversation_template import PRESET_TEMPLATES

    if name not in PRESET_TEMPLATES:
        raise ValueError(f"unknown conversation template {name!r}")
    return PRESET_TEMPLATES[name]


def assistant_text(instance):
    return "\n".join(
        str(message.get("content") or "")
        for message in instance.get("messages") or []
        if message.get("role") == "assistant"
    )


def called_tool_names(instance):
    calls, invalid = extract_tool_calls(assistant_text(instance))
    if invalid:
        return []
    names = []
    for call in calls:
        name = call.get("name")
        if name and name not in names:
            names.append(name)
    return names


def tool_name(tool):
    fn = tool.get("function", tool) if isinstance(tool, dict) else {}
    if isinstance(fn, dict) and fn.get("name"):
        return str(fn["name"])
    return None


def prune_tools_to_calls(instance):
    names = set(called_tool_names(instance))
    if not names:
        return None
    tools = instance.get("tools") or []
    pruned = [copy.deepcopy(tool) for tool in tools if tool_name(tool) in names]
    if not pruned:
        return None
    clone = copy.deepcopy(instance)
    clone["tools"] = pruned
    clone["source"] = f"{instance.get('source') or 'unknown'}_gold_tools"
    return clone


def conversation_for_template(instance):
    system = instance.get("system")
    conversation = [{"role": "system", "content": system if system is not None else DEFAULT_SYSTEM}]
    conversation.extend(copy.deepcopy(instance.get("messages") or []))
    return drop_none_fields(conversation)


def token_stats(tokenizer, chat_template, instance, block_size, truncation_side):
    encoded = tokenizer.apply_chat_template(
        conversation=conversation_for_template(instance),
        tools=drop_none_fields(instance.get("tools") or None),
        chat_template=chat_template,
        return_assistant_tokens_mask=True,
        return_dict=True,
    )
    labels = [
        token if mask == 1 else -100
        for token, mask in zip(encoded["input_ids"], encoded["assistant_masks"])
    ]
    full_labels = sum(label != -100 for label in labels)
    if len(labels) <= block_size:
        kept = labels
    elif truncation_side == "right":
        kept = labels[:block_size]
    elif truncation_side == "left":
        kept = labels[-block_size:]
    else:
        raise ValueError(f"unsupported truncation side {truncation_side!r}")
    kept_labels = sum(label != -100 for label in kept)
    return {
        "length": len(labels),
        "full_labels": full_labels,
        "kept_labels": kept_labels,
        "full_labels_kept": full_labels > 0 and kept_labels == full_labels,
        "zero_after_truncation": kept_labels == 0,
        "partial_after_truncation": 0 < kept_labels < full_labels,
    }


def label_ok(stats, require_full_labels, min_labels):
    if stats["kept_labels"] < min_labels:
        return False
    if require_full_labels and not stats["full_labels_kept"]:
        return False
    return True


def choose_labelaware_variant(tokenizer, chat_template, instance, args, audit_rows):
    candidates = [("full_tools", instance)]
    pruned = prune_tools_to_calls(instance)
    if pruned is not None:
        candidates.append(("gold_tools", pruned))

    scored = []
    for variant_name, candidate in candidates:
        stats = token_stats(tokenizer, chat_template, candidate, args.block_size, args.truncation_side)
        scored.append((variant_name, candidate, stats))
        audit_rows.append(
            {
                "source": instance.get("source") or "unknown",
                "variant": variant_name,
                "tool_count": len(candidate.get("tools") or []),
                **stats,
            }
        )

    passing = [
        item
        for item in scored
        if label_ok(item[2], args.require_full_labels, args.min_labels)
    ]
    if not passing:
        return None, scored

    full = next((item for item in passing if item[0] == "full_tools"), None)
    if full is not None and (args.prefer_full_tools or len(full[1].get("tools") or []) <= 1):
        chosen = full
    else:
        chosen = min(
            passing,
            key=lambda item: (
                len(item[1].get("tools") or []),
                item[2]["length"],
                0 if item[0] == "full_tools" else 1,
            ),
        )

    variant_name, candidate, stats = chosen
    out = copy.deepcopy(candidate)
    out["source"] = f"{instance.get('source') or 'unknown'}:{variant_name}"
    return out, scored


def public_onecall_instances(path, cap, rng):
    candidates = []
    for instance in load_conversation(path):
        item = normalized_instance(instance, "public_train_onecall")
        calls, invalid = extract_tool_calls(assistant_text(item))
        if len(calls) != 1 or invalid:
            continue
        item["messages"][-1]["content"] = compact_calls(calls)
        candidates.append(item)
    rng.shuffle(candidates)
    return candidates[:cap] if cap >= 0 else candidates


def percentiles(values):
    if not values:
        return {}
    values = sorted(values)

    def at(frac):
        idx = min(len(values) - 1, max(0, round((len(values) - 1) * frac)))
        return values[idx]

    return {
        "min": values[0],
        "p50": at(0.50),
        "p90": at(0.90),
        "max": values[-1],
    }


def summarize_audit(audit_rows, accepted):
    chosen_lengths = [row["length"] for row in accepted]
    chosen_labels = [row["kept_labels"] for row in accepted]
    return {
        "accepted_count": len(accepted),
        "chosen_length": percentiles(chosen_lengths),
        "chosen_kept_labels": percentiles(chosen_labels),
        "candidate_zero_after_truncation": sum(row["zero_after_truncation"] for row in audit_rows),
        "candidate_partial_after_truncation": sum(row["partial_after_truncation"] for row in audit_rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--conversation-template", default="fast_dllm_v2")
    parser.add_argument("--format-train", type=Path, default=DEFAULT_FORMAT_TRAIN)
    parser.add_argument("--public-train", type=Path, default=DEFAULT_PUBLIC_TRAIN)
    parser.add_argument("--public-eval", type=Path, default=DEFAULT_PUBLIC_EVAL)
    parser.add_argument("--public-teacher", type=Path, default=DEFAULT_PUBLIC_TEACHER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--teacher-train-eval-out", type=Path, default=DEFAULT_TEACHER_TRAIN_EVAL)
    parser.add_argument("--teacher-heldout-eval-out", type=Path, default=DEFAULT_TEACHER_HELDOUT_EVAL)
    parser.add_argument("--format-cap", type=int, default=96)
    parser.add_argument("--public-train-onecall-cap", type=int, default=40)
    parser.add_argument("--teacher-exact-cap", type=int, default=12)
    parser.add_argument("--heldout-limit", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=896)
    parser.add_argument("--truncation-side", choices=["left", "right"], default="right")
    parser.add_argument("--min-labels", type=int, default=1)
    parser.add_argument("--require-full-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefer-full-tools", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=97)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    chat_template = resolve_chat_template(args.conversation_template)

    format_instances = [normalized_instance(item, "format_curriculum") for item in load_conversation(args.format_train)]
    rng.shuffle(format_instances)
    if args.format_cap >= 0:
        format_instances = format_instances[: args.format_cap]

    public_instances = public_onecall_instances(args.public_train, args.public_train_onecall_cap, rng)
    teacher_records = teacher_exact_records(args.public_teacher, args.public_eval, args.teacher_exact_cap)
    teacher_instances = [item[2] for item in teacher_records]
    teacher_train_ids = {item[0].get("id") for item in teacher_records}

    raw_candidates = format_instances + public_instances + teacher_instances
    audit_rows = []
    accepted = []
    rejected = []
    for instance in raw_candidates:
        chosen, scored = choose_labelaware_variant(tokenizer, chat_template, instance, args, audit_rows)
        if chosen is None:
            rejected.append(
                {
                    "source": instance.get("source") or "unknown",
                    "tool_count": len(instance.get("tools") or []),
                    "candidate_stats": [
                        {"variant": variant, "tool_count": len(candidate.get("tools") or []), **stats}
                        for variant, candidate, stats in scored
                    ],
                }
            )
        else:
            accepted.append(chosen)

    instances = dedupe(accepted)
    rng.shuffle(instances)
    source_counts = Counter(instance.get("source") or "unknown" for instance in instances)
    source_family_counts = Counter((instance.get("source") or "unknown").split(":")[0] for instance in instances)

    public_cases = load_jsonl(args.public_eval)
    teacher_train_eval = [case for case in public_cases if case.get("id") in teacher_train_ids]
    teacher_heldout_eval = [case for case in public_cases if case.get("id") not in teacher_train_ids][: args.heldout_limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train_agentic_mix.json"
    train_path.write_text(
        json.dumps({"type": "conversation", "instances": [strip_source(item) for item in instances]}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    write_jsonl(args.teacher_train_eval_out, teacher_train_eval)
    write_jsonl(args.teacher_heldout_eval_out, teacher_heldout_eval)

    chosen_audit = []
    for instance in instances:
        stats = token_stats(tokenizer, chat_template, instance, args.block_size, args.truncation_side)
        chosen_audit.append(
            {
                "source": instance.get("source") or "unknown",
                "tool_count": len(instance.get("tools") or []),
                **stats,
            }
        )
    audit_path = args.out_dir / "train_agentic_mix.audit.jsonl"
    write_jsonl(audit_path, chosen_audit)

    rejected_by_source = defaultdict(int)
    for item in rejected:
        rejected_by_source[item["source"]] += 1

    manifest = {
        "train_path": str(train_path),
        "audit_path": str(audit_path),
        "count": len(instances),
        "raw_candidate_count": len(raw_candidates),
        "deduped_accepted_count": len(instances),
        "accepted_before_dedupe": len(accepted),
        "rejected_count": len(rejected),
        "source_counts": dict(sorted(source_counts.items())),
        "source_family_counts": dict(sorted(source_family_counts.items())),
        "rejected_by_source": dict(sorted(rejected_by_source.items())),
        "tokenizer_model": str(args.model),
        "conversation_template": args.conversation_template,
        "block_size": args.block_size,
        "truncation_side": args.truncation_side,
        "min_labels": args.min_labels,
        "require_full_labels": args.require_full_labels,
        "prefer_full_tools": args.prefer_full_tools,
        "chosen_audit_summary": summarize_audit(chosen_audit, chosen_audit),
        "all_candidate_audit_summary": summarize_audit(audit_rows, chosen_audit),
        "seed": args.seed,
        "format_cap": args.format_cap,
        "public_train_onecall_cap": args.public_train_onecall_cap,
        "teacher_exact_cap": args.teacher_exact_cap,
        "teacher_train_ids": sorted(teacher_train_ids),
        "teacher_train_eval_path": str(args.teacher_train_eval_out),
        "teacher_train_eval_count": len(teacher_train_eval),
        "teacher_heldout_eval_path": str(args.teacher_heldout_eval_out),
        "teacher_heldout_eval_count": len(teacher_heldout_eval),
        "rejected_examples": rejected[:20],
    }
    manifest_path = args.out_dir / "train_agentic_mix.manifest"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
