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
    compact_calls,
    dedupe,
    load_conversation,
    load_jsonl,
    normalized_instance,
    strip_source,
    teacher_exact_records,
    write_jsonl,
)
from build_toolcall_labelaware_public_mix import (
    choose_labelaware_variant,
    public_onecall_instances,
    resolve_chat_template,
    summarize_audit,
    token_stats,
)
from eval_toolcall_jsonl import extract_tool_calls


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_MODEL = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_OUT_DIR = ROOT / "data/qwen35_9b_toolcall_argument_curriculum"
DEFAULT_TEACHER_TRAIN_EVAL = ROOT / "data/toolcall_eval/public_onecall_teacher_train_argument_smoke.jsonl"
DEFAULT_TEACHER_HELDOUT_EVAL = ROOT / "data/toolcall_eval/public_onecall_teacher_heldout_argument_smoke.jsonl"

ARGUMENT_SYSTEM = (
    "You are a tool-call formatter. Return exactly one <tool_call> block with "
    "valid JSON and no prose."
)


def assistant_text(instance):
    return "\n".join(
        str(message.get("content") or "")
        for message in instance.get("messages") or []
        if message.get("role") == "assistant"
    )


def user_text(instance):
    return "\n\n".join(
        str(message.get("content") or "")
        for message in instance.get("messages") or []
        if message.get("role") == "user"
    ).strip()


def tool_name(tool):
    fn = tool.get("function", tool) if isinstance(tool, dict) else {}
    if isinstance(fn, dict) and fn.get("name"):
        return str(fn["name"])
    return None


def gold_tools(instance, name):
    tools = [copy.deepcopy(tool) for tool in instance.get("tools") or [] if tool_name(tool) == name]
    return tools or copy.deepcopy(instance.get("tools") or [])


def source_family(instance):
    return (instance.get("source") or "unknown").split(":")[0]


def one_call(instance):
    calls, invalid = extract_tool_calls(assistant_text(instance))
    if invalid or len(calls) != 1:
        return None
    call = calls[0]
    if not call.get("name"):
        return None
    return {"name": str(call["name"]), "arguments": call.get("arguments") or {}}


def argument_copy_variant(instance, call):
    name = call["name"]
    arguments = call.get("arguments") or {}
    return {
        "system": ARGUMENT_SYSTEM,
        "tools": gold_tools(instance, name),
        "messages": [
            {
                "role": "user",
                "content": (
                    "Format this exact function call as a Qwen tool call.\n"
                    f"Function name: {name}\n"
                    "Arguments JSON: "
                    + json.dumps(arguments, ensure_ascii=False, separators=(",", ": "))
                    + "\nReturn exactly one <tool_call> block."
                ),
            },
            {"role": "assistant", "content": compact_calls([call])},
        ],
        "source": f"{instance.get('source') or 'unknown'}:argument_copy",
    }


def argument_context_variant(instance, call):
    name = call["name"]
    arguments = call.get("arguments") or {}
    request = user_text(instance)
    return {
        "system": ARGUMENT_SYSTEM,
        "tools": gold_tools(instance, name),
        "messages": [
            {
                "role": "user",
                "content": (
                    "Return the Qwen tool call for this request using the selected function "
                    "and exact arguments.\n"
                    "User request:\n"
                    f"{request}\n\n"
                    f"Selected function: {name}\n"
                    "Arguments JSON: "
                    + json.dumps(arguments, ensure_ascii=False, separators=(",", ": "))
                ),
            },
            {"role": "assistant", "content": compact_calls([call])},
        ],
        "source": f"{instance.get('source') or 'unknown'}:argument_context",
    }


def argument_key_value_variant(instance, call):
    name = call["name"]
    arguments = call.get("arguments") or {}
    lines = []
    for key, value in arguments.items():
        lines.append(f"- {key}: {json.dumps(value, ensure_ascii=False, separators=(',', ': '))}")
    return {
        "system": ARGUMENT_SYSTEM,
        "tools": gold_tools(instance, name),
        "messages": [
            {
                "role": "user",
                "content": (
                    "Build the exact arguments object inside a Qwen tool call.\n"
                    f"Function name: {name}\n"
                    "Argument key-values:\n"
                    + "\n".join(lines)
                ),
            },
            {"role": "assistant", "content": compact_calls([call])},
        ],
        "source": f"{instance.get('source') or 'unknown'}:argument_key_values",
    }


ARGUMENT_VARIANTS = {
    "copy": argument_copy_variant,
    "context": argument_context_variant,
    "key_values": argument_key_value_variant,
}


def build_labelaware_originals(tokenizer, chat_template, args):
    rng = random.Random(args.seed)
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
    originals = []
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
            originals.append(chosen)
    return originals, rejected, audit_rows, teacher_train_ids, raw_candidates


def make_argument_candidates(originals, args):
    families = set(args.argument_sources.split(","))
    variants = args.argument_variants.split(",")
    out = []
    for instance in originals:
        if source_family(instance) not in families:
            continue
        call = one_call(instance)
        if call is None:
            continue
        for variant in variants:
            make_variant = ARGUMENT_VARIANTS.get(variant)
            if make_variant is None:
                raise ValueError(f"unknown argument variant {variant!r}")
            out.append(make_variant(instance, call))
    if args.argument_cap >= 0:
        rng = random.Random(args.seed + 1)
        rng.shuffle(out)
        out = out[: args.argument_cap]
    return out


def accept_argument_candidates(tokenizer, chat_template, candidates, args, audit_rows):
    accepted = []
    rejected = []
    for instance in candidates:
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
    return accepted, rejected


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
    parser.add_argument("--argument-sources", default="public_train_onecall,public_teacher_exact_onecall")
    parser.add_argument("--argument-variants", default="copy,context,key_values")
    parser.add_argument("--argument-cap", type=int, default=-1)
    parser.add_argument("--include-originals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=131)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    chat_template = resolve_chat_template(args.conversation_template)

    originals, original_rejected, original_audit, teacher_train_ids, raw_candidates = build_labelaware_originals(
        tokenizer,
        chat_template,
        args,
    )
    argument_candidates = make_argument_candidates(originals, args)
    argument_audit = []
    argument_accepted, argument_rejected = accept_argument_candidates(
        tokenizer,
        chat_template,
        argument_candidates,
        args,
        argument_audit,
    )

    accepted = []
    if args.include_originals:
        accepted.extend(originals)
    accepted.extend(argument_accepted)
    instances = dedupe(accepted)
    rng.shuffle(instances)

    source_counts = Counter(instance.get("source") or "unknown" for instance in instances)
    source_family_counts = Counter(source_family(instance) for instance in instances)
    variant_counts = Counter(":".join((instance.get("source") or "unknown").split(":")[1:]) or "original" for instance in instances)

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

    rejected = original_rejected + argument_rejected
    rejected_by_source = defaultdict(int)
    for item in rejected:
        rejected_by_source[item["source"]] += 1

    manifest = {
        "train_path": str(train_path),
        "audit_path": str(audit_path),
        "count": len(instances),
        "raw_original_candidate_count": len(raw_candidates),
        "labelaware_original_count": len(originals),
        "argument_candidate_count": len(argument_candidates),
        "argument_accepted_count": len(argument_accepted),
        "accepted_before_dedupe": len(accepted),
        "deduped_accepted_count": len(instances),
        "rejected_count": len(rejected),
        "source_counts": dict(sorted(source_counts.items())),
        "source_family_counts": dict(sorted(source_family_counts.items())),
        "variant_counts": dict(sorted(variant_counts.items())),
        "rejected_by_source": dict(sorted(rejected_by_source.items())),
        "tokenizer_model": str(args.model),
        "conversation_template": args.conversation_template,
        "block_size": args.block_size,
        "truncation_side": args.truncation_side,
        "min_labels": args.min_labels,
        "require_full_labels": args.require_full_labels,
        "prefer_full_tools": args.prefer_full_tools,
        "argument_sources": args.argument_sources,
        "argument_variants": args.argument_variants,
        "argument_cap": args.argument_cap,
        "include_originals": args.include_originals,
        "chosen_audit_summary": summarize_audit(chosen_audit, chosen_audit),
        "original_candidate_audit_summary": summarize_audit(original_audit, chosen_audit),
        "argument_candidate_audit_summary": summarize_audit(argument_audit, chosen_audit),
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
