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
    DEFAULT_PUBLIC_TRAIN,
    compact_calls,
    dedupe,
    load_conversation,
    load_jsonl,
    normalized_instance,
    strip_source,
    write_jsonl,
)
from build_toolcall_labelaware_public_mix import (
    choose_labelaware_variant,
    resolve_chat_template,
    summarize_audit,
    token_stats,
)
from eval_toolcall_jsonl import extract_tool_calls


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_MODEL = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_BASE_CURRICULUM = ROOT / "data/qwen35_9b_toolcall_model_repair_curriculum/train_agentic_mix.json"
DEFAULT_OUT_DIR = ROOT / "data/qwen35_9b_toolcall_multicall_curriculum"

MULTICALL_SYSTEM = (
    "You are a constrained Qwen tool-call model. Return only <tool_call> "
    "block(s) with valid JSON and no prose."
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


def called_tool_subset(instance, calls):
    names = {call.get("name") for call in calls if call.get("name")}
    tools = [copy.deepcopy(tool) for tool in instance.get("tools") or [] if tool_name(tool) in names]
    return tools or copy.deepcopy(instance.get("tools") or [])


def source_family(instance):
    return (instance.get("source") or "unknown").split(":")[0]


def multicall_from_instance(instance):
    calls, invalid = extract_tool_calls(assistant_text(instance))
    if invalid or len(calls) < 2:
        return None
    if any(not call.get("name") for call in calls):
        return None
    clone = copy.deepcopy(instance)
    clone["system"] = str(clone.get("system") or MULTICALL_SYSTEM).strip() or MULTICALL_SYSTEM
    clone["tools"] = copy.deepcopy(instance.get("tools") or [])
    clone["messages"] = [
        copy.deepcopy(message)
        for message in clone.get("messages") or []
        if message.get("role") in {"user", "assistant", "tool"} and str(message.get("content") or "").strip()
    ]
    clone["messages"][-1]["content"] = compact_calls(calls)
    clone["source"] = f"{instance.get('source') or 'public_train'}:multicall_full"
    return clone, calls


def continuation_variant(instance, calls, split_idx):
    prior = calls[:split_idx]
    remaining = calls[split_idx:]
    if not prior or not remaining:
        return None
    request = user_text(instance)
    return {
        "system": MULTICALL_SYSTEM,
        "tools": called_tool_subset(instance, calls),
        "messages": [
            {
                "role": "user",
                "content": (
                    "Continue the Qwen tool-call sequence for the request. The previous "
                    "tool calls are already completed; return only the remaining "
                    "tool_call block(s) in order.\n\n"
                    "Original request:\n"
                    f"{request}\n\n"
                    "Completed tool calls:\n"
                    f"{compact_calls(prior)}"
                ),
            },
            {"role": "assistant", "content": compact_calls(remaining)},
        ],
        "source": f"{instance.get('source') or 'public_train'}:multicall_continue_after_{split_idx}",
    }


def exact_plan_variant(instance, calls):
    request = user_text(instance)
    plan_lines = []
    for idx, call in enumerate(calls, start=1):
        plan_lines.append(
            f"{idx}. {call['name']} arguments="
            + json.dumps(call.get("arguments") or {}, ensure_ascii=False, separators=(",", ": "))
        )
    return {
        "system": MULTICALL_SYSTEM,
        "tools": called_tool_subset(instance, calls),
        "messages": [
            {
                "role": "user",
                "content": (
                    "Render this exact multi-call plan as Qwen tool-call JSON. Copy every "
                    "argument value exactly, preserve call order, and return no prose.\n\n"
                    "Original request:\n"
                    f"{request}\n\n"
                    "Exact ordered plan:\n"
                    + "\n".join(plan_lines)
                ),
            },
            {"role": "assistant", "content": compact_calls(calls)},
        ],
        "source": f"{instance.get('source') or 'public_train'}:multicall_exact_plan",
    }


def load_base_instances(path, cap, rng):
    if cap == 0 or not path.exists():
        return []
    instances = load_conversation(path)
    rng.shuffle(instances)
    if cap > 0:
        instances = instances[:cap]
    return instances


def load_public_multicall(path, cap, rng):
    candidates = []
    for raw in load_conversation(path):
        instance = normalized_instance(raw, "public_train_multicall")
        parsed = multicall_from_instance(instance)
        if parsed is None:
            continue
        item, calls = parsed
        candidates.append((item, calls))
    rng.shuffle(candidates)
    return candidates[:cap] if cap >= 0 else candidates


def make_multicall_candidates(public_items, args):
    candidates = []
    for instance, calls in public_items:
        if args.include_multicall_full:
            candidates.append(instance)
        if args.include_exact_plan:
            candidates.append(exact_plan_variant(instance, calls))
        if args.include_continuations:
            max_splits = min(len(calls), args.max_continuation_splits + 1)
            for split_idx in range(1, max_splits):
                item = continuation_variant(instance, calls, split_idx)
                if item is not None:
                    candidates.append(item)
    repeated = []
    for repeat_idx in range(max(1, args.multicall_repeat)):
        for instance in candidates:
            clone = copy.deepcopy(instance)
            clone["source"] = f"{instance.get('source')}:repeat{repeat_idx}"
            repeated.append(clone)
    return repeated


def accept_candidates(tokenizer, chat_template, candidates, args, audit_rows):
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
    parser.add_argument("--base-curriculum", type=Path, default=DEFAULT_BASE_CURRICULUM)
    parser.add_argument("--public-train", type=Path, default=DEFAULT_PUBLIC_TRAIN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--base-cap", type=int, default=-1)
    parser.add_argument("--public-multicall-cap", type=int, default=-1)
    parser.add_argument("--multicall-repeat", type=int, default=2)
    parser.add_argument("--max-continuation-splits", type=int, default=3)
    parser.add_argument("--include-multicall-full", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-continuations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-exact-plan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--dedupe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deduplicate final rows. Defaults false so repeated curriculum rows keep sampling weight.",
    )
    parser.add_argument("--block-size", type=int, default=896)
    parser.add_argument("--truncation-side", choices=["left", "right"], default="right")
    parser.add_argument("--min-labels", type=int, default=1)
    parser.add_argument("--require-full-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefer-full-tools", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=211)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "scripts"))
    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    chat_template = resolve_chat_template(args.conversation_template)

    base_instances = load_base_instances(args.base_curriculum, args.base_cap, rng)
    public_multicall = load_public_multicall(args.public_train, args.public_multicall_cap, rng)
    multicall_candidates = make_multicall_candidates(public_multicall, args)
    multicall_audit = []
    multicall_accepted, multicall_rejected = accept_candidates(
        tokenizer,
        chat_template,
        multicall_candidates,
        args,
        multicall_audit,
    )

    instances = base_instances + multicall_accepted
    accepted_before_dedupe = len(instances)
    if args.dedupe:
        instances = dedupe(instances)
    rng.shuffle(instances)

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

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train_agentic_mix.json"
    train_path.write_text(
        json.dumps({"type": "conversation", "instances": [strip_source(item) for item in instances]}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    audit_path = args.out_dir / "train_agentic_mix.audit.jsonl"
    write_jsonl(audit_path, chosen_audit)

    source_counts = Counter(instance.get("source") or "unknown" for instance in instances)
    source_family_counts = Counter(source_family(instance) for instance in instances)
    rejected_by_source = defaultdict(int)
    for item in multicall_rejected:
        rejected_by_source[item["source"]] += 1

    manifest = {
        "train_path": str(train_path),
        "audit_path": str(audit_path),
        "count": len(instances),
        "base_curriculum": str(args.base_curriculum),
        "base_count": len(base_instances),
        "accepted_before_dedupe": accepted_before_dedupe,
        "dedupe": args.dedupe,
        "public_train": str(args.public_train),
        "public_multicall_records": len(public_multicall),
        "multicall_candidate_count": len(multicall_candidates),
        "multicall_accepted_count": len(multicall_accepted),
        "multicall_rejected_count": len(multicall_rejected),
        "source_counts": dict(sorted(source_counts.items())),
        "source_family_counts": dict(sorted(source_family_counts.items())),
        "rejected_by_source": dict(sorted(rejected_by_source.items())),
        "chosen_audit_summary": summarize_audit(chosen_audit, chosen_audit),
        "multicall_candidate_audit_summary": summarize_audit(multicall_audit, chosen_audit),
        "tokenizer_model": str(args.model),
        "conversation_template": args.conversation_template,
        "block_size": args.block_size,
        "truncation_side": args.truncation_side,
        "min_labels": args.min_labels,
        "require_full_labels": args.require_full_labels,
        "prefer_full_tools": args.prefer_full_tools,
        "base_cap": args.base_cap,
        "public_multicall_cap": args.public_multicall_cap,
        "multicall_repeat": args.multicall_repeat,
        "max_continuation_splits": args.max_continuation_splits,
        "include_multicall_full": args.include_multicall_full,
        "include_continuations": args.include_continuations,
        "include_exact_plan": args.include_exact_plan,
        "seed": args.seed,
        "rejected_examples": multicall_rejected[:20],
    }
    manifest_path = args.out_dir / "train_agentic_mix.manifest"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
