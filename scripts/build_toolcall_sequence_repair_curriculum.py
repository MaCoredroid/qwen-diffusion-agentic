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
    load_conversation,
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
DEFAULT_OUT_DIR = ROOT / "data/qwen35_9b_toolcall_sequence_repair_curriculum"

REPAIR_SYSTEM = "You are a constrained fixed-sequence tool-call repair model."


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


def multicall_from_instance(instance):
    calls, invalid = extract_tool_calls(assistant_text(instance))
    if invalid or len(calls) < 2:
        return None
    if any(not call.get("name") for call in calls):
        return None
    clone = copy.deepcopy(instance)
    clone["system"] = str(clone.get("system") or REPAIR_SYSTEM).strip() or REPAIR_SYSTEM
    clone["tools"] = copy.deepcopy(instance.get("tools") or [])
    clone["messages"] = [
        copy.deepcopy(message)
        for message in clone.get("messages") or []
        if message.get("role") in {"user", "assistant", "tool"} and str(message.get("content") or "").strip()
    ]
    clone["messages"][-1]["content"] = compact_calls(calls)
    clone["source"] = f"{instance.get('source') or 'public_train'}:sequence_repair_source"
    return clone, calls


def scalar_paths(value, prefix=()):
    if isinstance(value, dict):
        out = []
        for key, item in value.items():
            out.extend(scalar_paths(item, (*prefix, key)))
        return out
    if isinstance(value, list):
        out = []
        for idx, item in enumerate(value):
            out.extend(scalar_paths(item, (*prefix, idx)))
        return out
    if isinstance(value, (str, int, float, bool)) or value is None:
        return [(prefix, value)]
    return []


def wrong_scalar(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, int) and not isinstance(value, bool):
        return value + 1
    if isinstance(value, float):
        return value + 1.0
    if isinstance(value, str):
        return "UNKNOWN"
    return "UNKNOWN"


def map_scalar_values(value, mode, counter=None):
    if isinstance(value, dict):
        return {key: map_scalar_values(item, mode, counter) for key, item in value.items()}
    if isinstance(value, list):
        return [map_scalar_values(item, mode, counter) for item in value]
    if mode == "null":
        return None
    if mode == "wrong":
        return wrong_scalar(value)
    if mode == "mixed_null":
        counter[0] += 1
        return value if counter[0] % 2 else None
    if mode == "mixed_wrong":
        counter[0] += 1
        return value if counter[0] % 2 else wrong_scalar(value)
    raise ValueError(f"unknown scalar map mode {mode!r}")


def drop_first_top_level_arg(arguments):
    if not isinstance(arguments, dict) or not arguments:
        return {}
    clone = copy.deepcopy(arguments)
    first_key = next(iter(clone))
    clone.pop(first_key, None)
    return clone


def sequence_draft(calls, variant):
    draft_calls = []
    if variant == "empty_args":
        for call in calls:
            draft_calls.append({"name": call["name"], "arguments": {}})
    elif variant == "null_args":
        for call in calls:
            draft_calls.append({"name": call["name"], "arguments": map_scalar_values(call.get("arguments") or {}, "null")})
    elif variant == "wrong_scalar":
        for call in calls:
            draft_calls.append({"name": call["name"], "arguments": map_scalar_values(call.get("arguments") or {}, "wrong")})
    elif variant == "drop_first_arg":
        for call in calls:
            draft_calls.append({"name": call["name"], "arguments": drop_first_top_level_arg(call.get("arguments") or {})})
    elif variant == "mixed_null":
        counter = [0]
        for call in calls:
            draft_calls.append(
                {"name": call["name"], "arguments": map_scalar_values(call.get("arguments") or {}, "mixed_null", counter)}
            )
    elif variant == "mixed_wrong":
        counter = [0]
        for call in calls:
            draft_calls.append(
                {"name": call["name"], "arguments": map_scalar_values(call.get("arguments") or {}, "mixed_wrong", counter)}
            )
    elif variant == "gold_skeleton":
        for call in calls:
            keys = {path[0]: None for path, _ in scalar_paths(call.get("arguments") or {}) if path}
            draft_calls.append({"name": call["name"], "arguments": keys})
    else:
        raise ValueError(f"unknown sequence repair variant {variant!r}")
    return compact_calls(draft_calls)


def sequence_repair_prompt(raw_draft):
    return (
        "The draft below contains the exact tool-call sequence to preserve. "
        "Keep the same number of tool calls, the same function names, and the "
        "same order. Repair only the arguments by copying exact values from the "
        "original user request and tool schema. Do not add, remove, rename, or "
        "reorder tools. Return only corrected Qwen <tool_call> blocks with JSON "
        "payloads and no prose.\n\n"
        "Fixed-sequence draft:\n"
        f"{raw_draft}"
    )


def repair_instance(instance, calls, draft, source):
    request = user_text(instance)
    return {
        "system": REPAIR_SYSTEM,
        "tools": called_tool_subset(instance, calls),
        "messages": [
            {"role": "user", "content": request},
            {"role": "user", "content": sequence_repair_prompt(draft)},
            {"role": "assistant", "content": compact_calls(calls)},
        ],
        "source": source,
    }


def load_public_multicall(path, cap, rng):
    candidates = []
    for raw in load_conversation(path):
        instance = normalized_instance(raw, "public_train_multicall")
        parsed = multicall_from_instance(instance)
        if parsed is not None:
            candidates.append(parsed)
    rng.shuffle(candidates)
    return candidates[:cap] if cap >= 0 else candidates


def make_sequence_repair_candidates(public_items, args):
    variants = [item for item in args.repair_variants.split(",") if item]
    candidates = []
    for instance, calls in public_items:
        for variant in variants:
            draft = sequence_draft(calls, variant)
            if not draft:
                continue
            candidates.append(
                repair_instance(
                    instance,
                    calls,
                    draft,
                    f"{instance.get('source') or 'public_train'}:{variant}",
                )
            )
    repeated = []
    for repeat_idx in range(max(1, args.repair_repeat)):
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


def source_family(instance):
    return (instance.get("source") or "unknown").split(":")[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--conversation-template", default="fast_dllm_v2")
    parser.add_argument("--public-train", type=Path, default=DEFAULT_PUBLIC_TRAIN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--public-multicall-cap", type=int, default=-1)
    parser.add_argument("--repair-repeat", type=int, default=2)
    parser.add_argument(
        "--repair-variants",
        default="empty_args,null_args,wrong_scalar,drop_first_arg,mixed_null,mixed_wrong,gold_skeleton",
    )
    parser.add_argument("--block-size", type=int, default=896)
    parser.add_argument("--truncation-side", choices=["left", "right"], default="right")
    parser.add_argument("--min-labels", type=int, default=1)
    parser.add_argument("--require-full-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefer-full-tools", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=317)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "scripts"))
    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    chat_template = resolve_chat_template(args.conversation_template)

    public_multicall = load_public_multicall(args.public_train, args.public_multicall_cap, rng)
    candidates = make_sequence_repair_candidates(public_multicall, args)
    candidate_audit = []
    accepted, rejected = accept_candidates(tokenizer, chat_template, candidates, args, candidate_audit)
    rng.shuffle(accepted)

    chosen_audit = []
    for instance in accepted:
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
        json.dumps({"type": "conversation", "instances": [strip_source(item) for item in accepted]}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    audit_path = args.out_dir / "train_agentic_mix.audit.jsonl"
    write_jsonl(audit_path, chosen_audit)

    source_counts = Counter(instance.get("source") or "unknown" for instance in accepted)
    source_family_counts = Counter(source_family(instance) for instance in accepted)
    rejected_by_source = defaultdict(int)
    for item in rejected:
        rejected_by_source[item["source"]] += 1

    manifest = {
        "train_path": str(train_path),
        "audit_path": str(audit_path),
        "count": len(accepted),
        "public_train": str(args.public_train),
        "public_multicall_records": len(public_multicall),
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "source_counts": dict(sorted(source_counts.items())),
        "source_family_counts": dict(sorted(source_family_counts.items())),
        "rejected_by_source": dict(sorted(rejected_by_source.items())),
        "chosen_audit_summary": summarize_audit(chosen_audit, chosen_audit),
        "candidate_audit_summary": summarize_audit(candidate_audit, chosen_audit),
        "tokenizer_model": str(args.model),
        "conversation_template": args.conversation_template,
        "block_size": args.block_size,
        "truncation_side": args.truncation_side,
        "min_labels": args.min_labels,
        "require_full_labels": args.require_full_labels,
        "prefer_full_tools": args.prefer_full_tools,
        "public_multicall_cap": args.public_multicall_cap,
        "repair_repeat": args.repair_repeat,
        "repair_variants": args.repair_variants,
        "seed": args.seed,
        "rejected_examples": rejected[:20],
    }
    manifest_path = args.out_dir / "train_agentic_mix.manifest"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
