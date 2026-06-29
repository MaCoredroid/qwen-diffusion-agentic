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
DEFAULT_BASE_CURRICULUM = ROOT / "data/qwen35_9b_toolcall_model_repair_curriculum/train_agentic_mix.json"
DEFAULT_OUT_DIR = ROOT / "data/qwen35_9b_toolcall_multicall_repair_curriculum"

REPAIR_SYSTEM = "You are a constrained multi-call tool-call repair model."


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


def compact_json_payloads(calls):
    return "\n".join(
        json.dumps(
            {"name": call["name"], "arguments": call.get("arguments") or {}},
            ensure_ascii=False,
            separators=(",", ": "),
        )
        for call in calls
    )


def repair_prompt(raw_draft):
    return (
        "The previous assistant draft below may have malformed tool-call syntax, "
        "wrong JSON punctuation, missing argument keys, or extra prose. Rewrite it "
        "using the same user request and available tools. Return only valid Qwen "
        "tool-call block(s) in this exact shape, with no prose before or after:\n"
        "<tool_call>\n"
        "{\"name\": \"tool_name\", \"arguments\": {}}\n"
        "</tool_call>\n\n"
        "Previous assistant draft:\n"
        f"{raw_draft}"
    )


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
    clone["source"] = f"{instance.get('source') or 'public_train'}:multicall_repair_source"
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


def set_path(value, path, replacement):
    if not path:
        return replacement
    clone = copy.deepcopy(value)
    cursor = clone
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    return clone


def corrupted_scalar_calls(calls):
    scalar_items = []
    for call_idx, call in enumerate(calls):
        for path, value in scalar_paths(call.get("arguments") or {}):
            scalar_items.append((call_idx, path, value))
    if not scalar_items:
        return None
    call_idx, path, value = scalar_items[0]
    replacement = next((item[2] for item in scalar_items[1:] if item[2] != value), None)
    if replacement is None:
        if isinstance(value, str):
            replacement = f"{value}_wrong"
        elif isinstance(value, bool):
            replacement = not value
        elif isinstance(value, int):
            replacement = value + 1
        elif isinstance(value, float):
            replacement = value + 1.0
        else:
            replacement = "UNKNOWN"
    corrupted = copy.deepcopy(calls)
    corrupted[call_idx]["arguments"] = set_path(corrupted[call_idx].get("arguments") or {}, path, replacement)
    return corrupted


def repair_draft(calls, variant):
    target = compact_calls(calls)
    if variant == "json_only":
        return compact_json_payloads(calls)
    if variant == "missing_wrapper":
        return target.replace("<tool_call>\n", "").replace("\n</tool_call>", "")
    if variant == "wrong_arguments_key":
        return target.replace('"arguments"', '"args"')
    if variant == "partial_chain":
        return compact_calls(calls[:-1])
    if variant == "truncated":
        cut = min(max(120, len(target) // 2), max(1, len(target) - 1), 420)
        return target[:cut]
    if variant == "reversed_order":
        return compact_calls(list(reversed(calls)))
    if variant == "wrong_scalar":
        corrupted = corrupted_scalar_calls(calls)
        return compact_calls(corrupted) if corrupted else None
    if variant == "prose":
        return (
            "I should call these tools in order:\n"
            + "\n".join(
                f"- {call['name']} with {json.dumps(call.get('arguments') or {}, ensure_ascii=False, separators=(',', ': '))}"
                for call in calls
            )
        )
    raise ValueError(f"unknown repair variant {variant!r}")


def repair_instance(instance, calls, draft, source):
    request = user_text(instance)
    return {
        "system": REPAIR_SYSTEM,
        "tools": called_tool_subset(instance, calls),
        "messages": [
            {"role": "user", "content": request},
            {"role": "user", "content": repair_prompt(draft)},
            {"role": "assistant", "content": compact_calls(calls)},
        ],
        "source": source,
    }


def load_base_instances(path, cap, rng):
    if cap == 0 or not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    instances = payload.get("instances") or []
    rng.shuffle(instances)
    if cap > 0:
        instances = instances[:cap]
    return instances


def load_public_multicall(path, cap, rng):
    candidates = []
    for raw in load_conversation(path):
        instance = normalized_instance(raw, "public_train_multicall")
        parsed = multicall_from_instance(instance)
        if parsed is not None:
            candidates.append(parsed)
    rng.shuffle(candidates)
    return candidates[:cap] if cap >= 0 else candidates


def make_multicall_repair_candidates(public_items, args):
    variants = [item for item in args.repair_variants.split(",") if item]
    candidates = []
    for instance, calls in public_items:
        for variant in variants:
            draft = repair_draft(calls, variant)
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
    parser.add_argument("--base-curriculum", type=Path, default=DEFAULT_BASE_CURRICULUM)
    parser.add_argument("--public-train", type=Path, default=DEFAULT_PUBLIC_TRAIN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--base-cap", type=int, default=120)
    parser.add_argument("--public-multicall-cap", type=int, default=-1)
    parser.add_argument("--repair-repeat", type=int, default=2)
    parser.add_argument(
        "--repair-variants",
        default="json_only,missing_wrapper,wrong_arguments_key,partial_chain,truncated,reversed_order,wrong_scalar,prose",
    )
    parser.add_argument("--block-size", type=int, default=896)
    parser.add_argument("--truncation-side", choices=["left", "right"], default="right")
    parser.add_argument("--min-labels", type=int, default=1)
    parser.add_argument("--require-full-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefer-full-tools", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=233)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "scripts"))
    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    chat_template = resolve_chat_template(args.conversation_template)

    base_instances = load_base_instances(args.base_curriculum, args.base_cap, rng)
    public_multicall = load_public_multicall(args.public_train, args.public_multicall_cap, rng)
    repair_candidates = make_multicall_repair_candidates(public_multicall, args)
    repair_audit = []
    repair_accepted, repair_rejected = accept_candidates(tokenizer, chat_template, repair_candidates, args, repair_audit)
    repair_chosen_audit = [
        {
            "source": instance.get("source") or "unknown",
            "tool_count": len(instance.get("tools") or []),
            **token_stats(tokenizer, chat_template, instance, args.block_size, args.truncation_side),
        }
        for instance in repair_accepted
    ]

    instances = base_instances + repair_accepted
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
    for item in repair_rejected:
        rejected_by_source[item["source"]] += 1

    manifest = {
        "train_path": str(train_path),
        "audit_path": str(audit_path),
        "count": len(instances),
        "base_curriculum": str(args.base_curriculum),
        "base_count": len(base_instances),
        "public_train": str(args.public_train),
        "public_multicall_records": len(public_multicall),
        "repair_candidate_count": len(repair_candidates),
        "repair_accepted_count": len(repair_accepted),
        "repair_rejected_count": len(repair_rejected),
        "source_counts": dict(sorted(source_counts.items())),
        "source_family_counts": dict(sorted(source_family_counts.items())),
        "rejected_by_source": dict(sorted(rejected_by_source.items())),
        "chosen_audit_summary": summarize_audit(chosen_audit, chosen_audit),
        "repair_accepted_audit_summary": summarize_audit(repair_chosen_audit, repair_chosen_audit),
        "repair_candidate_audit_summary": summarize_audit(repair_audit, repair_chosen_audit),
        "tokenizer_model": str(args.model),
        "conversation_template": args.conversation_template,
        "block_size": args.block_size,
        "truncation_side": args.truncation_side,
        "min_labels": args.min_labels,
        "require_full_labels": args.require_full_labels,
        "prefer_full_tools": args.prefer_full_tools,
        "base_cap": args.base_cap,
        "public_multicall_cap": args.public_multicall_cap,
        "repair_repeat": args.repair_repeat,
        "repair_variants": args.repair_variants,
        "seed": args.seed,
        "rejected_examples": repair_rejected[:20],
    }
    manifest_path = args.out_dir / "train_agentic_mix.manifest"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
