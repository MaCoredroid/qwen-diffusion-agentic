#!/usr/bin/env python3
import argparse
import copy
import json
import random
import re
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
DEFAULT_OUT_DIR = ROOT / "data/qwen35_9b_toolcall_multicall_scalar_curriculum"

SCALAR_SYSTEM = "You are a precise one-call tool argument extraction model."


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


def tool_subset(instance, name):
    tools = [copy.deepcopy(tool) for tool in instance.get("tools") or [] if tool_name(tool) == name]
    return tools or copy.deepcopy(instance.get("tools") or [])


def function_schema(tool):
    fn = tool.get("function", tool) if isinstance(tool, dict) else {}
    params = fn.get("parameters") if isinstance(fn, dict) else {}
    return params if isinstance(params, dict) else {}


def tool_for_name(instance, name):
    for tool in instance.get("tools") or []:
        if tool_name(tool) == name:
            return tool
    return None


def compact_one_call(name, arguments):
    return compact_calls([{"name": name, "arguments": arguments}])


def multicall_from_instance(instance):
    calls, invalid = extract_tool_calls(assistant_text(instance))
    if invalid or len(calls) < 2:
        return None
    if any(not call.get("name") for call in calls):
        return None
    return instance, calls


def load_public_multicall(path, cap, rng):
    candidates = []
    for raw in load_conversation(path):
        instance = normalized_instance(raw, "public_train_multicall")
        parsed = multicall_from_instance(instance)
        if parsed is not None:
            candidates.append(parsed)
    rng.shuffle(candidates)
    return candidates[:cap] if cap >= 0 else candidates


def scalar_top_level_props(arguments, tool):
    if not isinstance(arguments, dict):
        return []
    schema = function_schema(tool or {})
    properties = schema.get("properties") or {}
    out = []
    for prop, value in arguments.items():
        prop_schema = properties.get(prop) if isinstance(properties, dict) else {}
        expected = prop_schema.get("type") if isinstance(prop_schema, dict) else None
        if isinstance(expected, list):
            expected = next((item for item in expected if item != "null"), expected[0] if expected else None)
        if isinstance(value, (str, int, float, bool)) or value is None:
            if expected not in {"array", "object"}:
                out.append(prop)
    return out


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


def draft_arguments(arguments, prop, variant):
    draft = copy.deepcopy(arguments) if isinstance(arguments, dict) else {}
    if variant == "empty_args":
        return {}
    if variant == "missing_field":
        draft.pop(prop, None)
        return draft
    if variant == "wrong_scalar":
        draft[prop] = wrong_scalar(draft.get(prop))
        return draft
    if variant == "null_field":
        draft[prop] = None
        return draft
    raise ValueError(f"unknown scalar repair variant {variant!r}")


def value_needles(value):
    needles = []
    if isinstance(value, bool):
        needles.append(str(value).lower())
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        needles.append(str(value))
        if isinstance(value, float) and value.is_integer():
            needles.append(str(int(value)))
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped:
            needles.extend([stripped, stripped.replace("_", " "), stripped.replace("-", " ")])
            if stripped.endswith("Z"):
                needles.append(stripped[:-1])
            if "T" in stripped:
                date_part, _, time_part = stripped.partition("T")
                needles.append(date_part)
                if time_part:
                    needles.append(time_part.rstrip("Z"))
                    needles.append(time_part[:5])
    return [needle for needle in dict.fromkeys(needles) if needle]


def merge_spans(spans):
    if not spans:
        return []
    spans = sorted(spans)
    merged = [list(spans[0])]
    for start, end in spans[1:]:
        current = merged[-1]
        if start <= current[1] + 32:
            current[1] = max(current[1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def segment_span_for_call(text, call_index, call_count, radius):
    if call_count <= 0:
        return 0, min(len(text), radius)
    center = int((call_index + 0.5) * len(text) / call_count)
    return max(0, center - radius // 2), min(len(text), center + radius // 2)


def request_excerpt(request, call, call_index, call_count, max_chars):
    radius = max(240, max_chars // 2)
    spans = []
    arguments = call.get("arguments") or {}
    for value in arguments.values() if isinstance(arguments, dict) else []:
        for needle in value_needles(value):
            for match in re.finditer(re.escape(needle), request, flags=re.IGNORECASE):
                spans.append((max(0, match.start() - radius // 3), min(len(request), match.end() + radius // 3)))
    if not spans:
        spans.append(segment_span_for_call(request, call_index, call_count, radius))
    chunks = [request[start:end].strip() for start, end in merge_spans(spans)]
    excerpt = "\n...\n".join(chunk for chunk in chunks if chunk)
    if len(excerpt) > max_chars:
        excerpt = excerpt[: max_chars - 3].rstrip() + "..."
    return excerpt or request[:max_chars].strip()


def scalar_repair_prompt(name, prop, draft, excerpt):
    return (
        "Repair exactly one tool call. Keep the function name unchanged. "
        "Copy exact argument values from the request excerpt and tool schema. "
        f"Focus especially on the `{prop}` argument. Return one corrected Qwen "
        "<tool_call> block with JSON payload and no prose.\n\n"
        "Request excerpt:\n"
        f"{excerpt}\n\n"
        "Draft call:\n"
        f"{draft}"
    )


def scalar_instance(instance, calls, call_index, prop, variant, max_excerpt_chars):
    call = calls[call_index]
    name = call["name"]
    arguments = copy.deepcopy(call.get("arguments") or {})
    draft = compact_one_call(name, draft_arguments(arguments, prop, variant))
    excerpt = request_excerpt(user_text(instance), call, call_index, len(calls), max_excerpt_chars)
    return {
        "system": SCALAR_SYSTEM,
        "tools": tool_subset(instance, name),
        "messages": [
            {
                "role": "user",
                "content": scalar_repair_prompt(name, prop, draft, excerpt),
            },
            {"role": "assistant", "content": compact_one_call(name, arguments)},
        ],
        "source": f"{instance.get('source') or 'public_train'}:scalar_{variant}:call{call_index}:{prop}",
    }


def make_scalar_candidates(public_items, args):
    variants = [item for item in args.scalar_variants.split(",") if item]
    candidates = []
    skipped = Counter()
    for instance, calls in public_items:
        for call_index, call in enumerate(calls):
            tool = tool_for_name(instance, call.get("name"))
            props = scalar_top_level_props(call.get("arguments") or {}, tool)
            if not props:
                skipped["no_scalar_props"] += 1
                continue
            for prop in props:
                for variant in variants:
                    candidates.append(scalar_instance(instance, calls, call_index, prop, variant, args.max_excerpt_chars))
    repeated = []
    for repeat_idx in range(max(1, args.scalar_repeat)):
        for instance in candidates:
            clone = copy.deepcopy(instance)
            clone["source"] = f"{instance.get('source')}:repeat{repeat_idx}"
            repeated.append(clone)
    return repeated, skipped


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
    parts = (instance.get("source") or "unknown").split(":")
    return ":".join(parts[:2]) if len(parts) >= 2 else parts[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--conversation-template", default="fast_dllm_v2")
    parser.add_argument("--public-train", type=Path, default=DEFAULT_PUBLIC_TRAIN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--public-multicall-cap", type=int, default=-1)
    parser.add_argument("--scalar-repeat", type=int, default=1)
    parser.add_argument("--scalar-variants", default="empty_args,missing_field,wrong_scalar,null_field")
    parser.add_argument("--max-excerpt-chars", type=int, default=900)
    parser.add_argument("--block-size", type=int, default=896)
    parser.add_argument("--truncation-side", choices=["left", "right"], default="right")
    parser.add_argument("--min-labels", type=int, default=1)
    parser.add_argument("--require-full-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefer-full-tools", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=421)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "scripts"))
    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    chat_template = resolve_chat_template(args.conversation_template)

    public_multicall = load_public_multicall(args.public_train, args.public_multicall_cap, rng)
    candidates, skipped = make_scalar_candidates(public_multicall, args)
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
        "skipped_counts": dict(sorted(skipped.items())),
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
        "scalar_repeat": args.scalar_repeat,
        "scalar_variants": args.scalar_variants,
        "max_excerpt_chars": args.max_excerpt_chars,
        "seed": args.seed,
        "rejected_examples": rejected[:20],
    }
    manifest_path = args.out_dir / "train_agentic_mix.manifest"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
