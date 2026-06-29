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
from build_toolcall_multicall_gap_curriculum import assistant_text, user_text
from audit_toolcall_eval_overlap import eval_records, fingerprint
from eval_toolcall_jsonl import extract_tool_calls, score_tool_calls
from rescore_toolcall_sequence_planner_projection import planned_call_text


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_MODEL = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_OUT_DIR = ROOT / "data/qwen35_9b_toolcall_sequence_planner_distill_curriculum"

PLANNER_SYSTEM = "You are a constrained Qwen multi-call planning model."


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


def filter_eval_overlaps(public_items, eval_jsonl_paths):
    if not eval_jsonl_paths:
        return public_items, []
    eval_fingerprints = {row["fingerprint"] for row in eval_records(eval_jsonl_paths)}
    kept = []
    removed = []
    for idx, (instance, calls) in enumerate(public_items):
        row_fp = fingerprint(user_text(instance), assistant_text(instance))
        if row_fp in eval_fingerprints:
            removed.append(
                {
                    "idx": idx,
                    "source": instance.get("source") or "unknown",
                    "fingerprint": row_fp,
                    "gold_names": [call.get("name") for call in calls],
                    "user_excerpt": " ".join(user_text(instance).split())[:220],
                }
            )
        else:
            kept.append((instance, calls))
    return kept, removed


def planner_case(instance):
    return {
        "prompt_messages": [
            {"role": "user", "content": user_text(instance)},
        ],
        "tools": copy.deepcopy(instance.get("tools") or []),
        "gold_assistant": assistant_text(instance),
    }


def compact_schema(schema):
    if not isinstance(schema, dict):
        return {}
    out = {}
    expected = schema.get("type")
    if expected is not None:
        out["type"] = expected
    if schema.get("enum") is not None:
        out["enum"] = copy.deepcopy(schema["enum"])
    if schema.get("required") is not None:
        out["required"] = copy.deepcopy(schema["required"])
    props = schema.get("properties")
    if isinstance(props, dict):
        out["properties"] = {name: compact_schema(prop_schema) for name, prop_schema in props.items()}
    items = schema.get("items")
    if isinstance(items, dict):
        out["items"] = compact_schema(items)
    return out


def compact_tool(tool):
    if not isinstance(tool, dict):
        return tool
    fn = tool.get("function", tool)
    if not isinstance(fn, dict):
        return copy.deepcopy(tool)
    compact_fn = {"name": fn.get("name")}
    if fn.get("description"):
        compact_fn["description"] = str(fn["description"])[:160]
    if isinstance(fn.get("parameters"), dict):
        compact_fn["parameters"] = compact_schema(fn["parameters"])
    if "function" in tool:
        return {"type": tool.get("type", "function"), "function": compact_fn}
    return compact_fn


def tools_for_mode(instance, tool_schema_mode):
    tools = copy.deepcopy(instance.get("tools") or [])
    if tool_schema_mode == "compact":
        return [compact_tool(tool) for tool in tools]
    return tools


def planner_prompt(request, prompt_mode):
    if prompt_mode == "request_only":
        return request
    return (
        "Return the exact Qwen tool calls required by the request. "
        "Use request list/table order and the available tool schemas. "
        "Copy argument values from the request. Return only <tool_call> blocks "
        "with JSON payloads and no prose.\n\n"
        "Request:\n"
        f"{request}"
    )


def planner_candidate(instance, assistant_target, accept_mode, tool_schema_mode, prompt_mode):
    request = user_text(instance)
    return {
        "system": PLANNER_SYSTEM,
        "tools": tools_for_mode(instance, tool_schema_mode),
        "messages": [
            {"role": "user", "content": planner_prompt(request, prompt_mode)},
            {"role": "assistant", "content": assistant_target},
        ],
        "source": (
            f"{instance.get('source') or 'public_train'}:"
            f"sequence_planner_distill_{accept_mode}_{tool_schema_mode}_{prompt_mode}"
        ),
    }


def exact_planner_candidates(public_items, prefer_segment_args, accept_mode, tool_schema_modes, prompt_mode):
    accepted = []
    rejected = []
    exact_sequence_count = 0
    exact_argument_count = 0
    for idx, (instance, calls) in enumerate(public_items):
        case = planner_case(instance)
        planned_text, segment_audit, planned_names = planned_call_text(
            case,
            "",
            prefer_segment_args=prefer_segment_args,
        )
        gold_text = compact_calls(calls)
        metrics = score_tool_calls(planned_text, instance.get("tools") or [], gold_text)
        exact_sequence = bool(metrics.get("exact_tool_sequence"))
        exact_arguments = bool(metrics.get("exact_arguments"))
        exact_sequence_count += int(exact_sequence)
        exact_argument_count += int(exact_arguments)
        row = {
            "idx": idx,
            "source": instance.get("source") or "unknown",
            "gold_call_count": len(calls),
            "planned_call_count": len(planned_names),
            "planned_names": planned_names,
            "gold_names": [call.get("name") for call in calls],
            "exact_tool_sequence": exact_sequence,
            "exact_arguments": exact_arguments,
            "all_schema_valid": bool(metrics.get("all_schema_valid")),
            "all_required_args_present": bool(metrics.get("all_required_args_present")),
            "extra_call_count": metrics.get("extra_call_count"),
            "missing_call_count": metrics.get("missing_call_count"),
            "repeated_call_count": metrics.get("repeated_call_count"),
            "segment_audit": segment_audit,
        }
        accept = exact_sequence if accept_mode == "exact_sequence" else exact_sequence and exact_arguments
        if accept:
            target = gold_text if accept_mode == "exact_sequence" else planned_text
            for tool_schema_mode in tool_schema_modes:
                accepted.append((planner_candidate(instance, target, accept_mode, tool_schema_mode, prompt_mode), row))
        else:
            rejected.append(row)
    return accepted, rejected, {"exact_sequence": exact_sequence_count, "exact_arguments": exact_argument_count}


def accept_labelaware(tokenizer, chat_template, candidates, args):
    audit_rows = []
    accepted = []
    rejected = []
    for instance, planner_audit in candidates:
        chosen, scored = choose_labelaware_variant(tokenizer, chat_template, instance, args, audit_rows)
        if chosen is None:
            rejected.append(
                {
                    "source": instance.get("source") or "unknown",
                    "planner_audit": planner_audit,
                    "tool_count": len(instance.get("tools") or []),
                    "candidate_stats": [
                        {"variant": variant, "tool_count": len(candidate.get("tools") or []), **stats}
                        for variant, candidate, stats in scored
                    ],
                }
            )
        else:
            accepted.append(chosen)
    return accepted, rejected, audit_rows


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
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=896)
    parser.add_argument("--truncation-side", choices=["left", "right"], default="right")
    parser.add_argument("--min-labels", type=int, default=1)
    parser.add_argument("--require-full-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefer-full-tools", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefer-segment-args", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--accept-mode", choices=["exact_sequence", "exact_arguments"], default="exact_sequence")
    parser.add_argument("--tool-schema-mode", choices=["full", "compact", "both"], default="full")
    parser.add_argument("--prompt-mode", choices=["instruction", "request_only"], default="instruction")
    parser.add_argument("--exclude-eval-jsonl", type=Path, nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=613)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "scripts"))
    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    chat_template = resolve_chat_template(args.conversation_template)

    raw_public_multicall = load_public_multicall(args.public_train, args.public_multicall_cap, rng)
    public_multicall, eval_overlap_removed = filter_eval_overlaps(raw_public_multicall, args.exclude_eval_jsonl)
    tool_schema_modes = ["full", "compact"] if args.tool_schema_mode == "both" else [args.tool_schema_mode]
    exact_candidates, planner_rejected, planner_totals = exact_planner_candidates(
        public_multicall,
        args.prefer_segment_args,
        args.accept_mode,
        tool_schema_modes,
        args.prompt_mode,
    )

    repeated = []
    for repeat_idx in range(max(1, args.repeat)):
        for instance, planner_audit in exact_candidates:
            clone = copy.deepcopy(instance)
            clone["source"] = f"{instance.get('source')}:repeat{repeat_idx}"
            repeated.append((clone, planner_audit))

    accepted, label_rejected, candidate_audit = accept_labelaware(tokenizer, chat_template, repeated, args)
    instances = dedupe(accepted)
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
    planner_reject_path = args.out_dir / "planner_rejected.jsonl"
    write_jsonl(planner_reject_path, planner_rejected)
    label_reject_path = args.out_dir / "label_rejected.jsonl"
    write_jsonl(label_reject_path, label_rejected)

    source_counts = Counter(instance.get("source") or "unknown" for instance in instances)
    source_family_counts = Counter(source_family(instance) for instance in instances)
    rejected_by_source = defaultdict(int)
    for item in label_rejected:
        rejected_by_source[item["source"]] += 1

    manifest = {
        "train_path": str(train_path),
        "audit_path": str(audit_path),
        "planner_reject_path": str(planner_reject_path),
        "label_reject_path": str(label_reject_path),
        "count": len(instances),
        "public_train": str(args.public_train),
        "exclude_eval_jsonl": [str(path) for path in args.exclude_eval_jsonl],
        "raw_public_multicall_records": len(raw_public_multicall),
        "eval_overlap_removed_count": len(eval_overlap_removed),
        "eval_overlap_removed_examples": eval_overlap_removed[:20],
        "no_eval_leakage": bool(args.exclude_eval_jsonl),
        "public_multicall_records": len(public_multicall),
        "planner_candidates": len(public_multicall),
        "planner_exact_sequence": planner_totals["exact_sequence"],
        "planner_exact_arguments": planner_totals["exact_arguments"],
        "accept_mode": args.accept_mode,
        "accepted_planner_selected": len(exact_candidates),
        "rejected_not_exact": len(planner_rejected),
        "repeated_candidate_count": len(repeated),
        "accepted_before_dedupe": len(accepted),
        "deduped_accepted_count": len(instances),
        "label_rejected_count": len(label_rejected),
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
        "prefer_segment_args": args.prefer_segment_args,
        "tool_schema_mode": args.tool_schema_mode,
        "tool_schema_modes": tool_schema_modes,
        "prompt_mode": args.prompt_mode,
        "target_source": "gold_assistant" if args.accept_mode == "exact_sequence" else "planned_text",
        "public_multicall_cap": args.public_multicall_cap,
        "repeat": args.repeat,
        "seed": args.seed,
        "planner_rejected_examples": planner_rejected[:20],
        "label_rejected_examples": label_rejected[:20],
    }
    manifest_path = args.out_dir / "train_agentic_mix.manifest"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
