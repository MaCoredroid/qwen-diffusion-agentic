#!/usr/bin/env python3
import argparse
import copy
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from build_toolcall_argument_curriculum import build_labelaware_originals
from build_toolcall_format_public_mix import (
    DEFAULT_FORMAT_TRAIN,
    DEFAULT_PUBLIC_EVAL,
    DEFAULT_PUBLIC_TEACHER,
    DEFAULT_PUBLIC_TRAIN,
    compact_calls,
    load_jsonl,
    strip_source,
    write_jsonl,
)
from build_toolcall_labelaware_public_mix import (
    DEFAULT_MODEL,
    DEFAULT_TEACHER_HELDOUT_EVAL,
    DEFAULT_TEACHER_TRAIN_EVAL,
    choose_labelaware_variant,
    resolve_chat_template,
    summarize_audit,
    token_stats,
)
from eval_toolcall_jsonl import extract_tool_calls


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_OUT_DIR = ROOT / "data/qwen35_9b_toolcall_model_repair_curriculum"
DEFAULT_SYSTEM = "You are a constrained tool-call repair model."
DEFAULT_HARD_ARGUMENT_CASES = ROOT / "data/toolcall_eval/public_onecall_teacher_train_labelaware_smoke.jsonl"
DEFAULT_HARD_ARGUMENT_OUTPUTS = (
    ROOT
    / "runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300_eval96_modelrepair"
    / "teacher_train_labelaware_12_constrained_max1_rescore.jsonl"
)
DEFAULT_DRAFT_PAIRS = [
    (
        ROOT / "data/toolcall_eval/public_onecall_teacher_train_labelaware_smoke.jsonl",
        ROOT / "runs/fastdllm_qwen35_9b_toolcall_labelaware_argspanw1p5_b896_step300_eval96/teacher_train_labelaware_12.jsonl",
    ),
    (
        ROOT / "data/toolcall_eval/public_onecall_teacher_train_labelaware_smoke.jsonl",
        ROOT / "runs/fastdllm_qwen35_9b_toolcall_labelaware_argspanw2_b896_step300_eval96/teacher_train_labelaware_12.jsonl",
    ),
    (
        ROOT / "data/toolcall_eval/public_onecall_teacher_train_labelaware_smoke.jsonl",
        ROOT / "runs/fastdllm_qwen35_9b_toolcall_labelaware_argspanw3_b896_step300_eval96/teacher_train_labelaware_12.jsonl",
    ),
    (
        ROOT / "data/toolcall_eval/public_onecall_teacher_train_labelaware_smoke.jsonl",
        ROOT / "runs/fastdllm_qwen35_9b_toolcall_labelaware_structw2_b896_step300_eval96/teacher_train_labelaware_12.jsonl",
    ),
    (
        ROOT / "data/toolcall_eval/public_onecall_teacher_train_labelaware_smoke.jsonl",
        ROOT / "runs/fastdllm_qwen35_9b_toolcall_labelaware_structw2_argspanw2_b896_step300_eval96/teacher_train_labelaware_12.jsonl",
    ),
]


def parse_draft_pair(spec):
    left, sep, right = spec.partition(":")
    if not sep or not left or not right:
        raise argparse.ArgumentTypeError("--draft-pair must be cases_jsonl:raw_outputs_jsonl")
    return Path(left), Path(right)


def assistant_target(case):
    calls, invalid = extract_tool_calls(case.get("gold_assistant") or "")
    if invalid or not calls:
        return None
    return compact_calls(calls)


def compact_json_payloads(calls):
    return "\n".join(
        json.dumps({"name": call["name"], "arguments": call.get("arguments") or {}}, ensure_ascii=False, separators=(",", ": "))
        for call in calls
    )


def assistant_target_from_instance(instance):
    assistant_indices = [
        idx
        for idx, message in enumerate(instance.get("messages") or [])
        if message.get("role") == "assistant" and str(message.get("content") or "").strip()
    ]
    if not assistant_indices:
        return None
    last_idx = assistant_indices[-1]
    text = str((instance.get("messages") or [])[last_idx].get("content") or "")
    calls, invalid = extract_tool_calls(text)
    if invalid or not calls:
        return None
    return last_idx, calls, compact_calls(calls)


def prompt_parts_from_case(case):
    system = DEFAULT_SYSTEM
    messages = []
    for message in case.get("prompt_messages") or []:
        role = message.get("role")
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            system = content
        elif role in {"user", "tool", "assistant"}:
            messages.append({"role": role, "content": content})
    return system, messages


def repair_prompt(raw_draft):
    return (
        "Repair the previous assistant draft into valid Qwen tool-call output. "
        "Use the original user request and the available tools. Preserve the "
        "intended tool name and argument values when they are recoverable. "
        "Return only <tool_call> block(s) with JSON payloads and no prose.\n\n"
        "Previous assistant draft:\n"
        f"{raw_draft}"
    )


def clean_repair_draft(calls, target, variant):
    if variant == "json_only":
        return compact_json_payloads(calls)
    if variant == "missing_wrapper":
        return target.replace("<tool_call>\n", "").replace("\n</tool_call>", "")
    if variant == "wrong_arguments_key":
        return target.replace('"arguments"', '"argumentsarguments"', 1)
    if variant == "truncated":
        cut = min(max(80, len(target) // 2), max(1, len(target) - 1), 240)
        return target[:cut]
    if variant == "prose":
        first = calls[0]
        return (
            f"I should call {first['name']} with arguments "
            f"{json.dumps(first.get('arguments') or {}, ensure_ascii=False, separators=(',', ': '))}."
        )
    raise ValueError(f"unknown clean repair variant {variant!r}")


def repair_instance(case, raw_draft, source):
    target = assistant_target(case)
    if not target:
        return None
    system, messages = prompt_parts_from_case(case)
    messages = copy.deepcopy(messages)
    messages.append({"role": "user", "content": repair_prompt(raw_draft)})
    messages.append({"role": "assistant", "content": target})
    return {
        "system": system,
        "tools": copy.deepcopy(case.get("tools") or []),
        "messages": messages,
        "source": source,
    }


def clean_repair_instance(instance, raw_draft, source):
    target_info = assistant_target_from_instance(instance)
    if target_info is None:
        return None
    last_assistant_idx, _, target = target_info
    system = str(instance.get("system") or DEFAULT_SYSTEM).strip() or DEFAULT_SYSTEM
    messages = []
    for idx, message in enumerate(instance.get("messages") or []):
        role = message.get("role")
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            system = content
        elif idx == last_assistant_idx:
            continue
        elif role in {"user", "tool", "assistant"}:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": repair_prompt(raw_draft)})
    messages.append({"role": "assistant", "content": target})
    return {
        "system": system,
        "tools": copy.deepcopy(instance.get("tools") or []),
        "messages": messages,
        "source": source,
    }


def hard_argument_prompt(draft):
    return (
        "Complete and correct the previous tool-call arguments. Use the original "
        "user request and available tool schema as the source of truth. Preserve "
        "the intended tool name. Fill every required argument and copy exact "
        "scalar values. Return only the corrected <tool_call> block(s), with no "
        "prose.\n\n"
        "Previous constrained draft:\n"
        f"{draft}"
    )


def hard_argument_instance(case, draft, source):
    target = assistant_target(case)
    if not target:
        return None
    system, messages = prompt_parts_from_case(case)
    messages = copy.deepcopy(messages)
    messages.append({"role": "user", "content": hard_argument_prompt(draft)})
    messages.append({"role": "assistant", "content": target})
    return {
        "system": system,
        "tools": copy.deepcopy(case.get("tools") or []),
        "messages": messages,
        "source": source,
    }


def load_cases_by_id(path):
    return {case.get("id"): case for case in load_jsonl(path)}


def draft_rows(cases_path, rows_path):
    cases = load_cases_by_id(cases_path)
    out = []
    if not rows_path.exists():
        return out
    for row in load_jsonl(rows_path):
        if row.get("status") != "ok":
            continue
        case = cases.get(row.get("id"))
        raw = str(row.get("assistant") or "").strip()
        if not case or not raw:
            continue
        out.append((case, row, raw))
    return out


def make_repair_candidates(draft_pairs, cap, seed):
    candidates = []
    seen_sources = Counter()
    for cases_path, rows_path in draft_pairs:
        tag = rows_path.parent.name
        for case, row, raw in draft_rows(cases_path, rows_path):
            source = f"model_repair:{tag}:{row.get('idx', len(candidates))}"
            seen_sources[tag] += 1
            instance = repair_instance(case, raw, source)
            if instance is not None:
                candidates.append(instance)
    rng = random.Random(seed + 17)
    rng.shuffle(candidates)
    return candidates[:cap] if cap >= 0 else candidates, seen_sources


def make_clean_repair_candidates(raw_candidates, args):
    families = {item for item in args.clean_repair_sources.split(",") if item}
    variants = [item for item in args.clean_repair_variants.split(",") if item]
    candidates = []
    for instance in raw_candidates:
        family = source_family(instance)
        if families and family not in families:
            continue
        target_info = assistant_target_from_instance(instance)
        if target_info is None:
            continue
        _, calls, target = target_info
        if args.clean_repair_onecall_only and len(calls) != 1:
            continue
        for variant in variants:
            raw_draft = clean_repair_draft(calls, target, variant)
            source = f"clean_model_repair:{family}:{variant}"
            clean_instance = clean_repair_instance(instance, raw_draft, source)
            if clean_instance is not None:
                candidates.append(clean_instance)
    rng = random.Random(args.seed + 23)
    rng.shuffle(candidates)
    if args.clean_repair_cap >= 0:
        candidates = candidates[: args.clean_repair_cap]
    repeated = []
    for repeat_idx in range(max(1, args.clean_repair_repeat)):
        for instance in candidates:
            clone = copy.deepcopy(instance)
            clone["source"] = f"{instance.get('source')}:repeat{repeat_idx}"
            repeated.append(clone)
    return repeated


def make_hard_argument_candidates(args):
    if args.hard_argument_cap == 0:
        return []
    rows_path = args.hard_argument_outputs
    cases_path = args.hard_argument_cases
    if not rows_path.exists() or not cases_path.exists():
        return []
    cases = load_cases_by_id(cases_path)
    candidates = []
    for row in load_jsonl(rows_path):
        if not row.get(f"{args.hard_argument_prefix}_exact_tool_sequence"):
            continue
        if row.get(f"{args.hard_argument_prefix}_exact_arguments"):
            continue
        case = cases.get(row.get("id"))
        draft = str(row.get(f"{args.hard_argument_prefix}_assistant") or "").strip()
        if not case or not draft:
            continue
        source = f"hard_argument:{rows_path.parent.name}:{row.get('idx', len(candidates))}"
        instance = hard_argument_instance(case, draft, source)
        if instance is not None:
            candidates.append(instance)
    rng = random.Random(args.seed + 31)
    rng.shuffle(candidates)
    if args.hard_argument_cap > 0:
        candidates = candidates[: args.hard_argument_cap]
    repeated = []
    for repeat_idx in range(max(1, args.hard_argument_repeat)):
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
    parser.add_argument("--format-train", type=Path, default=DEFAULT_FORMAT_TRAIN)
    parser.add_argument("--public-train", type=Path, default=DEFAULT_PUBLIC_TRAIN)
    parser.add_argument("--public-eval", type=Path, default=DEFAULT_PUBLIC_EVAL)
    parser.add_argument("--public-teacher", type=Path, default=DEFAULT_PUBLIC_TEACHER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--teacher-train-eval-out", type=Path, default=DEFAULT_TEACHER_TRAIN_EVAL)
    parser.add_argument("--teacher-heldout-eval-out", type=Path, default=DEFAULT_TEACHER_HELDOUT_EVAL)
    parser.add_argument("--draft-pair", action="append", type=parse_draft_pair, default=[])
    parser.add_argument("--draft-cap", type=int, default=-1)
    parser.add_argument("--clean-repair-cap", type=int, default=0)
    parser.add_argument("--clean-repair-repeat", type=int, default=1)
    parser.add_argument("--clean-repair-sources", default="public_train_onecall,public_teacher_exact_onecall")
    parser.add_argument(
        "--clean-repair-variants",
        default="json_only,missing_wrapper,wrong_arguments_key,truncated,prose",
    )
    parser.add_argument("--clean-repair-onecall-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hard-argument-cases", type=Path, default=DEFAULT_HARD_ARGUMENT_CASES)
    parser.add_argument("--hard-argument-outputs", type=Path, default=DEFAULT_HARD_ARGUMENT_OUTPUTS)
    parser.add_argument("--hard-argument-prefix", default="constrained")
    parser.add_argument("--hard-argument-cap", type=int, default=0)
    parser.add_argument("--hard-argument-repeat", type=int, default=3)
    parser.add_argument("--include-originals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repair-repeat", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=896)
    parser.add_argument("--truncation-side", choices=["left", "right"], default="right")
    parser.add_argument("--min-labels", type=int, default=1)
    parser.add_argument("--require-full-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefer-full-tools", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--format-cap", type=int, default=96)
    parser.add_argument("--public-train-onecall-cap", type=int, default=40)
    parser.add_argument("--teacher-exact-cap", type=int, default=12)
    parser.add_argument("--heldout-limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=173)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "scripts"))
    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    chat_template = resolve_chat_template(args.conversation_template)

    originals, original_rejected, original_audit, teacher_train_ids, raw_candidates = build_labelaware_originals(
        tokenizer,
        chat_template,
        args,
    )
    draft_pairs = args.draft_pair or DEFAULT_DRAFT_PAIRS
    repair_candidates, draft_source_counts = make_repair_candidates(draft_pairs, args.draft_cap, args.seed)
    repeated_repair_candidates = []
    for repeat_idx in range(max(1, args.repair_repeat)):
        for instance in repair_candidates:
            clone = copy.deepcopy(instance)
            clone["source"] = f"{instance.get('source')}:repeat{repeat_idx}"
            repeated_repair_candidates.append(clone)
    repair_candidates = repeated_repair_candidates
    clean_repair_candidates = make_clean_repair_candidates(raw_candidates, args) if args.clean_repair_cap != 0 else []
    hard_argument_candidates = make_hard_argument_candidates(args)
    repair_candidates.extend(clean_repair_candidates)
    repair_candidates.extend(hard_argument_candidates)
    repair_audit = []
    repair_accepted, repair_rejected = accept_candidates(tokenizer, chat_template, repair_candidates, args, repair_audit)

    accepted = []
    if args.include_originals:
        accepted.extend(originals)
    accepted.extend(repair_accepted)
    instances = accepted
    rng.shuffle(instances)

    public_cases = load_jsonl(args.public_eval)
    teacher_train_eval = [case for case in public_cases if case.get("id") in teacher_train_ids]
    teacher_heldout_eval = [case for case in public_cases if case.get("id") not in teacher_train_ids][: args.heldout_limit]

    chosen_audit = []
    for instance in instances:
        stats = token_stats(tokenizer, chat_template, instance, args.block_size, args.truncation_side)
        chosen_audit.append({"source": instance.get("source") or "unknown", "tool_count": len(instance.get("tools") or []), **stats})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train_agentic_mix.json"
    train_path.write_text(
        json.dumps({"type": "conversation", "instances": [strip_source(item) for item in instances]}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    audit_path = args.out_dir / "train_agentic_mix.audit.jsonl"
    write_jsonl(audit_path, chosen_audit)
    write_jsonl(args.teacher_train_eval_out, teacher_train_eval)
    write_jsonl(args.teacher_heldout_eval_out, teacher_heldout_eval)

    rejected = original_rejected + repair_rejected
    rejected_by_source = defaultdict(int)
    for item in rejected:
        rejected_by_source[item["source"]] += 1
    source_counts = Counter(instance.get("source") or "unknown" for instance in instances)
    source_family_counts = Counter(source_family(instance) for instance in instances)

    manifest = {
        "train_path": str(train_path),
        "audit_path": str(audit_path),
        "count": len(instances),
        "labelaware_original_count": len(originals),
        "repair_candidate_count": len(repair_candidates),
        "clean_repair_candidate_count": len(clean_repair_candidates),
        "hard_argument_candidate_count": len(hard_argument_candidates),
        "repair_accepted_count": len(repair_accepted),
        "accepted_before_dedupe": len(accepted),
        "final_count": len(instances),
        "rejected_count": len(rejected),
        "source_counts": dict(sorted(source_counts.items())),
        "source_family_counts": dict(sorted(source_family_counts.items())),
        "draft_source_counts": dict(sorted(draft_source_counts.items())),
        "rejected_by_source": dict(sorted(rejected_by_source.items())),
        "tokenizer_model": str(args.model),
        "conversation_template": args.conversation_template,
        "block_size": args.block_size,
        "truncation_side": args.truncation_side,
        "min_labels": args.min_labels,
        "require_full_labels": args.require_full_labels,
        "prefer_full_tools": args.prefer_full_tools,
        "include_originals": args.include_originals,
        "repair_repeat": args.repair_repeat,
        "clean_repair_cap": args.clean_repair_cap,
        "clean_repair_repeat": args.clean_repair_repeat,
        "clean_repair_sources": args.clean_repair_sources,
        "clean_repair_variants": args.clean_repair_variants,
        "clean_repair_onecall_only": args.clean_repair_onecall_only,
        "hard_argument_cases": str(args.hard_argument_cases),
        "hard_argument_outputs": str(args.hard_argument_outputs),
        "hard_argument_prefix": args.hard_argument_prefix,
        "hard_argument_cap": args.hard_argument_cap,
        "hard_argument_repeat": args.hard_argument_repeat,
        "draft_pairs": [[str(a), str(b)] for a, b in draft_pairs],
        "chosen_audit_summary": summarize_audit(chosen_audit, chosen_audit),
        "original_candidate_audit_summary": summarize_audit(original_audit, chosen_audit),
        "repair_candidate_audit_summary": summarize_audit(repair_audit, chosen_audit),
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
