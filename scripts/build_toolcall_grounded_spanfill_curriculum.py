#!/usr/bin/env python3
import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from build_toolcall_format_public_mix import compact_calls, load_jsonl, strip_source, write_jsonl
from build_toolcall_labelaware_public_mix import choose_labelaware_variant, resolve_chat_template, summarize_audit, token_stats
from eval_toolcall_jsonl import extract_tool_calls


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_MODEL = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_CASES = ROOT / "data/toolcall_eval/public_onecall_teacher_train_labelaware_smoke.jsonl"
DEFAULT_RAW_OUTPUTS = (
    ROOT
    / "runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_eval96_modelrepair_max1"
    / "teacher_train_labelaware_12.jsonl"
)
DEFAULT_GROUNDED_OUTPUTS = (
    ROOT
    / "runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_eval96_modelrepair_max1"
    / "teacher_train_labelaware_12_grounded_projection_v2.jsonl"
)
DEFAULT_OUT_DIR = ROOT / "data/qwen35_9b_toolcall_grounded_spanfill_teacher_train_curriculum"
GROUNDING_SYSTEM = "You are a grounded tool-call argument repair model."


def cases_by_id(path):
    return {case.get("id"): case for case in load_jsonl(path)}


def rows_by_id(path):
    rows = {}
    if not path.exists():
        return rows
    for row in load_jsonl(path):
        row_id = row.get("id")
        if row_id:
            rows[row_id] = row
    return rows


def prompt_parts_from_case(case):
    system = GROUNDING_SYSTEM
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


def compact_target(text):
    calls, invalid = extract_tool_calls(text or "")
    if invalid or not calls:
        return None
    return compact_calls(calls)


def grounded_spanfill_prompt(raw_draft, constrained_draft=""):
    parts = [
        "Repair the previous assistant draft into the exact grounded Qwen tool call.",
        "Use the original user request and available tool schema as the source of truth.",
        "Copy request-evidence values exactly, fill required arguments, and return only valid <tool_call> block(s).",
        "",
        "Previous assistant draft:",
        raw_draft or "",
    ]
    if constrained_draft:
        parts.extend(["", "Previous constrained projection:", constrained_draft])
    return "\n".join(parts)


def build_instance(case, raw_row, grounded_row, source):
    target = compact_target(grounded_row.get("constrained_assistant") or "")
    if target is None:
        return None
    raw_draft = str((raw_row or grounded_row).get("assistant") or "").strip()
    constrained_draft = str((raw_row or {}).get("constrained_assistant") or "").strip()
    if not raw_draft and not constrained_draft:
        return None
    system, messages = prompt_parts_from_case(case)
    messages = copy.deepcopy(messages)
    messages.append({"role": "user", "content": grounded_spanfill_prompt(raw_draft, constrained_draft)})
    messages.append({"role": "assistant", "content": target})
    return {
        "system": system,
        "tools": copy.deepcopy(case.get("tools") or []),
        "messages": messages,
        "source": source,
    }


def make_candidates(args):
    cases = cases_by_id(args.cases_jsonl)
    raw_rows = rows_by_id(args.raw_outputs_jsonl)
    grounded_rows = rows_by_id(args.grounded_outputs_jsonl)
    candidates = []
    skipped = Counter()
    for row_id, grounded_row in grounded_rows.items():
        if args.require_grounded_exact and not grounded_row.get("constrained_exact_arguments"):
            skipped["not_grounded_exact"] += 1
            continue
        raw_row = raw_rows.get(row_id)
        if args.only_improved and raw_row and raw_row.get("constrained_exact_arguments"):
            skipped["already_exact"] += 1
            continue
        case = cases.get(row_id)
        if not case:
            skipped["missing_case"] += 1
            continue
        source = f"grounded_spanfill:{args.grounded_outputs_jsonl.parent.name}:{grounded_row.get('idx', len(candidates))}"
        instance = build_instance(case, raw_row, grounded_row, source)
        if instance is None:
            skipped["no_instance"] += 1
            continue
        candidates.append(instance)
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    if args.cap >= 0:
        candidates = candidates[: args.cap]
    repeated = []
    for repeat_idx in range(max(1, args.repeat)):
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
    return (instance.get("source") or "unknown").split(":")[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--conversation-template", default="fast_dllm_v2")
    parser.add_argument("--cases-jsonl", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--raw-outputs-jsonl", type=Path, default=DEFAULT_RAW_OUTPUTS)
    parser.add_argument("--grounded-outputs-jsonl", type=Path, default=DEFAULT_GROUNDED_OUTPUTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--cap", type=int, default=-1)
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--require-grounded-exact", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--only-improved", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--block-size", type=int, default=896)
    parser.add_argument("--truncation-side", choices=["left", "right"], default="right")
    parser.add_argument("--min-labels", type=int, default=1)
    parser.add_argument("--require-full-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefer-full-tools", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=197)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    chat_template = resolve_chat_template(args.conversation_template)
    candidates, skipped = make_candidates(args)
    audit_rows = []
    accepted, rejected = accept_candidates(tokenizer, chat_template, candidates, args, audit_rows)
    rng = random.Random(args.seed + 1)
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
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "skipped_counts": dict(sorted(skipped.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "source_family_counts": dict(sorted(source_family_counts.items())),
        "rejected_by_source": dict(sorted(rejected_by_source.items())),
        "tokenizer_model": str(args.model),
        "conversation_template": args.conversation_template,
        "cases_jsonl": str(args.cases_jsonl),
        "raw_outputs_jsonl": str(args.raw_outputs_jsonl),
        "grounded_outputs_jsonl": str(args.grounded_outputs_jsonl),
        "block_size": args.block_size,
        "truncation_side": args.truncation_side,
        "min_labels": args.min_labels,
        "require_full_labels": args.require_full_labels,
        "prefer_full_tools": args.prefer_full_tools,
        "cap": args.cap,
        "repeat": args.repeat,
        "require_grounded_exact": args.require_grounded_exact,
        "only_improved": args.only_improved,
        "chosen_audit_summary": summarize_audit(chosen_audit, chosen_audit),
        "candidate_audit_summary": summarize_audit(audit_rows, chosen_audit),
        "seed": args.seed,
        "rejected_examples": rejected[:20],
    }
    manifest_path = args.out_dir / "train_agentic_mix.manifest"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
