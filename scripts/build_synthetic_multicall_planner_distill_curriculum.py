#!/usr/bin/env python3
import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from audit_toolcall_eval_overlap import eval_records, fingerprint
from build_toolcall_format_public_mix import compact_calls, strip_source, write_jsonl
from build_toolcall_labelaware_public_mix import (
    choose_labelaware_variant,
    resolve_chat_template,
    summarize_audit,
    token_stats,
)
from build_toolcall_sequence_planner_distill_curriculum import (
    PLANNER_SYSTEM,
    planner_prompt,
    tools_for_mode,
)
from eval_toolcall_jsonl import extract_tool_calls, score_tool_calls


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_MODEL = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_CASES = ROOT / "data/toolcall_eval/synthetic_multicall_failure_analogues.jsonl"
DEFAULT_PLANNER_ROWS = ROOT / "runs/synthetic_multicall_failure_analogues/sequence_planner_bad_draft_safe_seqmismatch.jsonl"
DEFAULT_OUT_DIR = ROOT / "data/qwen35_9b_synthetic_multicall_planner_distill_curriculum"


def load_jsonl(path):
    if not path or not Path(path).exists():
        return []
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def case_key(row, fallback):
    return row.get("id") or row.get("case_id") or str(fallback)


def user_text(case):
    return "\n\n".join(
        str(message.get("content") or "").strip()
        for message in case.get("prompt_messages") or []
        if message.get("role") == "user" and str(message.get("content") or "").strip()
    )


def target_text(case, planner_row, target_source, planner_text_field):
    if target_source == "gold":
        return str(case.get("gold_assistant") or "")
    if target_source == "planner":
        return str((planner_row or {}).get(planner_text_field) or "")
    if target_source == "planner_or_gold":
        value = str((planner_row or {}).get(planner_text_field) or "")
        return value if value.strip() else str(case.get("gold_assistant") or "")
    raise ValueError(f"unsupported target source {target_source!r}")


def normalize_target(text):
    calls, invalid = extract_tool_calls(text)
    if invalid:
        return text
    return compact_calls(calls)


def source_family(case):
    return case.get("analogue_family") or case.get("source") or "unknown"


def instance_from_case(case, target, tool_schema_mode, prompt_mode, repeat_idx):
    request = user_text(case)
    return {
        "system": PLANNER_SYSTEM,
        "tools": tools_for_mode(case, tool_schema_mode),
        "messages": [
            {"role": "user", "content": planner_prompt(request, prompt_mode)},
            {"role": "assistant", "content": target},
        ],
        "source": (
            f"synthetic_multicall_planner:{source_family(case)}:"
            f"{case.get('id') or 'unknown'}:{tool_schema_mode}:{prompt_mode}:repeat{repeat_idx}"
        ),
    }


def target_metrics(case, target):
    return score_tool_calls(target, case.get("tools") or [], case.get("gold_assistant"))


def overlap_rows(instances, eval_jsonl_paths):
    if not eval_jsonl_paths:
        return []
    eval_fingerprints = {row["fingerprint"] for row in eval_records(eval_jsonl_paths)}
    overlaps = []
    for idx, instance in enumerate(instances):
        messages = instance.get("messages") or []
        request = next((message.get("content") or "" for message in messages if message.get("role") == "user"), "")
        assistant = "\n".join(str(message.get("content") or "") for message in messages if message.get("role") == "assistant")
        row_fp = fingerprint(request, assistant)
        if row_fp in eval_fingerprints:
            overlaps.append(
                {
                    "idx": idx,
                    "source": instance.get("source") or "unknown",
                    "fingerprint": row_fp,
                    "user_excerpt": " ".join(str(request).split())[:220],
                }
            )
    return overlaps


def build_raw_candidates(cases, planner_rows_by_key, args):
    candidates = []
    rejected = []
    target_metrics_rows = []
    tool_schema_modes = ["full", "compact"] if args.tool_schema_mode == "both" else [args.tool_schema_mode]
    for idx, case in enumerate(cases):
        key = case_key(case, idx)
        target = normalize_target(
            target_text(case, planner_rows_by_key.get(key), args.target_source, args.planner_text_field)
        )
        metrics = target_metrics(case, target)
        metric_row = {
            "idx": idx,
            "id": key,
            "family": source_family(case),
            "target_source": args.target_source,
            "exact_tool_sequence": bool(metrics.get("exact_tool_sequence")),
            "exact_arguments": bool(metrics.get("exact_arguments")),
            "all_schema_valid": bool(metrics.get("all_schema_valid")),
            "all_required_args_present": bool(metrics.get("all_required_args_present")),
            "called_names": metrics.get("called_names") or [],
            "gold_names": metrics.get("gold_called_names") or [],
        }
        target_metrics_rows.append(metric_row)
        if args.accept_mode == "exact_arguments":
            accept = metric_row["exact_tool_sequence"] and metric_row["exact_arguments"]
        elif args.accept_mode == "exact_sequence":
            accept = metric_row["exact_tool_sequence"]
        else:
            raise ValueError(f"unsupported accept mode {args.accept_mode!r}")
        if not accept:
            rejected.append(metric_row)
            continue
        for repeat_idx in range(max(1, args.repeat)):
            for tool_schema_mode in tool_schema_modes:
                candidates.append(instance_from_case(case, target, tool_schema_mode, args.prompt_mode, repeat_idx))
    return candidates, rejected, target_metrics_rows


def labelaware_filter(tokenizer, chat_template, instances, args):
    audit_rows = []
    accepted = []
    rejected = []
    for instance in instances:
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
    return accepted, rejected, audit_rows


def dedupe(instances):
    seen = set()
    out = []
    for instance in instances:
        key = json.dumps(strip_source(instance), sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        out.append(instance)
    return out


def source_family_from_instance(instance):
    parts = (instance.get("source") or "unknown").split(":")
    return ":".join(parts[:2]) if len(parts) >= 2 else parts[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--conversation-template", default="fast_dllm_v2")
    parser.add_argument("--cases-jsonl", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--planner-jsonl", type=Path, default=DEFAULT_PLANNER_ROWS)
    parser.add_argument("--planner-text-field", default="sequence_planner_assistant")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target-source", choices=["planner", "gold", "planner_or_gold"], default="planner")
    parser.add_argument("--accept-mode", choices=["exact_sequence", "exact_arguments"], default="exact_arguments")
    parser.add_argument("--tool-schema-mode", choices=["full", "compact", "both"], default="compact")
    parser.add_argument("--prompt-mode", choices=["instruction", "request_only"], default="instruction")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--dedupe", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--truncation-side", choices=["left", "right"], default="right")
    parser.add_argument("--min-labels", type=int, default=1)
    parser.add_argument("--require-full-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefer-full-tools", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exclude-eval-jsonl", type=Path, nargs="*", default=[])
    parser.add_argument("--contains-eval-slice", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--diagnostic-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    chat_template = resolve_chat_template(args.conversation_template)
    cases = load_jsonl(args.cases_jsonl)
    planner_rows = load_jsonl(args.planner_jsonl)
    planner_rows_by_key = {case_key(row, idx): row for idx, row in enumerate(planner_rows)}

    raw_candidates, target_rejected, target_metrics_rows = build_raw_candidates(cases, planner_rows_by_key, args)
    accepted, label_rejected, candidate_audit = labelaware_filter(tokenizer, chat_template, raw_candidates, args)
    deduped_instances = dedupe(accepted)
    instances = deduped_instances if args.dedupe else list(accepted)
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

    overlaps = overlap_rows(instances, args.exclude_eval_jsonl)
    source_counts = Counter(instance.get("source") or "unknown" for instance in instances)
    source_family_counts = Counter(source_family_from_instance(instance) for instance in instances)
    target_family_counts = Counter(row["family"] for row in target_metrics_rows)
    exact_by_family = defaultdict(lambda: {"records": 0, "exact_sequence": 0, "exact_arguments": 0})
    for row in target_metrics_rows:
        family = row["family"]
        exact_by_family[family]["records"] += 1
        exact_by_family[family]["exact_sequence"] += int(row["exact_tool_sequence"])
        exact_by_family[family]["exact_arguments"] += int(row["exact_arguments"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train_agentic_mix.json"
    train_path.write_text(
        json.dumps({"type": "conversation", "instances": [strip_source(item) for item in instances]}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    audit_path = args.out_dir / "train_agentic_mix.audit.jsonl"
    write_jsonl(audit_path, chosen_audit)
    candidate_audit_path = args.out_dir / "candidate_audit.jsonl"
    write_jsonl(candidate_audit_path, candidate_audit)
    target_metrics_path = args.out_dir / "target_metrics.jsonl"
    write_jsonl(target_metrics_path, target_metrics_rows)
    target_reject_path = args.out_dir / "target_rejected.jsonl"
    write_jsonl(target_reject_path, target_rejected)
    label_reject_path = args.out_dir / "label_rejected.jsonl"
    write_jsonl(label_reject_path, label_rejected)

    manifest = {
        "train_path": str(train_path),
        "audit_path": str(audit_path),
        "candidate_audit_path": str(candidate_audit_path),
        "target_metrics_path": str(target_metrics_path),
        "target_reject_path": str(target_reject_path),
        "label_reject_path": str(label_reject_path),
        "cases_jsonl": str(args.cases_jsonl),
        "planner_jsonl": str(args.planner_jsonl),
        "planner_text_field": args.planner_text_field,
        "count": len(instances),
        "raw_cases": len(cases),
        "planner_rows": len(planner_rows),
        "raw_candidate_count": len(raw_candidates),
        "accepted_before_dedupe": len(accepted),
        "deduped_accepted_count": len(deduped_instances),
        "dedupe": args.dedupe,
        "target_rejected_count": len(target_rejected),
        "label_rejected_count": len(label_rejected),
        "target_source": args.target_source,
        "accept_mode": args.accept_mode,
        "target_family_counts": dict(sorted(target_family_counts.items())),
        "target_exact_by_family": dict(sorted((key, dict(value)) for key, value in exact_by_family.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "source_family_counts": dict(sorted(source_family_counts.items())),
        "exclude_eval_jsonl": [str(path) for path in args.exclude_eval_jsonl],
        "eval_overlap_count": len(overlaps),
        "eval_overlap_examples": overlaps[:20],
        "no_eval_leakage": not overlaps,
        "chosen_audit_summary": summarize_audit(chosen_audit, chosen_audit),
        "candidate_audit_summary": summarize_audit(candidate_audit, chosen_audit),
        "tokenizer_model": str(args.model),
        "conversation_template": args.conversation_template,
        "block_size": args.block_size,
        "truncation_side": args.truncation_side,
        "min_labels": args.min_labels,
        "require_full_labels": args.require_full_labels,
        "prefer_full_tools": args.prefer_full_tools,
        "tool_schema_mode": args.tool_schema_mode,
        "prompt_mode": args.prompt_mode,
        "repeat": args.repeat,
        "seed": args.seed,
        "diagnostic_only": bool(args.diagnostic_only),
        "contains_eval_slice": bool(args.contains_eval_slice),
        "promotion_allowed": not bool(args.diagnostic_only) and not bool(args.contains_eval_slice),
        "promotion_note": (
            "Built from an eval/heldout slice. Use only for objective/debug gates; do not promote checkpoints trained on this corpus."
            if args.contains_eval_slice or args.diagnostic_only
            else "Promotion still requires separate heldout gates."
        ),
        "target_rejected_examples": target_rejected[:20],
        "label_rejected_examples": label_rejected[:20],
    }
    manifest_path = args.out_dir / "train_agentic_mix.manifest"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
