#!/usr/bin/env python3
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

from eval_fastdllm_toolcall_cases import (
    case_context_text,
    constrained_tool_call_text,
    generate_case,
    load_cases,
    load_model,
    model_repair_case,
    repaired_tool_call_text,
    resolve_chat_template,
    resolve_token_ids,
)
from eval_toolcall_jsonl import score_tool_calls


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_BASE = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_CASES = ROOT / "data/toolcall_eval/public_multicall_hermes_smoke.jsonl"
DEFAULT_INPUT = (
    ROOT
    / "runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_multicall_eval384_modelrepair/public_multicall_12.jsonl"
)
DEFAULT_OUT = ROOT / "runs/fastdllm_qwen35_9b_toolcall_multicall_repair_eval/public_multicall_12.jsonl"

METRIC_KEYS = [
    "valid_tool_call",
    "exact_tool_name_set",
    "exact_tool_name_multiset",
    "exact_tool_sequence",
    "same_tool_call_count",
    "exact_arguments",
    "all_schema_valid",
    "all_required_args_present",
]


def load_rows(path, limit=0):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def case_key(item, fallback_idx):
    return item.get("id") or item.get("case_id") or str(fallback_idx)


def sequence_repair_instruction(raw_text):
    return (
        "The draft below contains the exact tool-call sequence to preserve. "
        "Keep the same number of tool calls, the same function names, and the "
        "same order. Repair only the arguments by copying exact values from the "
        "original user request and tool schema. Do not add, remove, rename, or "
        "reorder tools. Return only corrected Qwen <tool_call> blocks with JSON "
        "payloads and no prose.\n\n"
        "Fixed-sequence draft:\n"
        f"{raw_text}"
    )


def repair_case(case, raw_text, prompt_mode):
    if prompt_mode == "broad":
        return model_repair_case(case, raw_text)
    if prompt_mode == "preserve_sequence":
        repaired = copy.deepcopy(case)
        repaired["prompt_messages"] = list(copy.deepcopy(case.get("prompt_messages") or []))
        repaired["prompt_messages"].append({"role": "user", "content": sequence_repair_instruction(raw_text)})
        return repaired
    raise ValueError(f"unknown repair prompt mode {prompt_mode!r}")


def index_cases(cases):
    by_key = {}
    for idx, case in enumerate(cases):
        by_key[case_key(case, idx)] = case
    return by_key


def add_metrics(row, prefix, metrics):
    row[f"{prefix}_called_names"] = metrics.get("called_names") or []
    row[f"{prefix}_calls"] = metrics.get("calls") or []
    row[f"{prefix}_invalid_tool_json_count"] = metrics.get("invalid_tool_call_count")
    row[f"{prefix}_valid_tool_json"] = bool(metrics.get("valid_tool_call"))
    row[f"{prefix}_exact_tool_name_set"] = bool(metrics.get("exact_tool_name_set"))
    row[f"{prefix}_exact_tool_name_multiset"] = bool(metrics.get("exact_tool_name_multiset"))
    row[f"{prefix}_exact_tool_sequence"] = bool(metrics.get("exact_tool_sequence"))
    row[f"{prefix}_same_tool_call_count"] = bool(metrics.get("same_tool_call_count"))
    row[f"{prefix}_exact_arguments"] = bool(metrics.get("exact_arguments"))
    row[f"{prefix}_all_schema_valid"] = bool(metrics.get("all_schema_valid"))
    row[f"{prefix}_all_required_args_present"] = bool(metrics.get("all_required_args_present"))
    row[f"{prefix}_schema_valid_count"] = metrics.get("schema_valid_count")
    row[f"{prefix}_required_args_count"] = metrics.get("required_args_count")
    row[f"{prefix}_extra_call_count"] = metrics.get("extra_call_count")
    row[f"{prefix}_missing_call_count"] = metrics.get("missing_call_count")
    row[f"{prefix}_repeated_call_count"] = metrics.get("repeated_call_count")
    row[f"{prefix}_call_errors"] = metrics.get("call_errors") or []


def ensure_generation_defaults(args):
    args.force_tool_call_prefix = getattr(args, "force_tool_call_prefix", False)
    args.forced_assistant_prefix = getattr(args, "forced_assistant_prefix", "")
    args.force_schedule_token_kinds = getattr(args, "force_schedule_token_kinds", set())
    args.force_argument_boundary_target_tokens = getattr(args, "force_argument_boundary_target_tokens", False)
    args.constrain_argument_candidate_tokens = getattr(args, "constrain_argument_candidate_tokens", False)
    args.force_selected_candidate_tokens = getattr(args, "force_selected_candidate_tokens", False)
    args.force_best_candidate_sequence = getattr(args, "force_best_candidate_sequence", False)
    args.force_best_tool_name_sequence = getattr(args, "force_best_tool_name_sequence", False)
    args.ban_argument_boundary_tokens = getattr(args, "ban_argument_boundary_tokens", False)
    args.ban_argument_json_boundary_tokens = getattr(args, "ban_argument_json_boundary_tokens", False)
    args.ban_argument_newline_tokens = getattr(args, "ban_argument_newline_tokens", False)
    args.argument_boundary_token_ids = getattr(args, "argument_boundary_token_ids", [])
    args.argument_newline_token_ids = getattr(args, "argument_newline_token_ids", [])
    args._argument_boundary_target_cache = getattr(args, "_argument_boundary_target_cache", {})


def empty_metric_totals():
    return {
        "valid_tool_json": 0,
        "exact_tool_name_set": 0,
        "exact_tool_name_multiset": 0,
        "exact_tool_sequence": 0,
        "same_tool_call_count": 0,
        "exact_arguments": 0,
        "all_schema_valid": 0,
        "all_required_args_present": 0,
        "records_with_extra_calls": 0,
        "records_with_missing_calls": 0,
        "records_with_repeated_calls": 0,
        "total_extra_calls": 0,
        "total_missing_calls": 0,
        "total_repeated_calls": 0,
    }


def add_to_totals(totals, prefix, metrics):
    bucket = totals[prefix]
    bucket["valid_tool_json"] += int(bool(metrics.get("valid_tool_call")))
    bucket["exact_tool_name_set"] += int(bool(metrics.get("exact_tool_name_set")))
    bucket["exact_tool_name_multiset"] += int(bool(metrics.get("exact_tool_name_multiset")))
    bucket["exact_tool_sequence"] += int(bool(metrics.get("exact_tool_sequence")))
    bucket["same_tool_call_count"] += int(bool(metrics.get("same_tool_call_count")))
    bucket["exact_arguments"] += int(bool(metrics.get("exact_arguments")))
    bucket["all_schema_valid"] += int(bool(metrics.get("all_schema_valid")))
    bucket["all_required_args_present"] += int(bool(metrics.get("all_required_args_present")))
    extra = int(metrics.get("extra_call_count") or 0)
    missing = int(metrics.get("missing_call_count") or 0)
    repeated = int(metrics.get("repeated_call_count") or 0)
    bucket["records_with_extra_calls"] += int(extra > 0)
    bucket["records_with_missing_calls"] += int(missing > 0)
    bucket["records_with_repeated_calls"] += int(repeated > 0)
    bucket["total_extra_calls"] += extra
    bucket["total_missing_calls"] += missing
    bucket["total_repeated_calls"] += repeated


def run_eval(model, tokenizer, args):
    cases = load_cases(args.cases_jsonl, args.limit)
    input_rows = load_rows(args.input_jsonl, args.limit)
    cases_by_key = index_cases(cases)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    totals = {
        "records": 0,
        "ok": 0,
        "errors": 0,
        "unresolved_mask_examples": 0,
        "draft": empty_metric_totals(),
        "repair": empty_metric_totals(),
        "repair_repaired": empty_metric_totals(),
        "repair_constrained": empty_metric_totals(),
    }
    generated_tokens = 0
    start = time.time()

    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for idx, draft_row in enumerate(input_rows):
            row = {
                "idx": idx,
                "source": draft_row.get("source"),
                "id": draft_row.get("id"),
                "draft_field": args.draft_field,
            }
            try:
                key = case_key(draft_row, idx)
                case = cases_by_key.get(key)
                if case is None and idx < len(cases):
                    case = cases[idx]
                if case is None:
                    raise KeyError(f"no matching case for row key {key!r}")
                if args.draft_field not in draft_row:
                    raise KeyError(f"draft field {args.draft_field!r} not present in input row")

                draft_text = str(draft_row.get(args.draft_field) or "")
                draft_metrics = score_tool_calls(draft_text, case.get("tools") or [], case.get("gold_assistant"))

                repair_args = copy.copy(args)
                repair_args.append_instruction = False
                repair_prompt_case = repair_case(case, draft_text, args.repair_prompt_mode)
                repair_start = time.time()
                repair_text, mask_count, token_count = generate_case(model, tokenizer, repair_prompt_case, repair_args)
                repair_seconds = time.time() - repair_start
                repair_metrics = score_tool_calls(repair_text, case.get("tools") or [], case.get("gold_assistant"))

                repaired_text = ""
                repaired_metrics = None
                if args.repair_mode != "none":
                    repaired_text = repaired_tool_call_text(repair_text, case.get("tools") or [])
                    repaired_metrics = score_tool_calls(
                        repaired_text,
                        case.get("tools") or [],
                        case.get("gold_assistant"),
                    )

                constrained_text = ""
                constrained_metrics = None
                if args.constrained_tool_decoding:
                    constrained_text = constrained_tool_call_text(
                        repair_text,
                        case.get("tools") or [],
                        context_text=case_context_text(case),
                        max_calls=args.constrained_max_calls,
                    )
                    constrained_metrics = score_tool_calls(
                        constrained_text,
                        case.get("tools") or [],
                        case.get("gold_assistant"),
                    )

                row.update(
                    {
                        "status": "ok",
                        "gold_tool_names": case.get("gold_tool_names") or [],
                        "available_tool_names": case.get("available_tool_names") or [],
                        "draft_assistant": draft_text,
                        "repair_assistant": repair_text,
                        "repair_mask_count": mask_count,
                        "repair_generated_token_count": token_count,
                        "repair_seconds": repair_seconds,
                    }
                )
                add_metrics(row, "draft", draft_metrics)
                add_metrics(row, "repair", repair_metrics)

                totals["ok"] += 1
                totals["unresolved_mask_examples"] += int(mask_count > 0)
                generated_tokens += token_count
                add_to_totals(totals, "draft", draft_metrics)
                add_to_totals(totals, "repair", repair_metrics)

                if repaired_metrics is not None:
                    row["repair_repaired_assistant"] = repaired_text
                    add_metrics(row, "repair_repaired", repaired_metrics)
                    add_to_totals(totals, "repair_repaired", repaired_metrics)

                if constrained_metrics is not None:
                    row["repair_constrained_assistant"] = constrained_text
                    row["constrained_max_calls"] = args.constrained_max_calls
                    add_metrics(row, "repair_constrained", constrained_metrics)
                    add_to_totals(totals, "repair_constrained", constrained_metrics)

            except Exception as exc:
                row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
                totals["errors"] += 1

            totals["records"] += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(
                f"{idx + 1}/{len(input_rows)} "
                f"draft_seq={totals['draft']['exact_tool_sequence']} "
                f"repair_seq={totals['repair']['exact_tool_sequence']} "
                f"repair_args={totals['repair']['exact_arguments']}",
                flush=True,
            )

    elapsed = time.time() - start
    summary = {
        "cases_jsonl": str(args.cases_jsonl),
        "input_jsonl": str(args.input_jsonl),
        "out_jsonl": str(args.out_jsonl),
        "base_model": str(args.base_model),
        "repair_adapter": str(args.repair_adapter),
        "tokenizer_path": str(args.tokenizer_path) if args.tokenizer_path else None,
        "draft_field": args.draft_field,
        "repair_prompt_mode": args.repair_prompt_mode,
        "merge_adapter": args.merge_adapter,
        "totals": totals,
        "elapsed_seconds": elapsed,
        "generated_tokens": generated_tokens,
        "generated_tokens_per_second": generated_tokens / elapsed if elapsed else 0.0,
        "block_size": args.block_size,
        "small_block_size": args.small_block_size,
        "max_new_tokens": args.max_new_tokens,
        "threshold": args.threshold,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "use_block_cache": args.use_block_cache,
        "full_context_sampling": args.full_context_sampling,
        "repair_mode": args.repair_mode,
        "constrained_tool_decoding": args.constrained_tool_decoding,
        "constrained_max_calls": args.constrained_max_calls,
        "mask_id": args.mask_id,
        "stop_token_id": args.stop_token_id,
        "conversation_template": args.conversation_template,
    }
    if torch.cuda.is_available():
        summary["cuda_max_memory_allocated_gib"] = torch.cuda.max_memory_allocated() / (1024**3)
        summary["cuda_max_memory_reserved_gib"] = torch.cuda.max_memory_reserved() / (1024**3)
    summary_path = args.out_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--repair-adapter", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--cases-jsonl", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--draft-field", default="assistant")
    parser.add_argument("--repair-prompt-mode", choices=["broad", "preserve_sequence"], default="broad")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--small-block-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--conversation-template", default="fast_dllm_v2")
    parser.add_argument("--use-block-cache", action="store_true")
    parser.add_argument("--full-context-sampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repair-mode", choices=["none", "schema"], default="schema")
    parser.add_argument(
        "--constrained-tool-decoding",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--constrained-max-calls", type=int, default=0)
    parser.add_argument("--merge-adapter", action="store_true")
    args = parser.parse_args()

    args.chat_template = resolve_chat_template(args.conversation_template)
    args.append_instruction = False
    ensure_generation_defaults(args)
    model, tokenizer = load_model(
        str(args.base_model),
        str(args.repair_adapter),
        merge_adapter=args.merge_adapter,
        tokenizer_path=str(args.tokenizer_path) if args.tokenizer_path else None,
    )
    args.mask_id, args.stop_token_id = resolve_token_ids(model, tokenizer)
    run_eval(model, tokenizer, args)


if __name__ == "__main__":
    main()
