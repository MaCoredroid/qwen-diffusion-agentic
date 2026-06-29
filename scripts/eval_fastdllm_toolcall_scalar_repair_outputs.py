#!/usr/bin/env python3
import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path

import torch

from eval_fastdllm_toolcall_cases import (
    case_context_text,
    closest_tool_name,
    clean_repeated_values,
    constrained_tool_call_text,
    generate_case,
    load_cases,
    load_model,
    resolve_chat_template,
    resolve_token_ids,
    sequence_preserving_constrained_tool_call_text,
)
from eval_toolcall_jsonl import extract_tool_calls, score_tool_calls, tool_schema_by_name


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_BASE = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_CASES = ROOT / "data/toolcall_eval/public_multicall_hermes_smoke.jsonl"
DEFAULT_INPUT = (
    ROOT
    / "runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_multicall_eval384_modelrepair/public_multicall_12.jsonl"
)
DEFAULT_OUT = ROOT / "runs/fastdllm_qwen35_9b_toolcall_scalar_repair_eval/public_multicall_12.jsonl"
SCALAR_SYSTEM = "You are a precise one-call tool argument extraction model."


def load_rows(path, limit=0):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def case_key(row, fallback_idx):
    return row.get("id") or row.get("case_id") or str(fallback_idx)


def index_cases(cases):
    return {case_key(case, idx): case for idx, case in enumerate(cases)}


def compact_call(name, arguments):
    payload = {"name": name, "arguments": arguments if isinstance(arguments, dict) else {}}
    return "<tool_call>\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ": ")) + "\n</tool_call>"


def compact_calls(calls):
    return "\n".join(compact_call(call.get("name"), call.get("arguments") or {}) for call in calls if call.get("name"))


def tool_name(tool):
    fn = tool.get("function", tool) if isinstance(tool, dict) else {}
    if isinstance(fn, dict) and fn.get("name"):
        return str(fn["name"])
    return None


def tool_subset(tools, name):
    subset = [copy.deepcopy(tool) for tool in tools or [] if tool_name(tool) == name]
    return subset or copy.deepcopy(tools or [])


def scalar_properties(schema, arguments):
    props = (schema or {}).get("properties") or {}
    keys = list(props) if props else sorted((arguments or {}).keys())
    out = []
    for key in keys:
        prop_schema = props.get(key, {}) if isinstance(props, dict) else {}
        expected = prop_schema.get("type") if isinstance(prop_schema, dict) else None
        if isinstance(expected, list):
            expected = next((item for item in expected if item != "null"), expected[0] if expected else None)
        if expected not in {"array", "object"}:
            out.append(key)
    return out


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


def segment_span(text, call_index, call_count, radius):
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
    spans.append(segment_span(request, call_index, call_count, radius))
    chunks = [request[start:end].strip() for start, end in merge_spans(spans)]
    excerpt = "\n...\n".join(chunk for chunk in chunks if chunk)
    if len(excerpt) > max_chars:
        excerpt = excerpt[: max_chars - 3].rstrip() + "..."
    return excerpt or request[:max_chars].strip()


def scalar_repair_prompt(name, focus_props, draft_text, excerpt):
    if focus_props:
        focus = ", ".join(f"`{prop}`" for prop in focus_props)
    else:
        focus = "all arguments"
    return (
        "Repair exactly one tool call. Keep the function name unchanged. "
        "Copy exact argument values from the request excerpt and tool schema. "
        f"Focus especially on {focus}. Return one corrected Qwen <tool_call> "
        "block with JSON payload and no prose.\n\n"
        "Request excerpt:\n"
        f"{excerpt}\n\n"
        "Draft call:\n"
        f"{draft_text}"
    )


def make_scalar_case(case, call, call_index, call_count, args):
    name = call.get("name")
    schemas = tool_schema_by_name(case.get("tools") or [])
    schema = schemas.get(name) or {}
    props = scalar_properties(schema, call.get("arguments") or {})
    excerpt = request_excerpt(case_context_text(case), call, call_index, call_count, args.max_excerpt_chars)
    draft_text = compact_call(name, call.get("arguments") or {})
    return {
        "prompt_messages": [
            {"role": "system", "content": SCALAR_SYSTEM},
            {"role": "user", "content": scalar_repair_prompt(name, props, draft_text, excerpt)},
        ],
        "tools": tool_subset(case.get("tools") or [], name),
    }


def repaired_call_from_text(text, original_call, tools, context_text):
    single_tool_text = constrained_tool_call_text(
        text,
        tools,
        context_text=context_text,
        max_calls=1,
    )
    calls, invalid = extract_tool_calls(single_tool_text or text)
    if invalid or not calls:
        return copy.deepcopy(original_call), False
    original_name = original_call.get("name")
    repaired = next((call for call in calls if call.get("name") == original_name), None)
    if repaired is None:
        repaired = calls[0]
    arguments = repaired.get("arguments") if isinstance(repaired.get("arguments"), dict) else {}
    if not arguments:
        return copy.deepcopy(original_call), False
    merged = copy.deepcopy(original_call.get("arguments") or {})
    changed = False
    for key, value in arguments.items():
        candidate = clean_repeated_values(value)
        if should_accept_scalar_value(merged.get(key), candidate):
            if merged.get(key) != candidate:
                changed = True
            merged[key] = candidate
    return {"name": original_name, "arguments": merged}, changed


def has_repetition_artifact(value):
    cleaned = clean_repeated_values(value)
    if cleaned != value:
        return True
    if isinstance(value, dict):
        return any(has_repetition_artifact(item) for item in value.values())
    if isinstance(value, list):
        return any(has_repetition_artifact(item) for item in value)
    return False


def is_truncated_or_noisy(value):
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if "\n" in stripped or "<tool_call" in stripped or "</" in stripped:
        return True
    if len(stripped) > 120 and any(marker in stripped for marker in [". ", "I've ", "Could you", "Please "]):
        return True
    return False


def should_accept_scalar_value(original, candidate):
    if candidate is None:
        return False
    if original is None:
        return True
    if isinstance(original, str) and not original.strip():
        return True
    if is_truncated_or_noisy(original):
        return True
    if has_repetition_artifact(original) and not has_repetition_artifact(candidate):
        return True
    return candidate == original


def metric_totals():
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


def add_metric_totals(totals, metrics):
    totals["valid_tool_json"] += int(bool(metrics.get("valid_tool_call")))
    totals["exact_tool_name_set"] += int(bool(metrics.get("exact_tool_name_set")))
    totals["exact_tool_name_multiset"] += int(bool(metrics.get("exact_tool_name_multiset")))
    totals["exact_tool_sequence"] += int(bool(metrics.get("exact_tool_sequence")))
    totals["same_tool_call_count"] += int(bool(metrics.get("same_tool_call_count")))
    totals["exact_arguments"] += int(bool(metrics.get("exact_arguments")))
    totals["all_schema_valid"] += int(bool(metrics.get("all_schema_valid")))
    totals["all_required_args_present"] += int(bool(metrics.get("all_required_args_present")))
    extra = int(metrics.get("extra_call_count") or 0)
    missing = int(metrics.get("missing_call_count") or 0)
    repeated = int(metrics.get("repeated_call_count") or 0)
    totals["records_with_extra_calls"] += int(extra > 0)
    totals["records_with_missing_calls"] += int(missing > 0)
    totals["records_with_repeated_calls"] += int(repeated > 0)
    totals["total_extra_calls"] += extra
    totals["total_missing_calls"] += missing
    totals["total_repeated_calls"] += repeated


def add_row_metrics(row, prefix, metrics):
    row[f"{prefix}_called_names"] = metrics.get("called_names") or []
    row[f"{prefix}_calls"] = metrics.get("calls") or []
    row[f"{prefix}_valid_tool_json"] = bool(metrics.get("valid_tool_call"))
    row[f"{prefix}_exact_tool_sequence"] = bool(metrics.get("exact_tool_sequence"))
    row[f"{prefix}_exact_arguments"] = bool(metrics.get("exact_arguments"))
    row[f"{prefix}_all_schema_valid"] = bool(metrics.get("all_schema_valid"))
    row[f"{prefix}_all_required_args_present"] = bool(metrics.get("all_required_args_present"))
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


def run_eval(model, tokenizer, args):
    cases = load_cases(args.cases_jsonl, args.limit)
    rows = load_rows(args.input_jsonl, args.limit)
    cases_by_key = index_cases(cases)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    totals = {
        "records": 0,
        "ok": 0,
        "errors": 0,
        "scalar_generation_calls": 0,
        "scalar_repaired_calls": 0,
        "unresolved_mask_examples": 0,
        "draft": metric_totals(),
        "scalar_repair": metric_totals(),
        "scalar_repair_constrained": metric_totals(),
    }
    generated_tokens = 0
    start = time.time()

    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for idx, draft_row in enumerate(rows):
            row = {"idx": idx, "id": draft_row.get("id"), "draft_field": args.draft_field}
            try:
                case = cases_by_key.get(case_key(draft_row, idx)) or (cases[idx] if idx < len(cases) else None)
                if case is None:
                    raise KeyError(f"no case for row {idx}")
                draft_text = str(draft_row.get(args.draft_field) or "")
                draft_metrics = score_tool_calls(draft_text, case.get("tools") or [], case.get("gold_assistant"))
                draft_calls, invalid = extract_tool_calls(draft_text)
                if invalid and not draft_calls:
                    raise ValueError(f"draft has {invalid} invalid tool call block(s) and no usable calls")

                repaired_calls = []
                scalar_outputs = []
                row_mask_count = 0
                for call_index, call in enumerate(draft_calls):
                    scalar_case = make_scalar_case(case, call, call_index, len(draft_calls), args)
                    scalar_args = copy.copy(args)
                    scalar_args.append_instruction = False
                    scalar_start = time.time()
                    scalar_text, mask_count, token_count = generate_case(model, tokenizer, scalar_case, scalar_args)
                    scalar_seconds = time.time() - scalar_start
                    one_tool = tool_subset(case.get("tools") or [], call.get("name"))
                    repaired_call, changed = repaired_call_from_text(
                        scalar_text,
                        call,
                        one_tool,
                        case_context_text(case),
                    )
                    repaired_calls.append(repaired_call)
                    scalar_outputs.append(
                        {
                            "call_index": call_index,
                            "name": call.get("name"),
                            "changed": changed,
                            "assistant": scalar_text,
                            "mask_count": mask_count,
                            "generated_token_count": token_count,
                            "seconds": scalar_seconds,
                        }
                    )
                    row_mask_count += mask_count
                    generated_tokens += token_count
                    totals["scalar_generation_calls"] += 1
                    totals["scalar_repaired_calls"] += int(changed)

                scalar_repair_text = compact_calls(repaired_calls)
                scalar_metrics = score_tool_calls(scalar_repair_text, case.get("tools") or [], case.get("gold_assistant"))
                constrained_text = sequence_preserving_constrained_tool_call_text(
                    scalar_repair_text,
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
                        "draft_assistant": draft_text,
                        "scalar_repair_assistant": scalar_repair_text,
                        "scalar_repair_constrained_assistant": constrained_text,
                        "scalar_outputs": scalar_outputs,
                        "scalar_repair_mask_count": row_mask_count,
                    }
                )
                add_row_metrics(row, "draft", draft_metrics)
                add_row_metrics(row, "scalar_repair", scalar_metrics)
                add_row_metrics(row, "scalar_repair_constrained", constrained_metrics)
                add_metric_totals(totals["draft"], draft_metrics)
                add_metric_totals(totals["scalar_repair"], scalar_metrics)
                add_metric_totals(totals["scalar_repair_constrained"], constrained_metrics)
                totals["ok"] += 1
                totals["unresolved_mask_examples"] += int(row_mask_count > 0)
            except Exception as exc:
                row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
                totals["errors"] += 1

            totals["records"] += 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"{idx + 1}/{len(rows)} "
                f"draft={totals['draft']['exact_tool_sequence']}/{totals['draft']['exact_arguments']} "
                f"scalar={totals['scalar_repair']['exact_tool_sequence']}/{totals['scalar_repair']['exact_arguments']} "
                f"constrained={totals['scalar_repair_constrained']['exact_tool_sequence']}/"
                f"{totals['scalar_repair_constrained']['exact_arguments']}",
                flush=True,
            )

    elapsed = time.time() - start
    summary = {
        "cases_jsonl": str(args.cases_jsonl),
        "input_jsonl": str(args.input_jsonl),
        "out_jsonl": str(args.out_jsonl),
        "base_model": str(args.base_model),
        "scalar_adapter": str(args.scalar_adapter),
        "tokenizer_path": str(args.tokenizer_path) if args.tokenizer_path else None,
        "draft_field": args.draft_field,
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
        "constrained_max_calls": args.constrained_max_calls,
        "max_excerpt_chars": args.max_excerpt_chars,
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
    parser.add_argument("--scalar-adapter", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--cases-jsonl", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--draft-field", default="constrained_assistant")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--small-block-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--conversation-template", default="fast_dllm_v2")
    parser.add_argument("--use-block-cache", action="store_true")
    parser.add_argument("--full-context-sampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--constrained-max-calls", type=int, default=0)
    parser.add_argument("--max-excerpt-chars", type=int, default=900)
    parser.add_argument("--merge-adapter", action="store_true")
    args = parser.parse_args()

    args.chat_template = resolve_chat_template(args.conversation_template)
    args.append_instruction = False
    ensure_generation_defaults(args)
    model, tokenizer = load_model(
        str(args.base_model),
        str(args.scalar_adapter),
        merge_adapter=args.merge_adapter,
        tokenizer_path=str(args.tokenizer_path) if args.tokenizer_path else None,
    )
    args.mask_id, args.stop_token_id = resolve_token_ids(model, tokenizer)
    run_eval(model, tokenizer, args)


if __name__ == "__main__":
    main()
