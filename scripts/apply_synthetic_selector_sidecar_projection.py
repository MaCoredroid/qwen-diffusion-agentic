#!/usr/bin/env python3
import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from eval_toolcall_jsonl import extract_tool_calls, score_tool_calls, tool_schema_by_name
from rescore_toolcall_sequence_planner_projection import (
    compact_calls,
    load_jsonl,
    task_segments,
    user_context_text,
    voice_command_evidence_for_segment,
)


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_CASES = ROOT / "data/toolcall_eval/synthetic_multicall_failure_analogues.jsonl"
DEFAULT_SELECTOR = ROOT / "runs/candidate_ranking/synthetic_multicall_failure_analogue_leaveone_voice003_ckpt20_index_rank.jsonl"
DEFAULT_OUT = ROOT / "runs/synthetic_multicall_failure_analogues/selector_sidecar_leaveone_ckpt20_projection.jsonl"


def case_key(row, fallback_idx):
    return row.get("id") or row.get("case_id") or str(fallback_idx)


def selector_key(row):
    return row.get("id")


def index_selectors(rows):
    out = {}
    for row in rows:
        key = selector_key(row)
        if key:
            out[key] = row
    return out


def schema_props(case, name):
    schemas = tool_schema_by_name(case.get("tools") or [])
    schema = schemas.get(name) or {}
    props = schema.get("properties") if isinstance(schema, dict) else {}
    return props if isinstance(props, dict) else {}


def compatible_arguments(case, name, arguments):
    props = schema_props(case, name)
    if not props:
        return dict(arguments or {})
    return {key: value for key, value in dict(arguments or {}).items() if key in props}


def fill_voice_command_arguments(case, call_index):
    text = user_context_text(case)
    segments = task_segments(text)
    segment = segments[call_index]["text"] if call_index < len(segments) else text
    evidence = voice_command_evidence_for_segment(text, segment) or {}
    args = {}
    for key in ["command", "device_type", "location"]:
        if key in evidence:
            args[key] = evidence[key]
    return args, evidence


def apply_selector(case, selector, text):
    calls, invalid = extract_tool_calls(text)
    audit = {
        "selector_id": selector.get("id"),
        "selector_kind": selector.get("kind"),
        "selector_correct": selector.get("correct"),
        "selector_target": selector.get("target"),
        "selector_predicted_value": selector.get("predicted_value"),
        "selector_target_margin": selector.get("target_margin"),
        "source_invalid_tool_json": invalid,
    }
    if invalid or not calls:
        audit["projection_status"] = "no_valid_source_calls"
        return text, audit
    index = int(selector.get("tool_call_index") or 0)
    if index < 0 or index >= len(calls):
        audit["projection_status"] = "selector_index_out_of_range"
        return text, audit

    out_calls = copy.deepcopy(calls)
    kind = selector.get("kind")
    predicted = selector.get("predicted_value")
    if kind == "tool_name":
        old_call = out_calls[index]
        old_name = old_call.get("name")
        new_name = str(predicted)
        if new_name == "activate_voice_command":
            args, evidence = fill_voice_command_arguments(case, index)
            audit["voice_command_evidence"] = evidence
        elif new_name == old_name:
            args = old_call.get("arguments") or {}
        else:
            args = compatible_arguments(case, new_name, old_call.get("arguments") or {})
        out_calls[index] = {"name": new_name, "arguments": args}
        audit.update({"projection_status": "tool_name_replaced", "old_name": old_name, "new_name": new_name})
    elif kind == "argument_value":
        key = selector.get("json_key")
        if not key:
            audit["projection_status"] = "missing_json_key"
            return text, audit
        call = out_calls[index]
        args = dict(call.get("arguments") or {})
        old_value = args.get(key)
        args[key] = predicted
        call["arguments"] = args
        audit.update({"projection_status": "argument_value_replaced", "json_key": key, "old_value": old_value, "new_value": predicted})
    else:
        audit["projection_status"] = f"unsupported_kind:{kind}"
        return text, audit
    return compact_calls(out_calls), audit


def add_metric_totals(totals, metrics):
    totals["valid_tool_json"] += int(bool(metrics.get("valid_tool_call")))
    totals["exact_tool_name_set"] += int(bool(metrics.get("exact_tool_name_set")))
    totals["exact_tool_name_multiset"] += int(bool(metrics.get("exact_tool_name_multiset")))
    totals["exact_tool_sequence"] += int(bool(metrics.get("exact_tool_sequence")))
    totals["same_tool_call_count"] += int(bool(metrics.get("same_tool_call_count")))
    totals["exact_arguments"] += int(bool(metrics.get("exact_arguments")))
    totals["all_schema_valid"] += int(bool(metrics.get("all_schema_valid")))
    totals["all_required_args_present"] += int(bool(metrics.get("all_required_args_present")))
    totals["records_with_extra_calls"] += int((metrics.get("extra_call_count") or 0) > 0)
    totals["records_with_missing_calls"] += int((metrics.get("missing_call_count") or 0) > 0)
    totals["records_with_repeated_calls"] += int((metrics.get("repeated_call_count") or 0) > 0)
    totals["total_extra_calls"] += int(metrics.get("extra_call_count") or 0)
    totals["total_missing_calls"] += int(metrics.get("missing_call_count") or 0)
    totals["total_repeated_calls"] += int(metrics.get("repeated_call_count") or 0)


def metric_totals():
    return Counter(
        {
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
    )


def add_row_metrics(row, prefix, metrics):
    row[f"{prefix}_valid_tool_json"] = bool(metrics.get("valid_tool_call"))
    row[f"{prefix}_exact_tool_name_set"] = bool(metrics.get("exact_tool_name_set"))
    row[f"{prefix}_exact_tool_name_multiset"] = bool(metrics.get("exact_tool_name_multiset"))
    row[f"{prefix}_exact_tool_sequence"] = bool(metrics.get("exact_tool_sequence"))
    row[f"{prefix}_same_tool_call_count"] = bool(metrics.get("same_tool_call_count"))
    row[f"{prefix}_exact_arguments"] = bool(metrics.get("exact_arguments"))
    row[f"{prefix}_all_schema_valid"] = bool(metrics.get("all_schema_valid"))
    row[f"{prefix}_all_required_args_present"] = bool(metrics.get("all_required_args_present"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-jsonl", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--selector-jsonl", type=Path, default=DEFAULT_SELECTOR)
    parser.add_argument("--text-field", default="bad_draft_assistant")
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    selectors = index_selectors(load_jsonl(args.selector_jsonl))
    totals = {"records": 0, "ok": 0, "errors": 0, "input": metric_totals(), "sidecar": metric_totals()}
    status_counts = Counter()
    rows = []
    for idx, case in enumerate(load_jsonl(args.cases_jsonl)):
        key = case_key(case, idx)
        out = {"idx": idx, "id": key, "status": "ok"}
        try:
            source_text = case.get(args.text_field) or ""
            selector = selectors.get(key)
            if not selector:
                out.update({"status": "missing_selector"})
                totals["errors"] += 1
                rows.append(out)
                continue
            sidecar_text, audit = apply_selector(case, selector, source_text)
            input_metrics = score_tool_calls(source_text, case.get("tools") or [], case.get("gold_assistant"))
            sidecar_metrics = score_tool_calls(sidecar_text, case.get("tools") or [], case.get("gold_assistant"))
            out.update(
                {
                    "text_field": args.text_field,
                    "selector_audit": audit,
                    "input_assistant": source_text,
                    "selector_sidecar_assistant": sidecar_text,
                    "input_called_names": [call.get("name") for call in extract_tool_calls(source_text)[0]],
                    "sidecar_called_names": [call.get("name") for call in extract_tool_calls(sidecar_text)[0]],
                }
            )
            add_row_metrics(out, "input", input_metrics)
            add_row_metrics(out, "sidecar", sidecar_metrics)
            add_metric_totals(totals["input"], input_metrics)
            add_metric_totals(totals["sidecar"], sidecar_metrics)
            status_counts[str(audit.get("projection_status"))] += 1
            totals["ok"] += 1
        except Exception as exc:
            out.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            totals["errors"] += 1
        totals["records"] += 1
        rows.append(out)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "cases_jsonl": str(args.cases_jsonl),
        "selector_jsonl": str(args.selector_jsonl),
        "text_field": args.text_field,
        "out_jsonl": str(args.out_jsonl),
        "totals": {
            "records": totals["records"],
            "ok": totals["ok"],
            "errors": totals["errors"],
            "input": dict(totals["input"]),
            "sidecar": dict(totals["sidecar"]),
        },
        "projection_status_counts": dict(sorted(status_counts.items())),
    }
    args.out_jsonl.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
