#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from eval_fastdllm_toolcall_cases import (
    case_context_text,
    constrained_tool_call_text,
    empty_totals,
    repaired_tool_call_text,
    sequence_preserving_constrained_tool_call_text,
)
from eval_toolcall_jsonl import score_tool_calls


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cases_by_id(cases):
    return {case.get("id"): case for case in cases}


def add_metrics(totals, row, prefix):
    totals[f"{prefix}_valid_tool_json"] += int(bool(row[f"{prefix}_valid_tool_json"]))
    totals[f"{prefix}_exact_tool_name_set"] += int(bool(row[f"{prefix}_exact_tool_name_set"]))
    totals[f"{prefix}_exact_tool_sequence"] += int(bool(row[f"{prefix}_exact_tool_sequence"]))
    totals[f"{prefix}_exact_arguments"] += int(bool(row[f"{prefix}_exact_arguments"]))
    totals[f"{prefix}_all_schema_valid"] += int(bool(row[f"{prefix}_all_schema_valid"]))
    totals[f"{prefix}_all_required_args_present"] += int(bool(row[f"{prefix}_all_required_args_present"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-jsonl", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument(
        "--text-field",
        default="assistant",
        help="Assistant text field to rescore. Defaults to the raw assistant field.",
    )
    parser.add_argument("--repair-mode", choices=["none", "schema"], default="schema")
    parser.add_argument("--constrained-tool-decoding", action="store_true")
    parser.add_argument(
        "--sequence-preserving-constrained",
        action="store_true",
        help="Preserve strict tool-call names/order/count from --text-field, including repeated calls.",
    )
    parser.add_argument(
        "--constrained-max-calls",
        type=int,
        default=0,
        help="When >0, cap constrained projection to this many tool calls. Use 1 for one-call eval slices.",
    )
    args = parser.parse_args()

    cases = cases_by_id(load_jsonl(args.cases_jsonl))
    rows = load_jsonl(args.input_jsonl)
    totals = empty_totals()
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(rows):
            case = cases.get(row.get("id"))
            if not case:
                row = {**row, "rescore_status": "missing_case"}
                totals["errors"] += 1
                totals["records"] += 1
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            text = row.get(args.text_field) or ""
            metrics = score_tool_calls(text, case.get("tools") or [], case.get("gold_assistant"))
            row.update(
                {
                    "rescore_status": "ok",
                    "valid_tool_json": metrics["valid_tool_call"],
                    "exact_tool_name_set": metrics.get("exact_tool_name_set"),
                    "exact_tool_sequence": metrics.get("exact_tool_sequence"),
                    "exact_tool_name_multiset": metrics.get("exact_tool_name_multiset"),
                    "same_tool_call_count": metrics.get("same_tool_call_count"),
                    "exact_arguments": metrics.get("exact_arguments"),
                    "all_schema_valid": metrics["all_schema_valid"],
                    "all_required_args_present": metrics["all_required_args_present"],
                    "extra_call_count": metrics.get("extra_call_count"),
                    "missing_call_count": metrics.get("missing_call_count"),
                    "repeated_call_count": metrics.get("repeated_call_count"),
                }
            )
            totals["ok"] += 1
            totals["valid_tool_json"] += int(bool(row["valid_tool_json"]))
            totals["exact_tool_name_set"] += int(bool(row["exact_tool_name_set"]))
            totals["exact_tool_sequence"] += int(bool(row["exact_tool_sequence"]))
            totals["exact_tool_name_multiset"] += int(bool(row["exact_tool_name_multiset"]))
            totals["same_tool_call_count"] += int(bool(row["same_tool_call_count"]))
            totals["exact_arguments"] += int(bool(row["exact_arguments"]))
            totals["all_schema_valid"] += int(bool(row["all_schema_valid"]))
            totals["all_required_args_present"] += int(bool(row["all_required_args_present"]))
            totals["records_with_extra_calls"] += int((row["extra_call_count"] or 0) > 0)
            totals["records_with_missing_calls"] += int((row["missing_call_count"] or 0) > 0)
            totals["records_with_repeated_calls"] += int((row["repeated_call_count"] or 0) > 0)
            totals["total_extra_calls"] += int(row["extra_call_count"] or 0)
            totals["total_missing_calls"] += int(row["missing_call_count"] or 0)
            totals["total_repeated_calls"] += int(row["repeated_call_count"] or 0)

            if args.repair_mode != "none":
                repaired_text = repaired_tool_call_text(text, case.get("tools") or [])
                repaired = score_tool_calls(repaired_text, case.get("tools") or [], case.get("gold_assistant"))
                row.update(
                    {
                        "repair_mode": args.repair_mode,
                        "repaired_assistant": repaired_text,
                        "repaired_called_names": repaired["called_names"],
                        "repaired_calls": repaired["calls"],
                        "repaired_valid_tool_json": repaired["valid_tool_call"],
                        "repaired_exact_tool_name_set": repaired.get("exact_tool_name_set"),
                        "repaired_exact_tool_sequence": repaired.get("exact_tool_sequence"),
                        "repaired_exact_arguments": repaired.get("exact_arguments"),
                        "repaired_all_schema_valid": repaired["all_schema_valid"],
                        "repaired_all_required_args_present": repaired["all_required_args_present"],
                        "repaired_call_errors": repaired["call_errors"],
                    }
                )
                add_metrics(totals, row, "repaired")

            if args.constrained_tool_decoding:
                if args.sequence_preserving_constrained:
                    constrained_text = sequence_preserving_constrained_tool_call_text(
                        text,
                        case.get("tools") or [],
                        context_text=case_context_text(case),
                        max_calls=args.constrained_max_calls,
                    )
                else:
                    constrained_text = constrained_tool_call_text(
                        text,
                        case.get("tools") or [],
                        context_text=case_context_text(case),
                        max_calls=args.constrained_max_calls,
                    )
                constrained = score_tool_calls(constrained_text, case.get("tools") or [], case.get("gold_assistant"))
                row.update(
                    {
                        "constrained_assistant": constrained_text,
                        "constrained_max_calls": args.constrained_max_calls,
                        "constrained_called_names": constrained["called_names"],
                        "constrained_calls": constrained["calls"],
                        "constrained_valid_tool_json": constrained["valid_tool_call"],
                        "constrained_exact_tool_name_set": constrained.get("exact_tool_name_set"),
                        "constrained_exact_tool_sequence": constrained.get("exact_tool_sequence"),
                        "constrained_exact_arguments": constrained.get("exact_arguments"),
                        "constrained_all_schema_valid": constrained["all_schema_valid"],
                        "constrained_all_required_args_present": constrained["all_required_args_present"],
                        "constrained_call_errors": constrained["call_errors"],
                    }
                )
                add_metrics(totals, row, "constrained")

            totals["unresolved_mask_examples"] += int((row.get("mask_count") or 0) > 0)
            totals["records"] += 1
            row["rescore_idx"] = idx
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_jsonl": str(args.input_jsonl),
        "cases_jsonl": str(args.cases_jsonl),
        "out_jsonl": str(args.out_jsonl),
        "repair_mode": args.repair_mode,
        "text_field": args.text_field,
        "constrained_tool_decoding": args.constrained_tool_decoding,
        "sequence_preserving_constrained": args.sequence_preserving_constrained,
        "constrained_max_calls": args.constrained_max_calls,
        "totals": totals,
    }
    summary_path = args.out_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
