#!/usr/bin/env python3
import argparse
from collections import Counter
import json
from pathlib import Path

from eval_toolcall_jsonl import coerce_arguments, extract_tool_calls, tool_schema_by_name


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cases_by_id(cases):
    return {case.get("id"): case for case in cases}


def schema_type(schema):
    if not isinstance(schema, dict):
        return None
    expected = schema.get("type")
    if isinstance(expected, list):
        return next((item for item in expected if item != "null"), expected[0] if expected else None)
    return expected


def is_complex(value):
    return isinstance(value, (dict, list))


def path_kind(path, schema):
    expected = schema_type(schema)
    if expected in {"object", "array"}:
        return "complex"
    if expected in {"string", "integer", "number", "boolean"}:
        return "scalar"
    return "complex" if "[" in path or isinstance(schema, dict) and schema.get("properties") else "scalar"


def compare_values(pred, gold, schema, path="$"):
    expected = schema_type(schema)
    diffs = []
    if expected == "object" or isinstance(gold, dict) or isinstance(pred, dict):
        if not isinstance(pred, dict):
            return [{"path": path, "kind": "type_mismatch", "expected": "object", "pred": pred, "gold": gold}]
        if not isinstance(gold, dict):
            return [{"path": path, "kind": "type_mismatch", "expected": type(gold).__name__, "pred": pred, "gold": gold}]
        props = (schema or {}).get("properties") or {}
        required = set((schema or {}).get("required") or [])
        for key in sorted(set(gold) | set(pred)):
            child_path = f"{path}.{key}"
            child_schema = props.get(key, {})
            if key not in pred:
                diffs.append(
                    {
                        "path": child_path,
                        "kind": "missing_required" if key in required else "missing_gold_key",
                        "value_kind": "complex" if is_complex(gold.get(key)) else path_kind(child_path, child_schema),
                        "gold": gold.get(key),
                    }
                )
            elif key not in gold:
                diffs.append(
                    {
                        "path": child_path,
                        "kind": "extra_key",
                        "value_kind": "complex" if is_complex(pred.get(key)) else path_kind(child_path, child_schema),
                        "pred": pred.get(key),
                    }
                )
            else:
                diffs.extend(compare_values(pred[key], gold[key], child_schema, child_path))
        return diffs

    if expected == "array" or isinstance(gold, list) or isinstance(pred, list):
        if not isinstance(pred, list):
            return [{"path": path, "kind": "type_mismatch", "expected": "array", "pred": pred, "gold": gold}]
        if not isinstance(gold, list):
            return [{"path": path, "kind": "type_mismatch", "expected": type(gold).__name__, "pred": pred, "gold": gold}]
        item_schema = (schema or {}).get("items") or {}
        if len(pred) != len(gold):
            diffs.append({"path": path, "kind": "array_length_mismatch", "pred_len": len(pred), "gold_len": len(gold)})
        for idx in range(min(len(pred), len(gold))):
            diffs.extend(compare_values(pred[idx], gold[idx], item_schema, f"{path}[{idx}]"))
        return diffs

    if pred != gold:
        return [{"path": path, "kind": "value_mismatch", "value_kind": "scalar", "pred": pred, "gold": gold}]
    return []


def calls_from_row(row, prefix):
    calls = row.get(f"{prefix}_calls")
    if isinstance(calls, list):
        return calls, 0
    text = row.get(f"{prefix}_assistant") or row.get("assistant") or ""
    return extract_tool_calls(text)


def diagnose_row(row, case, prefix):
    calls, invalid = calls_from_row(row, prefix)
    gold_calls, gold_invalid = extract_tool_calls(case.get("gold_assistant") or "")
    schemas = tool_schema_by_name(case.get("tools") or [])
    called_names = [call.get("name") for call in calls]
    gold_names = [call.get("name") for call in gold_calls]
    row_diffs = []
    paired = 0
    for idx, gold_call in enumerate(gold_calls):
        pred_call = calls[idx] if idx < len(calls) and calls[idx].get("name") == gold_call.get("name") else None
        if pred_call is None:
            for candidate in calls:
                if candidate.get("name") == gold_call.get("name"):
                    pred_call = candidate
                    break
        if pred_call is None:
            row_diffs.append(
                {
                    "call_index": idx,
                    "tool_name": gold_call.get("name"),
                    "path": "$",
                    "kind": "missing_tool_call",
                    "value_kind": "complex",
                }
            )
            continue
        paired += 1
        schema = schemas.get(gold_call.get("name"), {})
        pred_args = coerce_arguments(pred_call.get("arguments") or {}, schema)
        gold_args = coerce_arguments(gold_call.get("arguments") or {}, schema)
        for diff in compare_values(pred_args, gold_args, schema, "$"):
            diff = {"call_index": idx, "tool_name": gold_call.get("name"), **diff}
            row_diffs.append(diff)

    for idx, pred_call in enumerate(calls[len(gold_calls) :], start=len(gold_calls)):
        row_diffs.append(
            {
                "call_index": idx,
                "tool_name": pred_call.get("name"),
                "path": "$",
                "kind": "extra_tool_call",
                "value_kind": "complex",
            }
        )

    return {
        "id": row.get("id"),
        "idx": row.get("idx"),
        "source": row.get("source"),
        "prefix": prefix,
        "called_names": called_names,
        "gold_names": gold_names,
        "invalid_tool_calls": invalid,
        "gold_invalid_tool_calls": gold_invalid,
        "paired_gold_calls": paired,
        "exact_tool_sequence": called_names == gold_names,
        "exact_arguments": not row_diffs and called_names == gold_names,
        "diff_count": len(row_diffs),
        "diffs": row_diffs,
    }


def summarize(rows):
    kind_counts = Counter()
    path_counts = Counter()
    missing_required_paths = Counter()
    value_mismatch_paths = Counter()
    rows_with_diff = 0
    rows_exact_tool_not_args = 0
    rows_missing_complex = 0
    rows_scalar_mismatch = 0
    for row in rows:
        if row["diff_count"]:
            rows_with_diff += 1
        if row["exact_tool_sequence"] and not row["exact_arguments"]:
            rows_exact_tool_not_args += 1
        for diff in row["diffs"]:
            kind = diff["kind"]
            path = diff.get("path") or "$"
            kind_counts[kind] += 1
            path_counts[f"{diff.get('tool_name')}:{path}"] += 1
            if kind == "missing_required":
                missing_required_paths[f"{diff.get('tool_name')}:{path}"] += 1
                if diff.get("value_kind") == "complex":
                    rows_missing_complex += 1
            if kind == "value_mismatch":
                value_mismatch_paths[f"{diff.get('tool_name')}:{path}"] += 1
                if diff.get("value_kind") == "scalar":
                    rows_scalar_mismatch += 1
    return {
        "records": len(rows),
        "rows_with_diff": rows_with_diff,
        "rows_exact_tool_sequence_but_not_arguments": rows_exact_tool_not_args,
        "kind_counts": dict(kind_counts.most_common()),
        "top_paths": dict(path_counts.most_common(20)),
        "top_missing_required_paths": dict(missing_required_paths.most_common(20)),
        "top_value_mismatch_paths": dict(value_mismatch_paths.most_common(20)),
        "complex_missing_required_diff_count": rows_missing_complex,
        "scalar_value_mismatch_diff_count": rows_scalar_mismatch,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-jsonl", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--prefix", default="constrained")
    args = parser.parse_args()

    cases = cases_by_id(load_jsonl(args.cases_jsonl))
    rows = []
    missing = 0
    for row in load_jsonl(args.input_jsonl):
        case = cases.get(row.get("id"))
        if not case:
            missing += 1
            continue
        rows.append(diagnose_row(row, case, args.prefix))

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "cases_jsonl": str(args.cases_jsonl),
        "input_jsonl": str(args.input_jsonl),
        "out_jsonl": str(args.out_jsonl),
        "prefix": args.prefix,
        "missing_case_rows": missing,
        **summarize(rows),
    }
    summary_path = args.out_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
