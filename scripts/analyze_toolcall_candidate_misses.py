#!/usr/bin/env python3
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/home/mark/qwen_diffusion")
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from eval_toolcall_jsonl import extract_json_objects, extract_tool_calls  # noqa: E402


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def case_key(row, fallback_idx):
    return row.get("id") or row.get("case_id") or str(fallback_idx)


def normalize_scalar(value):
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def normalize_for_compare(value):
    value = normalize_scalar(value)
    if isinstance(value, str):
        return value.strip()
    return value


def flatten_value(value, prefix=""):
    rows = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(flatten_value(item, path))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            path = f"{prefix}[{idx}]"
            rows.extend(flatten_value(item, path))
    else:
        rows.append({"path": prefix, "key": leaf_key(prefix), "value": normalize_scalar(value)})
    return rows


def leaf_key(path):
    raw = str(path).split(".")[-1]
    if "[" in raw:
        raw = raw.split("[", 1)[0]
    return raw


def value_to_text(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def target_value(item):
    text = item.get("target_text")
    if text is None:
        return None
    try:
        return normalize_scalar(json.loads(text))
    except Exception:
        return str(text).strip().strip('"')


def schedule_candidate_index(schedule):
    out = defaultdict(list)
    for item in schedule or []:
        if item.get("kind") != "argument_value":
            continue
        try:
            tool_call_index = int(item.get("tool_call_index"))
        except Exception:
            continue
        key = item.get("json_key")
        target = target_value(item)
        candidates = item.get("candidate_sequence_values")
        if candidates is None:
            candidates = item.get("candidate_values") or []
        out[(tool_call_index, key)].append(
            {
                "target": target,
                "target_text": item.get("target_text"),
                "candidate_sequence_values": item.get("candidate_sequence_values"),
                "candidate_values": item.get("candidate_values"),
                "selected_candidate": item.get("selected_candidate"),
                "token_start": item.get("token_start"),
                "token_end": item.get("token_end"),
                "target_in_sequence_candidates": normalize_for_compare(target)
                in [normalize_for_compare(value) for value in (item.get("candidate_sequence_values") or [])],
                "target_in_candidates": normalize_for_compare(target)
                in [normalize_for_compare(value) for value in candidates],
            }
        )
    return out


def matching_schedule_entries(index, tool_call_index, key, gold_value):
    entries = index.get((tool_call_index, key), [])
    gold_norm = normalize_for_compare(gold_value)
    exact = [entry for entry in entries if normalize_for_compare(entry["target"]) == gold_norm]
    return exact or entries


def candidate_summary(entries):
    if not entries:
        return {
            "schedule_entries": 0,
            "gold_target_in_any_sequence_candidates": False,
            "gold_target_in_any_candidates": False,
            "candidate_sequence_values": [],
            "selected_candidates": [],
        }
    sequence_values = []
    selected = []
    for entry in entries:
        for value in entry.get("candidate_sequence_values") or []:
            if value not in sequence_values:
                sequence_values.append(value)
        value = entry.get("selected_candidate")
        if value is not None and value not in selected:
            selected.append(value)
    return {
        "schedule_entries": len(entries),
        "gold_target_in_any_sequence_candidates": any(entry.get("target_in_sequence_candidates") for entry in entries),
        "gold_target_in_any_candidates": any(entry.get("target_in_candidates") for entry in entries),
        "candidate_sequence_values": sequence_values,
        "selected_candidates": selected,
    }


def gold_calls_for_case(case):
    calls = case.get("gold_tool_calls")
    if isinstance(calls, list):
        return [
            {
                "name": call.get("name"),
                "arguments": call.get("arguments") or {},
            }
            for call in calls
            if isinstance(call, dict)
        ]
    calls, _ = extract_tool_calls(case.get("gold_assistant") or "")
    return calls


def invalid_tool_blocks(text):
    objects = extract_json_objects(text or "")
    blocks = []
    cursor = 0
    idx = 0
    while True:
        start = (text or "").find("<tool_call>", cursor)
        if start < 0:
            break
        content_start = start + len("<tool_call>")
        end = (text or "").find("</tool_call>", content_start)
        if end < 0:
            content = (text or "")[content_start:]
            cursor = len(text or "")
        else:
            content = (text or "")[content_start:end]
            cursor = end + len("</tool_call>")
        parsed = objects[idx] if idx < len(objects) else None
        if parsed is None:
            name_match = re.search(r'"name"\s*:\s*"([^"]+)"', content)
            blocks.append(
                {
                    "block_index": idx,
                    "name": name_match.group(1) if name_match else None,
                    "preview": content.strip()[:500],
                }
            )
        idx += 1
    return blocks


def analyze_row(row, case, schedule_row, fallback_idx):
    key = case_key(row, fallback_idx)
    generated_calls = row.get("calls") or []
    gold_calls = gold_calls_for_case(case)
    schedule_index = schedule_candidate_index((schedule_row or {}).get("schedule") or [])
    mismatches = []
    missing_calls = []
    extra_calls = []

    max_len = max(len(generated_calls), len(gold_calls))
    for call_idx in range(max_len):
        generated = generated_calls[call_idx] if call_idx < len(generated_calls) else None
        gold = gold_calls[call_idx] if call_idx < len(gold_calls) else None
        if generated is None and gold is not None:
            missing_calls.append({"tool_call_index": call_idx, "gold_name": gold.get("name")})
            continue
        if gold is None and generated is not None:
            extra_calls.append({"tool_call_index": call_idx, "generated_name": generated.get("name")})
            continue
        if generated.get("name") != gold.get("name"):
            mismatches.append(
                {
                    "tool_call_index": call_idx,
                    "kind": "tool_name",
                    "generated": generated.get("name"),
                    "gold": gold.get("name"),
                }
            )
            continue

        gen_flat = {item["path"]: item for item in flatten_value(generated.get("arguments") or {})}
        gold_flat = {item["path"]: item for item in flatten_value(gold.get("arguments") or {})}
        for path in sorted(set(gen_flat) | set(gold_flat)):
            gen_item = gen_flat.get(path)
            gold_item = gold_flat.get(path)
            gen_value = gen_item.get("value") if gen_item else None
            gold_value = gold_item.get("value") if gold_item else None
            if normalize_for_compare(gen_value) == normalize_for_compare(gold_value):
                continue
            key_name = (gold_item or gen_item or {}).get("key") or leaf_key(path)
            entries = matching_schedule_entries(schedule_index, call_idx, key_name, gold_value)
            mismatches.append(
                {
                    "tool_call_index": call_idx,
                    "tool_name": gold.get("name"),
                    "kind": "argument_value",
                    "path": path,
                    "json_key": key_name,
                    "generated": gen_value,
                    "gold": gold_value,
                    "candidate_summary": candidate_summary(entries),
                }
            )

    invalid_blocks = invalid_tool_blocks(row.get("assistant") or "")
    return {
        "id": key,
        "exact_tool_sequence": bool(row.get("exact_tool_sequence")),
        "exact_arguments": bool(row.get("exact_arguments")),
        "valid_tool_json": bool(row.get("valid_tool_json")),
        "called_names": row.get("called_names") or [],
        "gold_called_names": [call.get("name") for call in gold_calls],
        "missing_calls": missing_calls,
        "extra_calls": extra_calls,
        "invalid_tool_blocks": invalid_blocks,
        "mismatches": mismatches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--cases-jsonl", type=Path, required=True)
    parser.add_argument("--schedule-jsonl", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    args = parser.parse_args()

    cases = {case_key(row, idx): row for idx, row in enumerate(load_jsonl(args.cases_jsonl))}
    schedules = {case_key(row, idx): row for idx, row in enumerate(load_jsonl(args.schedule_jsonl))}
    records = []
    totals = Counter()
    for idx, row in enumerate(load_jsonl(args.eval_jsonl)):
        key = case_key(row, idx)
        case = cases.get(key)
        if not case:
            totals["missing_case"] += 1
            continue
        audit = analyze_row(row, case, schedules.get(key), idx)
        records.append(audit)
        totals["records"] += 1
        totals["failed_records"] += int(not audit["exact_arguments"] or not audit["exact_tool_sequence"] or not audit["valid_tool_json"])
        totals["mismatches"] += len(audit["mismatches"])
        totals["missing_calls"] += len(audit["missing_calls"])
        totals["extra_calls"] += len(audit["extra_calls"])
        totals["invalid_tool_blocks"] += len(audit["invalid_tool_blocks"])
        for mismatch in audit["mismatches"]:
            totals[f"mismatch_kind:{mismatch['kind']}"] += 1
            if mismatch["kind"] == "argument_value":
                summary = mismatch.get("candidate_summary") or {}
                totals["argument_mismatches_gold_in_sequence_candidates"] += int(
                    bool(summary.get("gold_target_in_any_sequence_candidates"))
                )
                totals["argument_mismatches_gold_in_candidates"] += int(
                    bool(summary.get("gold_target_in_any_candidates"))
                )

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "eval_jsonl": str(args.eval_jsonl),
        "cases_jsonl": str(args.cases_jsonl),
        "schedule_jsonl": str(args.schedule_jsonl),
        "out_jsonl": str(args.out_jsonl),
        "totals": dict(totals),
    }
    args.out_jsonl.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
