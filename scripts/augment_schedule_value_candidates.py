#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/home/mark/qwen_diffusion")
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from diagnose_schedule_value_candidates import (  # noqa: E402
    case_key,
    diagnose_record,
    load_jsonl,
    normalize_for_compare,
)
from eval_fastdllm_toolcall_cases import target_boundary_mask  # noqa: E402


def value_to_text(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def tool_name(tool):
    fn = tool.get("function", tool) if isinstance(tool, dict) else {}
    if isinstance(fn, dict) and fn.get("name"):
        return str(fn["name"])
    return None


def build_diagnostic_map(schedule_row, case):
    diagnostics = diagnose_record(schedule_row, case)
    out = {}
    for item in diagnostics:
        key = (int(item["tool_call_index"]), item["json_key"], normalize_for_compare(item["target"]))
        out[key] = item
    return out


def target_value(item):
    text = item.get("target_text")
    if text is None:
        return None
    try:
        return json.loads(text)
    except Exception:
        return str(text).strip().strip('"')


def group_argument_items(schedule):
    groups = defaultdict(list)
    for item in schedule:
        if item.get("kind") != "argument_value":
            continue
        key = (
            int(item.get("tool_call_index")),
            item.get("json_key"),
            item.get("target_text"),
        )
        groups[key].append(item)
    return groups


def group_tool_name_items(schedule):
    groups = defaultdict(list)
    for item in schedule:
        if item.get("kind") != "tool_name":
            continue
        key = (
            int(item.get("tool_call_index")),
            item.get("source_block_token_start"),
            item.get("source_block_token_end"),
            item.get("target_text"),
        )
        groups[key].append(item)
    return groups


def full_target_ids(items):
    ids = []
    for item in sorted(items, key=lambda row: int(row["token_start"])):
        ids.extend(int(token_id) for token_id in item.get("target_token_ids") or [])
    return ids


def candidate_allowed_by_offset(tokenizer, target_ids, candidate_values):
    boundary_cache = {}
    boundary_mask = target_boundary_mask(tokenizer, target_ids, boundary_cache)
    candidate_sequences = []
    for value in candidate_values:
        token_ids = tokenizer(value_to_text(value), add_special_tokens=False).input_ids
        if token_ids:
            candidate_sequences.append([int(token_id) for token_id in token_ids])

    allowed = []
    for offset, target_id in enumerate(target_ids):
        if boundary_mask[offset]:
            allowed.append([int(target_id)])
            continue
        ids = {int(target_id)}
        for sequence in candidate_sequences:
            if offset < len(sequence):
                ids.add(int(sequence[offset]))
        allowed.append(sorted(ids))
    return allowed


def selected_candidate_by_offset(tokenizer, target_ids, selected_value):
    if selected_value is None:
        return []
    selected_ids = [int(token_id) for token_id in tokenizer(value_to_text(selected_value), add_special_tokens=False).input_ids]
    if not selected_ids:
        return []
    boundary_cache = {}
    boundary_mask = target_boundary_mask(tokenizer, target_ids, boundary_cache)
    selected = []
    for offset, target_id in enumerate(target_ids):
        if boundary_mask[offset]:
            selected.append([int(target_id)])
        elif offset < len(selected_ids):
            selected.append([selected_ids[offset]])
        else:
            selected.append([])
    return selected


def same_value(left, right):
    return normalize_for_compare(left) == normalize_for_compare(right)


def compatible_candidate_sequences(tokenizer, target_ids, candidate_values, target=None):
    boundary_cache = {}
    boundary_mask = target_boundary_mask(tokenizer, target_ids, boundary_cache)
    content_positions = [idx for idx, is_boundary in enumerate(boundary_mask) if not is_boundary]
    sequences = []
    for value in candidate_values:
        if target is not None and same_value(value, target):
            sequences.append({"value": value, "token_ids_by_offset": [int(token_id) for token_id in target_ids]})
            continue
        candidate_ids = [int(token_id) for token_id in tokenizer(value_to_text(value), add_special_tokens=False).input_ids]
        if len(candidate_ids) != len(content_positions):
            continue
        sequence = [int(token_id) for token_id in target_ids]
        for offset, token_id in zip(content_positions, candidate_ids):
            sequence[offset] = token_id
        sequences.append({"value": value, "token_ids_by_offset": sequence})
    return sequences


def compatible_tool_name_sequences(tokenizer, target_ids, candidate_values):
    target_decoded = tokenizer.decode(target_ids)
    suffix = ""
    if target_decoded.endswith('","'):
        suffix = '","'
    elif target_decoded.endswith('",'):
        suffix = '",'
    elif target_decoded.endswith('"'):
        suffix = '"'
    sequences = []
    for value in candidate_values:
        candidate_ids = [
            int(token_id)
            for token_id in tokenizer(str(value) + suffix, add_special_tokens=False).input_ids
        ]
        if len(candidate_ids) != len(target_ids):
            continue
        sequences.append({"value": value, "token_ids_by_offset": candidate_ids})
    return sequences


def add_target_candidate(candidate_values, target):
    if normalize_for_compare(target) not in [normalize_for_compare(item) for item in candidate_values]:
        candidate_values.append(target)


def choose_selected_candidate(mode, target, diagnostic):
    if mode == "none":
        return None
    if mode == "target":
        return target
    candidate = diagnostic.get("candidate")
    if mode == "exact":
        return candidate if diagnostic.get("exact") else None
    return candidate


def augment_record(row, case, tokenizer, args):
    schedule = row.get("schedule") or []
    diagnostics = build_diagnostic_map(row, case)
    groups = group_argument_items(schedule)
    tool_name_groups = group_tool_name_items(schedule)
    totals = Counter()

    for _, items in groups.items():
        first = sorted(items, key=lambda item: int(item["token_start"]))[0]
        tool_idx = int(first.get("tool_call_index"))
        json_key = first.get("json_key")
        target = target_value(first)
        diagnostic = diagnostics.get((tool_idx, json_key, normalize_for_compare(target)))
        if not diagnostic:
            totals["argument_blocks_without_diagnostic"] += 1
            continue
        candidate_values = list(diagnostic.get("candidate_set") or [])
        if args.include_target_candidate:
            add_target_candidate(candidate_values, target)
        selected_candidate = choose_selected_candidate(args.selected_candidate_mode, target, diagnostic)
        target_ids = full_target_ids(items)
        if not target_ids or not candidate_values:
            totals["argument_blocks_without_candidates"] += 1
            continue
        allowed = candidate_allowed_by_offset(tokenizer, target_ids, candidate_values)
        selected = selected_candidate_by_offset(tokenizer, target_ids, selected_candidate)
        sequences = compatible_candidate_sequences(tokenizer, target_ids, candidate_values, target=target)
        totals["argument_blocks_augmented"] += 1
        totals["argument_candidate_values"] += len(candidate_values)
        totals["argument_candidate_positions"] += len(allowed)
        totals["argument_candidate_multi_id_positions"] += sum(int(len(ids) > 1) for ids in allowed)
        totals["argument_blocks_with_selected_candidate"] += int(bool(selected))
        totals["argument_blocks_with_sequence_candidates"] += int(bool(sequences))
        totals["argument_sequence_candidates"] += len(sequences)

        for item in items:
            offset = int(item["token_start"]) - int(first["token_start"])
            count = int(item["token_count"])
            item["candidate_values"] = candidate_values
            item["candidate_allowed_token_ids_by_offset"] = allowed[offset : offset + count]
            if sequences:
                item["candidate_sequence_values"] = [sequence["value"] for sequence in sequences]
                item["candidate_sequence_token_ids_by_offset"] = [
                    sequence["token_ids_by_offset"][offset : offset + count]
                    for sequence in sequences
                ]
            if selected:
                item["selected_candidate"] = selected_candidate
                item["selected_candidate_token_ids_by_offset"] = selected[offset : offset + count]
            item["candidate_target_in_set"] = bool(diagnostic.get("target_in_candidates"))
            item["candidate_exact_single_choice"] = bool(diagnostic.get("exact"))
            item["candidate_source"] = diagnostic.get("candidate_source")
            item["selected_candidate_mode"] = args.selected_candidate_mode
            item["candidate_schema_type"] = diagnostic.get("schema_type")

    available_tool_names = [name for name in (tool_name(tool) for tool in case.get("tools") or []) if name]
    for _, items in tool_name_groups.items():
        first = sorted(items, key=lambda item: int(item["token_start"]))[0]
        target = target_value(first)
        target_ids = full_target_ids(items)
        if not target_ids or not available_tool_names:
            totals["tool_name_blocks_without_candidates"] += 1
            continue
        sequences = compatible_tool_name_sequences(tokenizer, target_ids, available_tool_names)
        target_in_sequences = target in [sequence["value"] for sequence in sequences]
        totals["tool_name_blocks_augmented"] += 1
        totals["tool_name_candidate_values"] += len(available_tool_names)
        totals["tool_name_sequence_candidates"] += len(sequences)
        totals["tool_name_blocks_with_sequence_candidates"] += int(bool(sequences))
        totals["tool_name_blocks_with_target_candidate"] += int(bool(target_in_sequences))

        for item in items:
            offset = int(item["token_start"]) - int(first["token_start"])
            count = int(item["token_count"])
            item["candidate_values"] = available_tool_names
            item["candidate_sequence_values"] = [sequence["value"] for sequence in sequences]
            item["candidate_sequence_token_ids_by_offset"] = [
                sequence["token_ids_by_offset"][offset : offset + count]
                for sequence in sequences
            ]
            item["candidate_target_in_set"] = bool(target in available_tool_names)
            item["candidate_target_in_length_compatible_set"] = bool(target_in_sequences)
            item["candidate_source"] = "available_tools"
            item["candidate_schema_type"] = "tool_name"

    return totals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule-jsonl", type=Path, required=True)
    parser.add_argument("--cases-jsonl", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument(
        "--include-target-candidate",
        action="store_true",
        help="Add the schedule target value to each argument candidate set when absent.",
    )
    parser.add_argument(
        "--selected-candidate-mode",
        choices=["evidence", "exact", "target", "none"],
        default="evidence",
        help=(
            "How to populate selected_candidate_token_ids_by_offset for argument spans. "
            "'evidence' preserves the old extractor guess; 'target' selects the schedule target."
        ),
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_path), trust_remote_code=True)
    cases = {case_key(row, idx): row for idx, row in enumerate(load_jsonl(args.cases_jsonl))}
    records = []
    totals = Counter()
    for idx, row in enumerate(load_jsonl(args.schedule_jsonl)):
        key = case_key(row, idx)
        case = cases.get(key)
        if not case:
            totals["missing_case_records"] += 1
            continue
        totals["records"] += 1
        totals.update(augment_record(row, case, tokenizer, args))
        records.append(row)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "schedule_jsonl": str(args.schedule_jsonl),
        "cases_jsonl": str(args.cases_jsonl),
        "tokenizer_path": str(args.tokenizer_path),
        "out_jsonl": str(args.out_jsonl),
        "include_target_candidate": args.include_target_candidate,
        "selected_candidate_mode": args.selected_candidate_mode,
        "totals": dict(totals),
    }
    args.out_jsonl.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
