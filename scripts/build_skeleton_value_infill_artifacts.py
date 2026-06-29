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

from build_candidate_ranking_examples import (  # noqa: E402
    candidate_groups,
    case_key,
    compact_tools,
    concat_candidate_sequences,
    concat_target_ids,
    load_jsonl,
    local_peer_arguments,
    parse_target_value,
    same_call_arguments,
    target_index,
)
from diagnose_schedule_value_candidates import normalize_for_compare  # noqa: E402
from eval_fastdllm_toolcall_cases import case_context_text  # noqa: E402
from eval_toolcall_jsonl import extract_tool_calls  # noqa: E402


SENSITIVE_KINDS = {
    "tool_tag",
    "tool_name",
    "json_key",
    "json_structure",
    "argument_value",
}


def value_to_json_text(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def value_to_search_strings(value):
    values = []
    if isinstance(value, str):
        values.append(value)
    elif isinstance(value, (int, float, bool)) or value is None:
        values.append(str(value).lower() if isinstance(value, bool) else str(value))
    else:
        values.append(json.dumps(value, ensure_ascii=False))
    rendered = value_to_json_text(value)
    if rendered not in values:
        values.append(rendered)
    out = []
    for item in values:
        item = str(item)
        if item and item not in out:
            out.append(item)
    return out


def evidence_matches(context, value, limit=3, window=80):
    matches = []
    lower = context.lower()
    for needle in value_to_search_strings(value):
        needle_lower = needle.lower()
        if not needle_lower:
            continue
        start = 0
        while len(matches) < limit:
            idx = lower.find(needle_lower, start)
            if idx < 0:
                break
            lo = max(0, idx - window)
            hi = min(len(context), idx + len(needle) + window)
            matches.append(
                {
                    "needle": needle,
                    "start": idx,
                    "end": idx + len(needle),
                    "excerpt": context[lo:hi],
                }
            )
            start = idx + max(1, len(needle))
        if len(matches) >= limit:
            break
    return matches


def parse_path(path):
    if not path:
        return []
    parts = []
    for chunk in str(path).split("."):
        pos = 0
        match = re.match(r"^[^\[]+", chunk)
        if match:
            parts.append(match.group(0))
            pos = match.end()
        for idx_match in re.finditer(r"\[(\d+)\]", chunk[pos:]):
            parts.append(int(idx_match.group(1)))
    return parts


def set_path_placeholder(value, path_parts, placeholder):
    if not path_parts:
        return placeholder
    head, *tail = path_parts
    if isinstance(value, dict):
        out = {key: set_path_placeholder(item, [], item) for key, item in value.items()}
        if head in out:
            out[head] = set_path_placeholder(out[head], tail, placeholder)
        return out
    if isinstance(value, list):
        out = list(value)
        if isinstance(head, int) and 0 <= head < len(out):
            out[head] = set_path_placeholder(out[head], tail, placeholder)
        return out
    return value


def tool_calls_from_case(case):
    calls = case.get("gold_tool_calls")
    if isinstance(calls, list):
        return calls
    parsed, _ = extract_tool_calls(case.get("gold_assistant") or "")
    return parsed


def skeleton_calls_for_slot(case, slot):
    calls = tool_calls_from_case(case)
    out = []
    for call_idx, call in enumerate(calls):
        copied = {
            "name": call.get("name"),
            "arguments": call.get("arguments"),
        }
        if call_idx == slot["tool_call_index"]:
            placeholder = f"<VALUE_SLOT:{slot['json_path'] or slot['json_key']}>"
            copied["arguments"] = set_path_placeholder(
                copied.get("arguments"),
                parse_path(slot.get("json_path") or slot.get("argument_path") or slot.get("json_key")),
                placeholder,
            )
        out.append(copied)
    return out


def all_slot_skeleton_calls(case, slots):
    calls = tool_calls_from_case(case)
    slots_by_call = defaultdict(list)
    for slot in slots:
        slots_by_call[int(slot["tool_call_index"])].append(slot)
    out = []
    for call_idx, call in enumerate(calls):
        args = call.get("arguments")
        for slot in sorted(slots_by_call.get(call_idx, []), key=lambda row: row.get("schedule_token_start") or 0):
            placeholder = f"<VALUE_SLOT:{slot['json_path'] or slot['json_key']}>"
            args = set_path_placeholder(
                args,
                parse_path(slot.get("json_path") or slot.get("argument_path") or slot.get("json_key")),
                placeholder,
            )
        out.append({"name": call.get("name"), "arguments": args})
    return out


def group_slots(schedule_row, case):
    slots = []
    arguments_by_call = same_call_arguments(schedule_row.get("schedule") or [])
    for _, items in sorted(candidate_groups(schedule_row.get("schedule") or []).items()):
        items = sorted(items, key=lambda row: int(row["token_start"]))
        first = items[0]
        if first.get("kind") != "argument_value":
            continue
        try:
            tool_call_index = int(first.get("tool_call_index"))
        except Exception:
            tool_call_index = first.get("tool_call_index")
        target = parse_target_value(first.get("target_text"))
        candidate_values, candidate_token_ids = concat_candidate_sequences(items)
        target_idx = target_index(target, candidate_values)
        schedule_token_start = min(int(item["token_start"]) for item in items)
        schedule_token_end = max(int(item["token_end"]) for item in items)
        current_path = first.get("json_path") or first.get("argument_path") or first.get("json_key")
        call_arguments = arguments_by_call.get(tool_call_index, [])
        annotated = []
        for argument in call_arguments:
            row = dict(argument)
            row["_current_token_start"] = schedule_token_start
            annotated.append(row)
        local_peers = local_peer_arguments(annotated, current_path, first.get("target_text"))
        for row in local_peers:
            row.pop("_current_token_start", None)
        slot = {
            "slot_id": None,
            "id": schedule_row.get("id") or case.get("id"),
            "source": schedule_row.get("source") or case.get("source"),
            "provenance_case_id": case.get("id"),
            "kind": "argument_value",
            "tool_call_index": tool_call_index,
            "json_key": first.get("json_key"),
            "json_path": first.get("json_path") or first.get("argument_path"),
            "argument_path": first.get("argument_path") or first.get("json_path"),
            "schema_type": first.get("candidate_schema_type"),
            "target": target,
            "target_text": first.get("target_text"),
            "target_token_ids": concat_target_ids(items),
            "target_index": target_idx,
            "candidate_count": len(candidate_values),
            "candidate_values": candidate_values,
            "candidate_token_ids": candidate_token_ids,
            "selected_candidate": first.get("selected_candidate")
            or first.get("pairwise_tournament_selected_candidate"),
            "candidate_source": first.get("candidate_source"),
            "candidate_target_in_set": target_idx >= 0,
            "usable_for_value_training": target_idx >= 0 and bool(candidate_token_ids),
            "schedule_token_start": schedule_token_start,
            "schedule_token_end": schedule_token_end,
            "source_block_token_start": min(int(item.get("source_block_token_start", item["token_start"])) for item in items),
            "source_block_token_end": max(int(item.get("source_block_token_end", item["token_end"])) for item in items),
            "local_peer_arguments": local_peers,
        }
        slots.append(slot)
    for idx, slot in enumerate(slots):
        slot["slot_id"] = f"{slot['id']}::call{slot['tool_call_index']}::{idx}::{slot['json_path'] or slot['json_key']}"
    return slots


def boundary_label(item):
    kind = item.get("kind") or "unknown"
    token_count = int(item.get("token_count") or max(1, int(item.get("token_end", 0)) - int(item.get("token_start", 0))))
    if kind in {"tool_tag", "json_key", "json_structure"}:
        block_size = 1
        denoise_steps = max(1, int(item.get("denoise_steps") or 1))
    elif kind == "tool_name":
        block_size = min(4, token_count)
        denoise_steps = max(2, int(item.get("denoise_steps") or 2))
    elif kind == "argument_value":
        block_size = min(8, max(1, token_count))
        denoise_steps = max(8, int(item.get("denoise_steps") or 8))
    else:
        block_size = min(32, max(1, token_count))
        denoise_steps = int(item.get("denoise_steps") or 1)
    return {
        "kind": kind,
        "recommended_block_size": block_size,
        "recommended_denoise_steps": denoise_steps,
        "must_shrink": kind in SENSITIVE_KINDS,
        "must_constrain": kind in SENSITIVE_KINDS,
        "must_be_json_completable": kind in {"tool_tag", "tool_name", "json_key", "json_structure", "argument_value"},
        "structure_or_value": "value" if kind == "argument_value" else "structure" if kind in {"tool_tag", "tool_name", "json_key", "json_structure"} else "other",
    }


def train_prompt(case, slot, all_slots):
    context = case_context_text(case)
    tools = compact_tools(case.get("tools") or [])
    candidates = "\n".join(
        f"{idx}: {json.dumps(value, ensure_ascii=False)}"
        for idx, value in enumerate(slot["candidate_values"])
    )
    parts = [
        "Choose the grounded value for the active tool-call slot.",
        "Use the request, tool schemas, JSON skeleton, call index, and nearby peer arguments.",
        "Return only the exact JSON value, not the candidate index.",
        "",
        "User/tool context:",
        context,
        "",
        "Available tools:",
        json.dumps(tools, ensure_ascii=False, indent=2),
        "",
        "Full skeleton with value slots:",
        json.dumps(all_slot_skeleton_calls(case, all_slots), ensure_ascii=False, indent=2),
        "",
        "Focused skeleton:",
        json.dumps(skeleton_calls_for_slot(case, slot), ensure_ascii=False, indent=2),
        "",
        f"Tool call index: {slot['tool_call_index']}",
        f"JSON key: {slot.get('json_key')}",
        f"JSON path: {slot.get('json_path')}",
        f"Schema type: {slot.get('schema_type')}",
    ]
    if slot.get("local_peer_arguments"):
        parts.append("Nearby peer arguments:")
        for argument in slot["local_peer_arguments"][:12]:
            path = argument.get("json_path") or argument.get("argument_path") or argument.get("json_key")
            parts.append(f"- {path}: {json.dumps(argument.get('target'), ensure_ascii=False)}")
    parts.extend(["", "Candidates:", candidates])
    return "\n".join(parts).strip()


def train_instance(case, slot, all_slots):
    return {
        "messages": [
            {
                "role": "system",
                "content": "You fill one evidence-grounded Qwen tool-call argument value under a fixed JSON skeleton.",
            },
            {"role": "user", "content": train_prompt(case, slot, all_slots)},
            {"role": "assistant", "content": value_to_json_text(slot["target"])},
        ]
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule-jsonl", type=Path, required=True)
    parser.add_argument("--cases-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--provenance-label", default="unspecified")
    parser.add_argument("--promotion-allowed", action="store_true")
    parser.add_argument("--max-evidence-matches", type=int, default=3)
    args = parser.parse_args()

    cases = {case_key(row, idx): row for idx, row in enumerate(load_jsonl(args.cases_jsonl))}
    all_slots = []
    candidate_rows = []
    boundary_rows = []
    train_instances = []
    totals = Counter()

    for idx, schedule_row in enumerate(load_jsonl(args.schedule_jsonl)):
        key = case_key(schedule_row, idx)
        case = cases.get(key)
        if not case:
            totals["missing_case_records"] += 1
            continue
        totals["records"] += 1
        context = case_context_text(case)
        slots = group_slots(schedule_row, case)
        totals["slots"] += len(slots)
        totals["usable_slots"] += sum(int(slot["usable_for_value_training"]) for slot in slots)
        for slot in slots:
            slot["skeleton_calls_all_slots"] = all_slot_skeleton_calls(case, slots)
            slot["skeleton_calls_focused_slot"] = skeleton_calls_for_slot(case, slot)
            slot["context_evidence_matches"] = evidence_matches(context, slot["target"], args.max_evidence_matches)
            all_slots.append(slot)
            if slot["usable_for_value_training"]:
                train_instances.append(train_instance(case, slot, slots))
            for cand_idx, value in enumerate(slot["candidate_values"]):
                is_target = normalize_for_compare(value) == normalize_for_compare(slot["target"])
                candidate_rows.append(
                    {
                        "slot_id": slot["slot_id"],
                        "id": slot["id"],
                        "source": slot["source"],
                        "tool_call_index": slot["tool_call_index"],
                        "json_key": slot["json_key"],
                        "json_path": slot["json_path"],
                        "candidate_index": cand_idx,
                        "candidate_value": value,
                        "candidate_token_ids": slot["candidate_token_ids"][cand_idx]
                        if cand_idx < len(slot["candidate_token_ids"])
                        else [],
                        "is_target": is_target,
                        "is_selected": normalize_for_compare(value)
                        == normalize_for_compare(slot.get("selected_candidate")),
                        "candidate_source": slot.get("candidate_source"),
                        "evidence_matches": evidence_matches(context, value, args.max_evidence_matches),
                    }
                )
        for item in schedule_row.get("schedule") or []:
            label = boundary_label(item)
            boundary_rows.append(
                {
                    "id": schedule_row.get("id") or case.get("id"),
                    "source": schedule_row.get("source") or case.get("source"),
                    "tool_call_index": item.get("tool_call_index"),
                    "json_key": item.get("json_key"),
                    "json_path": item.get("json_path") or item.get("argument_path"),
                    "token_start": item.get("token_start"),
                    "token_end": item.get("token_end"),
                    "token_count": item.get("token_count"),
                    "target_text": item.get("target_text"),
                    **label,
                }
            )
            totals[f"boundary:{label['kind']}"] += 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    slots_path = args.out_dir / "skeleton_value_slots.jsonl"
    candidates_path = args.out_dir / "value_candidate_bank.jsonl"
    boundaries_path = args.out_dir / "boundary_labels.jsonl"
    train_path = args.out_dir / "value_infill_train.json"
    summary_path = args.out_dir / "summary.json"

    write_jsonl(slots_path, all_slots)
    write_jsonl(candidates_path, candidate_rows)
    write_jsonl(boundaries_path, boundary_rows)
    train_path.write_text(
        json.dumps({"type": "conversation", "instances": train_instances}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    totals["candidate_rows"] = len(candidate_rows)
    totals["boundary_rows"] = len(boundary_rows)
    totals["train_instances"] = len(train_instances)
    totals["target_candidate_rows"] = sum(int(row["is_target"]) for row in candidate_rows)
    totals["selected_candidate_rows"] = sum(int(row["is_selected"]) for row in candidate_rows)
    summary = {
        "schedule_jsonl": str(args.schedule_jsonl),
        "cases_jsonl": str(args.cases_jsonl),
        "out_dir": str(args.out_dir),
        "provenance_label": args.provenance_label,
        "promotion_allowed": bool(args.promotion_allowed),
        "artifacts": {
            "skeleton_value_slots": str(slots_path),
            "value_candidate_bank": str(candidates_path),
            "boundary_labels": str(boundaries_path),
            "value_infill_train": str(train_path),
        },
        "totals": dict(totals),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
