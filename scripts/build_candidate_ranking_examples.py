#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/home/mark/qwen_diffusion")
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from diagnose_schedule_value_candidates import normalize_for_compare  # noqa: E402
from eval_fastdllm_toolcall_cases import case_context_text  # noqa: E402


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def case_key(row, fallback_idx):
    return row.get("id") or row.get("case_id") or str(fallback_idx)


def parse_target_value(text):
    if text is None:
        return None
    try:
        return json.loads(text)
    except Exception:
        return str(text).strip().strip('"')


def group_key(item):
    kind = item.get("kind")
    if kind == "tool_name":
        return (
            kind,
            item.get("tool_call_index"),
            item.get("source_block_token_start"),
            item.get("source_block_token_end"),
            item.get("target_text"),
        )
    if kind == "argument_value":
        return (
            kind,
            item.get("tool_call_index"),
            item.get("json_key"),
            item.get("json_path") or item.get("argument_path"),
            item.get("target_text"),
        )
    return None


def candidate_groups(schedule):
    groups = defaultdict(list)
    for item in schedule:
        if item.get("kind") not in {"tool_name", "argument_value"}:
            continue
        if not item.get("candidate_sequence_values"):
            continue
        key = group_key(item)
        if key is None:
            continue
        groups[key].append(item)
    return groups


def concat_target_ids(items):
    ids = []
    for item in sorted(items, key=lambda row: int(row["token_start"])):
        ids.extend(int(token_id) for token_id in item.get("target_token_ids") or [])
    return ids


def concat_candidate_sequences(items):
    ordered = sorted(items, key=lambda row: int(row["token_start"]))
    values = ordered[0].get("candidate_sequence_values") or []
    sequences = [[] for _ in values]
    for item in ordered:
        current = item.get("candidate_sequence_token_ids_by_offset") or []
        if len(current) != len(values):
            return values, []
        for idx, span_ids in enumerate(current):
            sequences[idx].extend(int(token_id) for token_id in span_ids)
    return values, sequences


def target_index(target, candidates):
    normalized_target = normalize_for_compare(target)
    for idx, candidate in enumerate(candidates):
        if normalize_for_compare(candidate) == normalized_target:
            return idx
    return -1


def compact_tools(tools):
    out = []
    for tool in tools or []:
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        if not isinstance(fn, dict):
            continue
        out.append(
            {
                "name": fn.get("name"),
                "description": fn.get("description"),
                "parameters": fn.get("parameters"),
            }
        )
    return out


def ranking_prompt(case, example):
    context = case_context_text(case)
    tools = compact_tools(case.get("tools") or [])
    candidate_lines = "\n".join(
        f"{idx}: {json.dumps(value, ensure_ascii=False)}"
        for idx, value in enumerate(example["candidate_values"])
    )
    parts = [
        "Choose the correct candidate index for a Qwen tool-call trace.",
        "Use the user request, available tools, call index, and argument key.",
        "Resolve derived values when needed: ranges may map to a midpoint scalar, equal rounded splits should preserve totals, and policy enums require applying the stated condition.",
        "Return only the zero-based integer index.",
        "",
        "User/tool context:",
        context,
        "",
        "Available tools:",
        json.dumps(tools, ensure_ascii=False, indent=2),
        "",
        f"Span kind: {example['kind']}",
        f"Tool call index: {example['tool_call_index']}",
    ]
    if example.get("json_key") is not None:
        parts.append(f"JSON key: {example['json_key']}")
    if example.get("json_path"):
        parts.append(f"JSON path: {example['json_path']}")
    if example.get("kind") == "tool_name" and example.get("same_call_arguments"):
        parts.append("Same-call argument sketch:")
        for argument in example["same_call_arguments"]:
            path = argument.get("json_path") or argument.get("argument_path") or argument.get("json_key")
            value = json.dumps(argument.get("target"), ensure_ascii=False)
            parts.append(f"- {path}: {value}")
    if example.get("kind") == "argument_value":
        if example.get("local_peer_arguments"):
            parts.append("Local peer argument sketch:")
            for argument in example["local_peer_arguments"]:
                path = argument.get("json_path") or argument.get("argument_path") or argument.get("json_key")
                value = json.dumps(argument.get("target"), ensure_ascii=False)
                parts.append(f"- {path}: {value}")
        elif example.get("same_call_peer_arguments"):
            parts.append("Same-call peer argument sketch:")
            for argument in example["same_call_peer_arguments"][:12]:
                path = argument.get("json_path") or argument.get("argument_path") or argument.get("json_key")
                value = json.dumps(argument.get("target"), ensure_ascii=False)
                parts.append(f"- {path}: {value}")
    parts.extend(
        [
            "Candidates:",
            candidate_lines,
        ]
    )
    return "\n".join(parts).strip()


def same_call_arguments(schedule):
    out = defaultdict(list)
    seen = set()
    for item in schedule:
        if item.get("kind") != "argument_value":
            continue
        try:
            call_idx = int(item.get("tool_call_index"))
        except Exception:
            continue
        target_text = item.get("target_text")
        key = (
            call_idx,
            item.get("json_key"),
            item.get("json_path") or item.get("argument_path"),
            target_text,
            item.get("source_block_token_start"),
            item.get("source_block_token_end"),
        )
        if key in seen:
            continue
        seen.add(key)
        out[call_idx].append(
            {
                "json_key": item.get("json_key"),
                "json_path": item.get("json_path") or item.get("argument_path"),
                "argument_path": item.get("argument_path") or item.get("json_path"),
                "target": parse_target_value(target_text),
                "target_text": target_text,
                "schedule_token_start": item.get("source_block_token_start") or item.get("token_start"),
                "schedule_token_end": item.get("source_block_token_end") or item.get("token_end"),
            }
        )
    for call_idx, rows in out.items():
        rows.sort(key=lambda row: (row.get("schedule_token_start") is None, row.get("schedule_token_start") or 0))
    return out


def parent_path(path):
    if not path or "." not in path:
        return ""
    return path.rsplit(".", 1)[0]


def path_ancestors(path):
    parts = (path or "").split(".")
    return [".".join(parts[:idx]) for idx in range(len(parts) - 1, 0, -1)]


def peer_arguments(arguments, current_path, current_target_text):
    peers = []
    for argument in arguments:
        arg_path = argument.get("json_path") or argument.get("argument_path") or argument.get("json_key")
        if arg_path == current_path and argument.get("target_text") == current_target_text:
            continue
        peers.append(argument)
    return peers


def local_peer_arguments(arguments, current_path, current_target_text, limit=16):
    peers = peer_arguments(arguments, current_path, current_target_text)
    if not peers:
        return []
    current_parent = parent_path(current_path)
    ancestors = path_ancestors(current_path)

    def score(argument):
        arg_path = argument.get("json_path") or argument.get("argument_path") or argument.get("json_key") or ""
        arg_parent = parent_path(arg_path)
        if arg_parent == current_parent:
            group_score = 0
        elif current_parent and (arg_path.startswith(current_parent + ".") or current_parent.startswith(arg_parent + ".")):
            group_score = 1
        elif any(ancestor and (arg_path == ancestor or arg_path.startswith(ancestor + ".")) for ancestor in ancestors):
            group_score = 2
        elif not current_parent and not arg_parent:
            group_score = 2
        else:
            group_score = 3
        distance = abs((argument.get("schedule_token_start") or 0) - (argument.get("_current_token_start") or 0))
        return (group_score, distance, argument.get("schedule_token_start") or 0)

    return sorted(peers, key=score)[:limit]


def build_examples(schedule_row, case):
    examples = []
    arguments_by_call = same_call_arguments(schedule_row.get("schedule") or [])
    for _, items in sorted(candidate_groups(schedule_row.get("schedule") or {}).items()):
        items = sorted(items, key=lambda row: int(row["token_start"]))
        first = items[0]
        kind = first.get("kind")
        try:
            tool_call_index = int(first.get("tool_call_index"))
        except Exception:
            tool_call_index = first.get("tool_call_index")
        target = parse_target_value(first.get("target_text"))
        candidate_values, candidate_token_ids = concat_candidate_sequences(items)
        target_idx = target_index(target, candidate_values)
        example = {
            "id": schedule_row.get("id") or case.get("id"),
            "source": schedule_row.get("source") or case.get("source"),
            "kind": kind,
            "tool_call_index": tool_call_index,
            "json_key": first.get("json_key"),
            "json_path": first.get("json_path") or first.get("argument_path"),
            "argument_path": first.get("argument_path") or first.get("json_path"),
            "target": target,
            "target_text": first.get("target_text"),
            "target_token_ids": concat_target_ids(items),
            "candidate_values": candidate_values,
            "candidate_token_ids": candidate_token_ids,
            "target_index": target_idx,
            "candidate_count": len(candidate_values),
            "usable_for_training": target_idx >= 0 and bool(candidate_token_ids),
            "schedule_token_start": min(int(item["token_start"]) for item in items),
            "schedule_token_end": max(int(item["token_end"]) for item in items),
        }
        if kind == "argument_value":
            call_arguments = arguments_by_call.get(tool_call_index, [])
            current_path = first.get("json_path") or first.get("argument_path")
            current_start = example["schedule_token_start"]
            annotated_arguments = []
            for argument in call_arguments:
                annotated = dict(argument)
                annotated["_current_token_start"] = current_start
                annotated_arguments.append(annotated)
            same_call_peers = peer_arguments(annotated_arguments, current_path, first.get("target_text"))
            local_peers = local_peer_arguments(annotated_arguments, current_path, first.get("target_text"))
            for argument in same_call_peers:
                argument.pop("_current_token_start", None)
            for argument in local_peers:
                argument.pop("_current_token_start", None)
            example["same_call_peer_arguments"] = same_call_peers
            example["local_peer_arguments"] = local_peers
        if kind == "tool_name":
            example["same_call_arguments"] = arguments_by_call.get(tool_call_index, [])
        example["prompt"] = ranking_prompt(case, example)
        example["answer"] = str(target_idx) if target_idx >= 0 else ""
        examples.append(example)
    return examples


def conversation_instance(example):
    return {
        "messages": [
            {
                "role": "system",
                "content": "You select the correct candidate index for tool-call behavior preservation.",
            },
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["answer"]},
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule-jsonl", type=Path, required=True)
    parser.add_argument("--cases-jsonl", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-train-json", type=Path, default=None)
    args = parser.parse_args()

    cases = {case_key(row, idx): row for idx, row in enumerate(load_jsonl(args.cases_jsonl))}
    examples = []
    totals = Counter()
    for idx, schedule_row in enumerate(load_jsonl(args.schedule_jsonl)):
        key = case_key(schedule_row, idx)
        case = cases.get(key)
        if not case:
            totals["missing_case_records"] += 1
            continue
        totals["records"] += 1
        row_examples = build_examples(schedule_row, case)
        examples.extend(row_examples)
        for example in row_examples:
            totals["examples"] += 1
            totals[f"examples:{example['kind']}"] += 1
            totals["usable_for_training"] += int(bool(example["usable_for_training"]))
            totals[f"usable_for_training:{example['kind']}"] += int(bool(example["usable_for_training"]))
            totals["target_missing_from_candidates"] += int(example["target_index"] < 0)
            totals["candidate_values"] += int(example["candidate_count"])

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")

    train_path = args.out_train_json
    if train_path is None:
        train_path = args.out_jsonl.with_suffix(".train.json")
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_instances = [
        conversation_instance(example)
        for example in examples
        if example.get("usable_for_training")
    ]
    train_path.write_text(
        json.dumps({"type": "conversation", "instances": train_instances}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = {
        "schedule_jsonl": str(args.schedule_jsonl),
        "cases_jsonl": str(args.cases_jsonl),
        "out_jsonl": str(args.out_jsonl),
        "out_train_json": str(train_path),
        "totals": dict(totals),
    }
    args.out_jsonl.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
