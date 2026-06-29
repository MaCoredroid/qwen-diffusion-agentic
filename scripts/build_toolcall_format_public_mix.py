#!/usr/bin/env python3
import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path

from eval_toolcall_jsonl import extract_tool_calls


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_FORMAT_TRAIN = ROOT / "data/qwen35_9b_toolcall_format_curriculum/train_agentic_mix.json"
DEFAULT_PUBLIC_TRAIN = ROOT / "data/fastdllm_toolcall_train/train_toolcall.json"
DEFAULT_PUBLIC_EVAL = ROOT / "data/toolcall_eval/public_onecall_hermes_smoke.jsonl"
DEFAULT_PUBLIC_TEACHER = ROOT / "data/toolcall_eval/public_onecall_hermes_teacher_q36_nvfp4_arg24.jsonl"
DEFAULT_OUT_DIR = ROOT / "data/qwen35_9b_toolcall_format_public_curriculum"
DEFAULT_TEACHER_TRAIN_EVAL = ROOT / "data/toolcall_eval/public_onecall_teacher_train_smoke.jsonl"
DEFAULT_TEACHER_HELDOUT_EVAL = ROOT / "data/toolcall_eval/public_onecall_teacher_heldout_smoke.jsonl"
DEFAULT_SYSTEM = "You are a helpful assistant."


def load_conversation(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    instances = payload.get("instances")
    if not isinstance(instances, list):
        raise ValueError(f"{path} does not contain conversation instances")
    return instances


def load_jsonl(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compact_tool_call(name, arguments):
    payload = {"name": name, "arguments": arguments}
    return "<tool_call>\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ": ")) + "\n</tool_call>"


def compact_calls(calls):
    return "\n".join(compact_tool_call(call["name"], call.get("arguments") or {}) for call in calls)


def assistant_text(instance):
    return "\n".join(
        str(message.get("content") or "")
        for message in instance.get("messages") or []
        if message.get("role") == "assistant"
    )


def call_count(instance):
    calls, invalid = extract_tool_calls(assistant_text(instance))
    return len(calls), invalid


def normalized_instance(instance, source):
    item = copy.deepcopy(instance)
    messages = []
    for message in item.get("messages") or []:
        role = message.get("role")
        content = message.get("content")
        if role in {"system", "user", "assistant", "tool"} and content is not None:
            messages.append({"role": role, "content": str(content).strip()})
    item = {"messages": messages}
    if instance.get("system"):
        item["system"] = str(instance["system"]).strip()
    if instance.get("tools"):
        item["tools"] = copy.deepcopy(instance["tools"])
    if source:
        item["source"] = source
    return item


def public_onecall_instances(path, cap, rng):
    candidates = []
    for instance in load_conversation(path):
        count, invalid = call_count(instance)
        if count == 1 and invalid == 0:
            item = normalized_instance(instance, "public_train_onecall")
            calls, _ = extract_tool_calls(assistant_text(item))
            item["messages"][-1]["content"] = compact_calls(calls)
            candidates.append(item)
    rng.shuffle(candidates)
    return candidates[:cap] if cap >= 0 else candidates


def case_to_instance(case, assistant):
    system = DEFAULT_SYSTEM
    messages = []
    for message in case.get("prompt_messages") or []:
        if message.get("role") == "system":
            system = str(message.get("content") or DEFAULT_SYSTEM)
        elif message.get("role") in {"user", "assistant", "tool"}:
            messages.append({"role": message["role"], "content": str(message.get("content") or "")})
    messages.append({"role": "assistant", "content": assistant})
    return {
        "system": system,
        "tools": copy.deepcopy(case.get("tools") or []),
        "messages": messages,
        "source": "public_teacher_exact_onecall",
    }


def teacher_exact_records(teacher_path, eval_path, cap):
    cases = {case.get("id"): case for case in load_jsonl(eval_path)}
    records = []
    for record in load_jsonl(teacher_path):
        if record.get("status") != "ok":
            continue
        if not record.get("exact_arguments"):
            continue
        case = cases.get(record.get("id"))
        if not case:
            continue
        assistant = compact_calls(record.get("teacher_calls") or [])
        if not assistant:
            continue
        records.append((record, case, case_to_instance(case, assistant)))
    records.sort(key=lambda item: item[0].get("idx", 0))
    return records[:cap] if cap >= 0 else records


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def dedupe(instances):
    seen = set()
    out = []
    for instance in instances:
        key = json.dumps(
            {k: v for k, v in instance.items() if k != "source"},
            sort_keys=True,
            ensure_ascii=False,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(instance)
    return out


def strip_source(instance):
    item = copy.deepcopy(instance)
    item.pop("source", None)
    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format-train", type=Path, default=DEFAULT_FORMAT_TRAIN)
    parser.add_argument("--public-train", type=Path, default=DEFAULT_PUBLIC_TRAIN)
    parser.add_argument("--public-eval", type=Path, default=DEFAULT_PUBLIC_EVAL)
    parser.add_argument("--public-teacher", type=Path, default=DEFAULT_PUBLIC_TEACHER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--teacher-train-eval-out", type=Path, default=DEFAULT_TEACHER_TRAIN_EVAL)
    parser.add_argument("--teacher-heldout-eval-out", type=Path, default=DEFAULT_TEACHER_HELDOUT_EVAL)
    parser.add_argument("--format-cap", type=int, default=96)
    parser.add_argument("--public-train-onecall-cap", type=int, default=40)
    parser.add_argument("--teacher-exact-cap", type=int, default=12)
    parser.add_argument("--heldout-limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=79)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    format_instances = [normalized_instance(item, "format_curriculum") for item in load_conversation(args.format_train)]
    rng.shuffle(format_instances)
    if args.format_cap >= 0:
        format_instances = format_instances[: args.format_cap]

    public_instances = public_onecall_instances(args.public_train, args.public_train_onecall_cap, rng)
    teacher_records = teacher_exact_records(args.public_teacher, args.public_eval, args.teacher_exact_cap)
    teacher_instances = [item[2] for item in teacher_records]
    teacher_train_ids = {item[0].get("id") for item in teacher_records}

    public_cases = load_jsonl(args.public_eval)
    teacher_train_eval = [case for case in public_cases if case.get("id") in teacher_train_ids]
    teacher_heldout_eval = [case for case in public_cases if case.get("id") not in teacher_train_ids][: args.heldout_limit]

    instances = dedupe(format_instances + public_instances + teacher_instances)
    rng.shuffle(instances)
    source_counts = Counter(instance.get("source") or "unknown" for instance in instances)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train_agentic_mix.json"
    train_path.write_text(
        json.dumps({"type": "conversation", "instances": [strip_source(item) for item in instances]}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    write_jsonl(args.teacher_train_eval_out, teacher_train_eval)
    write_jsonl(args.teacher_heldout_eval_out, teacher_heldout_eval)

    manifest = {
        "train_path": str(train_path),
        "count": len(instances),
        "source_counts": dict(sorted(source_counts.items())),
        "seed": args.seed,
        "format_cap": args.format_cap,
        "public_train_onecall_cap": args.public_train_onecall_cap,
        "teacher_exact_cap": args.teacher_exact_cap,
        "teacher_train_ids": sorted(teacher_train_ids),
        "teacher_train_eval_path": str(args.teacher_train_eval_out),
        "teacher_train_eval_count": len(teacher_train_eval),
        "teacher_heldout_eval_path": str(args.teacher_heldout_eval_out),
        "teacher_heldout_eval_count": len(teacher_heldout_eval),
    }
    manifest_path = args.out_dir / "train_agentic_mix.manifest"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
