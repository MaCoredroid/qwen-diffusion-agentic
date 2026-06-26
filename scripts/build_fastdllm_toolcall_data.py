#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

from eval_toolcall_jsonl import extract_json_objects, required_tool_names


DEFAULT_IN = Path("/home/mark/qwen_diffusion/data/toolcall_seed/qwen_toolcall_seed.jsonl")
DEFAULT_TRAIN_DIR = Path("/home/mark/qwen_diffusion/data/fastdllm_toolcall_train")
DEFAULT_EVAL = Path("/home/mark/qwen_diffusion/data/toolcall_eval/fastdllm_toolcall_smoke.jsonl")
DEFAULT_SYSTEM = "You are a helpful assistant."


def first_assistant_index(messages):
    for idx, message in enumerate(messages):
        if message.get("role") == "assistant" and str(message.get("content", "")).strip():
            return idx
    return None


def assistant_tool_names(text):
    names = []
    for obj in extract_json_objects(text):
        if not isinstance(obj, dict):
            continue
        name = obj.get("name") or obj.get("function") or obj.get("tool_name")
        if isinstance(name, dict):
            name = name.get("name")
        if name:
            names.append(str(name))
    return names


def clean_messages(record):
    system = str(record.get("system") or "").strip()
    messages = []
    for message in record.get("messages") or []:
        role = message.get("role")
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        if role == "system":
            system = system or content
        elif role in {"user", "assistant", "tool"}:
            messages.append({"role": role, "content": content})
    return system, messages


def make_training_instance(record, allow_negative):
    tools = record.get("tools") or []
    _, messages = clean_messages(record)
    if not tools or not messages:
        return None
    first = first_assistant_index(messages)
    if first is None or first == 0:
        return None

    # Keep the first supervised assistant turn. This avoids accidentally
    # training on long traces where later observations dominate the tiny run.
    messages = messages[: first + 1]
    if messages[0]["role"] != "user":
        first_user = next((idx for idx, msg in enumerate(messages) if msg["role"] == "user"), None)
        if first_user is None or first_user >= first:
            return None
        messages = messages[first_user : first + 1]

    assistant_text = messages[-1]["content"]
    gold_names = assistant_tool_names(assistant_text)
    available_names = required_tool_names(tools)
    is_qwen_toolcall = bool(gold_names)
    is_plain_negative = not gold_names and record.get("source") == "glaive"

    if is_qwen_toolcall and available_names and not set(gold_names) & available_names:
        return None
    if not is_qwen_toolcall and not (allow_negative and is_plain_negative):
        return None

    return {"messages": messages, "tools": tools, "system": DEFAULT_SYSTEM}


def make_eval_case(record):
    tools = record.get("tools") or []
    _, messages = clean_messages(record)
    first = first_assistant_index(messages)
    if first is None or first == 0:
        return None
    gold_text = messages[first]["content"]
    gold_names = assistant_tool_names(gold_text)
    if not gold_names:
        return None

    prompt_messages = messages[:first]
    if not prompt_messages or prompt_messages[0].get("role") != "system":
        prompt_messages = [{"role": "system", "content": DEFAULT_SYSTEM}, *prompt_messages]

    return {
        "source": record.get("source"),
        "id": record.get("id"),
        "task": record.get("task"),
        "category": record.get("category"),
        "tools": tools,
        "prompt_messages": prompt_messages,
        "gold_assistant": gold_text,
        "gold_tool_names": gold_names,
        "available_tool_names": sorted(required_tool_names(tools)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--eval-out", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--train-count", type=int, default=96)
    parser.add_argument("--eval-count", type=int, default=8)
    parser.add_argument("--negative-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    records = []
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    rng = random.Random(args.seed)
    rng.shuffle(records)

    candidates = []
    source_counts = {}
    for record in records:
        instance = make_training_instance(record, allow_negative=args.negative_count > 0)
        eval_case = make_eval_case(record)
        if instance is not None or eval_case is not None:
            candidates.append((record, instance, eval_case))

    train_instances = []
    selected_keys = set()
    negative_instances = 0
    for record, instance, _ in candidates:
        if instance is None or len(train_instances) >= args.train_count:
            continue
        is_negative = not assistant_tool_names(instance["messages"][-1]["content"])
        if is_negative:
            if negative_instances >= args.negative_count:
                continue
            negative_instances += 1

        train_instances.append(instance)
        key = (record.get("source"), record.get("id"))
        selected_keys.add(key)
        source = record.get("source") or "unknown"
        source_counts[source] = source_counts.get(source, 0) + 1

    eval_cases = []
    seen_eval_keys = set()
    for record, _, eval_case in candidates:
        if eval_case is None:
            continue
        if (record.get("source"), record.get("id")) in selected_keys:
            continue
        key = (eval_case.get("source"), eval_case.get("id"), tuple(eval_case["gold_tool_names"]))
        if key in seen_eval_keys:
            continue
        seen_eval_keys.add(key)
        eval_cases.append(eval_case)
        if len(eval_cases) >= args.eval_count:
            break

    if not train_instances:
        raise SystemExit(f"No training instances produced from {args.input}")
    if not eval_cases:
        raise SystemExit(f"No eval cases produced from {args.input}")

    args.train_dir.mkdir(parents=True, exist_ok=True)
    args.eval_out.parent.mkdir(parents=True, exist_ok=True)

    train_path = args.train_dir / "train_toolcall.json"
    train_payload = {"type": "conversation", "instances": train_instances}
    train_path.write_text(json.dumps(train_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    with args.eval_out.open("w", encoding="utf-8") as f:
        for case in eval_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    manifest = {
        "input": str(args.input),
        "train_path": str(train_path),
        "eval_path": str(args.eval_out),
        "train_count": len(train_instances),
        "eval_count": len(eval_cases),
        "train_source_counts": source_counts,
        "seed": args.seed,
    }
    manifest_path = args.train_dir / "train_toolcall.manifest"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
