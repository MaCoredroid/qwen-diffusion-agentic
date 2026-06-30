#!/usr/bin/env python3
import argparse
import copy
import json
from pathlib import Path

from eval_toolcall_jsonl import extract_tool_calls, qwen_native_tool_call_text


NATIVE_EXAMPLE = (
    "<tool_call>\n"
    "<function=tool_name>\n"
    "<parameter=argument_name>\n"
    "argument value\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>"
)


def native_instruction_text(text):
    if not isinstance(text, str):
        return text
    updated = text
    updated = updated.replace(
        "Return exactly one <tool_call> block with valid JSON and no prose.",
        "Return exactly one Qwen-native <tool_call> block using <function=...> and <parameter=...> tags with no prose.",
    )
    updated = updated.replace(
        "Return only corrected Qwen <tool_call> blocks with JSON",
        "Return only corrected Qwen-native <tool_call> blocks using <function=...> and <parameter=...> tags",
    )
    updated = updated.replace("with JSON payloads", "with native function/parameter payloads")
    updated = updated.replace("with JSON payload", "with native function/parameter payload")
    updated = updated.replace("as Qwen tool-call JSON", "as Qwen-native tool calls")
    updated = updated.replace("Qwen tool-call JSON", "Qwen-native tool calls")
    updated = updated.replace("valid JSON", "valid Qwen-native function/parameter syntax")
    updated = updated.replace(
        '<tool_call>\n{"name": "tool_name", "arguments": {}}\n</tool_call>',
        NATIVE_EXAMPLE,
    )
    return updated


def native_from_text(text):
    calls, invalid = extract_tool_calls(text or "")
    if invalid or not calls:
        return None
    return qwen_native_tool_call_text(calls)


def convert_eval_row(row):
    row = copy.deepcopy(row)
    if "teacher_instruction" in row:
        row["teacher_instruction"] = native_instruction_text(row.get("teacher_instruction"))
    calls = row.get("gold_tool_calls")
    if not calls:
        calls, invalid = extract_tool_calls(row.get("gold_assistant") or "")
        if invalid or not calls:
            return row, False
    if row.get("gold_assistant") and "gold_assistant_legacy" not in row:
        row["gold_assistant_legacy"] = row["gold_assistant"]
    row["gold_assistant"] = qwen_native_tool_call_text(calls)
    row["gold_assistant_format"] = "qwen_native_function_parameter"
    row["gold_tool_calls"] = [
        {"name": call.get("name"), "arguments": call.get("arguments") or {}, "format": "qwen_native"}
        for call in calls
    ]
    return row, True


def convert_instance(instance):
    instance = copy.deepcopy(instance)
    changed = False
    if "system" in instance:
        instance["system"] = native_instruction_text(instance.get("system"))
    if "teacher_instruction" in instance:
        instance["teacher_instruction"] = native_instruction_text(instance.get("teacher_instruction"))
    for message in instance.get("messages") or []:
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            message["content"] = native_instruction_text(message["content"])
    for message in instance.get("messages") or []:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = str(message.get("content") or "")
        native = native_from_text(content)
        if native is None:
            continue
        if "content_legacy" not in message:
            message["content_legacy"] = content
        message["content"] = native
        changed = True
    if "gold_assistant" in instance:
        native = native_from_text(instance.get("gold_assistant") or "")
        if native is not None:
            instance.setdefault("gold_assistant_legacy", instance.get("gold_assistant"))
            instance["gold_assistant"] = native
            instance["gold_assistant_format"] = "qwen_native_function_parameter"
            changed = True
    return instance, changed


def convert_jsonl(input_path, output_path):
    count = 0
    changed = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row, did_change = convert_eval_row(json.loads(line))
            count += 1
            changed += int(did_change)
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"records": count, "converted": changed}


def convert_train_json(input_path, output_path):
    data = json.loads(input_path.read_text(encoding="utf-8"))
    changed = 0
    count = 0
    if isinstance(data, dict) and isinstance(data.get("instances"), list):
        instances = []
        for instance in data["instances"]:
            converted, did_change = convert_instance(instance)
            instances.append(converted)
            count += 1
            changed += int(did_change)
        data = copy.deepcopy(data)
        data["instances"] = instances
    elif isinstance(data, list):
        rows = []
        for instance in data:
            converted, did_change = convert_instance(instance)
            rows.append(converted)
            count += 1
            changed += int(did_change)
        data = rows
    else:
        raise ValueError(f"Unsupported JSON shape in {input_path}")
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"records": count, "converted": changed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=["eval-jsonl", "train-json"], required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.kind == "eval-jsonl":
        summary = convert_jsonl(args.input, args.output)
    else:
        summary = convert_train_json(args.input, args.output)
    summary.update({"input": str(args.input), "output": str(args.output), "kind": args.kind})
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
