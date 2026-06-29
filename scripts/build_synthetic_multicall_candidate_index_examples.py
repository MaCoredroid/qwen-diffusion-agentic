#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

from build_toolcall_format_public_mix import write_jsonl
from eval_toolcall_jsonl import extract_tool_calls


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_INPUT = ROOT / "data/toolcall_eval/synthetic_multicall_failure_analogues.jsonl"
DEFAULT_OUT = ROOT / "data/candidate_ranking/synthetic_multicall_failure_analogue_index_ranking.jsonl"


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def tool_name(tool):
    fn = tool.get("function", tool) if isinstance(tool, dict) else {}
    return str(fn.get("name")) if isinstance(fn, dict) and fn.get("name") else None


def available_tool_names(case):
    return [name for name in (tool_name(tool) for tool in case.get("tools") or []) if name]


def compact_tools(case):
    tools = []
    for tool in case.get("tools") or []:
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        if not isinstance(fn, dict):
            continue
        tools.append(
            {
                "name": fn.get("name"),
                "description": fn.get("description"),
                "parameters": fn.get("parameters"),
            }
        )
    return tools


def user_text(case):
    return "\n\n".join(
        str(message.get("content") or "").strip()
        for message in case.get("prompt_messages") or []
        if message.get("role") == "user" and str(message.get("content") or "").strip()
    )


def quoted_codes(text):
    values = []
    for match in re.finditer(r'"([A-Z0-9][A-Z0-9_.:-]{3,})"', text):
        value = match.group(1).strip()
        if re.search(r"\d", value) and value not in values:
            values.append(value)
    return values


def ranking_prompt(case, kind, tool_call_index, candidate_values, json_key=None):
    candidate_lines = "\n".join(
        f"{idx}: {json.dumps(value, ensure_ascii=False)}"
        for idx, value in enumerate(candidate_values)
    )
    parts = [
        "Choose the correct candidate index for preserving a Qwen tool-call trace.",
        "Use the user request, available tools, call index, and argument key.",
        "Return only the zero-based integer index.",
        "",
        "User request:",
        user_text(case),
        "",
        "Available tools:",
        json.dumps(compact_tools(case), ensure_ascii=False, indent=2),
        "",
        f"Span kind: {kind}",
        f"Tool call index: {tool_call_index}",
    ]
    if json_key is not None:
        parts.append(f"JSON key: {json_key}")
    parts.extend(["Candidates:", candidate_lines])
    return "\n".join(parts).strip()


def make_example(case, kind, tool_call_index, candidate_values, target, json_key=None):
    if target not in candidate_values:
        raise ValueError(f"target {target!r} is not in candidates for {case.get('id')}")
    target_index = candidate_values.index(target)
    return {
        "id": case.get("id"),
        "source": case.get("source"),
        "analogue_family": case.get("analogue_family"),
        "kind": kind,
        "tool_call_index": tool_call_index,
        "json_key": json_key,
        "target": target,
        "target_index": target_index,
        "candidate_values": candidate_values,
        "candidate_count": len(candidate_values),
        "usable_for_training": True,
        "prompt": ranking_prompt(case, kind, tool_call_index, candidate_values, json_key=json_key),
        "answer": str(target_index),
    }


def build_examples(case):
    calls, invalid = extract_tool_calls(case.get("gold_assistant") or "")
    if invalid:
        raise ValueError(f"invalid gold tool calls for {case.get('id')}")
    family = case.get("analogue_family")
    if family == "voice_command_camera":
        target = calls[2]["name"]
        candidates = []
        for name in ["activate_voice_command", "activate_security_cameras", "set_thermostat"]:
            if name in available_tool_names(case):
                candidates.append(name)
        return [make_example(case, "tool_name", 2, candidates, target)]
    if family == "security_installation_codes":
        target = calls[1]["arguments"]["installation_code"]
        candidates = quoted_codes(user_text(case))
        # Keep the diagnostic focused on the three competing security codes.
        candidates = [value for value in candidates if value in {calls[0]["arguments"]["installation_code"], target, calls[2]["arguments"]["system_code"]}]
        return [make_example(case, "argument_value", 1, candidates, target, json_key="installation_code")]
    return []


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
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out-train-json", type=Path, default=None)
    args = parser.parse_args()

    examples = []
    for case in load_jsonl(args.input_jsonl):
        examples.extend(build_examples(case))
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_jsonl, examples)

    train_path = args.out_train_json or args.out_jsonl.with_suffix(".train.json")
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.write_text(
        json.dumps(
            {
                "type": "conversation",
                "instances": [conversation_instance(example) for example in examples if example.get("usable_for_training")],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "input_jsonl": str(args.input_jsonl),
        "out_jsonl": str(args.out_jsonl),
        "out_train_json": str(train_path),
        "examples": len(examples),
        "usable_for_training": sum(int(bool(example.get("usable_for_training"))) for example in examples),
        "family_counts": {},
        "kind_counts": {},
    }
    for example in examples:
        summary["family_counts"][example["analogue_family"]] = summary["family_counts"].get(example["analogue_family"], 0) + 1
        summary["kind_counts"][example["kind"]] = summary["kind_counts"].get(example["kind"], 0) + 1
    args.out_jsonl.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
