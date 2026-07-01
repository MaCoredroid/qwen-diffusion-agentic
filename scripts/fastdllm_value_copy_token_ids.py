#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

from transformers import AutoTokenizer


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
NATIVE_PARAMETER_RE = re.compile(r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Derive token IDs for scalar argument values in Qwen tool-call labels."
    )
    parser.add_argument(
        "--tokenizer",
        default="/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-init",
        help="Tokenizer path or HF id.",
    )
    parser.add_argument(
        "--dataset",
        default="/home/mark/qwen_diffusion/data/qwen35_9b_toolcall_model_repair_curriculum",
        help="LMFlow dataset directory or JSON file.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional manifest path with extracted values and token IDs.",
    )
    return parser.parse_args()


def dataset_json_path(dataset):
    path = Path(dataset)
    if path.is_file():
        return path
    candidates = sorted(path.glob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"No JSON dataset file found under {path}")
    return candidates[0]


def iter_instances(dataset_path):
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    for instance in payload.get("instances", []):
        yield instance


def iter_tool_payloads(text):
    for match in TOOL_CALL_RE.finditer(text or ""):
        body = match.group(1).strip()
        try:
            yield json.loads(body)
        except json.JSONDecodeError:
            continue


def parse_native_parameter_value(raw):
    text = str(raw or "").strip()
    if not text:
        return text
    if text[0] in "[{\"" or text.lower() in {"true", "false", "null"}:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    try:
        if any(ch in text for ch in ".eE"):
            return float(text)
        return int(text)
    except ValueError:
        return text


def iter_native_tool_call_argument_objects(text):
    for match in TOOL_CALL_RE.finditer(text or ""):
        body = match.group(1)
        params = {}
        for param_match in NATIVE_PARAMETER_RE.finditer(body):
            name = param_match.group(1).strip()
            if not name:
                continue
            params[name] = parse_native_parameter_value(param_match.group(2))
        if params:
            yield params


def iter_tool_call_argument_objects(message):
    for payload in iter_tool_payloads(message.get("content", "")):
        yield payload.get("arguments", {})
    yield from iter_native_tool_call_argument_objects(message.get("content", ""))

    for tool_call in message.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") or {}
        if not isinstance(function, dict):
            continue
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass
        yield arguments


def iter_scalars(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from iter_scalars(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_scalars(child)
    elif value is None or isinstance(value, (str, int, float, bool)):
        yield value


def scalar_fragments(value):
    if isinstance(value, str):
        if value:
            yield value
            yield json.dumps(value, ensure_ascii=False)
    elif isinstance(value, bool):
        yield "true" if value else "false"
    elif value is None:
        yield "null"
    else:
        yield json.dumps(value, ensure_ascii=False)


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    dataset_path = dataset_json_path(args.dataset)

    values = []
    fragments = {}
    token_ids = set()
    tool_call_count = 0
    scalar_count = 0

    for instance in iter_instances(dataset_path):
        for message in instance.get("messages", []):
            if message.get("role") != "assistant":
                continue
            for arguments in iter_tool_call_argument_objects(message):
                tool_call_count += 1
                for scalar in iter_scalars(arguments):
                    scalar_count += 1
                    values.append(scalar)
                    for fragment in scalar_fragments(scalar):
                        ids = tokenizer.encode(fragment, add_special_tokens=False)
                        if not ids:
                            continue
                        fragments.setdefault(fragment, ids)
                        token_ids.update(ids)

    ordered_ids = sorted(token_ids)
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "tokenizer": args.tokenizer,
                    "dataset": str(dataset_path),
                    "tool_call_count": tool_call_count,
                    "scalar_count": scalar_count,
                    "unique_scalar_count": len({json.dumps(v, ensure_ascii=False, sort_keys=True) for v in values}),
                    "fragments": fragments,
                    "token_ids": ordered_ids,
                },
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    print(",".join(str(token_id) for token_id in ordered_ids))


if __name__ == "__main__":
    main()
