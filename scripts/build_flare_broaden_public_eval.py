#!/usr/bin/env python3
"""Build the broadened public-derived matched eval slice.

The output keeps the existing north-star 20 episodes first, then appends
leak-checked ToolACE-derived multi-turn tool-call episodes transcoded to the
Qwen-native XML function/parameter format.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from audit_toolcall_eval_overlap import normalize_text  # noqa: E402
from eval_toolcall_jsonl import extract_tool_calls, qwen_native_tool_call_text  # noqa: E402


DEFAULT_EXISTING = ROOT / "data/toolcall_eval_native/flare_scaleup_native_58.jsonl"
DEFAULT_OUT = ROOT / "data/toolcall_eval_native/flare_broaden_public_toolace.jsonl"
DEFAULT_MANIFEST = ROOT / "data/toolcall_eval_native/flare_broaden_public_toolace.manifest.json"
DEFAULT_MODEL = ROOT / "models/qwen3.5-9b-fastdllm-b1000-vllm-bf16"
DEFAULT_CHAT_TEMPLATE = Path("/home/mark/shared/lumoFlyWheel_codex_fork/docker/chat_templates/qwen3-openai-codex.jinja")
DEFAULT_XLAM_PROBE = ROOT / "data/toolcall_seed/xlam_probe.manifest.json"
TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*.*?\s*</tool_call>", re.DOTALL)
ASSISTANT_GENERATION_PROMPT = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-jsonl", type=Path, default=DEFAULT_EXISTING)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--rejected-jsonl", type=Path, default=None)
    parser.add_argument("--xlam-probe-json", type=Path, default=DEFAULT_XLAM_PROBE)
    parser.add_argument("--toolace-dataset", default="Team-ACE/ToolACE")
    parser.add_argument("--toolace-split", default="train")
    parser.add_argument("--existing-count", type=int, default=20)
    parser.add_argument("--existing-min-turns", type=int, default=3)
    parser.add_argument("--existing-max-turns", type=int, default=6)
    parser.add_argument("--toolace-count", type=int, default=40)
    parser.add_argument("--max-toolace-count", type=int, default=60)
    parser.add_argument("--public-min-turns", type=int, default=3)
    parser.add_argument("--public-max-turns", type=int, default=5)
    parser.add_argument("--min-total-episodes", type=int, default=60)
    parser.add_argument("--min-total-turns", type=int, default=180)
    parser.add_argument("--max-stream-rows", type=int, default=50000)
    parser.add_argument("--max-tools", type=int, default=10)
    parser.add_argument("--prompt-tokenizer-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--chat-template-path", type=Path, default=DEFAULT_CHAT_TEMPLATE)
    parser.add_argument("--max-prompt-tokens", type=int, default=3500)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--skip-token-filter", action="store_true")
    parser.add_argument("--train-path", type=Path, action="append", default=None)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def split_tool_call_blocks(text: str) -> list[str]:
    return [match.group(0).strip() for match in TOOL_CALL_BLOCK_RE.finditer(text or "")]


def safe_identifier(value: Any, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "")).strip("_")
    text = re.sub(r"_+", "_", text)
    if not text:
        text = fallback
    if not re.match(r"^[A-Za-z_]", text):
        text = f"{fallback}_{text}"
    return text


def unique_identifier(base: str, used: set[str]) -> str:
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def schema_type(value: Any) -> Any:
    if isinstance(value, list):
        return [schema_type(item) for item in value]
    mapping = {
        "dict": "object",
        "object": "object",
        "json": "object",
        "list": "array",
        "array": "array",
        "tuple": "array",
        "str": "string",
        "string": "string",
        "text": "string",
        "int": "integer",
        "integer": "integer",
        "float": "number",
        "double": "number",
        "number": "number",
        "bool": "boolean",
        "boolean": "boolean",
    }
    return mapping.get(str(value).lower(), value)


def convert_schema(schema: Any, *, top_level_name_map: dict[str, str] | None = None) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}, "required": []}
    out: dict[str, Any] = {}
    expected_type = schema_type(schema.get("type"))
    if expected_type is None and isinstance(schema.get("properties"), dict):
        expected_type = "object"
    if expected_type is not None:
        out["type"] = expected_type
    for key in ("description", "enum", "default"):
        if key in schema:
            out[key] = copy.deepcopy(schema[key])
    if isinstance(schema.get("properties"), dict):
        properties = {}
        for prop_name, prop_schema in schema["properties"].items():
            mapped = top_level_name_map.get(prop_name, prop_name) if top_level_name_map else prop_name
            properties[mapped] = convert_schema(prop_schema)
        out["properties"] = properties
        required = schema.get("required") or []
        if isinstance(required, list):
            out["required"] = [
                top_level_name_map.get(item, item) if top_level_name_map else item
                for item in required
                if isinstance(item, str)
            ]
        else:
            out["required"] = []
    if "items" in schema:
        out["items"] = convert_schema(schema["items"])
    if not out:
        out = {"type": "object", "properties": {}, "required": []}
    if out.get("type") == "object":
        out.setdefault("properties", {})
        out.setdefault("required", [])
    return out


def extract_json_array(text: str) -> list[dict[str, Any]]:
    start = text.find("[{")
    if start < 0:
        raise ValueError("no JSON function array found")
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                payload = json.loads(text[start : idx + 1])
                if not isinstance(payload, list):
                    raise ValueError("function payload is not a list")
                return [item for item in payload if isinstance(item, dict)]
    raise ValueError("unterminated JSON function array")


def convert_tools(raw_tools: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    used_tool_names: set[str] = set()
    mappings: dict[str, dict[str, Any]] = {}
    converted = []
    for idx, tool in enumerate(raw_tools):
        original_name = str(tool.get("name") or f"tool_{idx}")
        safe_name = unique_identifier(safe_identifier(original_name, "tool"), used_tool_names)
        params = tool.get("parameters") or {}
        used_param_names: set[str] = set()
        param_map: dict[str, str] = {}
        for prop_name in (params.get("properties") or {}) if isinstance(params, dict) else {}:
            param_map[str(prop_name)] = unique_identifier(safe_identifier(prop_name, "arg"), used_param_names)
        converted_schema = convert_schema(params, top_level_name_map=param_map)
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": safe_name,
                    "description": str(tool.get("description") or ""),
                    "parameters": converted_schema,
                },
            }
        )
        mappings[original_name] = {
            "safe_name": safe_name,
            "param_map": param_map,
            "original_name": original_name,
        }
    return converted, mappings


def matching_paren(text: str, open_idx: int) -> int:
    depth = 0
    quote: str | None = None
    escape = False
    for idx in range(open_idx, len(text)):
        char = text[idx]
        if quote is not None:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return idx
    return -1


def split_top_level(text: str, delimiter: str = ",") -> list[str]:
    parts = []
    start = 0
    depth = 0
    quote: str | None = None
    escape = False
    for idx, char in enumerate(text):
        if quote is not None:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == delimiter and depth == 0:
            parts.append(text[start:idx].strip())
            start = idx + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def split_key_value(text: str) -> tuple[str, str] | None:
    depth = 0
    quote: str | None = None
    escape = False
    for idx, char in enumerate(text):
        if quote is not None:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "=" and depth == 0:
            return text[:idx].strip(), text[idx + 1 :].strip()
    return None


def parse_value(text: str) -> Any:
    value = text.strip()
    if not value:
        return ""
    lower = value.lower()
    if lower in {"true", "false", "null", "none"}:
        return {"true": True, "false": False, "null": None, "none": None}[lower]
    try:
        return ast.literal_eval(value)
    except Exception:
        pass
    try:
        return json.loads(value)
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return value.strip("\"'")


def parse_arguments(text: str) -> dict[str, Any]:
    arguments: dict[str, Any] = {}
    if not text.strip():
        return arguments
    for part in split_top_level(text):
        key_value = split_key_value(part)
        if key_value is None:
            continue
        key, value = key_value
        if key:
            arguments[key] = parse_value(value)
    return arguments


def parse_toolace_calls(text: str) -> list[dict[str, Any]]:
    stripped = str(text or "").strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return []
    body = stripped[1:-1].strip()
    calls = []
    cursor = 0
    while cursor < len(body):
        while cursor < len(body) and body[cursor].isspace():
            cursor += 1
        open_idx = body.find("(", cursor)
        if open_idx < 0:
            break
        name = body[cursor:open_idx].strip()
        close_idx = matching_paren(body, open_idx)
        if close_idx < 0 or not name:
            return []
        calls.append({"name": name, "arguments": parse_arguments(body[open_idx + 1 : close_idx])})
        cursor = close_idx + 1
        while cursor < len(body) and body[cursor].isspace():
            cursor += 1
        if cursor < len(body):
            if body[cursor] != ",":
                return []
            cursor += 1
    return calls


def convert_call(call: dict[str, Any], tool_map: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    mapping = tool_map.get(str(call.get("name") or ""))
    if mapping is None:
        return None
    param_map = mapping.get("param_map") or {}
    used = set(param_map.values())
    arguments = {}
    for key, value in (call.get("arguments") or {}).items():
        safe_key = param_map.get(str(key))
        if safe_key is None:
            safe_key = unique_identifier(safe_identifier(key, "arg"), used)
        arguments[safe_key] = value
    return {"name": mapping["safe_name"], "arguments": arguments}


def content_or_json(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def continuation_suffix(payload: Any, next_user_message: str | None = None) -> str:
    content = content_or_json(payload)
    suffix = (
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "<tool_response>\n"
        + content
        + "\n</tool_response><|im_end|>\n"
    )
    if next_user_message is not None and str(next_user_message).strip():
        suffix += "<|im_start|>user\n" + str(next_user_message).strip() + "<|im_end|>\n"
    return suffix + ASSISTANT_GENERATION_PROMPT


def gold_tool_payload(gold_block: str, episode_id: str, turn_idx: int) -> dict[str, Any]:
    calls, invalid = extract_tool_calls(gold_block)
    if invalid or not calls:
        return {
            "ok": False,
            "error": "gold_tool_call_invalid_or_missing",
            "result_id": f"{episode_id}_turn_{turn_idx}",
        }
    call = calls[0]
    return {
        "ok": True,
        "tool": call.get("name"),
        "arguments": call.get("arguments") or {},
        "result_id": f"{episode_id}_turn_{turn_idx}",
        "summary": f"synthetic result for {call.get('name')}",
    }


def render_prompt(tokenizer, chat_template: str | None, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    if chat_template is not None:
        kwargs["chat_template"] = chat_template
    if tools:
        kwargs["tools"] = tools
    return tokenizer.apply_chat_template(messages, **kwargs)


def max_gold_path_tokens(tokenizer, chat_template: str | None, row: dict[str, Any]) -> tuple[int, int]:
    prompt = render_prompt(tokenizer, chat_template, row["prompt_messages"], row.get("tools") or [])
    max_prompt_tokens = 0
    max_gold_tokens = 0
    blocks = split_tool_call_blocks(row.get("gold_assistant") or "")
    turn_user_messages = row.get("turn_user_messages") or []
    for turn_idx, block in enumerate(blocks):
        max_prompt_tokens = max(max_prompt_tokens, len(tokenizer(prompt, add_special_tokens=False).input_ids))
        max_gold_tokens = max(max_gold_tokens, len(tokenizer(block, add_special_tokens=False).input_ids))
        next_user = None
        if turn_idx + 1 < len(turn_user_messages):
            next_user = turn_user_messages[turn_idx + 1]
        prompt += block + continuation_suffix(gold_tool_payload(block, row["id"], turn_idx), next_user)
    return max_prompt_tokens, max_gold_tokens


def toolace_row_to_episode(row: dict[str, Any], row_idx: int) -> tuple[dict[str, Any] | None, str]:
    try:
        raw_tools = extract_json_array(str(row.get("system") or ""))
    except Exception as exc:
        return None, f"tool_parse_error:{type(exc).__name__}"
    tools, tool_map = convert_tools(raw_tools)
    conversations = row.get("conversations") or []
    if not isinstance(conversations, list):
        return None, "bad_conversation_shape"
    blocks: list[str] = []
    turn_user_messages: list[str | None] = []
    first_user: str | None = None
    pending_user: str | None = None
    original_call_names: list[str] = []
    for message in conversations:
        role = message.get("from") or message.get("role")
        raw_value = message.get("value") if "value" in message else message.get("content")
        value = str(raw_value or "").strip()
        if role == "user":
            pending_user = value
            continue
        if role != "assistant":
            continue
        calls = parse_toolace_calls(value)
        if not calls:
            continue
        if pending_user is None:
            return None, "tool_call_without_preceding_user"
        for call_idx, call in enumerate(calls):
            converted_call = convert_call(call, tool_map)
            if converted_call is None:
                return None, "unknown_tool_name"
            if not blocks:
                first_user = pending_user
                turn_user_messages.append(None)
            elif call_idx == 0:
                turn_user_messages.append(pending_user)
            else:
                turn_user_messages.append(None)
            original_call_names.append(str(call.get("name") or ""))
            blocks.append(qwen_native_tool_call_text([converted_call]))
        pending_user = None
    if not blocks or first_user is None:
        return None, "no_tool_call_turns"
    episode_id = f"toolace_train_{row_idx:06d}"
    prompt_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": first_user},
    ]
    gold_assistant = "\n".join(blocks)
    gold_calls, gold_invalid = extract_tool_calls(gold_assistant)
    episode = {
        "source": "toolace_derived",
        "source_family": "ToolACE-derived",
        "source_dataset": "Team-ACE/ToolACE",
        "source_split": "train",
        "source_license": "Apache-2.0",
        "source_row_idx": row_idx,
        "id": episode_id,
        "tools": tools,
        "prompt_messages": prompt_messages,
        "turn_user_messages": turn_user_messages,
        "gold_assistant": gold_assistant,
        "gold_assistant_format": "qwen_native_function_parameter",
        "gold_tool_calls": gold_calls,
        "gold_invalid_tool_json_count": gold_invalid,
        "toolace_original_call_names": original_call_names,
        "toolace_name_sanitization": {
            name: {
                "safe_name": item["safe_name"],
                "param_map": item.get("param_map") or {},
            }
            for name, item in sorted(tool_map.items())
        },
    }
    episode["public_eval_hash"] = sha256_json(
        {
            "source_dataset": episode["source_dataset"],
            "source_row_idx": row_idx,
            "prompt_messages": prompt_messages,
            "turn_user_messages": turn_user_messages,
            "tools": tools,
            "gold_assistant": gold_assistant,
        }
    )
    return episode, "ok"


def messages_for_instance(instance: dict[str, Any]) -> list[dict[str, str]]:
    messages = []
    if instance.get("prompt_messages"):
        messages.extend(copy.deepcopy(instance.get("prompt_messages") or []))
    elif instance.get("messages"):
        if instance.get("system"):
            messages.append({"role": "system", "content": str(instance.get("system") or "")})
        messages.extend(copy.deepcopy(instance.get("messages") or []))
    if instance.get("gold_assistant") is not None:
        messages.append({"role": "assistant", "content": str(instance.get("gold_assistant") or "")})
    return [
        {"role": str(message.get("role") or ""), "content": str(message.get("content") or "")}
        for message in messages
        if isinstance(message, dict)
    ]


def row_user_text(row: dict[str, Any]) -> str:
    chunks = [
        str(message.get("content") or "").strip()
        for message in row.get("prompt_messages") or row.get("messages") or []
        if isinstance(message, dict) and message.get("role") == "user" and str(message.get("content") or "").strip()
    ]
    for value in row.get("turn_user_messages") or []:
        if value is not None and str(value).strip():
            chunks.append(str(value).strip())
    return "\n\n".join(chunks)


def row_assistant_text(row: dict[str, Any]) -> str:
    if row.get("gold_assistant") is not None:
        return str(row.get("gold_assistant") or "")
    return "\n".join(
        str(message.get("content") or "")
        for message in row.get("messages") or []
        if isinstance(message, dict) and message.get("role") == "assistant"
    )


def row_instance(row: dict[str, Any]) -> dict[str, Any]:
    messages = []
    for message in row.get("prompt_messages") or row.get("messages") or []:
        if isinstance(message, dict):
            messages.append({"role": str(message.get("role") or ""), "content": str(message.get("content") or "")})
    if row.get("gold_assistant") is not None:
        blocks = split_tool_call_blocks(str(row.get("gold_assistant") or ""))
        turn_user_messages = row.get("turn_user_messages") or []
        if blocks:
            for turn_idx, block in enumerate(blocks):
                messages.append({"role": "assistant", "content": block})
                if turn_idx + 1 < len(turn_user_messages):
                    next_user = turn_user_messages[turn_idx + 1]
                    if next_user is not None and str(next_user).strip():
                        messages.append({"role": "user", "content": str(next_user).strip()})
        else:
            messages.extend(messages_for_instance(row)[len(messages) :])
    system = ""
    filtered = []
    for message in messages:
        role = message.get("role")
        content = str(message.get("content") or "")
        if role == "system" and not system:
            system = content
        elif role in {"user", "assistant", "tool"} and content:
            filtered.append({"role": role, "content": content})
    return {"system": system or "You are a helpful assistant.", "messages": filtered, "tools": row.get("tools") or []}


def instance_fingerprint(instance: dict[str, Any]) -> str:
    return sha256_json(
        {
            "system": instance.get("system"),
            "messages": instance.get("messages") or [],
            "tools": instance.get("tools") or [],
        }
    )


def user_fingerprint(text: str) -> str:
    return sha256_text(normalize_text(text))


def normalize_scalar(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return f"bool:{str(value).lower()}"
    if isinstance(value, (int, float)):
        return f"num:{value}"
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return None
    return f"str:{text}"


def leaf_values(value: Any) -> list[str]:
    if isinstance(value, dict):
        out: list[str] = []
        for item in value.values():
            out.extend(leaf_values(item))
        return out
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(leaf_values(item))
        return out
    normalized = normalize_scalar(value)
    return [normalized] if normalized is not None else []


def distinctive_values(values: set[str]) -> set[str]:
    out = set()
    for value in values:
        _, _, raw = value.partition(":")
        if value.startswith("num:"):
            out.add(value)
        elif value.startswith("str:") and (len(raw) >= 3 or any(ch.isdigit() for ch in raw) or any(ch in raw for ch in "-_@:/.")):
            out.add(value)
    return out


def call_signatures_from_text(text: str) -> list[dict[str, Any]]:
    calls, _ = extract_tool_calls(text)
    signatures = []
    for idx, call in enumerate(calls):
        values = set(leaf_values(call.get("arguments") or {}))
        signatures.append(
            {
                "name": safe_identifier(call.get("name") or "", "tool"),
                "values": values,
                "distinctive_values": distinctive_values(values),
                "call_index": idx,
            }
        )
    return signatures


def default_train_paths() -> list[Path]:
    paths = []
    for pattern in ("train_agentic_mix.json", "train_toolcall*.json", "*.train.json"):
        paths.extend((ROOT / "data").rglob(pattern))
    return sorted({path.resolve() for path in paths if path.is_file()})


def load_instances(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("instances"), list):
        return [item for item in payload["instances"] if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def build_train_reference(paths: list[Path]) -> dict[str, Any]:
    exact_hashes: set[str] = set()
    user_hashes: set[str] = set()
    calls_by_name: dict[str, list[dict[str, Any]]] = {}
    file_counts = {}
    skipped = []
    total_records = 0
    total_calls = 0
    for path in paths:
        try:
            instances = load_instances(path)
        except Exception as exc:
            skipped.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        file_counts[str(path)] = len(instances)
        total_records += len(instances)
        for idx, instance in enumerate(instances):
            exact_hashes.add(instance_fingerprint(row_instance(instance)))
            user_hashes.add(user_fingerprint(row_user_text(instance)))
            for sig in call_signatures_from_text(row_assistant_text(instance)):
                if not sig["name"]:
                    continue
                calls_by_name.setdefault(sig["name"], []).append({"train_path": str(path), "train_idx": idx, **sig})
                total_calls += 1
    return {
        "paths": [str(path) for path in paths],
        "file_counts": file_counts,
        "skipped": skipped,
        "records": total_records,
        "calls": total_calls,
        "exact_hashes": exact_hashes,
        "user_hashes": user_hashes,
        "calls_by_name": calls_by_name,
    }


def near_leaks(row: dict[str, Any], train_ref: dict[str, Any]) -> list[dict[str, Any]]:
    leaks = []
    for sig in call_signatures_from_text(row_assistant_text(row)):
        values = sig.get("distinctive_values") or sig.get("values") or set()
        if not values:
            continue
        for train_call in train_ref["calls_by_name"].get(sig["name"], []):
            train_values = train_call.get("values") or set()
            if values.issubset(train_values):
                leaks.append(
                    {
                        "type": "same_tool_all_eval_arg_values",
                        "train_path": train_call["train_path"],
                        "train_idx": train_call["train_idx"],
                        "tool": sig["name"],
                        "matched_values": sorted(values),
                    }
                )
                break
    return leaks


def leak_reasons(row: dict[str, Any], train_ref: dict[str, Any], seen: dict[str, set[str]]) -> tuple[list[str], list[dict[str, Any]]]:
    instance_fp = instance_fingerprint(row_instance(row))
    user_fp = user_fingerprint(row_user_text(row))
    reasons = []
    if instance_fp in seen["instance"]:
        reasons.append("slice_exact_duplicate")
    if user_fp in seen["user"]:
        reasons.append("slice_user_duplicate")
    if instance_fp in train_ref["exact_hashes"]:
        reasons.append("train_exact_instance_overlap")
    if user_fp in train_ref["user_hashes"]:
        reasons.append("train_user_overlap")
    leaks = near_leaks(row, train_ref)
    if leaks:
        reasons.append("train_same_tool_all_eval_arg_values")
    return reasons, leaks


def mark_seen(row: dict[str, Any], seen: dict[str, set[str]]) -> None:
    seen["instance"].add(instance_fingerprint(row_instance(row)))
    seen["user"].add(user_fingerprint(row_user_text(row)))


def load_chat_template(path: Path | None) -> str | None:
    if path is None:
        return None
    return path.read_text(encoding="utf-8")


def token_filter(row: dict[str, Any], tokenizer, chat_template: str | None, args: argparse.Namespace) -> tuple[bool, dict[str, int]]:
    max_prompt, max_gold = max_gold_path_tokens(tokenizer, chat_template, row)
    stats = {"max_gold_path_prompt_tokens": max_prompt, "max_gold_block_tokens": max_gold}
    return max_prompt <= args.max_prompt_tokens and max_gold <= args.max_new_tokens, stats


def select_existing_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = []
    for row in load_jsonl(args.existing_jsonl):
        blocks = split_tool_call_blocks(row.get("gold_assistant") or "")
        if len(blocks) < args.existing_min_turns or len(blocks) > args.existing_max_turns:
            continue
        item = copy.deepcopy(row)
        item["source_family"] = "our-synthetic"
        item["source_dataset"] = str(args.existing_jsonl)
        item["source_license"] = "internal-clean-synthetic"
        item["public_eval_hash"] = sha256_json(
            {
                "id": item.get("id"),
                "prompt_messages": item.get("prompt_messages") or [],
                "tools": item.get("tools") or [],
                "gold_assistant": item.get("gold_assistant") or "",
            }
        )
        item.setdefault("turn_user_messages", [None for _ in blocks])
        selected.append(item)
        if len(selected) >= args.existing_count:
            return selected
    raise SystemExit(f"only selected {len(selected)} existing rows from {args.existing_jsonl}")


def build_toolace_rows(
    args: argparse.Namespace,
    train_ref: dict[str, Any],
    tokenizer,
    chat_template: str | None,
    initial_rows: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reject_counts: Counter = Counter()
    seen = {"instance": set(), "user": set()}
    for item in initial_rows or []:
        mark_seen(item, seen)
    stream = load_dataset(args.toolace_dataset, split=args.toolace_split, streaming=True)
    for row_idx, row in enumerate(stream):
        if row_idx >= args.max_stream_rows:
            break
        candidate, status = toolace_row_to_episode(row, row_idx)
        if candidate is None:
            reject_counts[status] += 1
            continue
        blocks = split_tool_call_blocks(candidate.get("gold_assistant") or "")
        reasons = []
        leaks: list[dict[str, Any]] = []
        if len(blocks) < args.public_min_turns or len(blocks) > args.public_max_turns:
            reasons.append("turn_count_out_of_range")
        if len(candidate.get("tools") or []) > args.max_tools:
            reasons.append("too_many_tools")
        leak_reason, leaks = leak_reasons(candidate, train_ref, seen)
        reasons.extend(leak_reason)
        token_stats = {}
        if not args.skip_token_filter and not reasons:
            token_ok, token_stats = token_filter(candidate, tokenizer, chat_template, args)
            if not token_ok:
                reasons.append("prompt_or_gold_too_long")
        if reasons:
            for reason in reasons:
                reject_counts[reason] += 1
            if len(rejected) < 200:
                rejected.append(
                    {
                        "source_row_idx": row_idx,
                        "id": candidate.get("id"),
                        "reasons": reasons,
                        "turns": len(blocks),
                        "tool_count": len(candidate.get("tools") or []),
                        "token_stats": token_stats,
                        "near_leaks": leaks[:5],
                    }
                )
            continue
        mark_seen(candidate, seen)
        candidate["leak_check"] = {
            "train_exact_instance_overlap": False,
            "train_user_overlap": False,
            "train_same_tool_all_eval_arg_values_overlap": False,
            "train_file_count": len(train_ref["paths"]),
            "train_records": train_ref["records"],
            "train_tool_calls_indexed": train_ref["calls"],
        }
        candidate["token_filter"] = token_stats
        selected.append(candidate)
        total_rows = args.existing_count + len(selected)
        total_turns = sum(len(split_tool_call_blocks(item.get("gold_assistant") or "")) for item in selected)
        total_turns += args.existing_count * 3
        if len(selected) >= args.toolace_count and (
            args.existing_count + len(selected) >= args.min_total_episodes
        ):
            if len(selected) >= args.max_toolace_count or total_turns >= args.min_total_turns:
                break
        if len(selected) >= args.max_toolace_count:
            break
    return selected, rejected, reject_counts


def xlam_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"dataset": "Salesforce/xlam-function-calling-60k", "status": "not_probed"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"dataset": "Salesforce/xlam-function-calling-60k", "status": "probe_unreadable", "error": str(exc)}
    return payload


def manifest_for(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    toolace_rows: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    reject_counts: Counter,
    train_ref: dict[str, Any],
) -> dict[str, Any]:
    source_counts = Counter(row.get("source_family") or row.get("source") or "unknown" for row in rows)
    turn_histogram = Counter(str(len(split_tool_call_blocks(row.get("gold_assistant") or ""))) for row in rows)
    public_turns = sum(len(split_tool_call_blocks(row.get("gold_assistant") or "")) for row in toolace_rows)
    manifest_rows = [
        {
            "id": row.get("id"),
            "source_family": row.get("source_family"),
            "source_dataset": row.get("source_dataset"),
            "source_license": row.get("source_license"),
            "source_row_idx": row.get("source_row_idx"),
            "turns": len(split_tool_call_blocks(row.get("gold_assistant") or "")),
            "public_eval_hash": row.get("public_eval_hash"),
            "tools_hash": sha256_json(row.get("tools") or []),
            "gold_hash": sha256_text(row.get("gold_assistant") or ""),
        }
        for row in rows
    ]
    return {
        "out_jsonl": str(args.out_jsonl),
        "records": len(rows),
        "turns": sum(item["turns"] for item in manifest_rows),
        "source_family_counts": dict(sorted(source_counts.items())),
        "turn_count_histogram": dict(sorted(turn_histogram.items())),
        "episode_set_hash": sha256_json(manifest_rows),
        "rows": manifest_rows,
        "existing_source": {
            "path": str(args.existing_jsonl),
            "selected_count": sum(1 for row in rows if row.get("source_family") == "our-synthetic"),
        },
        "public_sources": {
            "ToolACE": {
                "dataset": args.toolace_dataset,
                "split": args.toolace_split,
                "license": "Apache-2.0",
                "selected_count": len(toolace_rows),
                "selected_turns": public_turns,
                "name_sanitization": "non XML-safe characters collapsed to underscores; collisions receive numeric suffixes",
                "format_transcode": "Pythonic bracket calls -> Qwen-native <tool_call>/<function>/<parameter> blocks",
            },
            "xLAM": {
                "dataset": "Salesforce/xlam-function-calling-60k",
                "license": "CC-BY-4.0",
                "status": "not_used",
                "reason": "local probe indicates gated dataset unavailable without accepted Hugging Face terms/token",
                "probe": xlam_status(args.xlam_probe_json),
            },
        },
        "selection": {
            "existing_count": args.existing_count,
            "toolace_count_min": args.toolace_count,
            "toolace_count_max": args.max_toolace_count,
            "public_turn_range": [args.public_min_turns, args.public_max_turns],
            "max_tools": args.max_tools,
            "max_stream_rows": args.max_stream_rows,
            "max_prompt_tokens_gold_path": args.max_prompt_tokens,
            "max_new_tokens": args.max_new_tokens,
        },
        "train_leak_check": {
            "scope": "ToolACE-derived rows selected for this eval",
            "train_paths": train_ref["paths"],
            "train_file_count": len(train_ref["paths"]),
            "train_records": train_ref["records"],
            "train_tool_calls_indexed": train_ref["calls"],
            "skipped_train_files": train_ref["skipped"],
            "exact_instance_overlaps": 0,
            "user_overlaps": 0,
            "same_tool_all_eval_arg_values_overlaps": 0,
            "run1_copy_mix_covered": str(ROOT / "data/flare_redesign_run1_copy_retention_mix/train_agentic_mix.json")
            in train_ref["paths"],
        },
        "rejected_count_recorded": len(rejected),
        "rejected_reason_counts": dict(sorted(reject_counts.items())),
        "rejected_jsonl": str(args.rejected_jsonl or args.manifest_json.with_suffix(".rejected.jsonl")),
    }


def main() -> int:
    args = parse_args()
    train_paths = [path.resolve() for path in args.train_path] if args.train_path else default_train_paths()
    train_ref = build_train_reference(train_paths)
    chat_template = load_chat_template(args.chat_template_path)
    tokenizer = None
    if not args.skip_token_filter:
        tokenizer = AutoTokenizer.from_pretrained(str(args.prompt_tokenizer_path), trust_remote_code=True)
    existing = select_existing_rows(args)
    toolace_rows, rejected, reject_counts = build_toolace_rows(args, train_ref, tokenizer, chat_template, existing)
    rows = existing + toolace_rows
    total_turns = sum(len(split_tool_call_blocks(row.get("gold_assistant") or "")) for row in rows)
    if len(rows) < args.min_total_episodes:
        raise SystemExit(f"selected {len(rows)} episodes; need >= {args.min_total_episodes}")
    if total_turns < args.min_total_turns:
        raise SystemExit(f"selected {total_turns} turns; need >= {args.min_total_turns}")
    write_jsonl(args.out_jsonl, rows)
    rejected_path = args.rejected_jsonl or args.manifest_json.with_suffix(".rejected.jsonl")
    write_jsonl(rejected_path, rejected)
    manifest = manifest_for(args, rows, toolace_rows, rejected, reject_counts, train_ref)
    write_json(args.manifest_json, manifest)
    print(json.dumps({key: manifest[key] for key in ("records", "turns", "source_family_counts", "episode_set_hash")}, indent=2))
    print(f"wrote {args.out_jsonl}")
    print(f"wrote {args.manifest_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
