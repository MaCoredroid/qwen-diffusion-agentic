#!/usr/bin/env python3
import argparse
import ast
import json
import re
from pathlib import Path

from datasets import load_dataset


DEFAULT_OUT = Path("/home/mark/qwen_diffusion/data/toolcall_seed/qwen_toolcall_seed.jsonl")


SOURCES = {
    "hermes": "NousResearch/hermes-function-calling-v1",
    "glaive": "glaiveai/glaive-function-calling-v2",
    "toolace": "Team-ACE/ToolACE",
    "xlam": "Salesforce/xlam-function-calling-60k",
}


CHAT_MARKER_RE = re.compile(r"\b(FUNCTION RESPONSE|ASSISTANT|USER|FUNCTION|TOOL):", re.IGNORECASE)


def parse_maybe_literal(value):
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        try:
            return ast.literal_eval(value)
        except Exception:
            return value


def parse_first_json_value(text, start=0):
    opener = text[start]
    closer = "}" if opener == "{" else "]"
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
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                raw = text[start : idx + 1]
                try:
                    return json.loads(raw), idx + 1
                except Exception:
                    try:
                        return ast.literal_eval(raw), idx + 1
                    except Exception:
                        return None, idx + 1
    return None, len(text)


def iter_json_values(text):
    cursor = 0
    while cursor < len(text):
        positions = [pos for pos in (text.find("{", cursor), text.find("[", cursor)) if pos >= 0]
        if not positions:
            break
        start = min(positions)
        value, end = parse_first_json_value(text, start)
        if value is not None:
            yield value
        cursor = max(end, start + 1)


def normalize_tool_defs(value):
    value = parse_maybe_literal(value)
    tools = []
    if isinstance(value, list):
        for item in value:
            tools.extend(normalize_tool_defs(item))
    elif isinstance(value, dict):
        if value.get("type") == "function" and isinstance(value.get("function"), dict):
            tools.append(value)
        elif value.get("name") and isinstance(value.get("parameters"), dict):
            tools.append({"type": "function", "function": value})
        else:
            for key in ("tools", "functions", "api_list"):
                if key in value:
                    tools.extend(normalize_tool_defs(value[key]))
    return tools


def extract_tools_from_text(text):
    tools = []
    if not isinstance(text, str):
        return tools
    for value in iter_json_values(text):
        tools.extend(normalize_tool_defs(value))
    return tools


def role_name(role):
    role = str(role).lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant"}:
        return "assistant"
    if role == "tool":
        return "tool"
    if role == "system":
        return "system"
    return role


def clean_system_prompt(system):
    system = str(system or "").strip()
    if system.upper().startswith("SYSTEM:"):
        return system.split(":", 1)[1].strip()
    return system


def with_system_message(system, messages):
    system = clean_system_prompt(system)
    if not system:
        return messages
    if messages and messages[0].get("role") == "system":
        return messages
    return [{"role": "system", "content": system}, *messages]


def normalize_conversations(conversations):
    conversations = parse_maybe_literal(conversations)
    if not isinstance(conversations, list):
        return []
    messages = []
    for msg in conversations:
        if not isinstance(msg, dict):
            continue
        role = role_name(msg.get("role", msg.get("from", "")))
        content = msg.get("content", msg.get("value", ""))
        messages.append({"role": role, "content": str(content)})
    return messages


def split_tagged_chat(chat):
    chat = str(chat or "").replace("<|endoftext|>", "").strip()
    matches = list(CHAT_MARKER_RE.finditer(chat))
    if not matches:
        return [{"role": "conversation", "content": chat}] if chat else []

    messages = []
    for idx, match in enumerate(matches):
        role_token = match.group(1).upper()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(chat)
        content = chat[start:end].strip()
        if not content:
            continue
        if role_token == "USER":
            role = "user"
        elif role_token == "ASSISTANT":
            role = "assistant"
        else:
            role = "tool"
        messages.append({"role": role, "content": content})
    return messages


def normalize_hermes(example):
    return {
        "source": "hermes",
        "id": example.get("id"),
        "task": example.get("task"),
        "category": example.get("category"),
        "tools": parse_maybe_literal(example.get("tools", "[]")),
        "messages": normalize_conversations(example.get("conversations", [])),
    }


def normalize_toolace(example):
    system = example.get("system", "")
    return {
        "source": "toolace",
        "id": None,
        "task": None,
        "category": None,
        "tools": extract_tools_from_text(system),
        "system": clean_system_prompt(system),
        "messages": with_system_message(system, normalize_conversations(example.get("conversations", []))),
    }


def normalize_glaive(example):
    system = example.get("system", "")
    return {
        "source": "glaive",
        "id": None,
        "task": None,
        "category": None,
        "tools": extract_tools_from_text(system),
        "system": clean_system_prompt(system),
        "messages": with_system_message(system, split_tagged_chat(example.get("chat", ""))),
    }


def normalize_xlam(example):
    tools = []
    for key in ("tools", "functions", "api_list"):
        tools.extend(normalize_tool_defs(example.get(key, [])))

    messages = normalize_conversations(example.get("messages", example.get("conversations", [])))
    if not messages:
        system = clean_system_prompt(example.get("system", ""))
        user = example.get("query") or example.get("question") or example.get("instruction") or example.get("user")
        assistant = example.get("answer") or example.get("response") or example.get("assistant")
        messages = []
        if user:
            messages.append({"role": "user", "content": str(user)})
        if assistant:
            messages.append({"role": "assistant", "content": str(assistant)})
        messages = with_system_message(system, messages)

    return {
        "source": "xlam",
        "id": example.get("id"),
        "task": example.get("task") or example.get("query") or example.get("question"),
        "category": example.get("category"),
        "tools": tools,
        "system": clean_system_prompt(example.get("system", "")),
        "messages": messages,
    }


NORMALIZERS = {
    "hermes": normalize_hermes,
    "glaive": normalize_glaive,
    "toolace": normalize_toolace,
    "xlam": normalize_xlam,
}


def iter_source(source, limit):
    ds_name = SOURCES[source]
    ds = load_dataset(ds_name, split="train", streaming=True)
    normalizer = NORMALIZERS[source]
    count = 0
    for example in ds:
        item = normalizer(example)
        messages = item.get("messages") or []
        if not messages:
            continue
        yield item
        count += 1
        if count >= limit:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--per-source-limit", type=int, default=50)
    parser.add_argument("--sources", nargs="+", default=["hermes", "glaive", "toolace"])
    parser.add_argument("--include-xlam", action="store_true", help="Try gated xLAM dataset if HF access is available.")
    args = parser.parse_args()

    sources = list(args.sources)
    if args.include_xlam and "xlam" not in sources:
        sources.append("xlam")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "output": str(args.out),
        "per_source_limit": args.per_source_limit,
        "sources": {},
        "notes": [
            "Normalized seed data for Qwen-style tool-call eval/data loop.",
            "This is not final training data; teacher repair/verification is still required.",
        ],
    }

    with args.out.open("w", encoding="utf-8") as f:
        for source in sources:
            count = 0
            try:
                for item in iter_source(source, args.per_source_limit):
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    count += 1
                manifest["sources"][source] = {
                    "dataset": SOURCES[source],
                    "status": "ok",
                    "count": count,
                }
            except Exception as exc:
                manifest["sources"][source] = {
                    "dataset": SOURCES.get(source),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "count": count,
                }

    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
