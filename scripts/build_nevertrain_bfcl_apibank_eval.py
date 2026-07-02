#!/usr/bin/env python3
"""Build a never-train BFCL/API-Bank matched tool-call eval slice."""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import io
import json
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from audit_toolace40_leak_and_difficulty import (  # noqa: E402
    build_train_index,
    default_train_paths,
    difficulty_rows,
    overlap_audit,
    summarize_difficulty,
)
from build_flare_broaden_public_eval import max_gold_path_tokens  # noqa: E402
from eval_flare_multiturn_percall_waves import split_tool_call_blocks  # noqa: E402
from eval_toolcall_jsonl import extract_tool_calls, qwen_native_tool_call_text  # noqa: E402


DEFAULT_OUT = ROOT / "data/toolcall_eval_native/flare_nevertrain_bfcl_apibank.jsonl"
DEFAULT_MANIFEST = ROOT / "data/toolcall_eval_native/flare_nevertrain_bfcl_apibank.manifest.json"
DEFAULT_OVERLAP = ROOT / "data/toolcall_eval_native/flare_nevertrain_bfcl_apibank.overlap_rows.jsonl"
DEFAULT_REJECTED = ROOT / "data/toolcall_eval_native/flare_nevertrain_bfcl_apibank.rejected.jsonl"
DEFAULT_MODEL = ROOT / "models/qwen3.5-9b-fastdllm-b1000-vllm-bf16"
DEFAULT_CHAT_TEMPLATE = Path("/home/mark/shared/lumoFlyWheel_codex_fork/docker/chat_templates/qwen3-openai-codex.jinja")
DEFAULT_XLAM_PROBE = ROOT / "data/toolcall_seed/xlam_probe.manifest.json"

GORILLA_RAW = "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/bfcl_eval/data"
GORILLA_TREE = "https://api.github.com/repos/ShishirPatil/gorilla/git/trees/5e49f820dd465850cbaebc241806d2ef7b893471?recursive=1"
DAMO_RAW = "https://raw.githubusercontent.com/AlibabaResearch/DAMO-ConvAI/main/api-bank"
DAMO_TREE = "https://api.github.com/repos/AlibabaResearch/DAMO-ConvAI/git/trees/483554eae102996f5ec1f4feab4e78ef29c2a394?recursive=1"

BFCL_AST_CATEGORIES = ("multiple", "parallel", "parallel_multiple", "simple_python")
BFCL_MULTI_CATEGORIES = (
    "multi_turn_base",
    "multi_turn_miss_param",
    "multi_turn_miss_func",
    "multi_turn_long_context",
)
BFCL_MULTI_FUNC_DOC_FILES = (
    "gorilla_file_system.json",
    "math_api.json",
    "memory_kv.json",
    "memory_rec_sum.json",
    "memory_vector.json",
    "message_api.json",
    "posting_api.json",
    "ticket_api.json",
    "trading_bot.json",
    "travel_booking.json",
    "vehicle_control.json",
    "web_search.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--overlap-jsonl", type=Path, default=DEFAULT_OVERLAP)
    parser.add_argument("--rejected-jsonl", type=Path, default=DEFAULT_REJECTED)
    parser.add_argument("--bfcl-multi-count", type=int, default=30)
    parser.add_argument("--bfcl-ast-count", type=int, default=8)
    parser.add_argument("--apibank-lv1-count", type=int, default=13)
    parser.add_argument("--apibank-lv2-count", type=int, default=12)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-tools", type=int, default=24)
    parser.add_argument("--max-prompt-tokens", type=int, default=4500)
    parser.add_argument("--max-gold-tokens", type=int, default=384)
    parser.add_argument("--prompt-tokenizer-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--chat-template-path", type=Path, default=DEFAULT_CHAT_TEMPLATE)
    parser.add_argument("--xlam-probe-json", type=Path, default=DEFAULT_XLAM_PROBE)
    parser.add_argument("--train-path", type=Path, action="append", default=None)
    return parser.parse_args()


def fetch_text(url: str, timeout: float = 60.0, attempts: int = 4) -> tuple[str, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "qwen-diffusion-eval-builder"})
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
            break
        except Exception as exc:
            last_error = exc
            if attempt + 1 >= attempts:
                raise RuntimeError(f"failed to fetch {url}: {type(exc).__name__}: {exc}") from exc
            time.sleep(1.0 + attempt)
    else:
        raise RuntimeError(f"failed to fetch {url}: {last_error}")
    return raw.decode("utf-8-sig"), hashlib.sha256(raw).hexdigest()


def fetch_jsonl(url: str) -> tuple[list[dict[str, Any]], str]:
    text, digest = fetch_text(url)
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [row for row in rows if isinstance(row, dict)], digest


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def normalize_schema_type(value: Any) -> Any:
    if isinstance(value, list):
        return [normalize_schema_type(item) for item in value]
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


def convert_schema(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}, "required": []}
    out: dict[str, Any] = {}
    expected = normalize_schema_type(schema.get("type"))
    if expected is None and isinstance(schema.get("properties"), dict):
        expected = "object"
    if expected is not None:
        out["type"] = expected
    for key in ("description", "enum", "default"):
        if key in schema:
            out[key] = copy.deepcopy(schema[key])
    if isinstance(schema.get("properties"), dict):
        out["properties"] = {str(key): convert_schema(value) for key, value in schema["properties"].items()}
        required = schema.get("required") or []
        out["required"] = [str(item) for item in required if isinstance(item, str)] if isinstance(required, list) else []
    if "items" in schema:
        out["items"] = convert_schema(schema["items"])
    if out.get("type") == "object":
        out.setdefault("properties", {})
        out.setdefault("required", [])
    if not out:
        out = {"type": "object", "properties": {}, "required": []}
    return out


def as_tool(fn: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": str(fn.get("name") or ""),
            "description": str(fn.get("description") or ""),
            "parameters": convert_schema(fn.get("parameters") or {}),
        },
    }


def base_name(name: Any) -> str:
    text = str(name or "").strip()
    return text.split(".")[-1] if "." in text else text


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


def parse_value(text: str) -> Any:
    value = str(text).strip()
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


def parse_call_string(text: str, tool_by_name: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    open_idx = raw.find("(")
    close_idx = matching_paren(raw, open_idx) if open_idx >= 0 else -1
    if open_idx < 1 or close_idx < open_idx:
        return None
    name = raw[:open_idx].strip()
    body = raw[open_idx + 1 : close_idx].strip()
    positional: list[Any] = []
    kwargs: dict[str, Any] = {}
    for part in split_top_level(body):
        key_value = None
        depth = 0
        quote: str | None = None
        escape = False
        for idx, char in enumerate(part):
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
                key_value = (part[:idx].strip(), part[idx + 1 :].strip())
                break
        if key_value is None:
            positional.append(parse_value(part))
        else:
            key, value = key_value
            kwargs[key] = parse_value(value)
    if positional:
        props = list((((tool_by_name.get(name) or {}).get("function") or {}).get("parameters") or {}).get("properties") or {})
        for idx, value in enumerate(positional):
            key = props[idx] if idx < len(props) else f"arg_{idx}"
            kwargs.setdefault(key, value)
    return {"name": name, "arguments": kwargs}


def first_question_messages(question: Any) -> list[dict[str, str]]:
    if isinstance(question, list) and question:
        first = question[0]
        if isinstance(first, list):
            return [
                {"role": str(msg.get("role", "user")).lower(), "content": str(msg.get("content") or "")}
                for msg in first
                if isinstance(msg, dict) and str(msg.get("content") or "").strip()
            ]
    return []


def turn_user_text(turn_messages: Any) -> str:
    if not isinstance(turn_messages, list):
        return ""
    chunks = [
        str(msg.get("content") or "").strip()
        for msg in turn_messages
        if isinstance(msg, dict) and str(msg.get("role", "user")).lower() == "user" and str(msg.get("content") or "").strip()
    ]
    return "\n\n".join(chunks)


def choose_arg_value(values: Any, *, required: bool) -> tuple[bool, Any]:
    if isinstance(values, list):
        if not required and "" in values:
            return False, None
        for value in values:
            if value != "":
                return True, value
        return bool(values), values[0] if values else None
    return True, values


def bfcl_answer_calls(answer: dict[str, Any], tool_by_name: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    calls = []
    for item in answer.get("ground_truth") or []:
        if not isinstance(item, dict):
            continue
        for name, arg_choices in item.items():
            tool = tool_by_name.get(str(name)) or {}
            schema = ((tool.get("function") or {}).get("parameters") or {})
            required = set(schema.get("required") or [])
            args = {}
            if isinstance(arg_choices, dict):
                for key, values in arg_choices.items():
                    include, value = choose_arg_value(values, required=str(key) in required)
                    if include:
                        args[str(key)] = value
            calls.append({"name": str(name), "arguments": args})
    return calls


def episode_hash(row: dict[str, Any]) -> str:
    return sha256_json(
        {
            "source_dataset": row.get("source_dataset"),
            "source_row_idx": row.get("source_row_idx"),
            "prompt_messages": row.get("prompt_messages") or [],
            "turn_user_messages": row.get("turn_user_messages") or [],
            "tools": row.get("tools") or [],
            "gold_assistant": row.get("gold_assistant") or "",
        }
    )


def finalize_episode(row: dict[str, Any]) -> dict[str, Any]:
    calls, invalid = extract_tool_calls(row.get("gold_assistant") or "")
    row["gold_tool_calls"] = calls
    row["gold_invalid_tool_json_count"] = invalid
    row["gold_tool_names"] = [call.get("name") for call in calls]
    row["available_tool_names"] = sorted(
        str((tool.get("function") or {}).get("name") or "") for tool in row.get("tools") or []
    )
    row["gold_assistant_format"] = "qwen_native_function_parameter"
    row["public_eval_hash"] = episode_hash(row)
    return row


def load_bfcl_multi_docs(source_records: list[dict[str, Any]]) -> dict[str, Any]:
    docs_by_name: dict[str, dict[str, Any]] = {}
    docs_by_file: dict[str, list[dict[str, Any]]] = {}
    for filename in BFCL_MULTI_FUNC_DOC_FILES:
        url = f"{GORILLA_RAW}/multi_turn_func_doc/{filename}"
        rows, digest = fetch_jsonl(url)
        source_records.append(
            {"source": "BFCL", "kind": "multi_turn_func_doc", "url": url, "sha256": digest, "rows": len(rows)}
        )
        docs_by_file[filename] = rows
        for row in rows:
            if row.get("name"):
                docs_by_name[str(row["name"])] = as_tool(row)
    return {"by_name": docs_by_name, "by_file": docs_by_file}


def build_bfcl_ast(args: argparse.Namespace, source_records: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    per_category_target = max(1, args.bfcl_ast_count // len(BFCL_AST_CATEGORIES))
    for category in BFCL_AST_CATEGORIES:
        q_url = f"{GORILLA_RAW}/BFCL_v4_{category}.json"
        a_url = f"{GORILLA_RAW}/possible_answer/BFCL_v4_{category}.json"
        questions, q_sha = fetch_jsonl(q_url)
        answers, a_sha = fetch_jsonl(a_url)
        source_records.extend(
            [
                {"source": "BFCL", "kind": "question", "category": category, "url": q_url, "sha256": q_sha, "rows": len(questions)},
                {"source": "BFCL", "kind": "possible_answer", "category": category, "url": a_url, "sha256": a_sha, "rows": len(answers)},
            ]
        )
        answers_by_id = {str(row.get("id")): row for row in answers}
        accepted_here = 0
        for q_idx, question in enumerate(questions):
            answer = answers_by_id.get(str(question.get("id")))
            tools = [as_tool(item) for item in question.get("function") or [] if isinstance(item, dict)]
            tool_by_name = {tool["function"]["name"]: tool for tool in tools}
            calls = bfcl_answer_calls(answer or {}, tool_by_name)
            blocks = [qwen_native_tool_call_text([call]) for call in calls]
            if not calls or len(blocks) > args.max_turns or len(tools) > args.max_tools:
                rejected.append(
                    {
                        "source": "BFCL-AST",
                        "id": question.get("id"),
                        "reason": "empty_or_too_large",
                        "turns": len(blocks),
                        "tools": len(tools),
                    }
                )
                continue
            row = finalize_episode(
                {
                    "source": "bfcl_v4_ast",
                    "source_family": "BFCL-AST",
                    "source_dataset": f"ShishirPatil/gorilla BFCL_v4_{category}",
                    "source_split": "BFCL_v4_eval_non_live_ast",
                    "source_license": "Apache-2.0",
                    "source_row_idx": q_idx,
                    "source_category": category,
                    "id": f"bfcl_ast_{category}_{q_idx:04d}",
                    "tools": tools,
                    "prompt_messages": [{"role": "system", "content": "You are a helpful assistant."}]
                    + first_question_messages(question.get("question")),
                    "turn_user_messages": [None for _ in blocks],
                    "gold_assistant": "\n".join(blocks),
                }
            )
            rows.append(row)
            accepted_here += 1
            if accepted_here >= per_category_target or len(rows) >= args.bfcl_ast_count:
                break
        if len(rows) >= args.bfcl_ast_count:
            break
    return rows[: args.bfcl_ast_count]


def build_bfcl_multi(args: argparse.Namespace, source_records: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    docs = load_bfcl_multi_docs(source_records)
    docs_by_name: dict[str, dict[str, Any]] = docs["by_name"]
    rows = []
    per_category_target = max(1, args.bfcl_multi_count // len(BFCL_MULTI_CATEGORIES))
    for category in BFCL_MULTI_CATEGORIES:
        q_url = f"{GORILLA_RAW}/BFCL_v4_{category}.json"
        a_url = f"{GORILLA_RAW}/possible_answer/BFCL_v4_{category}.json"
        questions, q_sha = fetch_jsonl(q_url)
        answers, a_sha = fetch_jsonl(a_url)
        source_records.extend(
            [
                {"source": "BFCL", "kind": "question", "category": category, "url": q_url, "sha256": q_sha, "rows": len(questions)},
                {"source": "BFCL", "kind": "possible_answer", "category": category, "url": a_url, "sha256": a_sha, "rows": len(answers)},
            ]
        )
        answers_by_id = {str(row.get("id")): row for row in answers}
        accepted_here = 0
        for q_idx, question in enumerate(questions):
            answer = answers_by_id.get(str(question.get("id"))) or {}
            tool_names = {base_name(item) for item in (question.get("path") or [])}
            tool_names.update(base_name(item) for item in (question.get("excluded_function") or []))
            blocks = []
            turn_users = []
            for turn_idx, turn_calls in enumerate(answer.get("ground_truth") or []):
                user_text = turn_user_text((question.get("question") or [])[turn_idx] if turn_idx < len(question.get("question") or []) else [])
                first_call_in_turn = True
                for call_text in turn_calls if isinstance(turn_calls, list) else []:
                    parsed = parse_call_string(str(call_text), docs_by_name)
                    if parsed is None:
                        continue
                    tool_names.add(parsed["name"])
                    blocks.append(qwen_native_tool_call_text([parsed]))
                    if not turn_users:
                        turn_users.append(None)
                    elif first_call_in_turn:
                        turn_users.append(user_text or None)
                    else:
                        turn_users.append(None)
                    first_call_in_turn = False
            tools = [docs_by_name[name] for name in sorted(tool_names) if name in docs_by_name]
            if not blocks or len(blocks) > args.max_turns or len(tools) > args.max_tools:
                rejected.append(
                    {
                        "source": "BFCL-multi_turn",
                        "id": question.get("id"),
                        "reason": "empty_or_too_large",
                        "turns": len(blocks),
                        "tools": len(tools),
                    }
                )
                continue
            row = finalize_episode(
                {
                    "source": "bfcl_v4_multi_turn",
                    "source_family": "BFCL-multi_turn",
                    "source_dataset": f"ShishirPatil/gorilla BFCL_v4_{category}",
                    "source_split": "BFCL_v4_eval_non_live_multi_turn",
                    "source_license": "Apache-2.0",
                    "source_row_idx": q_idx,
                    "source_category": category,
                    "id": f"bfcl_mt_{category}_{q_idx:04d}",
                    "tools": tools,
                    "prompt_messages": [{"role": "system", "content": "You are a helpful assistant."}]
                    + first_question_messages(question.get("question")),
                    "turn_user_messages": turn_users,
                    "gold_assistant": "\n".join(blocks),
                    "bfcl_initial_config_hash": sha256_json(question.get("initial_config") or {}),
                }
            )
            rows.append(row)
            accepted_here += 1
            if accepted_here >= per_category_target or len(rows) >= args.bfcl_multi_count:
                break
        if len(rows) >= args.bfcl_multi_count:
            break
    return rows[: args.bfcl_multi_count]


def extract_assignment_dict(text: str, name: str) -> dict[str, Any]:
    marker = f"{name} ="
    idx = text.find(marker)
    if idx < 0:
        return {}
    open_idx = text.find("{", idx)
    if open_idx < 0:
        return {}
    depth = 0
    quote: str | None = None
    escape = False
    for pos in range(open_idx, len(text)):
        char = text[pos]
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
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    payload = ast.literal_eval(text[open_idx : pos + 1])
                    return payload if isinstance(payload, dict) else {}
                except Exception:
                    return {}
    return {}


def parse_api_bank_schemas(source_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    url = f"{DAMO_RAW}/data/all_apis.csv"
    text, digest = fetch_text(url)
    source_records.append({"source": "API-Bank", "kind": "api_schema_csv", "url": url, "sha256": digest})
    schemas = {}
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        name = str(row.get("类名") or "").strip()
        if not name:
            continue
        api_info = str(row.get("api_info") or "")
        description_match = re.search(r"description\s*=\s*(['\"])(.*?)\1", api_info, re.DOTALL)
        input_params = extract_assignment_dict(api_info, "input_parameters")
        properties = {}
        required = []
        for key, spec in input_params.items():
            if not isinstance(spec, dict):
                spec = {}
            properties[str(key)] = {
                "type": normalize_schema_type(spec.get("type") or "string"),
                "description": str(spec.get("description") or ""),
            }
            required.append(str(key))
        schemas[name] = {
            "type": "function",
            "function": {
                "name": name,
                "description": description_match.group(2).strip() if description_match else name,
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
        }
    return schemas


def api_bank_paths(source_records: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    text, digest = fetch_text(DAMO_TREE)
    source_records.append({"source": "API-Bank", "kind": "repo_tree", "url": DAMO_TREE, "sha256": digest})
    payload = json.loads(text)
    paths = [
        item["path"]
        for item in payload.get("tree", [])
        if item.get("type") == "blob"
        and item.get("path", "").startswith("api-bank/lv1-lv2-samples/level-1-given-desc/")
        and item.get("path", "").endswith(".jsonl")
        and int(item.get("size") or 0) > 0
    ]
    lv1 = sorted(path for path in paths if "-level-1-" in path)
    lv2 = sorted(path for path in paths if "-level-2-" in path)
    return lv1, lv2


def prompt_messages_from_api_bank(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for item in history:
        role = str(item.get("role") or "").lower()
        if role == "user":
            messages.append({"role": "user", "content": str(item.get("text") or "")})
        elif role in {"ai", "assistant"}:
            messages.append({"role": "assistant", "content": str(item.get("text") or "")})
        elif role == "api":
            payload = item.get("result") or {"api_name": item.get("api_name"), "input": item.get("param_dict")}
            messages.append(
                {
                    "role": "user",
                    "content": "<tool_response>\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n</tool_response>",
                }
            )
    return messages


def build_api_bank(
    args: argparse.Namespace,
    source_records: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    schemas = parse_api_bank_schemas(source_records)
    lv1_paths, lv2_paths = api_bank_paths(source_records)
    targets = [("API-Bank-Lv1", lv1_paths, args.apibank_lv1_count), ("API-Bank-Lv2", lv2_paths, args.apibank_lv2_count)]
    rows = []
    for source_family, paths, target_count in targets:
        accepted = 0
        for path in paths:
            if accepted >= target_count:
                break
            url = "https://raw.githubusercontent.com/AlibabaResearch/DAMO-ConvAI/main/" + path
            dialogue, digest = fetch_jsonl(url)
            source_records.append({"source": "API-Bank", "kind": "dialogue", "source_family": source_family, "url": url, "sha256": digest, "rows": len(dialogue)})
            api_names = sorted({str(item.get("api_name")) for item in dialogue if str(item.get("role") or "").lower() == "api"})
            tools = [schemas[name] for name in api_names if name in schemas]
            history = []
            call_idx = 0
            for item in dialogue:
                if str(item.get("role") or "").lower() != "api":
                    history.append(item)
                    continue
                api_name = str(item.get("api_name") or "")
                if api_name not in schemas:
                    rejected.append({"source": source_family, "path": path, "reason": "schema_missing", "api_name": api_name})
                    history.append(item)
                    continue
                if len(tools) > args.max_tools:
                    rejected.append({"source": source_family, "path": path, "reason": "too_many_tools", "tools": len(tools)})
                    history.append(item)
                    continue
                block = qwen_native_tool_call_text([{"name": api_name, "arguments": item.get("param_dict") or {}}])
                row = finalize_episode(
                    {
                        "source": "api_bank_raw",
                        "source_family": source_family,
                        "source_dataset": "AlibabaResearch/DAMO-ConvAI api-bank/lv1-lv2-samples/level-1-given-desc",
                        "source_split": "API-Bank evaluation Lv1/Lv2 samples",
                        "source_license": "MIT",
                        "source_row_idx": f"{path}:{call_idx}",
                        "source_path": path,
                        "id": f"apibank_{source_family.lower().replace('-', '_')}_{Path(path).stem}_{call_idx}",
                        "tools": tools,
                        "prompt_messages": prompt_messages_from_api_bank(history),
                        "turn_user_messages": [None],
                        "gold_assistant": block,
                        "api_bank_result_hash": sha256_json(item.get("result") or {}),
                    }
                )
                rows.append(row)
                accepted += 1
                call_idx += 1
                history.append(item)
                if accepted >= target_count:
                    break
    return rows


def load_chat_template(path: Path | None) -> str | None:
    return path.read_text(encoding="utf-8") if path else None


def token_filter_rows(
    rows: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    tokenizer,
    chat_template: str | None,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    kept = []
    for row in rows:
        max_prompt, max_gold = max_gold_path_tokens(tokenizer, chat_template, row)
        row["token_filter"] = {"max_gold_path_prompt_tokens": max_prompt, "max_gold_block_tokens": max_gold}
        if max_prompt > args.max_prompt_tokens or max_gold > args.max_gold_tokens:
            rejected.append(
                {
                    "source": row.get("source_family"),
                    "id": row.get("id"),
                    "reason": "prompt_or_gold_too_long",
                    "max_gold_path_prompt_tokens": max_prompt,
                    "max_gold_block_tokens": max_gold,
                }
            )
            continue
        kept.append(row)
    return kept


def source_status_xlam(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"dataset": "Salesforce/xlam-function-calling-60k", "status": "not_probed"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"dataset": "Salesforce/xlam-function-calling-60k", "status": "probe_unreadable", "error": str(exc)}


def source_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for source, source_rows in sorted(defaultdict(list, {k: [] for k in []}).items()):
        out[source] = source_rows
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("source_family") or "unknown")].append(row)
    return {
        source: {
            "episodes": len(items),
            "turns": sum(len(split_tool_call_blocks(item.get("gold_assistant") or "")) for item in items),
            "avg_tools": sum(len(item.get("tools") or []) for item in items) / len(items) if items else 0.0,
            "categories": dict(Counter(str(item.get("source_category") or item.get("source_path") or "") for item in items)),
        }
        for source, items in sorted(grouped.items())
    }


def manifest_for(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    source_records: list[dict[str, Any]],
    leak_check: dict[str, Any],
    difficulty: dict[str, Any],
) -> dict[str, Any]:
    manifest_rows = [
        {
            "id": row.get("id"),
            "source_family": row.get("source_family"),
            "source_dataset": row.get("source_dataset"),
            "source_split": row.get("source_split"),
            "source_row_idx": row.get("source_row_idx"),
            "turns": len(split_tool_call_blocks(row.get("gold_assistant") or "")),
            "tools": len(row.get("tools") or []),
            "public_eval_hash": row.get("public_eval_hash"),
            "token_filter": row.get("token_filter"),
        }
        for row in rows
    ]
    return {
        "created_by": "scripts/build_nevertrain_bfcl_apibank_eval.py",
        "out_jsonl": str(args.out_jsonl),
        "records": len(rows),
        "turns": sum(item["turns"] for item in manifest_rows),
        "episode_set_hash": sha256_json(manifest_rows),
        "source_family_counts": dict(Counter(str(row.get("source_family") or "unknown") for row in rows)),
        "source_summary": source_summary(rows),
        "rows": manifest_rows,
        "source_records": source_records,
        "selection": {
            "bfcl_multi_target": args.bfcl_multi_count,
            "bfcl_ast_target": args.bfcl_ast_count,
            "apibank_lv1_target": args.apibank_lv1_count,
            "apibank_lv2_target": args.apibank_lv2_count,
            "max_turns": args.max_turns,
            "max_tools": args.max_tools,
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_gold_tokens": args.max_gold_tokens,
            "qwen_contract": "Every generated turn has exactly one Qwen-native XML tool-call gold block; multi-call BFCL answers are split into sequential one-call turns.",
        },
        "never_train_claim": {
            "BFCL": "BFCL V4 eval-designated non-live AST and multi-turn categories; BFCL was never used as a training candidate in rl_dataset_plan; upstream repo license Apache-2.0.",
            "API-Bank": "Raw DAMO API-Bank lv1-lv2-samples/level-1-given-desc evaluation samples; Level-3/toolsearcher files are excluded; upstream repo license MIT.",
            "xLAM": {
                "status": "not_used",
                "reason": "local probe remains gated; not allowed to block this rebuild",
                "probe": source_status_xlam(args.xlam_probe_json),
            },
        },
        "leak_check": leak_check,
        "difficulty": difficulty,
        "rejected_count": len(rejected),
        "rejected_reason_counts": dict(Counter(str(item.get("reason") or "unknown") for item in rejected)),
        "rejected_jsonl": str(args.rejected_jsonl),
        "overlap_jsonl": str(args.overlap_jsonl),
    }


def main() -> int:
    args = parse_args()
    source_records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    rows = []
    rows.extend(build_bfcl_multi(args, source_records, rejected))
    rows.extend(build_bfcl_ast(args, source_records, rejected))
    rows.extend(build_api_bank(args, source_records, rejected))

    tokenizer = AutoTokenizer.from_pretrained(str(args.prompt_tokenizer_path), trust_remote_code=True)
    chat_template = load_chat_template(args.chat_template_path)
    rows = token_filter_rows(rows, rejected, tokenizer, chat_template, args)

    train_paths = [path.resolve() for path in args.train_path] if args.train_path else default_train_paths()
    train_index, _ = build_train_index(train_paths)
    leak_check, overlap_rows = overlap_audit(rows, train_index)
    leak_check["scope"] = "BFCL/API-Bank never-train eval rows selected for this rebuild"
    leak_check["hard_overlap_verdict"] = (
        "no-hard-overlap"
        if not leak_check.get("eval_episodes_with_any_hard_overlap")
        else "hard-overlap-found"
    )

    difficulty_episode_rows = []
    for source_family in sorted({str(row.get("source_family") or "unknown") for row in rows}):
        difficulty_episode_rows.extend(
            difficulty_rows([row for row in rows if row.get("source_family") == source_family], source_family)
        )
    grouped = defaultdict(list)
    for item in difficulty_episode_rows:
        grouped[item["source_family"]].append(item)
    difficulty = {
        "summary_by_source": {source: summarize_difficulty(items) for source, items in sorted(grouped.items())},
        "episode_count": len(difficulty_episode_rows),
    }

    write_jsonl(args.out_jsonl, rows)
    write_jsonl(args.overlap_jsonl, overlap_rows)
    write_jsonl(args.rejected_jsonl, rejected)
    manifest = manifest_for(args, rows, rejected, source_records, leak_check, difficulty)
    write_json(args.manifest_json, manifest)
    print(
        json.dumps(
            {
                "records": manifest["records"],
                "turns": manifest["turns"],
                "source_family_counts": manifest["source_family_counts"],
                "hard_overlap_verdict": manifest["leak_check"]["hard_overlap_verdict"],
                "episode_set_hash": manifest["episode_set_hash"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
