#!/usr/bin/env python3
"""Smoke-test an OpenAI-compatible server for native Qwen tool calls."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from eval_toolcall_jsonl import extract_tool_calls  # noqa: E402


def post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def get_json(url: str, timeout: float) -> dict:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def build_payload(model: str, tool_choice: str, max_tokens: int) -> dict:
    tool = {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up an order by id and customer name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "customer": {"type": "string"},
                },
                "required": ["order_id", "customer"],
            },
        },
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Use the available tool to look up order ORD-419 for customer Acme Labs. "
                    "Return only the tool call."
                ),
            }
        ],
        "tools": [tool],
        "tool_choice": tool_choice,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    return payload


def validate_tool_call(message: dict) -> tuple[bool, str, str]:
    tool_calls = message.get("tool_calls") or []
    source = "openai_tool_calls"
    if not tool_calls:
        content = message.get("content") or ""
        parsed_calls, invalid = extract_tool_calls(content)
        if invalid:
            return False, f"invalid Qwen-native tool-call blocks: {invalid}", "qwen_native_content"
        tool_calls = parsed_calls
        source = "qwen_native_content"
    if not tool_calls:
        return False, "no tool calls found", source
    first = tool_calls[0]
    if "function" in first:
        function = first.get("function") or {}
        name = function.get("name")
        raw_args = function.get("arguments") or "{}"
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError as exc:
                return False, f"arguments are not JSON: {exc}", source
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            return False, f"unexpected arguments type: {type(raw_args).__name__}", source
    else:
        name = first.get("name")
        args = first.get("arguments") or {}
    if name != "lookup_order":
        return False, f"unexpected tool name: {name!r}", source
    missing = [key for key in ("order_id", "customer") if key not in args]
    if missing:
        return False, f"missing required args: {missing}", source
    return True, "ok", source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", default="qwen3.5-9b-ar")
    parser.add_argument("--tool-choice", default="auto")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--out-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    endpoint = args.endpoint.rstrip("/")
    models = get_json(f"{endpoint}/models", args.timeout)
    response = post_json(
        f"{endpoint}/chat/completions",
        build_payload(args.model, args.tool_choice, args.max_tokens),
        args.timeout,
    )
    message = response["choices"][0]["message"]
    ok, reason, source = validate_tool_call(message)
    payload = {
        "ok": ok,
        "reason": reason,
        "source": source,
        "endpoint": endpoint,
        "model": args.model,
        "models": models,
        "message": message,
        "usage": response.get("usage"),
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(text, flush=True)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
