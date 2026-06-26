#!/usr/bin/env python3
import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from eval_toolcall_jsonl import score_tool_calls


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_INPUT = ROOT / "data/toolcall_eval/synthetic_onecall_smoke.jsonl"
DEFAULT_OUT = ROOT / "data/toolcall_eval/synthetic_onecall_teacher.jsonl"


def load_cases(path, limit):
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
            if limit and len(cases) >= limit:
                break
    return cases


def post_json(url, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ask_teacher(case, endpoint, model, timeout, temperature, max_tokens, enable_thinking):
    messages = list(case["prompt_messages"])
    messages.append(
        {
            "role": "user",
            "content": (
                "Return the necessary Qwen tool call or calls for the request above. "
                "Use only this format and no prose:\n"
                "<tool_call>\n"
                "{\"name\": \"tool_name\", \"arguments\": {}}\n"
                "</tool_call>"
            ),
        }
    )
    payload = {
        "model": model,
        "messages": messages,
        "tools": case.get("tools") or None,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    payload = {key: value for key, value in payload.items() if value is not None}
    response = post_json(endpoint.rstrip("/") + "/chat/completions", payload, timeout)
    message = response["choices"][0]["message"]
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        rendered = []
        for call in tool_calls:
            function = call.get("function") or {}
            name = function.get("name")
            arguments = function.get("arguments") or "{}"
            rendered.append(
                "<tool_call>\n"
                + json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
                + "\n</tool_call>"
            )
        return "\n".join(rendered)
    return message.get("content", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", default="qwen3.6-27b-teacher")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--enable-thinking", action="store_true")
    args = parser.parse_args()

    cases = load_cases(args.input_jsonl, args.limit)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    totals = {
        "records": 0,
        "ok": 0,
        "valid_tool_json": 0,
        "exact_tool_name_set": 0,
        "errors": 0,
    }
    start = time.time()
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for idx, case in enumerate(cases):
            row = {
                "idx": idx,
                "source": case.get("source"),
                "id": case.get("id"),
                "gold_tool_names": case.get("gold_tool_names") or [],
                "available_tool_names": case.get("available_tool_names") or [],
            }
            try:
                text = ask_teacher(
                    case,
                    args.endpoint,
                    args.model,
                    args.timeout,
                    args.temperature,
                    args.max_tokens,
                    args.enable_thinking,
                )
                metrics = score_tool_calls(text, case.get("tools") or [], case.get("gold_assistant"))
                names = metrics["called_names"]
                row.update(
                    {
                        "status": "ok",
                        "teacher_assistant": text,
                        "teacher_tool_names": names,
                        "teacher_calls": metrics["calls"],
                        "invalid_tool_json_count": metrics["invalid_tool_call_count"],
                        "valid_tool_json": metrics["valid_tool_call"],
                        "valid_tool_call": metrics["valid_tool_call"],
                        "exact_tool_name_set": metrics.get("exact_tool_name_set"),
                        "exact_tool_sequence": metrics.get("exact_tool_sequence"),
                        "exact_arguments": metrics.get("exact_arguments"),
                        "all_schema_valid": metrics["all_schema_valid"],
                        "all_required_args_present": metrics["all_required_args_present"],
                        "schema_valid_count": metrics["schema_valid_count"],
                        "required_args_count": metrics["required_args_count"],
                        "call_errors": metrics["call_errors"],
                    }
                )
                totals["ok"] += 1
                totals["valid_tool_json"] += int(row["valid_tool_json"])
                totals["exact_tool_name_set"] += int(bool(row["exact_tool_name_set"]))
                totals.setdefault("exact_tool_sequence", 0)
                totals.setdefault("exact_arguments", 0)
                totals.setdefault("all_schema_valid", 0)
                totals.setdefault("all_required_args_present", 0)
                totals["exact_tool_sequence"] += int(bool(row["exact_tool_sequence"]))
                totals["exact_arguments"] += int(bool(row["exact_arguments"]))
                totals["all_schema_valid"] += int(bool(row["all_schema_valid"]))
                totals["all_required_args_present"] += int(bool(row["all_required_args_present"]))
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
                totals["errors"] += 1
            totals["records"] += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    summary = {
        "input_jsonl": str(args.input_jsonl),
        "out_jsonl": str(args.out_jsonl),
        "endpoint": args.endpoint,
        "model": args.model,
        "totals": totals,
        "elapsed_seconds": time.time() - start,
    }
    summary_path = args.out_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
