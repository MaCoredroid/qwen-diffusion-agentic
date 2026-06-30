#!/usr/bin/env python3
import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from eval_toolcall_jsonl import score_tool_call_objects, score_tool_calls


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
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def ask_teacher(
    case,
    endpoint,
    model,
    timeout,
    temperature,
    top_p,
    top_k,
    presence_penalty,
    max_tokens,
    enable_thinking,
    instruction_mode,
):
    messages = list(case["prompt_messages"])
    if instruction_mode == "canonical":
        instruction = case.get("teacher_instruction") or (
            "Return the necessary Qwen tool call or calls for the request above. "
            "Use only this format and no prose:\n"
            "<tool_call>\n"
            "<function=tool_name>\n"
            "<parameter=argument_name>\n"
            "argument value\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        messages.append(
            {
                "role": "user",
                "content": instruction,
            }
        )
    payload = {
        "model": model,
        "messages": messages,
        "tools": case.get("tools") or None,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "presence_penalty": presence_penalty,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    payload = {key: value for key, value in payload.items() if value is not None}
    response = post_json(endpoint.rstrip("/") + "/chat/completions", payload, timeout)
    message = response["choices"][0]["message"]
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        return {"content": message.get("content") or "", "tool_calls": tool_calls, "response_message": message}
    return {"content": message.get("content", ""), "tool_calls": [], "response_message": message}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", default="qwen3.6-27b-teacher")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--presence-penalty", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--instruction-mode",
        choices=["canonical", "native"],
        default="native",
        help="native sends only prompt_messages+tools; canonical appends an explicit Qwen-native format reminder.",
    )
    args = parser.parse_args()

    cases = load_cases(args.input_jsonl, args.limit)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    totals = {
        "records": 0,
        "ok": 0,
        "valid_tool_json": 0,
        "exact_tool_name_set": 0,
        "exact_tool_sequence": 0,
        "exact_tool_name_multiset": 0,
        "same_tool_call_count": 0,
        "exact_arguments": 0,
        "all_schema_valid": 0,
        "all_required_args_present": 0,
        "records_with_extra_calls": 0,
        "records_with_missing_calls": 0,
        "records_with_repeated_calls": 0,
        "total_extra_calls": 0,
        "total_missing_calls": 0,
        "total_repeated_calls": 0,
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
                teacher_response = ask_teacher(
                    case,
                    args.endpoint,
                    args.model,
                    args.timeout,
                    args.temperature,
                    args.top_p,
                    args.top_k,
                    args.presence_penalty,
                    args.max_tokens,
                    args.enable_thinking,
                    args.instruction_mode,
                )
                text = teacher_response["content"]
                tool_calls = teacher_response["tool_calls"]
                if tool_calls:
                    metrics = score_tool_call_objects(
                        tool_calls,
                        case.get("tools") or [],
                        gold_text=case.get("gold_assistant"),
                    )
                else:
                    metrics = score_tool_calls(text, case.get("tools") or [], case.get("gold_assistant"))
                names = metrics["called_names"]
                row.update(
                    {
                        "status": "ok",
                        "teacher_assistant": text,
                        "teacher_response_tool_calls": tool_calls,
                        "teacher_response_message": teacher_response["response_message"],
                        "teacher_response_path": "tool_calls" if tool_calls else "content",
                        "teacher_tool_names": names,
                        "teacher_calls": metrics["calls"],
                        "invalid_tool_json_count": metrics["invalid_tool_call_count"],
                        "valid_tool_json": metrics["valid_tool_call"],
                        "valid_tool_call": metrics["valid_tool_call"],
                        "exact_tool_name_set": metrics.get("exact_tool_name_set"),
                        "exact_tool_name_multiset": metrics.get("exact_tool_name_multiset"),
                        "exact_tool_sequence": metrics.get("exact_tool_sequence"),
                        "same_tool_call_count": metrics.get("same_tool_call_count"),
                        "exact_arguments": metrics.get("exact_arguments"),
                        "all_schema_valid": metrics["all_schema_valid"],
                        "all_required_args_present": metrics["all_required_args_present"],
                        "schema_valid_count": metrics["schema_valid_count"],
                        "required_args_count": metrics["required_args_count"],
                        "extra_call_count": metrics.get("extra_call_count"),
                        "missing_call_count": metrics.get("missing_call_count"),
                        "repeated_call_count": metrics.get("repeated_call_count"),
                        "extra_call_names": metrics.get("extra_call_names"),
                        "missing_call_names": metrics.get("missing_call_names"),
                        "repeated_call_names": metrics.get("repeated_call_names"),
                        "call_errors": metrics["call_errors"],
                    }
                )
                totals["ok"] += 1
                totals["valid_tool_json"] += int(row["valid_tool_json"])
                totals["exact_tool_name_set"] += int(bool(row["exact_tool_name_set"]))
                totals["exact_tool_sequence"] += int(bool(row["exact_tool_sequence"]))
                totals["exact_tool_name_multiset"] += int(bool(row["exact_tool_name_multiset"]))
                totals["same_tool_call_count"] += int(bool(row["same_tool_call_count"]))
                totals["exact_arguments"] += int(bool(row["exact_arguments"]))
                totals["all_schema_valid"] += int(bool(row["all_schema_valid"]))
                totals["all_required_args_present"] += int(bool(row["all_required_args_present"]))
                totals["records_with_extra_calls"] += int((row["extra_call_count"] or 0) > 0)
                totals["records_with_missing_calls"] += int((row["missing_call_count"] or 0) > 0)
                totals["records_with_repeated_calls"] += int((row["repeated_call_count"] or 0) > 0)
                totals["total_extra_calls"] += int(row["extra_call_count"] or 0)
                totals["total_missing_calls"] += int(row["missing_call_count"] or 0)
                totals["total_repeated_calls"] += int(row["repeated_call_count"] or 0)
            except (RuntimeError, urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
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
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "presence_penalty": args.presence_penalty,
        "enable_thinking": args.enable_thinking,
        "instruction_mode": args.instruction_mode,
        "totals": totals,
        "elapsed_seconds": time.time() - start,
    }
    summary_path = args.out_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
