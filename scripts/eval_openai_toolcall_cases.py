#!/usr/bin/env python3
import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from eval_toolcall_jsonl import score_tool_call_objects, score_tool_calls


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_INPUT = ROOT / "data/toolcall_eval/synthetic_toolresult_openai_smoke.jsonl"
DEFAULT_OUT = ROOT / "data/toolcall_eval/synthetic_toolresult_openai_teacher.jsonl"


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


def message_content(message):
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def empty_totals():
    return {
        "records": 0,
        "ok": 0,
        "native_tool_call_response": 0,
        "text_fallback_response": 0,
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


def add_metrics_to_totals(totals, row):
    totals["valid_tool_json"] += int(bool(row["valid_tool_json"]))
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


def build_payload(case, args):
    messages = list(case[args.message_field])
    if args.append_instruction and case.get("teacher_instruction"):
        messages.append({"role": "user", "content": case["teacher_instruction"]})
    payload = {
        "model": args.model,
        "messages": messages,
        "tools": case.get("tools") or None,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
    }
    if args.tool_choice:
        payload["tool_choice"] = args.tool_choice
    return {key: value for key, value in payload.items() if value is not None}


def score_message(message, case, allow_text_fallback):
    native_tool_calls = message.get("tool_calls") or []
    content = message_content(message)
    if native_tool_calls:
        metrics = score_tool_call_objects(
            native_tool_calls,
            case.get("tools") or [],
            gold_tool_calls=case.get("gold_tool_calls"),
            gold_text=case.get("gold_assistant"),
        )
        return "native", metrics
    if allow_text_fallback:
        return "text", score_tool_calls(content, case.get("tools") or [], case.get("gold_assistant"))
    metrics = score_tool_call_objects([], case.get("tools") or [], gold_tool_calls=case.get("gold_tool_calls"))
    return "none", metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", default="qwen3.6-27b-teacher")
    parser.add_argument("--message-field", default="prompt_messages")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--tool-choice", default="auto")
    parser.add_argument("--append-instruction", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--allow-text-fallback", action="store_true")
    args = parser.parse_args()

    cases = load_cases(args.input_jsonl, args.limit)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    totals = empty_totals()
    start = time.time()
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for idx, case in enumerate(cases):
            row = {
                "idx": idx,
                "source": case.get("source"),
                "id": case.get("id"),
                "history_style": case.get("history_style"),
                "gold_tool_names": case.get("gold_tool_names") or [],
                "completed_tool_names": case.get("completed_tool_names") or [],
            }
            try:
                payload = build_payload(case, args)
                response = post_json(args.endpoint.rstrip("/") + "/chat/completions", payload, args.timeout)
                message = response["choices"][0]["message"]
                score_source, metrics = score_message(message, case, args.allow_text_fallback)
                row.update(
                    {
                        "status": "ok",
                        "score_source": score_source,
                        "assistant_content": message_content(message),
                        "native_tool_calls": message.get("tool_calls") or [],
                        "called_names": metrics["called_names"],
                        "calls": metrics["calls"],
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
                totals["native_tool_call_response"] += int(score_source == "native")
                totals["text_fallback_response"] += int(score_source == "text")
                add_metrics_to_totals(totals, row)
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
        "message_field": args.message_field,
        "tool_choice": args.tool_choice,
        "append_instruction": args.append_instruction,
        "allow_text_fallback": args.allow_text_fallback,
        "totals": totals,
        "elapsed_seconds": time.time() - start,
    }
    summary_path = args.out_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
