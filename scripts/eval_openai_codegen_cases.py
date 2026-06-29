#!/usr/bin/env python3
import argparse
import ast
import json
import re
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_INPUT = ROOT / "data/codegen_eval/synthetic_codegen_10.jsonl"
DEFAULT_OUT = ROOT / "data/codegen_eval/synthetic_codegen_q36_teacher_10.jsonl"


CHILD_RUNNER = r"""
import json
import resource
import sys

payload = json.loads(sys.stdin.read())
resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
resource.setrlimit(resource.RLIMIT_AS, (768 * 1024 * 1024, 768 * 1024 * 1024))

safe_builtins = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
}

env = {"__builtins__": safe_builtins}
try:
    exec(payload["code"], env, env)
    for test in payload["tests"]:
        exec(test, env, env)
except Exception as exc:
    print(json.dumps({"passed": False, "error": f"{type(exc).__name__}: {exc}"}))
    raise SystemExit(1)

print(json.dumps({"passed": True}))
"""


FORBIDDEN_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}


FORBIDDEN_ATTRS = {
    "system",
    "popen",
    "remove",
    "rmdir",
    "unlink",
    "rename",
    "socket",
    "connect",
}


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


def ask_model(case, endpoint, model, timeout, temperature, max_tokens, enable_thinking):
    messages = list(case["prompt_messages"])
    if case.get("teacher_instruction"):
        messages.append({"role": "user", "content": case["teacher_instruction"]})
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    response = post_json(endpoint.rstrip("/") + "/chat/completions", payload, timeout)
    return response["choices"][0]["message"].get("content", "")


def extract_code(text):
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    match = re.search(r"```(?:python)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return cleaned


def static_check(code, entrypoint):
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"

    found_entrypoint = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False, "imports are not allowed"
        if isinstance(node, (ast.With, ast.AsyncWith)):
            return False, "with statements are not allowed"
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            return False, "global/nonlocal statements are not allowed"
        if isinstance(node, ast.FunctionDef) and node.name == entrypoint:
            found_entrypoint = True
        if isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES or node.id.startswith("__"):
                return False, f"forbidden name: {node.id}"
        if isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRS or node.attr.startswith("__"):
                return False, f"forbidden attribute: {node.attr}"
    if not found_entrypoint:
        return False, f"missing function: {entrypoint}"
    return True, ""


def run_tests(code, tests, timeout):
    payload = json.dumps({"code": code, "tests": tests})
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", CHILD_RUNNER],
            input=payload,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "TimeoutExpired"

    stdout = proc.stdout.strip().splitlines()
    parsed = {}
    if stdout:
        try:
            parsed = json.loads(stdout[-1])
        except json.JSONDecodeError:
            parsed = {}
    if proc.returncode == 0 and parsed.get("passed"):
        return True, ""
    return False, parsed.get("error") or proc.stderr.strip() or proc.stdout.strip() or f"returncode {proc.returncode}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", default="qwen3.6-27b-teacher")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--test-timeout", type=float, default=5.0)
    args = parser.parse_args()

    cases = load_cases(args.input_jsonl, args.limit)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    totals = {
        "records": 0,
        "ok": 0,
        "code_extracted": 0,
        "static_check_passed": 0,
        "tests_passed": 0,
        "errors": 0,
    }
    start = time.time()
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for idx, case in enumerate(cases):
            row = {
                "idx": idx,
                "source": case.get("source"),
                "id": case.get("id"),
                "task": case.get("task"),
                "entrypoint": case.get("entrypoint"),
            }
            try:
                assistant = ask_model(
                    case,
                    args.endpoint,
                    args.model,
                    args.timeout,
                    args.temperature,
                    args.max_tokens,
                    args.enable_thinking,
                )
                code = extract_code(assistant)
                has_code = bool(code.strip())
                static_ok, static_error = static_check(code, case["entrypoint"]) if has_code else (False, "empty code")
                tests_ok = False
                test_error = ""
                if static_ok:
                    tests_ok, test_error = run_tests(code, case.get("tests") or [], args.test_timeout)
                row.update(
                    {
                        "status": "ok",
                        "assistant": assistant,
                        "code": code,
                        "code_extracted": has_code,
                        "static_check_passed": static_ok,
                        "static_check_error": static_error,
                        "tests_passed": tests_ok,
                        "test_error": test_error,
                    }
                )
                totals["ok"] += 1
                totals["code_extracted"] += int(has_code)
                totals["static_check_passed"] += int(static_ok)
                totals["tests_passed"] += int(tests_ok)
            except Exception as exc:
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
