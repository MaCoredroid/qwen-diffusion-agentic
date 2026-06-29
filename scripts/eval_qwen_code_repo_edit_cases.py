#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_INPUT = ROOT / "data/repo_edit_eval/tiny_repo_edit_5.jsonl"
DEFAULT_OUT = ROOT / "data/repo_edit_eval/tiny_repo_edit_qwen_code_q36_5.jsonl"
DEFAULT_WORK_ROOT = ROOT / "runs/qwen_code_repo_edit_eval/work"
DEFAULT_QWEN_BIN = ROOT / "node_modules/.bin/qwen"
DEFAULT_PROXY = ROOT / "scripts/qwen_code_sglang_proxy.py"
DEFAULT_SYSTEM_PROMPT = (
    "You are a concise coding agent. Use read_file to inspect files, edit to "
    "patch the smallest necessary source change, and run_shell_command to run "
    "the requested tests. Stop once tests pass."
)


def load_cases(path, limit):
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
            if limit and len(cases) >= limit:
                break
    return cases


def write_files(repo, files):
    for relpath, content in files.items():
        path = repo / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def run_cmd(cmd, cwd, timeout, env=None):
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=env,
        )
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": 124,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "timed_out": True,
        }


def init_git(repo):
    run_cmd(["git", "init", "-q"], repo, 10)
    run_cmd(["git", "config", "user.email", "repo-edit-eval@example.local"], repo, 10)
    run_cmd(["git", "config", "user.name", "Repo Edit Eval"], repo, 10)
    run_cmd(["git", "add", "."], repo, 10)
    run_cmd(["git", "commit", "-qm", "initial"], repo, 10)


def git_diff_summary(repo):
    name_proc = run_cmd(["git", "diff", "--name-only"], repo, 10)
    stat_proc = run_cmd(["git", "diff", "--stat"], repo, 10)
    diff_proc = run_cmd(["git", "diff", "--", "."], repo, 10)
    changed = [line.strip() for line in name_proc["stdout"].splitlines() if line.strip()]
    return {
        "changed_files": changed,
        "diff_stat": stat_proc["stdout"],
        "diff": diff_proc["stdout"][:6000],
    }


def wait_http(url, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status < 500:
                    return True
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(0.25)
    return False


def check_upstream(endpoint, timeout):
    url = endpoint.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200, resp.read().decode("utf-8", errors="replace")[:1000]
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def start_proxy(args):
    log_path = args.out_jsonl.with_suffix(".proxy.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    cmd = [
        sys.executable,
        str(args.proxy_script),
        "--host",
        args.proxy_host,
        "--port",
        str(args.proxy_port),
        "--upstream",
        args.endpoint,
        "--max-tokens",
        str(args.proxy_max_tokens),
    ]
    if args.proxy_tool_choice:
        cmd.extend(["--tool-choice", args.proxy_tool_choice])
    if args.proxy_tool_choice_turns:
        cmd.extend(["--tool-choice-turns", str(args.proxy_tool_choice_turns)])
    if args.proxy_dump_dir:
        cmd.extend(["--dump-dir", str(args.proxy_dump_dir)])
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    health = f"http://{args.proxy_host}:{args.proxy_port}/health"
    if not wait_http(health, 15):
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_file.close()
        raise RuntimeError(f"proxy did not become healthy; see {log_path}")
    return proc, log_file, log_path


def parse_qwen_json(stdout):
    stripped = stdout.strip()
    if not stripped:
        return None, "empty stdout"
    try:
        return json.loads(stripped), ""
    except json.JSONDecodeError:
        pass
    lines = [line for line in stripped.splitlines() if line.strip()]
    parsed_lines = []
    for line in lines:
        try:
            parsed_lines.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if parsed_lines:
        if len(parsed_lines) == 1:
            return parsed_lines[0], ""
        return parsed_lines, ""
    return None, "could not parse JSON output"


def extract_qwen_stats(parsed):
    stats = {
        "events": 0,
        "tool_calls": None,
        "tool_success": None,
        "tool_fail": None,
        "tool_by_name": None,
        "duration_ms": None,
        "usage": None,
        "result": "",
    }
    if isinstance(parsed, list):
        stats["events"] = len(parsed)
        events = parsed
    elif isinstance(parsed, dict):
        stats["events"] = 1
        events = [parsed]
    else:
        return stats

    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("type") == "result" or "result" in event:
            result = event.get("result")
            if isinstance(result, str):
                stats["result"] = result
        tools = event.get("tools")
        if not isinstance(tools, dict) and isinstance(event.get("stats"), dict):
            tools = event["stats"].get("tools")
        if isinstance(tools, dict):
            stats["tool_calls"] = tools.get("totalCalls", stats["tool_calls"])
            stats["tool_success"] = tools.get("totalSuccess", stats["tool_success"])
            stats["tool_fail"] = tools.get("totalFail", stats["tool_fail"])
            stats["tool_by_name"] = tools.get("byName", stats["tool_by_name"])
        if "durationMs" in event or "duration_ms" in event:
            stats["duration_ms"] = event.get("durationMs", event.get("duration_ms"))
        if "usage" in event:
            stats["usage"] = event["usage"]
    return stats


def build_prompt(case):
    return (
        f"{case['prompt']}\n\n"
        f"Run `{case['test_command']}` in this repository and edit only the files "
        "needed to make the tests pass. Stop once the tests pass."
    )


def run_qwen_code(case, repo, args):
    env = os.environ.copy()
    env.update(
        {
            "QWEN_CODE_MAX_OUTPUT_TOKENS": str(args.qwen_code_max_output_tokens),
            "QWEN_CODE_SUPPRESS_YOLO_WARNING": "1",
        }
    )
    cmd = [
        str(args.qwen_bin),
        "--bare",
        "--auth-type",
        "openai",
        "--openai-api-key",
        "dummy",
        "--openai-base-url",
        f"http://{args.proxy_host}:{args.proxy_port}/v1",
        "--model",
        args.model,
        "--approval-mode",
        args.approval_mode,
        "--max-tool-calls",
        str(args.max_tool_calls),
        "--max-wall-time",
        args.max_wall_time,
        "--output-format",
        "json",
        "--system-prompt",
        args.system_prompt,
        "--exclude-tools",
        "agent",
        "--exclude-tools",
        "web_fetch",
        "--exclude-tools",
        "notebook_edit",
        "-p",
        build_prompt(case),
    ]
    return run_cmd(cmd, repo, args.qwen_timeout, env=env)


def eval_case(case, idx, args):
    repo = args.work_root / f"{idx:03d}_{case['id']}"
    if repo.exists():
        shutil.rmtree(repo)
    repo.mkdir(parents=True)
    write_files(repo, case["files"])
    init_git(repo)

    initial = run_cmd(case["test_command"].split(), repo, args.test_timeout)
    qwen_start = time.time()
    qwen = run_qwen_code(case, repo, args)
    qwen_elapsed = time.time() - qwen_start
    final = run_cmd(case["test_command"].split(), repo, args.test_timeout)
    diff = git_diff_summary(repo)
    parsed, parse_error = parse_qwen_json(qwen["stdout"])
    qwen_stats = extract_qwen_stats(parsed)

    changed_expected = any(path in set(case.get("expected_files") or []) for path in diff["changed_files"])
    unexpected = [
        path
        for path in diff["changed_files"]
        if path not in set(case.get("expected_files") or [])
    ]
    return {
        "idx": idx,
        "source": case.get("source"),
        "id": case["id"],
        "task": case.get("task"),
        "repo": str(repo),
        "test_command": case["test_command"],
        "initial_tests_failed": initial["returncode"] != 0,
        "initial_returncode": initial["returncode"],
        "qwen_returncode": qwen["returncode"],
        "qwen_timed_out": qwen["timed_out"],
        "qwen_elapsed_sec": round(qwen_elapsed, 3),
        "qwen_json_parse_error": parse_error,
        "qwen_stats": qwen_stats,
        "final_tests_passed": final["returncode"] == 0,
        "final_returncode": final["returncode"],
        "changed_expected_file": changed_expected,
        "changed_unexpected_files": unexpected,
        "diff_nonempty": bool(diff["changed_files"]),
        "changed_files": diff["changed_files"],
        "diff_stat": diff["diff_stat"],
        "diff": diff["diff"],
        "initial_stdout": initial["stdout"][-3000:],
        "initial_stderr": initial["stderr"][-3000:],
        "qwen_stdout": qwen["stdout"][-6000:],
        "qwen_stderr": qwen["stderr"][-6000:],
        "final_stdout": final["stdout"][-3000:],
        "final_stderr": final["stderr"][-3000:],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", default="qwen3.6-27b-teacher")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--test-timeout", type=float, default=15.0)
    parser.add_argument("--qwen-timeout", type=float, default=240.0)
    parser.add_argument("--max-wall-time", default="180s")
    parser.add_argument("--max-tool-calls", type=int, default=30)
    parser.add_argument("--approval-mode", default="yolo")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--qwen-bin", type=Path, default=DEFAULT_QWEN_BIN)
    parser.add_argument("--proxy-script", type=Path, default=DEFAULT_PROXY)
    parser.add_argument("--proxy-host", default="127.0.0.1")
    parser.add_argument("--proxy-port", type=int, default=30001)
    parser.add_argument("--proxy-max-tokens", type=int, default=512)
    parser.add_argument("--proxy-tool-choice", default="required")
    parser.add_argument("--proxy-tool-choice-turns", type=int, default=0)
    parser.add_argument("--proxy-dump-dir", type=Path, default=None)
    parser.add_argument("--qwen-code-max-output-tokens", type=int, default=512)
    args = parser.parse_args()

    if not args.qwen_bin.exists():
        raise SystemExit(f"missing Qwen Code binary: {args.qwen_bin}; run npm install")
    if not args.proxy_script.exists():
        raise SystemExit(f"missing proxy script: {args.proxy_script}")

    ok, upstream_detail = check_upstream(args.endpoint, 5)
    if not ok:
        raise SystemExit(f"upstream is not ready at {args.endpoint}: {upstream_detail}")

    cases = load_cases(args.input_jsonl, args.limit)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.work_root.mkdir(parents=True, exist_ok=True)

    proxy_proc = None
    proxy_log = None
    try:
        proxy_proc, proxy_log, proxy_log_path = start_proxy(args)
        totals = {
            "records": 0,
            "initial_tests_failed": 0,
            "qwen_exit_zero": 0,
            "final_tests_passed": 0,
            "changed_expected_file": 0,
            "changed_unexpected_file": 0,
            "diff_nonempty": 0,
            "errors": 0,
        }
        start = time.time()
        with args.out_jsonl.open("w", encoding="utf-8") as f:
            for idx, case in enumerate(cases):
                totals["records"] += 1
                try:
                    row = eval_case(case, idx, args)
                except Exception as exc:
                    row = {
                        "idx": idx,
                        "id": case.get("id"),
                        "task": case.get("task"),
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    totals["errors"] += 1
                else:
                    row["status"] = "ok"
                    totals["initial_tests_failed"] += int(row["initial_tests_failed"])
                    totals["qwen_exit_zero"] += int(row["qwen_returncode"] == 0)
                    totals["final_tests_passed"] += int(row["final_tests_passed"])
                    totals["changed_expected_file"] += int(row["changed_expected_file"])
                    totals["changed_unexpected_file"] += int(bool(row["changed_unexpected_files"]))
                    totals["diff_nonempty"] += int(row["diff_nonempty"])
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                print(json.dumps(row, ensure_ascii=False))

        manifest = {
            "out": str(args.out_jsonl),
            "input": str(args.input_jsonl),
            "records": totals["records"],
            "totals": totals,
            "elapsed_sec": round(time.time() - start, 3),
            "endpoint": args.endpoint,
            "model": args.model,
            "proxy_log": str(proxy_log_path),
        }
        manifest_path = args.out_jsonl.with_suffix(".manifest.json")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(manifest, indent=2))
    finally:
        if proxy_proc is not None:
            proxy_proc.terminate()
            try:
                proxy_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proxy_proc.kill()
        if proxy_log is not None:
            proxy_log.close()


if __name__ == "__main__":
    main()
