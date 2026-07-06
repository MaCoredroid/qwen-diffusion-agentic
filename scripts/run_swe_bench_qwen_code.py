#!/usr/bin/env python3
"""SWE-Bench per-instance orchestrator driving QWEN CODE (headless) against a
local vLLM /v1/chat/completions endpoint (diffusion :9952 or stock-AR :9951).

USER DIRECTIVE (2026-07-05): Stage C runs through Qwen Code with LumoFlyWheel as
the REFERENCE implementation. This ports the flywheel Codex orchestrator
(`scripts/run_swe_bench_q36_a.py`, synced 2026-07-05) to Qwen Code as the agent
CLI, keeping the flywheel's episode / eval / reward conventions so results are
comparable with the flywheel's own SWE runs.

Per-instance protocol (unchanged from the flywheel, §11 of the bounded-time spec):
  1. Hydrate the workspace at the SWE-Bench base_commit (git worktree --detach),
     drop AGENTS.md carrying the problem_statement (flywheel `_write_agents_md`).
  2. Run the Qwen Code episode (headless, --output-format json) with the workspace
     as CWD, driven through the qwen_code_sglang_proxy.py adapter -> vLLM.
  3. Diff workspace vs base_commit -> patch.diff (`git diff --binary base_commit`).
  4. Evaluate the patch (flywheel exit-code contract 0=resolved/1=failed/2=crash),
     emit the flywheel eval_report.json / normalized_eval.json / predictions.jsonl.
  5. Write per-task artifacts under <out>/<dataset>/per_task/<instance_id>/.
  6. Aggregate predictions.jsonl + campaign_summary.json (flywheel `_aggregate`).

DIVERGENCES from the Codex driver (documented; user directive "adapt minimally"):
  A. AGENT: local `node_modules/.bin/qwen` (Qwen Code @0.19.2) via the
     qwen_code_sglang_proxy.py adapter -> vLLM Chat Completions, NOT `codex exec
     --json` in a docker container over the Responses API. Qwen Code speaks Chat
     Completions natively so no Responses proxy is needed (plan §C1). This reuses
     the Stage-A-proven agent invocation from runs/stage_a_smoke.
  B. EVAL: docker + the `swebench` package are ABSENT on this 5090 serving box,
     so the official docker harness eval CANNOT run locally here. Per plan §C3 the
     eval is OFFLOADED to the x86 `alienware` box (--eval-mode offload, ported from
     the flywheel `_run_eval_remote`), OR stubbed for no-docker plumbing dry-runs
     (--eval-mode mock: a gold-patch comparison stand-in, clearly labelled, NOT a
     real score), OR run in-process where swebench+docker DO exist (--eval-mode
     local, the flywheel codex_bench_eval_swe path).
  C. Dropped GB10-specific machinery (DCGM sampler, Prometheus per-request capture
     slicing, codex-on-alienware rsync offload) irrelevant to this topology; kept
     the load-bearing per-task artifact set + conventions.
  D. R4 (Stage-A finding, runs/stage_a_smoke/report.md): the diffusion qwen-code
     loop can exit NON-ZERO (loop-detector halt exit 1; budget exit 55) AFTER
     completing useful work. The verdict is therefore scored from the eval outcome
     + extracted patch, NEVER from the qwen CLI exit code (recorded separately for
     diagnostics). This is exactly how the flywheel scores (patch/eval, not exit).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_OUT_ROOT = REPO_ROOT / "runs" / "stage_c_driver" / "output"
DEFAULT_REPO_CACHE = REPO_ROOT / ".cache" / "swe_bench_repos"
DEFAULT_HF_HOME = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
DEFAULT_QWEN_BIN = REPO_ROOT / "node_modules" / ".bin" / "qwen"
DEFAULT_PROXY_SCRIPT = REPO_ROOT / "scripts" / "qwen_code_sglang_proxy.py"

# Endpoint defaults follow the Stage-A certified serving ports.
DEFAULT_ENDPOINT = "http://127.0.0.1:9951/v1"          # stock-AR arm
DEFAULT_MODEL = "qwen3.5-9b-ar"                         # AR served-model-name
DEFAULT_MODEL_NAME_TAG = "qwen3.5-9b-ar::qwen-code-0.19.2::stage-c"

# Per-attempt agent wall (subprocess timeout). 0 => rely on the qwen CLI's own
# --max-wall-time / --max-session-turns budgets (like the flywheel's codex idle
# timeout backstop). Set >0 for a hard harness wall.
DEFAULT_AGENT_WALL_S = 0
DEFAULT_MAX_SESSION_TURNS = 80    # flywheel QWEN_CODE_TEMPLATE default
DEFAULT_QWEN_MAX_WALL = "1800s"   # qwen CLI run-level budget (exit 55 on overrun)
DEFAULT_QWEN_MAX_OUTPUT_TOKENS = 32768   # flywheel R1 context-budget fix
DEFAULT_EVAL_TIMEOUT_S = 30 * 60

# Proxy adapter (Stage-A): inject enable_thinking=false, clamp max_tokens, pass
# `tools` through unchanged so the diffusion A2 grammar bridge (or AR qwen3_xml
# parser) runs server-side. Natural tool_choice (no forcing) so the AR arm can
# terminate with a free-text turn (R4).
DEFAULT_PROXY_HOST = "127.0.0.1"
DEFAULT_PROXY_PORT = 30021
DEFAULT_PROXY_MAX_TOKENS = 2048   # SWE edits need more than the toy-smoke 512

# ---------------------------------------------------------------------------
# Operator prompts (ported from the flywheel; wording generalized codex ->
# qwen-code: Qwen Code edits via its `edit`/`replace`/`write_file` tools, not
# codex's apply_patch, but the intent is identical).
# ---------------------------------------------------------------------------
DEFAULT_AGENT_PROMPT = (
    "Read the task prompt at ./AGENTS.md and complete it in this workspace. "
    "Edit the source files directly to implement the fix. Do not write a diff "
    "file -- modify the files in place so that running the project's tests passes "
    "the tests described in the prompt. Do NOT modify any test files."
)
RETRY_PROMPT_EMPTY = (
    "Your previous attempt finished WITHOUT leaving any code change in the working "
    "tree. Re-read ./AGENTS.md, inspect the relevant source files, and EDIT them "
    "now to implement the fix. Do not stop until you have made a concrete source "
    "edit. Do not waste time on environment setup or pip/conda installs -- the "
    "grader uses its own environment."
)
RETRY_PROMPT_SETUP_LOOP = (
    "Your previous attempt repeatedly hit the same failing command (likely an "
    "environment/install/build step) and never edited the source. STOP trying that "
    "approach entirely. The grader builds its own environment, so you do NOT need "
    "the project to install or import. Read ./AGENTS.md and the relevant source "
    "files, and directly EDIT the source to implement the fix."
)


def _iso_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Dataset / subset loading (ported verbatim from the flywheel).
# ---------------------------------------------------------------------------
def _load_subset(subset_json: Path) -> tuple[str, list[str]]:
    payload = json.loads(subset_json.read_text())
    return payload["dataset_name"], list(payload["instance_ids"])


def _load_dataset(dataset_name: str, split: str = "test") -> dict[str, dict]:
    # Local .json / .jsonl (e.g. the SWE-Gym probe pool dumped to a file) — no HF,
    # arbitrary split. Keeps the container/patch path identical; only the record
    # source differs.
    if dataset_name.endswith(".json") or dataset_name.endswith(".jsonl"):
        p = Path(dataset_name)
        if dataset_name.endswith(".jsonl"):
            recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        else:
            recs = json.loads(p.read_text())
        return {ex["instance_id"]: dict(ex) for ex in recs}
    os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    return {ex["instance_id"]: dict(ex) for ex in ds}


# ---------------------------------------------------------------------------
# Workspace hydrate / teardown (ported verbatim from the flywheel).
# ---------------------------------------------------------------------------
def _repo_clone_url(repo: str) -> str:
    return f"https://github.com/{repo}.git"


def _ensure_repo_cache(repo: str, cache_root: Path) -> Path:
    safe = repo.replace("/", "__")
    cache_path = cache_root / safe
    if not cache_path.is_dir():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--filter=blob:none", _repo_clone_url(repo), str(cache_path)],
            check=True,
        )
    return cache_path


def _fetch_commit(cache_path: Path, base_commit: str) -> None:
    rc = subprocess.run(
        ["git", "-C", str(cache_path), "cat-file", "-e", base_commit]
    ).returncode
    if rc != 0:
        subprocess.run(
            ["git", "-C", str(cache_path), "fetch", "origin", base_commit],
            check=False,
        )


def _hydrate_workspace(*, cache_path: Path, base_commit: str, workspace_path: Path) -> None:
    if workspace_path.exists():
        _remove_workspace(cache_path, workspace_path)
    workspace_path.parent.mkdir(parents=True, exist_ok=True)
    abs_workspace = workspace_path.resolve()
    subprocess.run(
        ["git", "-C", str(cache_path), "worktree", "add", "--detach",
         str(abs_workspace), base_commit],
        check=True,
    )


def _remove_workspace(cache_path: Path, workspace_path: Path) -> None:
    abs_workspace = workspace_path.resolve() if workspace_path.exists() else workspace_path
    if not abs_workspace.exists():
        return
    subprocess.run(
        ["git", "-C", str(cache_path), "worktree", "remove", "--force", str(abs_workspace)],
        check=False,
    )
    if abs_workspace.exists():
        shutil.rmtree(abs_workspace, ignore_errors=True)


def _write_agents_md(workspace: Path, instance: dict) -> None:
    """AGENTS.md drop — byte-for-byte the flywheel convention so the task prompt
    the agent sees is identical to the flywheel's Codex campaigns."""
    body: list[str] = []
    body.append(f"# SWE-Bench task: {instance['instance_id']}")
    body.append("")
    body.append(f"**Repo:** `{instance['repo']}`  ")
    body.append(f"**Base commit:** `{instance['base_commit']}`  ")
    if instance.get("version"):
        body.append(f"**Version:** `{instance['version']}`  ")
    body.append("")
    body.append("## Problem statement")
    body.append("")
    body.append(instance.get("problem_statement") or "(empty problem statement)")
    body.append("")
    body.append("## Required behavior")
    body.append("")
    body.append(
        "Implement the fix described in the problem statement by editing the "
        "source files in this workspace. Do NOT modify any test files. The "
        "hidden grader will apply its own test patch and run the test suite; "
        "your code must make those tests pass without breaking existing ones."
    )
    body.append("")
    body.append("## How to work (important)")
    body.append("")
    body.append(
        "- Reason carefully and thoroughly before each tool call. First inspect "
        "the relevant source files to confirm your understanding of the bug, "
        "then make the minimal correct edit.\n"
        "- Do NOT spend your time trying to `pip install` or build/conda the "
        "project -- the grader runs in its own prepared environment. If an "
        "install/build command fails, do not retry it; just edit the source.\n"
        "- You MUST finish by leaving an actual code change in the working tree. "
        "Do not stop until you have edited the source files to implement the fix."
    )
    body.append("")
    (workspace / "AGENTS.md").write_text("\n".join(body) + "\n", encoding="utf-8")


def _extract_patch(workspace: Path, base_commit: str) -> str:
    """Ported verbatim: tracked-file diff vs base_commit (binary-safe)."""
    proc = subprocess.run(
        ["git", "-C", str(workspace), "diff", "--no-color", "--binary", base_commit],
        capture_output=True, text=True, check=False,
    )
    return proc.stdout


# ---------------------------------------------------------------------------
# CONTAINER RUNTIME (RUNTIME_ALIGNMENT_DIRECTIVE, 2026-07-05).
#
# Official SWE-bench Verified hands the agent the per-instance RUNTIME: the repo
# checked out at base_commit and EDITABLE-INSTALLED into a conda env (`testbed`)
# with every dependency, plus build artifacts (*.egg-info, pytest _version.py,
# compiled extensions). Our host git-worktree path lacked all of that, so an
# in-episode `import <pkg>` or test run died on missing deps -> the whole N=5
# battery was deprecated. This wires each episode INSIDE the official image.
#
# PATTERN (documented design choice): WORKSPACE-MOUNT + DOCKER-EXEC hybrid.
#   * The image's /testbed is the ground-truth runtime; we do NOT rebuild it. We
#     SEED a host workspace from the image's /testbed (`docker cp`) so the
#     editable install + build artifacts are preserved verbatim, then bind-MOUNT
#     that workspace back over /testbed in a long-lived per-instance container.
#     qwen-code stays on the HOST with CWD=workspace, so its native FILE tools
#     (edit / write_file / replace) write the mount == the container's /testbed.
#   * Every SHELL action is routed INTO the container via `docker exec` with the
#     official conda-activation preamble (`conda activate testbed; cd /testbed`),
#     so imports / builds / tests use the prepared per-instance environment.
#     For a live qwen-code episode a `bash` PATH-shim (see _write_shell_shim)
#     forwards run_shell_command into that same `docker exec`.
#   * Patch = tracked `git diff base_commit` over the shared /testbed tree.
# Rejected alternative: bind-mount a BARE git checkout over /testbed -> shadows
# the editable install + build artifacts -> exactly the "troubled env" the
# directive rejects. Seeding the mount from the image is what avoids that.
# ---------------------------------------------------------------------------
CONTAINER_TESTBED = "/testbed"
# Byte-for-byte the swebench eval-script preamble (conda + locale + cd).
CONTAINER_ACTIVATE = (
    "source /opt/miniconda3/bin/activate testbed && cd /testbed && "
    "export LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8"
)


def _docker_base() -> list[str]:
    """Docker CLI prefix. `docker` where the docker group is active; override via
    SWE_DOCKER_CMD (e.g. 'sudo -A docker' with SUDO_ASKPASS) where it is not."""
    return shlex.split(os.environ.get("SWE_DOCKER_CMD", "docker"))


def _docker_arch() -> str:
    import platform
    m = platform.machine().lower()
    return "arm64" if m in ("arm64", "aarch64") else "x86_64"


def _container_image_for(instance_id: str) -> str:
    """Official Docker Hub image key for an instance. swebench substitutes the
    `__` in instance ids with `_1776_` (Docker Hub forbids `__` in repo names)."""
    slug = instance_id.replace("__", "_1776_")
    return f"swebench/sweb.eval.{_docker_arch()}.{slug}:latest"


def _drun(argv: list[str], *, timeout: int = 600, check: bool = False,
          text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(_docker_base() + argv, capture_output=True, text=text,
                          timeout=timeout, check=check)


def _image_present(image: str) -> bool:
    return _drun(["image", "inspect", image], timeout=60).returncode == 0


def _seed_workspace_from_image(*, image: str, workspace: Path) -> None:
    """Materialise the image's /testbed onto the host so it can be bind-mounted
    back (preserving the editable install + build artifacts). `docker cp` from a
    created-but-never-started throwaway container; the real container mounts the
    result over /testbed."""
    if workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True, exist_ok=True)
    cid = _drun(["create", image, "sleep", "3600"], timeout=120, check=True).stdout.strip()
    try:
        cp = _drun(["cp", f"{cid}:{CONTAINER_TESTBED}/.", str(workspace)], timeout=600)
        if cp.returncode != 0:
            raise RuntimeError(f"docker cp seed failed: {cp.stderr[-400:]}")
    finally:
        _drun(["rm", "-f", cid], timeout=60)


def _start_container(*, name: str, image: str, workspace: Path) -> None:
    """Long-lived per-instance container with the seeded workspace bind-mounted
    over /testbed. chown the mount to the host uid (via the container's own root)
    so host-side qwen-code file edits work; portable, needs no host sudo."""
    _drun(["rm", "-f", name], timeout=60)
    run = _drun(["run", "-d", "--name", name, "-v", f"{workspace}:{CONTAINER_TESTBED}",
                 image, "sleep", "infinity"], timeout=120)
    if run.returncode != 0:
        raise RuntimeError(f"docker run failed: {run.stderr[-400:]}")
    _cexec(name, f"chown -R {os.getuid()}:{os.getgid()} {CONTAINER_TESTBED} "
                 f"&& git config --global --add safe.directory {CONTAINER_TESTBED}",
           activate=False, timeout=180)


def _stop_container(name: str) -> None:
    _drun(["rm", "-f", name], timeout=120)


def _teardown_container(name: str, workspace: Path | None = None) -> None:
    """Robust teardown: chown the bind-mounted /testbed back to the host uid FIRST
    (the container runs as root, so builds/tests leave root-owned __pycache__/
    egg-info the host user otherwise cannot delete), THEN remove the container and
    the seeded workspace. Idempotent + never raises."""
    try:
        _cexec(name, f"chown -R {os.getuid()}:{os.getgid()} {CONTAINER_TESTBED}",
               activate=False, timeout=180)
    except Exception:  # noqa: BLE001
        pass
    _stop_container(name)
    if workspace is not None:
        shutil.rmtree(workspace, ignore_errors=True)


def _cexec(name: str, command: str, *, activate: bool = True, workdir: str = CONTAINER_TESTBED,
           timeout: int = 1800) -> subprocess.CompletedProcess:
    """Run a shell command INSIDE the per-instance container. When activate=True
    the official conda `testbed` env + locale + cd /testbed preamble is prepended
    (this is what makes in-episode imports / tests use the aligned runtime)."""
    inner = f"{CONTAINER_ACTIVATE} && {command}" if activate else command
    return _drun(["exec", "-w", workdir, name, "/bin/bash", "-lc", inner], timeout=timeout)


def _container_extract_patch(name: str, base_commit: str) -> str:
    """Tracked diff vs base_commit computed INSIDE the container (shares /testbed
    with the host mount, so identical to a host diff but avoids ownership races)."""
    cp = _cexec(name, f"git -c core.fileMode=false diff --no-color --binary {base_commit}",
                activate=False, timeout=300)
    return cp.stdout


def _write_shell_shim(*, shim_dir: Path, container: str) -> Path:
    """Write a `bash` shim (first on PATH) that forwards qwen-code's
    run_shell_command (`bash -c "<cmd>"`) into `docker exec` on the per-instance
    container, conda-activated. qwen-code's native FILE tools already reach the
    container through the /testbed bind-mount; this covers the SHELL tool so a
    live episode's builds/imports/tests run in the aligned env. Non-`-c`
    invocations fall through to the real bash."""
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim = shim_dir / "bash"
    shim.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, os\n"
        f"DOCKER = {json.dumps(_docker_base())}\n"
        f"CONTAINER = {json.dumps(container)}\n"
        f"PRE = {json.dumps(CONTAINER_ACTIVATE)}\n"
        "args = sys.argv[1:]\n"
        "cmd = None\n"
        "for i, a in enumerate(args):\n"
        "    if a in ('-c', '-lc', '-cl') and i + 1 < len(args):\n"
        "        cmd = args[i + 1]; break\n"
        "    if a.startswith('-') and 'c' in a and i + 1 < len(args):\n"
        "        cmd = args[i + 1]; break\n"
        "if cmd is None:\n"
        "    os.execv('/bin/bash', ['/bin/bash'] + args)\n"
        "os.execvp(DOCKER[0], DOCKER + ['exec', '-w', '/testbed', CONTAINER,\n"
        "          '/bin/bash', '-lc', PRE + ' && ' + cmd])\n",
        encoding="utf-8",
    )
    shim.chmod(0o755)
    return shim


def _run_mock_agent_container(*, container: str, instance: dict, base_commit: str,
                              workspace: Path, trace_path: Path) -> dict[str, Any]:
    """Scripted (NO model) agent for the container runtime: replay the dataset
    gold `patch` THROUGH the container (git apply via docker exec) to prove the
    hydrate -> in-container edit -> in-container patch-extract plumbing."""
    started = time.monotonic()
    gold = instance.get("patch") or ""
    applied = False
    apply_err = ""
    if gold.strip():
        (workspace / ".mock_gold.patch").write_text(gold, encoding="utf-8")
        cp = _cexec(container, "git apply --whitespace=nowarn .mock_gold.patch",
                    activate=False, timeout=180)
        applied = cp.returncode == 0
        apply_err = (cp.stderr or "").strip()[:800]
        (workspace / ".mock_gold.patch").unlink(missing_ok=True)
    trace_path.write_text(json.dumps({
        "mock_agent": True, "runtime": "container", "action": "replay_gold_patch_in_container",
        "gold_patch_bytes": len(gold), "applied": applied, "apply_stderr": apply_err,
    }, indent=2) + "\n", encoding="utf-8")
    return {
        "elapsed_s": round(time.monotonic() - started, 3), "exit_code": 0 if applied else 1,
        "timed_out": False, "cli_exit_is_verdict": False, "parsed": True,
        "subtype": "mock_gold_replay_container", "num_turns": 1, "duration_api_ms": 0,
        "usage": None, "tool_calls": 1, "tool_by_name": {"mock_apply_in_container": 1},
        "result_tail": f"gold_replay_in_container applied={applied}", "mock_apply_err": apply_err,
    }


# ---------------------------------------------------------------------------
# Qwen Code proxy lifecycle (Stage-A qwen_code_sglang_proxy.py adapter).
# ---------------------------------------------------------------------------
def _wait_http(url: str, timeout: float) -> bool:
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


def _check_upstream(endpoint: str, timeout: float) -> tuple[bool, str]:
    url = endpoint.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200, resp.read().decode("utf-8", errors="replace")[:400]
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def _start_proxy(args, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    cmd = [
        sys.executable, str(args.proxy_script),
        "--host", args.proxy_host,
        "--port", str(args.proxy_port),
        "--upstream", args.endpoint,
        "--max-tokens", str(args.proxy_max_tokens),
    ]
    if args.proxy_tool_choice:
        cmd += ["--tool-choice", args.proxy_tool_choice]
    if args.proxy_dump_dir:
        cmd += ["--dump-dir", str(args.proxy_dump_dir)]
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    if not _wait_http(f"http://{args.proxy_host}:{args.proxy_port}/health", 15):
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_file.close()
        raise RuntimeError(f"proxy did not become healthy; see {log_path}")
    return proc, log_file


def _stop_proxy(proc, log_file) -> None:
    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    if log_file is not None:
        log_file.close()


# ---------------------------------------------------------------------------
# Qwen Code episode runner (replaces the flywheel's `_run_codex`).
# ---------------------------------------------------------------------------
def _parse_qwen_result(stdout: str) -> dict[str, Any]:
    """Parse the qwen `--output-format json` output into a compact summary.

    Qwen Code emits either a single result object or a list of events; the final
    result object carries `subtype` (success | error_during_execution), `num_turns`,
    `duration_ms`/`duration_api_ms`, `usage` (token counts), `stats.tools`, and
    `result` (final text). Robust to either shape."""
    out: dict[str, Any] = {
        "parsed": False, "subtype": None, "num_turns": None,
        "duration_api_ms": None, "usage": None,
        "tool_calls": None, "tool_by_name": None, "result_tail": "",
    }
    stripped = (stdout or "").strip()
    if not stripped:
        return out
    obj = None
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        # try last JSON line
        for line in reversed(stripped.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    if obj is None:
        return out
    events = obj if isinstance(obj, list) else [obj]
    out["parsed"] = True
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("subtype"):
            out["subtype"] = ev.get("subtype")
        for k in ("num_turns",):
            if ev.get(k) is not None:
                out[k] = ev.get(k)
        for k in ("duration_api_ms", "durationApiMs", "duration_ms", "durationMs"):
            if ev.get(k) is not None:
                out["duration_api_ms"] = ev.get(k)
        if isinstance(ev.get("usage"), dict):
            out["usage"] = ev["usage"]
        stats = ev.get("stats") if isinstance(ev.get("stats"), dict) else None
        tools = ev.get("tools") if isinstance(ev.get("tools"), dict) else (
            stats.get("tools") if stats and isinstance(stats.get("tools"), dict) else None)
        if isinstance(tools, dict):
            out["tool_calls"] = tools.get("totalCalls", out["tool_calls"])
            out["tool_by_name"] = tools.get("byName", out["tool_by_name"])
        if isinstance(ev.get("result"), str) and ev["result"].strip():
            out["result_tail"] = ev["result"].strip()[-600:]
    return out


def _run_qwen_code(
    *,
    workspace: Path,
    proxy_base_url: str,
    model: str,
    timeout_s: int,
    instance_id: str,
    args,
    trace_path: Path,
    stderr_path: Path,
    prompt: str = DEFAULT_AGENT_PROMPT,
    extra_path: Path | None = None,
) -> dict[str, Any]:
    """Run the local Qwen Code CLI headless with CWD=workspace, capturing the
    --output-format json event stream to trace_path and CLI stderr to stderr_path.

    Mirrors the Stage-A-proven invocation (runs/stage_a_smoke) + the flywheel
    QWEN_CODE_TEMPLATE budgets (--max-session-turns, MAX_OUTPUT_TOKENS).

    extra_path: dir prepended to PATH (the container runtime passes the `bash`
    shell-shim dir here so run_shell_command routes into `docker exec`)."""
    env = os.environ.copy()
    env.update({
        "QWEN_CODE_MAX_OUTPUT_TOKENS": str(args.qwen_max_output_tokens),
        "QWEN_CODE_SUPPRESS_YOLO_WARNING": "1",
        "OPENAI_API_KEY": "dummy",
        "OPENAI_BASE_URL": proxy_base_url,
        "OPENAI_MODEL": model,
        "QWEN_MODEL": model,
    })
    if extra_path is not None:
        env["PATH"] = str(extra_path) + os.pathsep + env.get("PATH", "")
    cmd = [
        str(args.qwen_bin),
        "--bare",
        "--auth-type", "openai",
        "--openai-api-key", "dummy",
        "--openai-base-url", proxy_base_url,
        "--model", model,
        "--approval-mode", "yolo",
        "--max-session-turns", str(args.max_session_turns),
        "--max-wall-time", args.qwen_max_wall,
        "--output-format", "json",
        "--exclude-tools", "agent",
        "--exclude-tools", "web_fetch",
        "--exclude-tools", "notebook_edit",
    ]
    if args.system_prompt:
        cmd += ["--system-prompt", args.system_prompt]
    cmd += ["-p", prompt]

    started = time.monotonic()
    rc: int | None = None
    timed_out = False
    with trace_path.open("w", encoding="utf-8") as tf, stderr_path.open("w", encoding="utf-8") as ef:
        try:
            completed = subprocess.run(
                cmd, cwd=str(workspace), stdout=tf, stderr=ef, env=env,
                timeout=(None if timeout_s <= 0 else max(timeout_s, 30)),
                check=False,
            )
            rc = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            rc = -1
    elapsed = time.monotonic() - started
    parsed = _parse_qwen_result(trace_path.read_text(errors="replace") if trace_path.is_file() else "")
    return {
        "elapsed_s": round(elapsed, 3),
        "exit_code": rc if rc is not None else -1,
        "timed_out": timed_out,
        # R4: exit_code is DIAGNOSTIC ONLY — 1 (loop-detector halt) / 55 (budget)
        # are NOT failures if the patch/eval says otherwise.
        "cli_exit_is_verdict": False,
        **{k: parsed[k] for k in ("parsed", "subtype", "num_turns", "duration_api_ms",
                                  "usage", "tool_calls", "tool_by_name", "result_tail")},
    }


def _run_mock_agent(*, workspace: Path, instance: dict, trace_path: Path) -> dict[str, Any]:
    """Dry-run agent (NO model server): replay the dataset gold `patch` into the
    worktree via `git apply`. Proves the hydrate -> edit -> patch-extract plumbing
    end-to-end deterministically. Records what it did to trace_path."""
    started = time.monotonic()
    gold = instance.get("patch") or ""
    applied = False
    apply_err = ""
    if gold.strip():
        pf = workspace / ".mock_gold.patch"
        pf.write_text(gold, encoding="utf-8")
        proc = subprocess.run(
            ["git", "-C", str(workspace), "apply", "--whitespace=nowarn", str(pf)],
            capture_output=True, text=True, check=False,
        )
        applied = proc.returncode == 0
        apply_err = proc.stderr.strip()[:800]
        pf.unlink(missing_ok=True)
    trace_path.write_text(json.dumps({
        "mock_agent": True, "action": "replay_gold_patch",
        "gold_patch_bytes": len(gold), "applied": applied, "apply_stderr": apply_err,
    }, indent=2) + "\n", encoding="utf-8")
    return {
        "elapsed_s": round(time.monotonic() - started, 3),
        "exit_code": 0 if applied else 1,
        "timed_out": False, "cli_exit_is_verdict": False,
        "parsed": True, "subtype": "mock_gold_replay",
        "num_turns": 1, "duration_api_ms": 0, "usage": None,
        "tool_calls": 1, "tool_by_name": {"mock_apply": 1},
        "result_tail": f"gold_replay applied={applied}",
        "mock_apply_err": apply_err,
    }


# ---------------------------------------------------------------------------
# Empty-patch classification + state-conditional retry (ported from flywheel;
# default retries 0 = nudge-only, dormant unless SWE_EMPTY_PATCH_RETRIES>=1).
# ---------------------------------------------------------------------------
def _classify_empty_patch_cause(trace_path: Path) -> str:
    """>=3 identical failing shell commands => setup_loop, else agent_gave_up.
    Reads the qwen json event stream (tool call args)."""
    try:
        text = trace_path.read_text(errors="replace") if trace_path.is_file() else ""
        cmds: Counter = Counter()
        # qwen json: shell tool calls appear as run_shell_command with a `command`.
        for tok in text.split('"command"'):
            frag = tok[:400]
            start = frag.find(":")
            if start >= 0:
                cmds[frag[start:start + 200]] += 1
        if cmds and max(cmds.values()) >= 3:
            return "setup_loop"
    except Exception:  # noqa: BLE001
        pass
    return "agent_gave_up"


# ---------------------------------------------------------------------------
# Eval dispatch: mock (no docker) | offload (alienware x86) | local | skip.
# All paths emit the flywheel eval_report.json / normalized_eval.json /
# predictions.jsonl schema so `_aggregate` is identical to the flywheel.
# ---------------------------------------------------------------------------
def _emit_report(output_dir: Path, *, instance_id: str, dataset_name: str, model_name: str,
                 patch_path: Path, predictions_path: Path, verdict: str, passed: bool,
                 failure_mode: str | None, harness_exit_code: int, elapsed_s: float,
                 error: str | None, extra: dict[str, Any] | None = None) -> None:
    import platform
    report = {
        "track": "swe_bench", "instance_id": instance_id, "model_id": model_name,
        "dataset_name": dataset_name, "patch_path": str(patch_path),
        "prediction_path": str(predictions_path), "verdict": verdict, "passed": passed,
        "failure_mode": failure_mode, "harness_exit_code": harness_exit_code,
        "eval_wall_clock_seconds": round(elapsed_s, 3), "error": error,
    }
    if extra:
        report.update(extra)
    (output_dir / "eval_report.json").write_text(json.dumps(report, indent=2))
    (output_dir / "normalized_eval.json").write_text(json.dumps({
        "track": "swe_bench", "instance_id": instance_id, "outcome": verdict,
        "failure_mode": failure_mode, "dataset_name": dataset_name, "model_id": model_name,
        "eval_wall_clock_seconds": round(elapsed_s, 3), "arch": platform.machine().lower(),
    }, indent=2))


def _write_predictions(predictions_path: Path, *, instance_id: str, patch_text: str,
                       model_name: str) -> None:
    predictions_path.write_text(json.dumps({
        "instance_id": instance_id, "model_name_or_path": model_name, "model_patch": patch_text,
    }) + "\n")


def _changed_lines(p: str) -> set[str]:
    """The multiset (as a set) of added/removed *content* lines of a diff,
    excluding file headers (+++/---) and hunk headers. Robust to hunk-offset /
    context-line drift between the stored gold and a re-extracted git diff, while
    still capturing whether the same code change was made."""
    out: set[str] = set()
    for ln in p.splitlines():
        if ln.startswith(("+++", "---")):
            continue
        if ln.startswith(("+", "-")) and ln[1:].strip():
            out.add(ln.rstrip())
    return out


def _run_eval_mock(*, instance_id: str, instance: dict, patch_text: str, patch_path: Path,
                   output_dir: Path, dataset_name: str, model_name: str) -> dict[str, Any]:
    """No-docker stand-in for the swebench harness (dry-run plumbing only).

    NOT a real score: resolved iff the extracted patch equals the dataset gold
    code patch (normalized); empty -> patch_apply_failed; else -> tests_failed.
    Clearly labelled `mock=True` so it can never be confused with a docker run."""
    started = time.monotonic()
    predictions_path = output_dir / "predictions.jsonl"
    _write_predictions(predictions_path, instance_id=instance_id, patch_text=patch_text,
                       model_name=model_name)
    if not patch_text.strip():
        verdict, passed, fmode, rc = "failed", False, "patch_apply_failed", 1
    else:
        gold = instance.get("patch") or ""
        gold_lines = _changed_lines(gold)
        got_lines = _changed_lines(patch_text)
        if gold_lines and gold_lines.issubset(got_lines):
            verdict, passed, fmode, rc = "resolved", True, "tests_passed", 0
        else:
            verdict, passed, fmode, rc = "failed", False, "tests_failed", 1
    _emit_report(output_dir, instance_id=instance_id, dataset_name=dataset_name,
                 model_name=model_name, patch_path=patch_path, predictions_path=predictions_path,
                 verdict=verdict, passed=passed, failure_mode=fmode, harness_exit_code=rc,
                 elapsed_s=time.monotonic() - started, error=None,
                 extra={"mock": True, "eval_backend": "mock_gold_compare"})
    (output_dir / "eval.log").write_text(
        f"[MOCK EVAL — NOT a docker harness run] verdict={verdict} "
        f"patch_bytes={len(patch_text)} gold_bytes={len(instance.get('patch') or '')}\n",
        encoding="utf-8")
    return {"exit_code": rc, "elapsed_s": round(time.monotonic() - started, 3), "backend": "mock"}


_EVAL_SSH_OPTS = [
    "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
    "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
    "-o", "StrictHostKeyChecking=accept-new",
]
_REMOTE_BASE = "~/swe_eval_offload"
_REMOTE_WORKER = "~/swe_eval_offload/swe_eval_x86_worker.py"
_REMOTE_VENV_PY = "~/swe_eval_offload/venv/bin/python"
_REMOTE_HF_HOME = "~/.cache/huggingface"


def _run_eval_offload(*, host: str, instance_id: str, patch_path: Path, output_dir: Path,
                      dataset_name: str, model_name: str, timeout_s: int,
                      eval_log_path: Path) -> dict[str, Any]:
    """Offload eval to a native x86_64 box over SSH (plan §C3; ported from the
    flywheel `_run_eval_remote`). The remote runs swe_eval_x86_worker.py and we
    fetch the flywheel artifact set back."""
    started = time.monotonic()
    remote_dir = f"{_REMOTE_BASE}/work/{instance_id}"

    def _ssh(argv, timeout):
        return subprocess.run(["ssh", *_EVAL_SSH_OPTS, host, argv], capture_output=True,
                              text=True, timeout=timeout)

    mk = _ssh(f"mkdir -p {remote_dir} && echo ok", 30)
    if mk.returncode != 0:
        eval_log_path.write_text(f"remote mkdir failed rc={mk.returncode}\n{mk.stderr}", encoding="utf-8")
        return {"exit_code": -1, "elapsed_s": round(time.monotonic() - started, 3), "backend": "offload"}
    up = subprocess.run(["scp", *_EVAL_SSH_OPTS, str(patch_path), f"{host}:{remote_dir}/patch.diff"],
                        capture_output=True, text=True, timeout=120)
    if up.returncode != 0:
        eval_log_path.write_text(f"scp up failed rc={up.returncode}\n{up.stderr}", encoding="utf-8")
        return {"exit_code": -1, "elapsed_s": round(time.monotonic() - started, 3), "backend": "offload"}
    remote_cmd = (
        f"cd {_REMOTE_BASE} && HF_HOME={_REMOTE_HF_HOME} {_REMOTE_VENV_PY} {_REMOTE_WORKER} "
        f"--instance-id {instance_id} --patch-path {remote_dir}/patch.diff "
        f"--output-dir {remote_dir}/out --dataset-name '{dataset_name}' "
        f"--model-name '{model_name}' --timeout-s {timeout_s} --cache-level env"
    )
    ev = _ssh(remote_cmd, timeout_s + 900)
    output_dir.mkdir(parents=True, exist_ok=True)
    with eval_log_path.open("w", encoding="utf-8") as f:
        f.write(f"[offload host={host} worker_rc={ev.returncode}]\n{ev.stdout}\n-- stderr --\n{ev.stderr}")
    for fname in ("eval_report.json", "normalized_eval.json", "eval.log", "predictions.jsonl"):
        subprocess.run(["scp", *_EVAL_SSH_OPTS, f"{host}:{remote_dir}/out/{fname}",
                        str(output_dir / fname)], capture_output=True, text=True, timeout=120)
    _ssh(f"rm -rf {remote_dir}", 30)
    return {"exit_code": ev.returncode, "elapsed_s": round(time.monotonic() - started, 3),
            "backend": "offload", "eval_host": host}


def _run_eval_local(*, instance_id: str, patch_path: Path, output_dir: Path, dataset_name: str,
                    model_name: str, timeout_s: int, eval_log_path: Path) -> dict[str, Any]:
    """In-process flywheel codex_bench_eval_swe (needs swebench + docker locally).
    On this 5090 box both are absent -> returns crash/infra_error cleanly (never
    raises), which is the honest 'eval unavailable here' signal."""
    started = time.monotonic()
    sys.path.insert(0, str(Path("/home/mark/shared/lumoFlyWheel/src")))
    try:
        from lumo_flywheel_serving import codex_bench_eval_swe as cbe
    except Exception as exc:  # noqa: BLE001
        eval_log_path.write_text(f"import codex_bench_eval_swe failed: {exc}\n", encoding="utf-8")
        _write_predictions(output_dir / "predictions.jsonl", instance_id=instance_id,
                           patch_text=patch_path.read_text() if patch_path.is_file() else "",
                           model_name=model_name)
        _emit_report(output_dir, instance_id=instance_id, dataset_name=dataset_name,
                     model_name=model_name, patch_path=patch_path,
                     predictions_path=output_dir / "predictions.jsonl", verdict="crash",
                     passed=False, failure_mode="infra_error", harness_exit_code=-1,
                     elapsed_s=time.monotonic() - started, error=f"import_error: {exc}",
                     extra={"eval_backend": "local_unavailable"})
        return {"exit_code": 2, "elapsed_s": round(time.monotonic() - started, 3), "backend": "local"}
    rc = cbe.main([
        "--instance-id", instance_id, "--patch-path", str(patch_path),
        "--output-dir", str(output_dir), "--dataset-name", dataset_name,
        "--model-name", model_name, "--timeout-s", str(timeout_s), "--cache-level", "env",
    ])
    return {"exit_code": rc, "elapsed_s": round(time.monotonic() - started, 3), "backend": "local"}


def _run_eval(*, mode: str, eval_host: str | None, instance_id: str, instance: dict,
              patch_text: str, patch_path: Path, output_dir: Path, dataset_name: str,
              model_name: str, timeout_s: int, eval_log_path: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if mode == "skip":
        _write_predictions(output_dir / "predictions.jsonl", instance_id=instance_id,
                           patch_text=patch_text, model_name=model_name)
        _emit_report(output_dir, instance_id=instance_id, dataset_name=dataset_name,
                     model_name=model_name, patch_path=patch_path,
                     predictions_path=output_dir / "predictions.jsonl",
                     verdict="skipped", passed=False, failure_mode=None, harness_exit_code=0,
                     elapsed_s=0.0, error=None, extra={"eval_backend": "skip"})
        return {"exit_code": 0, "elapsed_s": 0.0, "backend": "skip"}
    if mode == "mock":
        return _run_eval_mock(instance_id=instance_id, instance=instance, patch_text=patch_text,
                              patch_path=patch_path, output_dir=output_dir,
                              dataset_name=dataset_name, model_name=model_name)
    if mode == "offload":
        if not eval_host:
            raise SystemExit("--eval-mode offload requires --eval-host")
        return _run_eval_offload(host=eval_host, instance_id=instance_id, patch_path=patch_path,
                                 output_dir=output_dir, dataset_name=dataset_name,
                                 model_name=model_name, timeout_s=timeout_s,
                                 eval_log_path=eval_log_path)
    if mode == "local":
        return _run_eval_local(instance_id=instance_id, patch_path=patch_path,
                               output_dir=output_dir, dataset_name=dataset_name,
                               model_name=model_name, timeout_s=timeout_s,
                               eval_log_path=eval_log_path)
    raise SystemExit(f"unknown eval mode: {mode}")


# ---------------------------------------------------------------------------
# Per-instance orchestration.
# ---------------------------------------------------------------------------
def _process_one(*, instance_id: str, instance: dict, dataset_name: str, per_task_root: Path,
                 repo_cache_root: Path, proxy_base_url: str | None, model: str, model_name: str,
                 agent: str, agent_wall_s: int, eval_mode: str, eval_host: str | None,
                 eval_timeout_s: int, skip_existing: bool, args) -> dict[str, Any]:
    task_dir = (per_task_root / instance_id).resolve()
    task_dir.mkdir(parents=True, exist_ok=True)
    runner_meta_path = task_dir / "runner_metadata.json"
    if skip_existing and runner_meta_path.is_file():
        return {"instance_id": instance_id, "status": "skipped_existing"}

    workspace_path = task_dir / "workspace"
    patch_path = task_dir / "patch.diff"
    qwen_trace = task_dir / "qwen_trace.json"
    qwen_stderr = task_dir / "qwen_stderr.log"
    prompt_md = task_dir / "prompt.md"
    eval_log = task_dir / "eval_invocation.log"
    eval_output = task_dir / "eval"
    eval_output.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "instance_id": instance_id, "dataset_name": dataset_name, "started_at": _iso_now(),
        "repo": instance.get("repo"), "base_commit": instance.get("base_commit"),
        "agent": agent, "eval_mode": eval_mode,
    }

    cache_path = None
    try:
        cache_path = _ensure_repo_cache(instance["repo"], repo_cache_root)
        _fetch_commit(cache_path, instance["base_commit"])
        _hydrate_workspace(cache_path=cache_path, base_commit=instance["base_commit"],
                           workspace_path=workspace_path)
        _write_agents_md(workspace_path, instance)
    except Exception as exc:  # noqa: BLE001
        summary["status"] = "hydration_failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["traceback"] = traceback.format_exc()
        runner_meta_path.write_text(json.dumps(summary, indent=2))
        if cache_path is not None:
            _remove_workspace(cache_path, workspace_path)
        return summary

    prompt_md.write_text(
        "## Qwen Code invocation prompt\n\n" + DEFAULT_AGENT_PROMPT + "\n\n"
        f"## AGENTS.md (workspace/{instance_id})\n\n"
        + (workspace_path / "AGENTS.md").read_text(encoding="utf-8"), encoding="utf-8")

    # --- agent episode -----------------------------------------------------
    if agent == "mock":
        agent_meta = _run_mock_agent(workspace=workspace_path, instance=instance,
                                     trace_path=qwen_trace)
    else:
        agent_meta = _run_qwen_code(workspace=workspace_path, proxy_base_url=proxy_base_url,
                                    model=model, timeout_s=agent_wall_s, instance_id=instance_id,
                                    args=args, trace_path=qwen_trace, stderr_path=qwen_stderr)
    summary["qwen"] = agent_meta

    # --- patch extraction --------------------------------------------------
    patch_text = ""
    try:
        patch_text = _extract_patch(workspace_path, instance["base_commit"])
    except Exception as exc:  # noqa: BLE001
        summary["patch_extract_error"] = f"{type(exc).__name__}: {exc}"

    # --- optional state-conditional empty-patch retry (default 0) ----------
    if not patch_text.strip() and agent != "mock":
        cause = _classify_empty_patch_cause(qwen_trace)
        max_retries = max(0, int(os.environ.get("SWE_EMPTY_PATCH_RETRIES", "0")))
        summary["empty_patch_retry"] = {"cause": cause, "max_retries": max_retries,
                                        "recovered_patch_bytes": 0}
        for ridx in range(1, max_retries + 1):
            retry_prompt = RETRY_PROMPT_SETUP_LOOP if cause == "setup_loop" else RETRY_PROMPT_EMPTY
            rtrace = task_dir / f"qwen_trace_retry{ridx}.json"
            rstderr = task_dir / f"qwen_stderr_retry{ridx}.log"
            rmeta = _run_qwen_code(workspace=workspace_path, proxy_base_url=proxy_base_url,
                                   model=model, timeout_s=agent_wall_s, instance_id=instance_id,
                                   args=args, trace_path=rtrace, stderr_path=rstderr,
                                   prompt=retry_prompt)
            summary[f"qwen_retry{ridx}"] = rmeta
            try:
                rp = _extract_patch(workspace_path, instance["base_commit"])
            except Exception:  # noqa: BLE001
                rp = ""
            if rp.strip():
                patch_text = rp
                summary["empty_patch_retry"]["recovered_patch_bytes"] = len(rp)
                break

    patch_path.write_text(patch_text, encoding="utf-8")
    summary["patch_bytes"] = len(patch_text)

    # --- eval (verdict source of truth; NOT the qwen CLI exit — R4) --------
    eval_meta = _run_eval(mode=eval_mode, eval_host=eval_host, instance_id=instance_id,
                          instance=instance, patch_text=patch_text, patch_path=patch_path,
                          output_dir=eval_output, dataset_name=dataset_name, model_name=model_name,
                          timeout_s=eval_timeout_s, eval_log_path=eval_log)
    summary["eval"] = eval_meta
    report_path = eval_output / "eval_report.json"
    if report_path.is_file():
        try:
            summary["eval_report"] = json.loads(report_path.read_text())
        except Exception:  # noqa: BLE001
            pass

    _remove_workspace(cache_path, workspace_path)
    summary["ended_at"] = _iso_now()
    runner_meta_path.write_text(json.dumps(summary, indent=2))
    return summary


def _process_one_container(*, instance_id: str, instance: dict, dataset_name: str,
                           per_task_root: Path, proxy_base_url: str | None, model: str,
                           model_name: str, agent: str, agent_wall_s: int, eval_mode: str,
                           eval_host: str | None, eval_timeout_s: int, skip_existing: bool,
                           args) -> dict[str, Any]:
    """CONTAINER runtime (RUNTIME_ALIGNMENT_DIRECTIVE): episode runs inside the
    official per-instance swebench image. Same artifact set as `_process_one`;
    the workspace is seeded from the image /testbed and bind-mounted into a
    long-lived container, the agent's shell routes through `docker exec`, and the
    patch is the in-container tracked diff vs base_commit."""
    task_dir = (per_task_root / instance_id).resolve()
    task_dir.mkdir(parents=True, exist_ok=True)
    runner_meta_path = task_dir / "runner_metadata.json"
    if skip_existing and runner_meta_path.is_file():
        return {"instance_id": instance_id, "status": "skipped_existing"}

    workspace_path = task_dir / "workspace"
    patch_path = task_dir / "patch.diff"
    qwen_trace = task_dir / "qwen_trace.json"
    qwen_stderr = task_dir / "qwen_stderr.log"
    prompt_md = task_dir / "prompt.md"
    eval_log = task_dir / "eval_invocation.log"
    eval_output = task_dir / "eval"
    eval_output.mkdir(parents=True, exist_ok=True)
    shim_dir = task_dir / "shim"

    base_commit = instance["base_commit"]
    image = _container_image_for(instance_id)
    container = f"{args.container_name_prefix}_{instance_id.replace('__', '_').replace('/', '_')}"

    summary: dict[str, Any] = {
        "instance_id": instance_id, "dataset_name": dataset_name, "started_at": _iso_now(),
        "repo": instance.get("repo"), "base_commit": base_commit, "agent": agent,
        "eval_mode": eval_mode, "runtime": "container", "image": image, "container": container,
    }

    try:
        if not _image_present(image):
            raise RuntimeError(f"official image not present locally: {image} "
                               f"(docker pull {image})")
        _seed_workspace_from_image(image=image, workspace=workspace_path)
        _start_container(name=container, image=image, workspace=workspace_path)
        _write_agents_md(workspace_path, instance)
    except Exception as exc:  # noqa: BLE001
        summary["status"] = "hydration_failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["traceback"] = traceback.format_exc()
        runner_meta_path.write_text(json.dumps(summary, indent=2))
        _teardown_container(container, workspace_path if not args.container_keep else None)
        return summary

    prompt_md.write_text(
        "## Qwen Code invocation prompt (container runtime)\n\n" + DEFAULT_AGENT_PROMPT + "\n\n"
        f"## AGENTS.md (workspace/{instance_id})\n\n"
        + (workspace_path / "AGENTS.md").read_text(encoding="utf-8"), encoding="utf-8")

    def _extract() -> str:
        try:
            return _container_extract_patch(container, base_commit)
        except Exception:  # noqa: BLE001
            return ""

    try:
        # --- agent episode (edits land in the container via mount + docker exec)
        if agent == "mock":
            agent_meta = _run_mock_agent_container(container=container, instance=instance,
                                                   base_commit=base_commit,
                                                   workspace=workspace_path, trace_path=qwen_trace)
        else:
            _write_shell_shim(shim_dir=shim_dir, container=container)
            agent_meta = _run_qwen_code(workspace=workspace_path, proxy_base_url=proxy_base_url,
                                        model=model, timeout_s=agent_wall_s, instance_id=instance_id,
                                        args=args, trace_path=qwen_trace, stderr_path=qwen_stderr,
                                        extra_path=shim_dir)
        summary["qwen"] = agent_meta

        patch_text = _extract()

        # --- optional state-conditional empty-patch retry (default 0) ----------
        if not patch_text.strip() and agent != "mock":
            cause = _classify_empty_patch_cause(qwen_trace)
            max_retries = max(0, int(os.environ.get("SWE_EMPTY_PATCH_RETRIES", "0")))
            summary["empty_patch_retry"] = {"cause": cause, "max_retries": max_retries,
                                            "recovered_patch_bytes": 0}
            for ridx in range(1, max_retries + 1):
                retry_prompt = RETRY_PROMPT_SETUP_LOOP if cause == "setup_loop" else RETRY_PROMPT_EMPTY
                rmeta = _run_qwen_code(workspace=workspace_path, proxy_base_url=proxy_base_url,
                                       model=model, timeout_s=agent_wall_s, instance_id=instance_id,
                                       args=args, trace_path=task_dir / f"qwen_trace_retry{ridx}.json",
                                       stderr_path=task_dir / f"qwen_stderr_retry{ridx}.log",
                                       prompt=retry_prompt, extra_path=shim_dir)
                summary[f"qwen_retry{ridx}"] = rmeta
                rp = _extract()
                if rp.strip():
                    patch_text = rp
                    summary["empty_patch_retry"]["recovered_patch_bytes"] = len(rp)
                    break

        patch_path.write_text(patch_text, encoding="utf-8")
        summary["patch_bytes"] = len(patch_text)

        eval_meta = _run_eval(mode=eval_mode, eval_host=eval_host, instance_id=instance_id,
                              instance=instance, patch_text=patch_text, patch_path=patch_path,
                              output_dir=eval_output, dataset_name=dataset_name,
                              model_name=model_name, timeout_s=eval_timeout_s, eval_log_path=eval_log)
        summary["eval"] = eval_meta
        report_path = eval_output / "eval_report.json"
        if report_path.is_file():
            try:
                summary["eval_report"] = json.loads(report_path.read_text())
            except Exception:  # noqa: BLE001
                pass
    finally:
        if not args.container_keep:
            _teardown_container(container, workspace_path)

    summary["ended_at"] = _iso_now()
    runner_meta_path.write_text(json.dumps(summary, indent=2))
    return summary


def _aggregate(per_task_root: Path, summary_path: Path, predictions_path: Path,
               started_at: str, ended_at: str, model_name: str) -> dict[str, Any]:
    """Ported from the flywheel: verdict_counts / resolved_rate / per-repo /
    wall percentiles, + predictions.jsonl concatenation."""
    instance_summaries: list[dict] = []
    verdict_counter: Counter = Counter()
    failure_counter: Counter = Counter()
    repo_counter: Counter = Counter()
    repo_pass_counter: Counter = Counter()
    agent_wall: list[float] = []
    predictions_lines: list[str] = []
    for task_dir in sorted(p for p in per_task_root.iterdir() if p.is_dir()):
        meta_path = task_dir / "runner_metadata.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text())
        instance_summaries.append(meta)
        verdict = (meta.get("eval_report") or {}).get("verdict", "missing")
        failure = (meta.get("eval_report") or {}).get("failure_mode", "missing")
        verdict_counter[verdict] += 1
        failure_counter[failure] += 1
        repo = meta.get("repo") or "unknown"
        repo_counter[repo] += 1
        if verdict == "resolved":
            repo_pass_counter[repo] += 1
        if (meta.get("qwen") or {}).get("elapsed_s") is not None:
            agent_wall.append(float(meta["qwen"]["elapsed_s"]))
        pred_file = task_dir / "eval" / "predictions.jsonl"
        if pred_file.is_file():
            predictions_lines.extend(l for l in pred_file.read_text().splitlines() if l.strip())

    def _pcts(xs):
        if not xs:
            return {}
        xs = sorted(xs)
        def _p(p):
            i = max(0, min(len(xs) - 1, int(round(p * (len(xs) - 1)))))
            return round(xs[i], 3)
        return {"p50": _p(0.5), "p90": _p(0.9), "min": round(min(xs), 3), "max": round(max(xs), 3)}

    summary = {
        "model_name_or_path": model_name, "started_at": started_at, "ended_at": ended_at,
        "instances_total": len(instance_summaries), "verdict_counts": dict(verdict_counter),
        "failure_mode_counts": dict(failure_counter), "per_repo_total": dict(repo_counter),
        "per_repo_resolved": dict(repo_pass_counter), "agent_wall_seconds": _pcts(agent_wall),
        "resolved_rate": (round(verdict_counter["resolved"] / len(instance_summaries), 4)
                          if instance_summaries else None),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    predictions_path.write_text("\n".join(predictions_lines) + ("\n" if predictions_lines else ""))
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subset", type=Path, required=True,
                   help="JSON subset from build_swe_bench_subset.py (dataset_name + instance_ids)")
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--dataset-tag", default=None)
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="vLLM /v1 endpoint (proxy upstream)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="served-model-name at --endpoint")
    p.add_argument("--model-name", default=DEFAULT_MODEL_NAME_TAG,
                   help="tag recorded in predictions.jsonl.model_name_or_path")
    p.add_argument("--agent", choices=("qwen_code", "mock"), default="qwen_code")
    p.add_argument("--agent-wall-s", type=int,
                   default=int(os.environ.get("SWE_AGENT_WALL_S", str(DEFAULT_AGENT_WALL_S))),
                   help="hard harness wall per attempt (0 = rely on qwen --max-wall-time)")
    p.add_argument("--max-session-turns", type=int, default=DEFAULT_MAX_SESSION_TURNS)
    p.add_argument("--qwen-max-wall", default=DEFAULT_QWEN_MAX_WALL)
    p.add_argument("--qwen-max-output-tokens", type=int, default=DEFAULT_QWEN_MAX_OUTPUT_TOKENS)
    p.add_argument("--system-prompt", default="")
    p.add_argument("--eval-mode", choices=("mock", "offload", "local", "skip"), default="mock")
    p.add_argument("--eval-host", default=None, help="SSH host for --eval-mode offload (x86)")
    p.add_argument("--eval-timeout-s", type=int, default=DEFAULT_EVAL_TIMEOUT_S)
    p.add_argument("--repo-cache", type=Path, default=DEFAULT_REPO_CACHE)
    p.add_argument("--limit", type=int, default=None, help="process only first N instances")
    p.add_argument("--only", default=None, help="comma-sep instance_ids to run (subset filter)")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--qwen-bin", type=Path, default=DEFAULT_QWEN_BIN)
    p.add_argument("--proxy-script", type=Path, default=DEFAULT_PROXY_SCRIPT)
    p.add_argument("--proxy-host", default=DEFAULT_PROXY_HOST)
    p.add_argument("--proxy-port", type=int, default=DEFAULT_PROXY_PORT)
    p.add_argument("--proxy-max-tokens", type=int, default=DEFAULT_PROXY_MAX_TOKENS)
    p.add_argument("--proxy-tool-choice", default="", help="'' = natural (default); 'required' forces")
    p.add_argument("--proxy-dump-dir", type=Path, default=None)
    p.add_argument("--runtime", choices=("host", "container"), default="host",
                   help="host = git-worktree bare checkout (legacy, deps absent); "
                        "container = official per-instance swebench image with the "
                        "prepared conda env (RUNTIME_ALIGNMENT_DIRECTIVE)")
    p.add_argument("--container-name-prefix", default="swe_ep",
                   help="prefix for the per-instance docker container name")
    p.add_argument("--container-keep", action="store_true",
                   help="keep the container + seeded workspace after each instance (debug)")
    args = p.parse_args(argv)

    dataset_name, instance_ids = _load_subset(args.subset)
    if args.only:
        want = {x.strip() for x in args.only.split(",") if x.strip()}
        instance_ids = [i for i in instance_ids if i in want]
    if args.limit is not None:
        instance_ids = instance_ids[: args.limit]
    dataset_tag = args.dataset_tag or ("pro" if "Pro" in dataset_name else "verified")
    dataset_out = args.out_root / dataset_tag
    per_task_root = dataset_out / "per_task"
    per_task_root.mkdir(parents=True, exist_ok=True)
    args.repo_cache.mkdir(parents=True, exist_ok=True)

    print(f"=== [{_iso_now()}] dataset={dataset_name} tag={dataset_tag} n={len(instance_ids)} "
          f"agent={args.agent} eval={args.eval_mode} endpoint={args.endpoint} ===", flush=True)

    # Load instance metadata (repo/base_commit/problem_statement/patch) from HF.
    dataset_records = _load_dataset(dataset_name)
    missing = [i for i in instance_ids if i not in dataset_records]
    if missing:
        print(f"WARNING: {len(missing)} subset instances missing from dataset: {missing[:5]}", flush=True)
        instance_ids = [i for i in instance_ids if i in dataset_records]

    # Start the qwen-code proxy once for the campaign (skip for the mock agent).
    proxy_proc = proxy_log = None
    proxy_base_url = None
    if args.agent == "qwen_code":
        if not args.qwen_bin.exists():
            raise SystemExit(f"missing qwen binary: {args.qwen_bin}")
        ok, detail = _check_upstream(args.endpoint, 5)
        if not ok:
            raise SystemExit(f"upstream not ready at {args.endpoint}: {detail}")
        proxy_base_url = f"http://{args.proxy_host}:{args.proxy_port}/v1"
        proxy_proc, proxy_log = _start_proxy(args, dataset_out / "proxy.log")
        print(f"[proxy] {proxy_base_url} -> {args.endpoint}", flush=True)

    started_at = _iso_now()
    try:
        for iid in instance_ids:
            t0 = time.time()
            print(f"[{_iso_now()}] -> {iid}", flush=True)
            try:
                if args.runtime == "container":
                    res = _process_one_container(
                        instance_id=iid, instance=dataset_records[iid],
                        dataset_name=dataset_name, per_task_root=per_task_root,
                        proxy_base_url=proxy_base_url, model=args.model,
                        model_name=args.model_name, agent=args.agent,
                        agent_wall_s=args.agent_wall_s, eval_mode=args.eval_mode,
                        eval_host=args.eval_host, eval_timeout_s=args.eval_timeout_s,
                        skip_existing=args.skip_existing, args=args)
                else:
                    res = _process_one(instance_id=iid, instance=dataset_records[iid],
                                       dataset_name=dataset_name, per_task_root=per_task_root,
                                       repo_cache_root=args.repo_cache, proxy_base_url=proxy_base_url,
                                       model=args.model, model_name=args.model_name, agent=args.agent,
                                       agent_wall_s=args.agent_wall_s, eval_mode=args.eval_mode,
                                       eval_host=args.eval_host, eval_timeout_s=args.eval_timeout_s,
                                       skip_existing=args.skip_existing, args=args)
            except Exception as exc:  # noqa: BLE001
                res = {"instance_id": iid, "status": "orchestrator_crash",
                       "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
            verdict = (res.get("eval_report") or {}).get("verdict", res.get("status", "?"))
            print(f"[{_iso_now()}] <- {iid} verdict={verdict} "
                  f"patch_bytes={res.get('patch_bytes')} elapsed={time.time()-t0:.1f}s", flush=True)
    finally:
        _stop_proxy(proxy_proc, proxy_log)

    ended_at = _iso_now()
    summary = _aggregate(per_task_root, dataset_out / "campaign_summary.json",
                         dataset_out / "predictions.jsonl", started_at, ended_at, args.model_name)
    print(f"=== [{ended_at}] DONE n={summary['instances_total']} "
          f"resolved_rate={summary.get('resolved_rate')} verdicts={summary['verdict_counts']} ===",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
