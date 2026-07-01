#!/usr/bin/env python3
"""Smoke-sized tau2 mock-domain agentic eval for FLARE vs AR baselines."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from eval_fastdllm_toolcall_cases import (
    full_context_sample,
    load_model as load_diffusion_model,
    resolve_token_ids,
    sequence_preserving_constrained_tool_call_text,
)
from eval_toolcall_jsonl import (
    extract_tool_call_objects,
    extract_tool_calls,
    qwen_native_tool_call_text,
    schema_errors,
    tool_schema_by_name,
)


DEFAULT_TAU2_ROOT = Path("/tmp/qwen_diffusion_external/tau2-bench")
DEFAULT_BASE = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_ADAPTER = ROOT / "runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000"
DEFAULT_TASK_IDS = ["create_task_1_with_env_assertions", "update_task_with_message_history"]
DEFAULT_OUT = ROOT / "runs/agentic_eval/tau2_mock_agentic.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau2-root", type=Path, default=DEFAULT_TAU2_ROOT)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--lanes", default="raw,constrained,protected")
    parser.add_argument("--backend", choices=["openai", "fastdllm-ar", "diffusion"], required=True)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--max-agent-turns", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--constrained-max-calls", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--served-model", default="qwen3.5-9b-ar")
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--no-merge-adapter", action="store_true")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--small-block-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--live-tool-native-grammar", action="store_true")
    parser.add_argument("--live-tool-json-topk", type=int, default=128)
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_json(obj: Any) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git_rev(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.replace(" ", ",").split(",") if item.strip()]


def function_schema(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def mock_tools() -> list[dict]:
    return [
        function_schema(
            "create_task",
            "Create a new task for a user.",
            {
                "user_id": {"type": "string", "description": "The ID of the user creating the task."},
                "title": {"type": "string", "description": "The title of the task."},
                "description": {"type": "string", "description": "Optional task description."},
            },
            ["user_id", "title"],
        ),
        function_schema(
            "get_users",
            "Get all users in the database.",
            {},
            [],
        ),
        function_schema(
            "update_task_status",
            "Update the status of a task.",
            {
                "task_id": {"type": "string", "description": "The ID of the task to update."},
                "status": {"type": "string", "enum": ["pending", "completed"], "description": "The new status."},
            },
            ["task_id", "status"],
        ),
        function_schema(
            "transfer_to_human_agents",
            "Transfer the user to a human agent when policy or available tools cannot solve the issue.",
            {"summary": {"type": "string", "description": "A summary of the user's issue."}},
            ["summary"],
        ),
    ]


def system_prompt(policy: str) -> str:
    return (
        "You are a customer service agent operating the public tau2 mock task-management environment. "
        "Use the available tools when a database read or write is required. In each assistant turn, either "
        "send a message to the user or make tool call(s), not both. Follow this policy exactly:\n\n"
        f"{policy.strip()}"
    )


def content_or_json(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def normalize_history_tool_call(item: dict) -> dict:
    return {
        "name": item.get("name") or item.get("function", {}).get("name"),
        "arguments": item.get("arguments") or item.get("function", {}).get("arguments") or {},
    }


def tool_response_message(payload: Any) -> dict:
    return {"role": "user", "content": "<tool_response>\n" + content_or_json(payload) + "\n</tool_response>"}


def task_user_message(task: dict) -> str:
    if task.get("ticket"):
        return str(task["ticket"])
    scenario = task.get("user_scenario") or {}
    return str(scenario.get("instructions") or task.get("description") or task.get("id"))


def build_initial_messages(policy: str, task: dict) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt(policy)}]
    for item in ((task.get("initial_state") or {}).get("message_history") or []):
        role = item.get("role")
        if role == "assistant" and item.get("tool_calls"):
            calls = [normalize_history_tool_call(call) for call in item.get("tool_calls") or []]
            messages.append({"role": "assistant", "content": qwen_native_tool_call_text(calls)})
        elif role == "tool":
            messages.append(tool_response_message(item.get("content") or ""))
        elif role in {"system", "user", "assistant"}:
            messages.append({"role": role, "content": content_or_json(item.get("content"))})
    messages.append({"role": "user", "content": task_user_message(task)})
    return messages


def conversation_text(messages: list[dict]) -> str:
    lines = []
    for message in messages:
        lines.append(f"{message.get('role')}: {content_or_json(message.get('content'))}")
    return "\n".join(lines)


def render_local_prompt(tokenizer, messages: list[dict], tools: list[dict]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


@dataclass
class Generation:
    text: str
    seconds: float
    tokens: int = 0
    backend_meta: dict[str, Any] | None = None


class OpenAIBackend:
    def __init__(self, args: argparse.Namespace):
        self.endpoint = args.endpoint.rstrip("/")
        self.model = args.served_model
        self.timeout = args.timeout
        self.max_new_tokens = args.max_new_tokens
        self.temperature = args.temperature
        self.seed = args.seed

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "openai",
            "endpoint": self.endpoint,
            "model": self.model,
            "dtype": "bf16",
            "quant": "none",
        }

    def complete(self, messages: list[dict], tools: list[dict]) -> Generation:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "seed": self.seed,
        }
        start = time.time()
        response = post_json(self.endpoint + "/chat/completions", payload, self.timeout)
        elapsed = time.time() - start
        message = response["choices"][0]["message"]
        text = content_or_json(message.get("content"))
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            native_text = qwen_native_tool_call_text(tool_calls)
            if native_text and native_text not in text:
                text = (text + "\n" + native_text).strip()
        usage = response.get("usage") or {}
        return Generation(
            text=text.strip(),
            seconds=elapsed,
            tokens=int(usage.get("completion_tokens") or 0),
            backend_meta={"finish_reason": response["choices"][0].get("finish_reason"), "usage": usage},
        )


class FastDllmARBackend:
    def __init__(self, args: argparse.Namespace):
        from eval_fastdllm_ar_toolcall_cases import greedy_ar_generate, load_model as load_ar_model

        self.greedy_ar_generate = greedy_ar_generate
        load_args = SimpleNamespace(
            base_model=args.base_model,
            adapter=args.adapter if args.adapter and args.adapter.exists() else None,
            tokenizer_path=args.tokenizer_path,
            no_merge_adapter=args.no_merge_adapter,
        )
        self.model, self.tokenizer = load_ar_model(load_args)
        self.mask_id, self.stop_token_id, self.stop_token_ids = resolve_token_ids(self.model, self.tokenizer)
        self.gen_args = SimpleNamespace(
            max_new_tokens=args.max_new_tokens,
            stop_token_ids=self.stop_token_ids,
            ban_mask_token=True,
            mask_id=self.mask_id,
            live_tool_native_grammar=args.live_tool_native_grammar,
            live_tool_json_topk=args.live_tool_json_topk,
        )
        self.base_model = args.base_model
        self.adapter = args.adapter
        self.merge_adapter = not args.no_merge_adapter

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "fastdllm-ar",
            "base_model": str(self.base_model),
            "adapter": str(self.adapter) if self.adapter else None,
            "merge_adapter": self.merge_adapter,
            "dtype": "bf16",
            "quant": "none",
            "mask_id": int(self.mask_id),
            "stop_token_ids": [int(item) for item in self.stop_token_ids],
            "live_tool_native_grammar": bool(self.gen_args.live_tool_native_grammar),
        }

    def complete(self, messages: list[dict], tools: list[dict]) -> Generation:
        import torch

        prompt = render_local_prompt(self.tokenizer, messages, tools)
        prompt_input_ids = self.tokenizer([prompt], return_tensors="pt").input_ids.to("cuda")
        start = time.time()
        generated, events = self.greedy_ar_generate(
            self.model,
            self.tokenizer,
            prompt_input_ids,
            prompt_input_ids,
            {"tools": tools},
            self.gen_args,
        )
        torch.cuda.synchronize()
        elapsed = time.time() - start
        new_ids = generated[prompt_input_ids.shape[1] :]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        return Generation(text=text, seconds=elapsed, tokens=int(new_ids.shape[0]), backend_meta={"grammar_events": events})


class DiffusionBackend:
    def __init__(self, args: argparse.Namespace):
        import torch

        os.environ["FASTDLLM_FLARE_GDN_ROUTE"] = "route_i"
        os.environ.setdefault("FASTDLLM_FLARE_TWO_STREAM", "1")
        os.environ.setdefault("FLARE_TWO_STREAM", "1")
        self.model, self.tokenizer = load_diffusion_model(
            args.base_model,
            args.adapter if args.adapter and args.adapter.exists() else None,
            merge_adapter=not args.no_merge_adapter,
            tokenizer_path=args.tokenizer_path,
        )
        self.model.eval()
        self.mask_id, self.stop_token_id, self.stop_token_ids = resolve_token_ids(self.model, self.tokenizer)
        self.gen_args = SimpleNamespace(
            block_size=args.block_size,
            small_block_size=args.small_block_size,
            max_new_tokens=args.max_new_tokens,
            threshold=args.threshold,
            temperature=args.temperature,
            top_p=args.top_p,
            mask_id=self.mask_id,
            stop_token_id=self.stop_token_id,
            stop_token_ids=self.stop_token_ids,
            use_block_cache=True,
            full_context_sampling=True,
            fresh_generation_blocks=False,
            denoise_logit_mode="flare_shift",
            force_argument_boundary_target_tokens=False,
            constrain_argument_candidate_tokens=False,
            force_selected_candidate_tokens=False,
            force_best_candidate_sequence=False,
            guard_tool_value_candidates=False,
            force_best_tool_name_sequence=False,
            guard_tool_name_candidates=False,
            ban_argument_boundary_tokens=False,
            ban_argument_json_boundary_tokens=False,
            ban_argument_newline_tokens=False,
            guard_tool_call_mode=False,
            guard_tool_json_prefix=False,
            json_prefix_guard_kinds=set(),
            json_prefix_guard_topk=32,
            json_prefix_guard_left_to_right=True,
            json_prefix_guard_target_fallback=False,
            live_tool_json_grammar=args.live_tool_native_grammar,
            live_tool_json_topk=args.live_tool_json_topk,
            force_schedule_token_kinds=set(),
            argument_boundary_token_ids=[],
            argument_newline_token_ids=[],
            _argument_boundary_target_cache={},
        )
        if hasattr(self.model, "config"):
            setattr(self.model.config, "bd_size", int(args.block_size))
        self.base_model = args.base_model
        self.adapter = args.adapter
        self.merge_adapter = not args.no_merge_adapter
        torch.cuda.reset_peak_memory_stats()

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "diffusion",
            "base_model": str(self.base_model),
            "adapter": str(self.adapter) if self.adapter else None,
            "merge_adapter": self.merge_adapter,
            "dtype": "bf16",
            "quant": "none",
            "block_size": int(self.gen_args.block_size),
            "small_block_size": int(self.gen_args.small_block_size),
            "denoise_logit_mode": self.gen_args.denoise_logit_mode,
            "use_block_cache": bool(self.gen_args.use_block_cache),
            "mask_id": int(self.mask_id),
            "stop_token_ids": [int(item) for item in self.stop_token_ids],
            "live_tool_native_grammar": bool(self.gen_args.live_tool_json_grammar),
        }

    def complete(self, messages: list[dict], tools: list[dict]) -> Generation:
        import torch

        prompt = render_local_prompt(self.tokenizer, messages, tools)
        prompt_input_ids = self.tokenizer([prompt], return_tensors="pt").input_ids.to("cuda")
        previous_schemas = getattr(self.gen_args, "_live_tool_schemas", None)
        self.gen_args._live_tool_schemas = tool_schema_by_name(tools)
        start = time.time()
        try:
            generated = full_context_sample(
                self.model,
                prompt_input_ids,
                self.tokenizer,
                self.gen_args,
                sampler_schedule=None,
                original_len_override=prompt_input_ids.shape[1],
            )
            torch.cuda.synchronize()
        finally:
            if previous_schemas is None:
                try:
                    delattr(self.gen_args, "_live_tool_schemas")
                except AttributeError:
                    pass
            else:
                self.gen_args._live_tool_schemas = previous_schemas
        elapsed = time.time() - start
        new_ids = generated[prompt_input_ids.shape[1] :]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        meta = {
            "sampler_schedule_events": getattr(self.gen_args, "_last_sampler_schedule_events", {}),
            "flare_cache_stats": getattr(self.gen_args, "_last_flare_cache_stats", {}),
        }
        return Generation(text=text, seconds=elapsed, tokens=int(new_ids.shape[0]), backend_meta=meta)


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


class MockEnvironment:
    def __init__(self, base_db: dict, task: dict):
        self.db = copy.deepcopy(base_db)
        self.tool_errors: list[str] = []
        self.blocked_calls: list[dict] = []
        self.apply_initial_state(task.get("initial_state") or {})

    def apply_initial_state(self, initial_state: dict) -> None:
        data = (initial_state.get("initialization_data") or {}).get("agent_data") or {}
        for task_id, task in (data.get("tasks") or {}).items():
            self.db.setdefault("tasks", {})[task_id] = {
                "task_id": task_id,
                "title": task.get("title") or task_id,
                "description": task.get("description"),
                "status": task.get("status") or "pending",
            }
        for user_id, user in (data.get("users") or {}).items():
            existing = self.db.setdefault("users", {}).setdefault(user_id, {"user_id": user_id, "tasks": []})
            for task_id in user.get("tasks") or []:
                if task_id not in existing.setdefault("tasks", []):
                    existing["tasks"].append(task_id)
        for action in initial_state.get("initialization_actions") or []:
            self.execute(action.get("func_name") or action.get("name"), action.get("arguments") or {})
        for message in initial_state.get("message_history") or []:
            if message.get("role") == "tool":
                self.absorb_tool_result(message.get("content"))

    def absorb_tool_result(self, content: Any) -> None:
        if isinstance(content, str):
            try:
                payload = json.loads(content)
            except Exception:
                return
        elif isinstance(content, dict):
            payload = content
        else:
            return
        task_id = payload.get("task_id")
        if not task_id:
            return
        task = {
            "task_id": task_id,
            "title": payload.get("title") or task_id,
            "description": payload.get("description"),
            "status": payload.get("status") or "pending",
        }
        self.db.setdefault("tasks", {})[task_id] = task
        user = self.db.setdefault("users", {}).setdefault("user_1", {"user_id": "user_1", "name": "Test User", "tasks": []})
        if task_id not in user.setdefault("tasks", []):
            user["tasks"].append(task_id)

    def guard_call(self, name: str, arguments: dict, tools: list[dict]) -> tuple[bool, str]:
        schemas = tool_schema_by_name(tools)
        schema = schemas.get(name)
        if schema is None:
            return False, "unknown_tool"
        errors = schema_errors(arguments, schema)
        if errors:
            return False, "schema_error:" + ";".join(errors)
        if name == "create_task":
            if arguments.get("user_id") not in self.db.get("users", {}):
                return False, "unknown_user"
            if not str(arguments.get("title") or "").strip():
                return False, "missing_title"
        if name == "update_task_status":
            if arguments.get("task_id") not in self.db.get("tasks", {}):
                return False, "unknown_task"
            if arguments.get("status") not in {"pending", "completed"}:
                return False, "bad_status"
        return True, "ok"

    def execute(self, name: str, arguments: dict) -> dict:
        arguments = arguments or {}
        try:
            if name == "create_task":
                user_id = str(arguments.get("user_id"))
                if user_id not in self.db.get("users", {}):
                    raise ValueError(f"User {user_id} not found")
                task_id = f"task_{len(self.db.setdefault('tasks', {})) + 1}"
                task = {
                    "task_id": task_id,
                    "title": arguments.get("title"),
                    "description": arguments.get("description"),
                    "status": "pending",
                }
                self.db["tasks"][task_id] = task
                self.db["users"][user_id].setdefault("tasks", []).append(task_id)
                return task
            if name == "get_users":
                return {"users": list(self.db.get("users", {}).values())}
            if name == "update_task_status":
                task_id = str(arguments.get("task_id"))
                if task_id not in self.db.get("tasks", {}):
                    raise ValueError(f"Task {task_id} not found")
                status = str(arguments.get("status"))
                if status not in {"pending", "completed"}:
                    raise ValueError(f"Invalid status {status}")
                self.db["tasks"][task_id]["status"] = status
                return self.db["tasks"][task_id]
            if name == "transfer_to_human_agents":
                return {"message": "Transfer successful"}
            raise ValueError(f"Unknown tool {name}")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self.tool_errors.append(error)
            return {"error": error}

    def evaluate_env_assertion(self, assertion: dict) -> bool:
        name = assertion.get("func_name")
        args = assertion.get("arguments") or {}
        try:
            if name == "assert_task_status":
                task = self.db.get("tasks", {}).get(args.get("task_id"))
                return bool(task and task.get("status") == args.get("expected_status"))
            if name == "assert_number_of_tasks":
                user = self.db.get("users", {}).get(args.get("user_id"))
                return bool(user and len(user.get("tasks") or []) == int(args.get("expected_number")))
        except Exception:
            return False
        return False


def coerce_args_for_schema(call: dict, tools: list[dict]) -> dict:
    schemas = tool_schema_by_name(tools)
    schema = schemas.get(call.get("name")) or {}
    properties = schema.get("properties") or {}
    coerced = {}
    for key, value in (call.get("arguments") or {}).items():
        expected = (properties.get(key) or {}).get("type")
        if isinstance(expected, list):
            expected = next((item for item in expected if item != "null"), expected[0] if expected else None)
        if expected in {"integer", "number"} and isinstance(value, str):
            try:
                value = int(value) if expected == "integer" else float(value)
            except Exception:
                pass
        coerced[key] = value
    return coerced


def action_matches(action: dict, call: dict, tools: list[dict]) -> bool:
    if action.get("name") != call.get("name"):
        return False
    call_args = coerce_args_for_schema(call, tools)
    if action.get("compare_args") is None:
        compare_args = list(call_args.keys())
    else:
        compare_args = list(action.get("compare_args") or [])
    if not compare_args:
        return True
    tool_args = {key: call_args.get(key) for key in compare_args if key in call_args}
    action_args = {key: (action.get("arguments") or {}).get(key) for key in compare_args if key in (action.get("arguments") or {})}
    return tool_args == action_args


def evaluate_actions(task: dict, calls: list[dict], tools: list[dict]) -> tuple[bool, list[dict]]:
    expected = ((task.get("evaluation_criteria") or {}).get("actions") or [])
    checks = []
    for action in expected:
        matched = any(action_matches(action, call, tools) for call in calls)
        checks.append({"action_id": action.get("action_id"), "name": action.get("name"), "matched": matched})
    return all(item["matched"] for item in checks), checks


def evaluate_db_state(task: dict, env: MockEnvironment) -> tuple[bool, list[dict]]:
    checks = []
    for action in ((task.get("evaluation_criteria") or {}).get("actions") or []):
        args = action.get("arguments") or {}
        ok = True
        detail = ""
        if action.get("name") == "create_task":
            user_id = args.get("user_id")
            title = args.get("title")
            user = env.db.get("users", {}).get(user_id)
            task_ids = set((user or {}).get("tasks") or [])
            matching = [
                task
                for task_id, task in env.db.get("tasks", {}).items()
                if task_id in task_ids and task.get("title") == title and task.get("status") == "pending"
            ]
            ok = bool(matching)
            detail = f"matching_created_tasks={len(matching)}"
        elif action.get("name") == "update_task_status":
            task = env.db.get("tasks", {}).get(args.get("task_id"))
            ok = bool(task and task.get("status") == args.get("status"))
            detail = f"actual_status={(task or {}).get('status')}"
        checks.append({"name": action.get("name"), "ok": ok, "detail": detail})
    return all(item["ok"] for item in checks), checks


def evaluate_env_assertions(task: dict, env: MockEnvironment) -> tuple[bool, list[dict]]:
    assertions = ((task.get("evaluation_criteria") or {}).get("env_assertions") or [])
    checks = []
    for assertion in assertions:
        ok = env.evaluate_env_assertion(assertion)
        checks.append({"func_name": assertion.get("func_name"), "ok": ok, "arguments": assertion.get("arguments") or {}})
    return all(item["ok"] for item in checks), checks


def parse_calls_from_text(text: str) -> tuple[list[dict], int]:
    calls, invalid = extract_tool_calls(text or "")
    return calls, invalid


def projected_text(raw_text: str, messages: list[dict], tools: list[dict], max_calls: int) -> str:
    return sequence_preserving_constrained_tool_call_text(
        raw_text or "",
        tools,
        context_text=conversation_text(messages),
        max_calls=max_calls,
    )


def failure_taxonomy(row: dict) -> list[str]:
    failures = []
    if row.get("status") != "ok":
        return ["backend_error"]
    if row.get("turn_budget_exhausted"):
        failures.append("stop_turn_budget")
    if not row.get("effective_calls"):
        failures.append("no_tool_call")
    if row.get("invalid_tool_call_count", 0) > 0:
        failures.append("invalid_call")
    expected_names = [item.get("name") for item in row.get("expected_actions") or []]
    called_names = [item.get("name") for item in row.get("effective_calls") or []]
    if called_names and expected_names and not (set(called_names) & set(expected_names)):
        failures.append("wrong_tool")
    if called_names and expected_names and not row.get("action_reward"):
        failures.append("wrong_args")
    if row.get("tool_execution_errors"):
        failures.append("tool_execution_error")
    if row.get("protected_blocked_calls"):
        failures.append("protected_blocked_call")
    if not row.get("db_reward"):
        failures.append("db_state_failure")
    if not row.get("env_assertion_reward"):
        failures.append("env_assertion_failure")
    return failures or ["pass"]


def request_hash(messages: list[dict], tools: list[dict], settings: dict) -> str:
    return sha256_json({"messages": messages, "tools": tools, "settings": settings})


def run_episode(
    *,
    backend,
    task: dict,
    lane: str,
    policy: str,
    base_db: dict,
    tools: list[dict],
    args: argparse.Namespace,
) -> dict:
    env = MockEnvironment(base_db, task)
    messages = build_initial_messages(policy, task)
    effective_calls: list[dict] = []
    turns = []
    invalid_total = 0
    turn_budget_exhausted = False
    settings = {
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "tool_choice": "auto",
        "enable_thinking": False,
    }

    for turn_idx in range(args.max_agent_turns):
        turn_request_hash = request_hash(messages, tools, settings)
        generation = backend.complete(messages, tools)
        raw_calls, raw_invalid = parse_calls_from_text(generation.text)
        invalid_total += raw_invalid
        assistant_for_context = generation.text
        selected_calls = raw_calls
        selected_text = generation.text
        projection = ""
        projection_invalid = 0

        if lane in {"constrained", "protected"}:
            projection = projected_text(generation.text, messages, tools, args.constrained_max_calls)
            projected_calls, projection_invalid = parse_calls_from_text(projection)
            selected_calls = projected_calls
            if projection.strip():
                selected_text = projection
                assistant_for_context = projection

        if lane == "protected":
            guarded = []
            for call in selected_calls:
                ok, reason = env.guard_call(call.get("name"), call.get("arguments") or {}, tools)
                if ok:
                    guarded.append(call)
                else:
                    env.blocked_calls.append({"call": call, "reason": reason})
            selected_calls = guarded
            if selected_calls:
                selected_text = qwen_native_tool_call_text(selected_calls)
                assistant_for_context = selected_text

        messages.append({"role": "assistant", "content": assistant_for_context})
        tool_results = []
        for call in selected_calls:
            result = env.execute(call.get("name"), call.get("arguments") or {})
            effective_calls.append({"name": call.get("name"), "arguments": call.get("arguments") or {}})
            tool_results.append({"call": call, "result": result})
            messages.append(tool_response_message(result))

        turns.append(
            {
                "turn_idx": turn_idx,
                "request_hash": turn_request_hash,
                "raw_assistant": generation.text,
                "raw_calls": raw_calls,
                "raw_invalid_tool_call_count": raw_invalid,
                "projection": projection,
                "projection_invalid_tool_call_count": projection_invalid,
                "selected_text": selected_text,
                "selected_calls": selected_calls,
                "tool_results": tool_results,
                "seconds": generation.seconds,
                "tokens": generation.tokens,
                "backend_meta": generation.backend_meta or {},
            }
        )
        if not selected_calls:
            break
    else:
        turn_budget_exhausted = True

    action_reward, action_checks = evaluate_actions(task, effective_calls, tools)
    db_reward, db_checks = evaluate_db_state(task, env)
    env_reward, env_checks = evaluate_env_assertions(task, env)
    reward = bool(action_reward and db_reward and env_reward)
    row = {
        "status": "ok",
        "task_id": task.get("id"),
        "lane": lane,
        "reward": reward,
        "action_reward": action_reward,
        "db_reward": db_reward,
        "env_assertion_reward": env_reward,
        "action_checks": action_checks,
        "db_checks": db_checks,
        "env_assertion_checks": env_checks,
        "expected_actions": ((task.get("evaluation_criteria") or {}).get("actions") or []),
        "effective_calls": effective_calls,
        "invalid_tool_call_count": invalid_total,
        "tool_execution_errors": list(env.tool_errors),
        "protected_blocked_calls": list(env.blocked_calls),
        "turn_budget_exhausted": turn_budget_exhausted,
        "turns": turns,
        "final_db": env.db,
    }
    row["failures"] = failure_taxonomy(row)
    return row


def load_tau2_mock(tau2_root: Path, selected_ids: list[str]) -> tuple[str, dict, list[dict]]:
    domain = tau2_root / "data/tau2/domains/mock"
    policy = (domain / "policy.md").read_text(encoding="utf-8")
    base_db = load_json(domain / "db.json")
    tasks = load_json(domain / "tasks.json")
    by_id = {task.get("id"): task for task in tasks}
    missing = [task_id for task_id in selected_ids if task_id not in by_id]
    if missing:
        raise ValueError(f"missing tau2 mock task ids: {missing}")
    return policy, base_db, [by_id[task_id] for task_id in selected_ids]


def build_backend(args: argparse.Namespace):
    if args.backend == "openai":
        return OpenAIBackend(args)
    if args.backend == "fastdllm-ar":
        return FastDllmARBackend(args)
    if args.backend == "diffusion":
        return DiffusionBackend(args)
    raise ValueError(args.backend)


def summarize(rows: list[dict], manifest: dict) -> dict:
    lanes = defaultdict(lambda: {"records": 0, "reward": 0, "action_reward": 0, "db_reward": 0, "env_assertion_reward": 0})
    failures = defaultdict(Counter)
    for row in rows:
        lane = row["lane"]
        lanes[lane]["records"] += 1
        for key in ("reward", "action_reward", "db_reward", "env_assertion_reward"):
            lanes[lane][key] += int(bool(row.get(key)))
        failures[lane].update(row.get("failures") or [])
    lane_summary = {}
    for lane, totals in lanes.items():
        records = max(1, totals["records"])
        lane_summary[lane] = {
            **totals,
            "score": totals["reward"] / records,
            "action_score": totals["action_reward"] / records,
            "db_score": totals["db_reward"] / records,
            "env_assertion_score": totals["env_assertion_reward"] / records,
            "failures": dict(failures[lane]),
        }
    return {"manifest": manifest, "lanes": lane_summary, "records": len(rows)}


def manifest_for(args: argparse.Namespace, backend_meta: dict, policy: str, base_db: dict, tasks: list[dict], tools: list[dict]) -> dict:
    task_ids = [task.get("id") for task in tasks]
    return {
        "benchmark": "tau2-bench public mock domain",
        "tau2_root": str(args.tau2_root),
        "tau2_commit": git_rev(args.tau2_root),
        "task_ids": task_ids,
        "task_subset_sha256": sha256_json(tasks),
        "policy_sha256": sha256_text(policy),
        "db_sha256": sha256_json(base_db),
        "tools_sha256": sha256_json(tools),
        "backend": backend_meta,
        "seed": int(args.seed),
        "max_agent_turns": int(args.max_agent_turns),
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": float(args.temperature),
        "lanes": split_csv(args.lanes),
        "parser_version": "eval_toolcall_jsonl.extract_tool_calls:qwen_native",
        "constrained_lane": {
            "projection": "sequence_preserving_constrained_tool_call_text",
            "schema_only": True,
            "max_calls": int(args.constrained_max_calls),
        },
        "protected_lane": {
            "projection": "same_as_constrained",
            "guards": ["known_tool", "json_schema", "mock_policy_write_preconditions"],
        },
    }


def main() -> None:
    args = parse_args()
    task_ids = args.task_id or DEFAULT_TASK_IDS
    lanes = split_csv(args.lanes)
    if not lanes:
        raise ValueError("--lanes resolved to empty list")
    policy, base_db, tasks = load_tau2_mock(args.tau2_root, task_ids)
    tools = mock_tools()
    backend = build_backend(args)
    manifest = manifest_for(args, backend.metadata(), policy, base_db, tasks, tools)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for lane in lanes:
            for task in tasks:
                try:
                    row = run_episode(
                        backend=backend,
                        task=task,
                        lane=lane,
                        policy=policy,
                        base_db=base_db,
                        tools=tools,
                        args=args,
                    )
                except Exception as exc:
                    row = {
                        "status": "error",
                        "task_id": task.get("id"),
                        "lane": lane,
                        "error": f"{type(exc).__name__}: {exc}",
                        "failures": ["backend_error"],
                    }
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"{args.backend} {lane} {task.get('id')} "
                    f"status={row.get('status')} reward={row.get('reward')}",
                    flush=True,
                )

    summary = summarize(rows, manifest)
    summary_path = args.summary_json or args.out_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
