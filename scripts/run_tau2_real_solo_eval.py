#!/usr/bin/env python3
"""Run a small public tau2 real-domain solo eval through the FLARE proxy."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAU2_ROOT = Path("/tmp/qwen_diffusion_external/tau2-bench")
DEFAULT_BASE = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_ADAPTER = ROOT / "runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000"
DEFAULT_OUT = ROOT / "runs/agentic_eval/tau2_real_solo.jsonl"

if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(DEFAULT_TAU2_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(DEFAULT_TAU2_ROOT / "src"))

from eval_fastdllm_toolcall_cases import sequence_preserving_constrained_tool_call_text
from eval_toolcall_jsonl import extract_tool_calls, qwen_native_tool_call_text, schema_errors, tool_schema_by_name
from run_tau2_mock_agentic_eval import DiffusionBackend, FastDllmARBackend, OpenAIBackend

from tau2 import registry
from tau2.agent.base_agent import ValidAgentInputMessage, is_valid_agent_history_message
from tau2.agent.llm_agent import AGENT_SOLO_INSTRUCTION, SYSTEM_PROMPT_SOLO, LLMAgentState, LLMSoloAgent
from tau2.data_model.message import AssistantMessage, Message, MultiToolMessage, SystemMessage, ToolCall, ToolMessage, UserMessage
from tau2.evaluator.evaluator import EvaluationType
from tau2.environment.tool import Tool, as_tool
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.runner.simulation import run_simulation
from tau2.user.user_simulator import DummyUser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau2-root", type=Path, default=DEFAULT_TAU2_ROOT)
    parser.add_argument("--domain", default="telecom")
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--num-tasks", type=int, default=12)
    parser.add_argument("--lanes", default="raw,constrained,protected")
    parser.add_argument("--backend", choices=["openai", "fastdllm-ar", "diffusion"], required=True)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-errors", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--constrained-max-calls", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--served-model", default="qwen3.5-9b-ar")
    parser.add_argument("--serving-context-length", type=int, default=None)
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
    parser.add_argument(
        "--kickoff-user-message",
        default="Begin the task.",
        help="Neutral user message required by the Qwen chat template in solo mode.",
    )
    return parser.parse_args()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str))


def split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.replace(" ", ",").split(",") if item.strip()]


def git_rev(path: Path) -> str | None:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def done() -> str:
    """Call this function when you are done with the task."""

    return LLMSoloAgent.STOP_TOKEN


def tool_schemas(tools: list[Tool]) -> list[dict]:
    return [tool.openai_schema for tool in tools]


def message_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def tool_response_content(message: ToolMessage) -> str:
    return "<tool_response>\n" + message_content(message.content) + "\n</tool_response>"


def tau_message_to_chat(message: Message) -> dict | None:
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content or ""}
    if isinstance(message, UserMessage):
        if message.is_tool_call():
            return {"role": "user", "content": qwen_native_tool_call_text([tc.model_dump() for tc in message.tool_calls])}
        return {"role": "user", "content": message.content or ""}
    if isinstance(message, AssistantMessage):
        if message.is_tool_call():
            return {"role": "assistant", "content": qwen_native_tool_call_text([tc.model_dump() for tc in message.tool_calls])}
        return {"role": "assistant", "content": message.content or ""}
    if isinstance(message, ToolMessage):
        return {"role": "user", "content": tool_response_content(message)}
    if isinstance(message, MultiToolMessage):
        return {"role": "user", "content": "\n".join(tool_response_content(item) for item in message.tool_messages)}
    return None


def tau_messages_to_chat(messages: list[Message]) -> list[dict]:
    rows = []
    for message in messages:
        row = tau_message_to_chat(message)
        if row is not None:
            rows.append(row)
    return rows


def conversation_text(messages: list[Message]) -> str:
    lines = []
    for message in messages:
        if isinstance(message, MultiToolMessage):
            for tool_message in message.tool_messages:
                lines.append(f"tool: {tool_message.content}")
        else:
            lines.append(f"{message.role}: {getattr(message, 'content', '')}")
    return "\n".join(lines)


def parse_calls(text: str) -> tuple[list[dict], int]:
    return extract_tool_calls(text or "")


def calls_are_schema_valid(calls: list[dict], invalid: int, schemas: list[dict]) -> bool:
    if invalid or not calls:
        return False
    by_name = tool_schema_by_name(schemas)
    for call in calls:
        schema = by_name.get(call.get("name"))
        if not schema:
            return False
        if schema_errors(call.get("arguments") or {}, schema):
            return False
    return True


def protected_calls(calls: list[dict], schemas: list[dict]) -> tuple[list[dict], list[dict]]:
    by_name = tool_schema_by_name(schemas)
    kept = []
    blocked = []
    for call in calls:
        schema = by_name.get(call.get("name"))
        if not schema:
            blocked.append({"call": call, "reason": "unknown_tool"})
            continue
        errors = schema_errors(call.get("arguments") or {}, schema)
        if errors:
            blocked.append({"call": call, "reason": "schema_error:" + ";".join(errors)})
            continue
        kept.append(call)
    return kept, blocked


class ProxySoloAgent(LLMSoloAgent):
    STOP_FUNCTION_NAME = LLMSoloAgent.STOP_FUNCTION_NAME
    STOP_TOKEN = LLMSoloAgent.STOP_TOKEN

    def __init__(
        self,
        *,
        tools: list[Tool],
        domain_policy: str,
        task,
        backend,
        lane: str,
        constrained_max_calls: int,
        kickoff_user_message: str,
    ):
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            task=task,
            llm="flare-proxy",
            llm_args={},
        )
        self.backend = backend
        self.lane = lane
        self.constrained_max_calls = constrained_max_calls
        self.kickoff_user_message = kickoff_user_message
        self.tool_schemas = tool_schemas(self.tools)
        self.events: list[dict] = []
        self.blocked_calls: list[dict] = []
        self._call_seq = 0

    @property
    def system_prompt(self) -> str:
        agent_instruction = AGENT_SOLO_INSTRUCTION.format(
            stop_function_name=self.STOP_FUNCTION_NAME,
            stop_token=self.STOP_TOKEN,
        )
        return SYSTEM_PROMPT_SOLO.format(
            agent_instruction=agent_instruction,
            domain_policy=self.domain_policy,
            ticket=self.task.ticket,
        )

    def set_seed(self, seed: int):
        self.seed = seed

    def get_init_state(self, message_history: list[Message] | None = None) -> LLMAgentState:
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Message history must contain only agent-valid messages."
        )
        messages = list(copy.deepcopy(message_history))
        if self.kickoff_user_message:
            messages.append(UserMessage(role="user", content=self.kickoff_user_message))
        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=messages,
        )

    @classmethod
    def is_stop(cls, message: AssistantMessage) -> bool:
        return bool(message.content and cls.STOP_TOKEN in message.content)

    def _calls_to_tau2(self, calls: list[dict]) -> list[ToolCall]:
        out = []
        for call in calls:
            self._call_seq += 1
            out.append(
                ToolCall(
                    id=f"call_{self._call_seq}",
                    name=str(call.get("name")),
                    arguments=call.get("arguments") or {},
                    requestor="assistant",
                )
            )
        return out

    def generate_next_message(
        self,
        message: ValidAgentInputMessage | None,
        state: LLMAgentState,
    ) -> tuple[AssistantMessage, LLMAgentState]:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        elif message is not None:
            state.messages.append(message)

        chat_messages = tau_messages_to_chat(state.system_messages + state.messages)
        generation = self.backend.complete(chat_messages, self.tool_schemas)
        raw_calls, raw_invalid = parse_calls(generation.text)
        selected_calls = raw_calls
        selected_text = generation.text
        projection = ""
        projection_invalid = 0

        if self.lane in {"constrained", "protected"} and not calls_are_schema_valid(raw_calls, raw_invalid, self.tool_schemas):
            projection = sequence_preserving_constrained_tool_call_text(
                generation.text,
                self.tool_schemas,
                context_text=conversation_text(state.messages),
                max_calls=self.constrained_max_calls,
            )
            projected_calls, projection_invalid = parse_calls(projection)
            selected_calls = projected_calls
            if projection.strip():
                selected_text = projection

        blocked = []
        if self.lane == "protected":
            selected_calls, blocked = protected_calls(selected_calls, self.tool_schemas)
            self.blocked_calls.extend(blocked)
            if selected_calls:
                selected_text = qwen_native_tool_call_text(selected_calls)

        stop_requested = any(call.get("name") == self.STOP_FUNCTION_NAME for call in selected_calls)
        if stop_requested:
            assistant_message = AssistantMessage(
                role="assistant",
                content=self.STOP_TOKEN,
                raw_data={"raw_text": generation.text, "projection": projection, "lane": self.lane},
                generation_time_seconds=generation.seconds,
            )
        elif selected_calls:
            assistant_message = AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=self._calls_to_tau2(selected_calls),
                raw_data={"raw_text": generation.text, "projection": projection, "lane": self.lane},
                generation_time_seconds=generation.seconds,
            )
        else:
            assistant_message = AssistantMessage(
                role="assistant",
                content=selected_text or generation.text or "",
                raw_data={"raw_text": generation.text, "projection": projection, "lane": self.lane},
                generation_time_seconds=generation.seconds,
            )

        self.events.append(
            {
                "raw_text": generation.text,
                "raw_calls": raw_calls,
                "raw_invalid_tool_call_count": raw_invalid,
                "projection": projection,
                "projection_invalid_tool_call_count": projection_invalid,
                "selected_calls": selected_calls,
                "blocked_calls": blocked,
                "stop_requested": stop_requested,
                "seconds": generation.seconds,
                "tokens": generation.tokens,
                "backend_meta": generation.backend_meta or {},
            }
        )
        state.messages.append(assistant_message)
        return assistant_message, state


def build_backend(args: argparse.Namespace):
    if args.backend == "openai":
        return OpenAIBackend(args)
    if args.backend == "fastdllm-ar":
        return FastDllmARBackend(args)
    if args.backend == "diffusion":
        return DiffusionBackend(args)
    raise ValueError(args.backend)


def select_tasks(domain: str, task_ids: list[str], num_tasks: int):
    tasks = registry.get_tasks_loader(domain)()
    if task_ids:
        by_id = {task.id: task for task in tasks}
        missing = [task_id for task_id in task_ids if task_id not in by_id]
        if missing:
            raise ValueError(f"missing task ids: {missing}")
        tasks = [by_id[task_id] for task_id in task_ids]
    else:
        tasks = [task for task in tasks if LLMSoloAgent.check_valid_task(task)]
        if num_tasks > 0:
            tasks = tasks[:num_tasks]
    return tasks


def tool_call_rows(messages: list[Message]) -> list[dict]:
    rows = []
    for message in messages:
        if isinstance(message, AssistantMessage) and message.is_tool_call():
            for call in message.tool_calls:
                rows.append(call.model_dump())
    return rows


def tool_error_count(messages: list[Message]) -> int:
    return sum(1 for message in messages if isinstance(message, ToolMessage) and message.error)


def reward_breakdown_dict(result) -> dict:
    info = result.reward_info
    if info is None:
        return {}
    out = {}
    if info.reward_breakdown:
        for key, value in info.reward_breakdown.items():
            out[str(key.value if hasattr(key, "value") else key)] = value
    return out


def action_reward(result) -> float | None:
    checks = (result.reward_info.action_checks if result.reward_info else None) or []
    if not checks:
        return None
    return 1.0 if all(check.action_match for check in checks) else 0.0


def partial_action_reward(result) -> float | None:
    checks = (result.reward_info.action_checks if result.reward_info else None) or []
    if not checks:
        return None
    return sum(1.0 for check in checks if check.action_match) / len(checks)


def action_checks_dicts(result) -> list[dict]:
    checks = (result.reward_info.action_checks if result.reward_info else None) or []
    return [check.model_dump() for check in checks]


def env_assertion_reward(result) -> float | None:
    checks = (result.reward_info.env_assertions if result.reward_info else None) or []
    if not checks:
        return None
    reward = 1.0
    for check in checks:
        reward *= float(check.reward)
    return reward


def failure_taxonomy(row: dict) -> list[str]:
    failures = []
    if row.get("status") != "ok":
        return ["backend_error"]
    if row.get("termination_reason") not in {"agent_stop", "user_stop"}:
        failures.append("stop_turn_budget")
    if not row.get("tool_calls"):
        failures.append("no_tool_call")
    if row.get("invalid_tool_call_count", 0):
        failures.append("invalid_call")
    expected = row.get("expected_actions") or []
    expected_names = [item.get("name") for item in expected]
    called_names = [item.get("name") for item in row.get("tool_calls") or []]
    if called_names and expected_names and not (set(called_names) & set(expected_names)):
        failures.append("wrong_tool")
    if row.get("action_reward") == 0.0:
        failures.append("wrong_args")
    if row.get("tool_error_count", 0):
        failures.append("tool_execution_error")
    if row.get("protected_blocked_calls"):
        failures.append("protected_blocked_call")
    if row.get("env_assertion_reward") == 0.0:
        failures.append("task_state_failure")
    if row.get("reward", 0.0) == 0.0 and not failures:
        failures.append("task_state_failure")
    return failures or ["pass"]


def cuda_memory_peaks(row: dict) -> tuple[float | None, float | None]:
    max_allocated = None
    max_reserved = None
    for event in row.get("events") or []:
        memory = ((event.get("backend_meta") or {}).get("cuda_memory") or {})
        allocated = memory.get("max_allocated_gib")
        reserved = memory.get("max_reserved_gib")
        if allocated is not None:
            max_allocated = float(allocated) if max_allocated is None else max(max_allocated, float(allocated))
        if reserved is not None:
            max_reserved = float(reserved) if max_reserved is None else max(max_reserved, float(reserved))
    return max_allocated, max_reserved


def event_speed_stats(row: dict) -> dict[str, float | int | None]:
    tokens = 0
    generation_seconds = 0.0
    denoise_forwards = 0
    cache_advance_calls = 0
    has_cache_stats = False
    for event in row.get("events") or []:
        tokens += int(event.get("tokens") or 0)
        generation_seconds += float(event.get("seconds") or 0.0)
        cache_stats = ((event.get("backend_meta") or {}).get("flare_cache_stats") or {})
        if cache_stats:
            has_cache_stats = True
            denoise_forwards += int(cache_stats.get("read_calls") or 0)
            cache_advance_calls += int(cache_stats.get("advance_calls") or 0)
    row_seconds = float(row.get("seconds") or 0.0)
    return {
        "generated_tokens": tokens,
        "backend_generation_seconds": generation_seconds,
        "backend_tokens_per_second": (tokens / generation_seconds) if generation_seconds > 0 else None,
        "end_to_end_seconds": row_seconds,
        "end_to_end_tokens_per_second": (tokens / row_seconds) if row_seconds > 0 else None,
        "denoise_forwards": denoise_forwards if has_cache_stats else None,
        "denoise_forwards_per_token": (denoise_forwards / tokens) if has_cache_stats and tokens > 0 else None,
        "cache_advance_calls": cache_advance_calls if has_cache_stats else None,
        "cache_advance_calls_per_token": (cache_advance_calls / tokens) if has_cache_stats and tokens > 0 else None,
    }


def run_one(args: argparse.Namespace, backend, task, lane: str) -> dict:
    env = registry.get_env_constructor(args.domain)(solo_mode=True)
    tools = env.get_tools()
    agent = ProxySoloAgent(
        tools=tools,
        domain_policy=env.get_policy(),
        task=task,
        backend=backend,
        lane=lane,
        constrained_max_calls=args.constrained_max_calls,
        kickoff_user_message=args.kickoff_user_message,
    )
    user = DummyUser()
    orchestrator = Orchestrator(
        domain=args.domain,
        agent=agent,
        user=user,
        environment=env,
        task=task,
        max_steps=args.max_steps,
        max_errors=args.max_errors,
        seed=args.seed,
        solo_mode=True,
        validate_communication=False,
        timeout=args.timeout,
    )
    result = run_simulation(orchestrator, evaluation_type=EvaluationType.ALL)
    messages = result.messages or []
    row = {
        "status": "ok",
        "domain": args.domain,
        "task_id": task.id,
        "lane": lane,
        "reward": float(result.reward_info.reward if result.reward_info else 0.0),
        "reward_breakdown": reward_breakdown_dict(result),
        "action_reward": action_reward(result),
        "partial_action_reward": partial_action_reward(result),
        "action_checks": action_checks_dicts(result),
        "env_assertion_reward": env_assertion_reward(result),
        "termination_reason": result.termination_reason,
        "tool_calls": tool_call_rows(messages),
        "tool_error_count": tool_error_count(messages),
        "invalid_tool_call_count": sum(int(event.get("raw_invalid_tool_call_count") or 0) for event in agent.events),
        "protected_blocked_calls": agent.blocked_calls,
        "expected_actions": [action.model_dump() for action in ((task.evaluation_criteria.actions if task.evaluation_criteria else []) or [])],
        "reward_basis": [str(item.value if hasattr(item, "value") else item) for item in ((task.evaluation_criteria.reward_basis if task.evaluation_criteria else []) or [])],
        "events": agent.events,
        "messages": [message.model_dump() for message in messages],
    }
    row["failures"] = failure_taxonomy(row)
    return row


def manifest(args: argparse.Namespace, backend_meta: dict, tasks: list) -> dict:
    env = registry.get_env_constructor(args.domain)(solo_mode=True)
    tools = env.get_tools() + [as_tool(done)]
    task_dump = [task.model_dump() for task in tasks]
    return {
        "benchmark": "tau2-bench public real domain solo mode",
        "tau2_root": str(args.tau2_root),
        "tau2_commit": git_rev(args.tau2_root),
        "domain": args.domain,
        "solo_mode": True,
        "qwen_kickoff_user_message": args.kickoff_user_message,
        "task_ids": [task.id for task in tasks],
        "task_subset_sha256": sha256_json(task_dump),
        "policy_sha256": sha256_text(env.get_policy()),
        "tools_sha256": sha256_json(tool_schemas(tools)),
        "backend": backend_meta,
        "seed": int(args.seed),
        "max_steps": int(args.max_steps),
        "max_errors": int(args.max_errors),
        "max_new_tokens": int(args.max_new_tokens),
        "serving_context_length": args.serving_context_length,
        "temperature": float(args.temperature),
        "threshold": float(args.threshold),
        "top_p": float(args.top_p),
        "lanes": split_csv(args.lanes),
        "parser_version": "eval_toolcall_jsonl.extract_tool_calls:qwen_native",
        "scoring": "official tau2 run_simulation/evaluate_simulation, EvaluationType.ALL",
    }


def summarize(rows: list[dict], manifest_obj: dict) -> dict:
    lanes = defaultdict(
        lambda: {
            "records": 0,
            "reward": 0.0,
            "action_reward": 0.0,
            "action_records": 0,
            "partial_action_reward": 0.0,
            "partial_action_records": 0,
            "env_assertion_reward": 0.0,
            "env_records": 0,
            "generated_tokens": 0,
            "backend_generation_seconds": 0.0,
            "end_to_end_seconds": 0.0,
            "denoise_forwards": 0,
            "denoise_records": 0,
            "cache_advance_calls": 0,
            "max_prompt_tokens": None,
            "cuda_max_memory_allocated_gib": None,
            "cuda_max_memory_reserved_gib": None,
        }
    )
    failures = defaultdict(Counter)
    for row in rows:
        lane = row["lane"]
        lanes[lane]["records"] += 1
        lanes[lane]["reward"] += float(row.get("reward") or 0.0)
        if row.get("action_reward") is not None:
            lanes[lane]["action_records"] += 1
            lanes[lane]["action_reward"] += float(row["action_reward"])
        if row.get("partial_action_reward") is not None:
            lanes[lane]["partial_action_records"] += 1
            lanes[lane]["partial_action_reward"] += float(row["partial_action_reward"])
        if row.get("env_assertion_reward") is not None:
            lanes[lane]["env_records"] += 1
            lanes[lane]["env_assertion_reward"] += float(row["env_assertion_reward"])
        speed = event_speed_stats(row)
        lanes[lane]["generated_tokens"] += int(speed["generated_tokens"] or 0)
        lanes[lane]["backend_generation_seconds"] += float(speed["backend_generation_seconds"] or 0.0)
        lanes[lane]["end_to_end_seconds"] += float(speed["end_to_end_seconds"] or 0.0)
        if speed["denoise_forwards"] is not None:
            lanes[lane]["denoise_records"] += 1
            lanes[lane]["denoise_forwards"] += int(speed["denoise_forwards"] or 0)
            lanes[lane]["cache_advance_calls"] += int(speed["cache_advance_calls"] or 0)
        for event in row.get("events") or []:
            prompt_tokens = (event.get("backend_meta") or {}).get("prompt_tokens")
            if prompt_tokens is not None:
                previous = lanes[lane]["max_prompt_tokens"]
                lanes[lane]["max_prompt_tokens"] = int(prompt_tokens) if previous is None else max(previous, int(prompt_tokens))
        max_allocated, max_reserved = cuda_memory_peaks(row)
        if max_allocated is not None:
            previous = lanes[lane]["cuda_max_memory_allocated_gib"]
            lanes[lane]["cuda_max_memory_allocated_gib"] = max_allocated if previous is None else max(previous, max_allocated)
        if max_reserved is not None:
            previous = lanes[lane]["cuda_max_memory_reserved_gib"]
            lanes[lane]["cuda_max_memory_reserved_gib"] = max_reserved if previous is None else max(previous, max_reserved)
        failures[lane].update(row.get("failures") or [])
    lane_summary = {}
    for lane, totals in lanes.items():
        records = max(1, totals["records"])
        lane_summary[lane] = {
            **totals,
            "score": totals["reward"] / records,
            "action_score": totals["action_reward"] / max(1, totals["action_records"]),
            "partial_action_score": totals["partial_action_reward"] / max(1, totals["partial_action_records"]),
            "env_assertion_score": totals["env_assertion_reward"] / max(1, totals["env_records"]),
            "backend_tokens_per_second": (
                totals["generated_tokens"] / totals["backend_generation_seconds"]
                if totals["backend_generation_seconds"] > 0
                else None
            ),
            "end_to_end_tokens_per_second": (
                totals["generated_tokens"] / totals["end_to_end_seconds"] if totals["end_to_end_seconds"] > 0 else None
            ),
            "denoise_forwards_per_token": (
                totals["denoise_forwards"] / totals["generated_tokens"]
                if totals["denoise_records"] and totals["generated_tokens"] > 0
                else None
            ),
            "cache_advance_calls_per_token": (
                totals["cache_advance_calls"] / totals["generated_tokens"]
                if totals["denoise_records"] and totals["generated_tokens"] > 0
                else None
            ),
            "failures": dict(failures[lane]),
        }
    speed_totals = {
        "generated_tokens": sum(int(lane.get("generated_tokens") or 0) for lane in lane_summary.values()),
        "backend_generation_seconds": sum(float(lane.get("backend_generation_seconds") or 0.0) for lane in lane_summary.values()),
        "end_to_end_seconds": sum(float(lane.get("end_to_end_seconds") or 0.0) for lane in lane_summary.values()),
        "denoise_forwards": sum(int(lane.get("denoise_forwards") or 0) for lane in lane_summary.values()),
        "denoise_records": sum(int(lane.get("denoise_records") or 0) for lane in lane_summary.values()),
        "cache_advance_calls": sum(int(lane.get("cache_advance_calls") or 0) for lane in lane_summary.values()),
    }
    speed = {
        **speed_totals,
        "backend_tokens_per_second": (
            speed_totals["generated_tokens"] / speed_totals["backend_generation_seconds"]
            if speed_totals["backend_generation_seconds"] > 0
            else None
        ),
        "end_to_end_tokens_per_second": (
            speed_totals["generated_tokens"] / speed_totals["end_to_end_seconds"]
            if speed_totals["end_to_end_seconds"] > 0
            else None
        ),
        "denoise_forwards_per_token": (
            speed_totals["denoise_forwards"] / speed_totals["generated_tokens"]
            if speed_totals["denoise_records"] and speed_totals["generated_tokens"] > 0
            else None
        ),
        "cache_advance_calls_per_token": (
            speed_totals["cache_advance_calls"] / speed_totals["generated_tokens"]
            if speed_totals["denoise_records"] and speed_totals["generated_tokens"] > 0
            else None
        ),
    }
    return {"manifest": manifest_obj, "lanes": lane_summary, "speed": speed, "records": len(rows)}


def main() -> None:
    args = parse_args()
    if str(args.tau2_root / "src") not in sys.path:
        sys.path.insert(0, str(args.tau2_root / "src"))
    tasks = select_tasks(args.domain, args.task_id, args.num_tasks)
    backend = build_backend(args)
    manifest_obj = manifest(args, backend.metadata(), tasks)
    lanes = split_csv(args.lanes)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for lane in lanes:
            for task in tasks:
                start = time.time()
                try:
                    row = run_one(args, backend, task, lane)
                    row["seconds"] = time.time() - start
                except Exception as exc:
                    row = {
                        "status": "error",
                        "domain": args.domain,
                        "task_id": task.id,
                        "lane": lane,
                        "error": f"{type(exc).__name__}: {exc}",
                        "failures": ["backend_error"],
                        "seconds": time.time() - start,
                    }
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                handle.flush()
                print(
                    f"{args.backend} {lane} {task.id} status={row.get('status')} "
                    f"reward={row.get('reward')} failures={row.get('failures')}",
                    flush=True,
                )
    summary = summarize(rows, manifest_obj)
    summary_path = args.summary_json or args.out_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
