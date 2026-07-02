#!/usr/bin/env python3
"""Matched north-star agentic eval: AR-vLLM vs diffusion per-call waves."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from eval_fastdllm_toolcall_cases import full_context_sample, load_model, resolve_single_token_ids, resolve_token_ids  # noqa: E402
from eval_flare_multiturn_percall_waves import (  # noqa: E402
    build_episodes,
    build_schedule,
    make_gen_args,
    synthetic_tool_result,
    trim_after_first_tool_call,
)
from eval_toolcall_jsonl import score_tool_calls, tool_schema_by_name  # noqa: E402
from flare_hf_cache import FlarePrefixCache  # noqa: E402


DEFAULT_INPUT = ROOT / "data/toolcall_eval_native/flare_scaleup_native_58.jsonl"
DEFAULT_OUT_DIR = ROOT / "runs/agentic_eval/northstar_matched_ar_vllm_vs_diffusion"
DEFAULT_AR_MODEL = ROOT / "models/qwen3.5-9b-fastdllm-b1000-vllm-bf16"
DEFAULT_DIFFUSION_BASE = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_DIFFUSION_ADAPTER = ROOT / "runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000"
DEFAULT_CHAT_TEMPLATE = Path("/home/mark/shared/lumoFlyWheel_codex_fork/docker/chat_templates/qwen3-openai-codex.jinja")
EMPTY_THINK_PREFIX = "<think>\n\n</think>\n\n"
ASSISTANT_GENERATION_PROMPT = "<|im_start|>assistant\n" + EMPTY_THINK_PREFIX


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["ar-vllm", "ar-vllm-guided", "diffusion", "report"], required=True)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--episode-limit", type=int, default=20)
    parser.add_argument("--min-turns", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--prompt-tokenizer-path", type=Path, default=DEFAULT_AR_MODEL)
    parser.add_argument("--chat-template-path", type=Path, default=DEFAULT_CHAT_TEMPLATE)
    parser.add_argument("--ar-base-url", default="http://127.0.0.1:9951")
    parser.add_argument("--ar-served-model", default="qwen3.5-9b-fastdllm-b1000-bf16")
    parser.add_argument("--ar-model-path", type=Path, default=DEFAULT_AR_MODEL)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_DIFFUSION_BASE)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_DIFFUSION_ADAPTER)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_AR_MODEL)
    parser.add_argument("--no-merge-adapter", action="store_true", default=True)
    parser.add_argument("--merge-adapter", dest="no_merge_adapter", action="store_false")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--small-block-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--max-extra-tokens", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--diffusion-structural-only", action="store_true")
    parser.add_argument("--diffusion-output-name", default="diffusion")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=20260701)
    return parser.parse_args()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


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


def load_chat_template(path: Path | None) -> str | None:
    if path is None:
        return None
    return path.read_text(encoding="utf-8")


def render_matched_prompt(tokenizer, messages: list[dict], tools: list[dict], chat_template: str | None) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    if tools:
        kwargs["tools"] = tools
    if chat_template is not None:
        kwargs["chat_template"] = chat_template
    return tokenizer.apply_chat_template(messages, **kwargs)


def decode_text(tokenizer, token_ids) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def trim_scored_assistant(text: str) -> str:
    return trim_after_first_tool_call(text.strip())


def tool_response_suffix(payload: Any, next_user_message: str | None = None) -> str:
    content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    suffix = (
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "<tool_response>\n"
        + content
        + "\n</tool_response><|im_end|>\n"
    )
    if next_user_message is not None and str(next_user_message).strip():
        suffix += "<|im_start|>user\n" + str(next_user_message).strip() + "<|im_end|>\n"
    return suffix + ASSISTANT_GENERATION_PROMPT


def next_turn_user_message(episode: dict, next_turn_idx: int) -> str | None:
    turn_user_messages = episode.get("turn_user_messages") or []
    if next_turn_idx < len(turn_user_messages):
        value = turn_user_messages[next_turn_idx]
        if value is not None and str(value).strip():
            return str(value)
    return None


def regex_literal(text: str) -> str:
    return re.escape(str(text))


def schema_type(schema: dict) -> str | None:
    expected = schema.get("type") if isinstance(schema, dict) else None
    if isinstance(expected, list):
        return next((item for item in expected if item != "null"), expected[0] if expected else None)
    return expected


def guided_value_regex(schema: dict) -> str:
    if not isinstance(schema, dict):
        return "[^<]*"
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and enum_values:
        choices = []
        for value in enum_values:
            if isinstance(value, str):
                choices.append(regex_literal(value))
            elif isinstance(value, bool):
                choices.extend([str(value).lower(), str(value)])
            elif value is None:
                choices.append("null")
            else:
                choices.append(regex_literal(json.dumps(value, ensure_ascii=False)))
        return "(?:" + "|".join(dict.fromkeys(choices)) + ")"
    expected = schema_type(schema)
    if expected == "integer":
        return "-?[0-9]+"
    if expected == "number":
        return "-?(?:[0-9]+(?:\\.[0-9]+)?|\\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
    if expected == "boolean":
        return "(?:true|false|True|False)"
    if expected == "array":
        return "\\[[^<]*\\]"
    if expected == "object":
        return "\\{[^<]*\\}"
    return "[^<]*"


def guided_tool_call_regex(tools: list[dict]) -> str:
    alternatives = []
    for tool in tools or []:
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        name = str(fn["name"])
        schema = fn.get("parameters") or {}
        props = schema.get("properties") if isinstance(schema, dict) else {}
        props = props if isinstance(props, dict) else {}
        required = set(schema.get("required") or []) if isinstance(schema, dict) else set()
        params = []
        for prop_name, prop_schema in props.items():
            body = (
                f"<parameter={regex_literal(prop_name)}>\\n"
                f"{guided_value_regex(prop_schema)}\\n"
                "</parameter>\\n"
            )
            if prop_name in required:
                params.append(body)
            else:
                params.append(f"(?:{body})?")
        alternatives.append(
            "<tool_call>\\n"
            f"<function={regex_literal(name)}>\\n"
            + "".join(params)
            + "</function>\\n"
            + "</tool_call>"
        )
    if not alternatives:
        return "<tool_call>\\n<function=[^>]+>\\n(?:<parameter=[^>]+>\\n[^<]*\\n</parameter>\\n)*</function>\\n</tool_call>"
    return "(?:" + "|".join(alternatives) + ")"


def write_manifest(args: argparse.Namespace, episodes: list[dict], tokenizer, chat_template: str | None) -> dict:
    episode_manifest = [
        {
            "episode_idx": episode["episode_idx"],
            "id": episode["id"],
            "source": episode.get("source"),
            "source_family": episode.get("source_family"),
            "source_dataset": episode.get("source_dataset"),
            "source_license": episode.get("source_license"),
            "source_row_idx": episode.get("source_row_idx"),
            "turns": len(episode["gold_blocks"]),
            "turn_user_message_count": sum(
                int(value is not None and bool(str(value).strip()))
                for value in (episode.get("turn_user_messages") or [])
            ),
            "tools_hash": sha256_json(episode.get("tools") or []),
            "gold_hash": sha256_json(episode.get("gold_blocks") or []),
        }
        for episode in episodes
    ]
    tool_close_ids = tokenizer("</tool_call>", add_special_tokens=False).input_ids
    manifest = {
        "input_jsonl": str(args.input_jsonl),
        "episode_count": len(episodes),
        "turn_count": sum(len(episode["gold_blocks"]) for episode in episodes),
        "source_family_counts": dict(
            sorted(Counter(episode.get("source_family") or episode.get("source") or "unknown" for episode in episodes).items())
        ),
        "episodes": episode_manifest,
        "episode_set_hash": sha256_json(episode_manifest),
        "prompt_tokenizer_path": str(args.prompt_tokenizer_path),
        "chat_template_path": str(args.chat_template_path) if args.chat_template_path else None,
        "chat_template_sha256": sha256_text(chat_template or ""),
        "prompt_tokenizer_hash": sha256_json(
            {
                "chat_template": chat_template or getattr(tokenizer, "chat_template", None),
                "vocab_size": getattr(tokenizer, "vocab_size", None),
            }
        ),
        "tool_close_token_ids": [int(token_id) for token_id in tool_close_ids],
        "stop_policy": {
            "temperature": 0.0,
            "stop_string": "</tool_call>",
            "ar_include_stop_str_in_output": True,
            "diffusion_stop_token_included": True,
            "max_new_tokens": args.max_new_tokens,
            "turn_budget": "same_absolute_max_new_tokens",
        },
        "prompt_loop": {
            "mode": "prefix_stable_incremental_completion_prompt",
            "initial_prompt": "chat_template_with_tools_and_generation_prompt",
            "followup_prompt": "previous_prompt_plus_sampled_assistant_plus_tool_response_plus_optional_next_user_plus_generation_prompt",
            "tool_response_role": "user",
            "next_user_role": "user",
            "assistant_generation_prompt": ASSISTANT_GENERATION_PROMPT,
        },
        "ar": {
            "backend": "vllm",
            "model_path": str(args.ar_model_path),
            "served_model": args.ar_served_model,
            "base_url": args.ar_base_url,
            "dtype": "bf16",
            "quant": "none",
            "fr13_apc": True,
        },
        "ar_guided": {
            "backend": "vllm",
            "model_path": str(args.ar_model_path),
            "served_model": args.ar_served_model,
            "base_url": args.ar_base_url,
            "dtype": "bf16",
            "quant": "none",
            "fr13_apc": True,
            "structured_outputs": "regex_from_qwen_xml_tool_schema",
            "constraint_scope": "one complete qwen-native XML tool call; allowed function names and schema properties from tools; values not gold-constrained",
        },
        "diffusion": {
            "backend": "hf_route_i_flare",
            "base_model": str(args.base_model),
            "adapter": str(args.adapter),
            "merge_adapter": not args.no_merge_adapter,
            "dtype": "bf16",
            "quant": "none",
            "decode": (
                "per_call_waves_tau095_structural_only_prefix_cache"
                if args.diffusion_structural_only
                else "per_call_waves_tau095_prefix_cache"
            ),
            "structural_only_projection": bool(args.diffusion_structural_only),
        },
    }
    prior_manifest_path = args.out_dir / "fairness_manifest.json"
    if prior_manifest_path.exists():
        try:
            prior_manifest = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
            prior_core = [
                {
                    "episode_idx": item.get("episode_idx"),
                    "id": item.get("id"),
                    "turns": item.get("turns"),
                    "tools_hash": item.get("tools_hash"),
                    "gold_hash": item.get("gold_hash"),
                }
                for item in prior_manifest.get("episodes", [])
            ]
            current_core = [
                {
                    "episode_idx": item.get("episode_idx"),
                    "id": item.get("id"),
                    "turns": item.get("turns"),
                    "tools_hash": item.get("tools_hash"),
                    "gold_hash": item.get("gold_hash"),
                }
                for item in manifest.get("episodes", [])
            ]
            if prior_core == current_core and prior_manifest.get("episode_set_hash"):
                manifest["episode_set_hash"] = prior_manifest["episode_set_hash"]
                manifest["episode_set_hash_note"] = "preserved from prior manifest; episode ids/tools/gold match"
            if "server_launch" in prior_manifest and "server_launch" not in manifest:
                manifest["server_launch"] = prior_manifest["server_launch"]
        except Exception:
            pass
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prior_manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def turn_budget(args: argparse.Namespace, tokenizer, gold_block: str) -> tuple[int, float]:
    _, schedule_record, schedule_build_seconds = build_schedule(tokenizer, gold_block)
    _ = int(schedule_record.get("token_count") or 0)
    return int(args.max_new_tokens), schedule_build_seconds


def row_from_generation(
    *,
    backend: str,
    episode: dict,
    turn_idx: int,
    prompt: str,
    tools: list[dict],
    gold_block: str,
    assistant_text: str,
    prompt_tokens: int,
    generated_tokens: int,
    turn_wall_seconds: float,
    schedule_build_seconds: float,
    backend_meta: dict,
) -> dict:
    metrics = score_tool_calls(assistant_text, tools, gold_block)
    tool_payload = synthetic_tool_result(assistant_text, gold_block, episode["id"], turn_idx, tools)
    return {
        "backend": backend,
        "episode_idx": episode["episode_idx"],
        "episode_id": episode["id"],
        "source": episode.get("source"),
        "source_family": episode.get("source_family"),
        "source_dataset": episode.get("source_dataset"),
        "source_license": episode.get("source_license"),
        "source_row_idx": episode.get("source_row_idx"),
        "turn_idx": turn_idx,
        "turns_in_episode": len(episode["gold_blocks"]),
        "prompt_sha256": sha256_text(prompt),
        "tools_sha256": sha256_json(tools),
        "gold_sha256": sha256_text(gold_block),
        "status": "ok",
        "prompt_tokens": int(prompt_tokens),
        "generated_token_count": int(generated_tokens),
        "assistant": assistant_text,
        "gold_assistant": gold_block,
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
        "call_errors": metrics["call_errors"],
        "tool_response_payload": tool_payload,
        "turn_wall_seconds": float(turn_wall_seconds),
        "schedule_build_seconds": float(schedule_build_seconds),
        "backend_meta": backend_meta,
    }


def run_ar_vllm(
    args: argparse.Namespace,
    episodes: list[dict],
    tokenizer,
    chat_template: str | None,
    *,
    guided: bool = False,
) -> list[dict]:
    rows = []
    for episode in episodes:
        messages = [dict(message) for message in episode["prompt_messages"]]
        prompt = render_matched_prompt(tokenizer, messages, episode["tools"], chat_template)
        guided_regex = guided_tool_call_regex(episode["tools"]) if guided else None
        for turn_idx, gold_block in enumerate(episode["gold_blocks"]):
            max_tokens, schedule_build_seconds = turn_budget(args, tokenizer, gold_block)
            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            payload = {
                "model": args.ar_served_model,
                "prompt": prompt,
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "stop": ["</tool_call>"],
                "include_stop_str_in_output": True,
            }
            if guided_regex is not None:
                payload["structured_outputs"] = {"regex": guided_regex}
            start = time.time()
            response = post_json(args.ar_base_url.rstrip("/") + "/v1/completions", payload, args.timeout)
            turn_wall_seconds = time.time() - start
            choice = response["choices"][0]
            history_text = str(choice.get("text") or "")
            usage = response.get("usage") or {}
            if "</tool_call>" not in history_text and choice.get("stop_reason") == "</tool_call>":
                history_text = history_text.rstrip() + "\n</tool_call>"
            assistant_text = trim_scored_assistant(history_text)
            row = row_from_generation(
                backend="ar_vllm_guided" if guided else "ar_vllm",
                episode=episode,
                turn_idx=turn_idx,
                prompt=prompt,
                tools=episode["tools"],
                gold_block=gold_block,
                assistant_text=assistant_text,
                prompt_tokens=int(usage.get("prompt_tokens") or len(prompt_ids)),
                generated_tokens=int(usage.get("completion_tokens") or 0),
                turn_wall_seconds=turn_wall_seconds,
                schedule_build_seconds=schedule_build_seconds,
                backend_meta={
                    "finish_reason": choice.get("finish_reason"),
                    "stop_reason": choice.get("stop_reason"),
                    "usage": usage,
                    "max_tokens": max_tokens,
                    "raw_model": response.get("model"),
                    "system_fingerprint": response.get("system_fingerprint"),
                    "structured_outputs": {
                        "type": "regex_from_qwen_xml_tool_schema",
                        "regex_sha256": sha256_text(guided_regex),
                        "regex_chars": len(guided_regex),
                    }
                    if guided_regex is not None
                    else None,
                },
            )
            row["assistant_history_sha256"] = sha256_text(history_text)
            next_user = next_turn_user_message(episode, turn_idx + 1)
            row["next_user_message_sha256"] = sha256_text(next_user) if next_user is not None else None
            rows.append(row)
            prompt = prompt + history_text + tool_response_suffix(row["tool_response_payload"], next_user)
            print(
                f"{'ar_vllm_guided' if guided else 'ar_vllm'} "
                f"episode={episode['episode_idx']} turn={turn_idx + 1}/{len(episode['gold_blocks'])} "
                f"exact_args={int(bool(row['exact_arguments']))} wall={turn_wall_seconds:.3f}s",
                flush=True,
            )
    return rows


def run_diffusion(args: argparse.Namespace, episodes: list[dict], tokenizer, chat_template: str | None) -> list[dict]:
    os.environ["FASTDLLM_FLARE_GDN_ROUTE"] = "route_i"
    os.environ.setdefault("FASTDLLM_FLARE_TWO_STREAM", "1")
    os.environ.setdefault("FLARE_TWO_STREAM", "1")
    model, model_tokenizer = load_model(
        args.base_model,
        args.adapter if args.adapter and args.adapter.exists() else None,
        merge_adapter=not args.no_merge_adapter,
        tokenizer_path=args.tokenizer_path,
    )
    model.eval()
    if hasattr(model, "config"):
        setattr(model.config, "bd_size", int(args.block_size))
    mask_id, stop_token_id, base_stop_token_ids = resolve_token_ids(model, model_tokenizer)
    tool_close_ids = model_tokenizer("</tool_call>", add_special_tokens=False).input_ids
    stop_token_ids = list(dict.fromkeys([int(item) for item in base_stop_token_ids + tool_close_ids]))
    argument_boundary_token_ids = resolve_single_token_ids(
        model_tokenizer, ["<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>"]
    )
    argument_newline_token_ids = resolve_single_token_ids(model_tokenizer, ["\n", "\n\n"])
    rows = []
    for episode in episodes:
        messages = [dict(message) for message in episode["prompt_messages"]]
        prompt = render_matched_prompt(model_tokenizer, messages, episode["tools"], chat_template)
        prefix_cache = FlarePrefixCache()
        gen_args = make_gen_args(
            args,
            condition="percall_waves_tau095",
            prefix_cache=prefix_cache,
            mask_id=mask_id,
            stop_token_id=stop_token_id,
            stop_token_ids=stop_token_ids,
            argument_boundary_token_ids=argument_boundary_token_ids,
            argument_newline_token_ids=argument_newline_token_ids,
        )
        if args.diffusion_structural_only:
            gen_args.two_wave_grammar_forced_only = True
            gen_args.record_projected_token_positions = True
            gen_args.two_wave_no_project_inside_parameter_value = True
        for turn_idx, gold_block in enumerate(episode["gold_blocks"]):
            schedule, schedule_record, schedule_build_seconds = build_schedule(model_tokenizer, gold_block)
            prompt_input_ids = model_tokenizer([prompt], return_tensors="pt").input_ids.to("cuda")
            _ = int(schedule_record.get("token_count") or 0)
            gen_args.max_new_tokens = int(gen_args.max_new_tokens_cap)
            previous_live_tool_schemas = getattr(gen_args, "_live_tool_schemas", None)
            gen_args._live_tool_schemas = tool_schema_by_name(episode["tools"])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.time()
            try:
                with torch.no_grad():
                    generated = full_context_sample(
                        model,
                        prompt_input_ids,
                        model_tokenizer,
                        gen_args,
                        sampler_schedule=schedule,
                        original_len_override=prompt_input_ids.shape[1],
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            finally:
                if previous_live_tool_schemas is None:
                    try:
                        delattr(gen_args, "_live_tool_schemas")
                    except AttributeError:
                        pass
                else:
                    gen_args._live_tool_schemas = previous_live_tool_schemas
            turn_wall_seconds = time.time() - start
            new_ids = generated[prompt_input_ids.shape[1] :]
            history_text = decode_text(model_tokenizer, new_ids)
            assistant_text = trim_scored_assistant(history_text)
            row = row_from_generation(
                backend=(
                    "diffusion_percall_waves_structural_only"
                    if args.diffusion_structural_only
                    else "diffusion_percall_waves"
                ),
                episode=episode,
                turn_idx=turn_idx,
                prompt=prompt,
                tools=episode["tools"],
                gold_block=gold_block,
                assistant_text=assistant_text,
                prompt_tokens=int(prompt_input_ids.shape[1]),
                generated_tokens=int((new_ids != gen_args.mask_id).sum().item()),
                turn_wall_seconds=turn_wall_seconds,
                schedule_build_seconds=schedule_build_seconds,
                backend_meta={
                    "sampler_schedule_events": getattr(gen_args, "_last_sampler_schedule_events", {}),
                    "flare_cache_stats": getattr(gen_args, "_last_flare_cache_stats", {}),
                    "flare_prefix_cache_stats": getattr(gen_args, "_last_flare_prefix_cache_stats", {}),
                    "flare_timing_stats": getattr(gen_args, "_last_flare_timing_stats", {}),
                    "max_new_tokens": gen_args.max_new_tokens,
                },
            )
            row["assistant_history_sha256"] = sha256_text(history_text)
            next_user = next_turn_user_message(episode, turn_idx + 1)
            row["next_user_message_sha256"] = sha256_text(next_user) if next_user is not None else None
            rows.append(row)
            prompt = prompt + history_text + tool_response_suffix(row["tool_response_payload"], next_user)
            cache_hit = bool((row["backend_meta"].get("flare_cache_stats") or {}).get("prefix_cache_hit"))
            print(
                f"diffusion episode={episode['episode_idx']} turn={turn_idx + 1}/{len(episode['gold_blocks'])} "
                f"exact_args={int(bool(row['exact_arguments']))} wall={turn_wall_seconds:.3f}s "
                f"cache_hit={int(cache_hit)}",
                flush=True,
            )
    return rows


def write_rows(out_dir: Path, backend: str, rows: list[dict]) -> None:
    backend_dir = out_dir / backend
    backend_dir.mkdir(parents=True, exist_ok=True)
    with (backend_dir / "turns.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize_backend(rows: list[dict]) -> dict:
    by_episode: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_episode[row["episode_id"]].append(row)
    turns = len(rows)
    wall = sum(float(row.get("turn_wall_seconds") or 0.0) for row in rows)
    generated_tokens = sum(int(row.get("generated_token_count") or 0) for row in rows)
    denoise_forwards = 0.0
    cache_hits = 0
    eligible_followups = 0
    for episode_rows in by_episode.values():
        eligible_followups += max(0, len(episode_rows) - 1)
    for row in rows:
        meta = row.get("backend_meta") or {}
        events = meta.get("sampler_schedule_events") or {}
        denoise_forwards += float(events.get("denoise_forwards_total") or 0.0)
        cache = meta.get("flare_cache_stats") or {}
        cache_hits += int(bool(cache.get("prefix_cache_hit")))
    return {
        "episodes": len(by_episode),
        "turns": turns,
        "valid_tool_json": sum(int(bool(row.get("valid_tool_json"))) for row in rows),
        "exact_tool_sequence": sum(int(bool(row.get("exact_tool_sequence"))) for row in rows),
        "exact_arguments": sum(int(bool(row.get("exact_arguments"))) for row in rows),
        "all_schema_valid": sum(int(bool(row.get("all_schema_valid"))) for row in rows),
        "all_required_args_present": sum(int(bool(row.get("all_required_args_present"))) for row in rows),
        "episode_exact_arguments_all_turns": sum(
            int(all(bool(row.get("exact_arguments")) for row in episode_rows))
            for episode_rows in by_episode.values()
        ),
        "turn_wall_seconds": wall,
        "sec_per_turn": wall / turns if turns else 0.0,
        "generated_tokens": generated_tokens,
        "generated_tokens_per_turn": generated_tokens / turns if turns else 0.0,
        "denoise_forwards_total": denoise_forwards,
        "denoise_forwards_per_turn": denoise_forwards / turns if turns else 0.0,
        "blended_tpf": generated_tokens / denoise_forwards if denoise_forwards else None,
        "prefix_cache_hit_turns": cache_hits,
        "prefix_cache_eligible_followup_turns": eligible_followups,
        "schedule_build_seconds": sum(float(row.get("schedule_build_seconds") or 0.0) for row in rows),
    }


def source_key(row: dict) -> str:
    return str(row.get("source_family") or row.get("source") or "unknown")


def summarize_by_source(rows: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[source_key(row)].append(row)
    return {name: summarize_backend(grouped[name]) for name in sorted(grouped)}


def paired_summary(reference_rows: list[dict], candidate_rows: list[dict]) -> dict:
    reference_by_turn = {(row["episode_id"], row["turn_idx"]): row for row in reference_rows}
    candidate_by_turn = {(row["episode_id"], row["turn_idx"]): row for row in candidate_rows}
    shared_turns = sorted(set(reference_by_turn) & set(candidate_by_turn))
    reference_by_episode: dict[str, list[dict]] = defaultdict(list)
    candidate_by_episode: dict[str, list[dict]] = defaultdict(list)
    for row in reference_rows:
        reference_by_episode[row["episode_id"]].append(row)
    for row in candidate_rows:
        candidate_by_episode[row["episode_id"]].append(row)
    shared_episodes = sorted(set(reference_by_episode) & set(candidate_by_episode))
    reference = summarize_backend(reference_rows)
    candidate = summarize_backend(candidate_rows)
    candidate_only_exact_arguments = sum(
        int(bool(candidate_by_turn[key].get("exact_arguments")) and not bool(reference_by_turn[key].get("exact_arguments")))
        for key in shared_turns
    )
    reference_only_exact_arguments = sum(
        int(bool(reference_by_turn[key].get("exact_arguments")) and not bool(candidate_by_turn[key].get("exact_arguments")))
        for key in shared_turns
    )
    both_exact_arguments = sum(
        int(bool(reference_by_turn[key].get("exact_arguments")) and bool(candidate_by_turn[key].get("exact_arguments")))
        for key in shared_turns
    )
    candidate_only_exact_argument_episodes = sum(
        int(
            all(bool(row.get("exact_arguments")) for row in candidate_by_episode[episode_id])
            and not all(bool(row.get("exact_arguments")) for row in reference_by_episode[episode_id])
        )
        for episode_id in shared_episodes
    )
    reference_only_exact_argument_episodes = sum(
        int(
            all(bool(row.get("exact_arguments")) for row in reference_by_episode[episode_id])
            and not all(bool(row.get("exact_arguments")) for row in candidate_by_episode[episode_id])
        )
        for episode_id in shared_episodes
    )
    both_exact_argument_episodes = sum(
        int(
            all(bool(row.get("exact_arguments")) for row in reference_by_episode[episode_id])
            and all(bool(row.get("exact_arguments")) for row in candidate_by_episode[episode_id])
        )
        for episode_id in shared_episodes
    )
    return {
        "paired_turns": len(shared_turns),
        "exact_arguments_delta": sum(
            int(bool(candidate_by_turn[key].get("exact_arguments")))
            - int(bool(reference_by_turn[key].get("exact_arguments")))
            for key in shared_turns
        ),
        "exact_sequence_delta": sum(
            int(bool(candidate_by_turn[key].get("exact_tool_sequence")))
            - int(bool(reference_by_turn[key].get("exact_tool_sequence")))
            for key in shared_turns
        ),
        "valid_delta": sum(
            int(bool(candidate_by_turn[key].get("valid_tool_json")))
            - int(bool(reference_by_turn[key].get("valid_tool_json")))
            for key in shared_turns
        ),
        "candidate_only_exact_arguments": candidate_only_exact_arguments,
        "reference_only_exact_arguments": reference_only_exact_arguments,
        "both_exact_arguments": both_exact_arguments,
        "neither_exact_arguments": (
            len(shared_turns) - candidate_only_exact_arguments - reference_only_exact_arguments - both_exact_arguments
        ),
        "schema_valid_delta": sum(
            int(bool(candidate_by_turn[key].get("all_schema_valid")))
            - int(bool(reference_by_turn[key].get("all_schema_valid")))
            for key in shared_turns
        ),
        "paired_episodes": len(shared_episodes),
        "episode_exact_arguments_delta": sum(
            int(all(bool(row.get("exact_arguments")) for row in candidate_by_episode[episode_id]))
            - int(all(bool(row.get("exact_arguments")) for row in reference_by_episode[episode_id]))
            for episode_id in shared_episodes
        ),
        "candidate_only_exact_argument_episodes": candidate_only_exact_argument_episodes,
        "reference_only_exact_argument_episodes": reference_only_exact_argument_episodes,
        "both_exact_argument_episodes": both_exact_argument_episodes,
        "neither_exact_argument_episodes": (
            len(shared_episodes)
            - candidate_only_exact_argument_episodes
            - reference_only_exact_argument_episodes
            - both_exact_argument_episodes
        ),
        "reference_wall_seconds": reference["turn_wall_seconds"],
        "candidate_wall_seconds": candidate["turn_wall_seconds"],
        "reference_over_candidate_wall_ratio": (
            reference["turn_wall_seconds"] / candidate["turn_wall_seconds"]
            if candidate["turn_wall_seconds"]
            else 0.0
        ),
        "candidate_over_reference_wall_ratio": (
            candidate["turn_wall_seconds"] / reference["turn_wall_seconds"]
            if reference["turn_wall_seconds"]
            else 0.0
        ),
    }


def paired_summary_by_source(reference_rows: list[dict], candidate_rows: list[dict]) -> dict[str, dict]:
    sources = sorted({source_key(row) for row in reference_rows} | {source_key(row) for row in candidate_rows})
    out: dict[str, dict] = {}
    for source in sources:
        reference_source_rows = [row for row in reference_rows if source_key(row) == source]
        candidate_source_rows = [row for row in candidate_rows if source_key(row) == source]
        if reference_source_rows and candidate_source_rows:
            out[source] = paired_summary(reference_source_rows, candidate_source_rows)
    return out


def write_report(args: argparse.Namespace) -> dict:
    manifest = json.loads((args.out_dir / "fairness_manifest.json").read_text(encoding="utf-8"))
    ar_rows = read_rows(args.out_dir / "ar-vllm" / "turns.jsonl")
    guided_path = args.out_dir / "ar-vllm-guided" / "turns.jsonl"
    ar_guided_rows = read_rows(guided_path) if guided_path.exists() else []
    contaminated_diffusion_rows = read_rows(args.out_dir / "diffusion" / "turns.jsonl")
    corrected_diffusion_path = args.out_dir / "diffusion_structural_only" / "turns.jsonl"
    corrected_diffusion_rows = read_rows(corrected_diffusion_path) if corrected_diffusion_path.exists() else []
    diffusion_rows = corrected_diffusion_rows or contaminated_diffusion_rows
    ar = summarize_backend(ar_rows)
    ar_guided = summarize_backend(ar_guided_rows) if ar_guided_rows else None
    contaminated_diffusion = summarize_backend(contaminated_diffusion_rows)
    diffusion = summarize_backend(diffusion_rows)
    source_breakdown = {
        "ar_vllm": summarize_by_source(ar_rows),
        "ar_vllm_guided": summarize_by_source(ar_guided_rows) if ar_guided_rows else {},
        "diffusion_percall_waves": summarize_by_source(diffusion_rows),
    }
    ar_by_turn = {(row["episode_id"], row["turn_idx"]): row for row in ar_rows}
    diffusion_by_turn = {(row["episode_id"], row["turn_idx"]): row for row in diffusion_rows}
    shared_turns = sorted(set(ar_by_turn) & set(diffusion_by_turn))
    ar_by_episode: dict[str, list[dict]] = defaultdict(list)
    diffusion_by_episode: dict[str, list[dict]] = defaultdict(list)
    for row in ar_rows:
        ar_by_episode[row["episode_id"]].append(row)
    for row in diffusion_rows:
        diffusion_by_episode[row["episode_id"]].append(row)
    shared_episodes = sorted(set(ar_by_episode) & set(diffusion_by_episode))
    paired = {
        "paired_turns": len(shared_turns),
        "exact_arguments_delta_diffusion_minus_ar": sum(
            int(bool(diffusion_by_turn[key].get("exact_arguments"))) - int(bool(ar_by_turn[key].get("exact_arguments")))
            for key in shared_turns
        ),
        "exact_sequence_delta_diffusion_minus_ar": sum(
            int(bool(diffusion_by_turn[key].get("exact_tool_sequence")))
            - int(bool(ar_by_turn[key].get("exact_tool_sequence")))
            for key in shared_turns
        ),
        "valid_delta_diffusion_minus_ar": sum(
            int(bool(diffusion_by_turn[key].get("valid_tool_json"))) - int(bool(ar_by_turn[key].get("valid_tool_json")))
            for key in shared_turns
        ),
        "paired_episodes": len(shared_episodes),
        "episode_exact_arguments_delta_diffusion_minus_ar": sum(
            int(all(bool(row.get("exact_arguments")) for row in diffusion_by_episode[episode_id]))
            - int(all(bool(row.get("exact_arguments")) for row in ar_by_episode[episode_id]))
            for episode_id in shared_episodes
        ),
        "ar_wall_seconds": ar["turn_wall_seconds"],
        "diffusion_wall_seconds": diffusion["turn_wall_seconds"],
        "ar_over_diffusion_wall_ratio": (
            ar["turn_wall_seconds"] / diffusion["turn_wall_seconds"] if diffusion["turn_wall_seconds"] else 0.0
        ),
        "diffusion_over_ar_wall_ratio": (
            diffusion["turn_wall_seconds"] / ar["turn_wall_seconds"] if ar["turn_wall_seconds"] else 0.0
        ),
    }
    paired_guided = None
    if ar_guided_rows:
        guided = paired_summary(ar_guided_rows, diffusion_rows)
        paired_guided = {
            "paired_turns": guided["paired_turns"],
            "exact_arguments_delta_diffusion_minus_ar_guided": guided["exact_arguments_delta"],
            "exact_sequence_delta_diffusion_minus_ar_guided": guided["exact_sequence_delta"],
            "valid_delta_diffusion_minus_ar_guided": guided["valid_delta"],
            "schema_valid_delta_diffusion_minus_ar_guided": guided["schema_valid_delta"],
            "diffusion_only_exact_arguments": guided["candidate_only_exact_arguments"],
            "ar_guided_only_exact_arguments": guided["reference_only_exact_arguments"],
            "both_exact_arguments": guided["both_exact_arguments"],
            "neither_exact_arguments": guided["neither_exact_arguments"],
            "paired_episodes": guided["paired_episodes"],
            "episode_exact_arguments_delta_diffusion_minus_ar_guided": guided[
                "episode_exact_arguments_delta"
            ],
            "diffusion_only_exact_argument_episodes": guided["candidate_only_exact_argument_episodes"],
            "ar_guided_only_exact_argument_episodes": guided["reference_only_exact_argument_episodes"],
            "both_exact_argument_episodes": guided["both_exact_argument_episodes"],
            "neither_exact_argument_episodes": guided["neither_exact_argument_episodes"],
            "ar_guided_wall_seconds": guided["reference_wall_seconds"],
            "diffusion_wall_seconds": guided["candidate_wall_seconds"],
            "ar_guided_over_diffusion_wall_ratio": guided["reference_over_candidate_wall_ratio"],
            "diffusion_over_ar_guided_wall_ratio": guided["candidate_over_reference_wall_ratio"],
        }
        paired_guided_by_source = paired_summary_by_source(ar_guided_rows, diffusion_rows)
    else:
        paired_guided_by_source = {}
    summary = {
        "manifest": manifest,
        "ar_vllm": ar,
        "ar_vllm_guided": ar_guided,
        "diffusion_percall_waves_contaminated": contaminated_diffusion,
        "diffusion_percall_waves": diffusion,
        "diffusion_corrected_structural_only": bool(corrected_diffusion_rows),
        "source_breakdown": source_breakdown,
        "paired": paired,
        "paired_guided": paired_guided,
        "paired_guided_by_source": paired_guided_by_source,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# FLARE North-Star Matched Eval",
        "",
        f"Slice: {manifest['episode_count']} episodes, {manifest['turn_count']} turns.",
        f"Episode set hash: `{manifest['episode_set_hash']}`.",
        "Generated-history loop: prefix-stable completion prompts; each backend appends its sampled assistant text, "
        "then the same synthetic tool-result schema and next generation prompt.",
        "",
        "| Backend | exact_args | episode exact | exact_seq | valid_xml | schema_ok | sec/turn | total wall | gen tok/turn | model forwards/turn |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| AR vLLM FR13 | {ar['exact_arguments']}/{ar['turns']} "
            f"| {ar['episode_exact_arguments_all_turns']}/{ar['episodes']} "
            f"| {ar['exact_tool_sequence']}/{ar['turns']} | {ar['valid_tool_json']}/{ar['turns']} "
            f"| {ar['all_schema_valid']}/{ar['turns']} "
            f"| {ar['sec_per_turn']:.3f} | {ar['turn_wall_seconds']:.3f}s "
            f"| {ar['generated_tokens_per_turn']:.3f} | n/a |"
        ),
    ]
    if ar_guided is not None:
        lines.append(
            f"| AR vLLM FR13 guided | {ar_guided['exact_arguments']}/{ar_guided['turns']} "
            f"| {ar_guided['episode_exact_arguments_all_turns']}/{ar_guided['episodes']} "
            f"| {ar_guided['exact_tool_sequence']}/{ar_guided['turns']} "
            f"| {ar_guided['valid_tool_json']}/{ar_guided['turns']} "
            f"| {ar_guided['all_schema_valid']}/{ar_guided['turns']} "
            f"| {ar_guided['sec_per_turn']:.3f} | {ar_guided['turn_wall_seconds']:.3f}s "
            f"| {ar_guided['generated_tokens_per_turn']:.3f} | n/a |"
        )
    if corrected_diffusion_rows:
        lines.append(
            "| Diffusion per-call waves (measurement-contaminated) "
            f"| {contaminated_diffusion['exact_arguments']}/{contaminated_diffusion['turns']} "
            f"| {contaminated_diffusion['episode_exact_arguments_all_turns']}/{contaminated_diffusion['episodes']} "
            f"| {contaminated_diffusion['exact_tool_sequence']}/{contaminated_diffusion['turns']} "
            f"| {contaminated_diffusion['valid_tool_json']}/{contaminated_diffusion['turns']} "
            f"| {contaminated_diffusion['all_schema_valid']}/{contaminated_diffusion['turns']} "
            f"| {contaminated_diffusion['sec_per_turn']:.3f} "
            f"| {contaminated_diffusion['turn_wall_seconds']:.3f}s "
            f"| {contaminated_diffusion['generated_tokens_per_turn']:.3f} "
            f"| {contaminated_diffusion['denoise_forwards_per_turn']:.3f} |"
        )
    lines.append(
        f"| Diffusion per-call waves{' (corrected structural-only)' if corrected_diffusion_rows else ''} "
        f"| {diffusion['exact_arguments']}/{diffusion['turns']} "
        f"| {diffusion['episode_exact_arguments_all_turns']}/{diffusion['episodes']} "
        f"| {diffusion['exact_tool_sequence']}/{diffusion['turns']} "
        f"| {diffusion['valid_tool_json']}/{diffusion['turns']} "
        f"| {diffusion['all_schema_valid']}/{diffusion['turns']} "
        f"| {diffusion['sec_per_turn']:.3f} | {diffusion['turn_wall_seconds']:.3f}s "
        f"| {diffusion['generated_tokens_per_turn']:.3f} | {diffusion['denoise_forwards_per_turn']:.3f} |"
    )
    if corrected_diffusion_rows:
        audit_path = args.out_dir / "diffusion_structural_only" / "projection_value_audit.json"
        audit_totals = {}
        if audit_path.exists():
            try:
                audit_totals = json.loads(audit_path.read_text(encoding="utf-8")).get("totals") or {}
            except Exception:
                audit_totals = {}
        lines.extend(
            [
                "",
                "## Contamination Note",
                "",
                (
                    "- The original 55/63 diffusion row is measurement-contaminated. "
                    "It was produced without structural-only projection, and the tokenizer-offset audit found "
                    "projected value tokens. The corrected row reruns diffusion with "
                    "`two_wave_grammar_forced_only=True`; AR rows are unchanged."
                ),
                (
                    "- Corrected projection audit: "
                    f"mode=`{audit_totals.get('verification_mode', 'n/a')}`, "
                    f"projected_value_tokens_exact={audit_totals.get('projected_value_tokens_exact', 'n/a')}, "
                    f"projected_token_records={audit_totals.get('projected_token_record_count', 'n/a')}."
                ),
            ]
        )
    headline_lines = []
    if ar_guided is not None:
        turn_holds = diffusion["exact_arguments"] >= ar_guided["exact_arguments"]
        episode_holds = (
            diffusion["episode_exact_arguments_all_turns"]
            >= ar_guided["episode_exact_arguments_all_turns"]
        )
        headline_lines = [
            "",
            "## Headline",
            "",
            (
                "- Diffusion >= AR-guided on exact-args and episode exactness: "
                f"{'YES' if turn_holds and episode_holds else 'NO'} "
                f"(turns {diffusion['exact_arguments']}/{diffusion['turns']} vs "
                f"{ar_guided['exact_arguments']}/{ar_guided['turns']}; episodes "
                f"{diffusion['episode_exact_arguments_all_turns']}/{diffusion['episodes']} vs "
                f"{ar_guided['episode_exact_arguments_all_turns']}/{ar_guided['episodes']})."
            ),
        ]
    guided_delta_lines = []
    if paired_guided is not None:
        guided_delta_lines = [
            f"- Turn exact-args delta, diffusion - AR guided: {paired_guided['exact_arguments_delta_diffusion_minus_ar_guided']} / {paired_guided['paired_turns']}",
            f"- Episode exact-args delta, diffusion - AR guided: {paired_guided['episode_exact_arguments_delta_diffusion_minus_ar_guided']} / {paired_guided['paired_episodes']}",
            f"- Turn exact-args flips: diffusion-only {paired_guided['diffusion_only_exact_arguments']}; AR-guided-only {paired_guided['ar_guided_only_exact_arguments']}; both {paired_guided['both_exact_arguments']}; neither {paired_guided['neither_exact_arguments']}",
            f"- Episode exact-args flips: diffusion-only {paired_guided['diffusion_only_exact_argument_episodes']}; AR-guided-only {paired_guided['ar_guided_only_exact_argument_episodes']}; both {paired_guided['both_exact_argument_episodes']}; neither {paired_guided['neither_exact_argument_episodes']}",
            f"- Valid XML delta, diffusion - AR guided: {paired_guided['valid_delta_diffusion_minus_ar_guided']} / {paired_guided['paired_turns']}",
            f"- Schema-valid delta, diffusion - AR guided: {paired_guided['schema_valid_delta_diffusion_minus_ar_guided']} / {paired_guided['paired_turns']}",
            f"- Wall latency ratio AR guided / diffusion: {paired_guided['ar_guided_over_diffusion_wall_ratio']:.3f}x",
            f"- Wall latency ratio diffusion / AR guided: {paired_guided['diffusion_over_ar_guided_wall_ratio']:.3f}x",
        ]
    source_lines = [
        "",
        "## Source Breakdown",
        "",
        "| Source | Backend | exact_args | episode exact | exact_seq | valid_xml | schema_ok | sec/turn |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    backend_labels = [
        ("AR vLLM FR13", source_breakdown["ar_vllm"]),
        ("AR vLLM FR13 guided", source_breakdown["ar_vllm_guided"]),
        ("Diffusion per-call waves", source_breakdown["diffusion_percall_waves"]),
    ]
    for source in sorted(
        set(source_breakdown["ar_vllm"]) | set(source_breakdown["ar_vllm_guided"]) | set(source_breakdown["diffusion_percall_waves"])
    ):
        for label, table in backend_labels:
            item = table.get(source)
            if not item:
                continue
            turns = int(item.get("turns") or 0)
            episodes_n = int(item.get("episodes") or 0)
            source_lines.append(
                f"| {source} | {label} | {int(item.get('exact_arguments') or 0)}/{turns} "
                f"| {int(item.get('episode_exact_arguments_all_turns') or 0)}/{episodes_n} "
                f"| {int(item.get('exact_tool_sequence') or 0)}/{turns} "
                f"| {int(item.get('valid_tool_json') or 0)}/{turns} "
                f"| {int(item.get('all_schema_valid') or 0)}/{turns} "
                f"| {float(item.get('sec_per_turn') or 0.0):.3f} |"
            )
    if paired_guided_by_source:
        source_lines.extend(["", "### Diffusion vs AR-Guided by Source", ""])
        for source, item in sorted(paired_guided_by_source.items()):
            source_lines.append(
                f"- {source}: exact_args delta {item['exact_arguments_delta']} / {item['paired_turns']}; "
                f"episode delta {item['episode_exact_arguments_delta']} / {item['paired_episodes']}; "
                f"turn flips diffusion-only {item['candidate_only_exact_arguments']}, "
                f"AR-guided-only {item['reference_only_exact_arguments']}; "
                f"episode flips diffusion-only {item['candidate_only_exact_argument_episodes']}, "
                f"AR-guided-only {item['reference_only_exact_argument_episodes']}."
            )
    lines.extend(
        [
        "",
        *headline_lines,
        "## Matched Deltas",
        "",
        f"- Turn exact-args delta, diffusion - AR: {paired['exact_arguments_delta_diffusion_minus_ar']} / {paired['paired_turns']}",
        f"- Episode exact-args delta, diffusion - AR: {paired['episode_exact_arguments_delta_diffusion_minus_ar']} / {paired['paired_episodes']}",
        f"- Wall latency ratio AR / diffusion: {paired['ar_over_diffusion_wall_ratio']:.3f}x",
        f"- Wall latency ratio diffusion / AR: {paired['diffusion_over_ar_wall_ratio']:.3f}x",
        *guided_delta_lines,
        *source_lines,
        "",
        "## Fairness Manifest",
        "",
        f"- AR: `{manifest['ar']['model_path']}`, bf16, no quant, FR13 APC on.",
        *(
            ["- AR guided: regex structured outputs from Qwen XML tool schemas, FR13 APC on."]
            if ar_guided is not None
            else []
        ),
        f"- Diffusion: `{manifest['diffusion']['base_model']}` + `{manifest['diffusion']['adapter']}`, bf16, no quant.",
        f"- Prompt tokenizer: `{manifest['prompt_tokenizer_path']}`.",
        f"- Chat template: `{manifest.get('chat_template_path')}` (`{manifest.get('chat_template_sha256')}`).",
        f"- Prompt loop: `{json.dumps(manifest.get('prompt_loop', {}), sort_keys=True)}`.",
        f"- Stop policy: `{json.dumps(manifest['stop_policy'], sort_keys=True)}`.",
        *(
            [f"- Server launch notes: `{json.dumps(manifest.get('server_launch', {}), sort_keys=True)}`."]
            if manifest.get("server_launch")
            else []
        ),
        f"- Full manifest: `{args.out_dir / 'fairness_manifest.json'}`.",
        f"- Per-turn rows: `{args.out_dir / 'ar-vllm/turns.jsonl'}`, `{args.out_dir / 'ar-vllm-guided/turns.jsonl'}`, `{args.out_dir / 'diffusion/turns.jsonl'}`"
        + (f", `{corrected_diffusion_path}`" if corrected_diffusion_rows else "")
        + ".",
        ]
    )
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"paired": paired, "paired_guided": paired_guided}, indent=2), flush=True)
    print(f"wrote {args.out_dir / 'report.md'}", flush=True)
    return summary


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.backend == "report":
        write_report(args)
        return 0
    tokenizer = AutoTokenizer.from_pretrained(str(args.prompt_tokenizer_path), trust_remote_code=True)
    chat_template = load_chat_template(args.chat_template_path)
    episodes = build_episodes(args)
    write_manifest(args, episodes, tokenizer, chat_template)
    if args.backend == "ar-vllm":
        rows = run_ar_vllm(args, episodes, tokenizer, chat_template)
        write_rows(args.out_dir, "ar-vllm", rows)
    elif args.backend == "ar-vllm-guided":
        rows = run_ar_vllm(args, episodes, tokenizer, chat_template, guided=True)
        write_rows(args.out_dir, "ar-vllm-guided", rows)
    elif args.backend == "diffusion":
        rows = run_diffusion(args, episodes, tokenizer, chat_template)
        write_rows(args.out_dir, args.diffusion_output_name, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
