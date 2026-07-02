#!/usr/bin/env python3
"""Paired multiturn FLARE eval for careful decode vs per-call waves.

The episodes are built from the leak-checked scale-up tool-call slice by
splitting each multi-call gold answer into sequential one-call turns, then
appending synthetic tool responses between sampled assistant turns.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from emit_tool_sensitive_sampler_schedule import schedule_for_record  # noqa: E402
from eval_fastdllm_toolcall_cases import (  # noqa: E402
    annotate_sampler_schedule,
    full_context_sample,
    load_model,
    parse_kind_set,
    resolve_single_token_ids,
    resolve_token_ids,
)
from eval_toolcall_jsonl import extract_tool_calls, score_tool_calls, tool_schema_by_name  # noqa: E402
from flare_hf_cache import FlarePrefixCache  # noqa: E402
from plan_tool_sensitive_blocks import add_token_spans, plan_text  # noqa: E402


DEFAULT_INPUT = ROOT / "data/toolcall_eval_native/flare_scaleup_native_58.jsonl"
DEFAULT_BASE = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_ADAPTER = ROOT / "runs/flare_redesign_run1_copy_grounded_qwen35_9b"
DEFAULT_OUT_DIR = ROOT / "runs/agentic_eval/multiturn_percall_waves_tau095"
TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*.*?\s*</tool_call>", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--no-merge-adapter", action="store_true", default=True)
    parser.add_argument("--merge-adapter", dest="no_merge_adapter", action="store_false")
    parser.add_argument("--episode-limit", type=int, default=12)
    parser.add_argument("--min-turns", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--small-block-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--max-extra-tokens", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--skip-warmup", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def split_tool_call_blocks(text: str) -> list[str]:
    return [match.group(0).strip() for match in TOOL_CALL_BLOCK_RE.finditer(text or "")]


def content_or_json(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def render_prompt(tokenizer, messages: list[dict], tools: list[dict]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    if tools:
        kwargs["tools"] = tools
    return tokenizer.apply_chat_template(messages, **kwargs)


def tool_response_message(payload: Any) -> dict:
    return {"role": "user", "content": "<tool_response>\n" + content_or_json(payload) + "\n</tool_response>"}


def trim_after_first_tool_call(text: str) -> str:
    end = text.find("</tool_call>")
    if end < 0:
        return text.strip()
    return text[: end + len("</tool_call>")].strip()


def build_episodes(args: argparse.Namespace) -> list[dict]:
    episodes = []
    passthrough_keys = (
        "turn_user_messages",
        "source_family",
        "source_dataset",
        "source_license",
        "source_row_idx",
        "public_eval_hash",
        "leak_check",
    )
    for row in load_jsonl(args.input_jsonl):
        blocks = split_tool_call_blocks(row.get("gold_assistant") or "")
        if len(blocks) < args.min_turns or len(blocks) > args.max_turns:
            continue
        episode = {
            "episode_idx": len(episodes),
            "id": row.get("id") or str(len(episodes)),
            "source": row.get("source"),
            "prompt_messages": row.get("prompt_messages") or row.get("messages") or [],
            "tools": row.get("tools") or [],
            "gold_blocks": blocks,
            "gold_assistant": row.get("gold_assistant") or "",
        }
        for key in passthrough_keys:
            if key in row:
                episode[key] = row[key]
        episodes.append(episode)
        if args.episode_limit and len(episodes) >= args.episode_limit:
            break
    if not episodes:
        raise SystemExit(
            f"no episodes found in {args.input_jsonl} with {args.min_turns}-{args.max_turns} tool-call turns"
        )
    return episodes


def build_schedule(tokenizer, gold_text: str) -> tuple[list[dict], dict, float]:
    start = time.time()
    record = {
        "id": "turn",
        "source": "multiturn_split",
        "text_field": "gold_assistant",
        "text": gold_text,
        "text_chars": len(gold_text),
        "tool_call_count": len(split_tool_call_blocks(gold_text)),
        "segments": plan_text(gold_text, max_prose_chars=512, max_json_structure_chars=96),
    }
    add_token_spans(record, tokenizer, include_token_ids=True)
    schedule_args = SimpleNamespace(
        prose_block_tokens=128,
        argument_value_block_tokens=8,
        json_structure_block_tokens=4,
        tiny_block_tokens=1,
        include_token_ids=True,
    )
    schedule = annotate_sampler_schedule(schedule_for_record(record, schedule_args))
    return schedule, record, time.time() - start


def make_gen_args(
    args: argparse.Namespace,
    *,
    condition: str,
    prefix_cache: FlarePrefixCache,
    mask_id: int,
    stop_token_id: int,
    stop_token_ids: list[int],
    argument_boundary_token_ids: list[int],
    argument_newline_token_ids: list[int],
) -> SimpleNamespace:
    base = {
        "block_size": args.block_size,
        "small_block_size": args.small_block_size,
        "max_new_tokens": args.max_new_tokens,
        "max_new_tokens_cap": args.max_new_tokens,
        "threshold": args.threshold,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "mask_id": mask_id,
        "stop_token_id": stop_token_id,
        "stop_token_ids": stop_token_ids,
        "use_block_cache": True,
        "full_context_sampling": True,
        "fresh_generation_blocks": False,
        "denoise_logit_mode": "flare_shift",
        "force_argument_boundary_target_tokens": False,
        "constrain_argument_candidate_tokens": False,
        "force_selected_candidate_tokens": False,
        "force_best_candidate_sequence": False,
        "guard_tool_value_candidates": False,
        "force_best_tool_name_sequence": False,
        "guard_tool_name_candidates": False,
        "ban_argument_boundary_tokens": False,
        "ban_argument_json_boundary_tokens": False,
        "ban_argument_newline_tokens": False,
        "guard_tool_call_mode": False,
        "guard_tool_json_prefix": False,
        "json_prefix_guard_kinds": {"tool_tag", "json_structure", "json_key", "tool_name", "argument_value"},
        "json_prefix_guard_topk": 32,
        "json_prefix_guard_left_to_right": True,
        "json_prefix_guard_target_fallback": False,
        "live_tool_json_grammar": False,
        "live_tool_json_topk": 128,
        "force_schedule_token_kinds": set(),
        "argument_boundary_token_ids": argument_boundary_token_ids,
        "argument_newline_token_ids": argument_newline_token_ids,
        "_argument_boundary_target_cache": {},
        "_flare_prefix_cache": prefix_cache,
    }
    if condition == "baseline_careful":
        base.update(
            {
                "parallel_commit_threshold": 0.95,
                "parallel_commit_kinds": set(),
                "two_wave_tool_schedule": False,
                "two_wave_per_call": False,
                "two_wave_wave1_mode": "confidence",
                "two_wave_grammar_forced_only": False,
                "two_wave_wave1_threshold": 0.0,
                "two_wave_wave1_kinds": parse_kind_set("tool_tag,json_structure,json_key,tool_name,prose"),
                "two_wave_grammar_project_kinds": parse_kind_set("tool_tag,json_structure,json_key,prose"),
                "two_wave_wave2_threshold": 0.95,
                "two_wave_wave2_kinds": parse_kind_set("argument_value"),
            }
        )
    elif condition == "percall_waves_tau095":
        base.update(
            {
                "parallel_commit_threshold": None,
                "parallel_commit_kinds": set(),
                "two_wave_tool_schedule": True,
                "two_wave_per_call": True,
                "two_wave_wave1_mode": "grammar_projected",
                "two_wave_grammar_forced_only": False,
                "two_wave_wave1_threshold": 0.0,
                "two_wave_wave1_kinds": parse_kind_set("tool_tag,json_structure,json_key,tool_name,prose"),
                "two_wave_grammar_project_kinds": parse_kind_set("tool_tag,json_structure,json_key,prose"),
                "two_wave_wave2_threshold": 0.95,
                "two_wave_wave2_kinds": parse_kind_set("argument_value"),
            }
        )
    else:
        raise ValueError(f"unknown condition {condition!r}")
    return SimpleNamespace(**base)


def synthetic_tool_result(assistant_text: str, gold_block: str, episode_id: str, turn_idx: int, tools: list[dict]) -> dict:
    calls, invalid = extract_tool_calls(assistant_text)
    gold_calls, _ = extract_tool_calls(gold_block)
    if invalid or not calls:
        return {
            "ok": False,
            "error": "assistant_tool_call_invalid_or_missing",
            "invalid_tool_call_count": int(invalid),
            "expected_tool": gold_calls[0]["name"] if gold_calls else None,
        }
    call = calls[0]
    schemas = tool_schema_by_name(tools)
    known = call.get("name") in schemas if schemas else True
    return {
        "ok": bool(known),
        "tool": call.get("name"),
        "arguments": call.get("arguments") or {},
        "result_id": f"{episode_id}_turn_{turn_idx}",
        "summary": f"synthetic result for {call.get('name')}",
    }


def numeric_event_sum(rows: list[dict], key: str) -> float:
    return sum(float((row.get("sampler_schedule_events") or {}).get(key) or 0.0) for row in rows)


def bool_sum(rows: list[dict], key: str) -> int:
    return sum(int(bool(row.get(key))) for row in rows)


def summarize_rows(rows: list[dict]) -> dict:
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)

    condition_summaries = {}
    for condition, condition_rows in sorted(by_condition.items()):
        episodes = sorted({row["episode_id"] for row in condition_rows})
        rows_by_episode: dict[str, list[dict]] = defaultdict(list)
        for row in condition_rows:
            rows_by_episode[row["episode_id"]].append(row)
        exact_arg_episodes = sum(
            int(bool(episode_rows) and all(bool(row.get("exact_arguments")) for row in episode_rows))
            for episode_rows in rows_by_episode.values()
        )
        exact_seq_episodes = sum(
            int(bool(episode_rows) and all(bool(row.get("exact_tool_sequence")) for row in episode_rows))
            for episode_rows in rows_by_episode.values()
        )
        generated_tokens = sum(int(row.get("generated_token_count") or 0) for row in condition_rows)
        denoise_forwards = numeric_event_sum(condition_rows, "denoise_forwards_total")
        timing_totals = Counter()
        cache_hits = 0
        prefix_reused_tokens = 0
        for row in condition_rows:
            for key, value in (row.get("flare_timing_stats") or {}).items():
                timing_totals[key] += float(value or 0.0)
            cache = row.get("flare_cache_stats") or {}
            cache_hits += int(bool(cache.get("prefix_cache_hit")))
            prefix_reused_tokens += int(cache.get("prefix_cache_reused_tokens") or 0)
        timed = (
            timing_totals["cache_reset_seconds"]
            + timing_totals["decode_loop_seconds"]
            + timing_totals["cache_finish_seconds"]
        )
        condition_summaries[condition] = {
            "episodes": len(episodes),
            "turns": len(condition_rows),
            "valid_tool_json": bool_sum(condition_rows, "valid_tool_json"),
            "exact_tool_sequence": bool_sum(condition_rows, "exact_tool_sequence"),
            "exact_arguments": bool_sum(condition_rows, "exact_arguments"),
            "episode_exact_arguments_all_turns": exact_arg_episodes,
            "episode_exact_sequence_all_turns": exact_seq_episodes,
            "all_schema_valid": bool_sum(condition_rows, "all_schema_valid"),
            "turn_wall_seconds": sum(float(row.get("turn_wall_seconds") or 0.0) for row in condition_rows),
            "sample_seconds": sum(float(row.get("sample_seconds") or 0.0) for row in condition_rows),
            "schedule_build_seconds": sum(float(row.get("schedule_build_seconds") or 0.0) for row in condition_rows),
            "prompt_tokenize_seconds": sum(float(row.get("prompt_tokenize_seconds") or 0.0) for row in condition_rows),
            "generated_tokens": generated_tokens,
            "generated_tokens_per_turn": generated_tokens / len(condition_rows) if condition_rows else 0.0,
            "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in condition_rows),
            "denoise_forwards_total": denoise_forwards,
            "denoise_forwards_per_turn": denoise_forwards / len(condition_rows) if condition_rows else 0.0,
            "blended_tpf": generated_tokens / denoise_forwards if denoise_forwards else 0.0,
            "prefix_cache_hit_turns": cache_hits,
            "prefix_cache_eligible_followup_turns": max(0, len(condition_rows) - len(episodes)),
            "prefix_cache_hit_rate_eligible": (
                cache_hits / max(1, len(condition_rows) - len(episodes))
                if len(condition_rows) > len(episodes)
                else 0.0
            ),
            "prefix_cache_reused_tokens": prefix_reused_tokens,
            "flare_timing_totals": dict(timing_totals),
            "prefill_share_of_timed": timing_totals["cache_reset_seconds"] / timed if timed else 0.0,
            "decode_share_of_timed": timing_totals["decode_loop_seconds"] / timed if timed else 0.0,
            "value_force_counters": {
                "forced_schedule_token_visits": numeric_event_sum(condition_rows, "forced_schedule_token_visits"),
                "tool_value_candidate_force_token_visits": numeric_event_sum(
                    condition_rows, "tool_value_candidate_force_token_visits"
                ),
                "wave1_value_tokens": numeric_event_sum(condition_rows, "two_wave_wave1_value_tokens"),
                "wave2_forced_tokens": numeric_event_sum(condition_rows, "two_wave_wave2_forced_tokens"),
                "parallel_commit_forced_tokens": numeric_event_sum(condition_rows, "parallel_commit_forced_tokens"),
                "wave1_projected_tokens": numeric_event_sum(condition_rows, "two_wave_wave1_projected_tokens"),
                "wave1_forced_tokens": numeric_event_sum(condition_rows, "two_wave_wave1_forced_tokens"),
            },
            "grammar_projection_seconds": numeric_event_sum(condition_rows, "two_wave_wave1_projection_seconds"),
        }

    paired = {}
    if {"baseline_careful", "percall_waves_tau095"} <= set(by_condition):
        baseline_by_turn = {
            (row["episode_id"], row["turn_idx"]): row for row in by_condition["baseline_careful"]
        }
        percall_by_turn = {
            (row["episode_id"], row["turn_idx"]): row for row in by_condition["percall_waves_tau095"]
        }
        shared = sorted(set(baseline_by_turn) & set(percall_by_turn))
        paired = {
            "paired_turns": len(shared),
            "exact_arguments_delta": sum(
                int(bool(percall_by_turn[key].get("exact_arguments")))
                - int(bool(baseline_by_turn[key].get("exact_arguments")))
                for key in shared
            ),
            "percall_only_exact_arguments": sum(
                int(bool(percall_by_turn[key].get("exact_arguments")) and not baseline_by_turn[key].get("exact_arguments"))
                for key in shared
            ),
            "baseline_only_exact_arguments": sum(
                int(bool(baseline_by_turn[key].get("exact_arguments")) and not percall_by_turn[key].get("exact_arguments"))
                for key in shared
            ),
        }
        baseline_by_episode: dict[str, list[dict]] = defaultdict(list)
        percall_by_episode: dict[str, list[dict]] = defaultdict(list)
        for row in by_condition["baseline_careful"]:
            baseline_by_episode[row["episode_id"]].append(row)
        for row in by_condition["percall_waves_tau095"]:
            percall_by_episode[row["episode_id"]].append(row)
        shared_episodes = sorted(set(baseline_by_episode) & set(percall_by_episode))
        paired["paired_episodes"] = len(shared_episodes)
        paired["episode_exact_arguments_delta"] = sum(
            int(all(bool(row.get("exact_arguments")) for row in percall_by_episode[episode_id]))
            - int(all(bool(row.get("exact_arguments")) for row in baseline_by_episode[episode_id]))
            for episode_id in shared_episodes
        )
        paired["percall_only_exact_argument_episodes"] = sum(
            int(
                all(bool(row.get("exact_arguments")) for row in percall_by_episode[episode_id])
                and not all(bool(row.get("exact_arguments")) for row in baseline_by_episode[episode_id])
            )
            for episode_id in shared_episodes
        )
        paired["baseline_only_exact_argument_episodes"] = sum(
            int(
                all(bool(row.get("exact_arguments")) for row in baseline_by_episode[episode_id])
                and not all(bool(row.get("exact_arguments")) for row in percall_by_episode[episode_id])
            )
            for episode_id in shared_episodes
        )
        b = condition_summaries["baseline_careful"]
        p = condition_summaries["percall_waves_tau095"]
        paired["turn_wall_speedup"] = (
            b["turn_wall_seconds"] / p["turn_wall_seconds"] if p["turn_wall_seconds"] else 0.0
        )
        paired["sample_speedup"] = b["sample_seconds"] / p["sample_seconds"] if p["sample_seconds"] else 0.0

    return {"conditions": condition_summaries, "paired": paired}


def run_turn(
    model,
    tokenizer,
    gen_args: SimpleNamespace,
    *,
    condition: str,
    episode: dict,
    turn_idx: int,
    messages: list[dict],
    gold_block: str,
    stop_token_ids: list[int],
    max_extra_tokens: int,
) -> tuple[dict, str, dict]:
    turn_wall_start = time.time()
    schedule, schedule_record, schedule_build_seconds = build_schedule(tokenizer, gold_block)
    prompt_start = time.time()
    prompt = render_prompt(tokenizer, messages, episode["tools"])
    prompt_input_ids = tokenizer([prompt], return_tensors="pt").input_ids.to("cuda")
    prompt_tokenize_seconds = time.time() - prompt_start
    gold_token_count = int(schedule_record.get("token_count") or 0)
    gen_args.max_new_tokens = min(int(gen_args.max_new_tokens_cap), max(16, gold_token_count + int(max_extra_tokens)))
    gen_args.stop_token_ids = stop_token_ids
    previous_live_tool_schemas = getattr(gen_args, "_live_tool_schemas", None)
    gen_args._live_tool_schemas = tool_schema_by_name(episode["tools"])

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    sample_start = time.time()
    try:
        with torch.no_grad():
            generated = full_context_sample(
                model,
                prompt_input_ids,
                tokenizer,
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
    sample_seconds = time.time() - sample_start

    new_ids = generated[prompt_input_ids.shape[1] :]
    mask_count = int((new_ids == gen_args.mask_id).sum().item())
    generated_token_count = int((new_ids != gen_args.mask_id).sum().item())
    assistant_text = trim_after_first_tool_call(tokenizer.decode(new_ids, skip_special_tokens=True).strip())
    metrics = score_tool_calls(assistant_text, episode["tools"], gold_block)
    tool_payload = synthetic_tool_result(assistant_text, gold_block, episode["id"], turn_idx, episode["tools"])
    turn_wall_seconds = time.time() - turn_wall_start

    row = {
        "condition": condition,
        "episode_idx": episode["episode_idx"],
        "episode_id": episode["id"],
        "source": episode.get("source"),
        "turn_idx": turn_idx,
        "turns_in_episode": len(episode["gold_blocks"]),
        "status": "ok",
        "prompt_tokens": int(prompt_input_ids.shape[1]),
        "gold_token_count": gold_token_count,
        "generated_token_count": generated_token_count,
        "mask_count": mask_count,
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
        "turn_wall_seconds": turn_wall_seconds,
        "sample_seconds": sample_seconds,
        "schedule_build_seconds": schedule_build_seconds,
        "prompt_tokenize_seconds": prompt_tokenize_seconds,
        "sampler_schedule_events": getattr(gen_args, "_last_sampler_schedule_events", {}),
        "flare_cache_stats": getattr(gen_args, "_last_flare_cache_stats", {}),
        "flare_prefix_cache_stats": getattr(gen_args, "_last_flare_prefix_cache_stats", {}),
        "flare_timing_stats": getattr(gen_args, "_last_flare_timing_stats", {}),
    }
    return row, assistant_text, tool_payload


def run_condition(
    model,
    tokenizer,
    args: argparse.Namespace,
    episodes: list[dict],
    condition: str,
    *,
    mask_id: int,
    stop_token_id: int,
    stop_token_ids: list[int],
    argument_boundary_token_ids: list[int],
    argument_newline_token_ids: list[int],
) -> list[dict]:
    rows = []
    for episode in episodes:
        messages = [dict(message) for message in episode["prompt_messages"]]
        prefix_cache = FlarePrefixCache()
        gen_args = make_gen_args(
            args,
            condition=condition,
            prefix_cache=prefix_cache,
            mask_id=mask_id,
            stop_token_id=stop_token_id,
            stop_token_ids=stop_token_ids,
            argument_boundary_token_ids=argument_boundary_token_ids,
            argument_newline_token_ids=argument_newline_token_ids,
        )
        for turn_idx, gold_block in enumerate(episode["gold_blocks"]):
            row, assistant_text, tool_payload = run_turn(
                model,
                tokenizer,
                gen_args,
                condition=condition,
                episode=episode,
                turn_idx=turn_idx,
                messages=messages,
                gold_block=gold_block,
                stop_token_ids=stop_token_ids,
                max_extra_tokens=args.max_extra_tokens,
            )
            rows.append(row)
            messages.append({"role": "assistant", "content": assistant_text})
            messages.append(tool_response_message(tool_payload))
            print(
                f"{condition} episode={episode['episode_idx']} turn={turn_idx + 1}/{len(episode['gold_blocks'])} "
                f"exact_args={int(bool(row['exact_arguments']))} "
                f"wall={row['turn_wall_seconds']:.3f}s cache_hit={int(bool((row.get('flare_cache_stats') or {}).get('prefix_cache_hit')))}",
                flush=True,
            )
    return rows


def write_report(out_dir: Path, args: argparse.Namespace, episodes: list[dict], rows: list[dict], summary: dict) -> None:
    b = summary["conditions"].get("baseline_careful", {})
    p = summary["conditions"].get("percall_waves_tau095", {})
    paired = summary.get("paired") or {}
    speedup = paired.get("turn_wall_speedup") or 0.0
    exact_delta = paired.get("exact_arguments_delta") or 0
    lines = [
        "# FLARE Multiturn Per-Call Waves Eval",
        "",
        f"Slice: {len(episodes)} generated-history episodes, {sum(len(ep['gold_blocks']) for ep in episodes)} turns, from `{args.input_jsonl}`.",
        "Prompting: previous sampled assistant tool call plus synthetic `<tool_response>` is appended before the next turn.",
        "Stop: `</tool_call>` added as a stop token so each turn measures one tool call.",
        "",
        "| Condition | exact_args | episode exact | valid_json | model forwards/turn | gen tok/turn | blended TPF | sec/turn | prefix hits |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in [("Baseline careful", b), ("Per-call waves tau 0.95", p)]:
        turns = int(item.get("turns") or 0)
        episodes_n = int(item.get("episodes") or 0)
        lines.append(
            f"| {name} | {int(item.get('exact_arguments') or 0)}/{turns} "
            f"| {int(item.get('episode_exact_arguments_all_turns') or 0)}/{episodes_n} "
            f"| {int(item.get('valid_tool_json') or 0)}/{turns} "
            f"| {float(item.get('denoise_forwards_per_turn') or 0.0):.3f} "
            f"| {float(item.get('generated_tokens_per_turn') or 0.0):.3f} "
            f"| {float(item.get('blended_tpf') or 0.0):.3f} "
            f"| {float(item.get('turn_wall_seconds') or 0.0) / turns if turns else 0.0:.3f} "
            f"| {int(item.get('prefix_cache_hit_turns') or 0)}/{int(item.get('prefix_cache_eligible_followup_turns') or 0)} |"
        )
    lines.extend(
        [
            "",
            "## Headline",
            "",
            f"- End-to-end turn/episode wall speedup: {speedup:.3f}x",
            f"- Paired exact-args delta (per-call - baseline): {exact_delta} / {int(paired.get('paired_turns') or 0)} turns",
            f"- Per-call only exact args: {int(paired.get('percall_only_exact_arguments') or 0)}; baseline only: {int(paired.get('baseline_only_exact_arguments') or 0)}",
            f"- Episode-level exact-args delta: {int(paired.get('episode_exact_arguments_delta') or 0)} / {int(paired.get('paired_episodes') or 0)} episodes",
            f"- Per-call only exact episodes: {int(paired.get('percall_only_exact_argument_episodes') or 0)}; baseline only: {int(paired.get('baseline_only_exact_argument_episodes') or 0)}",
            f"- TPF accounting: numerator is all generated visible tokens; denominator is model denoise forwards. Grammar-projected scaffold tokens count in generated tokens but consume no model forward.",
            f"- Baseline raw TPF: {int(b.get('generated_tokens') or 0)} generated tokens / {float(b.get('denoise_forwards_total') or 0.0):.0f} denoise forwards = {float(b.get('blended_tpf') or 0.0):.3f}",
            f"- Per-call raw TPF: {int(p.get('generated_tokens') or 0)} generated tokens / {float(p.get('denoise_forwards_total') or 0.0):.0f} denoise forwards = {float(p.get('blended_tpf') or 0.0):.3f}",
            f"- Baseline timed split: prefill/cache-reset {float(b.get('prefill_share_of_timed') or 0.0):.3f}, decode {float(b.get('decode_share_of_timed') or 0.0):.3f}",
            f"- Per-call timed split: prefill/cache-reset {float(p.get('prefill_share_of_timed') or 0.0):.3f}, decode {float(p.get('decode_share_of_timed') or 0.0):.3f}",
            f"- Schedule build overhead: baseline {float(b.get('schedule_build_seconds') or 0.0):.3f}s, per-call {float(p.get('schedule_build_seconds') or 0.0):.3f}s",
            f"- Grammar projection overhead in per-call: {float(p.get('grammar_projection_seconds') or 0.0):.3f}s",
            f"- Per-call value force counters: {json.dumps(p.get('value_force_counters') or {}, sort_keys=True)}",
            "",
            f"Full JSON summary: `{out_dir / 'summary.json'}`",
            f"Per-turn rows: `{out_dir / 'turns.jsonl'}`",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    os.environ["FASTDLLM_FLARE_GDN_ROUTE"] = "route_i"
    os.environ.setdefault("FASTDLLM_FLARE_TWO_STREAM", "1")
    os.environ.setdefault("FLARE_TWO_STREAM", "1")
    episodes = build_episodes(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(
        args.base_model,
        args.adapter if args.adapter and args.adapter.exists() else None,
        merge_adapter=not args.no_merge_adapter,
        tokenizer_path=args.tokenizer_path,
    )
    model.eval()
    if hasattr(model, "config"):
        setattr(model.config, "bd_size", int(args.block_size))
    mask_id, stop_token_id, base_stop_token_ids = resolve_token_ids(model, tokenizer)
    tool_close_ids = tokenizer("</tool_call>", add_special_tokens=False).input_ids
    stop_token_ids = list(dict.fromkeys([int(x) for x in base_stop_token_ids + tool_close_ids]))
    argument_boundary_token_ids = resolve_single_token_ids(
        tokenizer, ["<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>"]
    )
    argument_newline_token_ids = resolve_single_token_ids(tokenizer, ["\n", "\n\n"])

    rows = []
    conditions = ["baseline_careful", "percall_waves_tau095"]
    for condition in conditions:
        condition_rows = run_condition(
            model,
            tokenizer,
            args,
            episodes,
            condition,
            mask_id=mask_id,
            stop_token_id=stop_token_id,
            stop_token_ids=stop_token_ids,
            argument_boundary_token_ids=argument_boundary_token_ids,
            argument_newline_token_ids=argument_newline_token_ids,
        )
        rows.extend(condition_rows)

    out_jsonl = args.out_dir / "turns.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize_rows(rows)
    summary.update(
        {
            "input_jsonl": str(args.input_jsonl),
            "out_dir": str(args.out_dir),
            "base_model": str(args.base_model),
            "adapter": str(args.adapter),
            "merge_adapter": not args.no_merge_adapter,
            "episode_count": len(episodes),
            "turn_count": sum(len(ep["gold_blocks"]) for ep in episodes),
            "episode_ids": [ep["id"] for ep in episodes],
            "block_size": args.block_size,
            "small_block_size": args.small_block_size,
            "max_new_tokens": args.max_new_tokens,
            "max_extra_tokens": args.max_extra_tokens,
            "stop_token_ids": stop_token_ids,
            "tool_close_token_ids": tool_close_ids,
            "mask_id": int(mask_id),
        }
    )
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(args.out_dir, args, episodes, rows, summary)
    print(json.dumps(summary["paired"], indent=2), flush=True)
    print(f"wrote {args.out_dir / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
