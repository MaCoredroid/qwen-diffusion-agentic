#!/usr/bin/env python3
"""Matched multi-turn eval for clean hybrid constrained serving.

Hybrid policy:
- Bulk-commit only Qwen-native grammar tokens that are truly forced by the FSM.
- Use one denoise forward per non-forced token, including every parameter-value
  token and close timing inside value spans.
- Never project gold values.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from eval_fastdllm_toolcall_cases import (  # noqa: E402
    contiguous_decoded_prefix,
    grammar_legal_candidate_token_ids,
    live_tool_json_top_token,
    load_model,
    native_tool_prefix_can_stop,
    qwen_native_inside_parameter_value,
    resolve_token_ids,
    sample_with_top_p,
    tool_json_live_prefix_active,
)
from eval_flare_northstar_matched import (  # noqa: E402
    DEFAULT_AR_MODEL,
    DEFAULT_CHAT_TEMPLATE,
    DEFAULT_DIFFUSION_ADAPTER,
    DEFAULT_DIFFUSION_BASE,
    DEFAULT_INPUT,
    build_episodes,
    decode_text,
    load_chat_template,
    next_turn_user_message,
    render_matched_prompt,
    row_from_generation,
    sha256_text,
    tool_response_suffix,
    trim_scored_assistant,
    write_manifest,
    write_rows,
)
from eval_flare_stage1_ab_diffusion import set_block_size  # noqa: E402
from eval_toolcall_jsonl import tool_schema_by_name  # noqa: E402
from flare_hf_cache import FlarePrefixCache, RequestDiffusionState  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--episode-limit", type=int, default=20)
    parser.add_argument("--min-turns", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--prompt-tokenizer-path", type=Path, default=DEFAULT_AR_MODEL)
    parser.add_argument("--chat-template-path", type=Path, default=DEFAULT_CHAT_TEMPLATE)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_DIFFUSION_BASE)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_DIFFUSION_ADAPTER)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_AR_MODEL)
    parser.add_argument("--no-merge-adapter", action="store_true", default=True)
    parser.add_argument("--merge-adapter", dest="no_merge_adapter", action="store_false")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--grammar-topk", type=int, default=256)
    parser.add_argument("--diffusion-output-name", default="diffusion_hybrid_forced_grammar_seq_values")
    parser.add_argument("--seed", type=int, default=20260701)
    return parser.parse_args()


def summarize_backend(rows: list[dict]) -> dict:
    by_episode: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_episode[row["episode_id"]].append(row)
    turns = len(rows)
    wall = sum(float(row.get("turn_wall_seconds") or 0.0) for row in rows)
    generated_tokens = sum(int(row.get("generated_token_count") or 0) for row in rows)
    forwards = 0.0
    forced = 0
    value_tokens = 0
    structural_model_tokens = 0
    close_value_tokens = 0
    for row in rows:
        events = ((row.get("backend_meta") or {}).get("sampler_schedule_events") or {})
        forwards += float(events.get("denoise_forwards_total") or 0.0)
        forced += int(events.get("hybrid_forced_grammar_tokens") or 0)
        value_tokens += int(events.get("hybrid_model_value_tokens") or 0)
        structural_model_tokens += int(events.get("hybrid_model_structural_tokens") or 0)
        close_value_tokens += int(events.get("hybrid_value_close_timing_tokens") or 0)
    return {
        "episodes": len(by_episode),
        "turns": turns,
        "valid_tool_json": sum(int(bool(row.get("valid_tool_json"))) for row in rows),
        "valid_tool_call": sum(int(bool(row.get("valid_tool_call"))) for row in rows),
        "exact_tool_sequence": sum(int(bool(row.get("exact_tool_sequence"))) for row in rows),
        "exact_args": sum(int(bool(row.get("exact_arguments"))) for row in rows),
        "schema_ok": sum(int(bool(row.get("all_schema_valid"))) for row in rows),
        "required_args_present": sum(int(bool(row.get("all_required_args_present"))) for row in rows),
        "episode_exact": sum(
            int(all(bool(row.get("exact_arguments")) for row in episode_rows))
            for episode_rows in by_episode.values()
        ),
        "turn_wall_seconds_total": wall,
        "sec_per_turn": wall / turns if turns else 0.0,
        "generated_tokens": generated_tokens,
        "generated_tokens_per_turn": generated_tokens / turns if turns else 0.0,
        "denoise_forwards_total": forwards,
        "forwards_per_turn": forwards / turns if turns else 0.0,
        "tokens_per_forward": generated_tokens / forwards if forwards else None,
        "hybrid_forced_grammar_tokens": forced,
        "hybrid_model_value_tokens": value_tokens,
        "hybrid_model_structural_tokens": structural_model_tokens,
        "hybrid_value_close_timing_tokens": close_value_tokens,
    }


def _sample_unconstrained_token(model, row_logits: torch.Tensor, *, top_p: float, temperature: float) -> int:
    sampled, _probs = sample_with_top_p(
        model,
        row_logits.reshape(1, 1, -1),
        top_p=top_p,
        temperature=temperature,
    )
    return int(sampled.reshape(-1)[0].item())


def _generated_text(tokenizer, output_ids: torch.Tensor, original_len: int) -> str:
    return tokenizer.decode(
        output_ids[0, original_len:].detach().cpu().tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def _maybe_advance_cache(model, output_ids: torch.Tensor, cache_state: RequestDiffusionState) -> None:
    while int(output_ids.shape[1]) - int(cache_state.block_start) >= int(cache_state.block_size):
        block = output_ids[:, cache_state.block_start : cache_state.block_start + cache_state.block_size]
        cache_state.advance(model, block)


@torch.inference_mode()
def sample_hybrid_clean(
    model,
    tokenizer,
    prompt_input_ids: torch.Tensor,
    *,
    block_size: int,
    max_new_tokens: int,
    mask_id: int,
    stop_token_ids: set[int],
    top_p: float,
    temperature: float,
    schemas: dict[str, Any],
    grammar_topk: int,
    prefix_cache: FlarePrefixCache,
) -> tuple[torch.Tensor, dict[str, Any]]:
    output_ids = prompt_input_ids.to("cuda")
    original_len = int(output_ids.shape[1])
    cache_state = RequestDiffusionState.reset(model, output_ids, int(block_size), prefix_cache=prefix_cache)
    tool_close_token_ids = {
        int(token_id)
        for token_id in tokenizer("</tool_call>", add_special_tokens=False).input_ids
    }
    special_stop_token_ids = set(int(token_id) for token_id in stop_token_ids) - tool_close_token_ids
    metrics: dict[str, Any] = {
        "block_size": int(block_size),
        "max_new_tokens": int(max_new_tokens),
        "denoise_forwards": 0,
        "forced_grammar_tokens": 0,
        "model_value_tokens": 0,
        "model_structural_tokens": 0,
        "value_close_timing_tokens": 0,
        "grammar_checked_tokens": 0,
        "grammar_replacement_tokens": 0,
        "grammar_replacement_value_tokens": 0,
        "grammar_unsafe_fallback_tokens": 0,
        "stop_reason": None,
        "cache_stats": {},
    }

    while int(output_ids.shape[1]) - original_len < int(max_new_tokens):
        forced_this_round = 0
        while int(output_ids.shape[1]) - original_len < int(max_new_tokens):
            text = _generated_text(tokenizer, output_ids, original_len)
            if native_tool_prefix_can_stop(text, schemas=schemas):
                metrics["stop_reason"] = "complete_tool_call"
                prefix_cache.store(output_ids, cache_state)
                metrics["cache_stats"] = cache_state.stats()
                return output_ids.detach().cpu()[0], metrics
            if qwen_native_inside_parameter_value(text):
                break
            _maybe_advance_cache(model, output_ids, cache_state)
            mask = torch.full((output_ids.shape[0], 1), int(mask_id), dtype=torch.long, device=output_ids.device)
            sequence = torch.cat([output_ids, mask], dim=1)[0]
            abs_idx = int(sequence.numel()) - 1
            legal = grammar_legal_candidate_token_ids(
                tokenizer,
                sequence,
                original_len,
                abs_idx,
                int(mask_id),
                schemas,
                "qwen_native",
            )
            if len(legal) != 1:
                break
            token_id = int(legal[0])
            output_ids = torch.cat(
                [output_ids, torch.tensor([[token_id]], dtype=torch.long, device=output_ids.device)],
                dim=1,
            )
            metrics["forced_grammar_tokens"] += 1
            forced_this_round += 1
        if metrics.get("stop_reason"):
            break

        _maybe_advance_cache(model, output_ids, cache_state)
        mask = torch.full((output_ids.shape[0], 1), int(mask_id), dtype=torch.long, device=output_ids.device)
        x_t = torch.cat([output_ids, mask], dim=1)
        logits = cache_state.shifted_active_logits(model, x_t)[:, -1, :].clone().float()
        logits[:, int(mask_id)] = torch.finfo(logits.dtype).min
        row_logits = logits[0]
        text = _generated_text(tokenizer, output_ids, original_len)
        in_value = bool(qwen_native_inside_parameter_value(text))
        can_stop = native_tool_prefix_can_stop(text, schemas=schemas)
        if not can_stop:
            for stop_id in special_stop_token_ids:
                if 0 <= int(stop_id) < row_logits.shape[-1]:
                    row_logits[int(stop_id)] = torch.finfo(row_logits.dtype).min
        metrics["denoise_forwards"] += 1
        grammar_active = bool(tool_json_live_prefix_active(text))
        original_top = int(torch.argmax(row_logits).item())
        replacement = False
        if grammar_active:
            metrics["grammar_checked_tokens"] += 1
            if original_top in stop_token_ids and can_stop:
                token_id = original_top
                safe = True
            else:
                sequence = x_t[0].clone()
                token_id, safe = live_tool_json_top_token(
                    tokenizer,
                    sequence,
                    row_logits,
                    original_len,
                    int(sequence.numel()) - 1,
                    int(mask_id),
                    int(grammar_topk),
                    schemas=schemas,
                )
            replacement = int(token_id) != original_top
            if replacement:
                metrics["grammar_replacement_tokens"] += 1
            if not safe:
                metrics["grammar_unsafe_fallback_tokens"] += 1
        else:
            token_id = _sample_unconstrained_token(
                model,
                row_logits,
                top_p=float(top_p),
                temperature=float(temperature),
            )

        if in_value:
            metrics["model_value_tokens"] += 1
            piece = tokenizer.decode([int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            if "\n<" in piece or "</parameter" in piece or piece == "\n":
                metrics["value_close_timing_tokens"] += 1
            if replacement:
                metrics["grammar_replacement_value_tokens"] += 1
        else:
            metrics["model_structural_tokens"] += 1

        output_ids = torch.cat(
            [output_ids, torch.tensor([[int(token_id)]], dtype=torch.long, device=output_ids.device)],
            dim=1,
        )

    prefix_cache.store(output_ids, cache_state)
    metrics["cache_stats"] = cache_state.stats()
    if metrics["stop_reason"] is None:
        metrics["stop_reason"] = "max_new_tokens"
    return output_ids.detach().cpu()[0], metrics


def run_hybrid(args: argparse.Namespace, episodes: list[dict], tokenizer, chat_template: str | None) -> list[dict]:
    model, model_tokenizer = load_model(
        args.base_model,
        args.adapter if args.adapter and args.adapter.exists() else None,
        merge_adapter=not args.no_merge_adapter,
        tokenizer_path=args.tokenizer_path,
    )
    model.eval()
    set_block_size(model, int(args.block_size))
    mask_id, _stop_token_id, base_stop_token_ids = resolve_token_ids(model, model_tokenizer)
    tool_close_ids = model_tokenizer("</tool_call>", add_special_tokens=False).input_ids
    stop_token_ids = set(int(item) for item in base_stop_token_ids + tool_close_ids)
    rows = []
    backend_name = args.diffusion_output_name
    for episode in episodes:
        messages = [dict(message) for message in episode["prompt_messages"]]
        prompt = render_matched_prompt(model_tokenizer, messages, episode["tools"], chat_template)
        prefix_cache = FlarePrefixCache()
        for turn_idx, gold_block in enumerate(episode["gold_blocks"]):
            prompt_input_ids = model_tokenizer([prompt], return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.time()
            output_ids, sampler_metrics = sample_hybrid_clean(
                model,
                model_tokenizer,
                prompt_input_ids,
                block_size=int(args.block_size),
                max_new_tokens=int(args.max_new_tokens),
                mask_id=int(mask_id),
                stop_token_ids=stop_token_ids,
                top_p=float(args.top_p),
                temperature=float(args.temperature),
                schemas=tool_schema_by_name(episode["tools"]),
                grammar_topk=int(args.grammar_topk),
                prefix_cache=prefix_cache,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            turn_wall_seconds = time.time() - start
            new_ids = output_ids[int(prompt_input_ids.shape[1]) :]
            history_text = decode_text(model_tokenizer, new_ids)
            assistant_text = trim_scored_assistant(history_text)
            denoise_forwards = int(sampler_metrics.get("denoise_forwards") or 0)
            schedule_events = {
                "denoise_forwards_total": denoise_forwards,
                "hybrid_model_forwards": denoise_forwards,
                "hybrid_forced_grammar_tokens": int(sampler_metrics.get("forced_grammar_tokens") or 0),
                "hybrid_model_value_tokens": int(sampler_metrics.get("model_value_tokens") or 0),
                "hybrid_model_structural_tokens": int(sampler_metrics.get("model_structural_tokens") or 0),
                "hybrid_value_close_timing_tokens": int(sampler_metrics.get("value_close_timing_tokens") or 0),
                "hybrid_grammar_checked_tokens": int(sampler_metrics.get("grammar_checked_tokens") or 0),
                "hybrid_grammar_replacement_tokens": int(sampler_metrics.get("grammar_replacement_tokens") or 0),
                "hybrid_grammar_replacement_value_tokens": int(
                    sampler_metrics.get("grammar_replacement_value_tokens") or 0
                ),
                "hybrid_grammar_unsafe_fallback_tokens": int(
                    sampler_metrics.get("grammar_unsafe_fallback_tokens") or 0
                ),
                "parallel_commit_value_tokens": int(sampler_metrics.get("model_value_tokens") or 0),
                "parallel_commit_forced_tokens": 0,
                "two_wave_wave1_projected_tokens": 0,
                "two_wave_wave1_value_tokens": 0,
                "two_wave_wave2_value_tokens": 0,
                "two_wave_wave2_forced_tokens": 0,
            }
            row = row_from_generation(
                backend=backend_name,
                episode=episode,
                turn_idx=turn_idx,
                prompt=prompt,
                tools=episode["tools"],
                gold_block=gold_block,
                assistant_text=assistant_text,
                prompt_tokens=int(prompt_input_ids.shape[1]),
                generated_tokens=int(new_ids.numel()),
                turn_wall_seconds=turn_wall_seconds,
                schedule_build_seconds=0.0,
                backend_meta={
                    "sampler_schedule_events": schedule_events,
                    "hybrid_sampler_metrics": sampler_metrics,
                    "flare_cache_stats": sampler_metrics.get("cache_stats") or {},
                    "max_new_tokens": int(args.max_new_tokens),
                    "block_size": int(args.block_size),
                    "grammar_topk": int(args.grammar_topk),
                },
            )
            row["assistant_history_sha256"] = sha256_text(history_text)
            row["generated_token_ids"] = [int(token_id) for token_id in new_ids.detach().cpu().tolist()]
            next_user = next_turn_user_message(episode, turn_idx + 1)
            row["next_user_message_sha256"] = sha256_text(next_user) if next_user is not None else None
            rows.append(row)
            prompt = prompt + history_text + tool_response_suffix(row["tool_response_payload"], next_user)
            print(
                f"hybrid episode={episode['episode_idx']} turn={turn_idx + 1}/{len(episode['gold_blocks'])} "
                f"exact_args={int(bool(row['exact_arguments']))} valid={int(bool(row['valid_tool_call']))} "
                f"forwards={denoise_forwards} forced={schedule_events['hybrid_forced_grammar_tokens']} "
                f"value_tokens={schedule_events['hybrid_model_value_tokens']} wall={turn_wall_seconds:.3f}s",
                flush=True,
            )
    return rows


def main() -> int:
    args = parse_args()
    args.ar_model_path = args.prompt_tokenizer_path
    args.ar_served_model = "not_used_hybrid_clean"
    args.ar_base_url = "not_used_hybrid_clean"
    args.diffusion_condition = "hybrid_forced_grammar_seq_values"
    args.diffusion_structural_only = True
    torch.manual_seed(int(args.seed))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompt_tokenizer = AutoTokenizer.from_pretrained(str(args.prompt_tokenizer_path), trust_remote_code=True)
    chat_template = load_chat_template(args.chat_template_path)
    episodes = build_episodes(args)
    manifest = write_manifest(args, episodes, prompt_tokenizer, chat_template)
    manifest["diffusion"]["decode"] = "hybrid_forced_grammar_bulk_seq_values"
    manifest["diffusion"]["condition"] = "hybrid_forced_grammar_seq_values"
    manifest["diffusion"]["block_size"] = int(args.block_size)
    manifest["diffusion"]["grammar_topk"] = int(args.grammar_topk)
    (args.out_dir / "fairness_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rows = run_hybrid(args, episodes, prompt_tokenizer, chat_template)
    write_rows(args.out_dir, args.diffusion_output_name, rows)
    summary = summarize_backend(rows)
    summary.update(
        {
            "backend": args.diffusion_output_name,
            "adapter": str(args.adapter),
            "base_model": str(args.base_model),
            "input_jsonl": str(args.input_jsonl),
            "episode_set_hash": manifest.get("episode_set_hash"),
            "block_size": int(args.block_size),
            "grammar_topk": int(args.grammar_topk),
            "sampler": (
                "hybrid clean serving: truly-forced Qwen-native grammar tokens bulk-committed "
                "without denoise forwards; every non-forced/value token decoded sequentially with one denoise forward"
            ),
        }
    )
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
