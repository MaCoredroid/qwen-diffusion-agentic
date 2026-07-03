#!/usr/bin/env python3
"""Matched multi-turn eval with fixed-K schedule-gated block diffusion.

This is a diagnostic companion to eval_flare_northstar_matched.py. Here K means
denoise forwards per generated block, so a 32-token block at K=8 has nominal
4 generated tokens per forward. It intentionally does not use the confidence-run
parallel commit path.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from eval_fastdllm_toolcall_cases import (  # noqa: E402
    contiguous_decoded_prefix,
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
from measure_block_quality_curve import (  # noqa: E402
    first_stop_offset,
    sample_fixed_k_block_diffusion,
    shifted_full_context_logits,
    sync_cuda,
    visible_count_for_step,
)


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
    parser.add_argument("--denoise-steps", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--diffusion-output-name", default="diffusion_schedule_fixedk")
    parser.add_argument("--constrained-grammar", action="store_true")
    parser.add_argument("--grammar-topk", type=int, default=256)
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
    for row in rows:
        events = ((row.get("backend_meta") or {}).get("sampler_schedule_events") or {})
        forwards += float(events.get("denoise_forwards_total") or 0.0)
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
    }


def _sample_unconstrained_token(model, row_logits: torch.Tensor, *, top_p: float, temperature: float) -> tuple[int, float]:
    sampled, probs = sample_with_top_p(
        model,
        row_logits.reshape(1, 1, -1),
        top_p=top_p,
        temperature=temperature,
    )
    token_id = int(sampled.reshape(-1)[0].item())
    prob = float(probs.reshape(-1, probs.shape[-1])[0, token_id].detach().cpu().item())
    return token_id, prob


def _token_probability(row_logits: torch.Tensor, token_id: int, *, temperature: float) -> float:
    if float(temperature) > 0:
        probs = torch.softmax(row_logits / float(temperature), dim=-1)
    else:
        probs = torch.softmax(row_logits, dim=-1)
    return float(probs[int(token_id)].detach().cpu().item())


@torch.inference_mode()
def sample_fixed_k_constrained_grammar(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    *,
    block_size: int,
    denoise_steps: int,
    max_new_tokens: int,
    mask_id: int,
    stop_token_ids: set[int],
    top_p: float,
    temperature: float,
    schemas: dict[str, Any],
    grammar_topk: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    output_ids = input_ids.unsqueeze(0).to("cuda")
    original_len = int(output_ids.shape[1])
    metrics: dict[str, Any] = {
        "block_size": int(block_size),
        "denoise_steps_per_block": int(denoise_steps),
        "max_new_tokens": int(max_new_tokens),
        "denoise_forwards": 0,
        "denoise_seconds": 0.0,
        "blocks": [],
        "stop_offset": None,
        "stop_token_id": None,
        "unresolved_masks": 0,
        "visible_schedule": [],
        "committed_tokens": 0,
        "grammar_checked_tokens": 0,
        "grammar_replacement_tokens": 0,
        "grammar_unsafe_fallback_tokens": 0,
        "value_positions": 0,
        "model_chosen_value_tokens": 0,
        "grammar_replacement_value_tokens": 0,
        "structural_positions": 0,
        "gold_forced_tokens": 0,
        "projected_value_tokens": 0,
        "step_commits": [],
    }
    tool_close_token_ids = {
        int(token_id)
        for token_id in tokenizer("</tool_call>", add_special_tokens=False).input_ids
    }
    special_stop_token_ids = set(int(token_id) for token_id in stop_token_ids) - tool_close_token_ids

    while output_ids.shape[1] - original_len < max_new_tokens:
        remaining = max_new_tokens - (output_ids.shape[1] - original_len)
        block_pad = min(int(block_size), int(remaining))
        block_state = torch.full(
            (output_ids.shape[0], block_pad),
            int(mask_id),
            dtype=torch.long,
            device=output_ids.device,
        )
        committed = 0
        block_metrics: dict[str, Any] = {
            "block_pad": int(block_pad),
            "steps": int(denoise_steps),
            "step_visible": [],
            "step_committed": [],
            "step_conf_mean": [],
            "step_conf_min": [],
            "step_conf_max": [],
            "step_seconds": [],
        }

        for step_idx in range(int(denoise_steps)):
            x_t = torch.cat([output_ids, block_state], dim=1)
            sync_cuda()
            start = time.perf_counter()
            logits = shifted_full_context_logits(model, x_t, block_pad, mask_id)
            sync_cuda()
            seconds = time.perf_counter() - start

            target_visible = visible_count_for_step(block_pad, int(denoise_steps), step_idx)
            positions = list(range(committed, target_visible))
            chosen_probs: list[float] = []
            sequence = torch.cat([output_ids, block_state], dim=1)[0].clone()
            block_abs_start = int(output_ids.shape[1])

            for local_pos in positions:
                local_pos = int(local_pos)
                abs_idx = block_abs_start + local_pos
                row_logits = logits[0, local_pos].clone().float()
                row_logits[int(mask_id)] = torch.finfo(row_logits.dtype).min
                generated = sequence[original_len:].detach().tolist()
                text = contiguous_decoded_prefix(tokenizer, generated, mask_id)
                in_value = bool(qwen_native_inside_parameter_value(text))
                grammar_active = bool(tool_json_live_prefix_active(text))
                original_top = int(torch.argmax(row_logits).item())
                replacement = False
                safe = True

                if grammar_active:
                    metrics["grammar_checked_tokens"] += 1
                    can_stop = native_tool_prefix_can_stop(text, schemas=schemas)
                    if not can_stop:
                        for stop_id in special_stop_token_ids:
                            if 0 <= int(stop_id) < row_logits.shape[-1]:
                                row_logits[int(stop_id)] = torch.finfo(row_logits.dtype).min
                        original_top = int(torch.argmax(row_logits).item())
                    if original_top in stop_token_ids and can_stop:
                        token_id = original_top
                    else:
                        token_id, safe = live_tool_json_top_token(
                            tokenizer,
                            sequence.clone(),
                            row_logits,
                            original_len,
                            abs_idx,
                            int(mask_id),
                            int(grammar_topk),
                            schemas=schemas,
                        )
                    replacement = int(token_id) != original_top
                    if replacement:
                        metrics["grammar_replacement_tokens"] += 1
                    if not safe:
                        metrics["grammar_unsafe_fallback_tokens"] += 1
                    prob = _token_probability(row_logits, int(token_id), temperature=temperature)
                else:
                    token_id, prob = _sample_unconstrained_token(
                        model,
                        row_logits,
                        top_p=float(top_p),
                        temperature=float(temperature),
                    )

                if in_value:
                    metrics["value_positions"] += 1
                    metrics["model_chosen_value_tokens"] += 1
                    if replacement:
                        metrics["grammar_replacement_value_tokens"] += 1
                else:
                    metrics["structural_positions"] += 1

                block_state[0, local_pos] = int(token_id)
                sequence[abs_idx] = int(token_id)
                chosen_probs.append(float(prob))
                metrics["committed_tokens"] += 1

            committed = target_visible
            metrics["denoise_forwards"] += 1
            metrics["denoise_seconds"] += seconds
            metrics["step_commits"].append(len(positions))
            block_metrics["step_visible"].append(int(committed))
            block_metrics["step_committed"].append(int(len(positions)))
            block_metrics["step_conf_mean"].append(float(sum(chosen_probs) / len(chosen_probs)) if chosen_probs else 0.0)
            block_metrics["step_conf_min"].append(float(min(chosen_probs)) if chosen_probs else 0.0)
            block_metrics["step_conf_max"].append(float(max(chosen_probs)) if chosen_probs else 0.0)
            block_metrics["step_seconds"].append(seconds)

        unresolved = int((block_state == int(mask_id)).sum().item())
        metrics["unresolved_masks"] += unresolved
        output_ids = torch.cat([output_ids, block_state], dim=1)
        metrics["blocks"].append(block_metrics)
        metrics["visible_schedule"].append(block_metrics["step_visible"])

        generated = output_ids[0, original_len:]
        stop_offset = first_stop_offset(generated, stop_token_ids)
        if stop_offset is not None:
            metrics["stop_offset"] = int(stop_offset)
            metrics["stop_token_id"] = int(generated[stop_offset].item())
            output_ids = output_ids[:, : original_len + stop_offset + 1]
            break

    return output_ids[0].detach().cpu(), metrics


def run_fixedk(args: argparse.Namespace, episodes: list[dict], tokenizer, chat_template: str | None) -> list[dict]:
    if args.block_size % args.denoise_steps != 0:
        raise SystemExit("--block-size must be divisible by --denoise-steps for nominal B/K accounting")
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
    backend_prefix = "diffusion_schedule_fixedk_constrained" if args.constrained_grammar else "diffusion_schedule_fixedk"
    backend_name = f"{backend_prefix}_k{int(args.denoise_steps)}"
    for episode in episodes:
        messages = [dict(message) for message in episode["prompt_messages"]]
        prompt = render_matched_prompt(model_tokenizer, messages, episode["tools"], chat_template)
        for turn_idx, gold_block in enumerate(episode["gold_blocks"]):
            prompt_input_ids = model_tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.time()
            if args.constrained_grammar:
                output_ids, sampler_metrics = sample_fixed_k_constrained_grammar(
                    model,
                    model_tokenizer,
                    prompt_input_ids,
                    block_size=int(args.block_size),
                    denoise_steps=int(args.denoise_steps),
                    max_new_tokens=int(args.max_new_tokens),
                    mask_id=int(mask_id),
                    stop_token_ids=stop_token_ids,
                    top_p=float(args.top_p),
                    temperature=float(args.temperature),
                    schemas=tool_schema_by_name(episode["tools"]),
                    grammar_topk=int(args.grammar_topk),
                )
            else:
                output_ids, sampler_metrics = sample_fixed_k_block_diffusion(
                    model,
                    prompt_input_ids,
                    block_size=int(args.block_size),
                    denoise_steps=int(args.denoise_steps),
                    max_new_tokens=int(args.max_new_tokens),
                    mask_id=int(mask_id),
                    stop_token_ids=stop_token_ids,
                    top_p=float(args.top_p),
                    temperature=float(args.temperature),
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            turn_wall_seconds = time.time() - start
            new_ids = output_ids[int(prompt_input_ids.numel()) :]
            history_text = decode_text(model_tokenizer, new_ids)
            assistant_text = trim_scored_assistant(history_text)
            denoise_forwards = int(sampler_metrics.get("denoise_forwards") or 0)
            schedule_events = {
                "denoise_forwards_total": denoise_forwards,
                "fixed_k_denoise_forwards": denoise_forwards,
                "fixed_k_generated_tokens": int(new_ids.numel()),
                "fixed_k_nominal_tokens_per_forward": float(args.block_size) / float(args.denoise_steps),
                "parallel_commit_denoise_forwards": denoise_forwards,
                "parallel_commit_committed_tokens": int(sampler_metrics.get("committed_tokens") or int(new_ids.numel())),
                "parallel_commit_structural_tokens": int(sampler_metrics.get("structural_positions") or 0),
                "parallel_commit_value_tokens": int(sampler_metrics.get("model_chosen_value_tokens") or 0),
                "parallel_commit_forced_tokens": 0,
                "two_wave_wave1_projected_tokens": 0,
                "two_wave_wave1_value_tokens": 0,
                "two_wave_wave2_value_tokens": 0,
                "two_wave_wave2_forced_tokens": 0,
                "constrained_parallel_grammar_checked_tokens": int(sampler_metrics.get("grammar_checked_tokens") or 0),
                "constrained_parallel_grammar_replacement_tokens": int(
                    sampler_metrics.get("grammar_replacement_tokens") or 0
                ),
                "constrained_parallel_grammar_unsafe_fallback_tokens": int(
                    sampler_metrics.get("grammar_unsafe_fallback_tokens") or 0
                ),
                "constrained_parallel_value_positions": int(sampler_metrics.get("value_positions") or 0),
                "constrained_parallel_model_chosen_value_tokens": int(
                    sampler_metrics.get("model_chosen_value_tokens") or 0
                ),
                "constrained_parallel_projected_value_tokens": 0,
                "constrained_parallel_gold_forced_tokens": 0,
            }
            row = row_from_generation(
                backend=backend_name,
                episode=episode,
                turn_idx=turn_idx,
                prompt=prompt,
                tools=episode["tools"],
                gold_block=gold_block,
                assistant_text=assistant_text,
                prompt_tokens=int(prompt_input_ids.numel()),
                generated_tokens=int(new_ids.numel()),
                turn_wall_seconds=turn_wall_seconds,
                schedule_build_seconds=0.0,
                backend_meta={
                    "sampler_schedule_events": schedule_events,
                    "fixed_k_sampler_metrics": sampler_metrics,
                    "max_new_tokens": int(args.max_new_tokens),
                    "block_size": int(args.block_size),
                    "denoise_steps": int(args.denoise_steps),
                    "constrained_grammar": bool(args.constrained_grammar),
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
                f"fixedk K={args.denoise_steps} episode={episode['episode_idx']} "
                f"turn={turn_idx + 1}/{len(episode['gold_blocks'])} "
                f"exact_args={int(bool(row['exact_arguments']))} "
                f"forwards={denoise_forwards} wall={turn_wall_seconds:.3f}s",
                flush=True,
            )
    return rows


def main() -> int:
    args = parse_args()
    args.ar_model_path = args.prompt_tokenizer_path
    args.ar_served_model = "not_used_schedule_fixedk"
    args.ar_base_url = "not_used_schedule_fixedk"
    args.diffusion_condition = "schedule_fixed_k"
    args.diffusion_structural_only = False
    torch.manual_seed(int(args.seed))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompt_tokenizer = AutoTokenizer.from_pretrained(str(args.prompt_tokenizer_path), trust_remote_code=True)
    chat_template = load_chat_template(args.chat_template_path)
    episodes = build_episodes(args)
    manifest = write_manifest(args, episodes, prompt_tokenizer, chat_template)
    manifest["diffusion"]["decode"] = "schedule_fixed_k_mutable_visible_set"
    manifest["diffusion"]["condition"] = "schedule_fixed_k"
    manifest["diffusion"]["block_size"] = int(args.block_size)
    manifest["diffusion"]["denoise_steps"] = int(args.denoise_steps)
    manifest["diffusion"]["nominal_tokens_per_forward"] = float(args.block_size) / float(args.denoise_steps)
    if args.constrained_grammar:
        manifest["diffusion"]["decode"] = "schedule_fixed_k_constrained_grammar_prefix"
        manifest["diffusion"]["condition"] = "schedule_fixed_k_constrained_grammar"
        manifest["diffusion"]["grammar_topk"] = int(args.grammar_topk)
    (args.out_dir / "fairness_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rows = run_fixedk(args, episodes, prompt_tokenizer, chat_template)
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
            "denoise_steps": int(args.denoise_steps),
            "nominal_tokens_per_forward": float(args.block_size) / float(args.denoise_steps),
            "sampler": (
                "fixed-denoise schedule-gated diagnostic: mutable visible set, "
                "top-confidence positions retained per step, mask token banned"
            ),
            "constrained_grammar": bool(args.constrained_grammar),
            "grammar_topk": int(args.grammar_topk),
        }
    )
    if args.constrained_grammar:
        summary["sampler"] = (
            "fixed-denoise constrained grammar diagnostic: each forward commits the next B/K prefix "
            "positions using that forward's logits, Qwen-native grammar masks illegal tokens, no gold value projection"
        )
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
