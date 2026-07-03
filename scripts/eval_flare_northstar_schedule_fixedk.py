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

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from eval_fastdllm_toolcall_cases import load_model, resolve_token_ids  # noqa: E402
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
from measure_block_quality_curve import sample_fixed_k_block_diffusion  # noqa: E402


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
    backend_name = f"diffusion_schedule_fixedk_k{int(args.denoise_steps)}"
    for episode in episodes:
        messages = [dict(message) for message in episode["prompt_messages"]]
        prompt = render_matched_prompt(model_tokenizer, messages, episode["tools"], chat_template)
        for turn_idx, gold_block in enumerate(episode["gold_blocks"]):
            prompt_input_ids = model_tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.time()
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
                    "sampler_schedule_events": {
                        "denoise_forwards_total": denoise_forwards,
                        "fixed_k_denoise_forwards": denoise_forwards,
                        "fixed_k_generated_tokens": int(new_ids.numel()),
                        "fixed_k_nominal_tokens_per_forward": float(args.block_size) / float(args.denoise_steps),
                    },
                    "fixed_k_sampler_metrics": sampler_metrics,
                    "max_new_tokens": int(args.max_new_tokens),
                    "block_size": int(args.block_size),
                    "denoise_steps": int(args.denoise_steps),
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
