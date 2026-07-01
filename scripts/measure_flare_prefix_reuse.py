#!/usr/bin/env python3
"""Measure exact FLARE prefix-cache reuse on growing multi-turn prompts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from eval_fastdllm_toolcall_cases import load_model, resolve_token_ids
from flare_hf_cache import FlarePrefixCache, RequestDiffusionState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=ROOT / "models/qwen3.5-9b-fastdllm-init")
    parser.add_argument("--adapter", type=Path, default=ROOT / "runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000")
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--no-merge-adapter", action="store_true")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--prefix-blocks", type=int, default=240)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--append-blocks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--out-json", type=Path, default=ROOT / "runs/agentic_eval/flare_prefix_reuse_measurement.json")
    return parser.parse_args()


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_ids(model, seq_len: int, mask_id: int, device: torch.device) -> torch.Tensor:
    vocab_size = int(model.config.vocab_size)
    ids = torch.randint(4, vocab_size, (1, int(seq_len)), device=device)
    return torch.where(ids == int(mask_id), torch.full_like(ids, (int(mask_id) + 1) % vocab_size), ids)


def growing_prompts(
    model,
    *,
    mask_id: int,
    device: torch.device,
    block_size: int,
    prefix_blocks: int,
    turns: int,
    append_blocks: int,
) -> list[torch.Tensor]:
    current = make_ids(model, int(prefix_blocks) * int(block_size), mask_id, device)
    prompts = [current]
    for _ in range(1, int(turns)):
        extra = make_ids(model, int(append_blocks) * int(block_size), mask_id, device)
        current = torch.cat([current, extra], dim=1)
        prompts.append(current)
    return prompts


def timed_reset(
    model,
    prompt: torch.Tensor,
    block_size: int,
    *,
    prefix_cache: FlarePrefixCache | None,
) -> tuple[RequestDiffusionState, float]:
    synchronize()
    start = time.perf_counter()
    state = RequestDiffusionState.reset(model, prompt, block_size, prefix_cache=prefix_cache)
    synchronize()
    return state, time.perf_counter() - start


def max_cache_diff(a: RequestDiffusionState, b: RequestDiffusionState) -> float:
    if a.block_start != b.block_start:
        return float("inf")
    if len(a.layer_caches) != len(b.layer_caches):
        return float("inf")
    max_diff = 0.0
    for left, right in zip(a.layer_caches, b.layer_caches):
        if left.kind != right.kind:
            return float("inf")
        for name in ("gdn_state", "conv_tail", "key", "value"):
            x = getattr(left, name)
            y = getattr(right, name)
            if x is None and y is None:
                continue
            if x is None or y is None or tuple(x.shape) != tuple(y.shape):
                return float("inf")
            if x.numel():
                max_diff = max(max_diff, float((x.float() - y.float()).abs().max().item()))
    return max_diff


@torch.no_grad()
def active_logit_diff(
    model,
    prompt: torch.Tensor,
    full_state: RequestDiffusionState,
    reuse_state: RequestDiffusionState,
    mask_id: int,
) -> dict:
    active_len = int(prompt.shape[1]) - int(full_state.block_start)
    if active_len < 0 or active_len >= int(full_state.block_size):
        raise RuntimeError(f"invalid active_len={active_len} block_start={full_state.block_start}")
    pad = int(full_state.block_size) - active_len
    masks = torch.full((1, pad), int(mask_id), dtype=torch.long, device=prompt.device)
    x_t = torch.cat([prompt, masks], dim=1)
    full_logits = full_state.shifted_active_logits(model, x_t)
    reuse_logits = reuse_state.shifted_active_logits(model, x_t)
    max_abs = float((full_logits.float() - reuse_logits.float()).abs().max().item())
    argmax_flips = int((full_logits.argmax(dim=-1) != reuse_logits.argmax(dim=-1)).sum().item())
    return {
        "active_len": active_len,
        "compared_positions": int(full_logits.shape[1]),
        "logit_max_abs_diff": max_abs,
        "argmax_flips": argmax_flips,
    }


def main() -> int:
    args = parse_args()
    os.environ["FASTDLLM_FLARE_GDN_ROUTE"] = "route_i"
    os.environ.setdefault("FASTDLLM_FLARE_TWO_STREAM", "1")
    os.environ.setdefault("FLARE_TWO_STREAM", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(int(args.seed))
    model, tokenizer = load_model(
        args.base_model,
        args.adapter if args.adapter and args.adapter.exists() else None,
        merge_adapter=not args.no_merge_adapter,
        tokenizer_path=args.tokenizer_path,
    )
    model.eval()
    mask_id, _, _ = resolve_token_ids(model, tokenizer)
    device = torch.device("cuda")
    prompts = growing_prompts(
        model,
        mask_id=int(mask_id),
        device=device,
        block_size=int(args.block_size),
        prefix_blocks=int(args.prefix_blocks),
        turns=int(args.turns),
        append_blocks=int(args.append_blocks),
    )

    prefix_cache = FlarePrefixCache()
    turns = []
    full_total = 0.0
    reuse_total = 0.0
    for idx, prompt in enumerate(prompts):
        full_state, full_seconds = timed_reset(model, prompt, int(args.block_size), prefix_cache=None)
        reuse_state, reuse_seconds = timed_reset(model, prompt, int(args.block_size), prefix_cache=prefix_cache)
        full_total += full_seconds
        reuse_total += reuse_seconds
        cache_diff = max_cache_diff(full_state, reuse_state)
        logit_diff = active_logit_diff(model, prompt, full_state, reuse_state, int(mask_id))
        prefix_cache.store(prompt, reuse_state)
        turns.append(
            {
                "turn": int(idx),
                "prompt_tokens": int(prompt.shape[1]),
                "prompt_blocks": int(prompt.shape[1]) // int(args.block_size),
                "full_rebuild_reset_seconds": full_seconds,
                "prefix_reuse_reset_seconds": reuse_seconds,
                "reset_speedup": full_seconds / reuse_seconds if reuse_seconds > 0 else None,
                "full_cache_stats": full_state.stats(),
                "reuse_cache_stats": reuse_state.stats(),
                "state_cache_max_abs_diff": cache_diff,
                **logit_diff,
                "prefix_cache_stats_after_store": prefix_cache.stats(),
            }
        )
        full_state.free()
        reuse_state.free()
        del full_state, reuse_state

    total_speedup = full_total / reuse_total if reuse_total > 0 else None
    total_appended_blocks = max(0, int(args.turns) - 1) * int(args.append_blocks)
    payload = {
        "status": "pass"
        if all(item["argmax_flips"] == 0 and item["state_cache_max_abs_diff"] == 0.0 for item in turns)
        else "fail",
        "base_model": str(args.base_model),
        "adapter": str(args.adapter) if args.adapter else None,
        "merge_adapter": not args.no_merge_adapter,
        "block_size": int(args.block_size),
        "prefix_blocks": int(args.prefix_blocks),
        "turns": int(args.turns),
        "append_blocks_per_turn": int(args.append_blocks),
        "total_appended_blocks": int(total_appended_blocks),
        "full_rebuild_total_reset_seconds": full_total,
        "prefix_reuse_total_reset_seconds": reuse_total,
        "total_reset_speedup": total_speedup,
        "mask_id": int(mask_id),
        "turn_results": turns,
        "final_prefix_cache_stats": prefix_cache.stats(),
    }
    if torch.cuda.is_available():
        payload["cuda_max_memory_allocated_gib"] = torch.cuda.max_memory_allocated() / (1024**3)
        payload["cuda_max_memory_reserved_gib"] = torch.cuda.max_memory_reserved() / (1024**3)

    text = json.dumps(payload, indent=2)
    print(text, flush=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(text + "\n", encoding="utf-8")
    if payload["status"] != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
