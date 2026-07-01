#!/usr/bin/env python3
"""Measure HF FLARE cache throughput for lockstep batches."""

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
from flare_hf_cache import RequestDiffusionState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=ROOT / "models/qwen3.5-9b-fastdllm-init")
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--no-merge-adapter", action="store_true")
    parser.add_argument("--batch-sizes", default="1,16")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--prefix-blocks", type=int, default=8)
    parser.add_argument("--read-steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--gen-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--out-json", type=Path, default=None)
    return parser.parse_args()


def parse_batch_sizes(raw: str) -> list[int]:
    values = []
    for item in raw.replace(";", ",").replace(" ", ",").split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("--batch-sizes cannot be empty")
    return values


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_ids(model, batch_size: int, seq_len: int, mask_id: int, device):
    vocab_size = int(model.config.vocab_size)
    ids = torch.randint(4, vocab_size, (batch_size, seq_len), device=device)
    return torch.where(ids == mask_id, torch.full_like(ids, (mask_id + 1) % vocab_size), ids)


@torch.no_grad()
def measure_batch(model, *, batch_size: int, block_size: int, prefix_blocks: int, read_steps: int, warmup_steps: int, gen_tokens: int, mask_id: int, device):
    prefix_len = int(prefix_blocks) * int(block_size)
    prompt = make_ids(model, batch_size, prefix_len, mask_id, device)
    state = RequestDiffusionState.reset(model, prompt, block_size)
    active = torch.full((batch_size, block_size), int(mask_id), dtype=torch.long, device=device)
    x_t = torch.cat([prompt, active], dim=1)

    for _ in range(warmup_steps):
        _ = state.shifted_active_logits(model, x_t)
    synchronize()
    read_start = time.time()
    for _ in range(read_steps):
        logits = state.shifted_active_logits(model, x_t)
    synchronize()
    read_elapsed = time.time() - read_start
    del logits

    gen_state = RequestDiffusionState.reset(model, prompt, block_size)
    output_ids = prompt
    generated = 0
    synchronize()
    gen_start = time.time()
    while generated < gen_tokens:
        remaining = gen_tokens - generated
        active_len = output_ids.shape[1] - gen_state.block_start
        block_pad = min(block_size - active_len, remaining)
        masks = torch.full((batch_size, block_pad), int(mask_id), dtype=torch.long, device=device)
        x_gen = torch.cat([output_ids, masks], dim=1)
        while bool((x_gen[:, -block_pad:] == mask_id).any().item()):
            window_start = gen_state.block_start
            window = x_gen[:, window_start:]
            logits = gen_state.shifted_active_logits(model, x_gen)
            mask_positions = (window[0] == mask_id).nonzero(as_tuple=False)
            if mask_positions.numel() == 0:
                break
            local_pos = int(mask_positions[0, 0].item())
            token = logits[:, local_pos, :].argmax(dim=-1)
            x_gen[:, window_start + local_pos] = token
        active_block = x_gen[:, gen_state.block_start :]
        if active_block.shape[1] == block_size and not bool((active_block == mask_id).any().item()):
            gen_state.advance(model, active_block)
        generated += int(x_gen.shape[1] - output_ids.shape[1])
        output_ids = x_gen
    synchronize()
    gen_elapsed = time.time() - gen_start

    return {
        "batch_size": int(batch_size),
        "prefix_len": int(prefix_len),
        "block_size": int(block_size),
        "read_steps": int(read_steps),
        "read_elapsed_seconds": read_elapsed,
        "cached_reads_per_second": read_steps / read_elapsed if read_elapsed else 0.0,
        "active_logit_tokens_per_second": (batch_size * block_size * read_steps) / read_elapsed if read_elapsed else 0.0,
        "commit_one_generated_tokens": int(batch_size * generated),
        "commit_one_elapsed_seconds": gen_elapsed,
        "commit_one_policy_tokens_per_second": (batch_size * generated) / gen_elapsed if gen_elapsed else 0.0,
        "read_cache_stats": state.stats(),
        "generation_cache_stats": gen_state.stats(),
    }


def main() -> int:
    args = parse_args()
    os.environ["FASTDLLM_FLARE_GDN_ROUTE"] = "route_i"
    os.environ.setdefault("FASTDLLM_FLARE_TWO_STREAM", "1")
    os.environ.setdefault("FLARE_TWO_STREAM", "1")
    if not torch.cuda.is_available():
        raise RuntimeError("throughput measurement requires CUDA")
    torch.manual_seed(args.seed)
    model, tokenizer = load_model(
        args.base_model,
        args.adapter,
        merge_adapter=not args.no_merge_adapter,
        tokenizer_path=args.tokenizer_path,
    )
    mask_id, _, _ = resolve_token_ids(model, tokenizer)
    device = torch.device("cuda")
    results = []
    for batch_size in parse_batch_sizes(args.batch_sizes):
        torch.cuda.empty_cache()
        try:
            results.append(
                measure_batch(
                    model,
                    batch_size=batch_size,
                    block_size=int(args.block_size),
                    prefix_blocks=int(args.prefix_blocks),
                    read_steps=int(args.read_steps),
                    warmup_steps=int(args.warmup_steps),
                    gen_tokens=int(args.gen_tokens),
                    mask_id=int(mask_id),
                    device=device,
                )
            )
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            results.append({"batch_size": int(batch_size), "oom": True, "error": str(exc)})
    payload = {
        "base_model": str(args.base_model),
        "adapter": str(args.adapter) if args.adapter else None,
        "block_size": int(args.block_size),
        "prefix_blocks": int(args.prefix_blocks),
        "prefix_len": int(args.prefix_blocks * args.block_size),
        "mask_id": int(mask_id),
        "results": results,
    }
    if torch.cuda.is_available():
        payload["cuda_max_memory_allocated_gib"] = torch.cuda.max_memory_allocated() / (1024**3)
        payload["cuda_max_memory_reserved_gib"] = torch.cuda.max_memory_reserved() / (1024**3)
    text = json.dumps(payload, indent=2)
    print(text, flush=True)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
