#!/usr/bin/env python3
"""Validate the HF route-I FLARE serving cache.

T1: multi-block cache-on/cache-off active-token argmax parity.
T2: shifted serving logits/logprobs vs route-I FLARE training-style forward.
T3: greedy commit-one multi-block canary token/byte identity.
"""

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

from eval_fastdllm_toolcall_cases import flare_two_stream_noisy_logits, load_model, resolve_token_ids
from flare_hf_cache import RequestDiffusionState
from validate_flare_two_stream_forward import load_local_bridge, make_tiny_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/qwen3.5-9b-fastdllm-init")
    parser.add_argument("--real-weights", action="store_true")
    parser.add_argument("--base-model", type=Path, default=ROOT / "models/qwen3.5-9b-fastdllm-init")
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--no-merge-adapter", action="store_true")
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=0)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--t3-new-tokens", type=int, default=0)
    parser.add_argument("--prompt-text", default="")
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--out-json", type=Path, default=None)
    return parser.parse_args()


def max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0 and right.numel() == 0:
        return 0.0
    return float((left.float() - right.float()).abs().max().item())


def diff_distribution(diff: torch.Tensor) -> dict[str, float]:
    flat = diff.detach().float().reshape(-1).abs()
    if flat.numel() == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "p999": 0.0, "max": 0.0}
    quantiles = torch.quantile(flat, torch.tensor([0.5, 0.9, 0.99, 0.999], device=flat.device))
    return {
        "mean": float(flat.mean().item()),
        "p50": float(quantiles[0].item()),
        "p90": float(quantiles[1].item()),
        "p99": float(quantiles[2].item()),
        "p999": float(quantiles[3].item()),
        "max": float(flat.max().item()),
    }


def signed_diff_summary(diff: torch.Tensor) -> dict[str, float]:
    flat = diff.detach().float().reshape(-1)
    if flat.numel() == 0:
        return {"mean": 0.0, "positive_fraction": 0.0}
    return {
        "mean": float(flat.mean().item()),
        "positive_fraction": float((flat > 0).float().mean().item()),
    }


def load_validation_model(args):
    os.environ["FASTDLLM_FLARE_GDN_ROUTE"] = "route_i"
    os.environ.setdefault("FASTDLLM_FLARE_TWO_STREAM", "1")
    os.environ.setdefault("FLARE_TWO_STREAM", "1")
    if args.real_weights:
        if not torch.cuda.is_available():
            raise RuntimeError("--real-weights requires CUDA")
        model, tokenizer = load_model(
            args.base_model,
            args.adapter,
            merge_adapter=not args.no_merge_adapter,
            tokenizer_path=args.tokenizer_path,
        )
        mask_id, _, _ = resolve_token_ids(model, tokenizer)
        block_size = int(args.block_size or getattr(model.config, "bd_size", 32))
        return model.eval(), tokenizer, mask_id, block_size, torch.device("cuda")

    torch.set_num_threads(max(1, args.threads))
    config_module, modeling_module = load_local_bridge(args.model_dir.resolve())
    block_size = int(args.block_size or 4)
    model = make_tiny_model(config_module, modeling_module, seed=args.seed, block_size=block_size).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, None, int(model.config.mask_token_id), block_size, device


def shifted_reference(model, x_t: torch.Tensor, block_size: int, mask_id: int, active_start: int):
    noisy_logits = flare_two_stream_noisy_logits(
        model,
        x_t,
        x_t,
        block_size=block_size,
        mask_id=mask_id,
    )[: x_t.shape[0]]
    shifted = torch.cat([noisy_logits[:, :1, :], noisy_logits[:, :-1, :]], dim=1)
    return shifted[:, active_start:]


def deterministic_ids(model, batch_size: int, total_len: int, device: torch.device, mask_id: int):
    vocab_size = int(model.config.vocab_size)
    low = 4 if vocab_size > 8 else 0
    ids = torch.randint(low, vocab_size, (batch_size, total_len), device=device)
    ids = torch.where(ids == mask_id, torch.full_like(ids, (mask_id + 1) % vocab_size), ids)
    return ids


def noisy_block_from_clean(clean_block: torch.Tensor, mask_id: int, block_idx: int):
    positions = torch.arange(clean_block.shape[1], device=clean_block.device)
    pattern = ((positions + block_idx) % 3) == 0
    noisy = clean_block.clone()
    noisy[:, pattern] = int(mask_id)
    return noisy


@torch.no_grad()
def run_t1_t2(model, *, batch_size: int, blocks: int, block_size: int, mask_id: int, device: torch.device, atol: float):
    torch.manual_seed(20260701)
    clean_ids = deterministic_ids(model, batch_size, blocks * block_size, device, mask_id)
    state = RequestDiffusionState.reset(model, clean_ids[:, :0], block_size)
    total_positions = 0
    argmax_flips = 0
    max_logit_diff = 0.0
    max_logprob_diff = 0.0
    weighted_logit_abs_mean = 0.0
    weighted_logprob_abs_mean = 0.0
    weighted_kl_mean = 0.0
    max_top1_logprob_abs_delta = 0.0
    min_ref_top1_margin = float("inf")
    per_block = []

    for block_idx in range(blocks):
        start = block_idx * block_size
        end = start + block_size
        noisy_block = noisy_block_from_clean(clean_ids[:, start:end], mask_id, block_idx)
        x_t = torch.cat([clean_ids[:, :start], noisy_block], dim=1)
        cached = state.shifted_active_logits(model, x_t)
        reference = shifted_reference(model, x_t, block_size, mask_id, active_start=start)
        logit_delta = cached.float() - reference.float()
        logit_diff = max_abs_diff(cached, reference)
        cached_logp = torch.log_softmax(cached.float(), dim=-1)
        reference_logp = torch.log_softmax(reference.float(), dim=-1)
        logprob_delta = cached_logp - reference_logp
        logprob_diff = max_abs_diff(cached_logp, reference_logp)
        cached_argmax = cached.argmax(dim=-1)
        reference_argmax = reference.argmax(dim=-1)
        flips = int((cached_argmax != reference_argmax).sum().item())
        reference_top2 = torch.topk(reference_logp, k=2, dim=-1).values
        ref_margin = reference_top2[..., 0] - reference_top2[..., 1]
        ref_top_logprob_delta = torch.gather(logprob_delta, dim=-1, index=reference_argmax.unsqueeze(-1)).squeeze(-1)
        kl_ref_cached = (reference_logp.exp() * (reference_logp - cached_logp)).sum(dim=-1)
        elements = int(cached.numel())
        positions = int(cached.shape[0] * cached.shape[1])
        total_positions += int(cached.shape[0] * cached.shape[1])
        argmax_flips += flips
        max_logit_diff = max(max_logit_diff, logit_diff)
        max_logprob_diff = max(max_logprob_diff, logprob_diff)
        logit_distribution = diff_distribution(logit_delta)
        logprob_distribution = diff_distribution(logprob_delta)
        weighted_logit_abs_mean += logit_distribution["mean"] * elements
        weighted_logprob_abs_mean += logprob_distribution["mean"] * elements
        weighted_kl_mean += float(kl_ref_cached.mean().item()) * positions
        max_top1_logprob_abs_delta = max(
            max_top1_logprob_abs_delta,
            float(ref_top_logprob_delta.abs().max().item()),
        )
        min_ref_top1_margin = min(min_ref_top1_margin, float(ref_margin.min().item()))
        per_block.append(
            {
                "block": block_idx,
                "logit_max_abs_diff": logit_diff,
                "logprob_max_abs_diff": logprob_diff,
                "logit_abs_diff": logit_distribution,
                "logprob_abs_diff": logprob_distribution,
                "logit_signed_diff": signed_diff_summary(logit_delta),
                "logprob_signed_diff": signed_diff_summary(logprob_delta),
                "ref_top1_logprob_abs_delta": diff_distribution(ref_top_logprob_delta),
                "ref_top1_margin_min": float(ref_margin.min().item()),
                "ref_top1_margin_p50": float(torch.quantile(ref_margin.float().reshape(-1), 0.5).item()),
                "kl_ref_cached_mean": float(kl_ref_cached.mean().item()),
                "kl_ref_cached_max": float(kl_ref_cached.max().item()),
                "argmax_flips": flips,
            }
        )
        state.advance(model, clean_ids[:, start:end])

    t1_pass = argmax_flips == 0
    t2_pass = argmax_flips == 0 and max_logit_diff <= atol and max_logprob_diff <= atol
    return {
        "name": "T1_T2",
        "blocks": int(blocks),
        "batch_size": int(batch_size),
        "positions": int(total_positions),
        "logit_max_abs_diff": max_logit_diff,
        "logprob_max_abs_diff": max_logprob_diff,
        "logit_abs_diff_mean": weighted_logit_abs_mean / max(1, total_positions * int(model.config.vocab_size)),
        "logprob_abs_diff_mean": weighted_logprob_abs_mean / max(1, total_positions * int(model.config.vocab_size)),
        "kl_ref_cached_mean": weighted_kl_mean / max(1, total_positions),
        "ref_top1_logprob_abs_delta_max": max_top1_logprob_abs_delta,
        "ref_top1_margin_min": 0.0 if min_ref_top1_margin == float("inf") else min_ref_top1_margin,
        "argmax_flips": int(argmax_flips),
        "t1_argmax_pass": bool(t1_pass),
        "t2_logit_logprob_pass": bool(t2_pass),
        "cache_stats": state.stats(),
        "per_block": per_block,
    }


@torch.no_grad()
def greedy_canary(model, prompt_ids: torch.Tensor, *, block_size: int, mask_id: int, new_tokens: int, use_cache: bool):
    output_ids = prompt_ids.clone()
    state = RequestDiffusionState.reset(model, output_ids, block_size) if use_cache else None
    original_len = output_ids.shape[1]
    while output_ids.shape[1] - original_len < new_tokens:
        remaining = new_tokens - (output_ids.shape[1] - original_len)
        if use_cache:
            active_len = output_ids.shape[1] - state.block_start
            block_pad = min(block_size - active_len, remaining)
        else:
            block_pad = block_size - (output_ids.shape[1] % block_size)
            if block_pad == 0:
                block_pad = block_size
            block_pad = min(block_pad, remaining)
        masks = torch.full(
            (output_ids.shape[0], block_pad),
            int(mask_id),
            dtype=torch.long,
            device=output_ids.device,
        )
        x_t = torch.cat([output_ids, masks], dim=1)
        while bool((x_t[:, -block_pad:] == mask_id).any().item()):
            if use_cache:
                window_start = state.block_start
                window = x_t[:, window_start:]
                logits = state.shifted_active_logits(model, x_t)
            else:
                window_len = min(block_size, x_t.shape[1])
                window_start = x_t.shape[1] - window_len
                window = x_t[:, window_start:]
                logits = shifted_reference(model, x_t, block_size, mask_id, active_start=window_start)
            mask_positions = (window[0] == mask_id).nonzero(as_tuple=False)
            if mask_positions.numel() == 0:
                break
            local_pos = int(mask_positions[0, 0].item())
            token = logits[:, local_pos, :].argmax(dim=-1)
            x_t[:, window_start + local_pos] = token
        if use_cache:
            active_block = x_t[:, state.block_start :]
            if active_block.shape[1] == block_size and not bool((active_block == mask_id).any().item()):
                state.advance(model, active_block)
        output_ids = x_t
    stats = state.stats() if state is not None else {}
    return output_ids, stats


def build_t3_prompt(model, tokenizer, args, block_size: int, mask_id: int, device: torch.device):
    if tokenizer is not None and args.prompt_text:
        prompt = tokenizer(args.prompt_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        return prompt
    prompt_len = block_size // 2
    return deterministic_ids(model, 1, prompt_len, device, mask_id)


@torch.no_grad()
def run_t3(model, tokenizer, args, *, block_size: int, mask_id: int, device: torch.device):
    new_tokens = int(args.t3_new_tokens or (2 * block_size))
    prompt_ids = build_t3_prompt(model, tokenizer, args, block_size, mask_id, device)
    off, _ = greedy_canary(
        model,
        prompt_ids,
        block_size=block_size,
        mask_id=mask_id,
        new_tokens=new_tokens,
        use_cache=False,
    )
    on, cache_stats = greedy_canary(
        model,
        prompt_ids,
        block_size=block_size,
        mask_id=mask_id,
        new_tokens=new_tokens,
        use_cache=True,
    )
    token_exact = torch.equal(off, on)
    byte_exact = token_exact
    off_text = on_text = None
    if tokenizer is not None:
        off_text = tokenizer.decode(off[0].detach().cpu().tolist(), skip_special_tokens=False)
        on_text = tokenizer.decode(on[0].detach().cpu().tolist(), skip_special_tokens=False)
        byte_exact = off_text.encode("utf-8") == on_text.encode("utf-8")
    return {
        "name": "T3",
        "new_tokens": int(new_tokens),
        "prompt_len": int(prompt_ids.shape[1]),
        "token_exact": bool(token_exact),
        "byte_exact": bool(byte_exact),
        "cache_stats": cache_stats,
        "cache_off_ids": off[0].detach().cpu().tolist(),
        "cache_on_ids": on[0].detach().cpu().tolist(),
        "cache_off_text": off_text,
        "cache_on_text": on_text,
    }


def main() -> int:
    args = parse_args()
    start = time.time()
    model, tokenizer, mask_id, block_size, device = load_validation_model(args)
    result = {
        "real_weights": bool(args.real_weights),
        "base_model": str(args.base_model) if args.real_weights else str(args.model_dir),
        "adapter": str(args.adapter) if args.adapter else None,
        "block_size": int(block_size),
        "batch_size": int(args.batch_size),
        "blocks": int(args.blocks),
        "mask_id": int(mask_id),
        "device": str(device),
    }
    with torch.no_grad():
        result["t1_t2"] = run_t1_t2(
            model,
            batch_size=int(args.batch_size),
            blocks=int(args.blocks),
            block_size=block_size,
            mask_id=int(mask_id),
            device=device,
            atol=float(args.atol),
        )
        result["t3"] = run_t3(model, tokenizer, args, block_size=block_size, mask_id=int(mask_id), device=device)
    result["elapsed_seconds"] = time.time() - start
    result["passed"] = bool(
        result["t1_t2"]["t1_argmax_pass"]
        and result["t1_t2"]["t2_logit_logprob_pass"]
        and result["t3"]["token_exact"]
        and result["t3"]["byte_exact"]
    )
    if torch.cuda.is_available():
        result["cuda_max_memory_allocated_gib"] = torch.cuda.max_memory_allocated() / (1024**3)
        result["cuda_max_memory_reserved_gib"] = torch.cuda.max_memory_reserved() / (1024**3)
    text = json.dumps(result, indent=2)
    print(text, flush=True)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
