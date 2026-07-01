#!/usr/bin/env python3
"""Profile FLARE HF cache noisy/advance forwards by layer component."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from eval_fastdllm_toolcall_cases import load_model, resolve_token_ids
from flare_hf_cache import (
    FlareLayerCache,
    RequestDiffusionState,
    _assert_cache_shapes,
    _assert_route_i,
    _set_block_size,
    _tail_after_append,
    append_kv_cache,
    attention_project,
    clean_active_attention_mask,
    modeling_module_for,
    noisy_active_attention_mask,
    run_attention_from_kv,
    unwrap_lm_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=ROOT / "models/qwen3.5-9b-fastdllm-init")
    parser.add_argument("--adapter", type=Path, default=ROOT / "runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000")
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--no-merge-adapter", action="store_true")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--prefix-blocks", type=int, default=32)
    parser.add_argument("--read-steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--out-json", type=Path, default=ROOT / "runs/agentic_eval/flare_hf_cache_forward_profile.json")
    return parser.parse_args()


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    def __init__(self):
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)

    def measure(self, key: str, fn):
        synchronize()
        start = time.perf_counter()
        value = fn()
        synchronize()
        self.seconds[key] += time.perf_counter() - start
        self.calls[key] += 1
        return value

    def summary(self) -> dict:
        total = sum(self.seconds.values())
        return {
            key: {
                "seconds": self.seconds[key],
                "calls": self.calls[key],
                "mean_seconds": self.seconds[key] / max(1, self.calls[key]),
                "share": self.seconds[key] / total if total > 0 else None,
            }
            for key in sorted(self.seconds)
        }


def make_ids(model, batch_size: int, seq_len: int, mask_id: int, device):
    vocab_size = int(model.config.vocab_size)
    ids = torch.randint(4, vocab_size, (batch_size, seq_len), device=device)
    return torch.where(ids == mask_id, torch.full_like(ids, (mask_id + 1) % vocab_size), ids)


@torch.no_grad()
def profiled_noisy_logits(model, block_ids: torch.Tensor, state: RequestDiffusionState, timer: Timer) -> torch.Tensor:
    lm_model = unwrap_lm_model(model)
    modeling_module = modeling_module_for(lm_model)
    _assert_route_i(lm_model, modeling_module)
    _set_block_size(lm_model, state.block_size)
    core = lm_model.model
    hidden = timer.measure("embed", lambda: core.embed_tokens(block_ids))
    block_len = int(block_ids.shape[1])
    position_ids = torch.arange(
        state.block_start,
        state.block_start + block_len,
        device=block_ids.device,
    ).unsqueeze(0).expand(block_ids.shape[0], -1)
    position_embeddings = timer.measure("rotary_emb", lambda: core.rotary_emb(hidden, position_ids))

    for layer_idx, layer in enumerate(core.layers):
        layer_cache = state.layer_caches[layer_idx]
        residual = hidden
        normed = timer.measure("input_layernorm", lambda layer=layer, hidden=hidden: layer.input_layernorm(hidden))
        if layer.layer_type == "linear_attention":
            output, _, _, _ = timer.measure(
                "gdn",
                lambda layer=layer, normed=normed, layer_cache=layer_cache: modeling_module.run_gdn_manual_route_i(
                    layer.linear_attn,
                    normed,
                    chunk_size=state.block_size,
                    initial_state=layer_cache.gdn_state,
                    conv_tail=layer_cache.conv_tail,
                    output_chunk_states=False,
                ),
            )
        else:
            prefix_key = layer_cache.key
            prefix_value = layer_cache.value
            prefix_len = 0 if prefix_key is None else int(prefix_key.shape[2])
            mask = noisy_active_attention_mask(block_ids.shape[0], block_len, prefix_len, block_ids.device)
            output, _, _ = timer.measure(
                "attention",
                lambda layer=layer, normed=normed, position_embeddings=position_embeddings, prefix_key=prefix_key, prefix_value=prefix_value, mask=mask: run_attention_from_kv(
                    layer.self_attn,
                    normed,
                    position_embeddings,
                    prefix_key,
                    prefix_value,
                    mask,
                    modeling_module,
                ),
            )
        hidden = residual + output
        residual = hidden
        hidden = residual + timer.measure("mlp", lambda layer=layer, hidden=hidden: layer.mlp(layer.post_attention_layernorm(hidden)))

    hidden = timer.measure("final_norm", lambda: core.norm(hidden))
    return timer.measure("lm_head", lambda: lm_model.lm_head(hidden))


@torch.no_grad()
def profiled_clean_advance(model, block_ids: torch.Tensor, state: RequestDiffusionState, timer: Timer) -> None:
    lm_model = unwrap_lm_model(model)
    modeling_module = modeling_module_for(lm_model)
    _assert_route_i(lm_model, modeling_module)
    _set_block_size(lm_model, state.block_size)
    core = lm_model.model
    hidden = timer.measure("embed", lambda: core.embed_tokens(block_ids))
    block_len = int(block_ids.shape[1])
    position_ids = torch.arange(
        state.block_start,
        state.block_start + block_len,
        device=block_ids.device,
    ).unsqueeze(0).expand(block_ids.shape[0], -1)
    position_embeddings = timer.measure("rotary_emb", lambda: core.rotary_emb(hidden, position_ids))
    new_caches: list[FlareLayerCache] = []

    for layer_idx, layer in enumerate(core.layers):
        layer_cache = state.layer_caches[layer_idx]
        residual = hidden
        normed = timer.measure("input_layernorm", lambda layer=layer, hidden=hidden: layer.input_layernorm(hidden))
        if layer.layer_type == "linear_attention":
            output, final_state, chunk_states, raw_qkv = timer.measure(
                "gdn",
                lambda layer=layer, normed=normed, layer_cache=layer_cache: modeling_module.run_gdn_manual_route_i(
                    layer.linear_attn,
                    normed,
                    chunk_size=state.block_size,
                    initial_state=layer_cache.gdn_state,
                    conv_tail=layer_cache.conv_tail,
                    output_chunk_states=True,
                ),
            )
            if chunk_states is None:
                raise RuntimeError("clean advance did not return GDN chunk states")
            boundary_state = chunk_states[:, -1].detach().clone().to(torch.float32).contiguous()
            if final_state is not None:
                final_diff = (final_state.float() - boundary_state.float()).abs().max()
                if float(final_diff.item()) > 1e-3:
                    raise RuntimeError(f"GDN final_state/chunk_state mismatch: {float(final_diff.item()):.6g}")
            new_caches.append(
                FlareLayerCache(
                    kind="linear_attention",
                    gdn_state=boundary_state,
                    conv_tail=_tail_after_append(
                        layer_cache.conv_tail,
                        raw_qkv,
                        int(layer.linear_attn.conv_kernel_size) - 1,
                    ),
                )
            )
        else:
            prefix_key = layer_cache.key
            prefix_value = layer_cache.value
            prefix_len = 0 if prefix_key is None else int(prefix_key.shape[2])
            mask = clean_active_attention_mask(block_ids.shape[0], block_len, prefix_len, block_ids.device)
            output, active_key, active_value = timer.measure(
                "attention",
                lambda layer=layer, normed=normed, position_embeddings=position_embeddings, prefix_key=prefix_key, prefix_value=prefix_value, mask=mask: run_attention_from_kv(
                    layer.self_attn,
                    normed,
                    position_embeddings,
                    prefix_key,
                    prefix_value,
                    mask,
                    modeling_module,
                ),
            )
            key = timer.measure("kv_append", lambda prefix_key=prefix_key, active_key=active_key: append_kv_cache(prefix_key, active_key))
            value = timer.measure(
                "kv_append",
                lambda prefix_value=prefix_value, active_value=active_value: append_kv_cache(prefix_value, active_value),
            )
            new_caches.append(FlareLayerCache(kind="full_attention", key=key, value=value))
        hidden = residual + output
        residual = hidden
        hidden = residual + timer.measure("mlp", lambda layer=layer, hidden=hidden: layer.mlp(layer.post_attention_layernorm(hidden)))

    state.layer_caches = new_caches
    _assert_cache_shapes(state)


def ratio_summary(component_summary: dict) -> dict:
    total = sum(item["seconds"] for item in component_summary.values())
    grouped = {
        "attention": component_summary.get("attention", {}).get("seconds", 0.0),
        "gdn": component_summary.get("gdn", {}).get("seconds", 0.0),
        "mlp": component_summary.get("mlp", {}).get("seconds", 0.0),
        "other": 0.0,
    }
    grouped["other"] = max(0.0, total - grouped["attention"] - grouped["gdn"] - grouped["mlp"])
    return {
        key: {
            "seconds": value,
            "share": value / total if total > 0 else None,
        }
        for key, value in grouped.items()
    }


def main() -> int:
    args = parse_args()
    os.environ["FASTDLLM_FLARE_GDN_ROUTE"] = "route_i"
    os.environ.setdefault("FASTDLLM_FLARE_TWO_STREAM", "1")
    os.environ.setdefault("FLARE_TWO_STREAM", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    model, tokenizer = load_model(
        args.base_model,
        args.adapter if args.adapter and args.adapter.exists() else None,
        merge_adapter=not args.no_merge_adapter,
        tokenizer_path=args.tokenizer_path,
    )
    mask_id, _, _ = resolve_token_ids(model, tokenizer)
    device = torch.device("cuda")
    prefix_len = int(args.prefix_blocks) * int(args.block_size)
    prompt = make_ids(model, 1, prefix_len, int(mask_id), device)
    state = RequestDiffusionState.reset(model, prompt, int(args.block_size))
    active = torch.full((1, int(args.block_size)), int(mask_id), dtype=torch.long, device=device)
    x_t = torch.cat([prompt, active], dim=1)

    for _ in range(int(args.warmup_steps)):
        _ = state.shifted_active_logits(model, x_t)
    synchronize()
    torch.cuda.reset_peak_memory_stats()

    read_timer = Timer()
    read_start = time.perf_counter()
    for _ in range(int(args.read_steps)):
        _ = profiled_noisy_logits(model, active, state, read_timer)
    synchronize()
    read_seconds = time.perf_counter() - read_start

    advance_timer = Timer()
    next_block = make_ids(model, 1, int(args.block_size), int(mask_id), device)
    advance_start = time.perf_counter()
    profiled_clean_advance(model, next_block, state, advance_timer)
    synchronize()
    advance_seconds = time.perf_counter() - advance_start

    read_components = read_timer.summary()
    advance_components = advance_timer.summary()
    payload = {
        "base_model": str(args.base_model),
        "adapter": str(args.adapter) if args.adapter else None,
        "block_size": int(args.block_size),
        "prefix_blocks": int(args.prefix_blocks),
        "prefix_len": int(prefix_len),
        "read_steps": int(args.read_steps),
        "warmup_steps": int(args.warmup_steps),
        "read_seconds": read_seconds,
        "mean_read_seconds": read_seconds / max(1, int(args.read_steps)),
        "read_components": read_components,
        "read_grouped": ratio_summary(read_components),
        "advance_seconds": advance_seconds,
        "advance_components": advance_components,
        "advance_grouped": ratio_summary(advance_components),
        "cache_stats_before_advance": {
            "block_start": int(prefix_len),
            "block_size": int(args.block_size),
            "batch_size": 1,
        },
        "cuda_max_memory_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "cuda_max_memory_reserved_gib": torch.cuda.max_memory_reserved() / (1024**3),
    }
    text = json.dumps(payload, indent=2)
    print(text, flush=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
