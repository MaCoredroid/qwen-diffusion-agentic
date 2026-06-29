#!/usr/bin/env python3
"""CPU validation for a Qwen3.5 GDN-state cached active-block sampler.

This is a small random-weight harness. It does not load the 9B checkpoint.
It validates:
  1. full-attention prefix-KV + active-block equivalence,
  2. whole-model cached active-block shifted-logit equivalence, and
  3. token-exact small sampler matches against the existing golden sampler
     entry points.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
import types
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class LayerCache:
    kind: str
    gdn_state: torch.Tensor | None = None
    conv_tail: torch.Tensor | None = None
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None


@dataclass(frozen=True)
class StateCache:
    seq_len: int
    layer_caches: tuple[LayerCache, ...]
    last_token_logits: torch.Tensor | None


class DummyTokenizer:
    pad_token_id = 0

    def decode(self, token_ids, skip_special_tokens=True):
        if torch.is_tensor(token_ids):
            token_ids = token_ids.detach().cpu().tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return " ".join(str(int(token_id)) for token_id in token_ids)


class DeviceProxy:
    def __init__(self, model, device):
        self._model = model
        self.device = torch.device(device)

    def __getattr__(self, name):
        return getattr(self._model, name)

    def forward(self, *args, **kwargs):
        return self._model.forward(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/qwen3.5-9b-fastdllm-init")
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--threads", type=int, default=min(8, (os_cpu_count := __import__("os").cpu_count()) or 1))
    return parser.parse_args()


def load_local_bridge(model_dir: Path):
    package_name = "_fastdllm_qwen35_cache_validation"
    package = types.ModuleType(package_name)
    package.__path__ = [str(model_dir)]
    sys.modules[package_name] = package
    loaded = {}
    for module_name in ("configuration", "modeling"):
        spec = importlib.util.spec_from_file_location(
            f"{package_name}.{module_name}",
            model_dir / f"{module_name}.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not import {module_name} from {model_dir}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{package_name}.{module_name}"] = module
        spec.loader.exec_module(module)
        loaded[module_name] = module
    return loaded["configuration"], loaded["modeling"]


def make_small_config(config_module):
    return config_module.Fast_dLLM_Qwen3_5Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        head_dim=16,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
        bd_size=8,
        mask_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        gdn_mode="option_a_causal_gdn_v0",
        diffusion_bridge_status="implemented",
    )


def max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0 and right.numel() == 0:
        return 0.0
    return float((left.float() - right.float()).abs().max().item())


def project_and_conv(layer, hidden_states: torch.Tensor, conv_tail: torch.Tensor | None):
    seq_len = hidden_states.shape[1]
    raw_qkv = layer.in_proj_qkv(hidden_states)
    if conv_tail is None:
        conv_input = raw_qkv
        slice_start = 0
    else:
        conv_input = torch.cat([conv_tail, raw_qkv], dim=1)
        slice_start = conv_tail.shape[1]
    mixed = layer.conv1d(conv_input.transpose(1, 2))
    mixed = F.silu(mixed[:, :, : conv_input.shape[1]]).transpose(1, 2)
    return raw_qkv, mixed[:, slice_start : slice_start + seq_len]


def run_gdn_manual(
    layer,
    modeling_module,
    hidden_states: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    conv_tail: torch.Tensor | None = None,
    chunk_size: int = 64,
):
    batch_size, seq_len, _ = hidden_states.shape
    raw_qkv, mixed_qkv = project_and_conv(layer, hidden_states, conv_tail)
    query, key, value = torch.split(
        mixed_qkv,
        [layer.key_dim, layer.key_dim, layer.value_dim],
        dim=-1,
    )
    query = query.reshape(batch_size, seq_len, -1, layer.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, layer.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, layer.head_v_dim)
    z = layer.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, layer.head_v_dim)
    beta = layer.in_proj_b(hidden_states).sigmoid()
    g = -layer.A_log.float().exp() * F.softplus(
        layer.in_proj_a(hidden_states).float() + layer.dt_bias
    )
    if layer.num_v_heads // layer.num_k_heads > 1:
        repeat = layer.num_v_heads // layer.num_k_heads
        query = query.repeat_interleave(repeat, dim=2)
        key = key.repeat_interleave(repeat, dim=2)
    query = modeling_module.l2norm(query, dim=-1)
    key = modeling_module.l2norm(key, dim=-1)
    core, final_state = modeling_module.torch_chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        chunk_size=chunk_size,
        initial_state=initial_state,
        output_final_state=True,
    )
    core = layer.norm(core.reshape(-1, layer.head_v_dim), z.reshape(-1, layer.head_v_dim))
    output = layer.out_proj(core.reshape(batch_size, seq_len, -1))
    return output, final_state, raw_qkv


def tail_after_append(old_tail: torch.Tensor | None, raw_qkv: torch.Tensor, tail_len: int):
    if tail_len <= 0:
        return None
    if old_tail is None:
        joined = raw_qkv
    else:
        joined = torch.cat([old_tail, raw_qkv], dim=1)
    return joined[:, -tail_len:].contiguous()


def attention_project(layer, hidden_states, position_embeddings, modeling_module):
    q, k, v, gate = layer._project(hidden_states, position_embeddings)
    k = modeling_module.repeat_kv(k, layer.num_key_value_groups)
    v = modeling_module.repeat_kv(v, layer.num_key_value_groups)
    return q, k, v, gate


def run_attention_from_kv(layer, hidden_states, position_embeddings, prefix_key, prefix_value, mask, modeling_module):
    input_shape = hidden_states.shape[:-1]
    q, active_key, active_value, gate = attention_project(
        layer,
        hidden_states,
        position_embeddings,
        modeling_module,
    )
    key = active_key if prefix_key is None else torch.cat([prefix_key, active_key], dim=2)
    value = active_value if prefix_value is None else torch.cat([prefix_value, active_value], dim=2)
    weights = torch.matmul(q, key.transpose(2, 3)) * layer.scaling
    if mask is not None:
        if mask.dtype == torch.bool:
            weights = weights.masked_fill(~mask, torch.finfo(weights.dtype).min)
        else:
            weights = weights + mask
    weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(q.dtype)
    weights = F.dropout(weights, p=layer.attention_dropout, training=layer.training)
    output = torch.matmul(weights, value)
    output = output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    output = output * torch.sigmoid(gate)
    return layer.o_proj(output), active_key, active_value


def exact_active_attention_mask(model_core, full_ids, prefix_len):
    full_mask = model_core._attention_mask(full_ids, labels=None, mdm_split_size=None).to(full_ids.device)
    return full_mask[:, :, prefix_len:, : full_ids.shape[1]]


def describe_mask(model_core, full_ids, prefix_len):
    mask = exact_active_attention_mask(model_core, full_ids, prefix_len)
    if full_ids.shape[1] - prefix_len < 2:
        return "actual_inference_mask=causal_or_degenerate"
    first_active_sees_second = bool(mask[0, 0, 0, prefix_len + 1].item())
    last_active_sees_first = bool(mask[0, 0, -1, prefix_len].item())
    if first_active_sees_second and last_active_sees_first:
        return "actual_inference_mask=block_bidirectional_active_with_prefix"
    if (not first_active_sees_second) and last_active_sees_first:
        return "actual_inference_mask=causal"
    return "actual_inference_mask=other"


def build_state_cache(lm_model, input_ids, modeling_module) -> StateCache:
    core = lm_model.model
    hidden = core.embed_tokens(input_ids)
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
    position_embeddings = core.rotary_emb(hidden, position_ids)
    caches: list[LayerCache] = []

    for layer in core.layers:
        residual = hidden
        normed = layer.input_layernorm(hidden)
        if layer.layer_type == "linear_attention":
            output, final_state, raw_qkv = run_gdn_manual(
                layer.linear_attn,
                modeling_module,
                normed,
            )
            caches.append(
                LayerCache(
                    kind="linear_attention",
                    gdn_state=final_state.detach().clone(),
                    conv_tail=raw_qkv[:, -(layer.linear_attn.conv_kernel_size - 1) :].detach().clone(),
                )
            )
        else:
            q, key, value, gate = attention_project(
                layer.self_attn,
                normed,
                position_embeddings,
                modeling_module,
            )
            del q, gate
            full_mask = core._attention_mask(input_ids, labels=None, mdm_split_size=None).to(input_ids.device)
            output = layer.self_attn(
                hidden_states=normed,
                position_embeddings=position_embeddings,
                attention_mask=full_mask,
            )
            caches.append(
                LayerCache(
                    kind="full_attention",
                    key=key.detach().clone(),
                    value=value.detach().clone(),
                )
            )
        hidden = residual + output
        residual = hidden
        hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))

    logits = lm_model.lm_head(core.norm(hidden))
    return StateCache(
        seq_len=seq_len,
        layer_caches=tuple(caches),
        last_token_logits=logits[:, -1:, :].detach().clone() if seq_len else None,
    )


def cached_block_forward(
    lm_model,
    block_ids,
    cache: StateCache,
    modeling_module,
    *,
    update_cache: bool,
):
    core = lm_model.model
    if block_ids.shape[1] == 0:
        return torch.empty(block_ids.shape[0], 0, lm_model.vocab_size, device=block_ids.device), cache

    hidden = core.embed_tokens(block_ids)
    block_len = block_ids.shape[1]
    position_ids = torch.arange(
        cache.seq_len,
        cache.seq_len + block_len,
        device=block_ids.device,
    ).unsqueeze(0).expand(block_ids.shape[0], -1)
    position_embeddings = core.rotary_emb(hidden, position_ids)
    full_ids = torch.cat(
        [
            torch.zeros(block_ids.shape[0], cache.seq_len, dtype=block_ids.dtype, device=block_ids.device),
            block_ids,
        ],
        dim=1,
    )
    attention_mask = exact_active_attention_mask(core, full_ids, cache.seq_len)
    new_layer_caches: list[LayerCache] = []

    for layer_idx, layer in enumerate(core.layers):
        layer_cache = cache.layer_caches[layer_idx]
        residual = hidden
        normed = layer.input_layernorm(hidden)
        if layer.layer_type == "linear_attention":
            output, final_state, raw_qkv = run_gdn_manual(
                layer.linear_attn,
                modeling_module,
                normed,
                initial_state=layer_cache.gdn_state,
                conv_tail=layer_cache.conv_tail,
            )
            if update_cache:
                new_layer_caches.append(
                    LayerCache(
                        kind="linear_attention",
                        gdn_state=final_state.detach().clone(),
                        conv_tail=tail_after_append(
                            layer_cache.conv_tail,
                            raw_qkv,
                            layer.linear_attn.conv_kernel_size - 1,
                        ).detach().clone(),
                    )
                )
            else:
                new_layer_caches.append(layer_cache)
        else:
            output, active_key, active_value = run_attention_from_kv(
                layer.self_attn,
                normed,
                position_embeddings,
                layer_cache.key,
                layer_cache.value,
                attention_mask,
                modeling_module,
            )
            if update_cache:
                new_layer_caches.append(
                    LayerCache(
                        kind="full_attention",
                        key=torch.cat([layer_cache.key, active_key.detach()], dim=2).clone(),
                        value=torch.cat([layer_cache.value, active_value.detach()], dim=2).clone(),
                    )
                )
            else:
                new_layer_caches.append(layer_cache)
        hidden = residual + output
        residual = hidden
        hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))

    logits = lm_model.lm_head(core.norm(hidden))
    if update_cache:
        new_cache = StateCache(
            seq_len=cache.seq_len + block_len,
            layer_caches=tuple(new_layer_caches),
            last_token_logits=logits[:, -1:, :].detach().clone(),
        )
    else:
        new_cache = cache
    return logits, new_cache


def cached_shifted_logits(lm_model, x_t, cache: StateCache, modeling_module):
    block_ids = x_t[:, cache.seq_len :]
    logits, _ = cached_block_forward(lm_model, block_ids, cache, modeling_module, update_cache=False)
    if cache.last_token_logits is None:
        return torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
    return torch.cat([cache.last_token_logits, logits[:, :-1, :]], dim=1)


def sample_with_top_p(model, logits, top_p, temperature):
    return model.sample_with_top_p(logits, top_p=top_p, temperature=temperature)


def cached_full_context_sample(model, input_ids, args, modeling_module, *, trace=None):
    output_ids = input_ids
    original_len = input_ids.shape[1]
    cache = build_state_cache(model, output_ids, modeling_module)
    while output_ids.shape[1] - original_len < args.max_new_tokens:
        remaining = args.max_new_tokens - (output_ids.shape[1] - original_len)
        block_pad = args.block_size - (output_ids.shape[1] % args.block_size)
        if block_pad == 0:
            block_pad = args.block_size
        block_pad = min(block_pad, remaining)
        masks = torch.full(
            (output_ids.shape[0], block_pad),
            args.mask_id,
            dtype=torch.long,
            device=output_ids.device,
        )
        x_t = torch.cat([output_ids, masks], dim=1)
        while (x_t[:, -block_pad:] == args.mask_id).any():
            window_len = min(args.block_size, x_t.shape[1])
            num_intervals = (window_len + args.small_block_size - 1) // args.small_block_size
            for interval_idx in range(num_intervals):
                start = interval_idx * args.small_block_size
                end = min(window_len, start + args.small_block_size)
                while True:
                    mask_idx = x_t[:, -window_len:] == args.mask_id
                    current_mask = mask_idx[:, start:end]
                    if current_mask.sum() == 0:
                        break
                    active_shifted = cached_shifted_logits(model, x_t, cache, modeling_module)
                    prefix_in_window = window_len - block_pad
                    window_logits = torch.zeros(
                        x_t.shape[0],
                        window_len,
                        active_shifted.shape[-1],
                        device=x_t.device,
                        dtype=active_shifted.dtype,
                    )
                    window_logits[:, prefix_in_window:] = active_shifted
                    logits = window_logits[:, start:end]

                    if trace is not None:
                        full_logits = model(input_ids=x_t, use_cache=False).logits
                        full_shifted = torch.cat([full_logits[:, :1, :], full_logits[:, :-1, :]], dim=1)
                        full_slice = full_shifted[:, -window_len:][:, start:end]
                        trace.append(max_abs_diff(full_slice[:, current_mask[0]], logits[:, current_mask[0]]))

                    x_1, p_1t = sample_with_top_p(model, logits, args.top_p, args.temperature)
                    x1_p = torch.squeeze(torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1)
                    x1_p = torch.where(current_mask, x1_p, -torch.inf)
                    unmask_idx = x1_p > args.threshold
                    max_prob_idx = x1_p.argmax(dim=-1)
                    unmask_idx[torch.arange(x_1.shape[0], device=x_1.device), max_prob_idx] = True
                    unmask_idx = unmask_idx & current_mask
                    window = x_t[:, -window_len:]
                    span = window[:, start:end].clone()
                    span[unmask_idx] = x_1[unmask_idx]
                    window[:, start:end] = span
                    x_t[:, -window_len:] = window
            if (x_t[:, -block_pad:] == args.mask_id).all():
                break
        finalized = x_t[:, output_ids.shape[1] :]
        _, cache = cached_block_forward(model, finalized, cache, modeling_module, update_cache=True)
        output_ids = x_t
        generated = output_ids[:, original_len:]
        stop_positions = (generated == args.stop_token_id).nonzero(as_tuple=False)
        if len(stop_positions):
            stop_idx = int(stop_positions[0, 1].item()) + 1
            return output_ids[:, : original_len + stop_idx][0]
    return output_ids[0]


def cached_generation_functions_sample(model, input_ids, args, modeling_module, *, trace=None):
    # This mirrors the single-sample, no-finished-row path of v2 batch_sample,
    # but uses the same committed-prefix cache primitive for denoise logits.
    seq_len = input_ids.shape[1]
    output_ids = input_ids
    cache = build_state_cache(model, output_ids, modeling_module)
    num_blocks = args.max_new_tokens // args.block_size + seq_len // args.block_size
    start_block_idx = seq_len // args.block_size
    for block_idx in range(start_block_idx, num_blocks):
        if seq_len // args.block_size == block_idx:
            mask_len = args.block_size - input_ids.shape[1] % args.block_size
            if mask_len == 0:
                mask_len = args.block_size
            x_t = torch.cat(
                [
                    input_ids,
                    torch.full((1, mask_len), args.mask_id, dtype=torch.long, device=input_ids.device),
                ],
                dim=1,
            )
        else:
            x_t = input_ids[:, : (block_idx + 1) * args.block_size]
        while True:
            mask_idx = x_t[:, -args.block_size :] == args.mask_id
            if mask_idx.sum() == 0:
                finalized = x_t[:, output_ids.shape[1] :]
                commit_logits, cache = cached_block_forward(
                    model,
                    finalized,
                    cache,
                    modeling_module,
                    update_cache=True,
                )
                next_token = commit_logits[:, -1:, :].argmax(dim=-1)
                x_t = torch.cat([x_t, next_token], dim=1)
                break
            for small_block_idx in range(args.block_size // args.small_block_size):
                start = -args.block_size + small_block_idx * args.small_block_size
                end = None if small_block_idx == (args.block_size // args.small_block_size - 1) else start + args.small_block_size
                while True:
                    mask_idx = x_t[:, -args.block_size :] == args.mask_id
                    current_mask = mask_idx[:, start:end]
                    if current_mask.sum() == 0:
                        break
                    active_shifted = cached_shifted_logits(model, x_t, cache, modeling_module)
                    active_len = x_t.shape[1] - cache.seq_len
                    prefix_in_window = args.block_size - active_len
                    window_logits = torch.zeros(
                        x_t.shape[0],
                        args.block_size,
                        active_shifted.shape[-1],
                        device=x_t.device,
                        dtype=active_shifted.dtype,
                    )
                    window_logits[:, prefix_in_window:] = active_shifted
                    logits = window_logits[:, start:end]
                    if trace is not None:
                        full_logits = model(input_ids=x_t[:, -args.block_size :], use_cache=True).logits
                        full_shifted = torch.cat([full_logits[:, :1, :], full_logits[:, :-1, :]], dim=1)
                        full_slice = full_shifted[:, start:end]
                        trace.append(max_abs_diff(full_slice[:, current_mask[0]], logits[:, current_mask[0]]))
                    x_1, p_1t = sample_with_top_p(model, logits, args.top_p, args.temperature)
                    x1_p = torch.squeeze(torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1)
                    x1_p = torch.where(current_mask, x1_p, -torch.inf)
                    unmask_idx = x1_p > args.threshold
                    max_prob_idx = x1_p.argmax(dim=-1)
                    unmask_idx[torch.arange(x_1.shape[0], device=x_1.device), max_prob_idx] = True
                    unmask_idx = unmask_idx & current_mask
                    x_t[:, start:end][unmask_idx] = x_1[unmask_idx]
        input_ids = x_t
    return {0: input_ids[0]}


def validate_attention_kv(lm_model, modeling_module, atol):
    core = lm_model.model
    layer = next(layer for layer in core.layers if layer.layer_type == "full_attention")
    prefix_len, block_len = 5, 4
    hidden = torch.randn(1, prefix_len + block_len, core.config.hidden_size)
    normed = layer.input_layernorm(hidden)
    pos = torch.arange(prefix_len + block_len).unsqueeze(0)
    emb_full = core.rotary_emb(normed, pos)
    full_mask = core._attention_mask(
        torch.zeros(1, prefix_len + block_len, dtype=torch.long),
        labels=None,
        mdm_split_size=None,
    )
    full = layer.self_attn(normed, emb_full, attention_mask=full_mask)[:, prefix_len:]

    emb_prefix = core.rotary_emb(normed[:, :prefix_len], pos[:, :prefix_len])
    _, prefix_key, prefix_value, _ = attention_project(
        layer.self_attn,
        normed[:, :prefix_len],
        emb_prefix,
        modeling_module,
    )
    emb_active = core.rotary_emb(normed[:, prefix_len:], pos[:, prefix_len:])
    active_mask = full_mask[:, :, prefix_len:, :]
    cached, _, _ = run_attention_from_kv(
        layer.self_attn,
        normed[:, prefix_len:],
        emb_active,
        prefix_key,
        prefix_value,
        active_mask,
        modeling_module,
    )
    diff = max_abs_diff(full, cached)
    return diff, diff <= atol


def validate_whole_model_logits(lm_model, modeling_module, atol):
    prefix_len, block_len = 5, 7
    prefix = torch.randint(3, lm_model.config.vocab_size, (1, prefix_len))
    block = torch.randint(3, lm_model.config.vocab_size, (1, block_len))
    full_ids = torch.cat([prefix, block], dim=1)
    cache = build_state_cache(lm_model, prefix, modeling_module)
    cached_logits = cached_shifted_logits(lm_model, full_ids, cache, modeling_module)
    full_logits = lm_model(input_ids=full_ids, use_cache=False).logits
    full_shifted = torch.cat([full_logits[:, :1, :], full_logits[:, :-1, :]], dim=1)[:, prefix_len:]
    diff = max_abs_diff(full_shifted, cached_logits)
    mask_description = describe_mask(lm_model.model, full_ids, prefix_len)
    return diff, diff <= atol, mask_description


def validate_toolcall_full_context_entry(lm_model, modeling_module, atol):
    sys.path.insert(0, str(ROOT / "scripts"))
    toolcall_module = importlib.import_module("eval_fastdllm_toolcall_cases")
    args = SimpleNamespace(
        max_new_tokens=10,
        block_size=8,
        small_block_size=4,
        mask_id=1,
        stop_token_id=2,
        threshold=0.9,
        temperature=0.0,
        top_p=0.95,
        _last_sampler_schedule_events={},
        guard_tool_json_prefix=False,
        json_prefix_guard_kinds=set(),
    )
    input_ids = torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.long)
    trace: list[float] = []
    golden = toolcall_module.full_context_sample(
        lm_model,
        input_ids,
        DummyTokenizer(),
        args,
        sampler_schedule=None,
    )
    cached = cached_full_context_sample(lm_model, input_ids, args, modeling_module, trace=trace)
    token_exact = torch.equal(golden, cached)
    max_logit_diff = max(trace) if trace else 0.0
    return token_exact, max_logit_diff, max_logit_diff <= atol, golden, cached


def validate_generation_functions_entry(lm_model, modeling_module, atol):
    sys.path.insert(0, str(ROOT / "fast-dllm" / "v2"))
    generation_functions = importlib.import_module("generation_functions")
    proxy = DeviceProxy(lm_model, "cpu")
    args = SimpleNamespace(
        max_new_tokens=8,
        block_size=8,
        small_block_size=4,
        mask_id=1,
        stop_token_id=2,
        threshold=0.9,
        temperature=0.0,
        top_p=0.95,
    )
    input_ids = torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.long)
    trace: list[float] = []
    golden = generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample(
        proxy,
        input_ids,
        tokenizer=DummyTokenizer(),
        block_size=args.block_size,
        small_block_size=args.small_block_size,
        max_new_tokens=args.max_new_tokens,
        mask_id=args.mask_id,
        min_len=input_ids.shape[1],
        seq_len=torch.tensor([input_ids.shape[1]]),
        use_block_cache=False,
        threshold=args.threshold,
        stop_token=args.stop_token_id,
        top_p=args.top_p,
        temperature=args.temperature,
    )[0]
    cached = cached_generation_functions_sample(lm_model, input_ids, args, modeling_module, trace=trace)[0]
    token_exact = torch.equal(golden, cached)
    max_logit_diff = max(trace) if trace else 0.0
    return token_exact, max_logit_diff, max_logit_diff <= atol, golden, cached


def main() -> int:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    torch.manual_seed(args.seed)
    model_dir = (ROOT / args.model_dir).resolve()
    config_module, modeling_module = load_local_bridge(model_dir)
    config = make_small_config(config_module)
    lm_model = modeling_module.Fast_dLLM_Qwen3_5ForCausalLM(config).eval()

    print("Qwen3.5 state-cache sampler CPU validation")
    print(f"model_dir={model_dir}")
    print(f"seed={args.seed} threads={torch.get_num_threads()} atol={args.atol}")
    print(f"small_config layers={config.layer_types} hidden={config.hidden_size} bd_size={config.bd_size}")

    attn_diff, attn_ok = validate_attention_kv(lm_model, modeling_module, args.atol)
    print(f"attention_kv_max_abs_diff={attn_diff:.6g} status={'MATCH' if attn_ok else 'MISMATCH'}")

    logits_diff, logits_ok, mask_description = validate_whole_model_logits(
        lm_model,
        modeling_module,
        args.atol,
    )
    print(mask_description)
    print(f"whole_model_shifted_logits_max_abs_diff={logits_diff:.6g} status={'MATCH' if logits_ok else 'MISMATCH'}")

    tool_exact, tool_logit_diff, tool_logit_ok, tool_golden, tool_cached = validate_toolcall_full_context_entry(
        lm_model,
        modeling_module,
        args.atol,
    )
    print(
        "toolcall_full_context_sample "
        f"token_exact={tool_exact} max_step_logit_abs_diff={tool_logit_diff:.6g} "
        f"status={'MATCH' if tool_exact and tool_logit_ok else 'MISMATCH'}"
    )
    if not tool_exact:
        print(f"toolcall_golden={tool_golden.tolist()}")
        print(f"toolcall_cached={tool_cached.tolist()}")

    gen_exact, gen_logit_diff, gen_logit_ok, gen_golden, gen_cached = validate_generation_functions_entry(
        lm_model,
        modeling_module,
        args.atol,
    )
    print(
        "generation_functions_batch_sample "
        f"token_exact={gen_exact} max_step_logit_abs_diff={gen_logit_diff:.6g} "
        f"status={'MATCH' if gen_exact and gen_logit_ok else 'MISMATCH'}"
    )
    if not gen_exact:
        print(f"generation_golden={gen_golden.tolist()}")
        print(f"generation_cached={gen_cached.tolist()}")

    all_ok = attn_ok and logits_ok and tool_exact and tool_logit_ok and gen_exact and gen_logit_ok
    print(f"FINAL: {'MATCH' if all_ok else 'MISMATCH'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
