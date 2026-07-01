#!/usr/bin/env python3
"""HF route-I FLARE serving cache primitives.

The cache is per request batch and doc-anchored.  It stores only clean-stream
boundary carriers, then serves the active noisy block from those carriers.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def unwrap_lm_model(model):
    if hasattr(model, "get_base_model"):
        try:
            return model.get_base_model()
        except Exception:
            pass
    return model


def modeling_module_for(model):
    lm_model = unwrap_lm_model(model)
    return sys.modules[lm_model.__class__.__module__]


def _set_block_size(lm_model, block_size: int) -> None:
    if hasattr(lm_model, "_set_active_train_bd_size"):
        lm_model._set_active_train_bd_size(int(block_size))
    elif hasattr(lm_model, "model") and hasattr(lm_model.model, "bd_size"):
        lm_model.model.bd_size = int(block_size)


def _assert_route_i(lm_model, modeling_module) -> None:
    route = getattr(modeling_module, "flare_gdn_route", lambda: "route_i")()
    if route != "route_i":
        raise ValueError(f"FLARE HF cache requires route_i, got {route!r}")


@dataclass
class FlareLayerCache:
    kind: str
    gdn_state: torch.Tensor | None = None
    conv_tail: torch.Tensor | None = None
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None


@dataclass
class RequestDiffusionState:
    """Batched lockstep clean-boundary state for one active FLARE block."""

    block_start: int
    block_size: int
    batch_size: int
    layer_caches: list[FlareLayerCache]
    last_token_logits: torch.Tensor | None = None
    read_calls: int = 0
    advance_calls: int = 0
    residual_full_context_model_calls: int = 0

    @classmethod
    @torch.no_grad()
    def reset(cls, model, input_ids: torch.Tensor, block_size: int) -> "RequestDiffusionState":
        lm_model = unwrap_lm_model(model)
        modeling_module = modeling_module_for(lm_model)
        _assert_route_i(lm_model, modeling_module)
        _set_block_size(lm_model, block_size)
        state = cls(
            block_start=0,
            block_size=int(block_size),
            batch_size=int(input_ids.shape[0]),
            layer_caches=_empty_layer_caches(lm_model, input_ids),
            last_token_logits=None,
        )
        prompt_full_blocks = input_ids.shape[1] // int(block_size)
        for block_idx in range(prompt_full_blocks):
            start = block_idx * int(block_size)
            state.advance(model, input_ids[:, start : start + int(block_size)])
        return state

    @torch.no_grad()
    def shifted_active_logits(self, model, x_t: torch.Tensor) -> torch.Tensor:
        """Return train-matched shifted logits for x_t[:, block_start:].

        The returned tensor is active-block-only.  Callers that set
        window_len = x_t.shape[1] - block_start can use their normal
        ``[:, -window_len:]`` slicing without materializing prefix logits.
        """
        self._check_batch(x_t)
        if x_t.shape[1] < self.block_start:
            raise ValueError(
                f"x_t length {x_t.shape[1]} is before cache block_start={self.block_start}"
            )
        block_ids = x_t[:, self.block_start :]
        if block_ids.shape[1] <= 0:
            raise ValueError("FLARE cache read requires a non-empty active block")
        if block_ids.shape[1] > self.block_size:
            raise ValueError(
                f"active block length {block_ids.shape[1]} exceeds block_size={self.block_size}"
            )
        noisy_logits = cached_noisy_block_logits(model, block_ids, self)
        if self.block_start == 0:
            shifted = torch.cat([noisy_logits[:, :1, :], noisy_logits[:, :-1, :]], dim=1)
        else:
            if self.last_token_logits is None:
                raise RuntimeError("missing previous-token logits for shifted FLARE cache read")
            shifted = torch.cat([self.last_token_logits.to(noisy_logits.dtype), noisy_logits[:, :-1, :]], dim=1)
        self.read_calls += 1
        return shifted

    @torch.no_grad()
    def advance(self, model, block_ids: torch.Tensor) -> None:
        self._check_batch(block_ids)
        if block_ids.shape[1] != self.block_size:
            raise ValueError(
                f"advance() requires a full committed block of {self.block_size} tokens, "
                f"got {block_ids.shape[1]}"
            )
        # The next block's first shifted logit is the previous position's
        # FLARE noisy-stream logit, evaluated with the committed block clean.
        last_shift_logits = cached_noisy_block_logits(model, block_ids, self)[:, -1:, :].detach().clone()
        clean_advance_block(model, block_ids, self)
        self.last_token_logits = last_shift_logits
        self.block_start += int(block_ids.shape[1])
        self.advance_calls += 1

    def free(self) -> None:
        self.layer_caches.clear()
        self.last_token_logits = None

    def stats(self) -> dict[str, int]:
        return {
            "block_start": int(self.block_start),
            "block_size": int(self.block_size),
            "batch_size": int(self.batch_size),
            "read_calls": int(self.read_calls),
            "advance_calls": int(self.advance_calls),
            "residual_full_context_model_calls": int(self.residual_full_context_model_calls),
        }

    def _check_batch(self, tensor: torch.Tensor) -> None:
        if int(tensor.shape[0]) != self.batch_size:
            raise ValueError(f"batch mismatch: cache B={self.batch_size}, tensor B={tensor.shape[0]}")


def _empty_layer_caches(lm_model, input_ids: torch.Tensor) -> list[FlareLayerCache]:
    batch_size = int(input_ids.shape[0])
    device = input_ids.device
    caches: list[FlareLayerCache] = []
    for layer in lm_model.model.layers:
        if layer.layer_type == "linear_attention":
            gdn = layer.linear_attn
            caches.append(
                FlareLayerCache(
                    kind="linear_attention",
                    gdn_state=torch.zeros(
                        batch_size,
                        gdn.num_v_heads,
                        gdn.head_k_dim,
                        gdn.head_v_dim,
                        dtype=torch.float32,
                        device=device,
                    ),
                    conv_tail=None,
                )
            )
        elif layer.layer_type == "full_attention":
            caches.append(FlareLayerCache(kind="full_attention", key=None, value=None))
        else:
            raise ValueError(f"unsupported layer_type={layer.layer_type!r}")
    return caches


def _tail_after_append(old_tail: torch.Tensor | None, raw_qkv: torch.Tensor, tail_len: int):
    if tail_len <= 0:
        return None
    if old_tail is None or old_tail.numel() == 0:
        joined = raw_qkv
    else:
        joined = torch.cat([old_tail, raw_qkv], dim=1)
    return joined[:, -tail_len:].detach().clone().contiguous()


def attention_project(layer, hidden_states, position_embeddings, modeling_module):
    q, k, v, gate = layer._project(hidden_states, position_embeddings)
    return q, k, v, gate


def run_attention_from_kv(
    layer,
    hidden_states,
    position_embeddings,
    prefix_key,
    prefix_value,
    mask,
    modeling_module,
):
    input_shape = hidden_states.shape[:-1]
    query, active_key, active_value, gate = attention_project(
        layer,
        hidden_states,
        position_embeddings,
        modeling_module,
    )
    has_prefix = prefix_key is not None and prefix_key.numel() > 0
    key = active_key if not has_prefix else torch.cat([prefix_key, active_key], dim=2)
    value = active_value if not has_prefix else torch.cat([prefix_value, active_value], dim=2)
    key = modeling_module.repeat_kv(key, layer.num_key_value_groups)
    value = modeling_module.repeat_kv(value, layer.num_key_value_groups)
    weights = torch.matmul(query, key.transpose(2, 3)) * layer.scaling
    if mask is not None:
        if mask.dtype == torch.bool:
            weights = weights.masked_fill(~mask, torch.finfo(weights.dtype).min)
        else:
            weights = weights + mask
    weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
    weights = F.dropout(weights, p=layer.attention_dropout, training=layer.training)
    output = torch.matmul(weights, value)
    output = output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    output = output * torch.sigmoid(gate)
    return layer.o_proj(output), active_key.detach(), active_value.detach()


def append_kv_cache(prefix: torch.Tensor | None, active: torch.Tensor) -> torch.Tensor:
    active = active.detach()
    if prefix is None or prefix.numel() == 0:
        return active.clone()
    prefix_len = int(prefix.shape[2])
    active_len = int(active.shape[2])
    out = prefix.new_empty(prefix.shape[0], prefix.shape[1], prefix_len + active_len, prefix.shape[3])
    out[:, :, :prefix_len, :].copy_(prefix)
    out[:, :, prefix_len:, :].copy_(active)
    return out


def clean_active_attention_mask(batch_size: int, block_len: int, prefix_len: int, device) -> torch.Tensor:
    local = torch.tril(torch.ones(block_len, block_len, dtype=torch.bool, device=device))
    local = local.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, block_len, block_len)
    if prefix_len <= 0:
        return local
    prefix = torch.ones(batch_size, 1, block_len, prefix_len, dtype=torch.bool, device=device)
    return torch.cat([prefix, local], dim=-1)


def noisy_active_attention_mask(batch_size: int, block_len: int, prefix_len: int, device) -> torch.Tensor:
    return torch.ones(batch_size, 1, block_len, prefix_len + block_len, dtype=torch.bool, device=device)


@torch.no_grad()
def clean_advance_block(model, block_ids: torch.Tensor, state: RequestDiffusionState) -> None:
    lm_model = unwrap_lm_model(model)
    modeling_module = modeling_module_for(lm_model)
    _assert_route_i(lm_model, modeling_module)
    _set_block_size(lm_model, state.block_size)
    core = lm_model.model
    hidden = core.embed_tokens(block_ids)
    block_len = int(block_ids.shape[1])
    position_ids = torch.arange(
        state.block_start,
        state.block_start + block_len,
        device=block_ids.device,
    ).unsqueeze(0).expand(block_ids.shape[0], -1)
    position_embeddings = core.rotary_emb(hidden, position_ids)
    new_caches: list[FlareLayerCache] = []

    for layer_idx, layer in enumerate(core.layers):
        layer_cache = state.layer_caches[layer_idx]
        residual = hidden
        normed = layer.input_layernorm(hidden)
        if layer.layer_type == "linear_attention":
            output, final_state, chunk_states, raw_qkv = modeling_module.run_gdn_manual_route_i(
                layer.linear_attn,
                normed,
                chunk_size=state.block_size,
                initial_state=layer_cache.gdn_state,
                conv_tail=layer_cache.conv_tail,
                output_chunk_states=True,
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
            output, active_key, active_value = run_attention_from_kv(
                layer.self_attn,
                normed,
                position_embeddings,
                prefix_key,
                prefix_value,
                mask,
                modeling_module,
            )
            key = append_kv_cache(prefix_key, active_key)
            value = append_kv_cache(prefix_value, active_value)
            new_caches.append(FlareLayerCache(kind="full_attention", key=key, value=value))
        hidden = residual + output
        residual = hidden
        hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))

    state.layer_caches = new_caches
    _assert_cache_shapes(state)


@torch.no_grad()
def cached_noisy_block_logits(model, block_ids: torch.Tensor, state: RequestDiffusionState) -> torch.Tensor:
    lm_model = unwrap_lm_model(model)
    modeling_module = modeling_module_for(lm_model)
    _assert_route_i(lm_model, modeling_module)
    _set_block_size(lm_model, state.block_size)
    core = lm_model.model
    hidden = core.embed_tokens(block_ids)
    block_len = int(block_ids.shape[1])
    position_ids = torch.arange(
        state.block_start,
        state.block_start + block_len,
        device=block_ids.device,
    ).unsqueeze(0).expand(block_ids.shape[0], -1)
    position_embeddings = core.rotary_emb(hidden, position_ids)

    for layer_idx, layer in enumerate(core.layers):
        layer_cache = state.layer_caches[layer_idx]
        residual = hidden
        normed = layer.input_layernorm(hidden)
        if layer.layer_type == "linear_attention":
            output, _, _, _ = modeling_module.run_gdn_manual_route_i(
                layer.linear_attn,
                normed,
                chunk_size=state.block_size,
                initial_state=layer_cache.gdn_state,
                conv_tail=layer_cache.conv_tail,
                output_chunk_states=False,
            )
        else:
            prefix_key = layer_cache.key
            prefix_value = layer_cache.value
            prefix_len = 0 if prefix_key is None else int(prefix_key.shape[2])
            mask = noisy_active_attention_mask(block_ids.shape[0], block_len, prefix_len, block_ids.device)
            output, _, _ = run_attention_from_kv(
                layer.self_attn,
                normed,
                position_embeddings,
                prefix_key,
                prefix_value,
                mask,
                modeling_module,
            )
        hidden = residual + output
        residual = hidden
        hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))

    return lm_model.lm_head(core.norm(hidden))


def _assert_cache_shapes(state: RequestDiffusionState) -> None:
    for idx, layer_cache in enumerate(state.layer_caches):
        if layer_cache.kind == "linear_attention":
            if layer_cache.gdn_state is None:
                raise RuntimeError(f"missing GDN state for layer {idx}")
            if layer_cache.gdn_state.dtype != torch.float32:
                raise RuntimeError(f"GDN state layer {idx} must be fp32, got {layer_cache.gdn_state.dtype}")
            continue
        if layer_cache.kind == "full_attention":
            if state.block_start + state.block_size == 0:
                continue
            if layer_cache.key is None or layer_cache.value is None:
                raise RuntimeError(f"missing attention KV for layer {idx}")
            if layer_cache.key.shape[2] != state.block_start + state.block_size:
                raise RuntimeError(
                    f"attention key layer {idx} length={layer_cache.key.shape[2]} "
                    f"does not match committed length={state.block_start + state.block_size}"
                )
            if layer_cache.value.shape[2] != state.block_start + state.block_size:
                raise RuntimeError(
                    f"attention value layer {idx} length={layer_cache.value.shape[2]} "
                    f"does not match committed length={state.block_start + state.block_size}"
                )
            continue
        raise RuntimeError(f"unknown cache kind at layer {idx}: {layer_cache.kind!r}")
