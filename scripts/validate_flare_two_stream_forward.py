#!/usr/bin/env python3
"""Validate a FLARE Stage-1 two-stream forward on tiny CPU random weights.

This harness is intentionally validation-only. It does not load 9B weights,
does not train, and does not change the existing QLoRA/training entrypoint.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


IGNORE_INDEX = -100


@dataclass
class LossParts:
    ar_loss: torch.Tensor
    diff_loss: torch.Tensor
    total_loss: torch.Tensor
    ar_count: int
    diff_count: int
    diff_view0_count: int
    diff_view1_count: int


@dataclass
class TwoStreamOutput:
    clean_logits: torch.Tensor
    noisy_logits: torch.Tensor
    noisy_input_ids: torch.Tensor
    mask_view0: torch.Tensor
    mask_view1: torch.Tensor
    losses: LossParts


@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default="models/qwen3.5-9b-fastdllm-init",
        help="Local Fast-dLLM Qwen3.5 bridge directory.",
    )
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--atol", type=float, default=1e-6)
    return parser.parse_args()


def load_local_bridge(model_dir: Path):
    package_name = "_fastdllm_qwen35_stage1_local"
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


def max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0 and right.numel() == 0:
        return 0.0
    return float((left.float() - right.float()).abs().max().item())


def make_tiny_config(config_module, *, block_size: int = 2):
    return config_module.Fast_dLLM_Qwen3_5Config(
        vocab_size=41,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=128,
        initializer_range=0.05,
        rms_norm_eps=1e-6,
        use_cache=False,
        tie_word_embeddings=False,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=8,
        linear_conv_kernel_dim=3,
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
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000,
            "partial_rotary_factor": 1.0,
        },
        bd_size=block_size,
        mask_token="|<MASK>|",
        mask_token_id=3,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        gdn_mode="option_a_causal_gdn_v0",
    )


def make_tiny_model(config_module, modeling_module, *, seed: int, block_size: int = 2):
    torch.manual_seed(seed)
    config = make_tiny_config(config_module, block_size=block_size)
    model = modeling_module.Fast_dLLM_Qwen3_5ForCausalLM(config)
    for module in model.modules():
        if module.__class__.__name__ in {
            "Fast_dLLM_Qwen3_5RMSNorm",
            "Fast_dLLM_Qwen3_5RMSNormGated",
        }:
            module.weight.data.fill_(1.0)
    model.eval()
    return model


def local_position_ids(doc_ids: torch.Tensor) -> torch.Tensor:
    position_ids = torch.zeros_like(doc_ids)
    for row in range(doc_ids.shape[0]):
        last_doc = None
        local_pos = 0
        for col in range(doc_ids.shape[1]):
            current = int(doc_ids[row, col].item())
            if current < 0:
                position_ids[row, col] = 0
                last_doc = None
                local_pos = 0
                continue
            if last_doc is None or current != last_doc:
                local_pos = 0
            else:
                local_pos += 1
            position_ids[row, col] = local_pos
            last_doc = current
    return position_ids


def contiguous_doc_segments(doc_ids_row: torch.Tensor) -> list[tuple[int, int, int]]:
    segments: list[tuple[int, int, int]] = []
    length = doc_ids_row.numel()
    index = 0
    while index < length:
        doc_id = int(doc_ids_row[index].item())
        if doc_id < 0:
            index += 1
            continue
        end = index + 1
        while end < length and int(doc_ids_row[end].item()) == doc_id:
            end += 1
        segments.append((index, end, doc_id))
        index = end
    return segments


def build_clean_causal_mask(doc_ids: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len = doc_ids.shape
    local_pos = local_position_ids(doc_ids)
    mask = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool, device=doc_ids.device)
    for batch in range(batch_size):
        for query in range(seq_len):
            if doc_ids[batch, query] < 0:
                continue
            same_doc = doc_ids[batch] == doc_ids[batch, query]
            causal = local_pos[batch] <= local_pos[batch, query]
            valid = doc_ids[batch] >= 0
            mask[batch, query] = same_doc & causal & valid
    return mask[:, None, :, :]


def build_flare_two_stream_mask(doc_ids: torch.Tensor, block_size: int) -> torch.Tensor:
    """Build [clean, noisy] document-packed FLARE attention mask."""
    batch_size, seq_len = doc_ids.shape
    total_len = seq_len * 2
    local_pos = local_position_ids(doc_ids)
    local_block = torch.div(local_pos, block_size, rounding_mode="floor")
    mask = torch.zeros(batch_size, total_len, total_len, dtype=torch.bool, device=doc_ids.device)

    for batch in range(batch_size):
        for q_index in range(total_len):
            q_clean = q_index < seq_len
            q_pos = q_index if q_clean else q_index - seq_len
            q_doc = int(doc_ids[batch, q_pos].item())
            if q_doc < 0:
                continue
            q_local = int(local_pos[batch, q_pos].item())
            q_block = int(local_block[batch, q_pos].item())
            block_start = q_block * block_size

            for kv_index in range(total_len):
                kv_clean = kv_index < seq_len
                kv_pos = kv_index if kv_clean else kv_index - seq_len
                if int(doc_ids[batch, kv_pos].item()) != q_doc:
                    continue
                kv_local = int(local_pos[batch, kv_pos].item())
                kv_block = int(local_block[batch, kv_pos].item())

                if q_clean:
                    mask[batch, q_index, kv_index] = kv_clean and kv_local <= q_local
                elif kv_clean:
                    mask[batch, q_index, kv_index] = kv_local < block_start
                else:
                    mask[batch, q_index, kv_index] = kv_block == q_block

    return mask[:, None, :, :]


def project_and_conv(gdn_layer, hidden_states: torch.Tensor, conv_tail: torch.Tensor | None = None):
    seq_len = hidden_states.shape[1]
    raw_qkv = gdn_layer.in_proj_qkv(hidden_states)
    if conv_tail is None or conv_tail.numel() == 0:
        conv_input = raw_qkv
        slice_start = 0
    else:
        lag = int(gdn_layer.conv_kernel_size) - 1
        if conv_tail.shape[1] < lag:
            pad = torch.zeros(
                conv_tail.shape[0],
                lag - conv_tail.shape[1],
                conv_tail.shape[2],
                dtype=conv_tail.dtype,
                device=conv_tail.device,
            )
            conv_tail = torch.cat([pad, conv_tail], dim=1)
        conv_tail = conv_tail[:, -lag:]
        conv_input = torch.cat([conv_tail, raw_qkv], dim=1)
        slice_start = conv_tail.shape[1]

    conv_all = gdn_layer.conv1d(conv_input.transpose(1, 2))
    conv_all = F.silu(conv_all[:, :, : conv_input.shape[1]]).transpose(1, 2)
    mixed_qkv = conv_all[:, slice_start : slice_start + seq_len]
    return raw_qkv, mixed_qkv


def run_gdn_manual(
    gdn_layer,
    modeling_module,
    hidden_states: torch.Tensor,
    *,
    chunk_size: int,
    initial_state: torch.Tensor | None = None,
    conv_tail: torch.Tensor | None = None,
    output_chunk_states: bool = False,
):
    batch_size, seq_len, _ = hidden_states.shape
    raw_qkv, mixed_qkv = project_and_conv(gdn_layer, hidden_states, conv_tail=conv_tail)
    query, key, value = torch.split(
        mixed_qkv,
        [gdn_layer.key_dim, gdn_layer.key_dim, gdn_layer.value_dim],
        dim=-1,
    )
    query = query.reshape(batch_size, seq_len, -1, gdn_layer.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, gdn_layer.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, gdn_layer.head_v_dim)
    z = gdn_layer.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, gdn_layer.head_v_dim)
    beta = gdn_layer.in_proj_b(hidden_states).sigmoid()
    g = -gdn_layer.A_log.float().exp() * F.softplus(
        gdn_layer.in_proj_a(hidden_states).float() + gdn_layer.dt_bias
    )

    if gdn_layer.num_v_heads // gdn_layer.num_k_heads > 1:
        repeat = gdn_layer.num_v_heads // gdn_layer.num_k_heads
        query = query.repeat_interleave(repeat, dim=2)
        key = key.repeat_interleave(repeat, dim=2)

    query = modeling_module.l2norm(query, dim=-1)
    key = modeling_module.l2norm(key, dim=-1)
    delta_outputs = modeling_module.torch_chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        chunk_size=chunk_size,
        initial_state=initial_state,
        output_final_state=True,
        output_chunk_states=output_chunk_states,
    )
    if output_chunk_states:
        core_attn_out, final_state, chunk_states = delta_outputs
    else:
        core_attn_out, final_state = delta_outputs
        chunk_states = None
    core_attn_out = gdn_layer.norm(
        core_attn_out.reshape(-1, gdn_layer.head_v_dim),
        z.reshape(-1, gdn_layer.head_v_dim),
    )
    output = gdn_layer.out_proj(core_attn_out.reshape(batch_size, seq_len, -1))
    return output, final_state, chunk_states, raw_qkv


def clean_gdn_docwise_with_boundaries(
    gdn_layer,
    modeling_module,
    clean_states: torch.Tensor,
    doc_ids: torch.Tensor,
    block_size: int,
):
    batch_size, seq_len, _ = clean_states.shape
    clean_output = torch.zeros_like(clean_states)
    clean_raw_qkv = torch.zeros(
        batch_size,
        seq_len,
        gdn_layer.conv_dim,
        dtype=clean_states.dtype,
        device=clean_states.device,
    )
    boundary_states: dict[tuple[int, int], torch.Tensor] = {}

    for batch in range(batch_size):
        for start, end, _ in contiguous_doc_segments(doc_ids[batch]):
            if (end - start) % block_size:
                raise ValueError("Validation segments must be block-size aligned")
            segment = clean_states[batch : batch + 1, start:end]
            clean_output[batch : batch + 1, start:end] = gdn_layer(segment)
            _, _, chunk_states, raw_qkv = run_gdn_manual(
                gdn_layer,
                modeling_module,
                segment,
                chunk_size=block_size,
                output_chunk_states=True,
            )
            clean_raw_qkv[batch : batch + 1, start:end] = raw_qkv
            assert chunk_states is not None
            zero_state = torch.zeros_like(chunk_states[:, 0])
            num_blocks = (end - start) // block_size
            for block_index in range(num_blocks):
                block_start = start + block_index * block_size
                if block_index == 0:
                    boundary_states[(batch, block_start)] = zero_state
                else:
                    boundary_states[(batch, block_start)] = chunk_states[:, block_index - 1]

    return clean_output, boundary_states, clean_raw_qkv


def noisy_gdn_route_i(
    gdn_layer,
    modeling_module,
    noisy_states: torch.Tensor,
    noisy_doc_ids: torch.Tensor,
    clean_doc_ids: torch.Tensor,
    clean_boundary_states: dict[tuple[int, int], torch.Tensor],
    clean_raw_qkv: torch.Tensor,
    block_size: int,
):
    batch_size = clean_doc_ids.shape[0]
    noisy_output = torch.zeros_like(noisy_states)
    conv_lag = int(gdn_layer.conv_kernel_size) - 1

    for noisy_batch in range(noisy_states.shape[0]):
        clean_batch = noisy_batch % batch_size
        for start, end, _ in contiguous_doc_segments(noisy_doc_ids[noisy_batch]):
            if (end - start) % block_size:
                raise ValueError("Validation segments must be block-size aligned")
            for block_start in range(start, end, block_size):
                block_end = block_start + block_size
                initial_state = clean_boundary_states[(clean_batch, block_start)]
                tail_start = max(start, block_start - conv_lag)
                conv_tail = clean_raw_qkv[clean_batch : clean_batch + 1, tail_start:block_start]
                if conv_tail.numel() == 0:
                    conv_tail = None
                block_output, _, _, _ = run_gdn_manual(
                    gdn_layer,
                    modeling_module,
                    noisy_states[noisy_batch : noisy_batch + 1, block_start:block_end],
                    chunk_size=block_size,
                    initial_state=initial_state,
                    conv_tail=conv_tail,
                )
                noisy_output[noisy_batch : noisy_batch + 1, block_start:block_end] = block_output
    return noisy_output


def compute_flare_losses(
    clean_logits: torch.Tensor,
    noisy_logits: torch.Tensor,
    input_ids: torch.Tensor,
    doc_ids: torch.Tensor,
    mask_view0: torch.Tensor,
    mask_view1: torch.Tensor,
) -> LossParts:
    vocab_size = clean_logits.shape[-1]
    target_valid = (
        (doc_ids[:, :-1] >= 0)
        & (doc_ids[:, 1:] >= 0)
        & (doc_ids[:, :-1] == doc_ids[:, 1:])
    )
    targets = input_ids[:, 1:].contiguous()
    ar_logits = clean_logits[:, :-1].contiguous()
    ar_loss_flat = F.cross_entropy(
        ar_logits.view(-1, vocab_size),
        torch.where(target_valid, targets, torch.full_like(targets, IGNORE_INDEX)).view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    ar_count = int(target_valid.sum().item())
    ar_loss = ar_loss_flat / max(ar_count, 1)

    batch_size = input_ids.shape[0]
    noisy_view0 = noisy_logits[:batch_size, :-1].contiguous()
    noisy_view1 = noisy_logits[batch_size:, :-1].contiguous()
    diff_mask0 = mask_view0[:, 1:] & target_valid
    diff_mask1 = mask_view1[:, 1:] & target_valid
    labels0 = torch.where(diff_mask0, targets, torch.full_like(targets, IGNORE_INDEX))
    labels1 = torch.where(diff_mask1, targets, torch.full_like(targets, IGNORE_INDEX))
    diff_loss0 = F.cross_entropy(
        noisy_view0.view(-1, vocab_size),
        labels0.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    diff_loss1 = F.cross_entropy(
        noisy_view1.view(-1, vocab_size),
        labels1.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    diff_view0_count = int(diff_mask0.sum().item())
    diff_view1_count = int(diff_mask1.sum().item())
    diff_count = diff_view0_count + diff_view1_count
    diff_loss = (diff_loss0 + diff_loss1) / max(ar_count, 1)
    return LossParts(
        ar_loss=ar_loss,
        diff_loss=diff_loss,
        total_loss=ar_loss + diff_loss,
        ar_count=ar_count,
        diff_count=diff_count,
        diff_view0_count=diff_view0_count,
        diff_view1_count=diff_view1_count,
    )


def flare_two_stream_forward(
    model,
    modeling_module,
    input_ids: torch.Tensor,
    doc_ids: torch.Tensor,
    mask_indices: torch.Tensor,
    *,
    block_size: int,
) -> TwoStreamOutput:
    mask_token_id = int(model.config.mask_token_id)
    valid = doc_ids >= 0
    mask_view0 = mask_indices & valid
    mask_view1 = (~mask_indices) & valid
    noisy_view0_ids = torch.where(mask_view0, torch.full_like(input_ids, mask_token_id), input_ids)
    noisy_view1_ids = torch.where(mask_view1, torch.full_like(input_ids, mask_token_id), input_ids)
    noisy_input_ids = torch.cat([noisy_view0_ids, noisy_view1_ids], dim=0)
    noisy_doc_ids = torch.cat([doc_ids, doc_ids], dim=0)

    clean_hidden = model.model.embed_tokens(input_ids)
    noisy_hidden = model.model.embed_tokens(noisy_input_ids)

    clean_mask = build_clean_causal_mask(doc_ids)
    two_stream_mask = build_flare_two_stream_mask(noisy_doc_ids, block_size)
    clean_pos = local_position_ids(doc_ids)
    noisy_pos = local_position_ids(noisy_doc_ids)

    for layer in model.model.layers:
        clean_residual = clean_hidden
        noisy_residual = noisy_hidden
        clean_norm = layer.input_layernorm(clean_hidden)
        noisy_norm = layer.input_layernorm(noisy_hidden)

        if layer.layer_type == "linear_attention":
            clean_attn, clean_boundary_states, clean_raw_qkv = clean_gdn_docwise_with_boundaries(
                layer.linear_attn,
                modeling_module,
                clean_norm,
                doc_ids,
                block_size,
            )
            noisy_attn = noisy_gdn_route_i(
                layer.linear_attn,
                modeling_module,
                noisy_norm,
                noisy_doc_ids,
                doc_ids,
                clean_boundary_states,
                clean_raw_qkv,
                block_size,
            )
        else:
            clean_position_embeddings = model.model.rotary_emb(clean_norm, clean_pos)
            clean_attn = layer.self_attn(
                clean_norm,
                position_embeddings=clean_position_embeddings,
                attention_mask=clean_mask,
                split_size=None,
            )
            clean_for_noisy = clean_norm.repeat(2, 1, 1)
            combined_norm = torch.cat([clean_for_noisy, noisy_norm], dim=1)
            noisy_position_embeddings = model.model.rotary_emb(noisy_norm, noisy_pos)
            combined_attn = layer.self_attn(
                combined_norm,
                position_embeddings=noisy_position_embeddings,
                attention_mask=two_stream_mask,
                split_size=input_ids.shape[1],
            )
            noisy_attn = combined_attn[:, input_ids.shape[1] :]

        clean_hidden = clean_residual + clean_attn
        noisy_hidden = noisy_residual + noisy_attn
        clean_hidden = clean_hidden + layer.mlp(layer.post_attention_layernorm(clean_hidden))
        noisy_hidden = noisy_hidden + layer.mlp(layer.post_attention_layernorm(noisy_hidden))

    clean_logits = model.lm_head(model.model.norm(clean_hidden))
    noisy_logits = model.lm_head(model.model.norm(noisy_hidden))
    losses = compute_flare_losses(
        clean_logits,
        noisy_logits,
        input_ids,
        doc_ids,
        mask_view0,
        mask_view1,
    )
    return TwoStreamOutput(
        clean_logits=clean_logits,
        noisy_logits=noisy_logits,
        noisy_input_ids=noisy_input_ids,
        mask_view0=mask_view0,
        mask_view1=mask_view1,
        losses=losses,
    )


def test_mask_rules(block_size: int) -> TestResult:
    doc_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.long)
    mask = build_flare_two_stream_mask(doc_ids, block_size)[0, 0]
    seq_len = doc_ids.shape[1]

    checks = {
        "clean causal same doc": bool(mask[2, 0] and mask[2, 2] and not mask[2, 3]),
        "clean cannot see noisy": bool(not mask[2, seq_len + 0] and not mask[6, seq_len + 6]),
        "clean doc isolated": bool(not mask[4, 3] and not mask[3, 4]),
        "noisy first block bidir": bool(mask[seq_len + 0, seq_len + 1] and mask[seq_len + 1, seq_len + 0]),
        "noisy first block no clean prefix": bool(not mask[seq_len + 0, 0] and not mask[seq_len + 1, 0]),
        "noisy later block sees preceding clean": bool(mask[seq_len + 2, 0] and mask[seq_len + 2, 1]),
        "noisy later block hides current clean": bool(not mask[seq_len + 2, 2] and not mask[seq_len + 2, 3]),
        "noisy block isolated": bool(mask[seq_len + 2, seq_len + 3] and not mask[seq_len + 2, seq_len + 1]),
        "noisy doc isolated": bool(not mask[seq_len + 5, 0] and not mask[seq_len + 2, seq_len + 5]),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        return TestResult("mask rules", False, "failed=" + ",".join(failed))
    return TestResult("mask rules", True, f"true_edges={int(mask.sum().item())}")


def test_clean_logits_match_ar(config_module, modeling_module, *, seed: int, atol: float, block_size: int) -> TestResult:
    model = make_tiny_model(config_module, modeling_module, seed=seed, block_size=block_size)
    input_ids = torch.tensor([[5, 8, 13, 21, 7, 11, 17, 19]], dtype=torch.long)
    doc_ids = torch.zeros_like(input_ids)
    mask_indices = torch.tensor([[False, True, False, True, True, False, True, False]])

    with torch.inference_mode():
        golden_logits = model(input_ids=input_ids).logits
        two_stream = flare_two_stream_forward(
            model,
            modeling_module,
            input_ids,
            doc_ids,
            mask_indices,
            block_size=block_size,
        )
    diff = max_abs_diff(golden_logits, two_stream.clean_logits)
    passed = diff <= atol
    detail = f"clean_logits_max_abs_diff={diff:.6g} atol={atol:g}"
    return TestResult("clean logits vs AR", passed, detail)


def test_gdn_schedule_doc_reset(modeling_module, config_module, *, seed: int, atol: float, block_size: int) -> TestResult:
    torch.manual_seed(seed + 17)
    config = make_tiny_config(config_module, block_size=block_size)
    gdn = modeling_module.Fast_dLLM_Qwen3_5GatedDeltaNet(config, layer_idx=0)
    gdn.eval()
    hidden = torch.randn(1, 8, config.hidden_size)
    noisy_hidden = torch.randn(1, 8, config.hidden_size)
    doc_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.long)

    with torch.inference_mode():
        native_doc0, _ = gdn(hidden[:, :4], chunk_size=block_size, output_final_state=True)
        manual_doc0, _, chunk_states, raw_qkv_doc0 = run_gdn_manual(
            gdn,
            modeling_module,
            hidden[:, :4],
            chunk_size=block_size,
            output_chunk_states=True,
        )
        prefix_state = chunk_states[:, 0]
        conv_tail = raw_qkv_doc0[:, :block_size]
        seeded_block, _, _, _ = run_gdn_manual(
            gdn,
            modeling_module,
            noisy_hidden[:, block_size : block_size * 2],
            chunk_size=block_size,
            initial_state=prefix_state,
            conv_tail=conv_tail,
        )
        mixed = torch.cat([hidden[:, :block_size], noisy_hidden[:, block_size : block_size * 2]], dim=1)
        mixed_reference, _, _, _ = run_gdn_manual(
            gdn,
            modeling_module,
            mixed,
            chunk_size=block_size,
        )

        doc1_zero_ref, _, _, _ = run_gdn_manual(
            gdn,
            modeling_module,
            noisy_hidden[:, 4:6],
            chunk_size=block_size,
            initial_state=None,
            conv_tail=None,
        )
        _, clean_boundary_states, clean_raw_qkv = clean_gdn_docwise_with_boundaries(
            gdn,
            modeling_module,
            hidden,
            doc_ids,
            block_size,
        )
        route_noisy = noisy_gdn_route_i(
            gdn,
            modeling_module,
            noisy_hidden,
            doc_ids,
            doc_ids,
            clean_boundary_states,
            clean_raw_qkv,
            block_size,
        )
        wrong_doc1, _, _, _ = run_gdn_manual(
            gdn,
            modeling_module,
            noisy_hidden[:, 4:6],
            chunk_size=block_size,
            initial_state=prefix_state,
            conv_tail=raw_qkv_doc0[:, -block_size:],
        )

    native_manual_diff = max_abs_diff(native_doc0, manual_doc0)
    seeded_vs_mixed_diff = max_abs_diff(seeded_block, mixed_reference[:, block_size:])
    route_seeded_diff = max_abs_diff(route_noisy[:, block_size : block_size * 2], seeded_block)
    doc_reset_ref_diff = max_abs_diff(route_noisy[:, 4:6], doc1_zero_ref)
    wrong_reset_sensitivity = max_abs_diff(doc1_zero_ref, wrong_doc1)
    passed = (
        native_manual_diff <= atol
        and seeded_vs_mixed_diff <= atol
        and route_seeded_diff <= atol
        and doc_reset_ref_diff <= atol
        and wrong_reset_sensitivity > atol * 10
    )
    detail = (
        f"native_manual={native_manual_diff:.6g} "
        f"seeded_vs_mixed={seeded_vs_mixed_diff:.6g} "
        f"route_seeded={route_seeded_diff:.6g} "
        f"doc_reset_ref={doc_reset_ref_diff:.6g} "
        f"wrong_cross_doc_sensitivity={wrong_reset_sensitivity:.6g}"
    )
    return TestResult("GDN schedule/doc reset", passed, detail)


def test_loss_logit_shift_indexing() -> TestResult:
    vocab_size = 11
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    doc_ids = torch.zeros_like(input_ids)
    mask_view0 = torch.tensor([[False, True, False, True, False]])
    mask_view1 = ~mask_view0
    clean_logits = torch.full((1, 5, vocab_size), -8.0)
    noisy_logits = torch.full((2, 5, vocab_size), -8.0)

    for target_pos in range(1, input_ids.shape[1]):
        target = int(input_ids[0, target_pos].item())
        clean_logits[0, target_pos - 1, target] = 8.0
        view = 0 if bool(mask_view0[0, target_pos].item()) else 1
        noisy_logits[view, target_pos - 1, target] = 8.0

    shifted_losses = compute_flare_losses(
        clean_logits,
        noisy_logits,
        input_ids,
        doc_ids,
        mask_view0,
        mask_view1,
    )

    no_shift_logits = torch.full_like(noisy_logits, -8.0)
    for target_pos in range(1, input_ids.shape[1]):
        target = int(input_ids[0, target_pos].item())
        view = 0 if bool(mask_view0[0, target_pos].item()) else 1
        no_shift_logits[view, target_pos, target] = 8.0
    no_shift_losses = compute_flare_losses(
        clean_logits,
        no_shift_logits,
        input_ids,
        doc_ids,
        mask_view0,
        mask_view1,
    )

    shifted_total = float(shifted_losses.total_loss.item())
    no_shift_total = float(no_shift_losses.total_loss.item())
    passed = (
        shifted_losses.ar_count == 4
        and shifted_losses.diff_count == 4
        and shifted_losses.diff_view0_count == 2
        and shifted_losses.diff_view1_count == 2
        and shifted_total < 1e-4
        and no_shift_total > shifted_total + 1.0
    )
    detail = (
        f"shifted_total={shifted_total:.6g} no_shift_total={no_shift_total:.6g} "
        f"ar_count={shifted_losses.ar_count} diff_count={shifted_losses.diff_count} "
        f"view_counts={shifted_losses.diff_view0_count}/{shifted_losses.diff_view1_count}"
    )
    return TestResult("loss/logit-shift indexing", passed, detail)


def test_noisy_loss_finite(config_module, modeling_module, *, seed: int, block_size: int) -> TestResult:
    model = make_tiny_model(config_module, modeling_module, seed=seed + 31, block_size=block_size)
    input_ids = torch.tensor([[6, 10, 14, 18, 9, 12, 15, 20]], dtype=torch.long)
    doc_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.long)
    mask_indices = torch.tensor([[False, True, True, False, True, False, True, False]])

    with torch.inference_mode():
        output = flare_two_stream_forward(
            model,
            modeling_module,
            input_ids,
            doc_ids,
            mask_indices,
            block_size=block_size,
        )

    losses = output.losses
    finite = bool(torch.isfinite(losses.ar_loss) and torch.isfinite(losses.diff_loss) and torch.isfinite(losses.total_loss))
    complementary_ok = losses.diff_count == losses.ar_count
    total_ok = abs(float((losses.ar_loss + losses.diff_loss - losses.total_loss).item())) <= 1e-7
    passed = finite and complementary_ok and total_ok and losses.diff_view0_count > 0 and losses.diff_view1_count > 0
    detail = (
        f"L_AR={float(losses.ar_loss.item()):.6g} "
        f"L_diff={float(losses.diff_loss.item()):.6g} "
        f"L_total={float(losses.total_loss.item()):.6g} "
        f"ar_count={losses.ar_count} diff_count={losses.diff_count} "
        f"view_counts={losses.diff_view0_count}/{losses.diff_view1_count}"
    )
    return TestResult("noisy finite/complementary loss", passed, detail)


def test_flare_mask_rate_schedule(config_module, modeling_module, *, seed: int, block_size: int) -> TestResult:
    model = make_tiny_model(config_module, modeling_module, seed=seed + 41, block_size=block_size)
    labels = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
    forced_value_mask = torch.tensor([[True, False, False, False]], dtype=torch.bool)
    env_keys = [
        "FASTDLLM_FLARE_MASK_RATE_MIN",
        "FASTDLLM_FLARE_MASK_RATE_MAX",
        "FASTDLLM_FLARE_ADAPTIVE_COPY_SCHEDULE",
        "FASTDLLM_FLARE_HIGH_ENTROPY_MASK_RATE_MIN",
        "FASTDLLM_FLARE_HIGH_ENTROPY_MASK_RATE_MAX",
    ]
    saved = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ["FASTDLLM_FLARE_MASK_RATE_MIN"] = "1.0"
        os.environ["FASTDLLM_FLARE_MASK_RATE_MAX"] = "1.0"
        os.environ.pop("FASTDLLM_FLARE_ADAPTIVE_COPY_SCHEDULE", None)
        all_mask, all_visible = model._build_flare_mask_views(labels, block_size=block_size)

        os.environ["FASTDLLM_FLARE_MASK_RATE_MIN"] = "0.0"
        os.environ["FASTDLLM_FLARE_MASK_RATE_MAX"] = "0.0"
        no_mask, no_visible = model._build_flare_mask_views(labels, block_size=block_size)

        os.environ["FASTDLLM_FLARE_MASK_RATE_MIN"] = "1.0"
        os.environ["FASTDLLM_FLARE_MASK_RATE_MAX"] = "1.0"
        os.environ["FASTDLLM_FLARE_ADAPTIVE_COPY_SCHEDULE"] = "1"
        os.environ["FASTDLLM_FLARE_HIGH_ENTROPY_MASK_RATE_MIN"] = "0.0"
        os.environ["FASTDLLM_FLARE_HIGH_ENTROPY_MASK_RATE_MAX"] = "0.0"
        adaptive_mask, adaptive_visible = model._build_flare_mask_views(
            labels,
            block_size=block_size,
            forced_value_mask=forced_value_mask,
        )
    finally:
        restore_env(saved)

    expected_adaptive = torch.tensor([[True, True, False, False]], dtype=torch.bool)
    passed = (
        bool(torch.equal(all_mask, torch.ones_like(labels, dtype=torch.bool)))
        and bool(torch.equal(all_visible, torch.zeros_like(labels, dtype=torch.bool)))
        and bool(torch.equal(no_mask, torch.zeros_like(labels, dtype=torch.bool)))
        and bool(torch.equal(no_visible, torch.ones_like(labels, dtype=torch.bool)))
        and bool(torch.equal(adaptive_mask, expected_adaptive))
        and bool(torch.equal(adaptive_visible, ~expected_adaptive))
    )
    detail = (
        f"all_mask={int(all_mask.sum().item())}/{labels.numel()} "
        f"no_mask={int(no_mask.sum().item())}/{labels.numel()} "
        f"adaptive={adaptive_mask.int().tolist()}"
    )
    return TestResult("FLARE clipped/adaptive mask schedule", passed, detail)


def restore_env(saved: dict[str, str | None]):
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def run_production_two_stream(
    model,
    *,
    route: str,
    stride_blocks: int,
    bug: str = "",
    input_ids: torch.Tensor,
    doc_ids: torch.Tensor,
    mask_indices: torch.Tensor,
):
    env_keys = [
        "FASTDLLM_FLARE_TWO_STREAM",
        "FLARE_TWO_STREAM",
        "FASTDLLM_FLARE_GDN_ROUTE",
        "FASTDLLM_FLARE_ROUTE_II_STRIDE_BLOCKS",
        "FASTDLLM_FLARE_ROUTE_II_BUG",
    ]
    saved = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ["FASTDLLM_FLARE_TWO_STREAM"] = "1"
        os.environ["FLARE_TWO_STREAM"] = "1"
        os.environ["FASTDLLM_FLARE_GDN_ROUTE"] = route
        os.environ["FASTDLLM_FLARE_ROUTE_II_STRIDE_BLOCKS"] = str(stride_blocks)
        if bug:
            os.environ["FASTDLLM_FLARE_ROUTE_II_BUG"] = bug
        else:
            os.environ.pop("FASTDLLM_FLARE_ROUTE_II_BUG", None)
        model.train()
        model.zero_grad(set_to_none=True)
        labels = input_ids.clone()
        labels[doc_ids < 0] = IGNORE_INDEX
        output = model(
            input_ids=input_ids,
            labels=labels,
            doc_ids=doc_ids,
            flare_mask_indices=mask_indices,
        )
        output.loss.backward()
        parts = getattr(model, "_last_flare_loss_parts", {})
        grads = {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
        }
        return output.loss.detach().clone(), output.logits.detach().clone(), parts, grads
    finally:
        restore_env(saved)


def compare_grads(grads_i, grads_ii, *, atol: float = 1e-3, rtol: float = 1e-3):
    grad_equal = True
    grad_allclose = True
    grad_mismatches = []
    grad_not_allclose = []
    grad_max_abs = 0.0
    grad_max_rel = 0.0
    grad_max_margin = 0.0
    for name in sorted(grads_i):
        left = grads_i[name]
        right = grads_ii.get(name)
        if left is None or right is None:
            if left is not None or right is not None:
                grad_equal = False
                grad_allclose = False
                grad_mismatches.append(name)
                grad_not_allclose.append(name)
            continue
        diff = (right.float() - left.float()).abs()
        if diff.numel():
            grad_max_abs = max(grad_max_abs, float(diff.max().item()))
            rel = diff / (left.float().abs() + 1e-6)
            grad_max_rel = max(grad_max_rel, float(rel.max().item()))
            tolerance = atol + rtol * left.float().abs()
            margin = diff / tolerance
            grad_max_margin = max(grad_max_margin, float(margin.max().item()))
            if not torch.all(diff <= tolerance):
                grad_allclose = False
                grad_not_allclose.append(name)
        if not torch.equal(left, right):
            grad_equal = False
            grad_mismatches.append(name)
    return {
        "equal": grad_equal,
        "allclose": grad_allclose,
        "mismatches": grad_mismatches,
        "not_allclose": grad_not_allclose,
        "max_abs": grad_max_abs,
        "max_rel": grad_max_rel,
        "max_margin": grad_max_margin,
        "atol": atol,
        "rtol": rtol,
    }


def route_ii_stress_cases(block_size: int):
    return [
        (
            "multi_doc_aligned",
            torch.tensor([[5, 8, 13, 21, 7, 11, 17, 19]], dtype=torch.long),
            torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.long),
            torch.tensor([[False, True, False, True, True, False, True, False]], dtype=torch.bool),
        ),
        (
            "pad_isolated",
            torch.tensor([[6, 10, 14, 18, 0, 0, 0, 0]], dtype=torch.long),
            torch.tensor([[0, 0, 0, 0, -1, -1, -1, -1]], dtype=torch.long),
            torch.tensor([[False, True, True, False, True, False, True, False]], dtype=torch.bool),
        ),
        (
            "doc_boundary_before_noisy_block",
            torch.tensor([[9, 12, 15, 20, 23, 26, 29, 32]], dtype=torch.long),
            torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1]], dtype=torch.long),
            torch.tensor([[False, True, True, False, True, False, False, True]], dtype=torch.bool),
        ),
        (
            "non_aligned_doc_start",
            torch.tensor([[10, 13, 16, 19, 22, 25, 28, 31]], dtype=torch.long),
            torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]], dtype=torch.long),
            torch.tensor([[True, False, True, False, True, True, False, False]], dtype=torch.bool),
        ),
        (
            "partial_final_block",
            torch.tensor([[11, 14, 17, 20, 23, 26, 29, 0]], dtype=torch.long),
            torch.tensor([[0, 0, 0, 0, 1, 1, 1, -1]], dtype=torch.long),
            torch.tensor([[False, True, True, False, True, False, True, False]], dtype=torch.bool),
        ),
    ]


def test_route_ii_equivalence_matrix(
    config_module,
    modeling_module,
    *,
    seed: int,
    block_size: int,
):
    results: list[TestResult] = []
    base_model = make_tiny_model(config_module, modeling_module, seed=seed + 101, block_size=block_size)
    base_state = copy.deepcopy(base_model.state_dict())

    for stride_blocks in (1, 2, 4, 8):
        for case_name, input_ids, doc_ids, mask_indices in route_ii_stress_cases(block_size):
            route_i_model = make_tiny_model(config_module, modeling_module, seed=seed + 102, block_size=block_size)
            route_ii_model = make_tiny_model(config_module, modeling_module, seed=seed + 103, block_size=block_size)
            route_i_model.load_state_dict(base_state)
            route_ii_model.load_state_dict(base_state)

            loss_i, logits_i, parts_i, grads_i = run_production_two_stream(
                route_i_model,
                route="route_i",
                stride_blocks=stride_blocks,
                input_ids=input_ids,
                doc_ids=doc_ids,
                mask_indices=mask_indices,
            )
            loss_ii, logits_ii, parts_ii, grads_ii = run_production_two_stream(
                route_ii_model,
                route="route_ii",
                stride_blocks=stride_blocks,
                input_ids=input_ids,
                doc_ids=doc_ids,
                mask_indices=mask_indices,
            )

            loss_equal = bool(torch.equal(loss_i, loss_ii))
            logits_equal = bool(torch.equal(logits_i, logits_ii))
            loss_abs = float((loss_ii.float() - loss_i.float()).abs().item())
            ar_abs = float((parts_ii["ar"].float() - parts_i["ar"].float()).abs().item())
            diff_abs = float((parts_ii["diff"].float() - parts_i["diff"].float()).abs().item())
            grad_cmp = compare_grads(grads_i, grads_ii)
            ar_equal = torch.equal(parts_i["ar"], parts_ii["ar"])
            diff_equal = torch.equal(parts_i["diff"], parts_ii["diff"])
            passed = loss_abs <= 1e-6 and ar_abs <= 1e-6 and diff_abs <= 1e-6 and grad_cmp["allclose"]
            detail = (
                f"stride={stride_blocks} case={case_name} "
                f"loss_equal={loss_equal} logits_equal={logits_equal} "
                f"L_AR_equal={ar_equal} L_diff_equal={diff_equal} "
                f"loss_abs={loss_abs:.6g} L_AR_abs={ar_abs:.6g} L_diff_abs={diff_abs:.6g} "
                f"grad_equal={grad_cmp['equal']} grad_allclose={grad_cmp['allclose']} "
                f"grad_mismatches={len(grad_cmp['mismatches'])} "
                f"grad_not_allclose={len(grad_cmp['not_allclose'])} "
                f"grad_max_abs={grad_cmp['max_abs']:.6g} "
                f"grad_max_rel={grad_cmp['max_rel']:.6g} "
                f"grad_max_margin={grad_cmp['max_margin']:.6g}"
            )
            if grad_cmp["mismatches"]:
                detail += " first_grad_mismatch=" + grad_cmp["mismatches"][0]
            if grad_cmp["not_allclose"]:
                detail += " first_grad_not_allclose=" + grad_cmp["not_allclose"][0]
            results.append(TestResult("Route-II fp32 bit-exact", passed, detail))
    return results


def test_route_ii_shortconv_tail_control(modeling_module, config_module, *, seed: int, block_size: int) -> TestResult:
    torch.manual_seed(seed + 211)
    config = make_tiny_config(config_module, block_size=block_size)
    gdn = modeling_module.Fast_dLLM_Qwen3_5GatedDeltaNet(config, layer_idx=0)
    gdn.train()
    clean_hidden = torch.randn(1, 4, config.hidden_size)
    noisy_hidden = torch.randn(2, 4, config.hidden_size)
    doc_ids = torch.zeros(1, 4, dtype=torch.long)
    noisy_doc_ids = torch.cat([doc_ids, doc_ids], dim=0)

    clean_attn, noisy_attn = modeling_module.clean_noisy_gdn_route_ii(
        gdn,
        clean_hidden,
        noisy_hidden,
        noisy_doc_ids,
        doc_ids,
        block_size,
        stride_blocks=1,
    )
    _, boundary_states, clean_raw_qkv = modeling_module.clean_gdn_docwise_with_boundaries(
        gdn,
        clean_hidden,
        doc_ids,
        block_size,
    )
    initial_state = boundary_states[(0, block_size)]
    conv_tail = clean_raw_qkv[:, :block_size]
    correct_manual, _, _, _ = modeling_module.run_gdn_manual_route_i(
        gdn,
        noisy_hidden[:1, block_size : block_size * 2],
        chunk_size=block_size,
        initial_state=initial_state,
        conv_tail=conv_tail,
    )
    wrong_zero_tail, _, _, _ = modeling_module.run_gdn_manual_route_i(
        gdn,
        noisy_hidden[:1, block_size : block_size * 2],
        chunk_size=block_size,
        initial_state=initial_state,
        conv_tail=None,
    )
    route_diff = max_abs_diff(noisy_attn[:1, block_size : block_size * 2], correct_manual)
    sensitivity = max_abs_diff(correct_manual, wrong_zero_tail)
    clean_finite = bool(torch.isfinite(clean_attn).all().item())
    passed = route_diff <= 1e-6 and sensitivity > 1e-6 and clean_finite
    detail = f"route_vs_manual={route_diff:.6g} zero_tail_sensitivity={sensitivity:.6g} clean_finite={clean_finite}"
    return TestResult("Route-II ShortConv tail control", passed, detail)


def test_nested_view_gdn_state_discipline(
    modeling_module,
    config_module,
    *,
    seed: int,
    block_size: int,
    atol: float,
) -> TestResult:
    torch.manual_seed(seed + 251)
    config = make_tiny_config(config_module, block_size=block_size)
    gdn = modeling_module.Fast_dLLM_Qwen3_5GatedDeltaNet(config, layer_idx=0)
    gdn.eval()
    clean_hidden = torch.randn(1, block_size * 2, config.hidden_size)
    student_noisy = torch.randn(1, block_size * 2, config.hidden_size)
    teacher_noisy = student_noisy.clone()
    teacher_noisy[:, block_size : block_size + 1] = clean_hidden[:, block_size : block_size + 1]
    nested_noisy = torch.cat([student_noisy, teacher_noisy], dim=0)
    doc_ids = torch.zeros(1, block_size * 2, dtype=torch.long)
    noisy_doc_ids = torch.cat([doc_ids, doc_ids], dim=0)

    with torch.inference_mode():
        _, clean_boundary_states, clean_raw_qkv = clean_gdn_docwise_with_boundaries(
            gdn,
            modeling_module,
            clean_hidden,
            doc_ids,
            block_size,
        )
        seed_state = clean_boundary_states[(0, block_size)]
        seed_before = seed_state.detach().clone()
        conv_tail = clean_raw_qkv[:, :block_size].contiguous()
        conv_tail_before = conv_tail.detach().clone()
        route_out = noisy_gdn_route_i(
            gdn,
            modeling_module,
            nested_noisy,
            noisy_doc_ids,
            doc_ids,
            clean_boundary_states,
            clean_raw_qkv,
            block_size,
        )
        manual_student, _, _, _ = run_gdn_manual(
            gdn,
            modeling_module,
            student_noisy[:, block_size : block_size * 2],
            chunk_size=block_size,
            initial_state=seed_state,
            conv_tail=conv_tail,
        )
        manual_teacher, _, _, _ = run_gdn_manual(
            gdn,
            modeling_module,
            teacher_noisy[:, block_size : block_size * 2],
            chunk_size=block_size,
            initial_state=seed_state,
            conv_tail=conv_tail,
        )
        clean_commit, clean_commit_state, _, _ = run_gdn_manual(
            gdn,
            modeling_module,
            clean_hidden[:, block_size : block_size * 2],
            chunk_size=block_size,
            initial_state=seed_state,
            conv_tail=conv_tail,
        )
        full_clean, full_clean_state = gdn(
            clean_hidden,
            chunk_size=block_size,
            output_final_state=True,
        )

    route_student_diff = max_abs_diff(route_out[:1, block_size : block_size * 2], manual_student)
    route_teacher_diff = max_abs_diff(route_out[1:2, block_size : block_size * 2], manual_teacher)
    seed_readonly_diff = max(
        max_abs_diff(seed_state, seed_before),
        max_abs_diff(conv_tail, conv_tail_before),
    )
    commit_output_diff = max_abs_diff(full_clean[:, block_size : block_size * 2], clean_commit)
    commit_state_diff = max_abs_diff(full_clean_state, clean_commit_state)
    nested_views_differ = max_abs_diff(manual_student, manual_teacher)
    passed = (
        route_student_diff <= atol
        and route_teacher_diff <= atol
        and seed_readonly_diff <= atol
        and commit_output_diff <= atol
        and commit_state_diff <= atol
        and nested_views_differ > atol * 10
    )
    detail = (
        f"route_student={route_student_diff:.6g} "
        f"route_teacher={route_teacher_diff:.6g} "
        f"seed_readonly={seed_readonly_diff:.6g} "
        f"clean_commit_out={commit_output_diff:.6g} "
        f"clean_commit_state={commit_state_diff:.6g} "
        f"nested_view_sensitivity={nested_views_differ:.6g}"
    )
    return TestResult("nested-view GDN state discipline", passed, detail)


def test_route_ii_sensitivity_controls(
    config_module,
    modeling_module,
    *,
    seed: int,
    block_size: int,
) -> list[TestResult]:
    input_ids = torch.tensor([[5, 8, 13, 21, 7, 11, 17, 19]], dtype=torch.long)
    doc_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.long)
    mask_indices = torch.tensor([[False, True, False, True, True, False, True, False]], dtype=torch.bool)
    stride_blocks = 2
    base_model = make_tiny_model(config_module, modeling_module, seed=seed + 301, block_size=block_size)
    base_state = copy.deepcopy(base_model.state_dict())

    route_i_model = make_tiny_model(config_module, modeling_module, seed=seed + 302, block_size=block_size)
    route_i_model.load_state_dict(base_state)
    loss_i, _, parts_i, grads_i = run_production_two_stream(
        route_i_model,
        route="route_i",
        stride_blocks=stride_blocks,
        input_ids=input_ids,
        doc_ids=doc_ids,
        mask_indices=mask_indices,
    )

    controls = [
        ("legit", ""),
        ("zero_seed", "zero_seed"),
        ("doc_reset", "doc_reset"),
        ("window_offset", "window_offset"),
        ("zero_tail", "zero_tail"),
    ]
    results: list[TestResult] = []
    for label, bug in controls:
        route_ii_model = make_tiny_model(config_module, modeling_module, seed=seed + 303, block_size=block_size)
        route_ii_model.load_state_dict(base_state)
        loss_j, _, parts_j, grads_j = run_production_two_stream(
            route_ii_model,
            route="route_ii",
            stride_blocks=stride_blocks,
            bug=bug,
            input_ids=input_ids,
            doc_ids=doc_ids,
            mask_indices=mask_indices,
        )
        grad_cmp = compare_grads(grads_i, grads_j)
        loss_abs = float((loss_j.float() - loss_i.float()).abs().item())
        diff_abs = float((parts_j["diff"].float() - parts_i["diff"].float()).abs().item())
        if label == "legit":
            passed = loss_abs == 0.0 and diff_abs == 0.0 and grad_cmp["allclose"]
        else:
            passed = loss_abs > 1e-3 and (not grad_cmp["allclose"]) and grad_cmp["max_margin"] > 100.0
        detail = (
            f"control={label} stride={stride_blocks} "
            f"loss_abs={loss_abs:.6g} L_diff_abs={diff_abs:.6g} "
            f"grad_allclose={grad_cmp['allclose']} "
            f"grad_max_abs={grad_cmp['max_abs']:.6g} "
            f"grad_max_rel={grad_cmp['max_rel']:.6g} "
            f"grad_max_margin={grad_cmp['max_margin']:.6g} "
            f"grad_mismatches={len(grad_cmp['mismatches'])} "
            f"grad_not_allclose={len(grad_cmp['not_allclose'])}"
        )
        results.append(TestResult("Route-II sensitivity control", passed, detail))
    return results


def main() -> int:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    torch.manual_seed(args.seed)
    model_dir = Path(args.model_dir).resolve()
    config_module, modeling_module = load_local_bridge(model_dir)

    block_size = 2
    results = [
        test_mask_rules(block_size),
        test_clean_logits_match_ar(
            config_module,
            modeling_module,
            seed=args.seed,
            atol=args.atol,
            block_size=block_size,
        ),
        test_gdn_schedule_doc_reset(
            modeling_module,
            config_module,
            seed=args.seed,
            atol=args.atol,
            block_size=block_size,
        ),
        test_loss_logit_shift_indexing(),
        test_noisy_loss_finite(
            config_module,
            modeling_module,
            seed=args.seed,
            block_size=block_size,
        ),
        test_flare_mask_rate_schedule(
            config_module,
            modeling_module,
            seed=args.seed,
            block_size=block_size,
        ),
    ]
    results.extend(
        test_route_ii_equivalence_matrix(
            config_module,
            modeling_module,
            seed=args.seed,
            block_size=block_size,
        )
    )
    results.append(
        test_route_ii_shortconv_tail_control(
            modeling_module,
            config_module,
            seed=args.seed,
            block_size=block_size,
        )
    )
    results.append(
        test_nested_view_gdn_state_discipline(
            modeling_module,
            config_module,
            seed=args.seed,
            block_size=block_size,
            atol=args.atol,
        )
    )
    results.extend(
        test_route_ii_sensitivity_controls(
            config_module,
            modeling_module,
            seed=args.seed,
            block_size=block_size,
        )
    )

    print("FLARE Stage-1 two-stream forward validation")
    print(f"model_dir={model_dir}")
    print(f"device=cpu dtype=float32 threads={torch.get_num_threads()} seed={args.seed}")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status}\t{result.name}\t{result.detail}", flush=True)

    if all(result.passed for result in results):
        print("FINAL: PASS")
        return 0
    print("FINAL: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
