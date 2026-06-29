#!/usr/bin/env python3
"""Validate local Qwen3.5 GDN state snapshot equivalence.

This is an inference-only, random-weight harness for the local Fast-dLLM
Qwen3.5 Gated DeltaNet bridge. It does not load checkpoint weights.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass
class LayerResult:
    layer_idx: int
    output_max_abs_diff: float
    state_max_abs_diff: float
    native_manual_full_max_abs_diff: float
    shortconv_tail_match_max_abs_diff: float
    shortconv_zero_first_lags_max_abs_diff: float
    shortconv_zero_after_lags_max_abs_diff: float
    match: bool
    diagnosis: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default="models/qwen3.5-9b-fastdllm-init",
        help="Local Fast-dLLM Qwen3.5 bridge directory.",
    )
    parser.add_argument("--prompt-len", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument(
        "--layers",
        default="all",
        help="Comma-separated layer indices, or 'all' for every linear_attention layer.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--conv-atol", type=float, default=1e-5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=("float32", "bfloat16", "float16"),
        help="Computation dtype. CPU default should remain float32 for stable checks.",
    )
    return parser.parse_args()


def load_local_bridge(model_dir: Path):
    package_name = "_fastdllm_qwen35_local"
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


def load_config(config_module, model_dir: Path):
    with (model_dir / "config.json").open("r", encoding="utf-8") as handle:
        raw_config = json.load(handle)
    raw_config.pop("architectures", None)
    raw_config.pop("auto_map", None)
    raw_config.pop("model_type", None)
    raw_config.pop("transformers_version", None)
    return config_module.Fast_dLLM_Qwen3_5Config(**raw_config)


def resolve_layers(config, raw_layers: str) -> list[int]:
    gdn_layers = [
        idx for idx, layer_type in enumerate(config.layer_types) if layer_type == "linear_attention"
    ]
    if raw_layers == "all":
        return gdn_layers
    requested = [int(item.strip()) for item in raw_layers.split(",") if item.strip()]
    bad = [idx for idx in requested if idx not in gdn_layers]
    if bad:
        raise ValueError(f"Requested non-GDN layer(s): {bad}; GDN layers are {gdn_layers}")
    return requested


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def project_and_conv(layer, hidden_states: torch.Tensor, conv_tail: torch.Tensor | None = None):
    """Run the local layer's qkv projection and causal ShortConv.

    conv_tail is the raw in_proj_qkv tail from previous tokens, shape
    [batch, W-1, conv_dim]. It is prepended before the same Conv1d module and
    then sliced away, preserving causal boundary lags.
    """
    seq_len = hidden_states.shape[1]
    raw_qkv = layer.in_proj_qkv(hidden_states)
    if conv_tail is None:
        conv_input = raw_qkv
        slice_start = 0
    else:
        conv_input = torch.cat([conv_tail, raw_qkv], dim=1)
        slice_start = conv_tail.shape[1]

    conv_all = layer.conv1d(conv_input.transpose(1, 2))
    conv_all = F.silu(conv_all[:, :, : conv_input.shape[1]]).transpose(1, 2)
    mixed_qkv = conv_all[:, slice_start : slice_start + seq_len]
    return raw_qkv, mixed_qkv


def run_gdn_manual(
    layer,
    modeling_module,
    hidden_states: torch.Tensor,
    *,
    chunk_size: int,
    initial_state: torch.Tensor | None = None,
    conv_tail: torch.Tensor | None = None,
):
    batch_size, seq_len, _ = hidden_states.shape
    raw_qkv, mixed_qkv = project_and_conv(layer, hidden_states, conv_tail=conv_tail)
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
    core_attn_out, final_state = modeling_module.torch_chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        chunk_size=chunk_size,
        initial_state=initial_state,
        output_final_state=True,
    )
    core_attn_out = layer.norm(
        core_attn_out.reshape(-1, layer.head_v_dim),
        z.reshape(-1, layer.head_v_dim),
    )
    output = layer.out_proj(core_attn_out.reshape(batch_size, seq_len, -1))
    return output, final_state, raw_qkv, mixed_qkv


def max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().item())


def validate_layer(
    config,
    modeling_module,
    layer_idx: int,
    *,
    prompt_len: int,
    block_size: int,
    seed: int,
    atol: float,
    conv_atol: float,
    device: torch.device,
    dtype: torch.dtype,
) -> LayerResult:
    torch.manual_seed(seed + layer_idx)
    layer = modeling_module.Fast_dLLM_Qwen3_5GatedDeltaNet(config, layer_idx)
    layer.eval()
    layer.to(device=device, dtype=dtype)

    conv_lag = int(layer.conv_kernel_size) - 1
    prefix_len = prompt_len + block_size
    total_len = prefix_len + block_size
    if prompt_len < 1:
        raise ValueError("prompt_len must be positive")
    if block_size < conv_lag + 1:
        raise ValueError(f"block_size={block_size} must exceed ShortConv lag={conv_lag}")

    hidden_states = torch.randn(
        1,
        total_len,
        config.hidden_size,
        device=device,
        dtype=dtype,
    )

    with torch.inference_mode():
        native_full_out, native_full_state = layer(
            hidden_states,
            chunk_size=block_size,
            output_final_state=True,
        )
        manual_full_out, manual_full_state, raw_qkv_full, conv_qkv_full = run_gdn_manual(
            layer,
            modeling_module,
            hidden_states,
            chunk_size=block_size,
        )
        _, prefix_state, _, _ = run_gdn_manual(
            layer,
            modeling_module,
            hidden_states[:, :prefix_len],
            chunk_size=block_size,
        )
        conv_tail = raw_qkv_full[:, prefix_len - conv_lag : prefix_len].contiguous()
        block_out, block_state, _, conv_qkv_block_tail = run_gdn_manual(
            layer,
            modeling_module,
            hidden_states[:, prefix_len:],
            chunk_size=block_size,
            initial_state=prefix_state,
            conv_tail=conv_tail,
        )
        _, _, _, conv_qkv_block_zero = run_gdn_manual(
            layer,
            modeling_module,
            hidden_states[:, prefix_len:],
            chunk_size=block_size,
            initial_state=prefix_state,
            conv_tail=None,
        )

    native_manual_full_diff = max_abs_diff(native_full_out, manual_full_out)
    output_diff = max_abs_diff(native_full_out[:, prefix_len:], block_out)
    state_diff = max_abs_diff(native_full_state, block_state)
    manual_state_diff = max_abs_diff(native_full_state, manual_full_state)
    state_diff = max(state_diff, manual_state_diff)

    conv_full_block2 = conv_qkv_full[:, prefix_len:]
    shortconv_tail_diff = max_abs_diff(conv_full_block2, conv_qkv_block_tail)
    shortconv_zero_first_diff = max_abs_diff(
        conv_qkv_block_tail[:, :conv_lag],
        conv_qkv_block_zero[:, :conv_lag],
    )
    shortconv_zero_after_diff = max_abs_diff(
        conv_qkv_block_tail[:, conv_lag:],
        conv_qkv_block_zero[:, conv_lag:],
    )

    output_ok = output_diff <= atol
    state_ok = state_diff <= atol
    native_ok = native_manual_full_diff <= atol
    conv_tail_ok = shortconv_tail_diff <= conv_atol
    conv_reads_prefix = shortconv_zero_first_diff > conv_atol
    conv_after_ok = shortconv_zero_after_diff <= conv_atol
    match = output_ok and state_ok and native_ok and conv_tail_ok and conv_reads_prefix and conv_after_ok

    if match:
        diagnosis = "MATCH"
    elif not native_ok:
        diagnosis = "MISMATCH: manual wrapper diverges from local GDN forward"
    elif not conv_tail_ok or not conv_reads_prefix or not conv_after_ok:
        diagnosis = "MISMATCH: ShortConv boundary bug"
    else:
        diagnosis = "MISMATCH: GDN recurrence state snapshot mismatch"

    return LayerResult(
        layer_idx=layer_idx,
        output_max_abs_diff=output_diff,
        state_max_abs_diff=state_diff,
        native_manual_full_max_abs_diff=native_manual_full_diff,
        shortconv_tail_match_max_abs_diff=shortconv_tail_diff,
        shortconv_zero_first_lags_max_abs_diff=shortconv_zero_first_diff,
        shortconv_zero_after_lags_max_abs_diff=shortconv_zero_after_diff,
        match=match,
        diagnosis=diagnosis,
    )


def main() -> int:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))

    model_dir = Path(args.model_dir).resolve()
    config_module, modeling_module = load_local_bridge(model_dir)
    config = load_config(config_module, model_dir)
    block_size = int(args.block_size or config.bd_size)
    layer_indices = resolve_layers(config, args.layers)
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)

    required = modeling_module.torch_chunk_gated_delta_rule
    missing = [
        name
        for name in ("initial_state", "output_final_state")
        if name not in required.__code__.co_varnames
    ]
    if missing:
        raise RuntimeError(
            "Local torch_chunk_gated_delta_rule is missing required state API: "
            + ", ".join(missing)
        )

    print("GDN state-snapshot validation harness")
    print(f"model_dir={model_dir}")
    print(f"device={device} dtype={dtype} threads={torch.get_num_threads()}")
    print(f"prompt_len={args.prompt_len} block_size={block_size} atol={args.atol}")
    print(f"layers={layer_indices}")
    print(
        "layer\tout_max_abs\tstate_max_abs\tnative_manual_full_max_abs\t"
        "shortconv_tail_max_abs\tshortconv_zero_first_lags_max_abs\t"
        "shortconv_zero_after_lags_max_abs\tstatus"
    )

    results = []
    for layer_idx in layer_indices:
        result = validate_layer(
            config,
            modeling_module,
            layer_idx,
            prompt_len=args.prompt_len,
            block_size=block_size,
            seed=args.seed,
            atol=args.atol,
            conv_atol=args.conv_atol,
            device=device,
            dtype=dtype,
        )
        results.append(result)
        print(
            f"{result.layer_idx}\t"
            f"{result.output_max_abs_diff:.6g}\t"
            f"{result.state_max_abs_diff:.6g}\t"
            f"{result.native_manual_full_max_abs_diff:.6g}\t"
            f"{result.shortconv_tail_match_max_abs_diff:.6g}\t"
            f"{result.shortconv_zero_first_lags_max_abs_diff:.6g}\t"
            f"{result.shortconv_zero_after_lags_max_abs_diff:.6g}\t"
            f"{result.diagnosis}",
            flush=True,
        )

    all_match = all(result.match for result in results)
    if all_match:
        print("FINAL: MATCH (GDN causal-within-block, gate cleared)")
        return 0

    print("FINAL: MISMATCH")
    for result in results:
        if not result.match:
            print(f"layer={result.layer_idx} diagnosis={result.diagnosis}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
