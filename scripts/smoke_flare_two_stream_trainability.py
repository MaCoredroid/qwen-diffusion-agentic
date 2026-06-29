#!/usr/bin/env python3
"""Tiny trainability smoke for the FLARE Stage-1 two-stream objective.

This is a standalone validation rung: CPU/fp32, fixed synthetic packed docs,
fixed complementary masks, tiny random-init model, and a short optimizer loop.
It does not touch the production QLoRA path, model training forward, or data
pipeline.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class LossSnapshot:
    step: int
    total: float
    ar: float
    diff: float


@dataclass
class GradStats:
    total_norm: float
    max_abs: float
    tensors_with_grad: int
    finite: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default="models/qwen3.5-9b-fastdllm-init",
        help="Local Fast-dLLM Qwen3.5 bridge directory.",
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--final-threshold", type=float, default=0.15)
    parser.add_argument("--decrease-ratio", type=float, default=0.20)
    return parser.parse_args()


def load_stage1_module():
    script_path = Path(__file__).resolve().with_name("validate_flare_two_stream_forward.py")
    spec = importlib.util.spec_from_file_location("_flare_stage1_forward_validator", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import Stage-1 validator from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_smoke_config(config_module, *, block_size: int):
    return config_module.Fast_dLLM_Qwen3_5Config(
        vocab_size=29,
        hidden_size=48,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=64,
        initializer_range=0.04,
        rms_norm_eps=1e-6,
        use_cache=False,
        tie_word_embeddings=False,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=12,
        linear_conv_kernel_dim=3,
        linear_key_head_dim=12,
        linear_value_head_dim=12,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=["linear_attention", "full_attention"],
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


def make_smoke_model(config_module, modeling_module, *, seed: int, block_size: int):
    torch.manual_seed(seed)
    config = make_smoke_config(config_module, block_size=block_size)
    model = modeling_module.Fast_dLLM_Qwen3_5ForCausalLM(config)
    for module in model.modules():
        if module.__class__.__name__ in {
            "Fast_dLLM_Qwen3_5RMSNorm",
            "Fast_dLLM_Qwen3_5RMSNormGated",
        }:
            module.weight.data.fill_(1.0)
    model.train()
    return model


def fixed_batch():
    input_ids = torch.tensor([[5, 7, 9, 11, 13, 15, 17, 19]], dtype=torch.long)
    doc_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.long)
    mask_indices = torch.tensor(
        [[False, True, False, True, False, False, True, False]],
        dtype=torch.bool,
    )
    return input_ids, doc_ids, mask_indices


def snapshot_losses(stage1, model, modeling_module, input_ids, doc_ids, mask_indices, block_size: int, step: int):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        output = stage1.flare_two_stream_forward(
            model,
            modeling_module,
            input_ids,
            doc_ids,
            mask_indices,
            block_size=block_size,
        )
    if was_training:
        model.train()
    losses = output.losses
    return LossSnapshot(
        step=step,
        total=float(losses.total_loss.item()),
        ar=float(losses.ar_loss.item()),
        diff=float(losses.diff_loss.item()),
    )


def grad_stats(model) -> GradStats:
    total_sq = 0.0
    max_abs = 0.0
    tensors_with_grad = 0
    finite = True
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        tensors_with_grad += 1
        finite = finite and bool(torch.isfinite(grad).all().item())
        total_sq += float(grad.float().pow(2).sum().item())
        if grad.numel():
            max_abs = max(max_abs, float(grad.float().abs().max().item()))
    return GradStats(
        total_norm=total_sq**0.5,
        max_abs=max_abs,
        tensors_with_grad=tensors_with_grad,
        finite=finite,
    )


def backward_component(stage1, model, modeling_module, input_ids, doc_ids, mask_indices, block_size: int, component: str):
    model.zero_grad(set_to_none=True)
    output = stage1.flare_two_stream_forward(
        model,
        modeling_module,
        input_ids,
        doc_ids,
        mask_indices,
        block_size=block_size,
    )
    loss = getattr(output.losses, component)
    loss.backward()
    return float(loss.item()), grad_stats(model)


def format_snapshot(snapshot: LossSnapshot) -> str:
    return (
        f"step={snapshot.step:02d} "
        f"L={snapshot.total:.6f} "
        f"L_AR={snapshot.ar:.6f} "
        f"L_diff={snapshot.diff:.6f}"
    )


def main() -> int:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    torch.set_num_threads(max(1, args.threads))
    torch.manual_seed(args.seed)

    stage1 = load_stage1_module()
    config_module, modeling_module = stage1.load_local_bridge(Path(args.model_dir).resolve())
    block_size = 2
    model = make_smoke_model(config_module, modeling_module, seed=args.seed, block_size=block_size)
    input_ids, doc_ids, mask_indices = fixed_batch()

    ar_probe_loss, ar_probe_grad = backward_component(
        stage1,
        model,
        modeling_module,
        input_ids,
        doc_ids,
        mask_indices,
        block_size,
        "ar_loss",
    )
    diff_probe_loss, diff_probe_grad = backward_component(
        stage1,
        model,
        modeling_module,
        input_ids,
        doc_ids,
        mask_indices,
        block_size,
        "diff_loss",
    )
    model.zero_grad(set_to_none=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    curve: list[LossSnapshot] = [
        snapshot_losses(stage1, model, modeling_module, input_ids, doc_ids, mask_indices, block_size, 0)
    ]
    first_total_grad: GradStats | None = None
    last_total_grad: GradStats | None = None
    milestones = {1, 5, 10, 20, 30, 40, args.steps}
    milestones = {step for step in milestones if step <= args.steps}

    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = stage1.flare_two_stream_forward(
            model,
            modeling_module,
            input_ids,
            doc_ids,
            mask_indices,
            block_size=block_size,
        )
        output.losses.total_loss.backward()
        current_grad = grad_stats(model)
        if step == 1:
            first_total_grad = current_grad
        last_total_grad = current_grad
        optimizer.step()

        if step in milestones:
            curve.append(
                snapshot_losses(
                    stage1,
                    model,
                    modeling_module,
                    input_ids,
                    doc_ids,
                    mask_indices,
                    block_size,
                    step,
                )
            )

    initial = curve[0]
    final = curve[-1]
    assert first_total_grad is not None and last_total_grad is not None

    total_decreased = final.total < initial.total * args.decrease_ratio and final.total < args.final_threshold
    ar_decreased = final.ar < initial.ar * args.decrease_ratio and final.ar < args.final_threshold
    diff_decreased = final.diff < initial.diff * args.decrease_ratio and final.diff < args.final_threshold
    grad_ok = (
        ar_probe_grad.finite
        and diff_probe_grad.finite
        and first_total_grad.finite
        and last_total_grad.finite
        and ar_probe_grad.total_norm > 0.0
        and diff_probe_grad.total_norm > 0.0
        and first_total_grad.total_norm > 0.0
        and last_total_grad.tensors_with_grad > 0
    )
    passed = total_decreased and ar_decreased and diff_decreased and grad_ok

    print("FLARE Stage-1 two-stream trainability smoke")
    print(
        f"device=cpu dtype=float32 threads={torch.get_num_threads()} "
        f"seed={args.seed} steps={args.steps} lr={args.lr:g}"
    )
    print("fixed_batch input_ids=", input_ids.tolist(), "doc_ids=", doc_ids.tolist())
    print("fixed_mask view0=", mask_indices.tolist(), "view1=", (~mask_indices).tolist())
    print(
        "grad_probe "
        f"L_AR={ar_probe_loss:.6f} grad_norm={ar_probe_grad.total_norm:.6g} "
        f"max_abs={ar_probe_grad.max_abs:.6g} finite={ar_probe_grad.finite} "
        f"tensors={ar_probe_grad.tensors_with_grad}"
    )
    print(
        "grad_probe "
        f"L_diff={diff_probe_loss:.6f} grad_norm={diff_probe_grad.total_norm:.6g} "
        f"max_abs={diff_probe_grad.max_abs:.6g} finite={diff_probe_grad.finite} "
        f"tensors={diff_probe_grad.tensors_with_grad}"
    )
    print(
        "total_grad "
        f"step1_norm={first_total_grad.total_norm:.6g} "
        f"step1_finite={first_total_grad.finite} "
        f"last_norm={last_total_grad.total_norm:.6g} "
        f"last_finite={last_total_grad.finite}"
    )
    print("loss_curve")
    for snapshot in curve:
        print(format_snapshot(snapshot))
    print(
        "criteria "
        f"total_decreased={total_decreased} "
        f"ar_decreased={ar_decreased} "
        f"diff_decreased={diff_decreased} "
        f"grad_ok={grad_ok}"
    )
    print("FINAL:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
