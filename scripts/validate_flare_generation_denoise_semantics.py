#!/usr/bin/env python3
"""Validate generation denoise logits against FLARE two-stream training semantics."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0 and right.numel() == 0:
        return 0.0
    return float((left.float() - right.float()).abs().max().item())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/qwen3.5-9b-fastdllm-init")
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.threads))
    stage1 = load_module(ROOT / "scripts/validate_flare_two_stream_forward.py", "_stage1_forward_validator_for_gen")
    gen_diag = load_module(ROOT / "scripts/diagnose_flare_generation_speed.py", "_flare_gen_diag_for_validation")
    config_module, modeling_module = stage1.load_local_bridge((ROOT / args.model_dir).resolve())

    block_size = 4
    model = stage1.make_tiny_model(config_module, modeling_module, seed=args.seed, block_size=block_size)
    model.eval()
    input_ids = torch.tensor([[5, 8, 13, 21, 7, 11, 17, 19, 23, 29, 31, 37]], dtype=torch.long)
    doc_ids = torch.zeros_like(input_ids)
    mask_indices = torch.zeros_like(input_ids, dtype=torch.bool)
    mask_indices[:, 8:12] = torch.tensor([[True, False, True, True]])
    noisy_ids = torch.where(mask_indices, torch.full_like(input_ids, model.config.mask_token_id), input_ids)

    with torch.no_grad():
        helper = stage1.flare_two_stream_forward(
            model,
            modeling_module,
            input_ids,
            doc_ids,
            mask_indices,
            block_size=block_size,
        )
        sampler_noisy = gen_diag.flare_two_stream_noisy_logits(
            model,
            input_ids,
            noisy_ids,
            block_size=block_size,
            mask_id=int(model.config.mask_token_id),
        )

    helper_view0 = helper.noisy_logits[: input_ids.shape[0]]
    full_diff = max_abs_diff(sampler_noisy[: input_ids.shape[0]], helper_view0)
    shifted_sampler = torch.cat([sampler_noisy[:1, :1, :], sampler_noisy[:1, :-1, :]], dim=1)
    shifted_helper = torch.cat([helper_view0[:, :1, :], helper_view0[:, :-1, :]], dim=1)
    active_slice = slice(8, 12)
    active_diff = max_abs_diff(shifted_sampler[:, active_slice], shifted_helper[:, active_slice])
    token_exact = torch.equal(sampler_noisy[: input_ids.shape[0]], helper_view0)
    passed = full_diff <= args.atol and active_diff <= args.atol

    print("FLARE generation denoise semantics validation")
    print(f"device=cpu dtype=float32 threads={torch.get_num_threads()} seed={args.seed}")
    print(f"full_noisy_logits_max_abs_diff={full_diff:.6g}")
    print(f"active_shifted_logits_max_abs_diff={active_diff:.6g}")
    print(f"torch_equal_full_noisy_logits={token_exact}")
    print("FINAL:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
