#!/usr/bin/env python3
"""Strengthened FLA GDN integration checks on CUDA.

This covers the gaps the single-block bf16 spike did not:
1. fp32 tight-tolerance kernel parity at realistic GDN activation magnitudes;
2. detached initial_state parity (requires_grad=False seed);
3. production FLARE two-stream multi-block/multi-doc schedule parity;
4. legacy detached clean-state injection path parity and gradient isolation.

It intentionally uses tiny random-weight models. Real checkpoint logits/NLL
parity is handled by validate_fla_real_weight_parity.py.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.util
import json
import math
import os
import sys
import types
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models/qwen3.5-9b-fastdllm-init"


@contextlib.contextmanager
def patched_env(**updates):
    old = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_local_bridge(model_dir: Path):
    package_name = "_fastdllm_qwen35_fla_integration"
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


def load_stage1_module():
    script_path = ROOT / "scripts" / "validate_flare_two_stream_forward.py"
    spec = importlib.util.spec_from_file_location("_flare_stage1_for_fla_gate", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def diff_stats(left: torch.Tensor, right: torch.Tensor, *, rtol: float, atol: float):
    left_f = left.detach().float()
    right_f = right.detach().float()
    diff = (left_f - right_f).abs()
    denom = right_f.abs().clamp_min(1e-6)
    rel = diff / denom
    return {
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "max_rel": float(rel.max().item()) if rel.numel() else 0.0,
        "mean_rel": float(rel.mean().item()) if rel.numel() else 0.0,
        "allclose": bool(torch.allclose(left_f, right_f, rtol=rtol, atol=atol)),
    }


def grad_or_zero(tensor: torch.Tensor):
    return torch.zeros_like(tensor) if tensor.grad is None else tensor.grad.detach()


def clone_leaf(tensor: torch.Tensor, *, requires_grad: bool = True):
    return tensor.detach().clone().requires_grad_(requires_grad)


def make_kernel_inputs(modeling_module, *, seed: int, device: torch.device, dtype: torch.dtype):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    batch, seq_len, heads, key_dim, value_dim = 2, 128, 4, 32, 32
    query = torch.randn(batch, seq_len, heads, key_dim, device=device, dtype=torch.float32, generator=generator)
    key = torch.randn(batch, seq_len, heads, key_dim, device=device, dtype=torch.float32, generator=generator)
    query = modeling_module.l2norm(query, dim=-1).to(dtype)
    key = modeling_module.l2norm(key, dim=-1).to(dtype)
    value = (0.35 * torch.randn(batch, seq_len, heads, value_dim, device=device, dtype=torch.float32, generator=generator)).to(dtype)
    beta = torch.rand(batch, seq_len, heads, device=device, dtype=torch.float32, generator=generator)
    beta = beta.mul(0.90).add(0.05).to(dtype)
    g = -(torch.rand(batch, seq_len, heads, device=device, dtype=torch.float32, generator=generator).mul(0.075).add(0.002))
    h0 = (0.05 * torch.randn(batch, heads, key_dim, value_dim, device=device, dtype=torch.float32, generator=generator)).to(dtype)
    output_grad = 0.02 * torch.randn(batch, seq_len, heads, value_dim, device=device, dtype=torch.float32, generator=generator)
    state_grad = 0.02 * torch.randn(batch, heads, key_dim, value_dim, device=device, dtype=torch.float32, generator=generator)
    return {
        "query": query,
        "key": key,
        "value": value,
        "g": g,
        "beta": beta,
        "initial_state": h0,
        "output_grad": output_grad,
        "state_grad": state_grad,
    }


def run_kernel(modeling_module, base_inputs: dict[str, torch.Tensor], *, backend: str):
    inputs = {
        "query": clone_leaf(base_inputs["query"]),
        "key": clone_leaf(base_inputs["key"]),
        "value": clone_leaf(base_inputs["value"]),
        "g": clone_leaf(base_inputs["g"]),
        "beta": clone_leaf(base_inputs["beta"]),
        "initial_state": clone_leaf(base_inputs["initial_state"], requires_grad=False),
    }
    with patched_env(FASTDLLM_GDN_KERNEL=backend, FASTDLLM_COMPILE_GDN_SCAN="0"):
        output, final_state = modeling_module.torch_chunk_gated_delta_rule(
            inputs["query"],
            inputs["key"],
            inputs["value"],
            inputs["g"],
            inputs["beta"],
            chunk_size=64,
            initial_state=inputs["initial_state"],
            output_final_state=True,
            output_chunk_states=False,
        )
    loss = (output.float() * base_inputs["output_grad"]).sum() + (final_state.float() * base_inputs["state_grad"]).sum()
    loss.backward()
    torch.cuda.synchronize()
    return {
        "output": output.detach(),
        "final_state": final_state.detach(),
        "loss": loss.detach(),
        "grads": {
            name: grad_or_zero(tensor)
            for name, tensor in inputs.items()
            if name != "initial_state"
        },
        "initial_state_requires_grad": bool(inputs["initial_state"].requires_grad),
        "initial_state_grad_is_none": inputs["initial_state"].grad is None,
    }


def test_kernel_fp32_detached(modeling_module, args):
    base = make_kernel_inputs(modeling_module, seed=args.seed, device=torch.device("cuda"), dtype=torch.float32)
    torch_result = run_kernel(modeling_module, base, backend="torch")
    fla_result = run_kernel(modeling_module, base, backend="fla")
    checks = {
        "output": diff_stats(fla_result["output"], torch_result["output"], rtol=args.rtol_fp32, atol=args.atol_fp32),
        "final_state": diff_stats(
            fla_result["final_state"],
            torch_result["final_state"],
            rtol=args.rtol_fp32,
            atol=args.atol_fp32,
        ),
        "loss_abs": float((fla_result["loss"].float() - torch_result["loss"].float()).abs().item()),
        "detached_seed": {
            "fla_requires_grad": fla_result["initial_state_requires_grad"],
            "fla_grad_is_none": fla_result["initial_state_grad_is_none"],
            "torch_requires_grad": torch_result["initial_state_requires_grad"],
            "torch_grad_is_none": torch_result["initial_state_grad_is_none"],
        },
        "grads": {},
    }
    for name in ("query", "key", "value", "g", "beta"):
        checks["grads"][name] = diff_stats(
            fla_result["grads"][name],
            torch_result["grads"][name],
            rtol=args.rtol_fp32,
            atol=args.atol_fp32,
        )
    passed = (
        checks["output"]["allclose"]
        and checks["final_state"]["allclose"]
        and checks["loss_abs"] <= args.loss_atol_fp32
        and checks["detached_seed"]["fla_grad_is_none"]
        and all(item["allclose"] for item in checks["grads"].values())
    )
    return passed, checks


def loss_parts(model):
    return {
        key: (float(value.detach().float().cpu().item()) if torch.is_tensor(value) else value)
        for key, value in getattr(model, "_last_flare_loss_parts", {}).items()
        if key in {"total", "ar", "diff", "ar_count", "mask_view0", "mask_view1"}
    }


def run_two_stream(model, *, backend: str, input_ids, labels, attention_mask, doc_ids, mask_indices):
    with patched_env(
        FASTDLLM_GDN_KERNEL=backend,
        FASTDLLM_FLARE_TWO_STREAM="1",
        FLARE_TWO_STREAM=None,
        FASTDLLM_FLARE_GDN_ROUTE="route_i",
        FASTDLLM_BATCH_FLARE_NOISY_GDN="1",
        FASTDLLM_OPTIMIZE_FLARE_CLEAN_GDN="0",
        FASTDLLM_TRAIN_BD_SIZE="2",
        FASTDLLM_COMPILE_GDN_SCAN="0",
    ):
        model.train()
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                doc_ids=doc_ids,
                flare_mask_indices=mask_indices,
            )
        torch.cuda.synchronize()
        return {
            "loss": output.loss.detach(),
            "logits": output.logits.detach(),
            "parts": loss_parts(model),
        }


def test_two_stream_schedule(stage1, config_module, modeling_module, args):
    block_size = 2
    model = stage1.make_tiny_model(config_module, modeling_module, seed=args.seed + 1, block_size=block_size)
    model = model.to("cuda").float()
    input_ids = torch.tensor([[5, 8, 13, 21, 7, 11, 17, 19, 23, 29, 31, 37]], dtype=torch.long, device="cuda")
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)
    doc_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2]], dtype=torch.long, device="cuda")
    mask_indices = torch.tensor(
        [[False, True, False, True, True, False, True, False, False, True, True, False]],
        dtype=torch.bool,
        device="cuda",
    )
    torch_result = run_two_stream(
        model,
        backend="torch",
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        doc_ids=doc_ids,
        mask_indices=mask_indices,
    )
    fla_result = run_two_stream(
        model,
        backend="fla",
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        doc_ids=doc_ids,
        mask_indices=mask_indices,
    )
    logits = diff_stats(
        fla_result["logits"],
        torch_result["logits"],
        rtol=args.rtol_schedule_fp32,
        atol=args.atol_schedule_fp32,
    )
    loss_abs = float((fla_result["loss"].float() - torch_result["loss"].float()).abs().item())
    part_abs = {
        key: abs(float(fla_result["parts"][key]) - float(torch_result["parts"][key]))
        for key in ("total", "ar", "diff")
        if key in fla_result["parts"] and key in torch_result["parts"]
    }
    counts_match = {
        key: fla_result["parts"].get(key) == torch_result["parts"].get(key)
        for key in ("ar_count", "mask_view0", "mask_view1")
    }
    checks = {
        "logits": logits,
        "loss_abs": loss_abs,
        "part_abs": part_abs,
        "counts_match": counts_match,
        "torch_parts": torch_result["parts"],
        "fla_parts": fla_result["parts"],
        "doc_ids": doc_ids.detach().cpu().tolist()[0],
    }
    passed = logits["allclose"] and loss_abs <= args.loss_atol_schedule_fp32 and all(
        value <= args.loss_atol_schedule_fp32 for value in part_abs.values()
    ) and all(counts_match.values())
    return passed, checks


def run_detached_injection(layer, noisy_base, clean_base, *, backend: str):
    noisy = clone_leaf(noisy_base)
    clean = clone_leaf(clean_base)
    with patched_env(FASTDLLM_GDN_KERNEL=backend, FASTDLLM_COMPILE_GDN_SCAN="0"):
        output = layer._linear_attn_clean_state_injection(noisy, clean)
    seq_len = noisy.shape[1]
    noisy_output = output[:, :seq_len]
    loss = noisy_output.float().pow(2).mean()
    loss.backward()
    torch.cuda.synchronize()
    clean_grad = torch.zeros_like(clean) if clean.grad is None else clean.grad.detach()
    return {
        "noisy_output": noisy_output.detach(),
        "loss": loss.detach(),
        "noisy_grad": grad_or_zero(noisy),
        "clean_grad_max_abs": float(clean_grad.float().abs().max().item()),
    }


def test_detached_seed_module(stage1, config_module, modeling_module, args):
    block_size = 2
    config = stage1.make_tiny_config(config_module, block_size=block_size)
    layer = modeling_module.Fast_dLLM_Qwen3_5DecoderLayer(config, 0).to("cuda").float()
    layer.eval()
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed + 2)
    noisy = torch.randn(1, 8, config.hidden_size, device="cuda", dtype=torch.float32, generator=generator) * 0.25
    clean = torch.randn(1, 8, config.hidden_size, device="cuda", dtype=torch.float32, generator=generator) * 0.25
    torch_result = run_detached_injection(layer, noisy, clean, backend="torch")
    fla_result = run_detached_injection(layer, noisy, clean, backend="fla")
    checks = {
        "noisy_output": diff_stats(
            fla_result["noisy_output"],
            torch_result["noisy_output"],
            rtol=args.rtol_fp32,
            atol=args.atol_fp32,
        ),
        "noisy_grad": diff_stats(
            fla_result["noisy_grad"],
            torch_result["noisy_grad"],
            rtol=args.rtol_fp32,
            atol=args.atol_fp32,
        ),
        "loss_abs": float((fla_result["loss"].float() - torch_result["loss"].float()).abs().item()),
        "clean_grad_max_abs": {
            "torch": torch_result["clean_grad_max_abs"],
            "fla": fla_result["clean_grad_max_abs"],
        },
    }
    passed = (
        checks["noisy_output"]["allclose"]
        and checks["noisy_grad"]["allclose"]
        and checks["loss_abs"] <= args.loss_atol_fp32
        and checks["clean_grad_max_abs"]["torch"] == 0.0
        and checks["clean_grad_max_abs"]["fla"] == 0.0
    )
    return passed, checks


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--rtol-fp32", type=float, default=5e-4)
    parser.add_argument("--atol-fp32", type=float, default=5e-4)
    parser.add_argument("--loss-atol-fp32", type=float, default=5e-4)
    parser.add_argument("--rtol-schedule-fp32", type=float, default=6e-3)
    parser.add_argument("--atol-schedule-fp32", type=float, default=6e-3)
    parser.add_argument("--loss-atol-schedule-fp32", type=float, default=1e-3)
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FASTDLLM_GDN_KERNEL=fla")
    torch.cuda.set_device(0)
    config_module, modeling_module = load_local_bridge(args.model_dir.resolve())
    stage1 = load_stage1_module()

    tests = {}
    tests["fp32_kernel_detached_seed"] = dict(zip(("passed", "detail"), test_kernel_fp32_detached(modeling_module, args)))
    tests["two_stream_multi_doc_schedule"] = dict(
        zip(("passed", "detail"), test_two_stream_schedule(stage1, config_module, modeling_module, args))
    )
    tests["legacy_detached_seed_module"] = dict(
        zip(("passed", "detail"), test_detached_seed_module(stage1, config_module, modeling_module, args))
    )
    final = all(test["passed"] for test in tests.values())
    payload = {
        "status": "PASS" if final else "FAIL",
        "device": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "rtol_fp32": args.rtol_fp32,
        "atol_fp32": args.atol_fp32,
        "loss_atol_fp32": args.loss_atol_fp32,
        "rtol_schedule_fp32": args.rtol_schedule_fp32,
        "atol_schedule_fp32": args.atol_schedule_fp32,
        "loss_atol_schedule_fp32": args.loss_atol_schedule_fp32,
        "tests": tests,
    }
    if args.print_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("FLA GDN strengthened integration validation")
        print(f"device={payload['device']} capability={payload['capability']} seed={args.seed}")
        for name, test in tests.items():
            print(f"{'PASS' if test['passed'] else 'FAIL'}\t{name}\t{json.dumps(test['detail'], sort_keys=True)}")
        print("FINAL:", payload["status"])
    return 0 if final else 1


if __name__ == "__main__":
    raise SystemExit(main())
