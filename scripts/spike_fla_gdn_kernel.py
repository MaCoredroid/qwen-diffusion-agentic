#!/usr/bin/env python3
"""Spike-test FLA's fused GDN kernel against the local torch reference.

This is a standalone Step-0 gate. It does not switch the model code path.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import importlib.util
import json
import math
import os
import sys
import time
import traceback
import types
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models/qwen3.5-9b-fastdllm-init"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1024, help="One training GDN block by default.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def load_local_bridge(model_dir: Path):
    package_name = "_fastdllm_qwen35_fla_spike"
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
    return loaded["modeling"]


def version_or_missing(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "MISSING"


def make_leaf(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone().requires_grad_(True)


def make_inputs(args: argparse.Namespace, modeling_module):
    config_path = args.model_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)

    batch = args.batch_size
    seq_len = args.seq_len
    heads = int(cfg["linear_num_value_heads"])
    key_dim = int(cfg["linear_key_head_dim"])
    value_dim = int(cfg["linear_value_head_dim"])
    device = torch.device(args.device)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    q = torch.randn(batch, seq_len, heads, key_dim, device=device, dtype=torch.float32, generator=generator)
    k = torch.randn(batch, seq_len, heads, key_dim, device=device, dtype=torch.float32, generator=generator)
    q = modeling_module.l2norm(q, dim=-1).to(torch.bfloat16)
    k = modeling_module.l2norm(k, dim=-1).to(torch.bfloat16)
    v = (0.2 * torch.randn(batch, seq_len, heads, value_dim, device=device, dtype=torch.float32, generator=generator)).to(
        torch.bfloat16
    )
    beta = torch.rand(batch, seq_len, heads, device=device, dtype=torch.float32, generator=generator)
    beta = beta.mul(0.90).add(0.05).to(torch.bfloat16)

    # Raw per-token log-space decay. Do not cumsum here; FLA cumsums inside each chunk.
    g = -(torch.rand(batch, seq_len, heads, device=device, dtype=torch.float32, generator=generator).mul(0.050).add(0.001))

    h0 = (0.05 * torch.randn(batch, heads, key_dim, value_dim, device=device, dtype=torch.float32, generator=generator)).to(
        torch.bfloat16
    )
    if float(h0.float().abs().max().item()) == 0.0:
        raise RuntimeError("initial_state unexpectedly all zero")

    o_grad = 0.01 * torch.randn(batch, seq_len, heads, value_dim, device=device, dtype=torch.float32, generator=generator)
    ht_grad = 0.01 * torch.randn(batch, heads, key_dim, value_dim, device=device, dtype=torch.float32, generator=generator)

    base_inputs = {
        "query": q,
        "key": k,
        "value": v,
        "g": g,
        "beta": beta,
        "initial_state": h0,
    }
    grad_outputs = {"output": o_grad, "final_state": ht_grad}
    shape_meta = {
        "batch_size": batch,
        "seq_len": seq_len,
        "heads": heads,
        "key_dim": key_dim,
        "value_dim": value_dim,
    }
    return base_inputs, grad_outputs, shape_meta


def clone_inputs(base_inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: make_leaf(tensor) for name, tensor in base_inputs.items()}


def run_fla(inputs: dict[str, torch.Tensor], grad_outputs: dict[str, torch.Tensor], scale: float):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    start = time.perf_counter()
    output, final_state = chunk_gated_delta_rule(
        inputs["query"],
        inputs["key"],
        inputs["value"],
        g=inputs["g"],
        beta=inputs["beta"],
        scale=scale,
        initial_state=inputs["initial_state"],
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
        allow_neg_eigval=False,
    )
    torch.cuda.synchronize()
    fwd_seconds = time.perf_counter() - start

    loss = (output.float() * grad_outputs["output"]).sum() + (final_state.float() * grad_outputs["final_state"]).sum()
    start = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    bwd_seconds = time.perf_counter() - start
    return output, final_state, loss.detach(), {"fwd_seconds": fwd_seconds, "bwd_seconds": bwd_seconds}


def run_torch_reference(
    modeling_module,
    inputs: dict[str, torch.Tensor],
    grad_outputs: dict[str, torch.Tensor],
):
    start = time.perf_counter()
    output, final_state = modeling_module._torch_chunk_gated_delta_rule_impl(
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
    torch.cuda.synchronize()
    fwd_seconds = time.perf_counter() - start

    loss = (output.float() * grad_outputs["output"]).sum() + (final_state.float() * grad_outputs["final_state"]).sum()
    start = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    bwd_seconds = time.perf_counter() - start
    return output, final_state, loss.detach(), {"fwd_seconds": fwd_seconds, "bwd_seconds": bwd_seconds}


def finite_tensor(tensor: torch.Tensor | None) -> bool:
    return tensor is not None and bool(torch.isfinite(tensor).all().item())


def diff_stats(actual: torch.Tensor, expected: torch.Tensor, *, rtol: float, atol: float) -> dict[str, float | bool]:
    actual_f = actual.float()
    expected_f = expected.float()
    diff = (actual_f - expected_f).abs()
    denom = expected_f.abs().clamp_min(1e-3)
    rel = diff / denom
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "max_rel": float(rel.max().item()),
        "mean_rel": float(rel.mean().item()),
        "allclose": bool(torch.allclose(actual_f, expected_f, rtol=rtol, atol=atol)),
    }


def classify_exception(exc_text: str) -> str:
    lowered = exc_text.lower()
    if "tmem_store" in lowered or "no kernel image" in lowered:
        return "#607-no-fix"
    if "cumsum" in lowered or "chunk_local_cumsum" in lowered:
        return "#734-only"
    return "crash"


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the FLA spike")

    modeling_module = load_local_bridge(args.model_dir)
    base_inputs, grad_outputs, shape_meta = make_inputs(args, modeling_module)
    scale = 1.0 / math.sqrt(shape_meta["key_dim"])

    metadata_payload = {
        "flash_linear_attention": version_or_missing("flash-linear-attention"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": version_or_missing("triton"),
        "device": torch.cuda.get_device_name(0) if args.device == "cuda" else args.device,
        "capability": torch.cuda.get_device_capability(0) if args.device == "cuda" else None,
        "shape": shape_meta,
        "dtype": "bfloat16",
        "g_pre_cumsum": False,
        "scale": scale,
        "flags": {
            "use_qk_l2norm_in_kernel": False,
            "use_beta_sigmoid_in_kernel": False,
            "allow_neg_eigval": False,
        },
    }

    try:
        fla_inputs = clone_inputs(base_inputs)
        ref_inputs = clone_inputs(base_inputs)
        fla_output, fla_final, fla_loss, fla_timing = run_fla(fla_inputs, grad_outputs, scale)
        ref_output, ref_final, ref_loss, ref_timing = run_torch_reference(modeling_module, ref_inputs, grad_outputs)
    except Exception:
        exc_text = traceback.format_exc()
        payload = {
            "status": "CRASH",
            "verdict": classify_exception(exc_text),
            "metadata": metadata_payload,
            "exception": exc_text,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2

    parity = {
        "output": diff_stats(fla_output, ref_output, rtol=args.rtol, atol=args.atol),
        "final_state": diff_stats(fla_final, ref_final, rtol=args.rtol, atol=args.atol),
    }
    grad_names = ("query", "key", "value", "beta", "g", "initial_state")
    grad_finite = {}
    for name in grad_names:
        grad_finite[name] = {
            "fla": finite_tensor(fla_inputs[name].grad),
            "torch_reference": finite_tensor(ref_inputs[name].grad),
        }
        parity[f"d{name if name != 'initial_state' else 'h0'}"] = diff_stats(
            fla_inputs[name].grad,
            ref_inputs[name].grad,
            rtol=args.rtol,
            atol=args.atol,
        )

    all_finite = all(item["fla"] and item["torch_reference"] for item in grad_finite.values())
    all_parity = all(bool(stats["allclose"]) for stats in parity.values())
    no_crash = True
    verdict = "GREEN" if no_crash and all_finite and all_parity else "PARITY-FAIL"

    payload = {
        "status": "PASS" if verdict == "GREEN" else "FAIL",
        "verdict": verdict,
        "metadata": metadata_payload,
        "checks": {
            "no_607_tmem_store_or_no_kernel_image_crash": no_crash,
            "gradients_all_finite": all_finite,
            "gradients_finite_by_tensor": grad_finite,
            "parity_allclose": all_parity,
            "rtol": args.rtol,
            "atol": args.atol,
        },
        "losses": {
            "fla": float(fla_loss.float().item()),
            "torch_reference": float(ref_loss.float().item()),
        },
        "timing": {
            "fla": fla_timing,
            "torch_reference": ref_timing,
        },
        "parity": parity,
    }

    if args.print_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"FLA fused GDN spike verdict: {payload['verdict']}")
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if verdict == "GREEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
