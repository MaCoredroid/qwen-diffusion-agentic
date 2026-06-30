#!/usr/bin/env python3
"""End-to-end real-weight logits/NLL parity for FASTDLLM_GDN_KERNEL=fla."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_ADAPTER = ROOT / "runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000"


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


def configure_cuda_env():
    cuda_root = ROOT / ".venv-fastdllm" / "lib" / "python3.10" / "site-packages" / "nvidia" / "cu13"
    if cuda_root.exists():
        os.environ.setdefault("CUDA_HOME", str(cuda_root))
        os.environ["PATH"] = f"{cuda_root / 'bin'}:{os.environ.get('PATH', '')}"
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_root / 'lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    hasher = hashlib.sha256()
    for filename in ("modeling.py", "configuration.py"):
        path = DEFAULT_BASE / filename
        if path.exists():
            hasher.update(path.read_bytes())
    os.environ.setdefault("HF_MODULES_CACHE", str(ROOT / ".hf_modules_cache" / hasher.hexdigest()))


def diff_stats(left: torch.Tensor, right: torch.Tensor, *, rtol: float, atol: float):
    left_f = left.detach().float()
    right_f = right.detach().float()
    diff = (left_f - right_f).abs()
    denom = right_f.abs().clamp_min(1e-4)
    rel = diff / denom
    return {
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "max_rel": float(rel.max().item()) if rel.numel() else 0.0,
        "mean_rel": float(rel.mean().item()) if rel.numel() else 0.0,
        "allclose": bool(torch.allclose(left_f, right_f, rtol=rtol, atol=atol)),
    }


def load_model_and_tokenizer(args):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter = resolve_optional_path(args.adapter)
    tokenizer_path = str(resolve_optional_path(args.tokenizer_path) or adapter or args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.base_model),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, str(adapter))
    model.to("cuda").eval()
    model.config.use_cache = False
    return model, tokenizer


def resolve_optional_path(value):
    if value is None:
        return None
    raw = str(value).strip()
    if raw == "" or raw.lower() in {"none", "null", "false", "0"}:
        return None
    return Path(raw)


def build_batch(tokenizer, args):
    text = args.text
    encoded = tokenizer([text], return_tensors="pt", add_special_tokens=True)
    input_ids = encoded.input_ids[:, : args.max_length].to("cuda")
    if input_ids.shape[1] < 2:
        raise RuntimeError("Need at least two tokens for NLL parity")
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    return input_ids, attention_mask, labels


def find_decoder_layers(model):
    candidates = [model]
    for attr in ("base_model", "model"):
        value = getattr(model, attr, None)
        if value is not None:
            candidates.append(value)
    base_model = getattr(model, "base_model", None)
    if base_model is not None:
        for attr in ("model", "base_model"):
            value = getattr(base_model, attr, None)
            if value is not None:
                candidates.append(value)
    for candidate in candidates:
        current = candidate
        for attr in ("model", "layers"):
            current = getattr(current, attr, None)
            if current is None:
                break
        if current is not None:
            return list(current)
        current = getattr(candidate, "layers", None)
        if current is not None:
            return list(current)
    return []


def find_lm_model(model):
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model
    base_model = getattr(model, "base_model", None)
    if base_model is not None:
        inner = getattr(base_model, "model", None)
        if inner is not None and hasattr(inner, "model") and hasattr(inner.model, "embed_tokens"):
            return inner
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "model") and hasattr(inner.model, "embed_tokens"):
        return inner
    raise RuntimeError("Could not locate Fast-dLLM CausalLM module inside model")


@contextlib.contextmanager
def layer_trace(model, enabled: bool):
    if not enabled:
        yield None
        return
    layers = find_decoder_layers(model)
    traces = [None for _ in layers]
    handles = []
    for index, layer in enumerate(layers):
        def hook(_module, _inputs, output, *, layer_index=index):
            traces[layer_index] = output.detach().float().cpu()
        handles.append(layer.register_forward_hook(hook))
    try:
        yield traces
    finally:
        for handle in handles:
            handle.remove()


def summarize_layer_diffs(fla_traces, torch_traces):
    if fla_traces is None or torch_traces is None:
        return None
    summaries = []
    for index, (fla_tensor, torch_tensor) in enumerate(zip(fla_traces, torch_traces)):
        if fla_tensor is None or torch_tensor is None:
            summaries.append({"layer": index, "missing": True})
            continue
        diff = (fla_tensor - torch_tensor).abs()
        summaries.append(
            {
                "layer": index,
                "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
                "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
            }
        )
    return summaries


def first_gdn_fp32_parity(model, input_ids):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    lm_model = find_lm_model(model)
    modeling_module = __import__(lm_model.__class__.__module__, fromlist=["dummy"])
    gdn_layer = None
    for layer in lm_model.model.layers:
        if getattr(layer, "layer_type", None) == "linear_attention":
            gdn_layer = layer.linear_attn
            break
    if gdn_layer is None:
        raise RuntimeError("No linear_attention layer found for first-GDN parity")

    with torch.no_grad():
        hidden = lm_model.model.embed_tokens(input_ids)
        seq_len = hidden.shape[1]
        mixed_qkv = gdn_layer.in_proj_qkv(hidden).transpose(1, 2)
        mixed_qkv = F.silu(gdn_layer.conv1d(mixed_qkv)[:, :, :seq_len]).transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [gdn_layer.key_dim, gdn_layer.key_dim, gdn_layer.value_dim],
            dim=-1,
        )
        query = query.reshape(1, seq_len, -1, gdn_layer.head_k_dim)
        key = key.reshape(1, seq_len, -1, gdn_layer.head_k_dim)
        value = value.reshape(1, seq_len, -1, gdn_layer.head_v_dim)
        beta = gdn_layer.in_proj_b(hidden).sigmoid()
        g = -gdn_layer.A_log.float().exp() * F.softplus(gdn_layer.in_proj_a(hidden).float() + gdn_layer.dt_bias)
        if gdn_layer.num_v_heads // gdn_layer.num_k_heads > 1:
            repeat = gdn_layer.num_v_heads // gdn_layer.num_k_heads
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)
        query = modeling_module.l2norm(query, dim=-1).float()
        key = modeling_module.l2norm(key, dim=-1).float()
        value = value.float()
        beta = beta.float()
        g = g.float()
        torch_output, torch_state = modeling_module._torch_chunk_gated_delta_rule_impl(
            query,
            key,
            value,
            g,
            beta,
            chunk_size=64,
            output_final_state=True,
        )
        fla_output, fla_state = chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            scale=query.shape[-1] ** -0.5,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
            use_beta_sigmoid_in_kernel=False,
            allow_neg_eigval=False,
        )
        torch.cuda.synchronize()
    output_stats = diff_stats(fla_output, torch_output, rtol=5e-4, atol=5e-4)
    state_stats = diff_stats(fla_state, torch_state, rtol=5e-4, atol=5e-4)
    return {
        "output": output_stats,
        "final_state": state_stats,
        "g_min": float(g.min().item()),
        "g_max": float(g.max().item()),
        "g_mean": float(g.mean().item()),
        "passed": bool(output_stats["allclose"] and state_stats["allclose"]),
    }


def run_forward(model, input_ids, attention_mask, labels, *, backend: str, trace_layers: bool):
    with patched_env(
        FASTDLLM_GDN_KERNEL=backend,
        FASTDLLM_FLARE_TWO_STREAM=None,
        FLARE_TWO_STREAM=None,
        FASTDLLM_COMPILE_GDN_SCAN="0",
    ):
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with layer_trace(model, trace_layers) as traces:
            with torch.no_grad():
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        return {
            "loss": output.loss.detach().float().cpu(),
            "logits": output.logits.detach().float().cpu(),
            "layer_traces": traces,
            "seconds": seconds,
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / (1024**3),
        }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER), help="Adapter path, or 'none'.")
    parser.add_argument("--tokenizer-path", default=None, help="Tokenizer path, or 'none'.")
    parser.add_argument("--text", default="Q: What is 2 + 2?\\nA: The answer is")
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2.5e-1)
    parser.add_argument("--loss-atol", type=float, default=2e-3)
    parser.add_argument("--trace-layer-diffs", action="store_true")
    parser.add_argument("--skip-first-gdn-fp32", action="store_true")
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_cuda_env()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FASTDLLM_GDN_KERNEL=fla")
    torch.cuda.set_device(0)
    model, tokenizer = load_model_and_tokenizer(args)
    input_ids, attention_mask, labels = build_batch(tokenizer, args)
    torch_result = run_forward(
        model,
        input_ids,
        attention_mask,
        labels,
        backend="torch",
        trace_layers=args.trace_layer_diffs,
    )
    fla_result = run_forward(
        model,
        input_ids,
        attention_mask,
        labels,
        backend="fla",
        trace_layers=args.trace_layer_diffs,
    )
    logits = diff_stats(fla_result["logits"], torch_result["logits"], rtol=args.rtol, atol=args.atol)
    loss_abs = float((fla_result["loss"] - torch_result["loss"]).abs().item())
    first_gdn = None if args.skip_first_gdn_fp32 else first_gdn_fp32_parity(model, input_ids)
    passed = logits["allclose"] and loss_abs <= args.loss_atol and (first_gdn is None or first_gdn["passed"])
    payload = {
        "status": "PASS" if passed else "FAIL",
        "base_model": str(args.base_model),
        "adapter": str(resolve_optional_path(args.adapter)) if resolve_optional_path(args.adapter) else None,
        "device": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "input_tokens": int(input_ids.shape[1]),
        "rtol": args.rtol,
        "atol": args.atol,
        "loss_atol": args.loss_atol,
        "torch": {
            "loss": float(torch_result["loss"].item()),
            "seconds": torch_result["seconds"],
            "peak_allocated_gb": torch_result["peak_allocated_gb"],
            "peak_reserved_gb": torch_result["peak_reserved_gb"],
        },
        "fla": {
            "loss": float(fla_result["loss"].item()),
            "seconds": fla_result["seconds"],
            "peak_allocated_gb": fla_result["peak_allocated_gb"],
            "peak_reserved_gb": fla_result["peak_reserved_gb"],
        },
        "loss_abs": loss_abs,
        "logits": logits,
        "first_gdn_fp32": first_gdn,
        "layer_diffs": summarize_layer_diffs(fla_result.get("layer_traces"), torch_result.get("layer_traces")),
    }
    if args.print_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("FLA real-weight logits/NLL parity")
        print(f"base_model={args.base_model}")
        print(f"adapter={args.adapter}")
        print(f"device={payload['device']} capability={payload['capability']} input_tokens={payload['input_tokens']}")
        print(
            f"torch_loss={payload['torch']['loss']:.6g} fla_loss={payload['fla']['loss']:.6g} "
            f"loss_abs={loss_abs:.6g}"
        )
        print(f"logits={json.dumps(logits, sort_keys=True)}")
        print("FINAL:", payload["status"])
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
