#!/usr/bin/env python3
"""End-to-end real-weight logits/NLL parity for FASTDLLM_GDN_KERNEL=fla."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from pathlib import Path

import torch


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

    tokenizer_path = str(args.tokenizer_path or args.adapter or args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.base_model),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.to("cuda").eval()
    model.config.use_cache = False
    return model, tokenizer


def build_batch(tokenizer, args):
    text = args.text
    encoded = tokenizer([text], return_tensors="pt", add_special_tokens=True)
    input_ids = encoded.input_ids[:, : args.max_length].to("cuda")
    if input_ids.shape[1] < 2:
        raise RuntimeError("Need at least two tokens for NLL parity")
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    return input_ids, attention_mask, labels


def run_forward(model, input_ids, attention_mask, labels, *, backend: str):
    with patched_env(
        FASTDLLM_GDN_KERNEL=backend,
        FASTDLLM_FLARE_TWO_STREAM=None,
        FLARE_TWO_STREAM=None,
        FASTDLLM_COMPILE_GDN_SCAN="0",
    ):
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
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
            "seconds": seconds,
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / (1024**3),
        }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--text", default="Q: What is 2 + 2?\\nA: The answer is")
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--loss-atol", type=float, default=2e-2)
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
    torch_result = run_forward(model, input_ids, attention_mask, labels, backend="torch")
    fla_result = run_forward(model, input_ids, attention_mask, labels, backend="fla")
    logits = diff_stats(fla_result["logits"], torch_result["logits"], rtol=args.rtol, atol=args.atol)
    loss_abs = float((fla_result["loss"] - torch_result["loss"]).abs().item())
    passed = logits["allclose"] and loss_abs <= args.loss_atol
    payload = {
        "status": "PASS" if passed else "FAIL",
        "base_model": str(args.base_model),
        "adapter": str(args.adapter) if args.adapter else None,
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
