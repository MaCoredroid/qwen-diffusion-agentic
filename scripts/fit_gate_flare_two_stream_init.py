#!/usr/bin/env python3
"""One-step FLARE two-stream fit gate from the local Qwen3.5-9B init.

Loads the local bridge in 4-bit QLoRA, attaches a tiny LoRA adapter, runs one
optimizer step under FASTDLLM_FLARE_TWO_STREAM=1 on a fixed synthetic block,
and reports finite L_AR/L_diff/total plus trainable gradient norm.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/qwen3.5-9b-fastdllm-init")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    return parser.parse_args()


def configure_cuda_env(root: Path):
    cuda_root = root / ".venv-fastdllm" / "lib" / "python3.10" / "site-packages" / "nvidia" / "cu13"
    if cuda_root.exists():
        os.environ.setdefault("CUDA_HOME", str(cuda_root))
        os.environ["PATH"] = f"{cuda_root / 'bin'}:{os.environ.get('PATH', '')}"
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_root / 'lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def loss_parts_from_model(model):
    candidates = [model, getattr(model, "base_model", None)]
    base_model = getattr(model, "base_model", None)
    if base_model is not None:
        candidates.append(getattr(base_model, "model", None))
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "_last_flare_loss_parts"):
            return candidate._last_flare_loss_parts
    return {}


def grad_stats(model):
    total_sq = 0.0
    max_abs = 0.0
    tensors = 0
    finite = True
    for parameter in model.parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        tensors += 1
        finite = finite and bool(torch.isfinite(grad).all().item())
        total_sq += float(grad.float().pow(2).sum().item())
        if grad.numel():
            max_abs = max(max_abs, float(grad.float().abs().max().item()))
    return total_sq**0.5, max_abs, tensors, finite


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    model_dir = (root / args.model_dir).resolve()
    configure_cuda_env(root)
    os.environ["FASTDLLM_FLARE_TWO_STREAM"] = "1"
    os.environ["FASTDLLM_TRAIN_BD_SIZE"] = "32"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real-init 4-bit one-step gate")
    torch.cuda.set_device(0)

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.train()

    seq_len = args.seq_len
    if seq_len % 32:
        raise ValueError("--seq-len must be divisible by 32 for the default training bd_size")
    vocab_size = int(model.config.vocab_size)
    input_ids = (torch.arange(seq_len, device="cuda", dtype=torch.long)[None] * 13 + 1000) % (vocab_size - 10)
    input_ids = input_ids + 5
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)
    doc_ids = torch.zeros_like(input_ids)
    mask_indices = torch.zeros_like(input_ids, dtype=torch.bool)
    mask_indices[:, 1::2] = True

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        doc_ids=doc_ids,
        flare_mask_indices=mask_indices,
    )
    loss = output.loss
    loss.backward()
    grad_norm, grad_max_abs, grad_tensors, grads_finite = grad_stats(model)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    parts = loss_parts_from_model(model)
    total = float(loss.detach().float().cpu().item())
    ar = float(parts.get("ar", torch.tensor(float("nan"))).float().cpu().item())
    diff = float(parts.get("diff", torch.tensor(float("nan"))).float().cpu().item())
    all_finite = (
        torch.isfinite(loss.detach()).item()
        and torch.isfinite(torch.tensor(ar)).item()
        and torch.isfinite(torch.tensor(diff)).item()
        and grads_finite
    )
    grad_ok = grad_norm > 0.0 and grad_tensors > 0
    passed = bool(all_finite and grad_ok)

    allocated_gb = torch.cuda.max_memory_allocated() / (1024**3)
    reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)
    print("FLARE two-stream real-init one-step fit gate")
    print(f"model_dir={model_dir}")
    print(f"seq_len={seq_len} lr={args.lr:g} lora_r={args.lora_r} seed={args.seed}")
    print(f"L_total={total:.6g} L_AR={ar:.6g} L_diff={diff:.6g}")
    print(
        f"grad_norm={grad_norm:.6g} grad_max_abs={grad_max_abs:.6g} "
        f"grad_tensors={grad_tensors} grads_finite={grads_finite}"
    )
    print(f"cuda_peak_allocated_gb={allocated_gb:.3f} cuda_peak_reserved_gb={reserved_gb:.3f}")
    print("FINAL:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
