#!/usr/bin/env python3
"""Train one cached-SDTT fallback probe round from a cached top-k corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import types
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_STUDENT = ROOT / "runs/rl_multiturn_grpo_v2/from_selected_base_g4_step300/adapter_model"
DEFAULT_TOKENIZER = ROOT / "models/qwen3.5-9b-fastdllm-b1000-vllm-bf16"
DEFAULT_CORPUS = ROOT / "data/cached_sdtt_v2_teacher_probe/cached_sdtt_records.jsonl"
DEFAULT_OUT = ROOT / "runs/cached_sdtt_v2_teacher_probe/round1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--student-init-adapter", type=Path, default=DEFAULT_STUDENT)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--corpus-jsonl", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--micro-steps", type=int, default=4000)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--kbit-prepare", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-targets-per-record", type=int, default=0)
    parser.add_argument("--max-train-seq-tokens", type=int, default=640)
    parser.add_argument("--train-crop-tail-tokens", type=int, default=32)
    parser.add_argument("--gpu-index", type=int, default=0)
    return parser.parse_args()


def configure_env() -> None:
    os.environ.setdefault("FASTDLLM_FLARE_GDN_ROUTE", "route_i")
    os.environ.setdefault("FASTDLLM_FLARE_TWO_STREAM", "1")
    os.environ.setdefault("FLARE_TWO_STREAM", "1")
    os.environ.setdefault("FASTDLLM_GDN_KERNEL", "torch")
    os.environ.setdefault("FASTDLLM_BATCH_FLARE_NOISY_GDN", "1")
    os.environ.setdefault("FASTDLLM_OPTIMIZE_FLARE_CLEAN_GDN", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def sha256_file(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def unwrap_lm_model(model):
    if hasattr(model, "get_base_model"):
        try:
            return model.get_base_model()
        except Exception:
            pass
    return model


def differentiable_flare_noisy_hidden(
    model,
    clean_input_ids: torch.Tensor,
    noisy_input_ids: torch.Tensor,
    *,
    block_size: int,
    checkpoint_layers: bool = False,
):
    lm_model = unwrap_lm_model(model)
    if clean_input_ids.shape != noisy_input_ids.shape:
        raise ValueError("clean/noisy shape mismatch")
    if hasattr(lm_model, "_set_active_train_bd_size"):
        lm_model._set_active_train_bd_size(block_size)
    modeling_module = sys.modules[lm_model.__class__.__module__]
    doc_ids = torch.zeros_like(clean_input_ids, dtype=torch.long)
    noisy_pair_ids = noisy_input_ids.repeat(2, 1)
    noisy_doc_ids = doc_ids.repeat(2, 1)

    clean_hidden = lm_model.model.embed_tokens(clean_input_ids)
    noisy_hidden = lm_model.model.embed_tokens(noisy_pair_ids)
    clean_mask = modeling_module.doc_causal_bool_mask(doc_ids)
    two_stream_mask = modeling_module.flare_two_stream_bool_mask(noisy_doc_ids, block_size)
    clean_position_ids = modeling_module.local_position_ids_from_doc_ids(doc_ids)
    noisy_position_ids = modeling_module.local_position_ids_from_doc_ids(noisy_doc_ids)

    for layer in lm_model.model.layers:
        def layer_forward(clean_state, noisy_state):
            return lm_model._flare_two_stream_layer_forward(
                layer,
                clean_state,
                noisy_state,
                doc_ids=doc_ids,
                noisy_doc_ids=noisy_doc_ids,
                clean_mask=clean_mask,
                two_stream_mask=two_stream_mask,
                clean_position_ids=clean_position_ids,
                noisy_position_ids=noisy_position_ids,
                block_size=block_size,
            )

        if checkpoint_layers:
            clean_hidden, noisy_hidden = torch.utils.checkpoint.checkpoint(
                layer_forward,
                clean_hidden,
                noisy_hidden,
                use_reentrant=False,
            )
        else:
            clean_hidden, noisy_hidden = layer_forward(clean_hidden, noisy_hidden)

    noisy_hidden = lm_model.model.norm(noisy_hidden)
    return noisy_hidden[:1]


def crop_record_tensors(
    record: dict[str, Any],
    targets: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[int], list[int], list[dict[str, Any]], dict[str, Any]]:
    input_ids = list(record["input_ids"])
    noisy_ids = list(record["student_noisy_ids"])
    if len(input_ids) != len(noisy_ids):
        raise ValueError(f"record {record.get('record_id')} input/noisy length mismatch")
    max_len = int(args.max_train_seq_tokens or 0)
    if max_len <= 0 or len(input_ids) <= max_len:
        return input_ids, noisy_ids, targets, {
            "cropped": False,
            "crop_start": 0,
            "crop_end": len(input_ids),
            "orig_seq_len": len(input_ids),
        }

    positions = [int(item["pos"]) for item in targets]
    if not positions:
        return input_ids, noisy_ids, targets, {
            "cropped": False,
            "crop_start": 0,
            "crop_end": len(input_ids),
            "orig_seq_len": len(input_ids),
        }
    span_start = min(positions)
    span_end = max(positions) + 1
    span_len = span_end - span_start
    if span_len > max_len:
        start = span_start
        end = min(len(input_ids), start + max_len)
    else:
        tail = max(0, int(args.train_crop_tail_tokens or 0))
        end = min(len(input_ids), span_end + tail)
        start = max(0, end - max_len)
        if start > span_start:
            start = span_start
            end = min(len(input_ids), start + max_len)
    kept_targets: list[dict[str, Any]] = []
    for item in targets:
        pos = int(item["pos"])
        if start <= pos < end:
            copied = dict(item)
            copied["pos"] = pos - start
            kept_targets.append(copied)
    return input_ids[start:end], noisy_ids[start:end], kept_targets, {
        "cropped": True,
        "crop_start": start,
        "crop_end": end,
        "orig_seq_len": len(input_ids),
        "dropped_targets": len(targets) - len(kept_targets),
    }


def load_student(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_path), trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        str(args.base_model),
        trust_remote_code=True,
        local_files_only=True,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map={"": int(args.gpu_index)},
    )
    base.config.use_cache = False
    if args.gradient_checkpointing and hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable()
    elif hasattr(base, "gradient_checkpointing_disable"):
        base.gradient_checkpointing_disable()
    if args.kbit_prepare:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=bool(args.gradient_checkpointing))
    model = PeftModel.from_pretrained(base, str(args.student_init_adapter), is_trainable=True)
    model.config.use_cache = False
    repo_v2 = ROOT / "fast-dllm/v2"
    if str(repo_v2) not in sys.path:
        sys.path.insert(0, str(repo_v2))
    import generation_functions

    model.mdm_sample = types.MethodType(
        generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample,
        model,
    )
    model.train()
    return model, tokenizer


def trainable_parameter_count(model) -> tuple[int, int]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return int(trainable), int(total)


def record_loss(model, record: dict[str, Any], args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, Any]]:
    device = next(model.parameters()).device
    targets = list(record.get("targets") or [])
    if args.max_targets_per_record and len(targets) > int(args.max_targets_per_record):
        targets = targets[: int(args.max_targets_per_record)]
    cropped_input_ids, cropped_noisy_ids, targets, crop_info = crop_record_tensors(record, targets, args)
    if not targets:
        return torch.tensor(0.0, dtype=torch.float32, device=device), {"target_tokens": 0, **crop_info}
    input_ids = torch.tensor([cropped_input_ids], dtype=torch.long, device=device)
    noisy_ids = torch.tensor([cropped_noisy_ids], dtype=torch.long, device=device)

    noisy_hidden = differentiable_flare_noisy_hidden(
        model,
        input_ids,
        noisy_ids,
        block_size=int(args.block_size),
        checkpoint_layers=bool(args.gradient_checkpointing),
    )
    positions = torch.tensor([int(item["pos"]) for item in targets], dtype=torch.long, device=device)
    hidden_positions = torch.clamp(positions - 1, min=0)
    row_hidden = noisy_hidden[0].index_select(0, hidden_positions)
    row_logits = unwrap_lm_model(model).lm_head(row_hidden).float()
    target_ids = torch.tensor([item["top_ids"] for item in targets], dtype=torch.long, device=device)
    teacher_logprobs = torch.tensor([item["top_logprobs"] for item in targets], dtype=torch.float32, device=device)
    teacher_logprobs = teacher_logprobs - torch.logsumexp(teacher_logprobs, dim=-1, keepdim=True)

    temp = max(float(args.temperature), 1e-6)
    student_top_logits = row_logits.gather(dim=-1, index=target_ids) / temp
    student_logprobs = torch.log_softmax(student_top_logits, dim=-1)
    student_probs = student_logprobs.exp()
    reverse_kl = (student_probs * (student_logprobs - teacher_logprobs)).sum(dim=-1).mean()

    ce_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
    ce_weight = float(args.ce_weight)
    if ce_weight:
        gold = torch.tensor([int(item["gold_id"]) for item in targets], dtype=torch.long, device=device)
        ce_loss = F.cross_entropy(row_logits, gold, reduction="mean")
    loss = reverse_kl + ce_weight * ce_loss
    with torch.no_grad():
        top1 = target_ids[:, 0]
        student_argmax_on_support = target_ids.gather(
            dim=-1,
            index=student_top_logits.argmax(dim=-1, keepdim=True),
        ).squeeze(-1)
        support_top1_match = int(student_argmax_on_support.eq(top1).sum().item())
    return loss, {
        "target_tokens": int(len(targets)),
        "reverse_kl": float(reverse_kl.detach().cpu().item()),
        "ce": float(ce_loss.detach().cpu().item()),
        "support_top1_match": support_top1_match,
        **crop_info,
    }


def save_adapter(model, out_dir: Path, label: str) -> Path:
    path = out_dir / label
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    return path


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Cached-SDTT One-Probe Training",
        "",
        "Fallback: cached SDTT after DSCD teacher precheck failed.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Micro-steps | {summary['micro_steps_completed']} |",
        f"| Optimizer steps | {summary['optimizer_steps']} |",
        f"| Mean reverse KL | {summary.get('mean_reverse_kl')} |",
        f"| Mean loss | {summary.get('mean_loss')} |",
        f"| Trainable params | {summary['trainable_params']} |",
        f"| Peak CUDA allocated GiB | {summary.get('peak_cuda_allocated_gib')} |",
        "",
        f"Student init: `{summary['student_init_adapter']}`",
        f"Adapter out: `{summary['adapter_out']}`",
        "",
        "Loss caveat: sparse top-k reverse-KL over cached teacher support, with no full-vocab teacher mass.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    configure_env()
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    records = read_jsonl(args.corpus_jsonl)
    if not records:
        raise SystemExit(f"empty corpus: {args.corpus_jsonl}")
    model, _tokenizer = load_student(args)
    trainable, total = trainable_parameter_count(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    config = {
        "base_model": str(args.base_model),
        "student_init_adapter": str(args.student_init_adapter),
        "tokenizer_path": str(args.tokenizer_path),
        "corpus_jsonl": str(args.corpus_jsonl),
        "out_dir": str(args.out_dir),
        "micro_steps": int(args.micro_steps),
        "grad_accum": int(args.grad_accum),
        "learning_rate": float(args.learning_rate),
        "block_size": int(args.block_size),
        "temperature": float(args.temperature),
        "ce_weight": float(args.ce_weight),
        "max_train_seq_tokens": int(args.max_train_seq_tokens or 0),
        "train_crop_tail_tokens": int(args.train_crop_tail_tokens or 0),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "kbit_prepare": bool(args.kbit_prepare),
        "records": len(records),
        "trainable_params": trainable,
        "total_params": total,
        "git_head": git_head(),
        "script_sha256": sha256_file(Path(__file__)),
        "reverse_kl_caveat": "sparse top-k support only",
        "quality_rl_v5": "held",
    }
    write_json(args.out_dir / "config.json", config)

    order = list(range(len(records)))
    random.shuffle(order)
    order_cursor = 0
    loss_sum = 0.0
    kl_sum = 0.0
    target_sum = 0
    optimizer_steps = 0
    started = time.time()
    optimizer.zero_grad(set_to_none=True)

    for micro_step in range(1, int(args.micro_steps) + 1):
        if order_cursor >= len(order):
            random.shuffle(order)
            order_cursor = 0
        record = records[order[order_cursor]]
        order_cursor += 1
        step_start = time.time()
        loss, parts = record_loss(model, record, args)
        target_tokens = int(parts.get("target_tokens") or 0)
        if target_tokens <= 0:
            continue
        (loss / float(args.grad_accum)).backward()
        loss_value = float(loss.detach().cpu().item())
        loss_sum += loss_value
        kl_sum += float(parts.get("reverse_kl") or 0.0)
        target_sum += target_tokens

        did_step = False
        if micro_step % int(args.grad_accum) == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                float(args.max_grad_norm),
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            did_step = True
        else:
            grad_norm = torch.tensor(0.0)

        row = {
            "micro_step": micro_step,
            "optimizer_steps": optimizer_steps,
            "loss": loss_value,
            "reverse_kl": float(parts.get("reverse_kl") or 0.0),
            "ce": float(parts.get("ce") or 0.0),
            "target_tokens": target_tokens,
            "support_top1_match": int(parts.get("support_top1_match") or 0),
            "cropped": bool(parts.get("cropped") or False),
            "crop_start": parts.get("crop_start"),
            "crop_end": parts.get("crop_end"),
            "orig_seq_len": parts.get("orig_seq_len"),
            "dropped_targets": int(parts.get("dropped_targets") or 0),
            "optimizer_step": did_step,
            "grad_norm": float(grad_norm.detach().cpu().item()) if did_step else None,
            "seconds": time.time() - step_start,
        }
        if micro_step % int(args.log_every) == 0 or micro_step == 1:
            append_jsonl(metrics_path, row)
            print(
                f"micro_step={micro_step} opt={optimizer_steps} loss={loss_value:.6g} "
                f"rkl={row['reverse_kl']:.6g} targets={target_tokens} seconds={row['seconds']:.2f}",
                flush=True,
            )
        if int(args.save_every) > 0 and micro_step % int(args.save_every) == 0:
            save_adapter(model, args.out_dir, f"checkpoint-{micro_step}")

    if int(args.micro_steps) % int(args.grad_accum) != 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            float(args.max_grad_norm),
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
        append_jsonl(
            metrics_path,
            {
                "micro_step": int(args.micro_steps),
                "optimizer_steps": optimizer_steps,
                "event": "final_partial_optimizer_step",
                "grad_norm": float(grad_norm.detach().cpu().item()),
            },
        )

    adapter_out = save_adapter(model, args.out_dir, "adapter_model")
    elapsed = time.time() - started
    summary = {
        **config,
        "adapter_out": str(adapter_out),
        "micro_steps_completed": int(args.micro_steps),
        "optimizer_steps": optimizer_steps,
        "target_tokens_seen": int(target_sum),
        "mean_loss": loss_sum / max(1, int(args.micro_steps)),
        "mean_reverse_kl": kl_sum / max(1, int(args.micro_steps)),
        "elapsed_seconds": elapsed,
        "micro_steps_per_second": int(args.micro_steps) / max(elapsed, 1e-9),
        "peak_cuda_allocated_gib": (
            torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else None
        ),
    }
    write_json(args.out_dir / "summary.json", summary)
    write_report(args.out_dir / "report.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
