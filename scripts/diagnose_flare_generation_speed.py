#!/usr/bin/env python3
"""Diagnose Fast-dLLM block-diffusion generation speed on one heldout case."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
MASK_ID = 151665
STOP_TOKEN_ID = 151645


def configure_cuda_env() -> None:
    venv_root = Path(sys.executable).resolve().parents[1]
    cuda_root = (
        venv_root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "nvidia"
        / "cu13"
    )
    if not cuda_root.exists():
        cuda_root = ROOT / ".venv-fastdllm/lib/python3.10/site-packages/nvidia/cu13"
    if cuda_root.exists():
        os.environ.setdefault("CUDA_HOME", str(cuda_root))
        os.environ["PATH"] = f"{cuda_root / 'bin'}:{os.environ.get('PATH', '')}"
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_root / 'lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.pop("FASTDLLM_FLARE_TWO_STREAM", None)
    os.environ.pop("FLARE_TWO_STREAM", None)


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def load_eval_helpers():
    spec = importlib.util.spec_from_file_location("flare_eval", ROOT / "scripts/eval_flare_stage1_ab_diffusion.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_forward(fn):
    sync()
    start = time.perf_counter()
    out = fn()
    sync()
    return out, time.perf_counter() - start


def load_model(model_path: Path, adapter_path: Path | None, four_bit: bool):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": {"": 0},
    }
    if four_bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def sample_with_top_p(model, logits, top_p: float, temperature: float, mask_id: int | None = None):
    if mask_id is not None:
        logits = logits.clone()
        logits[..., mask_id] = torch.finfo(logits.dtype).min
    if hasattr(model, "sample_with_top_p"):
        return model.sample_with_top_p(logits, top_p=top_p, temperature=temperature)
    probs = torch.softmax(logits / max(temperature, 1e-12), dim=-1) if temperature > 0 else torch.softmax(logits, dim=-1)
    return probs.argmax(dim=-1), probs


def apply_repetition_penalty(logits: torch.Tensor, token_ids: torch.Tensor, penalty: float, mask_id: int) -> torch.Tensor:
    if penalty <= 1.0:
        return logits
    logits = logits.clone()
    for row_idx in range(logits.shape[0]):
        row_tokens = token_ids[row_idx]
        row_tokens = row_tokens[row_tokens != mask_id]
        if row_tokens.numel() == 0:
            continue
        unique_tokens = torch.unique(row_tokens)
        selected = logits[row_idx, :, unique_tokens]
        logits[row_idx, :, unique_tokens] = torch.where(selected < 0, selected * penalty, selected / penalty)
    return logits


def unwrap_lm_model(model):
    if hasattr(model, "get_base_model"):
        try:
            return model.get_base_model()
        except Exception:
            pass
    return model


@torch.no_grad()
def flare_two_stream_noisy_logits(model, clean_input_ids, noisy_input_ids, *, block_size: int, mask_id: int):
    lm_model = unwrap_lm_model(model)
    if clean_input_ids.shape != noisy_input_ids.shape:
        raise ValueError(
            f"clean/noisy shape mismatch: {tuple(clean_input_ids.shape)} vs {tuple(noisy_input_ids.shape)}"
        )
    if clean_input_ids.shape[1] % block_size:
        raise ValueError(f"sequence length {clean_input_ids.shape[1]} is not divisible by block_size={block_size}")
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
        clean_hidden, noisy_hidden = lm_model._flare_two_stream_layer_forward(
            layer,
            clean_hidden,
            noisy_hidden,
            doc_ids=doc_ids,
            noisy_doc_ids=noisy_doc_ids,
            clean_mask=clean_mask,
            two_stream_mask=two_stream_mask,
            clean_position_ids=clean_position_ids,
            noisy_position_ids=noisy_position_ids,
            block_size=block_size,
        )

    noisy_hidden = lm_model.model.norm(noisy_hidden)
    return lm_model.lm_head(noisy_hidden)


@torch.no_grad()
def diagnose_sample(model, input_ids, tokenizer, args) -> dict[str, Any]:
    device = input_ids.device
    block_size = args.block_size
    small_block_size = args.small_block_size
    min_len = input_ids.shape[1]
    seq_len = torch.tensor([min_len], device=device, dtype=torch.long)
    num_blocks = args.max_new_tokens // block_size + int(seq_len.max().item()) // block_size
    start_block_idx = min_len // block_size
    num_small_blocks = block_size // small_block_size
    seq_block_idx = seq_len // block_size
    finished_flag = torch.zeros((1,), device=device, dtype=torch.bool)
    metrics: dict[str, Any] = {
        "prompt_tokens": min_len,
        "block_size": block_size,
        "small_block_size": small_block_size,
        "max_new_tokens": args.max_new_tokens,
        "threshold": args.threshold,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "start_block_idx": int(start_block_idx),
        "num_blocks_loop": int(num_blocks - start_block_idx),
        "prefill_seconds": None,
        "denoise_forwards": 0,
        "commit_forwards": 0,
        "denoise_forward_seconds": 0.0,
        "commit_forward_seconds": 0.0,
        "total_committed_by_denoise": 0,
        "total_actual_nonmask_commits": 0,
        "total_selected_mask_token": 0,
        "total_natural_threshold_commits": 0,
        "total_forced_argmax_commits": 0,
        "blocks": [],
    }

    if min_len > block_size:
        prefix_len = min_len // block_size * block_size
        output, seconds = timed_forward(
            lambda: model.forward(
                input_ids=input_ids[:, :prefix_len],
                use_cache=True,
                update_past_key_values=True,
                block_size=block_size,
            )
        )
        metrics["prefill_seconds"] = seconds
        past_key_values = output.past_key_values
        if min_len % block_size == 0:
            next_token = output.logits[:, -1:, :].argmax(dim=-1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    else:
        past_key_values = None

    sample_indices = torch.arange(1, device=device)
    finished_samples = {}
    total_start = time.perf_counter()
    for block_idx in range(start_block_idx, num_blocks):
        if args.max_blocks is not None and len(metrics["blocks"]) >= args.max_blocks:
            metrics["stopped_early"] = "max_blocks"
            break
        block_start_time = time.perf_counter()
        if bool(finished_flag.all().item()):
            break
        if bool((seq_block_idx == block_idx).all().item()):
            pad_len = block_size if args.fresh_generation_blocks else block_size - input_ids.shape[1] % block_size
            x_init = args.mask_id * torch.ones((input_ids.shape[0], pad_len), device=device, dtype=torch.long)
            x_init = torch.cat([input_ids, x_init], dim=1)
            input_ids = x_init
        else:
            x_init = input_ids[:, : (block_idx + 1) * block_size]

        x_init[finished_flag, -block_size:] = tokenizer.pad_token_id
        x_t = x_init.clone()
        block_metrics: dict[str, Any] = {
            "block_idx": int(block_idx),
            "initial_masks": int((x_t[:, -block_size:] == args.mask_id).sum().item()),
            "denoise_steps": 0,
            "commit_forwards": 0,
            "tokens_committed": 0,
            "actual_nonmask_commits": 0,
            "selected_mask_token": 0,
            "natural_threshold_commits": 0,
            "forced_argmax_commits": 0,
            "denoise_forward_seconds": [],
            "commit_forward_seconds": [],
            "max_prob_min": None,
            "max_prob_mean": None,
            "step_trace": [],
        }
        max_probs = []
        block_past_key_values = None
        while True:
            mask_idx = x_t[:, -block_size:] == args.mask_id
            if int(mask_idx.sum().item()) == 0:
                if bool(finished_flag.all().item()):
                    break
                output, seconds = timed_forward(
                    lambda: model.forward(
                        input_ids=x_t[:, -block_size:],
                        use_cache=True,
                        past_key_values=past_key_values,
                        update_past_key_values=True,
                        block_size=block_size,
                    )
                )
                past_key_values = output.past_key_values
                next_token = output.logits[:, -1:, :].argmax(dim=-1)
                next_token[finished_flag] = tokenizer.pad_token_id
                x_t = torch.cat([x_t, next_token], dim=1)
                block_metrics["commit_forwards"] += 1
                block_metrics["commit_forward_seconds"].append(seconds)
                metrics["commit_forwards"] += 1
                metrics["commit_forward_seconds"] += seconds
                break

            for small_block_idx in range(num_small_blocks):
                small_block_start_idx = small_block_idx * small_block_size
                small_block_end_idx = small_block_start_idx + small_block_size
                start = -block_size + small_block_start_idx
                end = None if block_size == small_block_end_idx else -block_size + small_block_end_idx
                while True:
                    mask_idx = x_t[:, -block_size:] == args.mask_id
                    remaining = int(mask_idx[:, start:end].sum().item())
                    if remaining == 0:
                        break
                    output, seconds = timed_forward(
                        lambda: model.forward(
                            input_ids=x_t[:, -block_size:],
                            use_cache=True,
                            past_key_values=past_key_values,
                            update_past_key_values=False,
                        )
                    )
                    logits = torch.cat([output.logits[:, :1, :], output.logits[:, :-1, :]], dim=1)[:, start:end]
                    logits = apply_repetition_penalty(logits, x_t, args.repetition_penalty, args.mask_id)
                    x_1, p_1t = sample_with_top_p(
                        model,
                        logits,
                        args.top_p,
                        args.temperature,
                        mask_id=args.mask_id if args.ban_mask_logit else None,
                    )
                    x1_p = torch.squeeze(torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1)
                    active_probs = torch.where(mask_idx[:, start:end], x1_p, torch.full_like(x1_p, -torch.inf))
                    natural = active_probs > args.threshold
                    natural_count = int(natural.sum().item())
                    max_prob_idx = active_probs.argmax(dim=-1)
                    unmask_idx = natural.clone()
                    unmask_idx[torch.arange(x_1.shape[0], device=device), max_prob_idx] = True
                    unmask_idx = unmask_idx & mask_idx[:, start:end]
                    committed = int(unmask_idx.sum().item())
                    selected_mask_token = int(((x_1 == args.mask_id) & unmask_idx).sum().item())
                    actual_nonmask_commits = committed - selected_mask_token
                    forced = max(0, committed - natural_count)
                    x_t[:, start:end][unmask_idx] = x_1[unmask_idx]
                    max_prob = float(active_probs.max().detach().float().cpu())
                    max_probs.append(max_prob)
                    block_metrics["denoise_steps"] += 1
                    block_metrics["tokens_committed"] += committed
                    block_metrics["actual_nonmask_commits"] += actual_nonmask_commits
                    block_metrics["selected_mask_token"] += selected_mask_token
                    block_metrics["natural_threshold_commits"] += natural_count
                    block_metrics["forced_argmax_commits"] += forced
                    block_metrics["denoise_forward_seconds"].append(seconds)
                    if len(block_metrics["step_trace"]) < args.trace_steps:
                        block_metrics["step_trace"].append(
                            {
                                "remaining_before": remaining,
                                "committed": committed,
                                "actual_nonmask_commits": actual_nonmask_commits,
                                "selected_mask_token": selected_mask_token,
                                "natural_threshold": natural_count,
                                "forced_argmax": forced,
                                "max_prob": max_prob,
                                "forward_seconds": seconds,
                            }
                        )
                    metrics["denoise_forwards"] += 1
                    metrics["denoise_forward_seconds"] += seconds
                    metrics["total_committed_by_denoise"] += committed
                    metrics["total_actual_nonmask_commits"] += actual_nonmask_commits
                    metrics["total_selected_mask_token"] += selected_mask_token
                    metrics["total_natural_threshold_commits"] += natural_count
                    metrics["total_forced_argmax_commits"] += forced
                    stop_hits = torch.isin(x_1, torch.tensor(args.stop_token_ids, device=x_1.device))
                    finished_flag = finished_flag | (stop_hits & unmask_idx).any(dim=1)
                    if args.max_denoise_steps is not None and metrics["denoise_forwards"] >= args.max_denoise_steps:
                        metrics["stopped_early"] = "max_denoise_steps"
                        break
                if metrics.get("stopped_early"):
                    break
            if metrics.get("stopped_early"):
                break

        if input_ids.shape[1] == x_t.shape[1]:
            input_ids = x_t
        else:
            input_ids[:, : (block_idx + 1) * block_size] = x_t[:, :-1]
            if bool((seq_block_idx == block_idx).all().item()):
                input_ids = torch.cat([input_ids, x_t[:, -1:]], dim=1)
            elif input_ids.shape[1] <= (block_idx + 1) * block_size:
                input_ids = x_t
            else:
                input_ids[seq_block_idx == block_idx, (block_idx + 1) * block_size] = x_t[
                    seq_block_idx == block_idx, (block_idx + 1) * block_size
                ]
        seq_block_idx[seq_block_idx == block_idx] = block_idx + 1
        if bool(finished_flag.any().item()):
            for sample_idx in range(x_t.shape[0]):
                if bool(finished_flag[sample_idx].item()):
                    original_idx = int(sample_indices[sample_idx].item())
                    finished_samples[original_idx] = x_t[sample_idx : sample_idx + 1].clone().squeeze(dim=0)
            break

        block_metrics["block_seconds"] = time.perf_counter() - block_start_time
        if max_probs:
            block_metrics["max_prob_min"] = min(max_probs)
            block_metrics["max_prob_mean"] = sum(max_probs) / len(max_probs)
        metrics["blocks"].append(block_metrics)
        print(
            "[diag-block] "
            + json.dumps(
                {
                    "block_idx": block_metrics["block_idx"],
                    "initial_masks": block_metrics["initial_masks"],
                    "denoise_steps": block_metrics["denoise_steps"],
                    "tokens_committed": block_metrics["tokens_committed"],
                    "actual_nonmask_commits": block_metrics["actual_nonmask_commits"],
                    "selected_mask_token": block_metrics["selected_mask_token"],
                    "natural_threshold_commits": block_metrics["natural_threshold_commits"],
                    "forced_argmax_commits": block_metrics["forced_argmax_commits"],
                    "block_seconds": block_metrics["block_seconds"],
                    "max_prob_mean": block_metrics["max_prob_mean"],
                    "max_prob_min": block_metrics["max_prob_min"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if metrics.get("stopped_early"):
            break

    metrics["total_seconds"] = time.perf_counter() - total_start + (metrics["prefill_seconds"] or 0.0)
    metrics["output_tokens_including_prompt"] = int(input_ids.shape[1])
    metrics["new_tokens_materialized"] = int(max(0, input_ids.shape[1] - min_len))
    generated_ids = input_ids[0, min_len:].detach().cpu()
    metrics["generated_mask_count"] = int((generated_ids == args.mask_id).sum().item())
    metrics["generated_text"] = tokenizer.decode(generated_ids, skip_special_tokens=True)
    metrics["generated_blocks"] = []
    for block_start in range(0, int(generated_ids.numel()), block_size):
        block_ids = generated_ids[block_start : block_start + block_size]
        metrics["generated_blocks"].append(
            {
                "block": block_start // block_size,
                "tokens": int(block_ids.numel()),
                "mask_count": int((block_ids == args.mask_id).sum().item()),
                "text": tokenizer.decode(block_ids, skip_special_tokens=True),
            }
        )
    if metrics["denoise_forwards"]:
        metrics["mean_denoise_forward_seconds"] = metrics["denoise_forward_seconds"] / metrics["denoise_forwards"]
        metrics["mean_tokens_committed_per_denoise_forward"] = (
            metrics["total_committed_by_denoise"] / metrics["denoise_forwards"]
        )
        metrics["mean_actual_nonmask_commits_per_denoise_forward"] = (
            metrics["total_actual_nonmask_commits"] / metrics["denoise_forwards"]
        )
    else:
        metrics["mean_denoise_forward_seconds"] = None
        metrics["mean_tokens_committed_per_denoise_forward"] = None
        metrics["mean_actual_nonmask_commits_per_denoise_forward"] = None
    if metrics["commit_forwards"]:
        metrics["mean_commit_forward_seconds"] = metrics["commit_forward_seconds"] / metrics["commit_forwards"]
    else:
        metrics["mean_commit_forward_seconds"] = None
    return metrics


@torch.no_grad()
def diagnose_full_context_sample(model, input_ids, tokenizer, args) -> dict[str, Any]:
    output_ids = input_ids
    original_len = input_ids.shape[1]
    metrics: dict[str, Any] = {
        "mode": "full_context",
        "prompt_tokens": original_len,
        "block_size": args.block_size,
        "small_block_size": args.small_block_size,
        "max_new_tokens": args.max_new_tokens,
        "threshold": args.threshold,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "denoise_semantics": "two_stream" if args.two_stream_denoise else "single_stream_causal",
        "denoise_forwards": 0,
        "denoise_forward_seconds": 0.0,
        "total_committed_by_denoise": 0,
        "total_actual_nonmask_commits": 0,
        "total_selected_mask_token": 0,
        "total_natural_threshold_commits": 0,
        "total_forced_argmax_commits": 0,
        "blocks": [],
        "stop_token_ids": list(args.stop_token_ids),
    }
    stop_token_ids = torch.tensor(args.stop_token_ids, dtype=torch.long, device=output_ids.device)

    def truncate_if_stopped(sequence: torch.Tensor) -> torch.Tensor | None:
        generated = sequence[:, original_len:]
        if generated.numel() == 0:
            return None
        stop_mask = torch.isin(generated, stop_token_ids)
        if not bool(stop_mask.any().item()):
            return None
        first_stop = int(stop_mask.nonzero(as_tuple=False)[0, 1].item())
        prefix = generated[:, :first_stop]
        if bool((prefix == args.mask_id).any().item()):
            return None
        metrics["stop_token_hit"] = int(generated[:, first_stop].item())
        metrics["stop_offset"] = first_stop
        return sequence[:, : original_len + first_stop + 1]

    total_start = time.perf_counter()
    while output_ids.shape[1] - original_len < args.max_new_tokens:
        remaining_new = args.max_new_tokens - (output_ids.shape[1] - original_len)
        if args.fresh_generation_blocks:
            block_pad = args.block_size
        else:
            block_pad = args.block_size - (output_ids.shape[1] % args.block_size)
            if block_pad == 0:
                block_pad = args.block_size
        block_pad = min(block_pad, remaining_new)
        masks = torch.full(
            (output_ids.shape[0], block_pad),
            args.mask_id,
            dtype=torch.long,
            device=output_ids.device,
        )
        x_t = torch.cat([output_ids, masks], dim=1)
        window_len = min(args.block_size, x_t.shape[1])
        num_small_blocks = (window_len + args.small_block_size - 1) // args.small_block_size
        block_metrics: dict[str, Any] = {
            "generated_block": len(metrics["blocks"]),
            "block_pad": int(block_pad),
            "window_len": int(window_len),
            "initial_masks": int((x_t[:, -block_pad:] == args.mask_id).sum().item()),
            "denoise_steps": 0,
            "tokens_committed": 0,
            "actual_nonmask_commits": 0,
            "selected_mask_token": 0,
            "natural_threshold_commits": 0,
            "forced_argmax_commits": 0,
            "denoise_forward_seconds": [],
            "max_prob_min": None,
            "max_prob_mean": None,
            "step_trace": [],
        }
        block_start_time = time.perf_counter()
        max_probs = []
        while bool((x_t[:, -block_pad:] == args.mask_id).any().item()):
            for small_block_idx in range(num_small_blocks):
                small_block_start_idx = small_block_idx * args.small_block_size
                small_block_end_idx = min(small_block_start_idx + args.small_block_size, window_len)
                start = small_block_start_idx
                end = small_block_end_idx
                while True:
                    mask_idx = x_t[:, -window_len:] == args.mask_id
                    current_mask = mask_idx[:, start:end]
                    remaining = int(current_mask.sum().item())
                    if remaining == 0:
                        break
                    if args.two_stream_denoise:
                        noisy_logits, seconds = timed_forward(
                            lambda: flare_two_stream_noisy_logits(
                                model,
                                x_t,
                                x_t,
                                block_size=args.block_size,
                                mask_id=args.mask_id,
                            )
                        )
                        noisy_logits = noisy_logits[: x_t.shape[0]]
                        logits = torch.cat([noisy_logits[:, :1, :], noisy_logits[:, :-1, :]], dim=1)
                    else:
                        output, seconds = timed_forward(lambda: model(input_ids=x_t, use_cache=False))
                        logits = torch.cat([output.logits[:, :1, :], output.logits[:, :-1, :]], dim=1)
                    logits = logits[:, -window_len:][:, start:end]
                    logits = apply_repetition_penalty(logits, x_t, args.repetition_penalty, args.mask_id)
                    x_1, p_1t = sample_with_top_p(
                        model,
                        logits,
                        args.top_p,
                        args.temperature,
                        mask_id=args.mask_id if args.ban_mask_logit else None,
                    )
                    x1_p = torch.squeeze(torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1)
                    active_probs = torch.where(current_mask, x1_p, torch.full_like(x1_p, -torch.inf))
                    natural = active_probs > args.threshold
                    natural_count = int(natural.sum().item())
                    max_prob_idx = active_probs.argmax(dim=-1)
                    unmask_idx = natural.clone()
                    unmask_idx[torch.arange(x_1.shape[0], device=x_1.device), max_prob_idx] = True
                    unmask_idx = unmask_idx & current_mask
                    committed = int(unmask_idx.sum().item())
                    selected_mask_token = int(((x_1 == args.mask_id) & unmask_idx).sum().item())
                    actual_nonmask_commits = committed - selected_mask_token
                    forced = max(0, committed - natural_count)
                    window = x_t[:, -window_len:]
                    span = window[:, start:end].clone()
                    span[unmask_idx] = x_1[unmask_idx]
                    window[:, start:end] = span
                    x_t[:, -window_len:] = window
                    max_prob = float(active_probs.max().detach().float().cpu())
                    max_probs.append(max_prob)

                    block_metrics["denoise_steps"] += 1
                    block_metrics["tokens_committed"] += committed
                    block_metrics["actual_nonmask_commits"] += actual_nonmask_commits
                    block_metrics["selected_mask_token"] += selected_mask_token
                    block_metrics["natural_threshold_commits"] += natural_count
                    block_metrics["forced_argmax_commits"] += forced
                    block_metrics["denoise_forward_seconds"].append(seconds)
                    if len(block_metrics["step_trace"]) < args.trace_steps:
                        block_metrics["step_trace"].append(
                            {
                                "remaining_before": remaining,
                                "committed": committed,
                                "actual_nonmask_commits": actual_nonmask_commits,
                                "selected_mask_token": selected_mask_token,
                                "natural_threshold": natural_count,
                                "forced_argmax": forced,
                                "max_prob": max_prob,
                                "forward_seconds": seconds,
                            }
                        )
                    metrics["denoise_forwards"] += 1
                    metrics["denoise_forward_seconds"] += seconds
                    metrics["total_committed_by_denoise"] += committed
                    metrics["total_actual_nonmask_commits"] += actual_nonmask_commits
                    metrics["total_selected_mask_token"] += selected_mask_token
                    metrics["total_natural_threshold_commits"] += natural_count
                    metrics["total_forced_argmax_commits"] += forced
                    stopped = truncate_if_stopped(x_t)
                    if stopped is not None:
                        output_ids = stopped
                        block_metrics["block_seconds"] = time.perf_counter() - block_start_time
                        metrics["blocks"].append(block_metrics)
                        metrics["stopped_early"] = "stop_token"
                        break
                    if args.max_denoise_steps is not None and metrics["denoise_forwards"] >= args.max_denoise_steps:
                        metrics["stopped_early"] = "max_denoise_steps"
                        break
                if metrics.get("stopped_early"):
                    break
            if metrics.get("stopped_early"):
                break
            if bool((x_t[:, -block_pad:] == args.mask_id).all().item()):
                metrics["stopped_early"] = "no_progress_block"
                break
        if metrics.get("stopped_early") == "stop_token":
            break
        output_ids = x_t
        block_metrics["block_seconds"] = time.perf_counter() - block_start_time
        if max_probs:
            block_metrics["max_prob_min"] = min(max_probs)
            block_metrics["max_prob_mean"] = sum(max_probs) / len(max_probs)
        metrics["blocks"].append(block_metrics)
        print(
            "[diag-full-block] "
            + json.dumps(
                {
                    "generated_block": block_metrics["generated_block"],
                    "block_pad": block_metrics["block_pad"],
                    "denoise_steps": block_metrics["denoise_steps"],
                    "tokens_committed": block_metrics["tokens_committed"],
                    "actual_nonmask_commits": block_metrics["actual_nonmask_commits"],
                    "selected_mask_token": block_metrics["selected_mask_token"],
                    "natural_threshold_commits": block_metrics["natural_threshold_commits"],
                    "forced_argmax_commits": block_metrics["forced_argmax_commits"],
                    "block_seconds": block_metrics["block_seconds"],
                    "max_prob_mean": block_metrics["max_prob_mean"],
                    "max_prob_min": block_metrics["max_prob_min"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        stopped = truncate_if_stopped(output_ids)
        if stopped is not None:
            output_ids = stopped
            break
        if metrics.get("stopped_early"):
            break

    metrics["total_seconds"] = time.perf_counter() - total_start
    metrics["output_tokens_including_prompt"] = int(output_ids.shape[1])
    metrics["new_tokens_materialized"] = int(max(0, output_ids.shape[1] - original_len))
    generated_ids = output_ids[0, original_len:].detach().cpu()
    metrics["generated_mask_count"] = int((generated_ids == args.mask_id).sum().item())
    metrics["generated_text"] = tokenizer.decode(generated_ids, skip_special_tokens=True)
    metrics["generated_blocks"] = []
    for block_start in range(0, int(generated_ids.numel()), args.block_size):
        block_ids = generated_ids[block_start : block_start + args.block_size]
        metrics["generated_blocks"].append(
            {
                "block": block_start // args.block_size,
                "tokens": int(block_ids.numel()),
                "mask_count": int((block_ids == args.mask_id).sum().item()),
                "text": tokenizer.decode(block_ids, skip_special_tokens=True),
            }
        )
    if metrics["denoise_forwards"]:
        metrics["mean_denoise_forward_seconds"] = metrics["denoise_forward_seconds"] / metrics["denoise_forwards"]
        metrics["mean_tokens_committed_per_denoise_forward"] = (
            metrics["total_committed_by_denoise"] / metrics["denoise_forwards"]
        )
        metrics["mean_actual_nonmask_commits_per_denoise_forward"] = (
            metrics["total_actual_nonmask_commits"] / metrics["denoise_forwards"]
        )
    else:
        metrics["mean_denoise_forward_seconds"] = None
        metrics["mean_tokens_committed_per_denoise_forward"] = None
        metrics["mean_actual_nonmask_commits_per_denoise_forward"] = None
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(ROOT / "models/qwen3.5-9b-fastdllm-init"))
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--case-path", default=str(ROOT / "data/phaseA_retention/gsm8k_main_test_first20.jsonl"))
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--out-json", default=str(ROOT / "runs/flare_stage1_ab_pilot/generation_speed_diag_init_gsm8k0.json"))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--small-block-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--mask-id", type=int, default=None)
    parser.add_argument("--stop-token-id", type=int, default=None)
    parser.add_argument("--four-bit", action="store_true", default=True)
    parser.add_argument("--no-four-bit", action="store_false", dest="four_bit")
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--max-denoise-steps", type=int, default=None)
    parser.add_argument("--trace-steps", type=int, default=6)
    parser.add_argument("--ban-mask-logit", action="store_true")
    parser.add_argument("--full-context", action="store_true")
    parser.add_argument("--two-stream-denoise", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--force-gold-answer-tokens", type=int, default=0)
    parser.add_argument(
        "--tail-fill-generation",
        action="store_false",
        dest="fresh_generation_blocks",
        help="Compatibility mode: fill only the prompt-tail remainder before full generated blocks.",
    )
    parser.set_defaults(fresh_generation_blocks=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_cuda_env()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    helper = load_eval_helpers()
    rows = read_jsonl(Path(args.case_path))
    case = rows[args.case_index]
    model, tokenizer = load_model(Path(args.model), Path(args.adapter) if args.adapter else None, args.four_bit)
    config = helper.get_model_config(model)
    args.mask_id = helper.resolve_mask_id(config, tokenizer, args.mask_id)
    args.stop_token_ids = helper.resolve_stop_token_ids(config, tokenizer, args.stop_token_id)
    args.stop_token_id = int(args.stop_token_ids[0])
    print(
        "[diag-token-ids] "
        + json.dumps({"mask_id": args.mask_id, "stop_token_ids": args.stop_token_ids}, sort_keys=True),
        flush=True,
    )
    fewshot = read_jsonl(ROOT / "data/phaseA_retention/gsm8k_main_train_first5.jsonl", 5)
    prompt = helper.build_gsm8k_prompt(tokenizer, case, fewshot)
    input_ids = tokenizer([prompt], return_tensors="pt").input_ids.to("cuda")
    forced_prefix_text = ""
    forced_prefix_token_count = 0
    if args.force_gold_answer_tokens > 0:
        answer_ids = tokenizer(case["answer"], add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
        forced_prefix_token_count = min(int(args.force_gold_answer_tokens), int(answer_ids.shape[1]))
        forced_ids = answer_ids[:, :forced_prefix_token_count]
        input_ids = torch.cat([input_ids, forced_ids], dim=1)
        forced_prefix_text = tokenizer.decode(forced_ids[0].detach().cpu(), skip_special_tokens=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    print(
        f"[diag-start] prompt_tokens={input_ids.shape[1]} threshold={args.threshold} "
        f"block_size={args.block_size} max_new_tokens={args.max_new_tokens} "
        f"forced_gold_tokens={forced_prefix_token_count}",
        flush=True,
    )
    if args.full_context:
        result = diagnose_full_context_sample(model, input_ids, tokenizer, args)
    else:
        result = diagnose_sample(model, input_ids, tokenizer, args)
    result["case"] = {"path": args.case_path, "case_index": args.case_index, "idx": case.get("idx")}
    result["forced_gold_answer_tokens"] = forced_prefix_token_count
    result["forced_gold_answer_text"] = forced_prefix_text
    result["cuda_peak_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
    result["cuda_peak_reserved_gb"] = torch.cuda.max_memory_reserved() / (1024**3)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary_keys = [
        "prefill_seconds",
        "denoise_forwards",
        "commit_forwards",
        "mean_denoise_forward_seconds",
        "mean_commit_forward_seconds",
        "mean_tokens_committed_per_denoise_forward",
        "mean_actual_nonmask_commits_per_denoise_forward",
        "total_natural_threshold_commits",
        "total_forced_argmax_commits",
        "total_selected_mask_token",
        "total_actual_nonmask_commits",
        "total_seconds",
        "new_tokens_materialized",
    ]
    print("[diag-summary] " + json.dumps({k: result.get(k) for k in summary_keys}, sort_keys=True), flush=True)
    print(f"[diag-json] {out_path}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
