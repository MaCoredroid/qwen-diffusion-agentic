#!/usr/bin/env python3
"""Diffusion-mode eval for the FLARE Stage-1 A/B pilot.

This script intentionally avoids causal teacher-forced NLL. It computes a
single fixed-mask denoising NLL for init/A/B, then runs the same block-diffusion
sampler on the heldout GSM8K and MBPP slices.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import types
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_A = ROOT / "runs/flare_stage1_ab_pilot/diffusion_only_A_s1024_step200"
DEFAULT_B = ROOT / "runs/flare_stage1_ab_pilot/two_stream_B_s1024_step200"
DEFAULT_OUT = ROOT / "runs/flare_stage1_ab_pilot/diffusion_mode_eval"
IGNORE_INDEX = -100
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_row_hash(row: dict[str, Any]) -> str:
    payload = {
        "system": row.get("system"),
        "messages": row.get("messages"),
        "source": row.get("source"),
        "id": row.get("id"),
        "task": row.get("task"),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def content_hash(row: dict[str, Any]) -> str:
    payload = {
        "system": row.get("system"),
        "messages": row.get("messages"),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_train_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("instances"), list):
        return payload["instances"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported train payload shape in {path}")


def verify_disjoint(train_path: Path, heldout_path: Path) -> dict[str, Any]:
    train_rows = load_train_rows(train_path)
    heldout_rows = read_jsonl(heldout_path)
    train_full = {stable_row_hash(row) for row in train_rows}
    heldout_full = {stable_row_hash(row) for row in heldout_rows}
    train_content = {content_hash(row) for row in train_rows}
    heldout_content = {content_hash(row) for row in heldout_rows}
    full_overlap = sorted(train_full & heldout_full)
    content_overlap = sorted(train_content & heldout_content)
    result = {
        "train_path": str(train_path),
        "heldout_path": str(heldout_path),
        "train_count": len(train_rows),
        "heldout_count": len(heldout_rows),
        "full_hash_overlap_count": len(full_overlap),
        "content_hash_overlap_count": len(content_overlap),
        "sources": {
            "train": sorted({str(row.get("source")) for row in train_rows}),
            "heldout": sorted({str(row.get("source")) for row in heldout_rows}),
        },
    }
    if full_overlap or content_overlap:
        result["full_hash_overlap_prefix"] = full_overlap[:5]
        result["content_hash_overlap_prefix"] = content_overlap[:5]
        raise RuntimeError(f"Train/heldout overlap detected: {json.dumps(result, indent=2)}")
    return result


def apply_chat_template(tokenizer, messages: list[dict[str, str]], **kwargs):
    kwargs = dict(kwargs)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def row_messages(row: dict[str, Any], include_assistant: bool) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if row.get("system"):
        messages.append({"role": "system", "content": row["system"]})
    for message in row["messages"]:
        if not include_assistant and message.get("role") == "assistant":
            continue
        messages.append({"role": message["role"], "content": message["content"]})
    return messages


def template_encode(tokenizer, messages: list[dict[str, str]], add_generation_prompt: bool):
    encoded = apply_chat_template(
        tokenizer,
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=True,
    )
    if isinstance(encoded, str):
        tokenized = tokenizer(encoded, add_special_tokens=False)
        return list(tokenized["input_ids"]), list(tokenized.get("attention_mask", [1] * len(tokenized["input_ids"]))), None
    try:
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask", [1] * len(input_ids))
        assistant_mask = encoded.get("assistant_masks") or encoded.get("assistant_tokens_mask")
    except (TypeError, KeyError):
        input_ids = encoded
        attention_mask = [1] * len(input_ids)
        assistant_mask = None
    return list(input_ids), list(attention_mask), assistant_mask


def encode_labeled_row(tokenizer, row: dict[str, Any], max_length: int):
    messages = row_messages(row, include_assistant=True)
    input_ids, attention_mask, assistant_mask = template_encode(
        tokenizer,
        messages,
        add_generation_prompt=False,
    )
    labels = [IGNORE_INDEX] * len(input_ids)
    if assistant_mask is not None and any(assistant_mask):
        for idx, is_assistant in enumerate(assistant_mask):
            if idx < len(labels) and is_assistant:
                labels[idx] = input_ids[idx]
    else:
        for msg_idx, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            start_ids, _, _ = template_encode(tokenizer, messages[:msg_idx], add_generation_prompt=True)
            end_ids, _, _ = template_encode(tokenizer, messages[: msg_idx + 1], add_generation_prompt=False)
            for idx in range(min(len(start_ids), len(labels)), min(len(end_ids), len(labels))):
                labels[idx] = input_ids[idx]

    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
        attention_mask = attention_mask[-max_length:]
        labels = labels[-max_length:]
    return input_ids, attention_mask, labels


def pad_to_multiple(values: list[int], multiple: int, pad: int) -> list[int]:
    remainder = len(values) % multiple
    if remainder == 0:
        return values
    return values + [pad] * (multiple - remainder)


def build_fixed_nll_batches(
    tokenizer,
    rows: list[dict[str, Any]],
    *,
    max_length: int,
    block_size: int,
    seed: int,
    mask_id: int,
) -> list[dict[str, Any]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    batches = []
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for idx, row in enumerate(rows):
        input_ids, attention_mask, labels = encode_labeled_row(tokenizer, row, max_length=max_length)
        input_ids = pad_to_multiple(input_ids, block_size, pad_id)
        attention_mask = pad_to_multiple(attention_mask, block_size, 0)
        labels = pad_to_multiple(labels, block_size, IGNORE_INDEX)
        input_t = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        labels_t = torch.tensor(labels, dtype=torch.long).unsqueeze(0)
        label_valid = labels_t != IGNORE_INDEX
        block_count = input_t.shape[1] // block_size
        t = torch.rand((1, block_count), generator=generator)
        p_mask = ((1 - 1e-3) * t + 1e-3).unsqueeze(-1).expand(1, block_count, block_size)
        mask0 = (torch.rand((1, block_count, block_size), generator=generator) < p_mask).reshape_as(input_t)
        mask0 = mask0 & label_valid
        mask1 = (~mask0) & label_valid
        noisy0 = torch.where(mask0, torch.full_like(input_t, mask_id), input_t)
        noisy1 = torch.where(mask1, torch.full_like(input_t, mask_id), input_t)
        loss_mask0 = mask0[:, 1:] & (labels_t[:, 1:] != IGNORE_INDEX)
        loss_mask1 = mask1[:, 1:] & (labels_t[:, 1:] != IGNORE_INDEX)
        token_count = int(loss_mask0.sum().item() + loss_mask1.sum().item())
        if token_count == 0:
            continue
        batches.append(
            {
                "idx": idx,
                "id": row.get("id", str(idx)),
                "task": row.get("task", "all"),
                "input_ids": torch.cat([noisy0, noisy1], dim=0),
                "labels": labels_t.expand(2, -1).clone(),
                "loss_mask": torch.cat([loss_mask0, loss_mask1], dim=0),
                "token_count": token_count,
                "view0_tokens": int(loss_mask0.sum().item()),
                "view1_tokens": int(loss_mask1.sum().item()),
            }
        )
    return batches


class GpuMonitor:
    def __init__(self, gpu_index: int = 0, interval: float = 1.0):
        self.gpu_index = gpu_index
        self.interval = interval
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self):
        cmd = [
            "nvidia-smi",
            f"--id={self.gpu_index}",
            "--query-gpu=timestamp,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if proc.returncode == 0 and proc.stdout.strip():
                    parts = [part.strip() for part in proc.stdout.strip().split(",")]
                    if len(parts) >= 3:
                        self.samples.append(
                            {
                                "timestamp": parts[0],
                                "memory_mib": int(float(parts[1])),
                                "util_pct": int(float(parts[2])),
                            }
                        )
            except Exception:
                pass
            self._stop.wait(self.interval)

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {"samples": 0, "gpu_peak_memory_mib": None, "gpu_util_mean_pct": None, "gpu_util_max_pct": None}
        utils = [sample["util_pct"] for sample in self.samples]
        mem = [sample["memory_mib"] for sample in self.samples]
        return {
            "samples": len(self.samples),
            "gpu_peak_memory_mib": max(mem),
            "gpu_util_mean_pct": sum(utils) / len(utils),
            "gpu_util_max_pct": max(utils),
        }


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_model_and_tokenizer(model_path: Path, adapter_path: Path | None, four_bit: bool):
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
    repo_v2 = ROOT / "fast-dllm/v2"
    sys.path.insert(0, str(repo_v2))
    import generation_functions  # noqa: PLC0415

    model.mdm_sample = types.MethodType(generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample, model)
    model.eval()
    tokenizer_path = (
        adapter_path
        if adapter_path is not None and (adapter_path / "tokenizer.json").exists()
        else model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def resolve_mask_id(config, tokenizer, requested: int | None) -> int:
    if requested is not None:
        return int(requested)
    value = getattr(config, "mask_token_id", None)
    if value is not None:
        return int(value)
    token_ids = tokenizer("|<MASK>|", add_special_tokens=False).input_ids
    if token_ids:
        return int(token_ids[0])
    return int(MASK_ID)


def resolve_stop_token_ids(config, tokenizer, requested: int | None = None) -> list[int]:
    ids: list[int] = []

    def add(value):
        if value is None:
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                add(item)
            return
        item = int(value)
        if item not in ids:
            ids.append(item)

    add(requested)
    add(tokenizer.eos_token_id)
    add(getattr(config, "eos_token_id", None))
    for text in ("<|im_end|>", "<|im_start|>"):
        token_ids = tokenizer(text, add_special_tokens=False).input_ids
        if len(token_ids) == 1:
            add(token_ids[0])
    if not ids:
        add(STOP_TOKEN_ID)
    return ids


def get_model_config(model):
    return getattr(model, "config", None) or getattr(getattr(model, "base_model", None), "config", None)


def set_block_size(model, block_size: int) -> None:
    for module in model.modules():
        if hasattr(module, "bd_size"):
            module.bd_size = block_size
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "bd_size"):
            config.bd_size = block_size


def evaluate_fixed_denoising_nll(model, batches: list[dict[str, Any]]) -> dict[str, Any]:
    totals = defaultdict(lambda: {"loss_sum": 0.0, "tokens": 0, "examples": 0})
    rows = []
    vocab_size = int(get_model_config(model).vocab_size)
    with torch.inference_mode():
        for batch in batches:
            input_ids = batch["input_ids"].to("cuda", non_blocking=True)
            labels = batch["labels"].to("cuda", non_blocking=True)
            loss_mask = batch["loss_mask"].to("cuda", non_blocking=True)
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, :-1].contiguous().float()
            targets = labels[:, 1:].contiguous()
            masked_labels = torch.where(loss_mask, targets, torch.full_like(targets, IGNORE_INDEX))
            loss_sum_t = F.cross_entropy(
                logits.view(-1, vocab_size),
                masked_labels.view(-1),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
            token_count = int(loss_mask.sum().item())
            loss_sum = float(loss_sum_t.item())
            nll = loss_sum / max(token_count, 1)
            row = {
                "idx": batch["idx"],
                "id": batch["id"],
                "task": batch["task"],
                "nll": nll,
                "loss_sum": loss_sum,
                "tokens": token_count,
                "view0_tokens": batch["view0_tokens"],
                "view1_tokens": batch["view1_tokens"],
            }
            rows.append(row)
            for key in ("all", batch["task"]):
                totals[key]["loss_sum"] += loss_sum
                totals[key]["tokens"] += token_count
                totals[key]["examples"] += 1
    metrics = {}
    for key, value in sorted(totals.items()):
        nll = value["loss_sum"] / max(value["tokens"], 1)
        metrics[key] = {
            "nll": nll,
            "ppl": float(torch.exp(torch.tensor(nll)).item()),
            "tokens": value["tokens"],
            "examples": value["examples"],
        }
    return {"metrics": metrics, "per_example": rows}


def normalize_number(text: Any) -> str | None:
    if text is None:
        return None
    matches = re.findall(r"[-+]?\$?\d[\d,]*(?:\.\d+)?", str(text))
    if not matches:
        return None
    return matches[-1].replace(",", "").replace("$", "").rstrip(".").lower()


def gsm8k_gold(answer: str) -> str | None:
    if "####" in answer:
        return normalize_number(answer.split("####", 1)[1])
    return normalize_number(answer)


def gsm8k_strict(text: str) -> str | None:
    match = re.search(r"####\s*([-+]?\$?\d[\d,]*(?:\.\d+)?)", text)
    return normalize_number(match.group(1)) if match else None


def build_gsm8k_prompt(tokenizer, row: dict[str, Any], fewshot_rows: list[dict[str, Any]]):
    messages: list[dict[str, str]] = []
    for shot in fewshot_rows:
        messages.append({"role": "user", "content": f"Question: {shot['question']}\nAnswer:"})
        messages.append({"role": "assistant", "content": shot["answer"]})
    messages.append({"role": "user", "content": f"Question: {row['question']}\nAnswer:"})
    return apply_chat_template(tokenizer, messages, tokenize=False, add_generation_prompt=True)


def build_mbpp_prompt(tokenizer, row: dict[str, Any]):
    tests = "\n".join(row.get("test_list") or [])
    messages = [
        {
            "role": "system",
            "content": "You are a helpful coding assistant.",
        },
        {
            "role": "user",
            "content": (
                "Write a Python function for this task. Return code only.\n\n"
                f"Task: {row['text']}\n\nTests:\n{tests}"
            ),
        },
    ]
    return apply_chat_template(tokenizer, messages, tokenize=False, add_generation_prompt=True)


def strip_think(text: str) -> str:
    return re.sub(r"(?is)<think>.*?</think>", "", text).strip()


def extract_code(text: str) -> str:
    text = strip_think(text)
    fences = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.I | re.S)
    if fences:
        return fences[0].strip()
    if "[DONE]" in text:
        text = text.split("[DONE]", 1)[0]
    return text.strip()


def score_mbpp(code: str, row: dict[str, Any], timeout: float) -> dict[str, Any]:
    tests = "\n".join(row.get("test_list") or [])
    setup = row.get("test_setup_code") or ""
    script = f"{code}\n\n{setup}\n\n{tests}\n"
    with tempfile.TemporaryDirectory() as tmp:
        script_path = Path(tmp) / "candidate.py"
        script_path.write_text(script, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "PYTHONNOUSERSITE": "1"},
            )
        except subprocess.TimeoutExpired as exc:
            return {"passed": False, "error": "timeout", "stdout": exc.stdout, "stderr": exc.stderr}
    return {
        "passed": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-2000:],
        "stderr": proc.stderr[-2000:],
    }


def sample_with_top_p(logits: torch.Tensor, top_p: float, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    if temperature <= 0:
        probs = torch.softmax(logits, dim=-1)
        return probs.argmax(dim=-1), probs
    probs = torch.softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    indices_to_remove = torch.zeros_like(probs, dtype=torch.bool).scatter_(
        dim=-1,
        index=sorted_indices,
        src=sorted_indices_to_remove,
    )
    probs = probs.masked_fill(indices_to_remove, 0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    sampled = torch.multinomial(probs.reshape(-1, probs.shape[-1]), num_samples=1)
    return sampled.reshape(probs.shape[:-1]), probs


def ban_mask_token_logits(logits: torch.Tensor, mask_id: int) -> torch.Tensor:
    logits = logits.clone()
    logits[..., int(mask_id)] = torch.finfo(logits.dtype).min
    return logits


def full_context_sample_one(model, input_ids: torch.Tensor, args) -> tuple[torch.Tensor, dict[str, Any]]:
    output_ids = input_ids.unsqueeze(0).to("cuda")
    original_len = int(output_ids.shape[1])
    metrics = {
        "blocks": [],
        "denoise_forwards": 0,
        "selected_mask_tokens": 0,
        "natural_commits": 0,
        "forced_commits": 0,
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

    while output_ids.shape[1] - original_len < args.max_new_tokens:
        remaining = args.max_new_tokens - (output_ids.shape[1] - original_len)
        if args.fresh_generation_blocks:
            block_pad = args.block_size
        else:
            block_pad = args.block_size - (output_ids.shape[1] % args.block_size)
            if block_pad == 0:
                block_pad = args.block_size
        block_pad = min(block_pad, remaining)
        masks = torch.full(
            (output_ids.shape[0], block_pad),
            args.mask_id,
            dtype=torch.long,
            device=output_ids.device,
        )
        x_t = torch.cat([output_ids, masks], dim=1)
        block_metrics = {
            "block_pad": int(block_pad),
            "initial_masks": int((x_t[:, -block_pad:] == args.mask_id).sum().item()),
            "denoise_steps": 0,
            "selected_mask_tokens": 0,
        }
        while bool((x_t[:, -block_pad:] == args.mask_id).any().item()):
            window_len = min(args.block_size, x_t.shape[1])
            num_small_blocks = (window_len + args.small_block_size - 1) // args.small_block_size
            for small_block_idx in range(num_small_blocks):
                start = small_block_idx * args.small_block_size
                end = min(start + args.small_block_size, window_len)
                while True:
                    mask_idx = x_t[:, -window_len:] == args.mask_id
                    current_mask = mask_idx[:, start:end]
                    if not bool(current_mask.any().item()):
                        break
                    output = model(input_ids=x_t, use_cache=False)
                    logits = torch.cat([output.logits[:, :1, :], output.logits[:, :-1, :]], dim=1)
                    logits = logits[:, -window_len:][:, start:end]
                    logits = ban_mask_token_logits(logits, args.mask_id)
                    x_1, p_1t = sample_with_top_p(logits, args.top_p, args.temperature)
                    x1_p = torch.squeeze(torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1)
                    active_probs = torch.where(current_mask, x1_p, torch.full_like(x1_p, -torch.inf))
                    natural = active_probs > args.threshold
                    max_prob_idx = active_probs.argmax(dim=-1)
                    unmask_idx = natural.clone()
                    unmask_idx[torch.arange(x_1.shape[0], device=x_1.device), max_prob_idx] = True
                    unmask_idx = unmask_idx & current_mask
                    selected_mask = int(((x_1 == args.mask_id) & unmask_idx).sum().item())
                    natural_count = int(natural.sum().item())
                    committed = int(unmask_idx.sum().item())
                    window = x_t[:, -window_len:]
                    span = window[:, start:end].clone()
                    span[unmask_idx] = x_1[unmask_idx]
                    window[:, start:end] = span
                    x_t[:, -window_len:] = window
                    block_metrics["denoise_steps"] += 1
                    block_metrics["selected_mask_tokens"] += selected_mask
                    metrics["denoise_forwards"] += 1
                    metrics["selected_mask_tokens"] += selected_mask
                    metrics["natural_commits"] += natural_count
                    metrics["forced_commits"] += max(0, committed - natural_count)
                    stopped = truncate_if_stopped(x_t)
                    if stopped is not None:
                        output_ids = stopped
                        metrics["blocks"].append(block_metrics)
                        return output_ids[0].detach().cpu(), metrics
            if bool((x_t[:, -block_pad:] == args.mask_id).all().item()):
                break
        output_ids = x_t
        metrics["blocks"].append(block_metrics)
        stopped = truncate_if_stopped(output_ids)
        if stopped is not None:
            output_ids = stopped
            break
    return output_ids[0].detach().cpu(), metrics


def sample_batch(model, tokenizer, prompt_ids: list[torch.Tensor], args) -> tuple[dict[int, torch.Tensor], float]:
    if args.full_context_generation:
        sync_cuda()
        start = time.perf_counter()
        generated = {}
        sampler_metrics = []
        with torch.inference_mode():
            for idx, ids in enumerate(prompt_ids):
                output_ids, metrics = full_context_sample_one(model, ids, args)
                generated[idx] = output_ids
                sampler_metrics.append(metrics)
        sync_cuda()
        args._last_generation_sampler_metrics = sampler_metrics
        return generated, time.perf_counter() - start

    seq_lens = [int(ids.numel()) for ids in prompt_ids]
    max_len = max(seq_lens)
    padded = []
    for ids in prompt_ids:
        if ids.numel() < max_len:
            pad = torch.full((max_len - ids.numel(),), args.mask_id, dtype=torch.long)
            ids = torch.cat([ids, pad], dim=0)
        padded.append(ids.unsqueeze(0))
    input_ids = torch.cat(padded, dim=0).to("cuda")
    seq_len = torch.tensor(seq_lens, device="cuda")
    sync_cuda()
    start = time.perf_counter()
    with torch.inference_mode():
        generated = model.mdm_sample(
            input_ids,
            tokenizer=tokenizer,
            block_size=args.block_size,
            small_block_size=args.small_block_size,
            max_new_tokens=args.max_new_tokens,
            mask_id=args.mask_id,
            stop_token=args.stop_token_id,
            min_len=min(seq_lens),
            seq_len=seq_len,
            threshold=args.threshold,
            temperature=args.temperature,
            top_p=args.top_p,
            fresh_generation_blocks=args.fresh_generation_blocks,
        )
    sync_cuda()
    return generated, time.perf_counter() - start


def evaluate_generation(model, tokenizer, args) -> dict[str, Any]:
    generation_tasks = {
        task.strip().lower()
        for task in str(args.generation_tasks).replace(",", " ").split()
        if task.strip()
    }
    if not generation_tasks:
        generation_tasks = {"gsm8k", "mbpp"}
    unknown_tasks = generation_tasks - {"gsm8k", "mbpp"}
    if unknown_tasks:
        raise ValueError(f"unknown generation task(s): {sorted(unknown_tasks)}")

    items: list[dict[str, Any]] = []
    if "gsm8k" in generation_tasks:
        gsm_rows = read_jsonl(Path(args.gsm8k_path), args.generation_limit)
        fewshot_rows = read_jsonl(Path(args.gsm8k_fewshot_path), args.gsm8k_fewshot)
        for row in gsm_rows:
            prompt = build_gsm8k_prompt(tokenizer, row, fewshot_rows)
            input_ids = tokenizer([prompt], return_tensors="pt").input_ids[0].cpu()
            items.append({"task": "gsm8k", "row": row, "prompt_ids": input_ids})
    if "mbpp" in generation_tasks:
        mbpp_rows = read_jsonl(Path(args.mbpp_path), args.generation_limit)
        for row in mbpp_rows:
            prompt = build_mbpp_prompt(tokenizer, row)
            input_ids = tokenizer([prompt], return_tensors="pt").input_ids[0].cpu()
            items.append({"task": "mbpp", "row": row, "prompt_ids": input_ids})

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    generated_tokens = 0
    unresolved_examples = 0

    for batch_start in range(0, len(items), args.generation_batch_size):
        batch = items[batch_start : batch_start + args.generation_batch_size]
        generated, seconds = sample_batch(model, tokenizer, [item["prompt_ids"] for item in batch], args)
        per_example_seconds = seconds / max(len(batch), 1)
        print(
            f"[generation-batch] start={batch_start} size={len(batch)} seconds={seconds:.3f}",
            flush=True,
        )
        sampler_metrics = getattr(args, "_last_generation_sampler_metrics", [{} for _ in batch])
        for batch_pos, item in enumerate(batch):
            row = item["row"]
            prompt_len = int(item["prompt_ids"].numel())
            output_ids = generated[batch_pos]
            new_ids = output_ids[prompt_len:]
            mask_count = int((new_ids == args.mask_id).sum().item())
            text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            nonmask_tokens = int((new_ids != args.mask_id).sum().item())
            generated_tokens += nonmask_tokens
            unresolved_examples += int(mask_count > 0)
            if item["task"] == "gsm8k":
                strict_pred = gsm8k_strict(text)
                flex_pred = normalize_number(text)
                gold = gsm8k_gold(row["answer"])
                rows.append(
                    {
                        "task": "gsm8k",
                        "idx": row.get("idx"),
                        "id": f"gsm8k-{row.get('idx')}",
                        "gold": gold,
                        "strict_pred": strict_pred,
                        "flex_pred": flex_pred,
                        "strict_correct": strict_pred == gold,
                        "flex_correct": flex_pred == gold,
                        "mask_count": mask_count,
                        "nonmask_generated_tokens": nonmask_tokens,
                        "seconds": per_example_seconds,
                        "sampler": sampler_metrics[batch_pos] if batch_pos < len(sampler_metrics) else {},
                        "generated": text,
                    }
                )
            else:
                code = extract_code(text)
                score = score_mbpp(code, row, args.mbpp_timeout)
                rows.append(
                    {
                        "task": "mbpp",
                        "idx": row.get("idx"),
                        "id": f"mbpp-{row.get('task_id')}",
                        "task_id": row.get("task_id"),
                        "passed": bool(score["passed"]),
                        "mask_count": mask_count,
                        "nonmask_generated_tokens": nonmask_tokens,
                        "seconds": per_example_seconds,
                        "sampler": sampler_metrics[batch_pos] if batch_pos < len(sampler_metrics) else {},
                        "generated": text,
                        "extracted_code": code,
                        "score": score,
                    }
                )

    elapsed = time.perf_counter() - started
    gsm = [row for row in rows if row["task"] == "gsm8k"]
    mbpp = [row for row in rows if row["task"] == "mbpp"]
    summary = {
        "gsm8k": {
            "examples": len(gsm),
            "strict_correct": sum(int(row["strict_correct"]) for row in gsm),
            "strict_accuracy": sum(int(row["strict_correct"]) for row in gsm) / len(gsm) if gsm else 0.0,
            "flex_correct": sum(int(row["flex_correct"]) for row in gsm),
            "flex_accuracy": sum(int(row["flex_correct"]) for row in gsm) / len(gsm) if gsm else 0.0,
        },
        "mbpp": {
            "examples": len(mbpp),
            "passed": sum(int(row["passed"]) for row in mbpp),
            "pass_at_1": sum(int(row["passed"]) for row in mbpp) / len(mbpp) if mbpp else 0.0,
        },
        "elapsed_seconds": elapsed,
        "generated_tokens": generated_tokens,
        "generated_tokens_per_second": generated_tokens / elapsed if elapsed else 0.0,
        "unresolved_mask_examples": unresolved_examples,
    }
    return {"summary": summary, "per_example": rows}


def adapter_ready(path: Path | None) -> bool:
    if path is None:
        return True
    return (path / "adapter_model.safetensors").exists() and (path / "adapter_config.json").exists()


def model_specs(args) -> list[dict[str, Any]]:
    specs = [
        {"name": "init", "adapter": None},
        {"name": "A_diffusion_only", "adapter": Path(args.adapter_a)},
        {"name": "B_two_stream", "adapter": Path(args.adapter_b)},
    ]
    requested = {
        item.strip()
        for item in str(args.model_names).replace(",", " ").split()
        if item.strip()
    }
    if not requested:
        return specs
    known = {spec["name"] for spec in specs}
    unknown = requested - known
    if unknown:
        raise ValueError(f"unknown model name(s): {sorted(unknown)}; choices={sorted(known)}")
    return [spec for spec in specs if spec["name"] in requested]


def load_ar_reference(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        candidates = sorted(
            (ROOT / "runs/phaseA_retention_snapshot_n5_t384/ar_qwen35_9b_bf16_thinkingoff").glob(
                "*/results_*.json"
            ),
            key=lambda item: item.stat().st_mtime,
        )
        path = candidates[-1] if candidates else None
    if path is None or not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results", {})
    return {
        "path": str(path),
        "gsm8k_flex": results.get("phaseA_gsm8k_first20", {}).get("exact_match,flexible-extract"),
        "gsm8k_strict": results.get("phaseA_gsm8k_first20", {}).get("exact_match,strict-match"),
        "mbpp_pass_at_1": results.get("phaseA_mbpp_first20", {}).get("pass_at_1,none"),
    }


def del_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=str(DEFAULT_BASE))
    parser.add_argument("--adapter-a", default=str(DEFAULT_A))
    parser.add_argument("--adapter-b", default=str(DEFAULT_B))
    parser.add_argument(
        "--model-names",
        default="",
        help="Comma/space separated subset of init, A_diffusion_only, B_two_stream. Empty runs all.",
    )
    parser.add_argument("--train-path", default="data/flare_stage1_ab_pilot_train/train_agentic_mix.json")
    parser.add_argument("--heldout-nll", default="data/flare_stage1_ab_pilot/heldout_nll.jsonl")
    parser.add_argument("--gsm8k-path", default="data/phaseA_retention/gsm8k_main_test_first20.jsonl")
    parser.add_argument("--gsm8k-fewshot-path", default="data/phaseA_retention/gsm8k_main_train_first5.jsonl")
    parser.add_argument("--mbpp-path", default="data/phaseA_retention/mbpp_full_test_first20.jsonl")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--nll-block-size", type=int, default=32)
    parser.add_argument("--nll-seed", type=int, default=20260702)
    parser.add_argument("--generation-limit", type=int, default=20)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument(
        "--generation-tasks",
        default="gsm8k,mbpp",
        help="Comma/space separated generation tasks to run: gsm8k, mbpp.",
    )
    parser.add_argument(
        "--active-block-generation",
        action="store_false",
        dest="full_context_generation",
        help="Use the original mdm_sample active-block generation path instead of full-context recompute.",
    )
    parser.add_argument(
        "--tail-fill-generation",
        action="store_false",
        dest="fresh_generation_blocks",
        help="Compatibility mode: fill only the prompt-tail remainder before full generated blocks.",
    )
    parser.set_defaults(fresh_generation_blocks=True)
    parser.set_defaults(full_context_generation=True)
    parser.add_argument("--gsm8k-fewshot", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--small-block-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--mask-id", type=int, default=None)
    parser.add_argument("--stop-token-id", type=int, default=None)
    parser.add_argument("--mbpp-timeout", type=float, default=5.0)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-nll", action="store_true")
    parser.add_argument("--ar-reference-json", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_cuda_env()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    disjointness = verify_disjoint(Path(args.train_path), Path(args.heldout_nll))
    write_json(out_dir / "disjointness.json", disjointness)
    print("[disjointness] " + json.dumps(disjointness, sort_keys=True), flush=True)

    base_model = Path(args.base_model)
    specs = model_specs(args)
    for spec in specs:
        if not adapter_ready(spec["adapter"]):
            raise FileNotFoundError(f"Adapter missing/corrupt for {spec['name']}: {spec['adapter']}")

    from transformers import AutoTokenizer
    from transformers import AutoConfig

    base_tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    base_config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    args.mask_id = resolve_mask_id(base_config, base_tokenizer, args.mask_id)
    args.stop_token_ids = resolve_stop_token_ids(base_config, base_tokenizer, args.stop_token_id)
    args.stop_token_id = int(args.stop_token_ids[0])
    print(
        "[token_ids] "
        + json.dumps({"mask_id": args.mask_id, "stop_token_ids": args.stop_token_ids}, sort_keys=True),
        flush=True,
    )
    heldout_rows = read_jsonl(Path(args.heldout_nll))
    nll_batches = build_fixed_nll_batches(
        base_tokenizer,
        heldout_rows,
        max_length=args.max_length,
        block_size=args.nll_block_size,
        seed=args.nll_seed,
        mask_id=args.mask_id,
    )
    fixed_mask_manifest = {
        "heldout_rows": len(heldout_rows),
        "nll_batches": len(nll_batches),
        "total_eval_tokens": sum(batch["token_count"] for batch in nll_batches),
        "view0_tokens": sum(batch["view0_tokens"] for batch in nll_batches),
        "view1_tokens": sum(batch["view1_tokens"] for batch in nll_batches),
        "nll_seed": args.nll_seed,
        "nll_block_size": args.nll_block_size,
        "max_length": args.max_length,
    }
    write_json(out_dir / "fixed_mask_manifest.json", fixed_mask_manifest)
    print("[fixed_masks] " + json.dumps(fixed_mask_manifest, sort_keys=True), flush=True)

    all_results: dict[str, Any] = {
        "args": vars(args),
        "disjointness": disjointness,
        "fixed_mask_manifest": fixed_mask_manifest,
        "ar_reference": load_ar_reference(Path(args.ar_reference_json) if args.ar_reference_json else None),
        "models": {},
    }

    for spec in specs:
        name = spec["name"]
        adapter = spec["adapter"]
        print(f"[load] {name} adapter={adapter}", flush=True)
        model, tokenizer = load_model_and_tokenizer(base_model, adapter, four_bit=not args.no_4bit)
        set_block_size(model, args.block_size)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        model_result: dict[str, Any] = {"adapter": str(adapter) if adapter else None}
        if not args.skip_nll:
            print(f"[nll] {name}", flush=True)
            sync_cuda()
            start = time.perf_counter()
            with GpuMonitor(args.gpu_index) as monitor:
                nll_result = evaluate_fixed_denoising_nll(model, nll_batches)
                sync_cuda()
            nll_result["seconds"] = time.perf_counter() - start
            nll_result["gpu"] = monitor.summary()
            nll_result["cuda_peak_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
            nll_result["cuda_peak_reserved_gb"] = torch.cuda.max_memory_reserved() / (1024**3)
            write_json(out_dir / f"{name}_denoising_nll.json", nll_result)
            model_result["denoising_nll"] = nll_result
            print(f"[nll] {name} all_nll={nll_result['metrics']['all']['nll']:.6f}", flush=True)

        if not args.skip_generation:
            print(f"[generation] {name}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            sync_cuda()
            with GpuMonitor(args.gpu_index) as monitor:
                generation_result = evaluate_generation(model, tokenizer, args)
                sync_cuda()
            generation_result["gpu"] = monitor.summary()
            generation_result["cuda_peak_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
            generation_result["cuda_peak_reserved_gb"] = torch.cuda.max_memory_reserved() / (1024**3)
            write_json(out_dir / f"{name}_generation.json", generation_result)
            write_jsonl(out_dir / f"{name}_generation.jsonl", generation_result["per_example"])
            model_result["generation"] = generation_result
            print(
                "[generation] "
                f"{name} gsm8k_flex={generation_result['summary']['gsm8k']['flex_accuracy']:.4f} "
                f"mbpp={generation_result['summary']['mbpp']['pass_at_1']:.4f}",
                flush=True,
            )

        all_results["models"][name] = model_result
        write_json(out_dir / "summary.json", all_results)
        del_model(model)

    write_json(out_dir / "summary.json", all_results)
    print(json.dumps(all_results, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
