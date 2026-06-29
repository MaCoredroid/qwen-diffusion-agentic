#!/usr/bin/env python3
"""Evaluate clean AR-style assistant-continuation NLL for the FLARE A/B pilot."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


def configure_cuda_env(root: Path):
    venv_root = Path(sys.executable).resolve().parents[1]
    cuda_root = venv_root / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "nvidia" / "cu13"
    if not cuda_root.exists():
        cuda_root = root / ".venv-fastdllm" / "lib" / "python3.10" / "site-packages" / "nvidia" / "cu13"
    if cuda_root.exists():
        os.environ.setdefault("CUDA_HOME", str(cuda_root))
        os.environ["PATH"] = f"{cuda_root / 'bin'}:{os.environ.get('PATH', '')}"
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_root / 'lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.pop("FASTDLLM_FLARE_TWO_STREAM", None)
    os.environ.pop("FLARE_TWO_STREAM", None)


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def apply_chat_template(tokenizer, messages, **kwargs):
    kwargs = dict(kwargs)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def template_encode(tokenizer, messages, add_generation_prompt: bool):
    encoded = apply_chat_template(
        tokenizer,
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=True,
    )
    if isinstance(encoded, str):
        tokenized = tokenizer(encoded, add_special_tokens=False)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized.get("attention_mask", [1] * len(input_ids))
        assistant_mask = None
    else:
        try:
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask", [1] * len(input_ids))
            assistant_mask = encoded.get("assistant_masks") or encoded.get("assistant_tokens_mask")
        except (TypeError, KeyError):
            input_ids = encoded
            attention_mask = [1] * len(input_ids)
            assistant_mask = None
    return list(input_ids), list(attention_mask), assistant_mask


def encode_row(tokenizer, row, max_length: int):
    messages = []
    if row.get("system"):
        messages.append({"role": "system", "content": row["system"]})
    messages.extend(row["messages"])
    input_ids, attention_mask, assistant_mask = template_encode(
        tokenizer,
        messages,
        add_generation_prompt=False,
    )

    labels = [-100] * len(input_ids)
    if assistant_mask is not None and any(assistant_mask):
        for idx, is_assistant in enumerate(assistant_mask):
            if idx < len(labels) and is_assistant:
                labels[idx] = input_ids[idx]
    else:
        for msg_idx, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            start_ids, _, _ = template_encode(
                tokenizer,
                messages[:msg_idx],
                add_generation_prompt=True,
            )
            end_ids, _, _ = template_encode(
                tokenizer,
                messages[: msg_idx + 1],
                add_generation_prompt=False,
            )
            start = min(len(start_ids), len(labels))
            end = min(len(end_ids), len(labels))
            for idx in range(start, end):
                labels[idx] = input_ids[idx]

    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
        attention_mask = attention_mask[-max_length:]
        labels = labels[-max_length:]

    input_ids_t = torch.tensor([input_ids], dtype=torch.long)
    attention_mask_t = torch.tensor([attention_mask], dtype=torch.long)
    labels_t = torch.tensor([labels], dtype=torch.long)
    return input_ids_t, attention_mask_t, labels_t


def load_model_and_tokenizer(model_path: str, adapter_path: str | None, four_bit: bool):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    kwargs = {
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
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    tokenizer_path = adapter_path or model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def evaluate(model, tokenizer, rows, max_length: int):
    totals = defaultdict(lambda: {"loss_sum": 0.0, "tokens": 0, "examples": 0})
    with torch.no_grad():
        for row in rows:
            input_ids, attention_mask, labels = encode_row(tokenizer, row, max_length)
            input_ids = input_ids.to("cuda")
            attention_mask = attention_mask.to("cuda")
            labels = labels.to("cuda")
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits[:, :-1].contiguous().float()
            shift_labels = labels[:, 1:].contiguous()
            valid = shift_labels != -100
            token_count = int(valid.sum().item())
            if token_count == 0:
                continue
            loss_sum = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            task = row.get("task", "all")
            for key in ("all", task):
                totals[key]["loss_sum"] += float(loss_sum.item())
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
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--heldout", default="data/flare_stage1_ab_pilot/heldout_nll.jsonl")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--no-4bit", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    configure_cuda_env(root)
    rows = load_rows(Path(args.heldout))
    model, tokenizer = load_model_and_tokenizer(args.model, args.adapter, four_bit=not args.no_4bit)
    torch.cuda.reset_peak_memory_stats()
    metrics = evaluate(model, tokenizer, rows, args.max_length)
    result = {
        "model": args.model,
        "adapter": args.adapter,
        "heldout": args.heldout,
        "max_length": args.max_length,
        "metrics": metrics,
        "cuda_peak_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
        "cuda_peak_reserved_gb": torch.cuda.max_memory_reserved() / (1024**3),
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_json).open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
