#!/usr/bin/env python3
"""Measure real-9B cached sampler quality/speed on a tiny GSM8K slice."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_qwen35_state_cache_sampler import cached_full_context_sample  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/qwen3.5-9b-fastdllm-init")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--docs", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--small-block-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--trace-docs", type=int, default=3)
    return parser.parse_args()


def read_jsonl(path: Path, limit: int | None = None):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def apply_qwen_chat_template(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def build_gsm8k_prompt(tokenizer, train_rows, doc):
    messages = []
    for row in train_rows:
        messages.append({"role": "user", "content": f"Question: {row['question']}\nAnswer:"})
        messages.append({"role": "assistant", "content": row["answer"]})
    messages.append({"role": "user", "content": f"Question: {doc['question']}\nAnswer:"})
    return apply_qwen_chat_template(tokenizer, messages)


def normalize_number(raw):
    if raw is None:
        return None
    text = str(raw).strip()
    text = text.replace(",", "").replace("$", "")
    text = re.sub(r"\.$", "", text)
    return text.lower()


def target_answer(answer):
    if "####" in answer:
        return normalize_number(answer.split("####", 1)[1])
    return normalize_number(answer)


def strict_extract(text):
    match = re.search(r"#### (\-?[0-9\.\,]+)", text)
    return normalize_number(match.group(1)) if match else None


def flexible_extract(text):
    matches = re.findall(r"-?\$?[0-9][0-9\.,]*", text)
    return normalize_number(matches[-1]) if matches else None


def score(pred_text, gold_answer):
    target = target_answer(gold_answer)
    strict = strict_extract(pred_text)
    flex = flexible_extract(pred_text)
    return {
        "target": target,
        "strict_pred": strict,
        "flex_pred": flex,
        "strict_correct": strict == target,
        "flex_correct": flex == target,
    }


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def timed(device, fn, *args, **kwargs):
    sync(device)
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    sync(device)
    return out, time.perf_counter() - start


def make_sampler_args(model, tokenizer, cli_args):
    mask_id = getattr(model.config, "mask_token_id", None)
    if mask_id is None:
        mask_id = tokenizer.convert_tokens_to_ids("|<MASK>|")
    stop_token_id = getattr(model.config, "eos_token_id", None) or tokenizer.eos_token_id
    if isinstance(stop_token_id, (list, tuple)):
        stop_token_id = stop_token_id[0]
    return SimpleNamespace(
        max_new_tokens=cli_args.max_new_tokens,
        block_size=cli_args.block_size,
        small_block_size=cli_args.small_block_size,
        mask_id=int(mask_id),
        stop_token_id=int(stop_token_id),
        threshold=cli_args.threshold,
        temperature=cli_args.temperature,
        top_p=cli_args.top_p,
        _last_sampler_schedule_events={},
        guard_tool_json_prefix=False,
        json_prefix_guard_kinds=set(),
    )


def decode_new(tokenizer, generated_ids, prompt_len):
    return tokenizer.decode(generated_ids[prompt_len:], skip_special_tokens=True).strip()


def main() -> int:
    args = parse_args()
    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    model_dir = (ROOT / args.model_dir).resolve()
    train_rows = read_jsonl(ROOT / "data/phaseA_retention/gsm8k_main_train_first5.jsonl")
    docs = read_jsonl(ROOT / "data/phaseA_retention/gsm8k_main_test_first20.jsonl", args.docs)

    print("Real-9B cache eval-equivalence measurement: GSM8K tiny slice")
    print(f"model_dir={model_dir}")
    print(
        f"device={device} docs={len(docs)} max_new_tokens={args.max_new_tokens} "
        f"block_size={args.block_size} small_block_size={args.small_block_size} "
        f"threshold={args.threshold} temperature={args.temperature}"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": str(device)},
    ).eval()
    modeling_module = importlib.import_module(type(model).__module__)
    sampler_args = make_sampler_args(model, tokenizer, args)
    toolcall_module = importlib.import_module("eval_fastdllm_toolcall_cases")

    rows = []
    drift_values = []
    with torch.inference_mode():
        for idx, doc in enumerate(docs):
            prompt = build_gsm8k_prompt(tokenizer, train_rows, doc)
            input_ids = tokenizer([prompt], return_tensors="pt").input_ids.to(device)
            print(f"\ncase={idx} prompt_tokens={input_ids.shape[1]}")

            golden_ids, golden_seconds = timed(
                device,
                toolcall_module.full_context_sample,
                model,
                input_ids,
                tokenizer,
                sampler_args,
                None,
            )
            golden_text = decode_new(tokenizer, golden_ids, input_ids.shape[1])
            golden_score = score(golden_text, doc["answer"])
            torch.cuda.empty_cache()

            cached_ids, cached_seconds = timed(
                device,
                cached_full_context_sample,
                model,
                input_ids,
                sampler_args,
                modeling_module,
            )
            cached_text = decode_new(tokenizer, cached_ids, input_ids.shape[1])
            cached_score = score(cached_text, doc["answer"])
            torch.cuda.empty_cache()

            trace = []
            if idx < args.trace_docs:
                traced_cached = cached_full_context_sample(
                    model,
                    input_ids,
                    sampler_args,
                    modeling_module,
                    trace=trace,
                )
                if not torch.equal(cached_ids, traced_cached):
                    print("warning=trace_cached_tokens_differ_from_timed_cached")
                drift_values.extend(trace)
                torch.cuda.empty_cache()

            speedup = golden_seconds / cached_seconds if cached_seconds else float("inf")
            token_exact = torch.equal(golden_ids, cached_ids)
            metric_same = (
                golden_score["strict_correct"] == cached_score["strict_correct"]
                and golden_score["flex_correct"] == cached_score["flex_correct"]
            )
            row = {
                "idx": idx,
                "prompt_tokens": int(input_ids.shape[1]),
                "golden_seconds": golden_seconds,
                "cached_seconds": cached_seconds,
                "speedup": speedup,
                "token_exact": token_exact,
                "metric_same": metric_same,
                "golden_score": golden_score,
                "cached_score": cached_score,
                "trace_count": len(trace),
                "trace_mean": sum(trace) / len(trace) if trace else None,
                "trace_max": max(trace) if trace else None,
            }
            rows.append(row)
            print(
                f"golden strict={golden_score['strict_correct']} flex={golden_score['flex_correct']} "
                f"preds=({golden_score['strict_pred']},{golden_score['flex_pred']}) "
                f"seconds={golden_seconds:.3f}"
            )
            print(
                f"cached strict={cached_score['strict_correct']} flex={cached_score['flex_correct']} "
                f"preds=({cached_score['strict_pred']},{cached_score['flex_pred']}) "
                f"seconds={cached_seconds:.3f} speedup={speedup:.2f}x "
                f"token_exact={token_exact} metric_same={metric_same}"
            )
            if trace:
                print(
                    f"logit_drift count={len(trace)} mean={sum(trace)/len(trace):.6g} "
                    f"max={max(trace):.6g}"
                )
            gc.collect()

    golden_strict = sum(1 for row in rows if row["golden_score"]["strict_correct"]) / len(rows)
    cached_strict = sum(1 for row in rows if row["cached_score"]["strict_correct"]) / len(rows)
    golden_flex = sum(1 for row in rows if row["golden_score"]["flex_correct"]) / len(rows)
    cached_flex = sum(1 for row in rows if row["cached_score"]["flex_correct"]) / len(rows)
    speedups = [row["speedup"] for row in rows]
    print("\nSUMMARY")
    print(f"golden_strict={golden_strict:.3f} cached_strict={cached_strict:.3f}")
    print(f"golden_flexible={golden_flex:.3f} cached_flexible={cached_flex:.3f}")
    print(f"metric_same_cases={sum(1 for row in rows if row['metric_same'])}/{len(rows)}")
    print(f"token_exact_cases={sum(1 for row in rows if row['token_exact'])}/{len(rows)}")
    print(
        f"speedup_mean={sum(speedups)/len(speedups):.2f}x "
        f"speedup_min={min(speedups):.2f}x speedup_max={max(speedups):.2f}x"
    )
    if drift_values:
        print(
            f"logit_drift_mean={sum(drift_values)/len(drift_values):.6g} "
            f"logit_drift_max={max(drift_values):.6g} samples={len(drift_values)}"
        )
    print("FINAL: EVAL_EQUIVALENT" if golden_flex == cached_flex else "FINAL: METRIC_MISMATCH")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
