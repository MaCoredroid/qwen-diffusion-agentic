#!/usr/bin/env python3
import argparse
import json
import re
import sys
import time
import types
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BASE = "/home/mark/qwen_diffusion/models/qwen2.5-1.5b-fastdllm-init"
DEFAULT_ADAPTER = "/home/mark/qwen_diffusion/runs/fastdllm_qwen25_1p5b_alpaca_lora_full"
DEFAULT_OUT = "/home/mark/qwen_diffusion/runs/fastdllm_qwen25_1p5b_alpaca_lora_full/gsm8k_mini.jsonl"
MASK_ID = 151665


def normalize_number(text):
    if text is None:
        return None
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def gold_answer(answer):
    if "####" in answer:
        return normalize_number(answer.split("####")[-1])
    return normalize_number(answer)


def make_prompt(tokenizer, question):
    messages = [
        {
            "role": "user",
            "content": (
                "Solve this grade-school math problem. "
                "Return only the final numeric answer, with no explanation.\n\n"
                f"{question}"
            ),
        }
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=DEFAULT_BASE)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--small-block-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    repo_v2 = Path("/home/mark/qwen_diffusion/fast-dllm/v2")
    sys.path.insert(0, str(repo_v2))
    import generation_functions

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    model.mdm_sample = types.MethodType(
        generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample,
        model,
    )
    model.to("cuda").eval()

    ds = load_dataset("openai/gsm8k", "main", split=f"test[:{args.num_examples}]")

    correct = 0
    total = 0
    total_new_tokens = 0
    unresolved_mask_examples = 0
    start = time.time()

    with out_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(ds):
            prompt = make_prompt(tokenizer, row["question"])
            input_ids = tokenizer([prompt], return_tensors="pt").input_ids.to("cuda")
            seq_len = torch.tensor([input_ids.shape[1]], device="cuda")

            sample_start = time.time()
            with torch.no_grad():
                generated = model.mdm_sample(
                    input_ids,
                    tokenizer=tokenizer,
                    block_size=args.block_size,
                    small_block_size=args.small_block_size,
                    max_new_tokens=args.max_new_tokens,
                    mask_id=MASK_ID,
                    min_len=input_ids.shape[1],
                    seq_len=seq_len,
                    threshold=args.threshold,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )[0]
            sample_time = time.time() - sample_start

            new_ids = generated[input_ids.shape[1]:]
            mask_count = int((new_ids == MASK_ID).sum().item())
            text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            pred = normalize_number(text)
            gold = gold_answer(row["answer"])
            is_correct = pred == gold
            correct += int(is_correct)
            total += 1
            total_new_tokens += int((new_ids != MASK_ID).sum().item())
            unresolved_mask_examples += int(mask_count > 0)

            record = {
                "idx": idx,
                "question": row["question"],
                "gold": gold,
                "pred": pred,
                "correct": is_correct,
                "generated": text,
                "mask_count": mask_count,
                "seconds": sample_time,
            }
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
            f.flush()
            print(
                f"{idx + 1}/{args.num_examples} correct={correct}/{total} "
                f"gold={gold} pred={pred} seconds={sample_time:.2f}"
            )

    elapsed = time.time() - start
    summary = {
        "num_examples": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "elapsed_seconds": elapsed,
        "generated_tokens": total_new_tokens,
        "generated_tokens_per_second": total_new_tokens / elapsed if elapsed else 0.0,
        "unresolved_mask_examples": unresolved_mask_examples,
        "block_size": args.block_size,
        "small_block_size": args.small_block_size,
        "max_new_tokens": args.max_new_tokens,
        "threshold": args.threshold,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "output_jsonl": str(out_path),
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
