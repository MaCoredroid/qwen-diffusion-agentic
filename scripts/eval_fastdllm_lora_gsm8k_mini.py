#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import time
import types
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_ADAPTER = ROOT / "runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000"
DEFAULT_OUT = ROOT / "runs/fastdllm_apples_gsm8k/gsm8k_mini.jsonl"
DEFAULT_GSM8K = ROOT / "data/phaseA_retention/gsm8k_main_test_first20.jsonl"
DEFAULT_GSM8K_FEWSHOT = ROOT / "data/phaseA_retention/gsm8k_main_train_first5.jsonl"
MASK_ID = 151665
STOP_TOKEN_ID = 151645


def configure_cuda_env():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.pop("FASTDLLM_FLARE_TWO_STREAM", None)
    os.environ.pop("FLARE_TWO_STREAM", None)


def read_jsonl(path, limit=0):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def normalize_number(text):
    if text is None:
        return None
    matches = re.findall(r"[-+]?\$?\d[\d,]*(?:\.\d+)?", str(text))
    if not matches:
        return None
    return matches[-1].replace(",", "").replace("$", "").rstrip(".").lower()


def gold_answer(answer):
    if "####" in answer:
        return normalize_number(answer.split("####", 1)[1])
    return normalize_number(answer)


def strict_answer(text):
    match = re.search(r"####\s*([-+]?\$?\d[\d,]*(?:\.\d+)?)", str(text))
    return normalize_number(match.group(1)) if match else None


def make_final_only_prompt(tokenizer, question):
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


def make_phasea_prompt(tokenizer, row, fewshot_rows):
    messages = []
    for shot in fewshot_rows:
        messages.append({"role": "user", "content": f"Question: {shot['question']}\nAnswer:"})
        messages.append({"role": "assistant", "content": shot["answer"]})
    messages.append({"role": "user", "content": f"Question: {row['question']}\nAnswer:"})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def sample_with_top_p(logits, top_p, temperature):
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


def ban_mask_token_logits(logits, mask_id):
    logits = logits.clone()
    logits[..., int(mask_id)] = torch.finfo(logits.dtype).min
    return logits


def empty_sampler_metrics(args):
    return {
        "mode": args.decode_mode,
        "threshold": args.parallel_commit_threshold,
        "denoise_forwards": 0,
        "committed_tokens": 0,
        "natural_commits": 0,
        "forced_progress_commits": 0,
        "selected_mask_tokens": 0,
        "blocks": 0,
        "stop_token_ids": list(args.stop_token_ids),
    }


def full_context_commit_anywhere_sample(model, input_ids, args):
    output_ids = input_ids.unsqueeze(0).to("cuda")
    original_len = int(output_ids.shape[1])
    metrics = empty_sampler_metrics(args)
    stop_token_ids = torch.tensor(args.stop_token_ids, dtype=torch.long, device=output_ids.device)

    def truncate_if_stopped(sequence):
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
        metrics["blocks"] += 1

        while bool((x_t[:, -block_pad:] == args.mask_id).any().item()):
            window_len = min(args.block_size, x_t.shape[1])
            mask_idx = x_t[:, -window_len:] == args.mask_id
            current_mask = mask_idx
            if not bool(current_mask.any().item()):
                break

            output = model(input_ids=x_t, use_cache=False)
            logits = torch.cat([output.logits[:, :1, :], output.logits[:, :-1, :]], dim=1)
            logits = logits[:, -window_len:]
            logits = ban_mask_token_logits(logits, args.mask_id)
            x_1, p_1t = sample_with_top_p(logits, args.top_p, args.temperature)
            x1_p = torch.squeeze(torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1)
            active_probs = torch.where(current_mask, x1_p, torch.full_like(x1_p, -torch.inf))

            if args.decode_mode == "greedy_one":
                natural = torch.zeros_like(current_mask)
            else:
                natural = active_probs > args.parallel_commit_threshold
            unmask_idx = natural.clone()
            flat_idx = int(active_probs.reshape(-1).argmax().item())
            row_idx = flat_idx // active_probs.shape[1]
            pos_idx = flat_idx % active_probs.shape[1]
            unmask_idx[row_idx, pos_idx] = True
            unmask_idx = unmask_idx & current_mask

            selected_mask = int(((x_1 == args.mask_id) & unmask_idx).sum().item())
            natural_count = int((natural & current_mask).sum().item())
            committed = int(unmask_idx.sum().item())
            if committed <= 0:
                raise RuntimeError("commit-anywhere made no progress")

            window = x_t[:, -window_len:].clone()
            window[unmask_idx] = x_1[unmask_idx]
            x_t[:, -window_len:] = window

            metrics["denoise_forwards"] += 1
            metrics["committed_tokens"] += committed
            metrics["natural_commits"] += natural_count
            metrics["forced_progress_commits"] += max(0, committed - natural_count)
            metrics["selected_mask_tokens"] += selected_mask

            stopped = truncate_if_stopped(x_t)
            if stopped is not None:
                return stopped[0].detach().cpu(), metrics

        output_ids = x_t
        stopped = truncate_if_stopped(output_ids)
        if stopped is not None:
            output_ids = stopped
            break

    return output_ids[0].detach().cpu(), metrics


def resolve_token_ids(model, tokenizer, mask_id, stop_token_id):
    if mask_id is None:
        mask_id = getattr(model.config, "mask_token_id", None)
    if mask_id is None:
        converted = tokenizer.convert_tokens_to_ids("|<MASK>|")
        if converted != tokenizer.unk_token_id:
            mask_id = converted
    if mask_id is None:
        mask_id = MASK_ID

    stop_ids = []

    def add_stop(value):
        if value is None:
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                add_stop(item)
            return
        value = int(value)
        if value not in stop_ids:
            stop_ids.append(value)

    add_stop(stop_token_id)
    add_stop(tokenizer.eos_token_id)
    add_stop(getattr(model.config, "eos_token_id", None))
    for text in ("<|im_end|>", "<|im_start|>"):
        token_ids = tokenizer(text, add_special_tokens=False).input_ids
        if len(token_ids) == 1:
            add_stop(token_ids[0])
    if not stop_ids:
        add_stop(STOP_TOKEN_ID)
    return int(mask_id), int(stop_ids[0]), stop_ids


def load_cases(args):
    if args.gsm8k_path:
        rows = read_jsonl(args.gsm8k_path, args.num_examples)
        return rows
    ds = load_dataset("openai/gsm8k", "main", split=f"test[:{args.num_examples}]")
    return [
        {"idx": idx, "question": row["question"], "answer": row["answer"], "source": "openai/gsm8k:main:test"}
        for idx, row in enumerate(ds)
    ]


def load_model(args):
    repo_v2 = ROOT / "fast-dllm/v2"
    sys.path.insert(0, str(repo_v2))
    import generation_functions

    tokenizer_path = args.tokenizer_path or args.adapter or args.base_model
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(str(args.base_model), trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        str(args.base_model),
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(base, str(args.adapter))
        if not args.no_merge_adapter:
            model = model.merge_and_unload()
    else:
        model = base
    model.mdm_sample = types.MethodType(
        generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample,
        model,
    )
    model.to("cuda").eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gsm8k-path", type=Path, default=DEFAULT_GSM8K)
    parser.add_argument("--gsm8k-fewshot-path", type=Path, default=DEFAULT_GSM8K_FEWSHOT)
    parser.add_argument("--num-examples", type=int, default=20)
    parser.add_argument("--prompt-mode", choices=["phasea_fewshot", "final_only"], default="phasea_fewshot")
    parser.add_argument(
        "--decode-mode",
        choices=["greedy_one", "fastdllm_anywhere", "original_mdm"],
        default="fastdllm_anywhere",
    )
    parser.add_argument("--parallel-commit-threshold", type=float, default=0.9)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--small-block-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--mask-id", type=int, default=None)
    parser.add_argument("--stop-token-id", type=int, default=None)
    parser.add_argument("--tail-fill-generation", action="store_false", dest="fresh_generation_blocks")
    parser.add_argument("--no-merge-adapter", action="store_true")
    parser.set_defaults(fresh_generation_blocks=True)
    args = parser.parse_args()

    configure_cuda_env()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args)
    args.mask_id, args.stop_token_id, args.stop_token_ids = resolve_token_ids(
        model,
        tokenizer,
        args.mask_id,
        args.stop_token_id,
    )
    print(
        "[token_ids] "
        + json.dumps({"mask_id": args.mask_id, "stop_token_ids": args.stop_token_ids}, sort_keys=True),
        flush=True,
    )

    cases = load_cases(args)
    fewshot_rows = read_jsonl(args.gsm8k_fewshot_path, 0) if args.prompt_mode == "phasea_fewshot" else []

    correct = 0
    strict_correct = 0
    total = 0
    total_new_tokens = 0
    unresolved_mask_examples = 0
    sampler_totals = {
        "denoise_forwards": 0,
        "committed_tokens": 0,
        "natural_commits": 0,
        "forced_progress_commits": 0,
        "selected_mask_tokens": 0,
        "blocks": 0,
    }
    start = time.time()

    with out_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(cases):
            if args.prompt_mode == "phasea_fewshot":
                prompt = make_phasea_prompt(tokenizer, row, fewshot_rows)
            else:
                prompt = make_final_only_prompt(tokenizer, row["question"])
            input_ids = tokenizer([prompt], return_tensors="pt").input_ids

            sample_start = time.time()
            with torch.inference_mode():
                if args.decode_mode == "original_mdm":
                    seq_len = torch.tensor([input_ids.shape[1]], device="cuda")
                    generated = model.mdm_sample(
                        input_ids.to("cuda"),
                        tokenizer=tokenizer,
                        block_size=args.block_size,
                        small_block_size=args.small_block_size,
                        max_new_tokens=args.max_new_tokens,
                        mask_id=args.mask_id,
                        stop_token=args.stop_token_id,
                        min_len=input_ids.shape[1],
                        seq_len=seq_len,
                        threshold=args.threshold,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        fresh_generation_blocks=args.fresh_generation_blocks,
                    )[0].detach().cpu()
                    sampler_metrics = {}
                else:
                    generated, sampler_metrics = full_context_commit_anywhere_sample(
                        model,
                        input_ids[0].cpu(),
                        args,
                    )
            sample_time = time.time() - sample_start

            prompt_len = int(input_ids.shape[1])
            new_ids = generated[prompt_len:]
            mask_count = int((new_ids == args.mask_id).sum().item())
            text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            pred = normalize_number(text)
            strict_pred = strict_answer(text)
            gold = gold_answer(row["answer"])
            is_correct = pred == gold
            is_strict_correct = strict_pred == gold
            correct += int(is_correct)
            strict_correct += int(is_strict_correct)
            total += 1
            nonmask_tokens = int((new_ids != args.mask_id).sum().item())
            total_new_tokens += nonmask_tokens
            unresolved_mask_examples += int(mask_count > 0)
            for key in sampler_totals:
                sampler_totals[key] += int(sampler_metrics.get(key) or 0)

            record = {
                "idx": row.get("idx", idx),
                "source": row.get("source"),
                "question": row["question"],
                "gold": gold,
                "pred": pred,
                "strict_pred": strict_pred,
                "correct": is_correct,
                "strict_correct": is_strict_correct,
                "generated": text,
                "mask_count": mask_count,
                "nonmask_generated_tokens": nonmask_tokens,
                "seconds": sample_time,
                "sampler": sampler_metrics,
            }
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            handle.flush()
            print(
                f"{idx + 1}/{len(cases)} strict={strict_correct}/{total} flex={correct}/{total} "
                f"gold={gold} strict_pred={strict_pred} flex_pred={pred} seconds={sample_time:.2f}",
                flush=True,
            )

    elapsed = time.time() - start
    denoise_forwards = sampler_totals["denoise_forwards"]
    committed_tokens = sampler_totals["committed_tokens"]
    summary = {
        "num_examples": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "strict_correct": strict_correct,
        "strict_accuracy": strict_correct / total if total else 0.0,
        "elapsed_seconds": elapsed,
        "generated_tokens": total_new_tokens,
        "generated_tokens_per_second": total_new_tokens / elapsed if elapsed else 0.0,
        "unresolved_mask_examples": unresolved_mask_examples,
        "sampler_totals": sampler_totals,
        "tokens_per_forward": committed_tokens / denoise_forwards if denoise_forwards else None,
        "generated_tokens_per_forward": total_new_tokens / denoise_forwards if denoise_forwards else None,
        "forced_progress_fraction": (
            sampler_totals["forced_progress_commits"] / committed_tokens if committed_tokens else None
        ),
        "base_model": str(args.base_model),
        "adapter": str(args.adapter) if args.adapter else None,
        "merge_adapter": not args.no_merge_adapter,
        "prompt_mode": args.prompt_mode,
        "decode_mode": args.decode_mode,
        "parallel_commit_threshold": args.parallel_commit_threshold,
        "block_size": args.block_size,
        "small_block_size": args.small_block_size,
        "max_new_tokens": args.max_new_tokens,
        "threshold": args.threshold,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "fresh_generation_blocks": args.fresh_generation_blocks,
        "mask_id": args.mask_id,
        "stop_token_id": args.stop_token_id,
        "stop_token_ids": args.stop_token_ids,
        "output_jsonl": str(out_path),
    }
    if torch.cuda.is_available():
        summary["cuda_max_memory_allocated_gib"] = torch.cuda.max_memory_allocated() / (1024**3)
        summary["cuda_max_memory_reserved_gib"] = torch.cuda.max_memory_reserved() / (1024**3)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
