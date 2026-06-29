#!/usr/bin/env python3
import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch


ROOT = Path("/home/mark/qwen_diffusion")
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_synthetic_multicall_candidate_index_examples import load_jsonl  # noqa: E402
from eval_fastdllm_candidate_index_ranking import INDEX_SYSTEM  # noqa: E402
from eval_fastdllm_toolcall_cases import (  # noqa: E402
    apply_chat_template,
    load_model,
    resolve_chat_template,
    resolve_token_ids,
)


DEFAULT_BASE = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_EXAMPLES = ROOT / "data/candidate_ranking/public_multicall_targetcandidate_v5_miss_targets.jsonl"
DEFAULT_OUT = ROOT / "runs/candidate_ranking/public_multicall_targetcandidate_v5_miss_targets_ckpt275_index_generate.jsonl"
DEFAULT_ADAPTER = ROOT / "runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model"


def build_prompt(tokenizer, example, chat_template):
    messages = [
        {"role": "system", "content": INDEX_SYSTEM},
        {"role": "user", "content": example.get("prompt") or ""},
    ]
    return apply_chat_template(tokenizer, messages, None, chat_template=chat_template)


def parse_index(text):
    match = re.search(r"-?\d+", text)
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def generate_index(model, tokenizer, prompt, args, mask_id, stop_token_id):
    input_ids = tokenizer([prompt], return_tensors="pt").input_ids.to("cuda")
    seq_len = torch.tensor([input_ids.shape[1]], device="cuda")
    with torch.no_grad():
        generated = model.mdm_sample(
            input_ids,
            tokenizer=tokenizer,
            block_size=args.block_size,
            small_block_size=args.small_block_size,
            max_new_tokens=args.max_new_tokens,
            mask_id=mask_id,
            stop_token=stop_token_id,
            min_len=input_ids.shape[1],
            seq_len=seq_len,
            threshold=args.threshold,
            temperature=args.temperature,
            top_p=args.top_p,
            use_block_cache=args.use_block_cache,
        )[0]
    new_ids = generated[input_ids.shape[1] :]
    mask_count = int((new_ids == mask_id).sum().item())
    generated_token_count = int((new_ids != mask_id).sum().item())
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return text, mask_count, generated_token_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--examples-jsonl", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--conversation-template", default="fast_dllm_v2")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-merge-adapter", action="store_true")
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--small-block-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--use-block-cache", action="store_true")
    args = parser.parse_args()

    chat_template = resolve_chat_template(args.conversation_template)
    adapter_path = None if args.no_adapter else args.adapter
    model, tokenizer = load_model(
        str(args.base_model),
        str(adapter_path) if adapter_path else None,
        merge_adapter=not args.no_merge_adapter,
        tokenizer_path=str(args.tokenizer_path) if args.tokenizer_path else None,
    )
    mask_id, stop_token_id = resolve_token_ids(model, tokenizer)
    examples = [row for row in load_jsonl(args.examples_jsonl) if row.get("usable_for_training")]
    if args.limit and args.limit > 0:
        examples = examples[: args.limit]

    totals = Counter()
    start_time = time.time()
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for idx, example in enumerate(examples):
            row = {
                "idx": idx,
                "id": example.get("id"),
                "kind": example.get("kind"),
                "tool_call_index": example.get("tool_call_index"),
                "json_key": example.get("json_key"),
                "target": example.get("target"),
                "target_index": example.get("target_index"),
                "candidate_values": example.get("candidate_values") or [],
                "status": "ok",
            }
            try:
                target_idx = int(example["target_index"])
                prompt = build_prompt(tokenizer, example, chat_template)
                text, mask_count, generated_token_count = generate_index(
                    model,
                    tokenizer,
                    prompt,
                    args,
                    mask_id,
                    stop_token_id,
                )
                predicted = parse_index(text)
                candidate_values = example.get("candidate_values") or []
                in_range = predicted is not None and 0 <= predicted < len(candidate_values)
                row.update(
                    {
                        "generated_text": text,
                        "predicted_index": predicted,
                        "predicted_value": candidate_values[predicted] if in_range else None,
                        "correct": predicted == target_idx,
                        "in_range": in_range,
                        "mask_count": mask_count,
                        "generated_token_count": generated_token_count,
                    }
                )
                totals["examples"] += 1
                totals[f"examples:{example['kind']}"] += 1
                totals["correct"] += int(predicted == target_idx)
                totals["in_range"] += int(in_range)
            except Exception as exc:
                row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
                totals["errors"] += 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"index-generate {idx + 1}/{len(examples)} ok={totals['examples']} "
                f"correct={totals['correct']} errors={totals['errors']}",
                flush=True,
            )

    elapsed = time.time() - start_time
    summary = {
        "base_model": str(args.base_model),
        "adapter": str(adapter_path) if adapter_path else None,
        "merge_adapter": not args.no_merge_adapter,
        "tokenizer_path": str(args.tokenizer_path),
        "examples_jsonl": str(args.examples_jsonl),
        "out_jsonl": str(args.out_jsonl),
        "conversation_template": args.conversation_template,
        "generation": {
            "block_size": args.block_size,
            "small_block_size": args.small_block_size,
            "max_new_tokens": args.max_new_tokens,
            "threshold": args.threshold,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "use_block_cache": args.use_block_cache,
        },
        "totals": dict(totals),
        "accuracy": totals["correct"] / totals["examples"] if totals["examples"] else 0.0,
        "in_range_rate": totals["in_range"] / totals["examples"] if totals["examples"] else 0.0,
        "elapsed_seconds": elapsed,
        "examples_per_second": totals["examples"] / elapsed if elapsed else 0.0,
        "mask_id": mask_id,
        "stop_token_id": stop_token_id,
    }
    if torch.cuda.is_available():
        summary["cuda_max_memory_allocated_gib"] = torch.cuda.max_memory_allocated() / (1024**3)
        summary["cuda_max_memory_reserved_gib"] = torch.cuda.max_memory_reserved() / (1024**3)
    args.out_jsonl.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
