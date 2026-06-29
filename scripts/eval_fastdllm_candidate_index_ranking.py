#!/usr/bin/env python3
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch


ROOT = Path("/home/mark/qwen_diffusion")
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_synthetic_multicall_candidate_index_examples import load_jsonl  # noqa: E402
from eval_fastdllm_toolcall_cases import (  # noqa: E402
    apply_chat_template,
    load_model,
    resolve_chat_template,
    resolve_token_ids,
)


DEFAULT_BASE = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_EXAMPLES = ROOT / "data/candidate_ranking/synthetic_multicall_failure_analogue_index_ranking.jsonl"
DEFAULT_OUT = ROOT / "runs/candidate_ranking/synthetic_multicall_failure_analogue_ckpt275_index_rank.jsonl"
DEFAULT_ADAPTER = ROOT / "runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model"
INDEX_SYSTEM = "You select the correct candidate index for tool-call behavior preservation."


def score_index_candidate(model, prompt_ids, candidate_ids, mask_id):
    masks = torch.full(
        (prompt_ids.shape[0], len(candidate_ids)),
        mask_id,
        dtype=prompt_ids.dtype,
        device=prompt_ids.device,
    )
    x_t = torch.cat([prompt_ids, masks], dim=1)
    logits = model(input_ids=x_t, use_cache=False).logits
    logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
    log_probs = torch.log_softmax(logits[:, -len(candidate_ids) :, :], dim=-1)
    score = torch.zeros((), dtype=log_probs.dtype, device=log_probs.device)
    for pos, token_id in enumerate(candidate_ids):
        score = score + log_probs[0, pos, int(token_id)]
    return float(score.detach().cpu())


def rank_order(scores):
    return sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)


def build_prompt(tokenizer, example, chat_template):
    messages = [
        {"role": "system", "content": INDEX_SYSTEM},
        {"role": "user", "content": example.get("prompt") or ""},
    ]
    return apply_chat_template(tokenizer, messages, None, chat_template=chat_template)


def percentile_summary(values):
    if not values:
        return {}
    values = sorted(values)

    def at(frac):
        idx = min(len(values) - 1, max(0, round((len(values) - 1) * frac)))
        return values[idx]

    return {"min": values[0], "p50": at(0.50), "p90": at(0.90), "max": values[-1]}


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
    args = parser.parse_args()

    chat_template = resolve_chat_template(args.conversation_template)
    adapter_path = None if args.no_adapter else args.adapter
    model, tokenizer = load_model(
        str(args.base_model),
        str(adapter_path) if adapter_path else None,
        merge_adapter=not args.no_merge_adapter,
        tokenizer_path=str(args.tokenizer_path) if args.tokenizer_path else None,
    )
    mask_id, _ = resolve_token_ids(model, tokenizer)
    examples = [row for row in load_jsonl(args.examples_jsonl) if row.get("usable_for_training")]
    if args.limit and args.limit > 0:
        examples = examples[: args.limit]

    index_token_ids = []
    for idx in range(max((len(row.get("candidate_values") or []) for row in examples), default=0)):
        ids = tokenizer(str(idx), add_special_tokens=False).input_ids
        if not ids:
            raise ValueError(f"index {idx} tokenized to empty ids")
        index_token_ids.append(ids)

    totals = Counter()
    margins = []
    start_time = time.time()
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for idx, example in enumerate(examples):
            row = {
                "idx": idx,
                "id": example.get("id"),
                "analogue_family": example.get("analogue_family"),
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
                prompt_ids = tokenizer([prompt], return_tensors="pt").input_ids.to("cuda")
                scores = []
                for candidate_idx in range(len(example.get("candidate_values") or [])):
                    with torch.no_grad():
                        scores.append(score_index_candidate(model, prompt_ids, index_token_ids[candidate_idx], mask_id))
                order = rank_order(scores)
                predicted = order[0] if order else -1
                margin = (
                    float(scores[target_idx] - max(score for score_idx, score in enumerate(scores) if score_idx != target_idx))
                    if len(scores) > 1
                    else 0.0
                )
                margins.append(margin)
                row.update(
                    {
                        "scores": scores,
                        "rank_order": order,
                        "predicted_index": predicted,
                        "predicted_value": example["candidate_values"][predicted] if predicted >= 0 else None,
                        "correct": predicted == target_idx,
                        "target_margin": margin,
                        "candidate_count": len(scores),
                    }
                )
                totals["examples"] += 1
                totals[f"examples:{example['kind']}"] += 1
                totals[f"examples:{example.get('analogue_family')}"] += 1
                totals["correct"] += int(predicted == target_idx)
                totals[f"correct:{example['kind']}"] += int(predicted == target_idx)
                totals[f"correct:{example.get('analogue_family')}"] += int(predicted == target_idx)
            except Exception as exc:
                row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
                totals["errors"] += 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"index-rank {idx + 1}/{len(examples)} ok={totals['examples']} "
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
        "totals": dict(totals),
        "accuracy": totals["correct"] / totals["examples"] if totals["examples"] else 0.0,
        "margin_summary": percentile_summary(margins),
        "elapsed_seconds": elapsed,
        "examples_per_second": totals["examples"] / elapsed if elapsed else 0.0,
        "mask_id": mask_id,
    }
    if torch.cuda.is_available():
        summary["cuda_max_memory_allocated_gib"] = torch.cuda.max_memory_allocated() / (1024**3)
        summary["cuda_max_memory_reserved_gib"] = torch.cuda.max_memory_reserved() / (1024**3)
    args.out_jsonl.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
