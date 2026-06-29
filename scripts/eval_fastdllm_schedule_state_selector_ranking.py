#!/usr/bin/env python3
"""Rank constrained schedule-state selector decisions by masked likelihood.

Free-form selector generation failed to emit executable JSON. This evaluator
keeps the selector as control state: it scores fixed JSON decision templates, or
just the constrained `candidate_index` value after a forced JSON prefix.
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch


ROOT = Path("/home/mark/qwen_diffusion")
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from eval_fastdllm_schedule_state_selector import (  # noqa: E402
    DECISION_KEYS,
    build_prompt,
    expected_decision,
    infer_metadata,
    load_instances,
    normalized_decision,
)
from eval_fastdllm_toolcall_cases import (  # noqa: E402
    load_model,
    resolve_chat_template,
    resolve_token_ids,
)


DEFAULT_BASE = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_DATASET = ROOT / "data/qwen35_9b_schedule_state_selector_no_public_smoke_curriculum/train_agentic_mix.json"
DEFAULT_OUT = ROOT / "runs/schedule_state_selector/no_public_smoke_selector_rank.jsonl"
DEFAULT_ADAPTER = ROOT / "runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model"


def decision_with_index(expected: dict[str, Any], candidate_index: int) -> dict[str, Any]:
    decision = {key: expected[key] for key in DECISION_KEYS if key in expected}
    decision["candidate_index"] = candidate_index
    return decision


def compact_json(decision: dict[str, Any]) -> str:
    return json.dumps(decision, ensure_ascii=False, separators=(",", ":"))


def candidate_count_from_prompt(instance: dict[str, Any]) -> int:
    for message in instance.get("messages") or []:
        if message.get("role") != "user":
            continue
        match = re.search(r"^Candidate count:\s*(\d+)\s*$", str(message.get("content") or ""), flags=re.MULTILINE)
        if match:
            return int(match.group(1))
    raise ValueError("candidate count not found")


def score_masked_continuation(model, prefix_ids, candidate_ids, mask_id, normalize):
    masks = torch.full(
        (prefix_ids.shape[0], candidate_ids.shape[1]),
        mask_id,
        dtype=prefix_ids.dtype,
        device=prefix_ids.device,
    )
    x_t = torch.cat([prefix_ids, masks], dim=1)
    logits = model(input_ids=x_t, use_cache=False).logits
    logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
    log_probs = torch.log_softmax(logits[:, -candidate_ids.shape[1] :, :], dim=-1)
    token_log_probs = log_probs.gather(2, candidate_ids.unsqueeze(-1)).squeeze(-1)
    score = token_log_probs.sum()
    if normalize and candidate_ids.shape[1]:
        score = score / candidate_ids.shape[1]
    return float(score.detach().cpu())


def score_candidates(model, tokenizer, prompt: str, expected: dict[str, Any], candidate_count: int, args, mask_id: int):
    if args.score_mode == "index_only":
        forced_prefix = prompt + '{"candidate_index":'
        prefix_ids = tokenizer([forced_prefix], return_tensors="pt").input_ids.to("cuda")
        candidate_texts = [str(idx) for idx in range(candidate_count)]
    elif args.score_mode == "full_decision":
        prefix_ids = tokenizer([prompt], return_tensors="pt").input_ids.to("cuda")
        candidate_texts = [compact_json(decision_with_index(expected, idx)) for idx in range(candidate_count)]
    else:
        raise ValueError(f"unknown score mode {args.score_mode!r}")

    scores = []
    token_lengths = []
    for text in candidate_texts:
        candidate_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
        if candidate_ids.numel() == 0:
            raise ValueError(f"empty candidate tokenization for {text!r}")
        with torch.no_grad():
            scores.append(score_masked_continuation(model, prefix_ids, candidate_ids, mask_id, args.length_normalize))
        token_lengths.append(int(candidate_ids.shape[1]))
    return candidate_texts, scores, token_lengths


def rank_order(scores: list[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)


def percentile_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    values = sorted(values)

    def at(frac: float) -> float:
        idx = min(len(values) - 1, max(0, round((len(values) - 1) * frac)))
        return values[idx]

    return {"min": values[0], "p50": at(0.50), "p90": at(0.90), "max": values[-1]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--dataset-json", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--conversation-template", default="fast_dllm_v2")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--only-ambiguous", action="store_true")
    parser.add_argument("--no-merge-adapter", action="store_true")
    parser.add_argument("--score-mode", choices=["index_only", "full_decision"], default="index_only")
    parser.add_argument("--length-normalize", action="store_true")
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

    selected = []
    for source_idx, instance in enumerate(load_instances(args.dataset_json)):
        expected = normalized_decision(expected_decision(instance))
        meta = infer_metadata(instance, source_idx, expected)
        candidate_count = candidate_count_from_prompt(instance)
        meta["candidate_count"] = candidate_count
        if args.only_ambiguous and candidate_count <= 1:
            continue
        selected.append((source_idx, instance, expected, meta))
    if args.offset > 0:
        selected = selected[args.offset :]
    if args.limit and args.limit > 0:
        selected = selected[: args.limit]

    totals = Counter()
    margins = []
    start_time = time.time()
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for eval_idx, (source_idx, instance, expected, meta) in enumerate(selected):
            row = {
                "eval_idx": eval_idx,
                "source_idx": source_idx,
                **meta,
                "expected": expected,
                "status": "ok",
            }
            try:
                target_idx = int(expected["candidate_index"])
                candidate_count = int(meta["candidate_count"])
                if target_idx < 0 or target_idx >= candidate_count:
                    raise ValueError(f"target index {target_idx} out of range {candidate_count}")
                prompt = build_prompt(tokenizer, instance, chat_template)
                candidate_texts, scores, token_lengths = score_candidates(
                    model,
                    tokenizer,
                    prompt,
                    expected,
                    candidate_count,
                    args,
                    mask_id,
                )
                order = rank_order(scores)
                predicted = order[0] if order else -1
                target_rank = order.index(target_idx) + 1 if target_idx in order else None
                margin = (
                    float(scores[target_idx] - max(score for idx, score in enumerate(scores) if idx != target_idx))
                    if len(scores) > 1
                    else 0.0
                )
                margins.append(margin)
                correct = predicted == target_idx
                row.update(
                    {
                        "candidate_texts": candidate_texts,
                        "scores": scores,
                        "token_lengths": token_lengths,
                        "rank_order": order,
                        "predicted_index": predicted,
                        "target_rank": target_rank,
                        "target_score": scores[target_idx],
                        "best_score": scores[predicted] if predicted >= 0 else None,
                        "target_margin": margin,
                        "correct": correct,
                        "score_mode": args.score_mode,
                        "length_normalize": args.length_normalize,
                    }
                )
                totals["examples"] += 1
                totals["correct"] += int(correct)
                family = "ambiguous" if candidate_count > 1 else "singleton"
                totals[f"examples:{family}"] += 1
                totals[f"correct:{family}"] += int(correct)
                totals[f"rank1_or_2:{family}"] += int(target_rank is not None and target_rank <= 2)
                totals["rank1_or_2"] += int(target_rank is not None and target_rank <= 2)
            except Exception as exc:
                row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
                totals["errors"] += 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"selector-rank {eval_idx + 1}/{len(selected)} ok={totals['examples']} "
                f"correct={totals['correct']} errors={totals['errors']}",
                flush=True,
            )

    elapsed = time.time() - start_time
    examples = totals["examples"]
    ambiguous = totals["examples:ambiguous"]
    summary = {
        "base_model": str(args.base_model),
        "adapter": str(adapter_path) if adapter_path else None,
        "merge_adapter": not args.no_merge_adapter,
        "tokenizer_path": str(args.tokenizer_path),
        "dataset_json": str(args.dataset_json),
        "out_jsonl": str(args.out_jsonl),
        "conversation_template": args.conversation_template,
        "limit": args.limit,
        "offset": args.offset,
        "only_ambiguous": args.only_ambiguous,
        "score_mode": args.score_mode,
        "length_normalize": args.length_normalize,
        "totals": dict(totals),
        "accuracy": totals["correct"] / examples if examples else 0.0,
        "ambiguous_accuracy": totals["correct:ambiguous"] / ambiguous if ambiguous else 0.0,
        "rank1_or_2_rate": totals["rank1_or_2"] / examples if examples else 0.0,
        "ambiguous_rank1_or_2_rate": totals["rank1_or_2:ambiguous"] / ambiguous if ambiguous else 0.0,
        "margin_summary": percentile_summary(margins),
        "elapsed_seconds": elapsed,
        "examples_per_second": examples / elapsed if elapsed else 0.0,
        "mask_id": mask_id,
    }
    if torch.cuda.is_available():
        summary["cuda_max_memory_allocated_gib"] = torch.cuda.max_memory_allocated() / (1024**3)
        summary["cuda_max_memory_reserved_gib"] = torch.cuda.max_memory_reserved() / (1024**3)
    args.out_jsonl.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
