#!/usr/bin/env python3
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch


ROOT = Path("/home/mark/qwen_diffusion")
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_candidate_pairwise_curriculum import PAIRWISE_SYSTEM  # noqa: E402
from build_synthetic_multicall_candidate_index_examples import load_jsonl  # noqa: E402
from eval_fastdllm_candidate_index_ranking import rank_order, score_index_candidate  # noqa: E402
from eval_fastdllm_toolcall_cases import (  # noqa: E402
    apply_chat_template,
    load_model,
    resolve_chat_template,
    resolve_token_ids,
)


DEFAULT_BASE = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_EXAMPLES = ROOT / "data/qwen35_9b_public_multicall_v5_focused_miss_pairwise_diag_curriculum/pairwise_examples.jsonl"
DEFAULT_OUT = ROOT / "runs/candidate_ranking/public_multicall_targetcandidate_v5_miss_targets_ckpt275_pairwise_rank.jsonl"
DEFAULT_ADAPTER = ROOT / "runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model"


def build_prompt(tokenizer, example, chat_template):
    messages = [
        {"role": "system", "content": PAIRWISE_SYSTEM},
        {"role": "user", "content": example.get("prompt") or ""},
    ]
    return apply_chat_template(tokenizer, messages, None, chat_template=chat_template)


def row_key(row):
    return (
        row.get("id"),
        row.get("kind"),
        row.get("tool_call_index"),
        row.get("json_key"),
        json.dumps(row.get("target"), ensure_ascii=False),
        json.dumps(row.get("candidate_a"), ensure_ascii=False),
        json.dumps(row.get("candidate_b"), ensure_ascii=False),
        row.get("answer"),
    )


def group_key(row):
    if row.get("group_key"):
        return row["group_key"]
    return "|".join(
        "" if item is None else str(item)
        for item in (
            row.get("id"),
            row.get("kind"),
            row.get("tool_call_index"),
            row.get("json_key"),
            json.dumps(row.get("target"), ensure_ascii=False),
        )
    )


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
    parser.add_argument("--dedupe", action=argparse.BooleanOptionalAction, default=True)
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
    if args.dedupe:
        deduped = {}
        for row in examples:
            deduped.setdefault(row_key(row), row)
        examples = list(deduped.values())
    if args.limit and args.limit > 0:
        examples = examples[: args.limit]

    label_token_ids = {
        "A": tokenizer("A", add_special_tokens=False).input_ids,
        "B": tokenizer("B", add_special_tokens=False).input_ids,
    }
    for label, ids in label_token_ids.items():
        if not ids:
            raise ValueError(f"label {label} tokenized to empty ids")

    totals = Counter()
    margins = []
    group_results = defaultdict(list)
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
                "miss_path": example.get("miss_path"),
                "json_path": example.get("json_path"),
                "argument_path": example.get("argument_path"),
                "candidate_a": example.get("candidate_a"),
                "candidate_b": example.get("candidate_b"),
                "candidate_a_index": example.get("candidate_a_index"),
                "candidate_b_index": example.get("candidate_b_index"),
                "answer": example.get("answer"),
                "group_key": group_key(example),
                "status": "ok",
            }
            try:
                prompt = build_prompt(tokenizer, example, chat_template)
                prompt_ids = tokenizer([prompt], return_tensors="pt").input_ids.to("cuda")
                scores = []
                for label in ["A", "B"]:
                    with torch.no_grad():
                        scores.append(score_index_candidate(model, prompt_ids, label_token_ids[label], mask_id))
                order = rank_order(scores)
                predicted = ["A", "B"][order[0]]
                target_label = str(example["answer"])
                target_pos = 0 if target_label == "A" else 1
                margin = float(scores[target_pos] - scores[1 - target_pos])
                correct = predicted == target_label
                margins.append(margin)
                group_results[row["group_key"]].append(correct)
                row.update(
                    {
                        "scores": {"A": scores[0], "B": scores[1]},
                        "predicted": predicted,
                        "correct": correct,
                        "target_margin": margin,
                    }
                )
                totals["examples"] += 1
                totals[f"examples:{example['kind']}"] += 1
                totals["correct"] += int(correct)
                totals[f"correct:{example['kind']}"] += int(correct)
            except Exception as exc:
                row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
                totals["errors"] += 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"pairwise-rank {idx + 1}/{len(examples)} ok={totals['examples']} "
                f"correct={totals['correct']} errors={totals['errors']}",
                flush=True,
            )

    group_count = len(group_results)
    group_all_correct = sum(1 for values in group_results.values() if values and all(values))
    elapsed = time.time() - start_time
    summary = {
        "base_model": str(args.base_model),
        "adapter": str(adapter_path) if adapter_path else None,
        "merge_adapter": not args.no_merge_adapter,
        "tokenizer_path": str(args.tokenizer_path),
        "examples_jsonl": str(args.examples_jsonl),
        "out_jsonl": str(args.out_jsonl),
        "conversation_template": args.conversation_template,
        "dedupe": args.dedupe,
        "totals": dict(totals),
        "pairwise_accuracy": totals["correct"] / totals["examples"] if totals["examples"] else 0.0,
        "group_count": group_count,
        "group_all_correct": group_all_correct,
        "group_all_correct_rate": group_all_correct / group_count if group_count else 0.0,
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
