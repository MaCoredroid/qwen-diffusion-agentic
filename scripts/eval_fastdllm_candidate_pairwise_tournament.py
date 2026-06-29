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

from build_candidate_pairwise_curriculum import PAIRWISE_SYSTEM, pairwise_prompt  # noqa: E402
from build_synthetic_multicall_candidate_index_examples import load_jsonl  # noqa: E402
from eval_fastdllm_candidate_index_ranking import rank_order, score_index_candidate  # noqa: E402
from eval_fastdllm_toolcall_cases import (  # noqa: E402
    apply_chat_template,
    load_model,
    resolve_chat_template,
    resolve_token_ids,
)


DEFAULT_BASE = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_EXAMPLES = ROOT / "data/candidate_ranking/public_multicall_targetcandidate_v5_miss_targets.jsonl"
DEFAULT_OUT = ROOT / "runs/candidate_ranking/public_multicall_targetcandidate_v5_miss_targets_ckpt275_pairwise_tournament.jsonl"
DEFAULT_ADAPTER = ROOT / "runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model"


def build_prompt(tokenizer, user_prompt, chat_template):
    messages = [
        {"role": "system", "content": PAIRWISE_SYSTEM},
        {"role": "user", "content": user_prompt},
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

    label_token_ids = {
        "A": tokenizer("A", add_special_tokens=False).input_ids,
        "B": tokenizer("B", add_special_tokens=False).input_ids,
    }
    for label, ids in label_token_ids.items():
        if not ids:
            raise ValueError(f"label {label} tokenized to empty ids")

    examples = [row for row in load_jsonl(args.examples_jsonl) if row.get("usable_for_training")]
    if args.limit and args.limit > 0:
        examples = examples[: args.limit]

    totals = Counter()
    target_margins = []
    pair_count = 0
    start_time = time.time()
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for idx, example in enumerate(examples):
            candidates = example.get("candidate_values") or []
            target_idx = int(example.get("target_index", -1))
            wins = [0 for _ in candidates]
            margin_sum = [0.0 for _ in candidates]
            comparisons = []
            row = {
                "idx": idx,
                "id": example.get("id"),
                "kind": example.get("kind"),
                "tool_call_index": example.get("tool_call_index"),
                "json_key": example.get("json_key"),
                "target": example.get("target"),
                "target_index": target_idx,
                "miss_path": example.get("miss_path"),
                "miss_generated": example.get("miss_generated"),
                "json_path": example.get("json_path"),
                "argument_path": example.get("argument_path"),
                "same_call_arguments": example.get("same_call_arguments") or [],
                "same_call_peer_arguments": example.get("same_call_peer_arguments") or [],
                "local_peer_arguments": example.get("local_peer_arguments") or [],
                "schedule_token_start": example.get("schedule_token_start"),
                "schedule_token_end": example.get("schedule_token_end"),
                "candidate_values": candidates,
                "status": "ok",
            }
            try:
                for left in range(len(candidates)):
                    for right in range(left + 1, len(candidates)):
                        user_prompt = pairwise_prompt(example, candidates[left], candidates[right])
                        prompt = build_prompt(tokenizer, user_prompt, chat_template)
                        prompt_ids = tokenizer([prompt], return_tensors="pt").input_ids.to("cuda")
                        scores = []
                        for label in ["A", "B"]:
                            with torch.no_grad():
                                scores.append(score_index_candidate(model, prompt_ids, label_token_ids[label], mask_id))
                        order = rank_order(scores)
                        winner = left if order[0] == 0 else right
                        loser = right if winner == left else left
                        score_margin = float(abs(scores[0] - scores[1]))
                        signed_left_margin = float(scores[0] - scores[1])
                        wins[winner] += 1
                        margin_sum[winner] += score_margin
                        margin_sum[loser] -= score_margin
                        comparisons.append(
                            {
                                "left_index": left,
                                "right_index": right,
                                "left_value": candidates[left],
                                "right_value": candidates[right],
                                "scores": {"A": scores[0], "B": scores[1]},
                                "winner_index": winner,
                                "winner_value": candidates[winner],
                                "signed_left_margin": signed_left_margin,
                            }
                        )
                        pair_count += 1
                order = sorted(range(len(candidates)), key=lambda item: (wins[item], margin_sum[item]), reverse=True)
                predicted = order[0] if order else -1
                target_win_margin = (
                    float(wins[target_idx] - max(wins[item] for item in range(len(candidates)) if item != target_idx))
                    if 0 <= target_idx < len(candidates) and len(candidates) > 1
                    else 0.0
                )
                target_margins.append(target_win_margin)
                row.update(
                    {
                        "wins": wins,
                        "margin_sum": margin_sum,
                        "rank_order": order,
                        "predicted_index": predicted,
                        "predicted_value": candidates[predicted] if predicted >= 0 else None,
                        "correct": predicted == target_idx,
                        "target_win_margin": target_win_margin,
                        "comparisons": comparisons,
                    }
                )
                totals["examples"] += 1
                totals[f"examples:{example['kind']}"] += 1
                totals["correct"] += int(predicted == target_idx)
                totals[f"correct:{example['kind']}"] += int(predicted == target_idx)
            except Exception as exc:
                row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
                totals["errors"] += 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"pairwise-tournament {idx + 1}/{len(examples)} ok={totals['examples']} "
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
        "target_win_margin_summary": percentile_summary(target_margins),
        "pair_count": pair_count,
        "elapsed_seconds": elapsed,
        "pairs_per_second": pair_count / elapsed if elapsed else 0.0,
        "mask_id": mask_id,
    }
    if torch.cuda.is_available():
        summary["cuda_max_memory_allocated_gib"] = torch.cuda.max_memory_allocated() / (1024**3)
        summary["cuda_max_memory_reserved_gib"] = torch.cuda.max_memory_reserved() / (1024**3)
    args.out_jsonl.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
