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

from build_candidate_ranking_examples import case_key, load_jsonl  # noqa: E402
from eval_fastdllm_toolcall_cases import (  # noqa: E402
    load_model,
    make_prompt,
    resolve_chat_template,
    resolve_token_ids,
)


DEFAULT_BASE = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_ADAPTER = (
    ROOT
    / "runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model"
)
DEFAULT_EXAMPLES = ROOT / "data/candidate_ranking/public_multicall_toolname_argument_ranking_v3_12.jsonl"
DEFAULT_CASES = ROOT / "data/toolcall_eval/public_multicall_hermes_smoke.jsonl"
DEFAULT_OUT = ROOT / "runs/candidate_ranking/public_multicall_qwen35_ckpt275_masked_span_rank_v3_12.jsonl"


def load_cases(path):
    return {case_key(row, idx): row for idx, row in enumerate(load_jsonl(path))}


def score_candidate_sequences(model, x_t, abs_start, abs_end, candidate_token_ids):
    logits = model(input_ids=x_t, use_cache=False).logits
    logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
    log_probs = torch.log_softmax(logits[:, abs_start:abs_end, :], dim=-1)
    scores = []
    for sequence in candidate_token_ids:
        score = torch.zeros((), dtype=log_probs.dtype, device=log_probs.device)
        for pos, token_id in enumerate(sequence):
            score = score + log_probs[0, pos, int(token_id)]
        scores.append(float(score.detach().cpu()))
    return scores


def rank_order(scores):
    return sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)


def candidate_context(prompt_ids, assistant_ids, span_start, span_end, mask_id, mode):
    if span_end > assistant_ids.shape[1]:
        raise ValueError(
            f"span end {span_end} exceeds assistant token count {assistant_ids.shape[1]}"
        )
    if mode == "full_gold":
        x_t = torch.cat([prompt_ids, assistant_ids.clone()], dim=1)
        abs_start = prompt_ids.shape[1] + span_start
        abs_end = prompt_ids.shape[1] + span_end
        x_t[:, abs_start:abs_end] = mask_id
        return x_t, abs_start, abs_end
    if mode == "prefix_only":
        prefix = assistant_ids[:, :span_start]
        masks = torch.full(
            (assistant_ids.shape[0], span_end - span_start),
            mask_id,
            dtype=assistant_ids.dtype,
            device=assistant_ids.device,
        )
        x_t = torch.cat([prompt_ids, prefix, masks], dim=1)
        abs_start = prompt_ids.shape[1] + span_start
        abs_end = prompt_ids.shape[1] + span_end
        return x_t, abs_start, abs_end
    if mode == "future_masked":
        x_t = torch.cat([prompt_ids, assistant_ids.clone()], dim=1)
        abs_start = prompt_ids.shape[1] + span_start
        abs_end = prompt_ids.shape[1] + span_end
        x_t[:, abs_start:] = mask_id
        return x_t, abs_start, abs_end
    raise ValueError(f"unknown context mode {mode!r}")


def percentile_summary(values):
    if not values:
        return {}
    values = sorted(values)

    def at(frac):
        idx = min(len(values) - 1, max(0, round((len(values) - 1) * frac)))
        return values[idx]

    return {
        "min": values[0],
        "p50": at(0.50),
        "p90": at(0.90),
        "max": values[-1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument(
        "--no-adapter",
        action="store_true",
        help="Evaluate the base converted diffusion model without loading a PEFT adapter.",
    )
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--examples-jsonl", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--cases-jsonl", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--conversation-template", default="fast_dllm_v2")
    parser.add_argument(
        "--context-mode",
        choices=["full_gold", "prefix_only", "future_masked"],
        default="full_gold",
        help=(
            "Context used when scoring candidates: full_gold masks only the span, "
            "prefix_only removes future assistant tokens, future_masked masks the "
            "span and all future assistant tokens."
        ),
    )
    parser.add_argument("--append-instruction", action="store_true")
    parser.add_argument("--no-merge-adapter", action="store_true")
    args = parser.parse_args()

    args.chat_template = resolve_chat_template(args.conversation_template)
    adapter_path = None if args.no_adapter else args.adapter
    model, tokenizer = load_model(
        str(args.base_model),
        str(adapter_path) if adapter_path else None,
        merge_adapter=not args.no_merge_adapter,
        tokenizer_path=str(args.tokenizer_path) if args.tokenizer_path else None,
    )
    mask_id, _ = resolve_token_ids(model, tokenizer)
    cases = load_cases(args.cases_jsonl)
    examples = [
        row
        for row in load_jsonl(args.examples_jsonl)
        if row.get("usable_for_training")
    ]
    if args.limit and args.limit > 0:
        examples = examples[: args.limit]

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    totals = Counter()
    margins = []
    start_time = time.time()
    prompt_cache = {}
    assistant_cache = {}

    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for idx, example in enumerate(examples):
            case = cases.get(example.get("id"))
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
                if not case:
                    raise ValueError(f"missing case for id={example.get('id')!r}")
                candidate_token_ids = example.get("candidate_token_ids") or []
                span_start = int(example["schedule_token_start"])
                span_end = int(example["schedule_token_end"])
                span_len = span_end - span_start
                if not candidate_token_ids or any(len(seq) != span_len for seq in candidate_token_ids):
                    raise ValueError("candidate token ids are missing or not span-length compatible")
                if int(example.get("target_index", -1)) < 0:
                    raise ValueError("target index missing from candidate set")

                cache_key = example.get("id")
                if cache_key not in prompt_cache:
                    prompt = make_prompt(
                        tokenizer,
                        case,
                        args.append_instruction,
                        chat_template=args.chat_template,
                    )
                    prompt_cache[cache_key] = tokenizer([prompt], return_tensors="pt").input_ids.to("cuda")
                    assistant_ids = tokenizer(
                        case.get("gold_assistant") or "",
                        add_special_tokens=False,
                        return_tensors="pt",
                    ).input_ids.to("cuda")
                    assistant_cache[cache_key] = assistant_ids

                prompt_ids = prompt_cache[cache_key]
                assistant_ids = assistant_cache[cache_key]
                x_t, abs_start, abs_end = candidate_context(
                    prompt_ids,
                    assistant_ids,
                    span_start,
                    span_end,
                    mask_id,
                    args.context_mode,
                )
                with torch.no_grad():
                    scores = score_candidate_sequences(
                        model,
                        x_t,
                        abs_start,
                        abs_end,
                        candidate_token_ids,
                    )
                order = rank_order(scores)
                predicted = order[0] if order else -1
                target_idx = int(example["target_index"])
                target_rank = order.index(target_idx) + 1 if target_idx in order else None
                best_score = scores[predicted] if predicted >= 0 else None
                target_score = scores[target_idx]
                margin = float(target_score - max(score for i, score in enumerate(scores) if i != target_idx)) if len(scores) > 1 else 0.0
                margins.append(margin)
                row.update(
                    {
                        "scores": scores,
                        "rank_order": order,
                        "predicted_index": predicted,
                        "predicted_value": example["candidate_values"][predicted] if predicted >= 0 else None,
                        "target_rank": target_rank,
                        "target_score": target_score,
                        "best_score": best_score,
                        "target_margin": margin,
                        "correct": predicted == target_idx,
                        "candidate_count": len(scores),
                        "span_len": span_len,
                        "context_mode": args.context_mode,
                    }
                )
                totals["examples"] += 1
                totals[f"examples:{example['kind']}"] += 1
                totals["correct"] += int(predicted == target_idx)
                totals[f"correct:{example['kind']}"] += int(predicted == target_idx)
                totals["single_candidate_examples"] += int(len(scores) == 1)
                totals[f"single_candidate_examples:{example['kind']}"] += int(len(scores) == 1)
                totals["multi_candidate_examples"] += int(len(scores) > 1)
                totals[f"multi_candidate_examples:{example['kind']}"] += int(len(scores) > 1)
                totals["multi_candidate_correct"] += int(len(scores) > 1 and predicted == target_idx)
                totals[f"multi_candidate_correct:{example['kind']}"] += int(
                    len(scores) > 1 and predicted == target_idx
                )
            except Exception as exc:
                row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
                totals["errors"] += 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"rank {idx + 1}/{len(examples)} ok={totals['examples']} "
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
        "cases_jsonl": str(args.cases_jsonl),
        "out_jsonl": str(args.out_jsonl),
        "conversation_template": args.conversation_template,
        "context_mode": args.context_mode,
        "append_instruction": args.append_instruction,
        "totals": dict(totals),
        "accuracy": totals["correct"] / totals["examples"] if totals["examples"] else 0.0,
        "multi_candidate_accuracy": (
            totals["multi_candidate_correct"] / totals["multi_candidate_examples"]
            if totals["multi_candidate_examples"]
            else 0.0
        ),
        "margin_summary": percentile_summary(margins),
        "elapsed_seconds": elapsed,
        "examples_per_second": totals["examples"] / elapsed if elapsed else 0.0,
        "mask_id": mask_id,
    }
    if torch.cuda.is_available():
        summary["cuda_max_memory_allocated_gib"] = torch.cuda.max_memory_allocated() / (1024**3)
        summary["cuda_max_memory_reserved_gib"] = torch.cuda.max_memory_reserved() / (1024**3)
    args.out_jsonl.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
