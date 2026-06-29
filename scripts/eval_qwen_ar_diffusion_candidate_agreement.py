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

from eval_fastdllm_candidate_index_ranking import (  # noqa: E402
    INDEX_SYSTEM,
    rank_order,
    score_index_candidate as score_fastdllm_index_candidate,
)
from eval_fastdllm_toolcall_cases import (  # noqa: E402
    apply_chat_template,
    load_model as load_fastdllm_model,
    resolve_chat_template,
    resolve_token_ids,
)


DEFAULT_EXAMPLES = ROOT / "data/candidate_ranking/synthetic_multicall_failure_analogue_index_ranking.jsonl"
DEFAULT_OUT = ROOT / "runs/candidate_agreement/qwen35_ar_or_diffusion_scores.jsonl"
DEFAULT_FASTDLLM_BASE = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_FASTDLLM_ADAPTER = (
    ROOT / "runs/fastdllm_qwen35_9b_sequence_value_retention_mix_bd16_from_ckpt275_step10"
    / "checkpoint-5/adapter_model"
)


def local_qwen35_ar_default():
    snapshot = (
        Path.home()
        / ".cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots"
        / "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    )
    return snapshot if snapshot.exists() else Path("Qwen/Qwen3.5-9B")


def load_jsonl(path, limit=0, min_candidates=1):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("usable_for_training", True):
                continue
            if len(row.get("candidate_values") or []) < min_candidates:
                continue
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def percentile_summary(values):
    if not values:
        return {}
    values = sorted(float(v) for v in values)

    def at(frac):
        idx = min(len(values) - 1, max(0, round((len(values) - 1) * frac)))
        return values[idx]

    return {"min": values[0], "p50": at(0.50), "p90": at(0.90), "max": values[-1]}


def example_key(row):
    return "|".join(
        str(row.get(key, ""))
        for key in ("id", "kind", "tool_call_index", "json_key", "target")
    )


def build_prompt(tokenizer, example, chat_template=None):
    messages = [
        {"role": "system", "content": INDEX_SYSTEM},
        {"role": "user", "content": example.get("prompt") or ""},
    ]
    return apply_chat_template(tokenizer, messages, None, chat_template=chat_template)


def load_ar_model(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.torch_dtype]
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "device_map": args.device_map,
        "local_files_only": args.local_files_only,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        args.ar_model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(args.ar_model, **model_kwargs).eval()
    return model, tokenizer


def first_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def score_ar_candidate(model, prompt_ids, candidate_ids):
    device = first_device(model)
    candidate = torch.tensor([candidate_ids], dtype=prompt_ids.dtype, device=device)
    full_ids = torch.cat([prompt_ids.to(device), candidate], dim=1)
    with torch.inference_mode():
        logits = model(input_ids=full_ids, use_cache=False).logits
    start = prompt_ids.shape[1] - 1
    end = start + len(candidate_ids)
    log_probs = torch.log_softmax(logits[:, start:end, :], dim=-1)
    score = torch.zeros((), dtype=log_probs.dtype, device=log_probs.device)
    for pos, token_id in enumerate(candidate_ids):
        score = score + log_probs[0, pos, int(token_id)]
    return float(score.detach().cpu())


def token_ids_for_indices(tokenizer, examples):
    max_candidates = max((len(row.get("candidate_values") or []) for row in examples), default=0)
    token_ids = []
    for idx in range(max_candidates):
        ids = tokenizer(str(idx), add_special_tokens=False).input_ids
        if not ids:
            raise ValueError(f"index {idx} tokenized to empty ids")
        token_ids.append([int(item) for item in ids])
    return token_ids


def score_examples(args, model, tokenizer, model_kind):
    chat_template = resolve_chat_template(args.conversation_template) if args.conversation_template else None
    examples = load_jsonl(args.examples_jsonl, limit=args.limit, min_candidates=args.min_candidates)
    index_token_ids = token_ids_for_indices(tokenizer, examples)
    if model_kind == "fastdllm":
        mask_id, _ = resolve_token_ids(model, tokenizer)
    else:
        mask_id = None

    totals = Counter()
    margins = []
    start_time = time.time()
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for idx, example in enumerate(examples):
            row = {
                "idx": idx,
                "key": example_key(example),
                "id": example.get("id"),
                "source": example.get("source"),
                "analogue_family": example.get("analogue_family"),
                "kind": example.get("kind"),
                "tool_call_index": example.get("tool_call_index"),
                "json_key": example.get("json_key"),
                "target": example.get("target"),
                "target_index": example.get("target_index"),
                "candidate_values": example.get("candidate_values") or [],
                "model_kind": model_kind,
                "status": "ok",
            }
            try:
                target_idx = int(example["target_index"])
                prompt = build_prompt(tokenizer, example, chat_template=chat_template)
                prompt_ids = tokenizer([prompt], return_tensors="pt").input_ids.to(first_device(model))
                scores = []
                for candidate_idx in range(len(row["candidate_values"])):
                    if model_kind == "fastdllm":
                        with torch.inference_mode():
                            score = score_fastdllm_index_candidate(
                                model,
                                prompt_ids,
                                index_token_ids[candidate_idx],
                                mask_id,
                            )
                    else:
                        score = score_ar_candidate(model, prompt_ids, index_token_ids[candidate_idx])
                    scores.append(score)
                order = rank_order(scores)
                predicted = order[0] if order else -1
                margin = (
                    float(scores[target_idx] - max(score for score_idx, score in enumerate(scores) if score_idx != target_idx))
                    if len(scores) > 1 and 0 <= target_idx < len(scores)
                    else 0.0
                )
                margins.append(margin)
                row.update(
                    {
                        "scores": scores,
                        "rank_order": order,
                        "predicted_index": predicted,
                        "predicted_value": row["candidate_values"][predicted] if predicted >= 0 else None,
                        "correct": predicted == target_idx,
                        "target_margin": margin,
                        "candidate_count": len(scores),
                    }
                )
                totals["examples"] += 1
                totals[f"examples:{row['kind']}"] += 1
                totals[f"examples:{row.get('analogue_family')}"] += 1
                totals["correct"] += int(row["correct"])
                totals[f"correct:{row['kind']}"] += int(row["correct"])
                totals[f"correct:{row.get('analogue_family')}"] += int(row["correct"])
            except Exception as exc:
                row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
                totals["errors"] += 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"{model_kind} {idx + 1}/{len(examples)} correct={totals['correct']} "
                f"errors={totals['errors']}",
                flush=True,
            )

    elapsed = time.time() - start_time
    summary = {
        "mode": args.mode,
        "model_kind": model_kind,
        "examples_jsonl": str(args.examples_jsonl),
        "out_jsonl": str(args.out_jsonl),
        "conversation_template": args.conversation_template,
        "min_candidates": args.min_candidates,
        "limit": args.limit,
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


def load_score_rows(path):
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        file_idx = 0
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "ok":
                file_idx += 1
                continue
            semantic_key = row.get("key") or example_key(row)
            rows[f"{row.get('idx', file_idx)}|{semantic_key}"] = row
            file_idx += 1
    return rows


def compare_scores(args):
    ref_rows = load_score_rows(args.reference_scores)
    cand_rows = load_score_rows(args.candidate_scores)
    keys = sorted(set(ref_rows) & set(cand_rows))
    totals = Counter()
    margins = {"reference": [], "candidate": [], "delta": []}
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for idx, key in enumerate(keys):
            ref = ref_rows[key]
            cand = cand_rows[key]
            ref_correct = bool(ref.get("correct"))
            cand_correct = bool(cand.get("correct"))
            agreement = ref.get("predicted_index") == cand.get("predicted_index")
            row = {
                "idx": idx,
                "key": key,
                "id": ref.get("id"),
                "kind": ref.get("kind"),
                "tool_call_index": ref.get("tool_call_index"),
                "json_key": ref.get("json_key"),
                "target": ref.get("target"),
                "target_index": ref.get("target_index"),
                "candidate_values": ref.get("candidate_values") or [],
                "reference_predicted_index": ref.get("predicted_index"),
                "candidate_predicted_index": cand.get("predicted_index"),
                "reference_predicted_value": ref.get("predicted_value"),
                "candidate_predicted_value": cand.get("predicted_value"),
                "reference_correct": ref_correct,
                "candidate_correct": cand_correct,
                "prediction_agreement": agreement,
                "reference_margin": ref.get("target_margin"),
                "candidate_margin": cand.get("target_margin"),
                "margin_delta_candidate_minus_reference": (
                    float(cand.get("target_margin", 0.0)) - float(ref.get("target_margin", 0.0))
                ),
            }
            totals["overlap"] += 1
            totals[f"overlap:{row['kind']}"] += 1
            totals["reference_correct"] += int(ref_correct)
            totals["candidate_correct"] += int(cand_correct)
            totals["prediction_agreement"] += int(agreement)
            totals["candidate_regression_vs_reference"] += int(ref_correct and not cand_correct)
            totals["candidate_improvement_vs_reference"] += int(cand_correct and not ref_correct)
            totals[f"reference_correct:{row['kind']}"] += int(ref_correct)
            totals[f"candidate_correct:{row['kind']}"] += int(cand_correct)
            totals[f"prediction_agreement:{row['kind']}"] += int(agreement)
            totals[f"candidate_regression_vs_reference:{row['kind']}"] += int(ref_correct and not cand_correct)
            totals[f"candidate_improvement_vs_reference:{row['kind']}"] += int(cand_correct and not ref_correct)
            for name, source in [("reference", ref), ("candidate", cand)]:
                if source.get("target_margin") is not None:
                    margins[name].append(float(source["target_margin"]))
            margins["delta"].append(float(row["margin_delta_candidate_minus_reference"]))
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    overlap = totals["overlap"]
    summary = {
        "mode": "compare",
        "reference_scores": str(args.reference_scores),
        "candidate_scores": str(args.candidate_scores),
        "out_jsonl": str(args.out_jsonl),
        "totals": dict(totals),
        "reference_accuracy": totals["reference_correct"] / overlap if overlap else 0.0,
        "candidate_accuracy": totals["candidate_correct"] / overlap if overlap else 0.0,
        "prediction_agreement_rate": totals["prediction_agreement"] / overlap if overlap else 0.0,
        "margin_summary": {key: percentile_summary(value) for key, value in margins.items()},
    }
    args.out_jsonl.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score candidate-index choices with AR or Fast-DLLM models and compare behavior retention."
    )
    parser.add_argument("--mode", choices=["ar", "fastdllm", "fastdllm_causal", "compare"], required=True)
    parser.add_argument("--examples-jsonl", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-candidates", type=int, default=2)
    parser.add_argument("--conversation-template", default="fast_dllm_v2")
    parser.add_argument("--reference-scores", type=Path)
    parser.add_argument("--candidate-scores", type=Path)

    parser.add_argument("--ar-model", default=str(local_qwen35_ar_default()))
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", default="nf4", choices=["nf4", "fp4"])
    parser.add_argument("--bnb-4bit-use-double-quant", action="store_true")
    parser.add_argument("--torch-dtype", default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--fastdllm-base-model", type=Path, default=DEFAULT_FASTDLLM_BASE)
    parser.add_argument("--fastdllm-adapter", type=Path, default=DEFAULT_FASTDLLM_ADAPTER)
    parser.add_argument("--fastdllm-tokenizer-path", type=Path, default=DEFAULT_FASTDLLM_BASE)
    parser.add_argument("--fastdllm-no-adapter", action="store_true")
    parser.add_argument("--fastdllm-no-merge-adapter", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "compare":
        if not args.reference_scores or not args.candidate_scores:
            raise SystemExit("--reference-scores and --candidate-scores are required in compare mode")
        compare_scores(args)
        return

    if args.mode == "ar":
        model, tokenizer = load_ar_model(args)
        score_examples(args, model, tokenizer, "ar")
        return

    adapter = None if args.fastdllm_no_adapter else args.fastdllm_adapter
    model, tokenizer = load_fastdllm_model(
        str(args.fastdllm_base_model),
        str(adapter) if adapter else None,
        merge_adapter=not args.fastdllm_no_merge_adapter,
        tokenizer_path=str(args.fastdllm_tokenizer_path) if args.fastdllm_tokenizer_path else None,
    )
    model_kind = "fastdllm" if args.mode == "fastdllm" else "fastdllm_causal"
    score_examples(args, model, tokenizer, model_kind)


if __name__ == "__main__":
    main()
