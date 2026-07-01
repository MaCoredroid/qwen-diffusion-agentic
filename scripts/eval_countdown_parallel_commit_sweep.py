#!/usr/bin/env python3
"""Measure constrained Countdown multi-token commit quality/speed tradeoff."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_ADAPTER = ROOT / "runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000"
DEFAULT_OUT = ROOT / "runs/countdown_parallel_commit_sweep"

if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from eval_countdown_sample_decode_bestofn import (  # noqa: E402
    DATASET_SPECS,
    configure_hf_modules_cache,
    make_dataset_entries,
    parse_dataset_names,
)
from rl_pilot_countdown import (  # noqa: E402
    GpuMonitor,
    build_token_grammar,
    configure_cuda_env,
    constrained_countdown_rollout,
    load_model_and_tokenizer,
    sync_cuda,
    write_json,
)


def parse_thresholds(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("at least one commit threshold is required")
    for value in values:
        if value < 0.0 or value > 1.0:
            raise ValueError(f"commit thresholds must be in [0, 1], got {values}")
    return values


def sample_token_count(rollout) -> int:
    return sum(len(token_ids) for token_ids in rollout.token_ids)


def evaluate_baseline_or_tau(
    model,
    tokenizer,
    dataset,
    entries: list[dict[str, Any]],
    grammar,
    *,
    tau: float | None,
    max_new_tokens: int,
    use_fast_cache: bool,
    block_size: int,
    generator: torch.Generator,
) -> dict[str, Any]:
    strict_correct = 0
    rg_sum = 0.0
    committed_tokens = 0
    denoise_forwards = 0
    cache_reads = 0
    cache_advances = 0
    rows: list[dict[str, Any]] = []

    sync_cuda()
    started = time.perf_counter()
    for idx, entry in enumerate(entries):
        rollout = constrained_countdown_rollout(
            model,
            tokenizer,
            dataset,
            entry,
            idx,
            grammar,
            group_size=1,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            record_steps=False,
            use_fast_cache=use_fast_cache,
            block_size=block_size,
            multi_commit=tau is not None,
            commit_threshold=1.0 if tau is None else float(tau),
            generator=generator,
        )
        tokens = sample_token_count(rollout)
        forwards = int(rollout.denoise_forwards)
        committed_tokens += tokens
        denoise_forwards += forwards
        cache_reads += int(rollout.cache_read_calls)
        cache_advances += int(rollout.cache_advance_calls)
        strict = bool(rollout.strict_rewards[0] >= 1.0 - 1e-9)
        strict_correct += int(strict)
        rg_score = float(rollout.rg_scores[0])
        rg_sum += rg_score
        rows.append(
            {
                "idx": idx,
                "target": entry["metadata"]["target"],
                "numbers": entry["metadata"]["numbers"],
                "gold": entry["answer"],
                "expression": rollout.expressions[0],
                "reasoning_gym_score": rg_score,
                "strict": strict,
                "tokens": tokens,
                "denoise_forwards": forwards,
                "tokens_per_forward": tokens / forwards if forwards > 0 else None,
                "cache_read_calls": rollout.cache_read_calls,
                "cache_advance_calls": rollout.cache_advance_calls,
            }
        )
    sync_cuda()
    seconds = time.perf_counter() - started
    n = len(entries)
    return {
        "mode": "single_token_baseline" if tau is None else "multi_commit",
        "tau": tau,
        "examples": n,
        "strict_correct": strict_correct,
        "strict_accuracy": strict_correct / n if n else 0.0,
        "mean_reasoning_gym_score": rg_sum / n if n else 0.0,
        "committed_tokens": committed_tokens,
        "denoise_forwards": denoise_forwards,
        "tokens_per_forward": committed_tokens / denoise_forwards if denoise_forwards > 0 else None,
        "seconds": seconds,
        "samples_per_second": n / seconds if seconds > 0 else None,
        "committed_tokens_per_second": committed_tokens / seconds if seconds > 0 else None,
        "cache_read_calls": cache_reads,
        "cache_advance_calls": cache_advances,
        "rows": rows,
    }


def held_quality_headline(dataset_result: dict[str, Any]) -> dict[str, Any]:
    baseline = dataset_result["baseline"]
    baseline_correct = int(baseline["strict_correct"])
    baseline_rg = float(baseline["mean_reasoning_gym_score"])
    candidates = []
    for row in dataset_result["tau_rows"]:
        if int(row["strict_correct"]) >= baseline_correct and float(row["mean_reasoning_gym_score"]) >= baseline_rg - 1e-12:
            candidates.append(row)
    if not candidates:
        return {
            "held_quality": False,
            "baseline_strict_correct": baseline_correct,
            "baseline_mean_reasoning_gym_score": baseline_rg,
            "best_tau": None,
            "tokens_per_forward": None,
        }
    best = max(candidates, key=lambda item: float(item["tokens_per_forward"] or 0.0))
    return {
        "held_quality": True,
        "baseline_strict_correct": baseline_correct,
        "baseline_mean_reasoning_gym_score": baseline_rg,
        "best_tau": best["tau"],
        "tokens_per_forward": best["tokens_per_forward"],
        "strict_correct": best["strict_correct"],
        "mean_reasoning_gym_score": best["mean_reasoning_gym_score"],
    }


def make_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Countdown constrained parallel-commit sweep")
    lines.append("")
    lines.append("Greedy constrained decoding; first token per forward preserves baseline progress, additional same-forward commits require confidence > tau.")
    lines.append("")
    for dataset_name, result in summary["datasets"].items():
        lines.append(f"## {result['label']}")
        lines.append("")
        baseline = result["baseline"]
        lines.append(
            f"Baseline one-token/forward: tokens/forward {baseline['tokens_per_forward']:.3f}, "
            f"strict {baseline['strict_correct']}/{baseline['examples']}, "
            f"mean RG {baseline['mean_reasoning_gym_score']:.4f}."
        )
        lines.append("")
        lines.append("| tau | tokens/forward | strict pass | mean RG | committed tokens | forwards |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in result["tau_rows"]:
            lines.append(
                f"| {row['tau']:.2f} | {row['tokens_per_forward']:.3f} | "
                f"{row['strict_correct']}/{row['examples']} | "
                f"{row['mean_reasoning_gym_score']:.4f} | "
                f"{row['committed_tokens']} | {row['denoise_forwards']} |"
            )
        headline = result["held_quality_headline"]
        if headline["held_quality"]:
            lines.append("")
            lines.append(
                f"Held-quality headline: tau={headline['best_tau']:.2f} gives "
                f"{headline['tokens_per_forward']:.3f} tokens/forward at "
                f"{headline['strict_correct']}/{baseline['examples']} strict, "
                f"mean RG {headline['mean_reasoning_gym_score']:.4f}."
            )
        else:
            lines.append("")
            lines.append("Held-quality headline: no tau matched the one-token/forward baseline.")
        lines.append("")
    lines.append("No promotion decision is made by this script.")
    lines.append("")
    return "\n".join(lines)


def evaluate(args) -> dict[str, Any]:
    configure_cuda_env()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the parallel-commit sweep")
    torch.cuda.set_device(args.gpu_index)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    configure_hf_modules_cache(Path(args.base_model))

    dataset_names = parse_dataset_names(args.datasets)
    thresholds = parse_thresholds(args.commit_thresholds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, mask_id, stop_ids = load_model_and_tokenizer(args)
    model.eval()
    grammar = build_token_grammar(tokenizer, mask_id, stop_ids)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)

    config = {
        "args": vars(args),
        "datasets": dataset_names,
        "commit_thresholds": thresholds,
        "mask_id": mask_id,
        "stop_ids": stop_ids,
        "token_grammar": grammar.char_to_id,
        "decode_policy": "greedy temperature=0; one baseline token per forward, extra same-forward tokens require allowed-softmax confidence > tau",
        "serving_forward": "RequestDiffusionState cached route_i FLARE noisy forward"
        if args.use_fast_serving_cache
        else "cache_off_full_context_model_forward",
        "promotion": "none; constrained decoder tradeoff measurement",
    }
    write_json(out_dir / "config.json", config)
    print("[config] " + json.dumps(config, sort_keys=True), flush=True)

    if args.warmup:
        warmup_spec = DATASET_SPECS[dataset_names[0]]
        warmup_ds, warmup_entries = make_dataset_entries(warmup_spec, seed=args.eval_seed, size=1)
        print("[warmup] starting", flush=True)
        _ = evaluate_baseline_or_tau(
            model,
            tokenizer,
            warmup_ds,
            warmup_entries,
            grammar,
            tau=max(thresholds),
            max_new_tokens=warmup_spec["max_new_tokens"],
            use_fast_cache=args.use_fast_serving_cache,
            block_size=args.block_size,
            generator=generator,
        )
        print("[warmup] done", flush=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    dataset_results: dict[str, Any] = {}
    with GpuMonitor(args.gpu_index, interval=args.gpu_monitor_interval) as monitor:
        for dataset_name in dataset_names:
            spec = DATASET_SPECS[dataset_name]
            dataset, entries = make_dataset_entries(spec, seed=args.eval_seed, size=args.eval_size)
            print(f"[baseline] dataset={dataset_name} prompts={len(entries)}", flush=True)
            baseline = evaluate_baseline_or_tau(
                model,
                tokenizer,
                dataset,
                entries,
                grammar,
                tau=None,
                max_new_tokens=spec["max_new_tokens"],
                use_fast_cache=args.use_fast_serving_cache,
                block_size=args.block_size,
                generator=generator,
            )
            print(
                f"[baseline] {dataset_name} tpf={baseline['tokens_per_forward']:.3f} "
                f"strict={baseline['strict_correct']}/{baseline['examples']} "
                f"rg={baseline['mean_reasoning_gym_score']:.4f}",
                flush=True,
            )

            tau_rows = []
            for tau in thresholds:
                print(f"[tau] dataset={dataset_name} tau={tau}", flush=True)
                row = evaluate_baseline_or_tau(
                    model,
                    tokenizer,
                    dataset,
                    entries,
                    grammar,
                    tau=tau,
                    max_new_tokens=spec["max_new_tokens"],
                    use_fast_cache=args.use_fast_serving_cache,
                    block_size=args.block_size,
                    generator=generator,
                )
                tau_rows.append(row)
                print(
                    f"[tau] {dataset_name} tau={tau:.2f} "
                    f"tpf={row['tokens_per_forward']:.3f} "
                    f"strict={row['strict_correct']}/{row['examples']} "
                    f"rg={row['mean_reasoning_gym_score']:.4f}",
                    flush=True,
                )

            dataset_results[dataset_name] = {
                "label": spec["label"],
                "seed": args.eval_seed,
                "config": spec,
                "baseline": baseline,
                "tau_rows": tau_rows,
            }
            dataset_results[dataset_name]["held_quality_headline"] = held_quality_headline(dataset_results[dataset_name])
            write_json(out_dir / f"{dataset_name}_sweep.json", dataset_results[dataset_name])

    total_seconds = time.perf_counter() - started
    summary = {
        "output_dir": str(out_dir),
        "total_seconds": total_seconds,
        "block_size": int(args.block_size),
        "eval_size": int(args.eval_size),
        "eval_seed": int(args.eval_seed),
        "use_fast_serving_cache": bool(args.use_fast_serving_cache),
        "gpu": monitor.summary(),
        "cuda_peak_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
        "cuda_peak_reserved_gb": torch.cuda.max_memory_reserved() / (1024**3),
        "datasets": dataset_results,
        "promotion": "none",
    }
    write_json(out_dir / "summary.json", summary)
    (out_dir / "summary.md").write_text(make_markdown(summary), encoding="utf-8")
    print("[summary] " + json.dumps({k: v for k, v in summary.items() if k != "datasets"}, sort_keys=True), flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=str(DEFAULT_BASE))
    parser.add_argument("--adapter-in", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--eval-seed", type=int, default=2000)
    parser.add_argument("--eval-size", type=int, default=16)
    parser.add_argument("--datasets", default="easy3,standard4")
    parser.add_argument("--commit-thresholds", default="0.99,0.95,0.9,0.8,0.7,0.5")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--use-fast-serving-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-monitor-interval", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
