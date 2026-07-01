#!/usr/bin/env python3
"""Evaluate Countdown sample-and-decode best-of-N.

The experiment is inference-only:
1. sample N constrained Countdown expressions per prompt with the FLARE HF cache;
2. verify each expression against the known Countdown target;
3. count a prompt as solved when any of the N samples verifies.

The score curve is computed from nested prefixes of one max-N rollout so pass@N
is monotonic for each prompt. Throughput is measured in separate batched sweeps
for each N because the wall-clock question is about batching efficiency.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
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
DEFAULT_OUT = ROOT / "runs/countdown_sample_decode_bestofn"

if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from rl_pilot_countdown import (  # noqa: E402
    GpuMonitor,
    build_token_grammar,
    configure_cuda_env,
    constrained_countdown_rollout,
    load_model_and_tokenizer,
    sync_cuda,
    write_json,
)


DATASET_SPECS: dict[str, dict[str, Any]] = {
    "easy3": {
        "label": "easy-3-number Countdown",
        "min_numbers": 3,
        "max_numbers": 3,
        "min_value": 1,
        "max_value": 10,
        "min_target": 3,
        "max_target": 30,
        "max_new_tokens": 24,
    },
    "standard4": {
        "label": "standard 4-number Countdown",
        "min_numbers": 4,
        "max_numbers": 4,
        "min_value": 1,
        "max_value": 20,
        "min_target": 10,
        "max_target": 100,
        "max_new_tokens": 32,
    },
}


def configure_hf_modules_cache(base_model: Path) -> None:
    modeling_py = base_model / "modeling.py"
    digest = "unknown"
    if modeling_py.exists():
        digest = hashlib.sha1(modeling_py.read_bytes()).hexdigest()[:12]
    cache_dir = ROOT / ".hf_modules_cache" / f"countdown_bestofn_{digest}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_MODULES_CACHE"] = str(cache_dir)


def parse_n_values(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("at least one N value is required")
    if any(value <= 0 for value in values):
        raise ValueError(f"N values must be positive, got {values}")
    return sorted(dict.fromkeys(values))


def parse_dataset_names(raw: str) -> list[str]:
    names = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [name for name in names if name not in DATASET_SPECS]
    if unknown:
        raise ValueError(f"unknown datasets {unknown}; available={sorted(DATASET_SPECS)}")
    return names


def make_dataset_entries(spec: dict[str, Any], *, seed: int, size: int):
    import reasoning_gym.games.countdown  # noqa: F401
    from reasoning_gym.factory import create_dataset

    dataset = create_dataset(
        "countdown",
        seed=seed,
        size=size,
        min_numbers=spec["min_numbers"],
        max_numbers=spec["max_numbers"],
        min_value=spec["min_value"],
        max_value=spec["max_value"],
        min_target=spec["min_target"],
        max_target=spec["max_target"],
    )
    return dataset, [dataset[idx] for idx in range(size)]


def rollout_timed(
    model,
    tokenizer,
    dataset,
    entry: dict[str, Any],
    idx: int,
    grammar,
    *,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    use_fast_cache: bool,
    block_size: int,
    generator: torch.Generator,
):
    sync_cuda()
    started = time.perf_counter()
    rollout = constrained_countdown_rollout(
        model,
        tokenizer,
        dataset,
        entry,
        idx,
        grammar,
        group_size=group_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        record_steps=False,
        use_fast_cache=use_fast_cache,
        block_size=block_size,
        generator=generator,
    )
    sync_cuda()
    wall_seconds = time.perf_counter() - started
    return rollout, wall_seconds


def sample_token_count(rollout) -> int:
    return sum(len(token_ids) for token_ids in rollout.token_ids)


def run_score_curve(
    model,
    tokenizer,
    dataset,
    entries: list[dict[str, Any]],
    grammar,
    *,
    n_values: list[int],
    max_new_tokens: int,
    temperature: float,
    use_fast_cache: bool,
    block_size: int,
    generator: torch.Generator,
) -> dict[str, Any]:
    max_n = max(n_values)
    correct_by_n = {n: 0 for n in n_values}
    rg_sum_by_n = {n: 0.0 for n in n_values}
    prompt_rows: list[dict[str, Any]] = []
    total_wall = 0.0
    total_rollout_seconds = 0.0
    total_denoise_forwards = 0
    total_cache_reads = 0
    total_cache_advances = 0
    total_tokens = 0

    for idx, entry in enumerate(entries):
        rollout, wall_seconds = rollout_timed(
            model,
            tokenizer,
            dataset,
            entry,
            idx,
            grammar,
            group_size=max_n,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            use_fast_cache=use_fast_cache,
            block_size=block_size,
            generator=generator,
        )
        strict = [bool(item >= 1.0 - 1e-9) for item in rollout.strict_rewards]
        rg_scores = [float(item) for item in rollout.rg_scores]
        total_wall += wall_seconds
        total_rollout_seconds += float(rollout.seconds)
        total_denoise_forwards += int(rollout.denoise_forwards)
        total_cache_reads += int(rollout.cache_read_calls)
        total_cache_advances += int(rollout.cache_advance_calls)
        total_tokens += sample_token_count(rollout)

        prefix_rows: dict[str, Any] = {}
        for n in n_values:
            prefix_strict = strict[:n]
            prefix_rg = rg_scores[:n]
            hit = any(prefix_strict)
            correct_by_n[n] += int(hit)
            rg_sum_by_n[n] += max(prefix_rg) if prefix_rg else 0.0
            selected_index = next((sample_idx for sample_idx, ok in enumerate(prefix_strict) if ok), None)
            if selected_index is None and prefix_rg:
                selected_index = max(range(len(prefix_rg)), key=lambda sample_idx: prefix_rg[sample_idx])
            prefix_rows[str(n)] = {
                "hit": hit,
                "selected_index": selected_index,
                "selected_expression": None
                if selected_index is None
                else rollout.expressions[int(selected_index)],
                "best_reasoning_gym_score": max(prefix_rg) if prefix_rg else 0.0,
            }

        prompt_rows.append(
            {
                "idx": idx,
                "target": entry["metadata"]["target"],
                "numbers": entry["metadata"]["numbers"],
                "gold": entry["answer"],
                "wall_seconds": wall_seconds,
                "rollout_seconds": rollout.seconds,
                "denoise_forwards": rollout.denoise_forwards,
                "cache_read_calls": rollout.cache_read_calls,
                "cache_advance_calls": rollout.cache_advance_calls,
                "samples": [
                    {
                        "sample_idx": sample_idx,
                        "expression": expression,
                        "reasoning_gym_score": float(rollout.rg_scores[sample_idx]),
                        "strict": bool(rollout.strict_rewards[sample_idx] >= 1.0 - 1e-9),
                        "graded_reward": float(rollout.rewards[sample_idx]),
                        "tokens": len(rollout.token_ids[sample_idx]),
                    }
                    for sample_idx, expression in enumerate(rollout.expressions)
                ],
                "prefix_best_of_n": prefix_rows,
            }
        )

    n_prompts = len(entries)
    curve = {
        str(n): {
            "strict_correct": correct_by_n[n],
            "strict_accuracy": correct_by_n[n] / n_prompts if n_prompts else 0.0,
            "pass_at_n": correct_by_n[n] / n_prompts if n_prompts else 0.0,
            "mean_best_reasoning_gym_score": rg_sum_by_n[n] / n_prompts if n_prompts else 0.0,
        }
        for n in n_values
    }
    return {
        "examples": n_prompts,
        "max_n": max_n,
        "temperature": temperature,
        "score_method": "nested prefixes of one max-N constrained sample-and-decode rollout per prompt",
        "curve": curve,
        "seconds": total_wall,
        "rollout_seconds": total_rollout_seconds,
        "total_samples": n_prompts * max_n,
        "total_expression_tokens": total_tokens,
        "samples_per_second": (n_prompts * max_n) / total_wall if total_wall > 0 else None,
        "expression_tokens_per_second": total_tokens / total_wall if total_wall > 0 else None,
        "denoise_forwards": total_denoise_forwards,
        "cache_read_calls": total_cache_reads,
        "cache_advance_calls": total_cache_advances,
        "rows": prompt_rows,
    }


def run_throughput_sweep(
    model,
    tokenizer,
    dataset,
    entries: list[dict[str, Any]],
    grammar,
    *,
    n_values: list[int],
    max_new_tokens: int,
    temperature: float,
    use_fast_cache: bool,
    block_size: int,
    ar_tok_s: float,
    generator: torch.Generator,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for n in n_values:
        total_wall = 0.0
        total_rollout_seconds = 0.0
        total_tokens = 0
        total_denoise_forwards = 0
        total_cache_reads = 0
        total_cache_advances = 0
        strict_hits = 0
        for idx, entry in enumerate(entries):
            rollout, wall_seconds = rollout_timed(
                model,
                tokenizer,
                dataset,
                entry,
                idx,
                grammar,
                group_size=n,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                use_fast_cache=use_fast_cache,
                block_size=block_size,
                generator=generator,
            )
            total_wall += wall_seconds
            total_rollout_seconds += float(rollout.seconds)
            total_tokens += sample_token_count(rollout)
            total_denoise_forwards += int(rollout.denoise_forwards)
            total_cache_reads += int(rollout.cache_read_calls)
            total_cache_advances += int(rollout.cache_advance_calls)
            strict_hits += int(any(item >= 1.0 - 1e-9 for item in rollout.strict_rewards))

        total_samples = len(entries) * n
        samples_per_second = total_samples / total_wall if total_wall > 0 else None
        expression_tokens_per_second = total_tokens / total_wall if total_wall > 0 else None
        rows[str(n)] = {
            "n": n,
            "examples": len(entries),
            "total_samples": total_samples,
            "seconds": total_wall,
            "rollout_seconds": total_rollout_seconds,
            "seconds_per_prompt": total_wall / len(entries) if entries else None,
            "samples_per_second": samples_per_second,
            "expression_tokens_per_second": expression_tokens_per_second,
            "useful_token_throughput_vs_ar": None
            if expression_tokens_per_second is None
            else expression_tokens_per_second / float(ar_tok_s),
            "mean_expression_tokens_per_sample": total_tokens / total_samples if total_samples else 0.0,
            "strict_correct_observed": strict_hits,
            "strict_accuracy_observed": strict_hits / len(entries) if entries else 0.0,
            "denoise_forwards": total_denoise_forwards,
            "cache_read_calls": total_cache_reads,
            "cache_advance_calls": total_cache_advances,
        }

    n1 = rows.get("1")
    if n1 and n1["seconds_per_prompt"]:
        n1_seconds_per_prompt = float(n1["seconds_per_prompt"])
        n1_samples_per_second = float(n1["samples_per_second"])
        for key, row in rows.items():
            n = int(row["n"])
            seconds_per_prompt = float(row["seconds_per_prompt"])
            row["wall_multiplier_vs_n1"] = seconds_per_prompt / n1_seconds_per_prompt
            row["batched_speedup_vs_sequential_n1"] = (
                (n * n1_seconds_per_prompt) / seconds_per_prompt if seconds_per_prompt > 0 else None
            )
            row["sample_throughput_multiplier_vs_n1"] = (
                float(row["samples_per_second"]) / n1_samples_per_second
                if n1_samples_per_second > 0
                else None
            )
    return {
        "examples": len(entries),
        "temperature": temperature,
        "ar_single_stream_tok_s_baseline": ar_tok_s,
        "rows": rows,
    }


def make_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Countdown sample-and-decode best-of-N")
    lines.append("")
    lines.append("Inference-only verifier: Countdown target check; no gold expression is used for selection.")
    lines.append("")
    for dataset_name, result in summary["datasets"].items():
        lines.append(f"## {result['label']}")
        lines.append("")
        lines.append("| N | pass@N | correct | mean best RG score |")
        lines.append("| ---: | ---: | ---: | ---: |")
        examples = result["score_curve"]["examples"]
        for n, row in result["score_curve"]["curve"].items():
            lines.append(
                f"| {n} | {row['pass_at_n']:.4f} | "
                f"{row['strict_correct']}/{examples} | {row['mean_best_reasoning_gym_score']:.4f} |"
            )
        lines.append("")
        lines.append("| N | samples/sec | expr tok/sec | vs 89 tok/s AR | wall x N=1 | batched speedup vs sequential N=1 |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
        for n, row in result["throughput"]["rows"].items():
            lines.append(
                f"| {n} | {row['samples_per_second']:.3f} | "
                f"{row['expression_tokens_per_second']:.1f} | "
                f"{row['useful_token_throughput_vs_ar']:.2f}x | "
                f"{row.get('wall_multiplier_vs_n1', 1.0):.2f}x | "
                f"{row.get('batched_speedup_vs_sequential_n1', 1.0):.2f}x |"
            )
        lines.append("")
    lines.append("No promotion decision is made by this script.")
    lines.append("")
    return "\n".join(lines)


def evaluate(args) -> dict[str, Any]:
    configure_cuda_env()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Countdown best-of-N eval")
    torch.cuda.set_device(args.gpu_index)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    configure_hf_modules_cache(Path(args.base_model))

    n_values = parse_n_values(args.n_values)
    dataset_names = parse_dataset_names(args.datasets)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, mask_id, stop_ids = load_model_and_tokenizer(args)
    model.eval()
    grammar = build_token_grammar(tokenizer, mask_id, stop_ids)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)

    config = {
        "args": vars(args),
        "n_values": n_values,
        "datasets": dataset_names,
        "mask_id": mask_id,
        "stop_ids": stop_ids,
        "token_grammar": grammar.char_to_id,
        "serving_forward": "RequestDiffusionState cached route_i FLARE noisy forward"
        if args.use_fast_serving_cache
        else "cache_off_full_context_model_forward",
        "verifier": "reasoning_gym Countdown score_answer(expression, entry) target check",
        "selection": "best-of-N keeps the first verified-correct expression in the prefix, else records best RG score",
        "promotion": "none; inference-only constrained lane measurement",
    }
    write_json(out_dir / "config.json", config)
    print("[config] " + json.dumps(config, sort_keys=True), flush=True)

    if args.warmup:
        warmup_name = dataset_names[0]
        warmup_spec = DATASET_SPECS[warmup_name]
        warmup_ds, warmup_entries = make_dataset_entries(warmup_spec, seed=args.eval_seed, size=1)
        print("[warmup] starting", flush=True)
        rollout_timed(
            model,
            tokenizer,
            warmup_ds,
            warmup_entries[0],
            0,
            grammar,
            group_size=min(max(n_values), max(1, args.warmup_group_size)),
            max_new_tokens=warmup_spec["max_new_tokens"],
            temperature=args.temperature,
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
        for dataset_offset, dataset_name in enumerate(dataset_names):
            spec = DATASET_SPECS[dataset_name]
            score_size = int(args.score_size)
            throughput_size = int(args.throughput_size or args.score_size)
            size = max(score_size, throughput_size)
            seed = int(args.eval_seed) + dataset_offset * int(args.dataset_seed_stride)
            dataset, entries = make_dataset_entries(spec, seed=seed, size=size)
            score_entries = entries[:score_size]
            throughput_entries = entries[:throughput_size]

            print(
                f"[score] dataset={dataset_name} prompts={len(score_entries)} max_n={max(n_values)}",
                flush=True,
            )
            score_curve = run_score_curve(
                model,
                tokenizer,
                dataset,
                score_entries,
                grammar,
                n_values=n_values,
                max_new_tokens=spec["max_new_tokens"],
                temperature=args.temperature,
                use_fast_cache=args.use_fast_serving_cache,
                block_size=args.block_size,
                generator=generator,
            )
            write_json(out_dir / f"{dataset_name}_score_curve.json", score_curve)
            print(
                "[score] "
                + dataset_name
                + " "
                + " ".join(
                    f"N={n}:{score_curve['curve'][str(n)]['strict_correct']}/{score_curve['examples']}"
                    for n in n_values
                ),
                flush=True,
            )

            print(
                f"[throughput] dataset={dataset_name} prompts={len(throughput_entries)} "
                f"n_values={n_values}",
                flush=True,
            )
            throughput = run_throughput_sweep(
                model,
                tokenizer,
                dataset,
                throughput_entries,
                grammar,
                n_values=n_values,
                max_new_tokens=spec["max_new_tokens"],
                temperature=args.temperature,
                use_fast_cache=args.use_fast_serving_cache,
                block_size=args.block_size,
                ar_tok_s=args.ar_tok_s,
                generator=generator,
            )
            write_json(out_dir / f"{dataset_name}_throughput.json", throughput)
            for n in n_values:
                row = throughput["rows"][str(n)]
                print(
                    f"[throughput] {dataset_name} N={n} "
                    f"samples_s={row['samples_per_second']:.3f} "
                    f"tok_s={row['expression_tokens_per_second']:.1f} "
                    f"vs_ar={row['useful_token_throughput_vs_ar']:.2f}x",
                    flush=True,
                )

            dataset_results[dataset_name] = {
                "label": spec["label"],
                "seed": seed,
                "config": spec,
                "score_curve": score_curve,
                "throughput": throughput,
            }

    total_seconds = time.perf_counter() - started
    summary = {
        "output_dir": str(out_dir),
        "total_seconds": total_seconds,
        "temperature": float(args.temperature),
        "block_size": int(args.block_size),
        "use_fast_serving_cache": bool(args.use_fast_serving_cache),
        "score_size": int(args.score_size),
        "throughput_size": int(args.throughput_size or args.score_size),
        "ar_single_stream_tok_s_baseline": float(args.ar_tok_s),
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
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--eval-seed", type=int, default=2000)
    parser.add_argument("--dataset-seed-stride", type=int, default=10000)
    parser.add_argument("--datasets", default="easy3,standard4")
    parser.add_argument("--score-size", type=int, default=16)
    parser.add_argument("--throughput-size", type=int, default=16)
    parser.add_argument("--n-values", default="1,2,4,8,16")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--ar-tok-s", type=float, default=89.0)
    parser.add_argument("--use-fast-serving-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup-group-size", type=int, default=4)
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
