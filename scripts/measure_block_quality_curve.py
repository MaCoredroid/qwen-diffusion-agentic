#!/usr/bin/env python3
"""Measure the block-diffusion quality curve for the B@1000 Qwen3.5 adapter.

The sampler here is deliberately fixed-K and full-context:

* append a fresh masked block of size B,
* run exactly K denoise forwards for that block,
* keep a mutable block estimate between steps,
* re-mask lower-confidence positions and re-sample them on later steps,
* finalize the full block after the Kth denoise forward.

This is a measurement harness only; it does not change serving or vLLM code.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from eval_fastdllm_toolcall_cases import make_prompt  # noqa: E402
from eval_flare_stage1_ab_diffusion import (  # noqa: E402
    build_gsm8k_prompt,
    configure_cuda_env,
    gsm8k_gold,
    gsm8k_strict,
    load_model_and_tokenizer,
    read_jsonl,
    resolve_mask_id,
    resolve_stop_token_ids,
    sample_with_top_p,
    set_block_size,
)
from eval_toolcall_jsonl import score_tool_calls  # noqa: E402


DEFAULT_BASE = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_ADAPTER = ROOT / "runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000"
DEFAULT_GSM8K = ROOT / "data/phaseA_retention/gsm8k_main_test_first20.jsonl"
DEFAULT_GSM8K_FEWSHOT = ROOT / "data/phaseA_retention/gsm8k_main_train_first5.jsonl"
DEFAULT_TOOLCALL = ROOT / "runs/planner_decomposition/heldout_seed_multicall_policy_targets.jsonl"
DEFAULT_OUT_DIR = ROOT / "runs/block_quality_curve_b1000"


@dataclass(frozen=True)
class PromptItem:
    slice_name: str
    row_index: int
    item_id: str
    prompt: str
    row: dict[str, Any]


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for piece in str(raw).replace(";", ",").replace(" ", ",").split(","):
        piece = piece.strip()
        if piece:
            values.append(int(piece))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def default_k_values(block_size: int) -> list[int]:
    values = []
    k = int(block_size)
    while k >= 1:
        values.append(k)
        k //= 2
    return values


def expand_sweep(block_sizes: list[int], requested_k: list[int] | None) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for block_size in block_sizes:
        k_values = requested_k if requested_k else default_k_values(block_size)
        for denoise_steps in k_values:
            if denoise_steps < 1:
                continue
            if denoise_steps > block_size:
                continue
            if block_size % denoise_steps != 0:
                continue
            pairs.append((int(block_size), int(denoise_steps)))
    return pairs


def ban_mask_token_logits(logits: torch.Tensor, mask_id: int) -> torch.Tensor:
    logits = logits.clone()
    logits[..., int(mask_id)] = torch.finfo(logits.dtype).min
    return logits


def shifted_full_context_logits(model, sequence: torch.Tensor, active_len: int, mask_id: int) -> torch.Tensor:
    output = model(input_ids=sequence, use_cache=False)
    logits = torch.cat([output.logits[:, :1, :], output.logits[:, :-1, :]], dim=1)
    logits = logits[:, -active_len:, :].float()
    return ban_mask_token_logits(logits, mask_id)


def first_stop_offset(generated: torch.Tensor, stop_token_ids: set[int]) -> int | None:
    for idx, token_id in enumerate(generated.detach().cpu().tolist()):
        if int(token_id) in stop_token_ids:
            return idx
    return None


def visible_count_for_step(block_len: int, denoise_steps: int, step_index: int) -> int:
    if step_index + 1 >= denoise_steps:
        return int(block_len)
    return max(1, min(int(block_len), math.ceil((step_index + 1) * block_len / denoise_steps)))


@torch.inference_mode()
def sample_fixed_k_block_diffusion(
    model,
    input_ids: torch.Tensor,
    *,
    block_size: int,
    denoise_steps: int,
    max_new_tokens: int,
    mask_id: int,
    stop_token_ids: set[int],
    top_p: float,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    output_ids = input_ids.unsqueeze(0).to("cuda")
    original_len = int(output_ids.shape[1])
    metrics: dict[str, Any] = {
        "block_size": int(block_size),
        "denoise_steps_per_block": int(denoise_steps),
        "max_new_tokens": int(max_new_tokens),
        "denoise_forwards": 0,
        "denoise_seconds": 0.0,
        "blocks": [],
        "stop_offset": None,
        "stop_token_id": None,
        "unresolved_masks": 0,
        "visible_schedule": [],
        "mutable_token_changes": 0,
        "remasked_visible_tokens": 0,
    }

    while output_ids.shape[1] - original_len < max_new_tokens:
        remaining = max_new_tokens - (output_ids.shape[1] - original_len)
        block_pad = min(int(block_size), int(remaining))
        block_state = torch.full(
            (output_ids.shape[0], block_pad),
            int(mask_id),
            dtype=torch.long,
            device=output_ids.device,
        )
        visible = torch.zeros_like(block_state, dtype=torch.bool)
        block_metrics: dict[str, Any] = {
            "block_pad": int(block_pad),
            "steps": int(denoise_steps),
            "step_visible": [],
            "step_conf_mean": [],
            "step_conf_min": [],
            "step_conf_max": [],
            "step_seconds": [],
            "mutable_token_changes": 0,
            "remasked_visible_tokens": 0,
        }

        for step_idx in range(int(denoise_steps)):
            x_t = torch.cat([output_ids, block_state], dim=1)
            sync_cuda()
            start = time.perf_counter()
            logits = shifted_full_context_logits(model, x_t, block_pad, mask_id)
            sampled, probs = sample_with_top_p(logits, top_p=top_p, temperature=temperature)
            chosen_probs = torch.gather(probs, dim=-1, index=sampled.unsqueeze(-1)).squeeze(-1)
            sync_cuda()
            seconds = time.perf_counter() - start

            target_visible = visible_count_for_step(block_pad, int(denoise_steps), step_idx)
            topk = torch.topk(chosen_probs, k=target_visible, dim=-1).indices
            next_visible = torch.zeros_like(visible)
            next_visible.scatter_(1, topk, True)

            previous_visible = visible
            previous_state = block_state
            next_state = torch.full_like(block_state, int(mask_id))
            next_state[next_visible] = sampled[next_visible]

            stayed_visible = previous_visible & next_visible
            if bool(stayed_visible.any().item()):
                changed = (previous_state != next_state) & stayed_visible
                changed_count = int(changed.sum().item())
            else:
                changed_count = 0
            remasked_count = int((previous_visible & ~next_visible).sum().item())

            block_state = next_state
            visible = next_visible
            metrics["denoise_forwards"] += 1
            metrics["denoise_seconds"] += seconds
            metrics["mutable_token_changes"] += changed_count
            metrics["remasked_visible_tokens"] += remasked_count
            block_metrics["mutable_token_changes"] += changed_count
            block_metrics["remasked_visible_tokens"] += remasked_count
            block_metrics["step_visible"].append(int(visible.sum().item()))
            block_metrics["step_conf_mean"].append(float(chosen_probs.mean().detach().cpu().item()))
            block_metrics["step_conf_min"].append(float(chosen_probs.min().detach().cpu().item()))
            block_metrics["step_conf_max"].append(float(chosen_probs.max().detach().cpu().item()))
            block_metrics["step_seconds"].append(seconds)

        unresolved = int((block_state == int(mask_id)).sum().item())
        metrics["unresolved_masks"] += unresolved
        output_ids = torch.cat([output_ids, block_state], dim=1)
        metrics["blocks"].append(block_metrics)
        metrics["visible_schedule"].append(block_metrics["step_visible"])

        generated = output_ids[0, original_len:]
        stop_offset = first_stop_offset(generated, stop_token_ids)
        if stop_offset is not None:
            metrics["stop_offset"] = int(stop_offset)
            metrics["stop_token_id"] = int(generated[stop_offset].item())
            output_ids = output_ids[:, : original_len + stop_offset + 1]
            break

    return output_ids[0].detach().cpu(), metrics


@torch.inference_mode()
def measure_ar_cached_step_seconds(
    model,
    prompt_ids: list[torch.Tensor],
    *,
    steps: int,
    mask_id: int,
) -> dict[str, Any]:
    timings = []
    generated_steps = 0
    for ids in prompt_ids:
        input_ids = ids.unsqueeze(0).to("cuda")
        sync_cuda()
        output = model(input_ids=input_ids, use_cache=True)
        sync_cuda()
        past = output.past_key_values
        next_token = output.logits[:, -1:, :].float()
        next_token[..., int(mask_id)] = torch.finfo(next_token.dtype).min
        token = torch.argmax(next_token, dim=-1)
        for _ in range(int(steps)):
            sync_cuda()
            start = time.perf_counter()
            output = model(input_ids=token, use_cache=True, past_key_values=past)
            logits = output.logits[:, -1:, :].float()
            logits[..., int(mask_id)] = torch.finfo(logits.dtype).min
            token = torch.argmax(logits, dim=-1)
            past = output.past_key_values
            sync_cuda()
            timings.append(time.perf_counter() - start)
            generated_steps += 1
    return {
        "rows": len(prompt_ids),
        "steps": generated_steps,
        "mean_step_seconds": statistics.mean(timings) if timings else None,
        "median_step_seconds": statistics.median(timings) if timings else None,
        "min_step_seconds": min(timings) if timings else None,
        "max_step_seconds": max(timings) if timings else None,
    }


def build_prompt_items(tokenizer, args: argparse.Namespace) -> dict[str, list[PromptItem]]:
    fewshot_rows = read_jsonl(args.gsm8k_fewshot_jsonl, args.gsm8k_fewshot)
    gsm_rows = read_jsonl(args.gsm8k_jsonl, args.gsm8k_limit)
    gsm_items = []
    for idx, row in enumerate(gsm_rows):
        prompt = build_gsm8k_prompt(tokenizer, row, fewshot_rows)
        gsm_items.append(
            PromptItem(
                slice_name="gsm8k_first20_strict",
                row_index=idx,
                item_id=f"gsm8k-{row.get('idx', idx)}",
                prompt=prompt,
                row=row,
            )
        )

    tool_rows = read_jsonl(args.toolcall_jsonl, args.toolcall_limit)
    tool_items = []
    for idx, row in enumerate(tool_rows):
        prompt = make_prompt(tokenizer, row, append_instruction=False, chat_template=None)
        tool_items.append(
            PromptItem(
                slice_name="toolcall_heldout12_exact_args",
                row_index=idx,
                item_id=str(row.get("id") or f"toolcall-{idx}"),
                prompt=prompt,
                row=row,
            )
        )
    return {"gsm8k": gsm_items, "toolcall": tool_items}


def score_generated_text(item: PromptItem, generated_text: str) -> dict[str, Any]:
    if item.slice_name.startswith("gsm8k"):
        gold = gsm8k_gold(item.row["answer"])
        pred = gsm8k_strict(generated_text)
        return {
            "metric": "gsm8k_strict",
            "gold": gold,
            "prediction": pred,
            "correct": bool(pred is not None and gold is not None and pred == gold),
        }
    metrics = score_tool_calls(generated_text, item.row.get("tools") or [], item.row.get("gold_assistant"))
    return {
        "metric": "toolcall_exact_args",
        "correct": bool(metrics.get("exact_arguments")),
        "valid_tool_call": bool(metrics.get("valid_tool_call")),
        "exact_arguments": bool(metrics.get("exact_arguments")),
        "exact_tool_name_set": bool(metrics.get("exact_tool_name_set")),
        "exact_tool_name_multiset": bool(metrics.get("exact_tool_name_multiset")),
        "exact_tool_sequence": bool(metrics.get("exact_tool_sequence")),
        "same_tool_call_count": bool(metrics.get("same_tool_call_count")),
        "called_names": metrics.get("called_names"),
        "gold_called_names": metrics.get("gold_called_names"),
        "tool_call_count": metrics.get("tool_call_count"),
        "gold_tool_call_count": metrics.get("gold_tool_call_count"),
        "invalid_tool_call_count": metrics.get("invalid_tool_call_count"),
        "all_schema_valid": bool(metrics.get("all_schema_valid")),
        "all_required_args_present": bool(metrics.get("all_required_args_present")),
    }


def run_slice(
    model,
    tokenizer,
    items: list[PromptItem],
    *,
    block_size: int,
    denoise_steps: int,
    args: argparse.Namespace,
    mask_id: int,
    stop_token_ids: set[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    set_block_size(model, block_size)
    rows = []
    totals = Counter()
    denoise_seconds = []
    denoise_forwards = 0
    generated_tokens = 0
    wall_start = time.perf_counter()
    for item in items:
        input_ids = tokenizer(item.prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
        sample_start = time.perf_counter()
        output_ids, sampler_metrics = sample_fixed_k_block_diffusion(
            model,
            input_ids,
            block_size=block_size,
            denoise_steps=denoise_steps,
            max_new_tokens=args.max_new_tokens,
            mask_id=mask_id,
            stop_token_ids=stop_token_ids,
            top_p=args.top_p,
            temperature=args.temperature,
        )
        sample_seconds = time.perf_counter() - sample_start
        new_ids = output_ids[int(input_ids.numel()) :]
        generated_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        score = score_generated_text(item, generated_text)
        row = {
            "slice": item.slice_name,
            "row_index": item.row_index,
            "id": item.item_id,
            "block_size": int(block_size),
            "denoise_steps": int(denoise_steps),
            "tokens_per_forward_nominal": float(block_size) / float(denoise_steps),
            "prompt_tokens": int(input_ids.numel()),
            "generated_tokens": int(new_ids.numel()),
            "sample_seconds": sample_seconds,
            "generated_text": generated_text,
            "score": score,
            "sampler": sampler_metrics,
        }
        rows.append(row)
        totals["examples"] += 1
        totals["correct"] += int(bool(score.get("correct")))
        totals["generated_tokens"] += int(new_ids.numel())
        totals["unresolved_masks"] += int(sampler_metrics.get("unresolved_masks") or 0)
        totals["mutable_token_changes"] += int(sampler_metrics.get("mutable_token_changes") or 0)
        totals["remasked_visible_tokens"] += int(sampler_metrics.get("remasked_visible_tokens") or 0)
        denoise_forwards += int(sampler_metrics.get("denoise_forwards") or 0)
        generated_tokens += int(new_ids.numel())
        denoise_seconds.append(float(sampler_metrics.get("denoise_seconds") or 0.0))
        print(
            "[row] "
            f"slice={item.slice_name} B={block_size} K={denoise_steps} "
            f"idx={item.row_index} correct={int(bool(score.get('correct')))} "
            f"tokens={int(new_ids.numel())} forwards={int(sampler_metrics.get('denoise_forwards') or 0)}",
            flush=True,
        )

    elapsed = time.perf_counter() - wall_start
    examples = int(totals["examples"])
    summary = {
        "slice": items[0].slice_name if items else "empty",
        "block_size": int(block_size),
        "denoise_steps": int(denoise_steps),
        "tokens_per_forward_nominal": float(block_size) / float(denoise_steps),
        "examples": examples,
        "correct": int(totals["correct"]),
        "accuracy": float(totals["correct"]) / examples if examples else None,
        "generated_tokens": int(generated_tokens),
        "denoise_forwards": int(denoise_forwards),
        "generated_tokens_per_forward_actual": (
            float(generated_tokens) / float(denoise_forwards) if denoise_forwards else None
        ),
        "elapsed_seconds": elapsed,
        "generated_tokens_per_second": float(generated_tokens) / elapsed if elapsed else None,
        "denoise_seconds": float(sum(denoise_seconds)),
        "mean_denoise_seconds_per_forward": (
            float(sum(denoise_seconds)) / float(denoise_forwards) if denoise_forwards else None
        ),
        "unresolved_masks": int(totals["unresolved_masks"]),
        "mutable_token_changes": int(totals["mutable_token_changes"]),
        "remasked_visible_tokens": int(totals["remasked_visible_tokens"]),
    }
    return rows, summary


def add_wallclock_estimates(summary: dict[str, Any], ar_timing: dict[str, Any] | None) -> None:
    ar_step = (ar_timing or {}).get("mean_step_seconds")
    if not ar_step:
        summary["wallclock_vs_cached_ar_estimate"] = None
        return
    diff_step = summary.get("mean_denoise_seconds_per_forward")
    tpf = summary.get("generated_tokens_per_forward_actual") or summary.get("tokens_per_forward_nominal")
    if not diff_step or not tpf:
        summary["wallclock_vs_cached_ar_estimate"] = None
        return
    summary["wallclock_vs_cached_ar_estimate"] = float(tpf) * float(ar_step) / float(diff_step)


def held_quality_max_tpf(
    summaries: list[dict[str, Any]],
    *,
    slice_name: str,
    baseline_by_block: dict[int, float],
    tolerance: float,
) -> dict[str, Any] | None:
    eligible = []
    for row in summaries:
        if row.get("slice") != slice_name:
            continue
        baseline = baseline_by_block.get(int(row["block_size"]))
        if baseline is None:
            continue
        accuracy = row.get("accuracy")
        if accuracy is None:
            continue
        if float(accuracy) + 1e-12 >= float(baseline) - tolerance:
            eligible.append(row)
    if not eligible:
        return None
    best = max(eligible, key=lambda row: (float(row["tokens_per_forward_nominal"]), -int(row["denoise_steps"])))
    return {
        "slice": slice_name,
        "block_size": int(best["block_size"]),
        "denoise_steps": int(best["denoise_steps"]),
        "tokens_per_forward_nominal": float(best["tokens_per_forward_nominal"]),
        "generated_tokens_per_forward_actual": best.get("generated_tokens_per_forward_actual"),
        "accuracy": best.get("accuracy"),
        "baseline_accuracy": baseline_by_block.get(int(best["block_size"])),
        "tolerance": float(tolerance),
        "wallclock_vs_cached_ar_estimate": best.get("wallclock_vs_cached_ar_estimate"),
    }


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Qwen3.5 B@1000 Block Quality Curve",
        "",
        f"Run name: `{payload['run_name']}`",
        f"Anchor gate passed: `{payload['anchor_gate']['passed']}`",
        f"Anchor strict accuracy: `{payload['anchor_gate'].get('accuracy')}`",
        "",
        "## Configuration",
        "",
        f"- Base model: `{payload['args']['base_model']}`",
        f"- Adapter: `{payload['args']['adapter']}`",
        f"- Max new tokens: `{payload['args']['max_new_tokens']}`",
        f"- Temperature: `{payload['args']['temperature']}`",
        f"- Top-p: `{payload['args']['top_p']}`",
        f"- Sampler: mutable-remask fixed-K full-context fresh blocks, mask-token banned",
        "",
        "## AR Timing",
        "",
    ]
    ar_timing = payload.get("ar_timing") or {}
    if ar_timing.get("mean_step_seconds") is None:
        lines.append("- Not measured.")
    else:
        lines.append(f"- Cached AR mean decode-step seconds: `{ar_timing['mean_step_seconds']:.6f}`")
        lines.append(f"- Cached AR timing steps: `{ar_timing['steps']}`")
    lines.extend(["", "## Quality Curve", ""])
    lines.append(
        "| Slice | B | K | Nominal toks/fwd | Actual toks/fwd | Accuracy | Correct | Mean diff fwd s | Wall vs cached AR |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("summaries", []):
        acc = row.get("accuracy")
        actual = row.get("generated_tokens_per_forward_actual")
        mean_diff = row.get("mean_denoise_seconds_per_forward")
        wall = row.get("wallclock_vs_cached_ar_estimate")
        lines.append(
            "| {slice} | {b} | {k} | {nom:.3f} | {actual} | {acc} | {correct}/{examples} | {mean_diff} | {wall} |".format(
                slice=row.get("slice"),
                b=row.get("block_size"),
                k=row.get("denoise_steps"),
                nom=float(row.get("tokens_per_forward_nominal") or 0.0),
                actual="-" if actual is None else f"{float(actual):.3f}",
                acc="-" if acc is None else f"{float(acc):.3f}",
                correct=row.get("correct"),
                examples=row.get("examples"),
                mean_diff="-" if mean_diff is None else f"{float(mean_diff):.4f}",
                wall="-" if wall is None else f"{float(wall):.4f}x",
            )
        )
    lines.extend(["", "## Held-Quality Headline", ""])
    for row in payload.get("held_quality", []):
        if row is None:
            continue
        lines.append(
            "- `{slice}`: B=`{b}`, K=`{k}`, nominal toks/fwd=`{tpf:.3f}`, "
            "accuracy=`{acc:.3f}`, baseline=`{base:.3f}`, wall-vs-cached-AR=`{wall}`".format(
                slice=row["slice"],
                b=row["block_size"],
                k=row["denoise_steps"],
                tpf=float(row["tokens_per_forward_nominal"]),
                acc=float(row["accuracy"]),
                base=float(row["baseline_accuracy"]),
                wall=(
                    "-"
                    if row.get("wallclock_vs_cached_ar_estimate") is None
                    else f"{float(row['wallclock_vs_cached_ar_estimate']):.4f}x"
                ),
            )
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--gsm8k-jsonl", type=Path, default=DEFAULT_GSM8K)
    parser.add_argument("--gsm8k-fewshot-jsonl", type=Path, default=DEFAULT_GSM8K_FEWSHOT)
    parser.add_argument("--toolcall-jsonl", type=Path, default=DEFAULT_TOOLCALL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--block-sizes", type=parse_int_list, default=parse_int_list("8,16,32"))
    parser.add_argument("--k-values", type=parse_int_list, default=None)
    parser.add_argument("--gsm8k-limit", type=int, default=20)
    parser.add_argument("--toolcall-limit", type=int, default=12)
    parser.add_argument("--gsm8k-fewshot", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--anchor-block-size", type=int, default=32)
    parser.add_argument("--anchor-denoise-steps", type=int, default=32)
    parser.add_argument("--anchor-min-strict-accuracy", type=float, default=0.60)
    parser.add_argument("--anchor-only", action="store_true")
    parser.add_argument("--skip-anchor-gate", action="store_true")
    parser.add_argument("--quality-tolerance", type=float, default=0.05)
    parser.add_argument("--ar-timing-rows", type=int, default=4)
    parser.add_argument("--ar-timing-steps", type=int, default=16)
    parser.add_argument("--no-4bit", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_cuda_env()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_name = args.run_name or time.strftime("block_quality_curve_%Y%m%d_%H%M%S")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"{run_name}.jsonl"
    summary_path = out_dir / f"{run_name}.summary.json"
    report_path = out_dir / f"{run_name}.report.md"

    print(f"[load] base={args.base_model} adapter={args.adapter}", flush=True)
    model, tokenizer = load_model_and_tokenizer(args.base_model, args.adapter, four_bit=not args.no_4bit)
    # Resolve generation sentinels from the base config/tokenizer, matching the
    # validation scripts' discipline and avoiding ad hoc hard-coded IDs.
    from transformers import AutoConfig, AutoTokenizer

    base_config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    mask_id = resolve_mask_id(base_config, base_tokenizer, None)
    stop_token_ids_list = resolve_stop_token_ids(base_config, base_tokenizer, None)
    stop_token_ids = set(int(token_id) for token_id in stop_token_ids_list)
    print(
        "[token_ids] "
        + json.dumps({"mask_id": mask_id, "stop_token_ids": sorted(stop_token_ids)}, sort_keys=True),
        flush=True,
    )

    prompt_items = build_prompt_items(tokenizer, args)
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    anchor_gate = {
        "passed": True,
        "required": not args.skip_anchor_gate,
        "accuracy": None,
        "min_accuracy": float(args.anchor_min_strict_accuracy),
        "block_size": int(args.anchor_block_size),
        "denoise_steps": int(args.anchor_denoise_steps),
    }
    if not args.skip_anchor_gate:
        print(
            f"[anchor] B={args.anchor_block_size} K={args.anchor_denoise_steps} "
            f"gsm8k_limit={args.gsm8k_limit}",
            flush=True,
        )
        anchor_rows, anchor_summary = run_slice(
            model,
            tokenizer,
            prompt_items["gsm8k"],
            block_size=args.anchor_block_size,
            denoise_steps=args.anchor_denoise_steps,
            args=args,
            mask_id=mask_id,
            stop_token_ids=stop_token_ids,
        )
        anchor_summary["is_anchor"] = True
        all_rows.extend(anchor_rows)
        summaries.append(anchor_summary)
        anchor_gate["accuracy"] = anchor_summary.get("accuracy")
        anchor_gate["correct"] = anchor_summary.get("correct")
        anchor_gate["examples"] = anchor_summary.get("examples")
        anchor_gate["passed"] = bool(
            anchor_summary.get("accuracy") is not None
            and float(anchor_summary["accuracy"]) + 1e-12 >= float(args.anchor_min_strict_accuracy)
        )
        write_jsonl(rows_path, all_rows)
        write_json(
            summary_path,
            {
                "run_name": run_name,
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "mask_id": mask_id,
                "stop_token_ids": sorted(stop_token_ids),
                "anchor_gate": anchor_gate,
                "summaries": summaries,
                "rows_path": str(rows_path),
            },
        )
        if not anchor_gate["passed"]:
            payload = {
                "run_name": run_name,
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "mask_id": mask_id,
                "stop_token_ids": sorted(stop_token_ids),
                "anchor_gate": anchor_gate,
                "summaries": summaries,
                "held_quality": [],
                "ar_timing": None,
            }
            report_path.write_text(markdown_report(payload), encoding="utf-8")
            print("[anchor] FAILED gate; stopping before speed sweep", flush=True)
            print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
            return 2
        print(
            f"[anchor] PASS strict={anchor_gate['correct']}/{anchor_gate['examples']} "
            f"accuracy={anchor_gate['accuracy']:.4f}",
            flush=True,
        )

    if args.anchor_only:
        payload = {
            "run_name": run_name,
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "mask_id": mask_id,
            "stop_token_ids": sorted(stop_token_ids),
            "anchor_gate": anchor_gate,
            "summaries": summaries,
            "held_quality": [],
            "ar_timing": None,
            "rows_path": str(rows_path),
            "summary_path": str(summary_path),
            "report_path": str(report_path),
        }
        write_jsonl(rows_path, all_rows)
        write_json(summary_path, payload)
        report_path.write_text(markdown_report(payload), encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        return 0

    ar_prompt_ids = []
    for item in (prompt_items["gsm8k"] + prompt_items["toolcall"])[: args.ar_timing_rows]:
        ar_prompt_ids.append(tokenizer(item.prompt, return_tensors="pt", add_special_tokens=False).input_ids[0])
    ar_timing = measure_ar_cached_step_seconds(
        model,
        ar_prompt_ids,
        steps=args.ar_timing_steps,
        mask_id=mask_id,
    )
    print("[ar_timing] " + json.dumps(ar_timing, sort_keys=True), flush=True)

    completed_keys = {
        (row["slice"], int(row["block_size"]), int(row["denoise_steps"]))
        for row in summaries
        if row.get("is_anchor")
    }
    sweep_pairs = expand_sweep(args.block_sizes, args.k_values)
    for block_size, denoise_steps in sweep_pairs:
        for slice_key in ("gsm8k", "toolcall"):
            items = prompt_items[slice_key]
            if not items:
                continue
            slice_name = items[0].slice_name
            key = (slice_name, int(block_size), int(denoise_steps))
            if key in completed_keys:
                print(f"[sweep] reuse anchor slice={slice_name} B={block_size} K={denoise_steps}", flush=True)
                continue
            print(f"[sweep] slice={slice_name} B={block_size} K={denoise_steps}", flush=True)
            rows, summary = run_slice(
                model,
                tokenizer,
                items,
                block_size=block_size,
                denoise_steps=denoise_steps,
                args=args,
                mask_id=mask_id,
                stop_token_ids=stop_token_ids,
            )
            add_wallclock_estimates(summary, ar_timing)
            all_rows.extend(rows)
            summaries.append(summary)
            write_jsonl(rows_path, all_rows)
            write_json(
                summary_path,
                {
                    "run_name": run_name,
                    "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                    "mask_id": mask_id,
                    "stop_token_ids": sorted(stop_token_ids),
                    "anchor_gate": anchor_gate,
                    "ar_timing": ar_timing,
                    "summaries": summaries,
                    "rows_path": str(rows_path),
                },
            )

    for summary in summaries:
        add_wallclock_estimates(summary, ar_timing)

    baseline_by_slice: dict[str, dict[int, float]] = defaultdict(dict)
    for summary in summaries:
        if int(summary.get("block_size") or 0) == int(summary.get("denoise_steps") or -1):
            if summary.get("accuracy") is not None:
                baseline_by_slice[summary["slice"]][int(summary["block_size"])] = float(summary["accuracy"])
    held_quality = [
        held_quality_max_tpf(
            summaries,
            slice_name=slice_name,
            baseline_by_block=baseline,
            tolerance=float(args.quality_tolerance),
        )
        for slice_name, baseline in sorted(baseline_by_slice.items())
    ]

    payload = {
        "run_name": run_name,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "mask_id": mask_id,
        "stop_token_ids": sorted(stop_token_ids),
        "anchor_gate": anchor_gate,
        "ar_timing": ar_timing,
        "summaries": summaries,
        "held_quality": held_quality,
        "rows_path": str(rows_path),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
    }
    write_jsonl(rows_path, all_rows)
    write_json(summary_path, payload)
    report_path.write_text(markdown_report(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
