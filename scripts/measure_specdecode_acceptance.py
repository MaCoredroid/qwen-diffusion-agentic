#!/usr/bin/env python3
"""Measure lossless self-speculative decode acceptance in the HF FLARE stack.

Draft = one route_i FLARE diffusion masked-block forward.
Verify = one clean causal AR forward over prefix + draft.

The emitted sequence is lossless by construction: accept the longest draft
prefix that matches the AR verifier, and on the first mismatch emit the AR
token instead.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from eval_fastdllm_toolcall_cases import (  # noqa: E402
    denoise_logits_for_mode,
    load_model,
    make_prompt,
    resolve_token_ids,
)


DEFAULT_BASE = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_ADAPTER = ROOT / "runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000"
DEFAULT_TOOLCALL = ROOT / "runs/planner_decomposition/heldout_seed_multicall_policy_targets.jsonl"
DEFAULT_GSM8K = ROOT / "data/phaseA_retention/gsm8k_main_test_first20.jsonl"
DEFAULT_GSM8K_FEWSHOT = ROOT / "data/phaseA_retention/gsm8k_main_train_first5.jsonl"
DEFAULT_OUT_DIR = ROOT / "runs/specdecode_acceptance_b1000"


def configure_env() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["FASTDLLM_FLARE_GDN_ROUTE"] = "route_i"
    os.environ.setdefault("FASTDLLM_FLARE_TWO_STREAM", "1")
    os.environ.setdefault("FLARE_TWO_STREAM", "1")
    venv_root = Path(sys.executable).resolve().parents[1]
    cuda_root = (
        venv_root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "nvidia"
        / "cu13"
    )
    if cuda_root.exists():
        os.environ.setdefault("CUDA_HOME", str(cuda_root))
        os.environ["PATH"] = f"{cuda_root / 'bin'}:{os.environ.get('PATH', '')}"
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_root / 'lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}"


def read_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for item in str(raw).replace(";", ",").replace(" ", ",").split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise argparse.ArgumentTypeError("integer list cannot be empty")
    return values


def apply_chat_template(tokenizer, messages: list[dict[str, str]], **kwargs):
    kwargs = dict(kwargs)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def make_gsm8k_prompt(tokenizer, row: dict[str, Any], fewshot_rows: list[dict[str, Any]], mode: str) -> str:
    if mode == "final_only":
        messages = [
            {
                "role": "user",
                "content": (
                    "Solve this grade-school math problem. "
                    "Return only the final numeric answer, with no explanation.\n\n"
                    f"{row['question']}"
                ),
            }
        ]
        return apply_chat_template(tokenizer, messages, tokenize=False, add_generation_prompt=True)

    messages: list[dict[str, str]] = []
    for shot in fewshot_rows:
        messages.append({"role": "user", "content": f"Question: {shot['question']}\nAnswer:"})
        messages.append({"role": "assistant", "content": shot["answer"]})
    messages.append({"role": "user", "content": f"Question: {row['question']}\nAnswer:"})
    return apply_chat_template(tokenizer, messages, tokenize=False, add_generation_prompt=True)


def build_prompt_rows(tokenizer, args: argparse.Namespace) -> list[dict[str, Any]]:
    prompt_rows: list[dict[str, Any]] = []
    tool_rows = read_jsonl(args.toolcall_jsonl, args.toolcall_limit)
    for idx, row in enumerate(tool_rows):
        prompt = make_prompt(tokenizer, row, append_instruction=False, chat_template=None)
        prompt_rows.append(
            {
                "slice": "toolcall_heldout12",
                "row_index": idx,
                "id": row.get("id") or f"toolcall_{idx}",
                "prompt": prompt,
                "source": str(args.toolcall_jsonl),
            }
        )

    fewshot_rows = read_jsonl(args.gsm8k_fewshot_jsonl, 0) if args.gsm8k_prompt_mode == "phasea_fewshot" else []
    gsm_rows = read_jsonl(args.gsm8k_jsonl, args.gsm8k_limit)
    for idx, row in enumerate(gsm_rows):
        prompt = make_gsm8k_prompt(tokenizer, row, fewshot_rows, args.gsm8k_prompt_mode)
        prompt_rows.append(
            {
                "slice": f"gsm8k_{args.gsm8k_prompt_mode}",
                "row_index": idx,
                "id": row.get("idx", idx),
                "prompt": prompt,
                "source": str(args.gsm8k_jsonl),
            }
        )
    return prompt_rows


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def shifted_ar_logits_for_next_tokens(model, sequence: torch.Tensor, start_pos: int, count: int, mask_id: int | None, ban_mask: bool) -> torch.Tensor:
    if start_pos <= 0:
        raise ValueError("start_pos must be > 0 so the verifier has a previous token")
    output = model(input_ids=sequence, use_cache=False)
    logits = output.logits[:, start_pos - 1 : start_pos - 1 + count, :].float()
    if ban_mask and mask_id is not None:
        logits = ban_mask_token_logits(logits, int(mask_id))
    return logits


def ban_mask_token_logits(logits: torch.Tensor, mask_id: int) -> torch.Tensor:
    logits = logits.clone()
    logits[..., int(mask_id)] = torch.finfo(logits.dtype).min
    return logits


def draft_tokens(model, prefix: torch.Tensor, k: int, denoise_args: SimpleNamespace) -> tuple[torch.Tensor, float]:
    masks = torch.full(
        (prefix.shape[0], k),
        int(denoise_args.mask_id),
        dtype=prefix.dtype,
        device=prefix.device,
    )
    x_t = torch.cat([prefix, masks], dim=1)
    cuda_sync()
    start = time.perf_counter()
    logits = denoise_logits_for_mode(model, x_t, denoise_args)[:, -k:, :].float()
    logits = ban_mask_token_logits(logits, int(denoise_args.mask_id))
    tokens = torch.argmax(logits, dim=-1)
    cuda_sync()
    return tokens, time.perf_counter() - start


def ar_greedy_reference(
    model,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    stop_token_ids: set[int],
    mask_id: int,
    ban_mask: bool,
) -> torch.Tensor:
    output_ids = prompt_ids.clone()
    for _ in range(max_new_tokens):
        logits = model(input_ids=output_ids, use_cache=False).logits[:, -1, :].float()
        if ban_mask:
            logits = ban_mask_token_logits(logits, int(mask_id))
        token = int(torch.argmax(logits, dim=-1)[0].item())
        next_token = torch.tensor([[token]], dtype=output_ids.dtype, device=output_ids.device)
        output_ids = torch.cat([output_ids, next_token], dim=1)
        if token in stop_token_ids:
            break
    return output_ids


@torch.no_grad()
def lossless_self_spec_decode(
    model,
    prompt_ids: torch.Tensor,
    *,
    k: int,
    max_new_tokens: int,
    stop_token_ids: set[int],
    mask_id: int,
    denoise_args: SimpleNamespace,
    ban_mask_in_verify: bool,
) -> dict[str, Any]:
    output_ids = prompt_ids.clone()
    original_len = int(prompt_ids.shape[1])
    rounds: list[dict[str, Any]] = []
    draft_seconds = 0.0
    verify_seconds = 0.0
    stopped = False

    while output_ids.shape[1] - original_len < max_new_tokens and not stopped:
        remaining = max_new_tokens - (output_ids.shape[1] - original_len)
        round_k = min(k, remaining)
        round_index = len(rounds)
        draft, draft_s = draft_tokens(model, output_ids, round_k, denoise_args)
        draft_seconds += draft_s

        verify_input = torch.cat([output_ids, draft], dim=1)
        cuda_sync()
        verify_start = time.perf_counter()
        verify_logits = shifted_ar_logits_for_next_tokens(
            model,
            verify_input,
            start_pos=int(output_ids.shape[1]),
            count=round_k,
            mask_id=mask_id,
            ban_mask=ban_mask_in_verify,
        )
        verify_tokens = torch.argmax(verify_logits, dim=-1)
        cuda_sync()
        verify_s = time.perf_counter() - verify_start
        verify_seconds += verify_s

        accepted = 0
        emitted: list[int] = []
        mismatch_token: int | None = None
        mismatch_draft: int | None = None
        mismatch_pos: int | None = None
        for pos in range(round_k):
            draft_token = int(draft[0, pos].item())
            ar_token = int(verify_tokens[0, pos].item())
            if draft_token == ar_token:
                accepted += 1
                emitted.append(draft_token)
                if draft_token in stop_token_ids:
                    stopped = True
                    break
                continue
            mismatch_pos = pos
            mismatch_draft = draft_token
            mismatch_token = ar_token
            emitted.append(ar_token)
            if ar_token in stop_token_ids:
                stopped = True
            break

        if not emitted:
            # This should only be reachable for zero-length rounds, which the loop prevents.
            raise RuntimeError("self-spec decode emitted no token")
        emitted_tensor = torch.tensor([emitted], dtype=output_ids.dtype, device=output_ids.device)
        output_ids = torch.cat([output_ids, emitted_tensor], dim=1)
        rounds.append(
            {
                "round_index": round_index,
                "k": int(k),
                "round_draft_len": int(round_k),
                "accepted_draft_tokens": int(accepted),
                "emitted_tokens": int(len(emitted)),
                "full_draft_accepted": bool(accepted == round_k),
                "mismatch_pos": mismatch_pos,
                "mismatch_draft_token": mismatch_draft,
                "mismatch_ar_token": mismatch_token,
                "stopped": bool(stopped),
                "draft_seconds": draft_s,
                "verify_seconds": verify_s,
            }
        )

    return {
        "output_ids": output_ids,
        "rounds": rounds,
        "draft_seconds": draft_seconds,
        "verify_seconds": verify_seconds,
        "generated_tokens": int(output_ids.shape[1] - original_len),
        "stopped": bool(stopped),
    }


def summarize_rounds(records: list[dict[str, Any]]) -> dict[str, Any]:
    all_rounds: list[dict[str, Any]] = []
    for record in records:
        all_rounds.extend(record.get("rounds") or [])
    hist = Counter(int(row["accepted_draft_tokens"]) for row in all_rounds)
    emitted = sum(int(row["emitted_tokens"]) for row in all_rounds)
    accepted = sum(int(row["accepted_draft_tokens"]) for row in all_rounds)
    drafted = sum(int(row["round_draft_len"]) for row in all_rounds)
    full_accepts = sum(int(bool(row["full_draft_accepted"])) for row in all_rounds)
    round_count = len(all_rounds)
    draft_forwards = round_count
    verify_forwards = round_count
    total_forwards = draft_forwards + verify_forwards
    seconds = sum(float(record.get("seconds") or 0.0) for record in records)
    draft_seconds = sum(float(record.get("draft_seconds") or 0.0) for record in records)
    verify_seconds = sum(float(record.get("verify_seconds") or 0.0) for record in records)
    acceptance_lengths = [int(row["accepted_draft_tokens"]) for row in all_rounds]
    emitted_lengths = [int(row["emitted_tokens"]) for row in all_rounds]
    return {
        "records": len(records),
        "rounds": round_count,
        "generated_tokens": emitted,
        "accepted_draft_tokens": accepted,
        "drafted_tokens": drafted,
        "full_draft_accept_rounds": full_accepts,
        "full_draft_accept_rate": full_accepts / round_count if round_count else None,
        "mean_accepted_draft_tokens_per_round": accepted / round_count if round_count else None,
        "median_accepted_draft_tokens_per_round": statistics.median(acceptance_lengths) if acceptance_lengths else None,
        "mean_emitted_tokens_per_round": emitted / round_count if round_count else None,
        "median_emitted_tokens_per_round": statistics.median(emitted_lengths) if emitted_lengths else None,
        "draft_token_acceptance_fraction": accepted / drafted if drafted else None,
        "histogram_accepted_draft_tokens": {str(key): hist[key] for key in sorted(hist)},
        "draft_forwards": draft_forwards,
        "verify_forwards": verify_forwards,
        "total_forwards": total_forwards,
        "draft_denoise_forwards_per_round": 1,
        "ar_verify_forwards_per_round": 1,
        "net_decode_speedup_emitted_over_2fwds": (emitted / total_forwards) if total_forwards else None,
        "net_decode_speedup_accepted_only_over_2fwds": (accepted / total_forwards) if total_forwards else None,
        "forwards_per_emitted_token": total_forwards / emitted if emitted else None,
        "seconds": seconds,
        "draft_seconds": draft_seconds,
        "verify_seconds": verify_seconds,
        "emitted_tokens_per_second_wall": emitted / seconds if seconds else None,
    }


def build_summary(args: argparse.Namespace, records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["slice"]), int(record["k"]))].append(record)
    by_slice_k = {
        f"{slice_name}/K{k}": {
            "slice": slice_name,
            "k": k,
            **summarize_rounds(rows),
        }
        for (slice_name, k), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1]))
    }
    by_slice: dict[str, dict[str, Any]] = {}
    for slice_name in sorted({str(record["slice"]) for record in records}):
        by_slice[slice_name] = summarize_rounds([record for record in records if record["slice"] == slice_name])
    by_k: dict[str, dict[str, Any]] = {}
    for k in sorted({int(record["k"]) for record in records}):
        by_k[f"K{k}"] = summarize_rounds([record for record in records if int(record["k"]) == k])
    summary = {
        "schema": "qwen35.self_spec_acceptance.v1",
        "base_model": str(args.base_model),
        "adapter": str(args.adapter) if args.adapter else None,
        "merge_adapter": not args.no_merge_adapter,
        "toolcall_jsonl": str(args.toolcall_jsonl),
        "gsm8k_jsonl": str(args.gsm8k_jsonl),
        "gsm8k_prompt_mode": args.gsm8k_prompt_mode,
        "k_values": list(args.k_values),
        "max_new_tokens": int(args.max_new_tokens),
        "draft_mode": "one route_i FLARE diffusion masked-block forward, argmax per position",
        "verify_mode": "one clean causal AR forward over prefix + draft",
        "forwards_per_round": 2,
        "lossless": "by construction; mismatch emits the AR verifier token",
        "records": len(records),
        "overall": summarize_rounds(records),
        "by_slice": by_slice,
        "by_k": by_k,
        "by_slice_k": by_slice_k,
        "ar_baseline_checks": {
            "checked_records": sum(1 for record in records if record.get("ar_baseline_checked")),
            "passed": sum(1 for record in records if record.get("ar_baseline_match")),
            "failed": sum(1 for record in records if record.get("ar_baseline_checked") and not record.get("ar_baseline_match")),
        },
    }
    if torch.cuda.is_available():
        summary["cuda_max_memory_allocated_gib"] = torch.cuda.max_memory_allocated() / (1024**3)
        summary["cuda_max_memory_reserved_gib"] = torch.cuda.max_memory_reserved() / (1024**3)
    return summary


def should_check_ar_baseline(slice_counts: dict[str, int], slice_name: str, verify_rows: int) -> bool:
    if verify_rows <= 0:
        return False
    seen = slice_counts.get(slice_name, 0)
    slice_counts[slice_name] = seen + 1
    return seen < verify_rows


def run_measurement(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("spec-decode acceptance measurement requires CUDA")
    model, tokenizer = load_model(
        args.base_model,
        args.adapter,
        merge_adapter=not args.no_merge_adapter,
        tokenizer_path=args.tokenizer_path,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    mask_id, stop_token_id, stop_token_ids = resolve_token_ids(model, tokenizer)
    stop_set = {int(token_id) for token_id in stop_token_ids}
    prompt_rows = build_prompt_rows(tokenizer, args)
    records: list[dict[str, Any]] = []
    denoise_args = SimpleNamespace(
        denoise_logit_mode="flare_shift",
        block_size=int(args.block_size),
        mask_id=int(mask_id),
        _flare_cache_state=None,
        _flare_cache_required=False,
    )

    for k in args.k_values:
        verify_slice_counts: dict[str, int] = {}
        for row_idx, row in enumerate(prompt_rows):
            encoded = tokenizer([row["prompt"]], return_tensors="pt", add_special_tokens=False)
            prompt_ids = encoded.input_ids.to("cuda")
            start = time.perf_counter()
            with torch.inference_mode():
                result = lossless_self_spec_decode(
                    model,
                    prompt_ids,
                    k=int(k),
                    max_new_tokens=int(args.max_new_tokens),
                    stop_token_ids=stop_set,
                    mask_id=int(mask_id),
                    denoise_args=denoise_args,
                    ban_mask_in_verify=bool(args.ban_mask_in_verify),
                )
            cuda_sync()
            seconds = time.perf_counter() - start
            output_ids = result.pop("output_ids")
            new_ids = output_ids[0, prompt_ids.shape[1] :].detach().cpu()
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            ar_baseline_checked = should_check_ar_baseline(
                verify_slice_counts,
                str(row["slice"]),
                int(args.verify_ar_baseline_rows),
            )
            ar_baseline_match = None
            if ar_baseline_checked:
                with torch.inference_mode():
                    ref = ar_greedy_reference(
                        model,
                        prompt_ids,
                        max_new_tokens=int(args.max_new_tokens),
                        stop_token_ids=stop_set,
                        mask_id=int(mask_id),
                        ban_mask=bool(args.ban_mask_in_verify),
                    )
                ar_baseline_match = bool(torch.equal(output_ids, ref))
            record = {
                "slice": row["slice"],
                "row_index": int(row["row_index"]),
                "id": row["id"],
                "source": row["source"],
                "k": int(k),
                "prompt_tokens": int(prompt_ids.shape[1]),
                "generated_tokens": int(result["generated_tokens"]),
                "generated_text_excerpt": text[:500],
                "seconds": seconds,
                "draft_seconds": float(result["draft_seconds"]),
                "verify_seconds": float(result["verify_seconds"]),
                "rounds": result["rounds"],
                "stopped": bool(result["stopped"]),
                "ar_baseline_checked": ar_baseline_checked,
                "ar_baseline_match": ar_baseline_match,
            }
            records.append(record)
            short_summary = summarize_rounds([record])
            print(
                f"K={k} {row['slice']}#{row['row_index']} "
                f"rounds={short_summary['rounds']} "
                f"mean_emit={short_summary['mean_emitted_tokens_per_round']:.3f} "
                f"mean_accept={short_summary['mean_accepted_draft_tokens_per_round']:.3f} "
                f"net={short_summary['net_decode_speedup_emitted_over_2fwds']:.3f} "
                f"seconds={seconds:.2f}",
                flush=True,
            )

    summary = build_summary(args, records)
    summary["mask_id"] = int(mask_id)
    summary["stop_token_id"] = int(stop_token_id)
    summary["stop_token_ids"] = [int(token_id) for token_id in stop_token_ids]
    return records, summary


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Qwen3.5-9B B@1000 Self-Spec Decode Acceptance",
        "",
        "This is a measurement only. No promotion decision is made.",
        "",
        "## Setup",
        "",
        f"- Base: `{summary['base_model']}`",
        f"- Adapter: `{summary['adapter']}`",
        f"- K values: `{summary['k_values']}`",
        f"- Max new tokens per row/K: `{summary['max_new_tokens']}`",
        "- Draft: one route_i FLARE diffusion masked-block forward, argmax per position.",
        "- Verify: one clean AR forward over `prefix + draft`.",
        "- Cost model: `2` full 9B forwards per round (`D=1` draft + `1` verifier).",
        "- Losslessness: by construction; the first mismatch emits the AR verifier token.",
        "",
        "## Summary By Slice And K",
        "",
        "| Slice | K | Rounds | Mean accepted draft toks/round | Mean emitted toks/round | Net speedup emitted/2fwds | Accepted-only/2fwds | Full-draft accept % | Histogram accepted length |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for key, row in summary["by_slice_k"].items():
        hist = ", ".join(f"{k}:{v}" for k, v in row["histogram_accepted_draft_tokens"].items())
        lines.append(
            "| {slice} | {k} | {rounds} | {mean_acc:.3f} | {mean_emit:.3f} | {net:.3f}x | {net_acc:.3f}x | {full:.1f}% | `{hist}` |".format(
                slice=row["slice"],
                k=row["k"],
                rounds=row["rounds"],
                mean_acc=row["mean_accepted_draft_tokens_per_round"] or 0.0,
                mean_emit=row["mean_emitted_tokens_per_round"] or 0.0,
                net=row["net_decode_speedup_emitted_over_2fwds"] or 0.0,
                net_acc=row["net_decode_speedup_accepted_only_over_2fwds"] or 0.0,
                full=(row["full_draft_accept_rate"] or 0.0) * 100.0,
                hist=hist,
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    best = max(
        summary["by_slice_k"].values(),
        key=lambda row: row["net_decode_speedup_emitted_over_2fwds"] or 0.0,
    )
    lines.append(
        "- Best measured emitted-token speedup was `{:.3f}x` on `{}` at `K={}`.".format(
            best["net_decode_speedup_emitted_over_2fwds"] or 0.0,
            best["slice"],
            best["k"],
        )
    )
    if (best["net_decode_speedup_emitted_over_2fwds"] or 0.0) >= 2.0:
        lines.append("- This clears the ~2x lossless gate; P2 vLLM spec-decode integration is worth building.")
    else:
        lines.append(
            "- This does not clear the ~2x lossless gate; a full-price 9B diffusion draft is not enough by itself."
        )
    lines.append(
        "- AR equality spot-checks: `{passed}/{checked}` passed.".format(
            passed=summary["ar_baseline_checks"]["passed"],
            checked=summary["ar_baseline_checks"]["checked_records"],
        )
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--no-merge-adapter", action="store_true")
    parser.add_argument("--toolcall-jsonl", type=Path, default=DEFAULT_TOOLCALL)
    parser.add_argument("--gsm8k-jsonl", type=Path, default=DEFAULT_GSM8K)
    parser.add_argument("--gsm8k-fewshot-jsonl", type=Path, default=DEFAULT_GSM8K_FEWSHOT)
    parser.add_argument("--toolcall-limit", type=int, default=12)
    parser.add_argument("--gsm8k-limit", type=int, default=12)
    parser.add_argument("--gsm8k-prompt-mode", choices=["phasea_fewshot", "final_only"], default="phasea_fewshot")
    parser.add_argument("--k-values", type=parse_int_list, default=parse_int_list("4,8,16,32"))
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--verify-ar-baseline-rows", type=int, default=1)
    parser.add_argument("--ban-mask-in-verify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--run-name", default="full")
    args = parser.parse_args()

    configure_env()
    start = time.time()
    records, summary = run_measurement(args)
    summary["elapsed_seconds"] = time.time() - start
    out_dir = Path(args.out_dir)
    records_path = out_dir / f"{args.run_name}.jsonl"
    summary_path = out_dir / f"{args.run_name}.summary.json"
    report_path = out_dir / f"{args.run_name}.report.md"
    write_jsonl(records_path, records)
    summary["records_jsonl"] = str(records_path)
    summary["summary_json"] = str(summary_path)
    summary["report_md"] = str(report_path)
    write_json(summary_path, summary)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(summary), encoding="utf-8")
    print(json.dumps({"summary_json": str(summary_path), "report_md": str(report_path), "overall": summary["overall"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
