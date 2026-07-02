#!/usr/bin/env python3
"""Summarize the powered native scale-up eval: baseline careful vs per-call waves."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("/home/mark/qwen_diffusion")
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from eval_fastdllm_toolcall_cases import case_context_text  # noqa: E402
from eval_toolcall_jsonl import extract_tool_calls  # noqa: E402


DEFAULT_ROOT = ROOT / "runs/flare_scaleup_eval"
DEFAULT_CASES = ROOT / "data/toolcall_eval_native/flare_scaleup_native_58.jsonl"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ratio(num: float, den: float) -> float | None:
    if not den:
        return None
    return float(num) / float(den)


def round_or_none(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def denoise_forwards(totals: dict[str, Any]) -> int:
    explicit = int(totals.get("sampler_denoise_forwards_total") or 0)
    if explicit:
        return explicit
    return int(
        (totals.get("sampler_parallel_commit_denoise_forwards") or 0)
        + (totals.get("sampler_two_wave_wave1_denoise_forwards") or 0)
        + (totals.get("sampler_two_wave_wave2_denoise_forwards") or 0)
        + (totals.get("sampler_fallback_denoise_forwards") or 0)
    )


def summarize_condition(summary_path: Path, label: str) -> dict[str, Any]:
    summary = read_json(summary_path)
    totals = summary["totals"]
    records = int(totals.get("records") or 0)
    generated = int(summary.get("generated_tokens") or 0)
    forwards = denoise_forwards(totals)
    wave1_tokens = int(totals.get("sampler_two_wave_wave1_committed_tokens") or 0)
    wave1_forwards = int(totals.get("sampler_two_wave_wave1_denoise_forwards") or 0)
    wave2_tokens = int(totals.get("sampler_two_wave_wave2_value_tokens") or 0)
    wave2_forwards = int(totals.get("sampler_two_wave_wave2_denoise_forwards") or 0)
    elapsed = float(summary.get("elapsed_seconds") or 0.0)
    force_counters = {
        "forced_schedule_token_visits": int(totals.get("sampler_forced_schedule_token_visits") or 0),
        "tool_value_candidate_force_token_visits": int(
            totals.get("sampler_tool_value_candidate_force_token_visits") or 0
        ),
        "wave1_value_tokens": int(totals.get("sampler_two_wave_wave1_value_tokens") or 0),
        "wave2_forced_tokens": int(totals.get("sampler_two_wave_wave2_forced_tokens") or 0),
        "parallel_commit_forced_tokens": int(totals.get("sampler_parallel_commit_forced_tokens") or 0),
        "wave1_projected_tokens": int(totals.get("sampler_two_wave_wave1_projected_tokens") or 0),
        "wave1_forced_tokens": int(totals.get("sampler_two_wave_wave1_forced_tokens") or 0),
    }
    return {
        "label": label,
        "summary_json": str(summary_path),
        "output_jsonl": summary.get("out_jsonl"),
        "records": records,
        "exact_args": int(totals.get("exact_arguments") or 0),
        "exact_seq": int(totals.get("exact_tool_sequence") or 0),
        "valid_json": int(totals.get("valid_tool_json") or 0),
        "generated_tokens": generated,
        "denoise_forwards": forwards,
        "blended_tpf": round_or_none(ratio(generated, forwards)),
        "scaffold_tpf": round_or_none(ratio(wave1_tokens, wave1_forwards)),
        "value_tpf": round_or_none(ratio(wave2_tokens, wave2_forwards)),
        "elapsed_seconds": round_or_none(elapsed, 3),
        "seconds_per_record": round_or_none(ratio(elapsed, records), 3),
        "force_counters": force_counters,
        "raw_summary": summary,
    }


def normalize_text(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def leaf_items(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        out: list[tuple[str, Any]] = []
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.extend(leaf_items(item, path))
        return out
    if isinstance(value, list):
        out = []
        for idx, item in enumerate(value):
            out.extend(leaf_items(item, f"{prefix}[{idx}]"))
        return out
    return [(prefix, value)]


def value_in_context(value: Any, context: str) -> bool:
    raw = str(value).strip()
    if not raw:
        return False
    if raw in context:
        return True
    return normalize_text(raw) in normalize_text(context)


def call_args(call: dict[str, Any] | None) -> dict[str, Any]:
    if not call:
        return {}
    args = call.get("arguments") or {}
    return args if isinstance(args, dict) else {"arguments": args}


def get_path_value(value: Any, path: str) -> Any:
    current = value
    for key, index in re.findall(r"([^\.\[\]]+)|\[(\d+)\]", path):
        if key:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        else:
            idx = int(index)
            if not isinstance(current, list) or idx >= len(current):
                return None
            current = current[idx]
    return current


def missed_value_split(case: dict[str, Any], generated_calls: list[dict[str, Any]]) -> dict[str, Any]:
    context = case_context_text(case)
    gold_calls, _ = extract_tool_calls(case.get("gold_assistant") or "")
    counts = Counter()
    examples: list[dict[str, Any]] = []
    for call_idx, gold_call in enumerate(gold_calls):
        generated_call = generated_calls[call_idx] if call_idx < len(generated_calls) else None
        generated_args = call_args(generated_call)
        if not generated_call or generated_call.get("name") != gold_call.get("name"):
            generated_args = {}
        for path, gold_value in leaf_items(call_args(gold_call)):
            generated_value = get_path_value(generated_args, path)
            if normalize_text(generated_value) == normalize_text(gold_value):
                continue
            bucket = "copy" if value_in_context(gold_value, context) else "derived"
            counts[bucket] += 1
            if len(examples) < 12:
                examples.append(
                    {
                        "tool_call_index": call_idx,
                        "tool": gold_call.get("name"),
                        "path": path,
                        "gold_value": gold_value,
                        "generated_value": generated_value,
                        "bucket": bucket,
                    }
                )
    return {"counts": dict(sorted(counts.items())), "examples": examples}


def rows_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("id") or row.get("idx")): row for row in rows}


def failure_taxonomy(
    cases_path: Path,
    baseline_jsonl: Path,
    percall_jsonl: Path,
) -> dict[str, Any]:
    cases = rows_by_id(read_jsonl(cases_path))
    baseline = rows_by_id(read_jsonl(baseline_jsonl))
    percall = rows_by_id(read_jsonl(percall_jsonl))
    b_only = []
    all_percall_misses = []
    split_counts = Counter()
    split_examples = []
    for case_id, row in percall.items():
        case = cases.get(case_id)
        if case is None:
            continue
        split = missed_value_split(case, row.get("calls") or [])
        for key, value in split["counts"].items():
            split_counts[key] += int(value)
        split_examples.extend(split["examples"])
        if row.get("exact_arguments"):
            continue
        item = {
            "idx": row.get("idx"),
            "id": case_id,
            "source": case.get("source"),
            "valid_json": bool(row.get("valid_tool_json")),
            "exact_seq": bool(row.get("exact_tool_sequence")),
            "called_names": row.get("called_names") or [],
            "missing_call_count": row.get("missing_call_count"),
            "extra_call_count": row.get("extra_call_count"),
            "repeated_call_count": row.get("repeated_call_count"),
            "copy_vs_derived": split["counts"],
            "argument_examples": split["examples"][:5],
        }
        all_percall_misses.append(item)
        base_row = baseline.get(case_id)
        if base_row and base_row.get("exact_arguments"):
            b_only.append(item)
    return {
        "percall_misses": all_percall_misses,
        "baseline_exact_percall_miss": b_only,
        "percall_miss_value_split": dict(sorted(split_counts.items())),
        "percall_miss_value_examples": split_examples[:25],
    }


def fmt_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}".rstrip("0").rstrip(".")


def markdown(report: dict[str, Any]) -> str:
    a = report["conditions"]["baseline_careful"]
    b = report["conditions"]["percall_waves_tau095"]
    manifest = report.get("manifest") or {}
    leak = manifest.get("train_leak_check") or {}
    lines = [
        "# FLARE Scale-Up Native Eval",
        "",
        f"Slice: {manifest.get('records', a['records'])} leak-checked native records.",
        f"Sources: {manifest.get('source_path_counts', {})}",
        (
            "Leak check: "
            f"exact_instance={leak.get('exact_instance_overlaps', 'n/a')}, "
            f"user={leak.get('user_overlaps', 'n/a')} against all Run-1 train records; "
            f"same-tool/all-value={leak.get('same_tool_all_eval_arg_values_overlaps', 'n/a')} "
            f"against near-leak scope `{leak.get('near_leak_scope', 'n/a')}` "
            f"({leak.get('near_leak_scope_records', 'n/a')} records)."
        ),
        "",
        "| Condition | exact_args | exact_seq | valid_json | blended TPF | sec/rec | total wall |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| Baseline careful | {a['exact_args']}/{a['records']} | {a['exact_seq']}/{a['records']} | "
            f"{a['valid_json']}/{a['records']} | {fmt_float(a['blended_tpf'])} | "
            f"{fmt_float(a['seconds_per_record'])} | {fmt_float(a['elapsed_seconds'])}s |"
        ),
        (
            f"| Per-call waves tau 0.95 | {b['exact_args']}/{b['records']} | {b['exact_seq']}/{b['records']} | "
            f"{b['valid_json']}/{b['records']} | {fmt_float(b['blended_tpf'])} | "
            f"{fmt_float(b['seconds_per_record'])} | {fmt_float(b['elapsed_seconds'])}s |"
        ),
        "",
        "## Headline",
        "",
        f"- exact_args delta (per-call - baseline): {report['headline']['exact_args_delta']} / {a['records']}",
        f"- honest wall speedup: {fmt_float(report['headline']['wall_speedup'])}x",
        f"- per-call misses: {len(report['failures']['percall_misses'])}/{b['records']}",
        f"- per-call miss value split: {report['failures']['percall_miss_value_split']}",
        f"- value force counters: {b['force_counters']}",
        f"- full per-call miss list: `{report['out_json']}`" if "out_json" in report else "",
        "",
        "## B-Only Misses",
        "",
    ]
    b_only = report["failures"]["baseline_exact_percall_miss"]
    if not b_only:
        lines.append("- None.")
    else:
        for item in b_only:
            lines.append(
                f"- idx {item['idx']} id `{item['id']}`: valid={item['valid_json']} "
                f"seq={item['exact_seq']} copy/derived={item['copy_vs_derived']} names={item['called_names']}"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--cases-jsonl", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_ROOT / "scaleup_native_58_manifest.json")
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_ROOT / "baseline_careful/scaleup_native_58.summary.json")
    parser.add_argument("--percall-summary", type=Path, default=DEFAULT_ROOT / "percall_waves_tau095/scaleup_native_58.summary.json")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_ROOT / "scaleup_eval_report.json")
    parser.add_argument("--out-md", type=Path, default=DEFAULT_ROOT / "scaleup_eval_report.md")
    args = parser.parse_args()

    baseline = summarize_condition(args.baseline_summary, "baseline_careful")
    percall = summarize_condition(args.percall_summary, "percall_waves_tau095")
    failures = failure_taxonomy(
        args.cases_jsonl,
        Path(baseline["output_jsonl"]),
        Path(percall["output_jsonl"]),
    )
    wall_speedup = ratio(float(baseline["elapsed_seconds"]), float(percall["elapsed_seconds"]))
    report = {
        "out_json": str(args.out_json),
        "out_md": str(args.out_md),
        "manifest": read_json(args.manifest_json) if args.manifest_json.exists() else {},
        "conditions": {
            "baseline_careful": {k: v for k, v in baseline.items() if k != "raw_summary"},
            "percall_waves_tau095": {k: v for k, v in percall.items() if k != "raw_summary"},
        },
        "headline": {
            "exact_args_delta": int(percall["exact_args"]) - int(baseline["exact_args"]),
            "wall_speedup": round_or_none(wall_speedup),
        },
        "failures": failures,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    args.out_md.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
