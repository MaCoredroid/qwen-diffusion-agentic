#!/usr/bin/env python3
"""Write the final stock-vs-merged-vs-hybrid endgame scoreboard."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from eval_flare_northstar_matched import read_rows, summarize_backend  # noqa: E402


DEFAULT_OUT = ROOT / "runs/endgame_scoreboard"
DEFAULT_STOCK_ROOT = ROOT / "runs/endgame_stock_qwen35_ar_guided"
DEFAULT_V6_GATES = ROOT / "runs/rl_multiturn_grpo_v6/from_v2_hybrid_mixed35_kl005_g4_step300_gates"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stock-root", type=Path, default=DEFAULT_STOCK_ROOT)
    parser.add_argument("--v6-gates-root", type=Path, default=DEFAULT_V6_GATES)
    parser.add_argument(
        "--stock-snapshot",
        default="/home/mark/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def audit_clean(path: Path) -> bool:
    payload = load_json(path)
    totals = payload.get("totals") if isinstance(payload, dict) else {}
    if not isinstance(totals, dict):
        return False
    return (
        int(totals.get("zero_projected_value_tokens_verified") or 0) == 1
        and int(totals.get("projected_value_tokens_exact") or 0) == 0
        and int(totals.get("parallel_commit_forced_tokens_counter") or 0) == 0
    )


def candidate_rows(name: str, matched_path: Path, never_path: Path, *, valid: bool, reason: str) -> dict[str, Any]:
    if valid and matched_path.exists() and never_path.exists():
        matched = read_rows(matched_path)
        never = read_rows(never_path)
        return {
            "name": name,
            "valid": True,
            "reason": reason,
            "matched_rows": matched,
            "never_rows": never,
            "aggregate_rows": matched + never,
        }
    return {"name": name, "valid": False, "reason": reason}


def choose_hybrid(v6_gates_root: Path) -> dict[str, Any]:
    v2 = candidate_rows(
        "v2_hybrid_clean",
        ROOT / "runs/hybrid_forced_grammar_seq_values_v2/matched20/diffusion_hybrid_forced_grammar_seq_values/turns.jsonl",
        ROOT
        / "runs/hybrid_broaden_nevertrain_v2/nevertrain_bfcl_apibank60/diffusion_hybrid_forced_grammar_seq_values/turns.jsonl",
        valid=True,
        reason="accepted v2 hybrid-clean baseline",
    )
    v6_retention = load_json(v6_gates_root / "retention_gate.json")
    v6_matched_audit = (
        v6_gates_root
        / "matched20_hybrid/diffusion_hybrid_forced_grammar_seq_values/projection_value_audit.json"
    )
    v6_never_audit = (
        v6_gates_root
        / "nevertrain_bfcl_apibank60_hybrid/diffusion_hybrid_forced_grammar_seq_values/projection_value_audit.json"
    )
    v6_valid = (
        bool(v6_retention.get("passed"))
        and audit_clean(v6_matched_audit)
        and audit_clean(v6_never_audit)
    )
    v6 = candidate_rows(
        "v6_hybrid_clean",
        v6_gates_root / "matched20_hybrid/diffusion_hybrid_forced_grammar_seq_values/turns.jsonl",
        v6_gates_root / "nevertrain_bfcl_apibank60_hybrid/diffusion_hybrid_forced_grammar_seq_values/turns.jsonl",
        valid=v6_valid,
        reason=(
            "retention/audit passed"
            if v6_valid
            else f"not eligible: retention_passed={bool(v6_retention.get('passed'))}, "
            f"matched_audit={audit_clean(v6_matched_audit)}, never_audit={audit_clean(v6_never_audit)}"
        ),
    )
    candidates = [item for item in [v2, v6] if item.get("valid")]
    if not candidates:
        raise SystemExit("no valid hybrid candidate rows found")
    scored = []
    for item in candidates:
        matched_summary = summarize_backend(item["matched_rows"])
        never_summary = summarize_backend(item["never_rows"])
        aggregate_summary = summarize_backend(item["aggregate_rows"])
        scored.append(
            {
                "item": item,
                "matched_exact": int(matched_summary["exact_arguments"]),
                "never_exact": int(never_summary["exact_arguments"]),
                "aggregate_exact": int(aggregate_summary["exact_arguments"]),
            }
        )
    scored.sort(key=lambda row: (row["aggregate_exact"], row["matched_exact"], row["never_exact"]), reverse=True)
    selected = scored[0]["item"]
    return {
        "selected": selected,
        "candidates": [
            {
                "name": row["item"]["name"],
                "valid": True,
                "reason": row["item"]["reason"],
                "matched_exact": row["matched_exact"],
                "never_exact": row["never_exact"],
                "aggregate_exact": row["aggregate_exact"],
            }
            for row in scored
        ]
        + ([{"name": v6["name"], "valid": False, "reason": v6["reason"]}] if not v6.get("valid") else []),
    }


def compact(summary: dict[str, Any], *, steps_key: str) -> dict[str, Any]:
    return {
        "exact_args": f"{int(summary['exact_arguments'])}/{int(summary['turns'])}",
        "episode_exact": f"{int(summary['episode_exact_arguments_all_turns'])}/{int(summary['episodes'])}",
        "valid": f"{int(summary['valid_tool_json'])}/{int(summary['turns'])}",
        "sec_per_turn": float(summary["sec_per_turn"]),
        "steps_per_turn": float(summary[steps_key] or 0.0),
    }


def row_metrics(row: dict[str, Any], rows_by_slice: dict[str, list[dict]], *, steps_key: str) -> dict[str, Any]:
    out = {"label": row["label"], "runtime": row["runtime"], "steps_label": row["steps_label"]}
    for slice_name, rows in rows_by_slice.items():
        out[slice_name] = compact(summarize_backend(rows), steps_key=steps_key)
    return out


def table_lines(title: str, rows: list[dict[str, Any]], slice_name: str) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| row | exact_args | episode_exact | valid | sec/turn | forwards-or-steps/turn | runtime |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        cell = row[slice_name]
        lines.append(
            f"| {row['label']} | {cell['exact_args']} | {cell['episode_exact']} | {cell['valid']} "
            f"| {cell['sec_per_turn']:.3f} | {cell['steps_per_turn']:.2f} {row['steps_label']} | {row['runtime']} |"
        )
    return lines


def quant_lines(quant_comparison: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "## Stock FP8 Quantization",
        "",
        "| slice | exact_args delta | sec/turn delta | FP8 speedup vs bf16 |",
        "|---|---:|---:|---:|",
    ]
    for slice_name in ["matched20", "nevertrain", "aggregate"]:
        item = quant_comparison[slice_name]
        speedup = item["fp8_speedup_vs_bf16"]
        speedup_text = f"{speedup:.3f}x" if isinstance(speedup, float) else "n/a"
        lines.append(
            f"| {slice_name} | {int(item['exact_args_delta_fp8_minus_bf16']):+d} "
            f"| {float(item['sec_per_turn_delta_fp8_minus_bf16']):+.3f} | {speedup_text} |"
        )
    return lines


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    stock_bf16_matched = read_rows(args.stock_root / "bf16/matched20/ar-vllm-guided/turns.jsonl")
    stock_bf16_never = read_rows(args.stock_root / "bf16/nevertrain_bfcl_apibank60/ar-vllm-guided/turns.jsonl")
    stock_fp8_matched = read_rows(args.stock_root / "fp8/matched20/ar-vllm-guided/turns.jsonl")
    stock_fp8_never = read_rows(args.stock_root / "fp8/nevertrain_bfcl_apibank60/ar-vllm-guided/turns.jsonl")
    merged_matched = read_rows(ROOT / "runs/hybrid_broaden_nevertrain_v2/matched20/ar-vllm-guided/turns.jsonl")
    merged_never = read_rows(
        ROOT / "runs/hybrid_broaden_nevertrain_v2/nevertrain_bfcl_apibank60/ar-vllm-guided/turns.jsonl"
    )
    hybrid = choose_hybrid(args.v6_gates_root)
    selected = hybrid["selected"]

    rows = [
        row_metrics(
            {
                "label": "stock-bf16-AR-guided",
                "runtime": "vLLM bf16 guided",
                "steps_label": "decode tokens/turn",
            },
            {
                "matched20": stock_bf16_matched,
                "nevertrain": stock_bf16_never,
                "aggregate": stock_bf16_matched + stock_bf16_never,
            },
            steps_key="generated_tokens_per_turn",
        ),
        row_metrics(
            {
                "label": "stock-FP8-AR-guided",
                "runtime": "vLLM fp8 guided",
                "steps_label": "decode tokens/turn",
            },
            {
                "matched20": stock_fp8_matched,
                "nevertrain": stock_fp8_never,
                "aggregate": stock_fp8_matched + stock_fp8_never,
            },
            steps_key="generated_tokens_per_turn",
        ),
        row_metrics(
            {
                "label": "merged-AR guided",
                "runtime": "vLLM bf16 guided",
                "steps_label": "decode tokens/turn",
            },
            {"matched20": merged_matched, "nevertrain": merged_never, "aggregate": merged_matched + merged_never},
            steps_key="generated_tokens_per_turn",
        ),
        row_metrics(
            {
                "label": f"OUR SYSTEM hybrid-clean ({selected['name']})",
                "runtime": "HF diffusion hybrid-clean",
                "steps_label": "denoise forwards/turn",
            },
            {
                "matched20": selected["matched_rows"],
                "nevertrain": selected["never_rows"],
                "aggregate": selected["aggregate_rows"],
            },
            steps_key="denoise_forwards_per_turn",
        ),
    ]

    stock_matched_summary = summarize_backend(stock_bf16_matched)
    merged_matched_summary = summarize_backend(merged_matched)
    stock_bar_note = (
        f"Stock bf16 matched-20 exact_args is {stock_matched_summary['exact_arguments']}/{stock_matched_summary['turns']}; "
        f"merged-AR guided is {merged_matched_summary['exact_arguments']}/{merged_matched_summary['turns']}."
    )
    if int(stock_matched_summary["exact_arguments"]) > int(merged_matched_summary["exact_arguments"]):
        stock_bar_note += " The maintains-AR bar rises to the stock result."
    else:
        stock_bar_note += " The stock control does not raise the matched-20 bar above the merged-AR row."

    def quant_delta(slice_name: str) -> dict[str, Any]:
        bf16_row = next(row for row in rows if row["label"] == "stock-bf16-AR-guided")[slice_name]
        fp8_row = next(row for row in rows if row["label"] == "stock-FP8-AR-guided")[slice_name]
        bf16_exact = int(bf16_row["exact_args"].split("/")[0])
        fp8_exact = int(fp8_row["exact_args"].split("/")[0])
        bf16_sec = float(bf16_row["sec_per_turn"])
        fp8_sec = float(fp8_row["sec_per_turn"])
        return {
            "exact_args_delta_fp8_minus_bf16": fp8_exact - bf16_exact,
            "sec_per_turn_delta_fp8_minus_bf16": fp8_sec - bf16_sec,
            "fp8_speedup_vs_bf16": (bf16_sec / fp8_sec) if fp8_sec > 0 else None,
        }

    quant_comparison = {name: quant_delta(name) for name in ["matched20", "nevertrain", "aggregate"]}

    summary = {
        "stock_snapshot": args.stock_snapshot,
        "stock_note": "Qwen/Qwen3.5-9B cached snapshot served on vLLM as bf16 and as online FP8; bf16 is not NVFP4.",
        "quant_comparison": quant_comparison,
        "hybrid_selection": {
            "selected": selected["name"],
            "candidates": hybrid["candidates"],
            "rule": "highest aggregate exact_args among retention-valid, zero-value-projection hybrid candidates",
        },
        "rows": rows,
        "stock_bar_note": stock_bar_note,
        "wall_clock_note": (
            "AR rows are vLLM engine measurements; the hybrid row is the current HF diffusion stack. "
            "The hybrid wall-clock column is intentionally honest and remains the P2 engine deliverable."
        ),
        "git_hash": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "artifacts": {
            "stock_root": str(args.stock_root),
            "stock_bf16_root": str(args.stock_root / "bf16"),
            "stock_fp8_root": str(args.stock_root / "fp8"),
            "v6_gates_root": str(args.v6_gates_root),
            "merged_ar_matched": "runs/hybrid_broaden_nevertrain_v2/matched20/ar-vllm-guided/turns.jsonl",
            "merged_ar_nevertrain": "runs/hybrid_broaden_nevertrain_v2/nevertrain_bfcl_apibank60/ar-vllm-guided/turns.jsonl",
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Endgame Scoreboard",
        "",
        "Rows are the user-defined comparison set with the added stock FP8 control: stock bf16 guided, stock FP8 guided, merged-AR guided, and our best diffusion hybrid-clean system.",
        "",
        f"- Stock model: `{args.stock_snapshot}`",
        "- Stock precision/runtime: vLLM bf16 guided decoding and vLLM `--quantization fp8` guided decoding. The bf16 stock row is not NVFP4; NVFP4 was only the 27B teacher context.",
        f"- Hybrid selected: `{selected['name']}`.",
        f"- Selection rule: {summary['hybrid_selection']['rule']}.",
        f"- {stock_bar_note}",
        "- Quant tax/speedup: FP8-vs-bf16 deltas are reported in `summary.json` under `quant_comparison`.",
        "- Wall-clock: hybrid-clean is still on the HF stack and is expected to be slower than vLLM AR here; closing that column is the P2 engine deliverable.",
        "",
    ]
    lines.extend(table_lines("Matched-20", rows, "matched20"))
    lines.append("")
    lines.extend(table_lines("Never-Train BFCL/API-Bank", rows, "nevertrain"))
    lines.append("")
    lines.extend(table_lines("Aggregate", rows, "aggregate"))
    lines.append("")
    lines.extend(quant_lines(quant_comparison))
    lines.append("")
    lines.extend(
        [
            "## Artifacts",
            "",
            f"- Stock bf16 AR-guided root: `{args.stock_root / 'bf16'}`",
            f"- Stock FP8 AR-guided root: `{args.stock_root / 'fp8'}`",
            f"- v6 gates root: `{args.v6_gates_root}`",
            "- Merged-AR rows: `runs/hybrid_broaden_nevertrain_v2/.../ar-vllm-guided/turns.jsonl`",
            "- Hybrid rows: selected from v2/v6 retention-valid hybrid-clean artifacts.",
        ]
    )
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
