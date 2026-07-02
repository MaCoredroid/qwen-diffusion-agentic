#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


DEFAULT_ROOT = Path("runs/flare_redesign_run1_threewave_grammar")
DEFAULT_CONFIDENCE_ROOT = Path("runs/flare_redesign_run1_twowave/w1_090")


CONDITIONS = [
    ("A", "grammar_projected", "grammar_projected", "Wave-1 grammar-projected scaffold; wave-2 raw values"),
    ("B", "careful", "careful", "Wave-1 careful scaffold; wave-2 raw values"),
]


CONFIDENCE_CONDITIONS = [
    ("C", "confidence_bulk", "tau_08", "Existing confidence-bulk wave-1 contrast, value tau 0.80"),
    ("C", "confidence_bulk", "tau_095", "Existing confidence-bulk wave-1 contrast, value tau 0.95"),
]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ratio(num, den):
    if den is None or int(den) == 0:
        return None
    return float(num) / float(den)


def rounded(value, digits=3):
    if value is None:
        return None
    return round(float(value), digits)


def summarize_summary(path: Path, condition: str, mode: str, tau: str, note: str) -> dict:
    summary = read_json(path)
    totals = summary["totals"]
    generated_tokens = int(summary.get("generated_tokens") or 0)
    forwards = int(totals.get("sampler_denoise_forwards_total") or 0)
    records = int(totals.get("records") or 0)
    wave1_tokens = int(totals.get("sampler_two_wave_wave1_committed_tokens") or 0)
    wave1_forwards = int(totals.get("sampler_two_wave_wave1_denoise_forwards") or 0)
    wave1_steps = int(totals.get("sampler_two_wave_wave1_projection_steps") or 0)
    wave2_value_tokens = int(totals.get("sampler_two_wave_wave2_value_tokens") or 0)
    wave2_forwards = int(totals.get("sampler_two_wave_wave2_denoise_forwards") or 0)
    forced_value_counters = {
        "wave1_value_tokens": int(totals.get("sampler_two_wave_wave1_value_tokens") or 0),
        "wave2_forced_tokens": int(totals.get("sampler_two_wave_wave2_forced_tokens") or 0),
        "selected_candidate_force_token_visits": int(totals.get("sampler_selected_candidate_force_token_visits") or 0),
        "candidate_sequence_force_token_visits": int(totals.get("sampler_candidate_sequence_force_token_visits") or 0),
        "tool_value_candidate_force_token_visits": int(totals.get("sampler_tool_value_candidate_force_token_visits") or 0),
        "forced_schedule_token_visits": int(totals.get("sampler_forced_schedule_token_visits") or 0),
    }
    return {
        "condition": condition,
        "mode": mode,
        "tau": tau,
        "split": summary["eval_name"],
        "summary_json": str(path),
        "output_jsonl": summary.get("out_jsonl"),
        "note": note,
        "records": records,
        "exact_args": int(totals.get("exact_arguments") or 0),
        "exact_seq": int(totals.get("exact_tool_sequence") or 0),
        "valid_json": int(totals.get("valid_tool_json") or 0),
        "generated_tokens": generated_tokens,
        "denoise_forwards": forwards,
        "blended_tpf": rounded(ratio(generated_tokens, forwards)),
        "scaffold_tpf_model_forwards": rounded(ratio(wave1_tokens, wave1_forwards)),
        "scaffold_tokens_per_wave1_step": rounded(ratio(wave1_tokens, wave1_forwards + wave1_steps)),
        "value_tpf": rounded(ratio(wave2_value_tokens, wave2_forwards)),
        "seconds_per_record": rounded(ratio(float(summary.get("elapsed_seconds") or 0.0), records), 2),
        "wave1_committed_tokens": wave1_tokens,
        "wave1_denoise_forwards": wave1_forwards,
        "wave1_projected_tokens": int(totals.get("sampler_two_wave_wave1_projected_tokens") or 0),
        "wave1_projection_steps": wave1_steps,
        "wave1_forced_tokens": int(totals.get("sampler_two_wave_wave1_forced_tokens") or 0),
        "wave2_value_tokens": wave2_value_tokens,
        "wave2_denoise_forwards": wave2_forwards,
        "forced_value_counters": forced_value_counters,
        "value_nonleakage_pass": all(value == 0 for value in forced_value_counters.values()),
    }


def split_label(row: dict) -> str:
    if "heldout" in row["split"]:
        return "heldout"
    if "public" in row["split"]:
        return "public"
    return row["split"]


def metric_cell(row: dict) -> str:
    return (
        f"{row['exact_args']}/{row['records']} args, "
        f"{row['exact_seq']}/{row['records']} seq, "
        f"{row['valid_json']}/{row['records']} valid"
    )


def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def tau_label(tau_dir: str) -> str:
    if tau_dir == "tau_080":
        return "0.80"
    if tau_dir == "tau_095":
        return "0.95"
    return tau_dir.replace("tau_", "0.")


def markdown_report(rows: list[dict], output_json: Path) -> str:
    by_key = {(row["mode"], row["tau"], split_label(row)): row for row in rows}
    lines = [
        "# Three-Wave Grammar Wave-1 Comparison",
        "",
        "Run-1 checkpoint, raw value lane, native heldout/public slices. Wave 2 is raw `argument_value` parallel commit at value tau 0.80 and 0.95. Condition C is the previously measured confidence-bulk wave-1 contrast.",
        "",
        "## Headline",
        "",
        "- Condition B (careful wave 1) does not reproduce the native exact-args anchor: best B is tau 0.95 with heldout 2/12 and public 4/12 exact args, below the requested 3/12 heldout and 8/12 public anchor.",
        "- Condition A (grammar-projected wave 1) materially lifts blended TPF while holding or improving B's quality, but still misses the public exactness anchor: best A is tau 0.95 with heldout 4/12 and public 6/12 exact args.",
        "- Condition C (confidence-bulk wave 1) remains the negative contrast: 0/12 exact args on both slices at tau 0.80 and tau 0.95, with degraded validity.",
        "- Values were not forced or projected in A/B/C: all value force counters are zero in every row.",
        "",
        "## Results",
        "",
        "| Cond | Mode | Value tau | Split | Quality | Blended TPF | Scaffold TPF | Value TPF | sec/rec | Wave1 projected | Value force pass |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["condition"],
                    row["mode"],
                    row["tau"],
                    split_label(row),
                    metric_cell(row),
                    fmt(row["blended_tpf"]),
                    fmt(row["scaffold_tpf_model_forwards"]),
                    fmt(row["value_tpf"]),
                    fmt(row["seconds_per_record"]),
                    str(row["wave1_projected_tokens"]),
                    "yes" if row["value_nonleakage_pass"] else "NO",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A answers the narrow red-team: confidence-bulk scaffold was the wrong wave-1 implementation. Grammar-projected scaffold recovers validity and tool sequence quality while lifting blended TPF to 1.66-1.93 on the native slices.",
            "",
            "The decisive negative is different: the careful left-context-first rescue from the copy-span slice does not generalize to native exact arguments. Even with conservative value tau 0.95, B reaches only 2/12 heldout and 4/12 public exact args. A is faster and better than B, but public exact args remain 6/12, below the 8/12 public anchor.",
            "",
            "The honest speed ceiling at held exactness on both slices is therefore not established. There is a material speed signal for grammar-projected wave 1, but no operating point here preserves the requested public exactness quality.",
            "",
            f"Machine-readable report: `{output_json}`",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--confidence-root", type=Path, default=DEFAULT_CONFIDENCE_ROOT)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_ROOT / "threewave_grammar_report.json")
    parser.add_argument("--out-md", type=Path, default=DEFAULT_ROOT / "threewave_grammar_report.md")
    args = parser.parse_args()

    rows = []
    for condition, mode, directory, note in CONDITIONS:
        for tau in ["tau_080", "tau_095"]:
            for split in ["heldout", "public"]:
                rows.append(
                    summarize_summary(
                        args.root / directory / tau / f"{split}_native_12.summary.json",
                        condition,
                        mode,
                        tau_label(tau),
                        note,
                    )
                )
    for condition, mode, tau_dir, note in CONFIDENCE_CONDITIONS:
        confidence_tau_label = "0.80" if tau_dir == "tau_08" else "0.95"
        for split in ["heldout", "public"]:
            rows.append(
                summarize_summary(
                    args.confidence_root / tau_dir / f"{split}_native_12.summary.json",
                    condition,
                    mode,
                    confidence_tau_label,
                    note,
                )
            )

    report = {
        "baseline_anchor": {
            "heldout_exact_args": "3/12",
            "public_exact_args": "8/12",
            "source": "flare_threewave_grammar steer",
        },
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(markdown_report(rows, args.out_json), encoding="utf-8")
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md)}, indent=2))


if __name__ == "__main__":
    main()
