#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


ROOT = Path("runs/flare_redesign_run1_percall_waves")
PLAIN = Path("runs/flare_redesign_run1_eval/tau_095/public_native_12.jsonl")
OLD_A = Path("runs/flare_redesign_run1_threewave_grammar/grammar_projected/tau_095/public_native_12.jsonl")
OLD_B = Path("runs/flare_redesign_run1_threewave_grammar/careful/tau_095/public_native_12.jsonl")


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ratio(num, den):
    if not den:
        return None
    return float(num) / float(den)


def round_or_none(value, digits=3):
    if value is None:
        return None
    return round(float(value), digits)


def summarize(path: Path, label: str, tau: str) -> dict:
    summary = read_json(path)
    totals = summary["totals"]
    generated = int(summary.get("generated_tokens") or 0)
    forwards = int(totals.get("sampler_denoise_forwards_total") or 0)
    wave1_tokens = int(totals.get("sampler_two_wave_wave1_committed_tokens") or 0)
    wave1_forwards = int(totals.get("sampler_two_wave_wave1_denoise_forwards") or 0)
    wave2_tokens = int(totals.get("sampler_two_wave_wave2_value_tokens") or 0)
    wave2_forwards = int(totals.get("sampler_two_wave_wave2_denoise_forwards") or 0)
    records = int(totals.get("records") or 0)
    value_force_counters = {
        "wave1_value_tokens": int(totals.get("sampler_two_wave_wave1_value_tokens") or 0),
        "wave2_forced_tokens": int(totals.get("sampler_two_wave_wave2_forced_tokens") or 0),
        "tool_value_candidate_force_token_visits": int(totals.get("sampler_tool_value_candidate_force_token_visits") or 0),
        "forced_schedule_token_visits": int(totals.get("sampler_forced_schedule_token_visits") or 0),
    }
    return {
        "label": label,
        "tau": tau,
        "split": summary["eval_name"],
        "records": records,
        "exact_args": int(totals.get("exact_arguments") or 0),
        "exact_seq": int(totals.get("exact_tool_sequence") or 0),
        "valid_json": int(totals.get("valid_tool_json") or 0),
        "blended_tpf": round_or_none(ratio(generated, forwards)),
        "scaffold_tpf": round_or_none(ratio(wave1_tokens, wave1_forwards)),
        "value_tpf": round_or_none(ratio(wave2_tokens, wave2_forwards)),
        "seconds_per_record": round_or_none(ratio(float(summary.get("elapsed_seconds") or 0.0), records), 2),
        "wave1_projected_tokens": int(totals.get("sampler_two_wave_wave1_projected_tokens") or 0),
        "value_force_counters": value_force_counters,
        "value_nonleakage_pass": all(value == 0 for value in value_force_counters.values()),
        "summary_json": str(path),
        "output_jsonl": summary.get("out_jsonl"),
    }


def exact_set(rows):
    return {int(row["idx"]) for row in rows if row.get("exact_arguments")}


def row_brief(row):
    return {
        "idx": int(row["idx"]),
        "id": row.get("id"),
        "exact_args": bool(row.get("exact_arguments")),
        "exact_seq": bool(row.get("exact_tool_sequence")),
        "valid_json": bool(row.get("valid_tool_json")),
        "called_names": row.get("called_names") or [],
        "calls": row.get("calls") or [],
        "assistant_excerpt": (row.get("assistant") or "")[:500],
    }


def per_row_diff() -> dict:
    plain = read_jsonl(PLAIN)
    old_a = read_jsonl(OLD_A)
    old_b = read_jsonl(OLD_B)
    percall = read_jsonl(ROOT / "fullproj/tau_095/public_native_12.jsonl")
    plain_exact = exact_set(plain)
    return {
        "plain_exact_rows": sorted(plain_exact),
        "old_A_lost_vs_plain": [
            {
                "idx": 1,
                "classification": "iii_cross_call_separator_stop",
                "finding": "Whole-block A omitted the deterministic inter-call prose/newline separator from wave 1; after call 1 the model emitted stop instead of exposing call 2. Per-call waves plus prose scaffold fixed this row.",
                "plain": row_brief(plain[1]),
                "old_A": row_brief(old_a[1]),
                "percall_tau095": row_brief(percall[1]),
            },
            {
                "idx": 9,
                "classification": "ii_value_corrupt",
                "finding": "Final target value was truncated/corrupted in old A. Per-call waves removed the extra trailing tool call and restored validity, but the final value remained corrupt at tau 0.95 and tau 0.99.",
                "plain": row_brief(plain[9]),
                "old_A": row_brief(old_a[9]),
                "percall_tau095": row_brief(percall[9]),
            },
        ],
        "old_B_lost_vs_plain": [
            {
                "idx": 4,
                "classification": "i_scaffold_tool_name_corrupt",
                "finding": "Careful wave-1 whole-block ordering produced `sschedule_watering` instead of `schedule_watering`.",
                "plain": row_brief(plain[4]),
                "old_B": row_brief(old_b[4]),
            },
            {
                "idx": 5,
                "classification": "i_scaffold_tool_name_corrupt",
                "finding": "Careful wave-1 whole-block ordering produced `set_thermostatmostat_temperature`.",
                "plain": row_brief(plain[5]),
                "old_B": row_brief(old_b[5]),
            },
            {
                "idx": 6,
                "classification": "i_scaffold_tool_name_corrupt",
                "finding": "Careful wave-1 whole-block ordering produced `set_thermostatmostat_schedule`.",
                "plain": row_brief(plain[6]),
                "old_B": row_brief(old_b[6]),
            },
            {
                "idx": 7,
                "classification": "ii_value_boundary_corrupt",
                "finding": "Tool sequence remained correct, but the final `schedule_time` value/close boundary became `</parameterparameter>`, dropping the argument.",
                "plain": row_brief(plain[7]),
                "old_B": row_brief(old_b[7]),
            },
        ],
    }


def split_name(row):
    return "heldout" if "heldout" in row["split"] else "public"


def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def markdown(report: dict) -> str:
    lines = [
        "# Per-Call Wave Schedule Report",
        "",
        "Run-1 checkpoint, raw value lane. Main rows are the requested per-call waves at value tau 0.80 and 0.95; tau 0.99 is an extra conservative diagnostic for the remaining public miss.",
        "",
        "## Per-Row Diff",
        "",
        "- Old A lost public rows 1 and 9 versus the plain-careful 8/12 anchor. Row 1 is a cross-call separator/stop issue and is fixed by per-call waves. Row 9 is value corruption and remains the single public miss after per-call waves.",
        "- Old B lost public rows 4, 5, 6, and 7. Rows 4-6 are tool-name scaffold corruptions; row 7 is a value/close-boundary corruption.",
        "",
        "## Sweep",
        "",
        "| Tau | Split | Quality | Blended TPF | Scaffold TPF | Value TPF | sec/rec | Wave1 projected | Value force pass |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["sweep_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["tau"],
                    split_name(row),
                    f"{row['exact_args']}/{row['records']} args, {row['exact_seq']}/{row['records']} seq, {row['valid_json']}/{row['records']} valid",
                    fmt(row["blended_tpf"]),
                    fmt(row["scaffold_tpf"]),
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
            "## Verdict",
            "",
            "Per-call waves are a real improvement over whole-block A: tau 0.95 moves public exact_args from 6/12 to 7/12 and keeps heldout at 5/12, with blended TPF 1.80 heldout and 2.47 public.",
            "",
            "The target is not met: public remains below the 8/12 anchor. The remaining row 9 miss persists even at tau 0.99, so it is not explained solely by whole-block right-context infill or residual value parallelism at tau 0.95.",
            "",
            "Values remained raw: value force counters are zero in all per-call rows.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", type=Path, default=ROOT / "percall_waves_report.json")
    parser.add_argument("--out-md", type=Path, default=ROOT / "percall_waves_report.md")
    args = parser.parse_args()

    rows = []
    tau_dirs = {"0.80": "tau_080", "0.95": "tau_095", "0.99": "tau_099"}
    for tau in ["0.80", "0.95", "0.99"]:
        tau_dir = tau_dirs[tau]
        for split in ["heldout", "public"]:
            rows.append(summarize(ROOT / f"fullproj/{tau_dir}/{split}_native_12.summary.json", "percall_fullproj", tau))
    report = {
        "anchor": {"plain_careful_public_tau095_exact_args": "8/12"},
        "per_row_diff": per_row_diff(),
        "sweep_rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md)}, indent=2))


if __name__ == "__main__":
    main()
