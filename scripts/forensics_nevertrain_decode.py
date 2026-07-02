#!/usr/bin/env python3
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_toolcall_jsonl import (  # noqa: E402
    extract_tool_calls,
    normalize_call_for_compare,
    tool_schema_by_name,
)
from plan_tool_sensitive_blocks import load_tokenizer  # noqa: E402


PARAMETER_RE = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.DOTALL)


def load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def events(row):
    return ((row.get("backend_meta") or {}).get("sampler_schedule_events") or {})


def event_int(row, key):
    try:
        return int(events(row).get(key) or 0)
    except Exception:
        return 0


def value_spans(text):
    spans = []
    for match in PARAMETER_RE.finditer(text or ""):
        start, end = match.span(2)
        if start < end and text[start] == "\n":
            start += 1
        if end > start and text[end - 1] == "\n":
            end -= 1
        if start < end:
            spans.append(
                {
                    "key": match.group(1),
                    "start": start,
                    "end": end,
                    "text": text[start:end],
                }
            )
    return spans


def token_count_for_spans(tokenizer, text, spans):
    if not spans:
        return 0
    encoded = tokenizer(text or "", add_special_tokens=False, return_offsets_mapping=True)
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    count = 0
    for token_start, token_end in offsets:
        if token_end <= token_start:
            continue
        if any(token_end > span["start"] and token_start < span["end"] for span in spans):
            count += 1
    return count


def output_value_token_count(tokenizer, row):
    text = row.get("assistant") or ""
    return token_count_for_spans(tokenizer, text, value_spans(text))


def backend_summary(rows):
    totals = Counter()
    forwards_hist = Counter()
    episodes = defaultdict(list)
    source_rows = defaultdict(list)
    has_model_forward_counters = False
    for row in rows:
        totals["turns"] += 1
        totals["exact_args"] += int(bool(row.get("exact_arguments")))
        totals["exact_seq"] += int(bool(row.get("exact_tool_sequence")))
        totals["valid_xml"] += int(bool(row.get("valid_tool_call")))
        totals["schema_ok"] += int(bool(row.get("all_schema_valid")))
        totals["generated_tokens"] += int(row.get("generated_token_count") or 0)
        totals["wall_seconds_millis"] += round(float(row.get("turn_wall_seconds") or 0.0) * 1000)
        fw = event_int(row, "denoise_forwards_total")
        has_counter = "denoise_forwards_total" in events(row)
        has_model_forward_counters = has_model_forward_counters or has_counter
        totals["model_forwards"] += fw
        if has_counter:
            forwards_hist[fw] += 1
        episodes[row.get("episode_idx")].append(bool(row.get("exact_arguments")))
        source_rows[row.get("source_family") or row.get("source") or "unknown"].append(row)
    total_turns = int(totals["turns"])
    total_wall = float(totals["wall_seconds_millis"]) / 1000.0
    summary = {
        "turns": total_turns,
        "exact_args": int(totals["exact_args"]),
        "episode_exact": sum(1 for values in episodes.values() if all(values)),
        "episodes": len(episodes),
        "exact_seq": int(totals["exact_seq"]),
        "valid_xml": int(totals["valid_xml"]),
        "schema_ok": int(totals["schema_ok"]),
        "total_wall_seconds": total_wall,
        "sec_per_turn": total_wall / total_turns if total_turns else 0.0,
        "generated_tokens_per_turn": float(totals["generated_tokens"]) / total_turns if total_turns else 0.0,
        "model_forwards_total": int(totals["model_forwards"]) if has_model_forward_counters else None,
        "model_forwards_per_turn": (
            float(totals["model_forwards"]) / total_turns
            if total_turns and has_model_forward_counters
            else None
        ),
        "forwards_histogram": {str(key): int(value) for key, value in sorted(forwards_hist.items())},
    }
    by_source = {}
    for source, source_subset in sorted(source_rows.items()):
        by_source[source] = backend_summary_no_sources(source_subset)
    summary["by_source"] = by_source
    return summary


def backend_summary_no_sources(rows):
    totals = Counter()
    episodes = defaultdict(list)
    for row in rows:
        totals["turns"] += 1
        totals["exact_args"] += int(bool(row.get("exact_arguments")))
        totals["exact_seq"] += int(bool(row.get("exact_tool_sequence")))
        totals["valid_xml"] += int(bool(row.get("valid_tool_call")))
        totals["schema_ok"] += int(bool(row.get("all_schema_valid")))
        totals["wall_seconds_millis"] += round(float(row.get("turn_wall_seconds") or 0.0) * 1000)
        episodes[row.get("episode_idx")].append(bool(row.get("exact_arguments")))
    turns = int(totals["turns"])
    wall = float(totals["wall_seconds_millis"]) / 1000.0
    return {
        "turns": turns,
        "exact_args": int(totals["exact_args"]),
        "episode_exact": sum(1 for values in episodes.values() if all(values)),
        "episodes": len(episodes),
        "exact_seq": int(totals["exact_seq"]),
        "valid_xml": int(totals["valid_xml"]),
        "schema_ok": int(totals["schema_ok"]),
        "sec_per_turn": wall / turns if turns else 0.0,
    }


def projected_value_audit(tokenizer, rows):
    per_turn = []
    totals = Counter()
    for idx, row in enumerate(rows):
        fw = event_int(row, "denoise_forwards_total")
        projected = event_int(row, "two_wave_wave1_projected_tokens")
        true_value_tokens = output_value_token_count(tokenizer, row)
        model_value_tokens = max(
            event_int(row, "two_wave_wave2_value_tokens"),
            event_int(row, "parallel_commit_value_tokens"),
        ) + event_int(row, "two_wave_wave1_value_tokens")
        if fw == 0:
            projected_value_lb = true_value_tokens
        else:
            projected_value_lb = max(0, true_value_tokens - model_value_tokens)
        projected_value_lb = min(projected_value_lb, projected)
        projected_scaffold_ub = max(0, projected - projected_value_lb)
        dependent = bool(row.get("exact_arguments")) and projected_value_lb > 0
        out = {
            "row_idx": idx,
            "episode_idx": row.get("episode_idx"),
            "episode_id": row.get("episode_id"),
            "turn_idx": row.get("turn_idx"),
            "source": row.get("source"),
            "source_family": row.get("source_family"),
            "exact_arguments": bool(row.get("exact_arguments")),
            "denoise_forwards_total": fw,
            "wave1_projected_tokens": projected,
            "true_xml_value_tokens": true_value_tokens,
            "reported_model_value_tokens": model_value_tokens,
            "projected_true_value_tokens_lower_bound": projected_value_lb,
            "projected_scaffold_tokens_upper_bound": projected_scaffold_ub,
            "exact_depends_on_projected_values": dependent,
        }
        per_turn.append(out)
        totals["turns"] += 1
        totals["zero_forward_turns"] += int(fw == 0)
        totals["zero_forward_turns_with_values"] += int(fw == 0 and true_value_tokens > 0)
        totals["exact_args"] += int(bool(row.get("exact_arguments")))
        totals["exact_turns_dependent_on_projected_values"] += int(dependent)
        totals["wave1_projected_tokens"] += projected
        totals["true_xml_value_tokens"] += true_value_tokens
        totals["reported_model_value_tokens"] += model_value_tokens
        totals["projected_true_value_tokens_lower_bound"] += projected_value_lb
        totals["projected_scaffold_tokens_upper_bound"] += projected_scaffold_ub
        totals["strict_old_no_value_projection_exact"] += int(
            bool(row.get("exact_arguments")) and projected_value_lb == 0
        )
    return {"totals": dict(totals), "turns": per_turn}


def no_value_projection_check(tokenizer, rows):
    totals = Counter()
    for row in rows:
        fw = event_int(row, "denoise_forwards_total")
        true_value_tokens = output_value_token_count(tokenizer, row)
        totals["turns"] += 1
        totals["zero_forward_turns"] += int(fw == 0)
        totals["zero_forward_turns_with_values"] += int(fw == 0 and true_value_tokens > 0)
        totals["wave1_value_tokens"] += event_int(row, "two_wave_wave1_value_tokens")
        totals["wave2_value_tokens"] += event_int(row, "two_wave_wave2_value_tokens")
        totals["parallel_commit_value_tokens"] += event_int(row, "parallel_commit_value_tokens")
        totals["wave1_projected_tokens"] += event_int(row, "two_wave_wave1_projected_tokens")
        totals["model_forwards"] += fw
    return dict(totals)


def canonical_call_list(calls, schemas):
    return [normalize_call_for_compare(call, schemas) for call in calls]


def canonical_sort_key(call):
    return json.dumps(call, ensure_ascii=False, sort_keys=True, default=str)


def classify_ar_guided_failure(row, case):
    calls, invalid = extract_tool_calls(row.get("assistant") or "")
    gold_calls, gold_invalid = extract_tool_calls(row.get("gold_assistant") or "")
    schemas = tool_schema_by_name(case.get("tools") or [])
    normalized = canonical_call_list(calls, schemas)
    normalized_gold = canonical_call_list(gold_calls, schemas)
    if invalid or not calls or gold_invalid:
        return "formatting/type variant"
    if sorted(normalized, key=canonical_sort_key) == sorted(normalized_gold, key=canonical_sort_key):
        return "key-order/canonicalization artifact"
    if len(normalized) != len(normalized_gold):
        return "missing/extra arg"
    if [call.get("name") for call in normalized] != [call.get("name") for call in normalized_gold]:
        return "missing/extra arg"
    for got, want in zip(normalized, normalized_gold):
        got_args = got.get("arguments") if isinstance(got.get("arguments"), dict) else {}
        want_args = want.get("arguments") if isinstance(want.get("arguments"), dict) else {}
        if set(got_args) != set(want_args):
            return "missing/extra arg"
    if normalized == normalized_gold:
        return "formatting/type variant"
    return "wrong value"


def order_insensitive_exact(row, case):
    calls, invalid = extract_tool_calls(row.get("assistant") or "")
    gold_calls, gold_invalid = extract_tool_calls(row.get("gold_assistant") or "")
    if invalid or gold_invalid:
        return False
    schemas = tool_schema_by_name(case.get("tools") or [])
    normalized = canonical_call_list(calls, schemas)
    normalized_gold = canonical_call_list(gold_calls, schemas)
    return sorted(normalized, key=canonical_sort_key) == sorted(normalized_gold, key=canonical_sort_key)


def ar_guided_taxonomy(rows, cases, sample_size):
    failures = []
    counts = Counter()
    all_failure_counts = Counter()
    order_insensitive_recovers = 0
    for idx, row in enumerate(rows):
        case = cases[int(row.get("episode_idx") or 0)]
        recovered = order_insensitive_exact(row, case)
        if not row.get("exact_arguments") and recovered:
            order_insensitive_recovers += 1
        if row.get("exact_arguments"):
            continue
        category = classify_ar_guided_failure(row, case)
        calls, invalid = extract_tool_calls(row.get("assistant") or "")
        gold_calls, gold_invalid = extract_tool_calls(row.get("gold_assistant") or "")
        all_failure_counts[category] += 1
        if len(failures) < sample_size:
            counts[category] += 1
            failures.append(
                {
                    "row_idx": idx,
                    "episode_idx": row.get("episode_idx"),
                    "episode_id": row.get("episode_id"),
                    "turn_idx": row.get("turn_idx"),
                    "source": row.get("source"),
                    "source_family": row.get("source_family"),
                    "category": category,
                    "called_names": row.get("called_names"),
                    "generated_calls": calls,
                    "generated_invalid_count": invalid,
                    "gold_calls": gold_calls,
                    "gold_invalid_count": gold_invalid,
                    "gold_excerpt": (row.get("gold_assistant") or "")[:400],
                    "assistant_excerpt": (row.get("assistant") or "")[:400],
                }
            )
    return {
        "failure_count": sum(all_failure_counts.values()),
        "sample_size": len(failures),
        "sample_counts": dict(counts),
        "all_failure_counts": dict(all_failure_counts),
        "order_insensitive_recovers": order_insensitive_recovers,
        "sample": failures,
    }


def corrected_table(backends):
    rows = []
    for label, summary in backends.items():
        rows.append(
            {
                "backend": label,
                "exact_args": f"{summary['exact_args']}/{summary['turns']}",
                "episode_exact": f"{summary['episode_exact']}/{summary['episodes']}",
                "exact_seq": f"{summary['exact_seq']}/{summary['turns']}",
                "valid_xml": f"{summary['valid_xml']}/{summary['turns']}",
                "schema_ok": f"{summary['schema_ok']}/{summary['turns']}",
                "sec_per_turn": round(summary["sec_per_turn"], 3),
                "total_wall_seconds": round(summary["total_wall_seconds"], 3),
                "model_forwards_per_turn": (
                    "n/a"
                    if summary["model_forwards_per_turn"] is None
                    else round(summary["model_forwards_per_turn"], 3)
                ),
            }
        )
    return rows


def markdown_table(rows, columns):
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---" for _ in columns]) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def write_report(path, summary):
    corrected_rows = summary["corrected_three_way_table"]
    audit = summary["old_diffusion_projection_audit"]["totals"]
    corrected_check = summary["corrected_diffusion_no_value_projection_check"]
    taxonomy = summary["ar_guided_failure_taxonomy"]
    backends = summary["backend_summaries"]
    lines = [
        "# Never-Train Decode Forensics",
        "",
        "## Verdict",
        "",
        "- The published diffusion `181/184` never-train number is invalid for model-quality claims.",
        f"- Old diffusion forwards histogram: `{backends['Diffusion per-call waves (tainted)']['forwards_histogram']}`.",
        f"- Old diffusion zero-forward turns: {audit['zero_forward_turns']}/{audit['turns']}; zero-forward turns with XML values: {audit['zero_forward_turns_with_values']}/{audit['turns']}.",
        f"- Exact turns dependent on projected values, lower bound: {audit['exact_turns_dependent_on_projected_values']}/{audit['exact_args']}.",
        f"- Strict old-artifact no-value-projection exact, lower-bound filter: {audit['strict_old_no_value_projection_exact']}/{audit['turns']}.",
        f"- Corrected no-value-projection rerun: {backends['Diffusion per-call waves (corrected)']['exact_args']}/{backends['Diffusion per-call waves (corrected)']['turns']} exact_args, {backends['Diffusion per-call waves (corrected)']['model_forwards_per_turn']:.3f} forwards/turn.",
        f"- Corrected forwards histogram: `{backends['Diffusion per-call waves (corrected)']['forwards_histogram']}`.",
        f"- Corrected wave-1 value tokens: {corrected_check['wave1_value_tokens']}; corrected zero-forward turns: {corrected_check['zero_forward_turns']}.",
        "",
        "## Corrected Three-Way Table",
        "",
        markdown_table(
            corrected_rows,
            [
                "backend",
                "exact_args",
                "episode_exact",
                "exact_seq",
                "valid_xml",
                "schema_ok",
                "sec_per_turn",
                "total_wall_seconds",
                "model_forwards_per_turn",
            ],
        ),
        "",
        "## Value Projection Split",
        "",
        f"- Old wave-1 projected tokens: {audit['wave1_projected_tokens']}.",
        f"- True XML value tokens in old outputs: {audit['true_xml_value_tokens']}.",
        f"- Projected true-value tokens lower bound: {audit['projected_true_value_tokens_lower_bound']}.",
        f"- Projected scaffold tokens upper bound: {audit['projected_scaffold_tokens_upper_bound']}.",
        "",
        "The lower bound is exact for zero-forward turns. For nonzero turns the old logs do not retain per-token source, so the split credits model-sampled value tokens as generously as the counters allow.",
        "",
        "## AR-Guided Failure Taxonomy",
        "",
        f"- AR-guided failures: {taxonomy['failure_count']}.",
        f"- First-20 sample counts: {taxonomy['sample_counts']}.",
        f"- All-failure counts: {taxonomy['all_failure_counts']}.",
        f"- Order-insensitive/type-coerced recoveries: {taxonomy['order_insensitive_recovers']}.",
        "",
        "Formatting/type/canonicalization artifacts do not dominate, so no scorer fix was applied.",
        "",
        "## Source Breakdown",
        "",
    ]
    source_rows = []
    for backend, backend_summary_value in summary["backend_summaries"].items():
        if backend == "Diffusion per-call waves (tainted)":
            continue
        for source, source_summary in backend_summary_value["by_source"].items():
            source_rows.append(
                {
                    "source": source,
                    "backend": backend,
                    "exact_args": f"{source_summary['exact_args']}/{source_summary['turns']}",
                    "episode_exact": f"{source_summary['episode_exact']}/{source_summary['episodes']}",
                    "sec_per_turn": round(source_summary["sec_per_turn"], 3),
                }
            )
    lines.append(markdown_table(source_rows, ["source", "backend", "exact_args", "episode_exact", "sec_per_turn"]))
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Forensic summary JSON: `{summary['artifacts']['summary_json']}`",
            f"- Per-turn projection audit JSONL: `{summary['artifacts']['projection_audit_jsonl']}`",
            f"- AR-guided taxonomy JSONL: `{summary['artifacts']['taxonomy_jsonl']}`",
            f"- Corrected diffusion turns: `{summary['inputs']['corrected_diffusion_turns']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--old-run-dir", type=Path, required=True)
    parser.add_argument("--corrected-run-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--taxonomy-sample-size", type=int, default=20)
    args = parser.parse_args()

    cases = load_jsonl(args.input_jsonl)
    old_diffusion = load_jsonl(args.old_run_dir / "diffusion" / "turns.jsonl")
    corrected_diffusion = load_jsonl(args.corrected_run_dir / "diffusion" / "turns.jsonl")
    ar_rows = load_jsonl(args.old_run_dir / "ar-vllm" / "turns.jsonl")
    ar_guided_rows = load_jsonl(args.old_run_dir / "ar-vllm-guided" / "turns.jsonl")
    tokenizer = load_tokenizer(args.tokenizer_path)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    projection_audit = projected_value_audit(tokenizer, old_diffusion)
    corrected_projection_check = no_value_projection_check(tokenizer, corrected_diffusion)
    projection_audit_jsonl = args.out_dir / "projection_value_audit.jsonl"
    write_jsonl(projection_audit_jsonl, projection_audit["turns"])

    taxonomy = ar_guided_taxonomy(ar_guided_rows, cases, args.taxonomy_sample_size)
    taxonomy_jsonl = args.out_dir / "ar_guided_failure_taxonomy.jsonl"
    write_jsonl(taxonomy_jsonl, taxonomy["sample"])

    backend_summaries = {
        "AR vLLM FR13": backend_summary(ar_rows),
        "AR vLLM FR13 guided": backend_summary(ar_guided_rows),
        "Diffusion per-call waves (tainted)": backend_summary(old_diffusion),
        "Diffusion per-call waves (corrected)": backend_summary(corrected_diffusion),
    }
    corrected_backends = {
        "AR vLLM FR13": backend_summaries["AR vLLM FR13"],
        "AR vLLM FR13 guided": backend_summaries["AR vLLM FR13 guided"],
        "Diffusion per-call waves (corrected)": backend_summaries[
            "Diffusion per-call waves (corrected)"
        ],
    }
    summary_json = args.out_dir / "forensics_summary.json"
    report_md = args.out_dir / "forensics_report.md"
    summary = {
        "inputs": {
            "input_jsonl": str(args.input_jsonl),
            "old_run_dir": str(args.old_run_dir),
            "corrected_run_dir": str(args.corrected_run_dir),
            "corrected_diffusion_turns": str(args.corrected_run_dir / "diffusion" / "turns.jsonl"),
            "tokenizer_path": str(args.tokenizer_path),
        },
        "backend_summaries": backend_summaries,
        "old_diffusion_projection_audit": {
            "totals": projection_audit["totals"],
            "per_turn_jsonl": str(projection_audit_jsonl),
        },
        "corrected_diffusion_no_value_projection_check": corrected_projection_check,
        "ar_guided_failure_taxonomy": taxonomy,
        "corrected_three_way_table": corrected_table(corrected_backends),
        "artifacts": {
            "summary_json": str(summary_json),
            "report_md": str(report_md),
            "projection_audit_jsonl": str(projection_audit_jsonl),
            "taxonomy_jsonl": str(taxonomy_jsonl),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(report_md, summary)
    print(json.dumps(summary["corrected_three_way_table"], indent=2), flush=True)
    print(f"wrote {summary_json}")
    print(f"wrote {report_md}")


if __name__ == "__main__":
    main()
