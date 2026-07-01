#!/usr/bin/env python3
"""Compare tau2 real solo AR and diffusion runs, including capability and speed."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_AR = Path("runs/agentic_eval/tau2_real_solo_openai_ar9b.jsonl")
DEFAULT_DIFFUSION = Path("runs/agentic_eval/tau2_real_solo_diffusion9b_memfix.jsonl")
DEFAULT_OUT_JSON = Path("runs/agentic_eval/tau2_real_solo_memfix_comparison.json")
DEFAULT_OUT_MD = Path("runs/agentic_eval/tau2_real_solo_memfix_report.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ar-jsonl", type=Path, default=DEFAULT_AR)
    parser.add_argument("--diffusion-jsonl", type=Path, default=DEFAULT_DIFFUSION)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--title", default="tau2 real solo memfix comparison")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def generation_events(row: dict[str, Any]) -> list[dict[str, Any]]:
    if "events" in row:
        return list(row.get("events") or [])
    if "turns" in row:
        return list(row.get("turns") or [])
    return []


def tool_names(row: dict[str, Any]) -> list[str]:
    calls = row.get("tool_calls")
    if calls is None:
        calls = row.get("effective_calls")
    return [str(call.get("name")) for call in (calls or []) if call.get("name")]


def action_matches_tool_call(action: dict[str, Any], tool_call: dict[str, Any]) -> bool:
    if action.get("name") != tool_call.get("name"):
        return False
    tool_args = tool_call.get("arguments") or {}
    action_args = action.get("arguments") or {}
    compare_args = action.get("compare_args")
    if compare_args is None:
        compare_args = list(tool_args.keys())
    if len(compare_args) == 0:
        return True
    predicted = {key: tool_args.get(key) for key in compare_args if key in tool_args}
    expected = {key: action_args.get(key) for key in compare_args if key in action_args}
    return predicted == expected


def partial_action_stats(row: dict[str, Any]) -> dict[str, Any]:
    expected = list(row.get("expected_actions") or [])
    calls = list(row.get("tool_calls") or row.get("effective_calls") or [])
    matches = []
    for action in expected:
        matched = any(action_matches_tool_call(action, call) for call in calls)
        matches.append(
            {
                "action_id": action.get("action_id"),
                "name": action.get("name"),
                "matched": matched,
            }
        )
    matched_count = sum(1 for item in matches if item["matched"])
    total = len(matches)
    return {
        "action_matches": matched_count,
        "action_total": total,
        "partial_action_score": matched_count / total if total else None,
        "action_match_details": matches,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lane_totals: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "records": 0,
            "reward": 0.0,
            "partial_action_records": 0,
            "action_matches": 0,
            "action_total": 0,
            "generated_tokens": 0,
            "backend_generation_seconds": 0.0,
            "end_to_end_seconds": 0.0,
            "denoise_forwards": 0,
            "cache_advance_calls": 0,
            "denoise_records": 0,
            "max_prompt_tokens": None,
            "cuda_max_memory_allocated_gib": None,
            "cuda_max_memory_reserved_gib": None,
            "failures": Counter(),
            "termination_reasons": Counter(),
            "called_tools": Counter(),
            "tool_error_count": 0,
            "protected_blocked_calls": 0,
        }
    )
    for row in rows:
        lane = str(row.get("lane") or "unknown")
        totals = lane_totals[lane]
        totals["records"] += 1
        totals["reward"] += float(row.get("reward") or 0.0)
        action_stats = partial_action_stats(row)
        if action_stats["action_total"]:
            totals["partial_action_records"] += 1
            totals["action_matches"] += int(action_stats["action_matches"])
            totals["action_total"] += int(action_stats["action_total"])
        totals["failures"].update(row.get("failures") or [])
        if row.get("termination_reason"):
            totals["termination_reasons"].update([row["termination_reason"]])
        totals["called_tools"].update(tool_names(row))
        totals["tool_error_count"] += int(row.get("tool_error_count") or 0)
        totals["protected_blocked_calls"] += len(row.get("protected_blocked_calls") or [])
        totals["end_to_end_seconds"] += float(row.get("seconds") or 0.0)
        for event in generation_events(row):
            totals["generated_tokens"] += int(event.get("tokens") or 0)
            totals["backend_generation_seconds"] += float(event.get("seconds") or 0.0)
            meta = event.get("backend_meta") or {}
            prompt_tokens = meta.get("prompt_tokens")
            if prompt_tokens is not None:
                previous = totals["max_prompt_tokens"]
                totals["max_prompt_tokens"] = int(prompt_tokens) if previous is None else max(previous, int(prompt_tokens))
            memory = meta.get("cuda_memory") or {}
            allocated = memory.get("max_allocated_gib")
            reserved = memory.get("max_reserved_gib")
            if allocated is not None:
                previous = totals["cuda_max_memory_allocated_gib"]
                totals["cuda_max_memory_allocated_gib"] = (
                    float(allocated) if previous is None else max(previous, float(allocated))
                )
            if reserved is not None:
                previous = totals["cuda_max_memory_reserved_gib"]
                totals["cuda_max_memory_reserved_gib"] = (
                    float(reserved) if previous is None else max(previous, float(reserved))
                )
            cache_stats = meta.get("flare_cache_stats") or {}
            if cache_stats:
                totals["denoise_records"] += 1
                totals["denoise_forwards"] += int(cache_stats.get("read_calls") or 0)
                totals["cache_advance_calls"] += int(cache_stats.get("advance_calls") or 0)

    lanes = {}
    for lane, totals in lane_totals.items():
        records = max(1, int(totals["records"]))
        tokens = int(totals["generated_tokens"])
        backend_seconds = float(totals["backend_generation_seconds"])
        end_to_end_seconds = float(totals["end_to_end_seconds"])
        denoise_records = int(totals["denoise_records"])
        lanes[lane] = {
            **{key: value for key, value in totals.items() if not isinstance(value, Counter)},
            "score": float(totals["reward"]) / records,
            "partial_action_score": (
                float(totals["action_matches"]) / float(totals["action_total"])
                if totals["action_total"]
                else None
            ),
            "backend_tokens_per_second": tokens / backend_seconds if backend_seconds > 0 else None,
            "end_to_end_tokens_per_second": tokens / end_to_end_seconds if end_to_end_seconds > 0 else None,
            "denoise_forwards_per_token": (
                float(totals["denoise_forwards"]) / tokens if denoise_records and tokens > 0 else None
            ),
            "cache_advance_calls_per_token": (
                float(totals["cache_advance_calls"]) / tokens if denoise_records and tokens > 0 else None
            ),
            "failures": dict(totals["failures"]),
            "termination_reasons": dict(totals["termination_reasons"]),
            "called_tools": dict(totals["called_tools"].most_common(20)),
        }

    aggregate = aggregate_lanes(lanes)
    return {"records": len(rows), "lanes": lanes, "overall": aggregate}


def aggregate_lanes(lanes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    records = sum(int(lane.get("records") or 0) for lane in lanes.values())
    reward = sum(float(lane.get("reward") or 0.0) for lane in lanes.values())
    action_matches = sum(int(lane.get("action_matches") or 0) for lane in lanes.values())
    action_total = sum(int(lane.get("action_total") or 0) for lane in lanes.values())
    tokens = sum(int(lane.get("generated_tokens") or 0) for lane in lanes.values())
    backend_seconds = sum(float(lane.get("backend_generation_seconds") or 0.0) for lane in lanes.values())
    end_to_end_seconds = sum(float(lane.get("end_to_end_seconds") or 0.0) for lane in lanes.values())
    denoise_forwards = sum(int(lane.get("denoise_forwards") or 0) for lane in lanes.values())
    denoise_records = sum(int(lane.get("denoise_records") or 0) for lane in lanes.values())
    cache_advance_calls = sum(int(lane.get("cache_advance_calls") or 0) for lane in lanes.values())
    failures = Counter()
    terminations = Counter()
    called = Counter()
    for lane in lanes.values():
        failures.update(lane.get("failures") or {})
        terminations.update(lane.get("termination_reasons") or {})
        called.update(lane.get("called_tools") or {})
    max_allocated_values = [
        float(lane["cuda_max_memory_allocated_gib"])
        for lane in lanes.values()
        if lane.get("cuda_max_memory_allocated_gib") is not None
    ]
    max_reserved_values = [
        float(lane["cuda_max_memory_reserved_gib"])
        for lane in lanes.values()
        if lane.get("cuda_max_memory_reserved_gib") is not None
    ]
    prompt_values = [int(lane["max_prompt_tokens"]) for lane in lanes.values() if lane.get("max_prompt_tokens") is not None]
    return {
        "records": records,
        "score": reward / max(1, records),
        "partial_action_score": action_matches / action_total if action_total else None,
        "action_matches": action_matches,
        "action_total": action_total,
        "generated_tokens": tokens,
        "backend_generation_seconds": backend_seconds,
        "backend_tokens_per_second": tokens / backend_seconds if backend_seconds > 0 else None,
        "end_to_end_seconds": end_to_end_seconds,
        "end_to_end_tokens_per_second": tokens / end_to_end_seconds if end_to_end_seconds > 0 else None,
        "denoise_forwards": denoise_forwards if denoise_records else None,
        "denoise_forwards_per_token": denoise_forwards / tokens if denoise_records and tokens > 0 else None,
        "cache_advance_calls": cache_advance_calls if denoise_records else None,
        "cache_advance_calls_per_token": cache_advance_calls / tokens if denoise_records and tokens > 0 else None,
        "max_prompt_tokens": max(prompt_values) if prompt_values else None,
        "cuda_max_memory_allocated_gib": max(max_allocated_values) if max_allocated_values else None,
        "cuda_max_memory_reserved_gib": max(max_reserved_values) if max_reserved_values else None,
        "failures": dict(failures),
        "termination_reasons": dict(terminations),
        "called_tools": dict(called.most_common(20)),
    }


def divide(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return a / b


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_report(report: dict[str, Any]) -> str:
    ar = report["backends"]["ar"]
    diffusion = report["backends"]["diffusion"]
    ratio = report["speed_ratio"]
    lines = [
        f"# {report['title']}",
        "",
        "## Overall",
        "",
        "| backend | records | binary score | partial action | backend tok/s | e2e tok/s | denoise forwards/token | max prompt | max CUDA reserved GiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, summary in [("AR", ar), ("diffusion", diffusion)]:
        overall = summary["overall"]
        lines.append(
            "| {name} | {records} | {score} | {partial} | {backend_tps} | {e2e_tps} | {denoise} | {prompt} | {reserved} |".format(
                name=name,
                records=overall["records"],
                score=fmt(overall["score"]),
                partial=fmt(overall["partial_action_score"]),
                backend_tps=fmt(overall["backend_tokens_per_second"]),
                e2e_tps=fmt(overall["end_to_end_tokens_per_second"]),
                denoise=fmt(overall["denoise_forwards_per_token"]),
                prompt=fmt(overall["max_prompt_tokens"], 0),
                reserved=fmt(overall["cuda_max_memory_reserved_gib"]),
            )
        )
    lines.extend(
        [
            "",
            "## Speed Ratio",
            "",
            f"- diffusion / AR backend tok/s: {fmt(ratio['backend_tokens_per_second'])}",
            f"- diffusion / AR end-to-end tok/s: {fmt(ratio['end_to_end_tokens_per_second'])}",
            "- target: diffusion / AR backend tok/s >= 10.0",
            "",
            "## Lanes",
            "",
            "| backend | lane | n | binary score | partial action | backend tok/s | failures |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for backend_name, summary in [("AR", ar), ("diffusion", diffusion)]:
        for lane in sorted(summary["lanes"]):
            lane_summary = summary["lanes"][lane]
            failures = ", ".join(f"{key}:{value}" for key, value in lane_summary["failures"].items()) or "none"
            lines.append(
                "| {backend} | {lane} | {records} | {score} | {partial} | {tps} | {failures} |".format(
                    backend=backend_name,
                    lane=lane,
                    records=lane_summary["records"],
                    score=fmt(lane_summary["score"]),
                    partial=fmt(lane_summary["partial_action_score"]),
                    tps=fmt(lane_summary["backend_tokens_per_second"]),
                    failures=failures,
                )
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    ar = summarize_rows(load_jsonl(args.ar_jsonl))
    diffusion = summarize_rows(load_jsonl(args.diffusion_jsonl))
    report = {
        "inputs": {"ar_jsonl": str(args.ar_jsonl), "diffusion_jsonl": str(args.diffusion_jsonl)},
        "title": args.title,
        "backends": {"ar": ar, "diffusion": diffusion},
        "speed_ratio": {
            "backend_tokens_per_second": divide(
                diffusion["overall"]["backend_tokens_per_second"], ar["overall"]["backend_tokens_per_second"]
            ),
            "end_to_end_tokens_per_second": divide(
                diffusion["overall"]["end_to_end_tokens_per_second"], ar["overall"]["end_to_end_tokens_per_second"]
            ),
        },
        "speed_target": {"diffusion_over_ar_backend_tokens_per_second": 10.0},
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
