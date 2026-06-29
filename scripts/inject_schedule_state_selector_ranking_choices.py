#!/usr/bin/env python3
"""Inject schedule-state selector ranking choices into sampler schedules.

The schedule-state selector ranker predicts a candidate index for each active
argument-value span. This script maps those ranked indices back onto an existing
sampler schedule's candidate_sequence_* fields and restricts each matching span
to the top-k ranked candidate sequences.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_SCHEDULE = ROOT / "runs/tool_sensitive_block_plans/public_train_no_public_smoke_sampler_schedule_with_candidates_targetincluded.jsonl"
DEFAULT_RANKING = ROOT / "runs/schedule_state_selector/no_public_smoke_ckpt275_indexonly_rank_all_ambiguous.jsonl"
DEFAULT_OUT = ROOT / "runs/tool_sensitive_block_plans/public_train_no_public_smoke_sampler_schedule_with_schedule_state_selector_rank1.jsonl"


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def case_key(row: dict[str, Any], fallback_idx: int) -> str:
    return str(row.get("id") or row.get("case_id") or fallback_idx)


def selector_key(case_id: str, tool_call_index: Any, json_path: Any, json_key: Any) -> tuple[str, int, str]:
    path = json_path or json_key or ""
    return (str(case_id), int(tool_call_index), str(path))


def selected_token_ids_by_offset(sequence: list[int]) -> list[list[int]]:
    return [[int(token_id)] for token_id in sequence]


def normalize_rank_order(row: dict[str, Any]) -> list[int]:
    order = row.get("rank_order")
    if isinstance(order, list) and order:
        return [int(idx) for idx in order]
    predicted = row.get("predicted_index")
    return [int(predicted)] if predicted is not None else []


def ranking_map(path: Path) -> tuple[dict[tuple[str, int, str], dict[str, Any]], Counter]:
    out: dict[tuple[str, int, str], dict[str, Any]] = {}
    totals = Counter()
    for idx, row in enumerate(load_jsonl(path)):
        try:
            key = selector_key(
                row.get("case_id") or row.get("id"),
                row.get("tool_call_index"),
                row.get("json_path"),
                row.get("json_key"),
            )
        except Exception:
            totals["ranking_rows_without_key"] += 1
            continue
        order = normalize_rank_order(row)
        if not order:
            totals["ranking_rows_without_order"] += 1
            continue
        existing = out.get(key)
        compact = {
            "rank_order": order,
            "predicted_index": int(row.get("predicted_index")),
            "target_index": row.get("target_index"),
            "target_rank": row.get("target_rank"),
            "correct": row.get("correct"),
            "target_margin": row.get("target_margin"),
            "source_idx": row.get("source_idx"),
            "eval_idx": row.get("eval_idx", idx),
        }
        if existing and existing["rank_order"] != compact["rank_order"]:
            totals["ranking_key_conflicts"] += 1
            continue
        out[key] = compact
        totals["ranking_rows"] += 1
    totals["ranking_keys"] = len(out)
    return out, totals


def restrict_item(item: dict[str, Any], selector: dict[str, Any], top_k: int) -> tuple[bool, str | None]:
    values = item.get("candidate_sequence_values") or []
    sequences = item.get("candidate_sequence_token_ids_by_offset") or []
    if not values or not sequences:
        return False, "missing_candidates"
    kept_indices = []
    for idx in selector["rank_order"]:
        if 0 <= idx < len(values) and idx < len(sequences) and idx not in kept_indices:
            kept_indices.append(idx)
        if len(kept_indices) >= top_k:
            break
    if not kept_indices:
        return False, "ranked_indices_out_of_range"

    kept_values = [values[idx] for idx in kept_indices]
    kept_sequences = [[int(token_id) for token_id in sequences[idx]] for idx in kept_indices]
    item["schedule_state_selector_rank_order"] = selector["rank_order"]
    item["schedule_state_selector_kept_indices"] = kept_indices
    item["schedule_state_selector_top_k"] = top_k
    item["schedule_state_selector_correct"] = selector.get("correct")
    item["schedule_state_selector_target_rank"] = selector.get("target_rank")
    item["schedule_state_selector_target_margin"] = selector.get("target_margin")
    item["schedule_state_selector_selected_candidate"] = kept_values[0]
    item["candidate_sequence_values"] = kept_values
    item["candidate_sequence_token_ids_by_offset"] = kept_sequences
    item["candidate_source"] = f"schedule_state_selector_rank{top_k}"
    if top_k == 1:
        item["selected_candidate"] = kept_values[0]
        item["selected_candidate_token_ids_by_offset"] = selected_token_ids_by_offset(kept_sequences[0])
    else:
        item.pop("selected_candidate", None)
        item.pop("selected_candidate_token_ids_by_offset", None)
    return True, None


def item_selector_key(row_key: str, item: dict[str, Any]) -> tuple[str, int, str] | None:
    if item.get("kind") != "argument_value":
        return None
    try:
        return selector_key(
            row_key,
            item.get("tool_call_index"),
            item.get("json_path") or item.get("argument_path"),
            item.get("json_key"),
        )
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule-jsonl", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--ranking-jsonl", type=Path, default=DEFAULT_RANKING)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument(
        "--clear-existing-selected",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clear pre-existing selected_candidate fields before injecting ranked choices.",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        raise SystemExit("--top-k must be >= 1")

    selectors, ranking_totals = ranking_map(args.ranking_jsonl)
    records = []
    totals = Counter(ranking_totals)
    for idx, row in enumerate(load_jsonl(args.schedule_jsonl)):
        row_key = case_key(row, idx)
        row_audit = []
        row_restricted = 0
        row_missing = 0
        for item in row.get("schedule") or []:
            if args.clear_existing_selected:
                item.pop("selected_candidate", None)
                item.pop("selected_candidate_token_ids_by_offset", None)
                item.pop("selector_sidecar_selected_candidate", None)
                item.pop("pairwise_tournament_selected_candidate", None)
            key = item_selector_key(row_key, item)
            if key is None:
                continue
            selector = selectors.get(key)
            if not selector:
                totals["argument_items_without_selector"] += 1
                continue
            ok, reason = restrict_item(item, selector, args.top_k)
            if ok:
                row_restricted += 1
                totals["restricted_argument_items"] += 1
            else:
                row_missing += 1
                totals[f"restriction_failed:{reason}"] += 1
            row_audit.append(
                {
                    "tool_call_index": key[1],
                    "json_path": key[2],
                    "top_k": args.top_k,
                    "rank_order": selector.get("rank_order"),
                    "predicted_index": selector.get("predicted_index"),
                    "target_index": selector.get("target_index"),
                    "target_rank": selector.get("target_rank"),
                    "correct": selector.get("correct"),
                    "target_margin": selector.get("target_margin"),
                    "restricted": ok,
                    "failure_reason": reason,
                }
            )
        if row_audit:
            row["schedule_state_selector_ranking_choices"] = row_audit
            row["schedule_state_selector_ranking_summary"] = {
                "top_k": args.top_k,
                "restricted_argument_items": row_restricted,
                "failed_argument_items": row_missing,
            }
            totals["records_with_selector_choices"] += 1
        else:
            totals["records_without_selector_choices"] += 1
        totals["records"] += 1
        records.append(row)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "schedule_jsonl": str(args.schedule_jsonl),
        "ranking_jsonl": str(args.ranking_jsonl),
        "out_jsonl": str(args.out_jsonl),
        "top_k": args.top_k,
        "clear_existing_selected": args.clear_existing_selected,
        "totals": dict(totals),
    }
    args.out_jsonl.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
