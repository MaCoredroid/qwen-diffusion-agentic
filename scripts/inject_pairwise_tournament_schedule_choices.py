#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_SCHEDULE = ROOT / "runs/tool_sensitive_block_plans/public_multicall_gold_sampler_schedule_with_candidates_targetselected_v5_12.jsonl"
DEFAULT_SELECTOR = ROOT / "runs/candidate_ranking/public_multicall_targetcandidate_v5_miss_targets_ckpt275_pairwise_path_tournament.jsonl"
DEFAULT_OUT = ROOT / "runs/tool_sensitive_block_plans/public_multicall_gold_sampler_schedule_with_pairwise_path_choices_v5_12.jsonl"


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def case_key(row, fallback_idx):
    return row.get("id") or row.get("case_id") or str(fallback_idx)


def normalize(value):
    return str(value).strip().strip('"')


def selected_token_ids_by_offset(sequence):
    return [[int(token_id)] for token_id in sequence]


def restrict_item(item, predicted):
    values = item.get("candidate_sequence_values") or []
    sequences = item.get("candidate_sequence_token_ids_by_offset") or []
    predicted_norm = normalize(predicted)
    for idx, value in enumerate(values):
        if normalize(value) != predicted_norm or idx >= len(sequences):
            continue
        sequence = [int(token_id) for token_id in sequences[idx]]
        item["pairwise_tournament_selected_candidate"] = predicted
        item["candidate_sequence_values"] = [value]
        item["candidate_sequence_token_ids_by_offset"] = [sequence]
        item["selected_candidate"] = predicted
        item["selected_candidate_token_ids_by_offset"] = selected_token_ids_by_offset(sequence)
        item["candidate_source"] = "pairwise_path_tournament"
        return True
    return False


def selector_rows(paths):
    out = defaultdict(list)
    for path in paths:
        for idx, row in enumerate(load_jsonl(path)):
            key = case_key(row, idx)
            if row.get("predicted_value") is None:
                continue
            out[key].append(row)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule-jsonl", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument(
        "--selector-jsonl",
        type=Path,
        action="append",
        default=[],
        help="Tournament selector JSONL. May be passed more than once.",
    )
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--clear-existing-selected",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove pre-existing selected-candidate fields before injecting tournament choices.",
    )
    parser.add_argument(
        "--include-kinds",
        nargs="+",
        choices=["tool_name", "argument_value"],
        default=None,
        help="Restrict injection to these selector kinds. Defaults to all selector kinds.",
    )
    args = parser.parse_args()

    selector_paths = args.selector_jsonl or [DEFAULT_SELECTOR]
    selectors = selector_rows(selector_paths)
    records = []
    totals = Counter()
    for idx, row in enumerate(load_jsonl(args.schedule_jsonl)):
        key = case_key(row, idx)
        if args.clear_existing_selected:
            for item in row.get("schedule") or []:
                if "selected_candidate" in item or "selected_candidate_token_ids_by_offset" in item:
                    totals["cleared_existing_selected_items"] += 1
                item.pop("selected_candidate", None)
                item.pop("selected_candidate_token_ids_by_offset", None)
                item.pop("selector_sidecar_selected_candidate", None)
                if item.get("candidate_source") in {"selector_sidecar", "pairwise_path_tournament"}:
                    item.pop("candidate_source", None)
        row_selectors = selectors.get(key) or []
        if not row_selectors:
            totals["records_without_selector"] += 1
            records.append(row)
            continue
        selector_audit = []
        for selector in row_selectors:
            if args.include_kinds and selector.get("kind") not in set(args.include_kinds):
                totals["selectors_filtered_by_kind"] += 1
                continue
            predicted = selector.get("predicted_value")
            try:
                start = int(selector["schedule_token_start"])
                end = int(selector["schedule_token_end"])
            except Exception:
                totals["selectors_without_token_span"] += 1
                continue
            restricted = 0
            candidates_missing = 0
            matching_schedule_items = 0
            selector_path = selector.get("json_path") or selector.get("argument_path") or selector.get("miss_path")
            for item in row.get("schedule") or []:
                try:
                    item_start = int(item.get("token_start"))
                    item_end = int(item.get("token_end"))
                except Exception:
                    continue
                if item_start < start or item_end > end:
                    continue
                if item.get("kind") != selector.get("kind"):
                    continue
                if item.get("tool_call_index") != selector.get("tool_call_index"):
                    continue
                if item.get("json_key") != selector.get("json_key"):
                    continue
                item_path = item.get("json_path") or item.get("argument_path")
                if selector_path and item_path and selector_path != item_path:
                    continue
                matching_schedule_items += 1
                if restrict_item(item, predicted):
                    restricted += 1
                else:
                    candidates_missing += 1
            if not matching_schedule_items:
                totals["selectors_without_matching_schedule_span"] += 1
            selector_audit.append(
                {
                    "kind": selector.get("kind"),
                    "tool_call_index": selector.get("tool_call_index"),
                    "json_key": selector.get("json_key"),
                    "miss_path": selector.get("miss_path"),
                    "json_path": selector.get("json_path"),
                    "argument_path": selector.get("argument_path"),
                    "schedule_token_start": start,
                    "schedule_token_end": end,
                    "predicted": predicted,
                    "correct": selector.get("correct"),
                    "target_win_margin": selector.get("target_win_margin"),
                    "matching_schedule_items": matching_schedule_items,
                    "restricted_schedule_items": restricted,
                    "candidate_missing_items": candidates_missing,
                }
            )
            totals["selectors"] += 1
            totals["selectors_correct"] += int(bool(selector.get("correct")))
            totals["restricted_schedule_items"] += restricted
            totals["candidate_missing_items"] += candidates_missing
            totals["selectors_with_restrictions"] += int(restricted > 0)
        row["pairwise_tournament_schedule_choices"] = selector_audit
        totals["records"] += 1
        records.append(row)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "schedule_jsonl": str(args.schedule_jsonl),
        "selector_jsonl": [str(path) for path in selector_paths],
        "out_jsonl": str(args.out_jsonl),
        "totals": dict(totals),
    }
    args.out_jsonl.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
