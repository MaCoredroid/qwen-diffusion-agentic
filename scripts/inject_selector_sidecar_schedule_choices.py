#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_SCHEDULE = ROOT / "runs/tool_sensitive_block_plans/synthetic_selector_sidecar_sampler_schedule_augmented.jsonl"
DEFAULT_SELECTOR = ROOT / "runs/synthetic_multicall_failure_analogues/selector_sidecar_leaveone_ckpt20_projection.jsonl"
DEFAULT_OUT = ROOT / "runs/tool_sensitive_block_plans/synthetic_selector_sidecar_sampler_schedule_with_selector_choices.jsonl"


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
        item["selector_sidecar_selected_candidate"] = predicted
        item["candidate_sequence_values"] = [value]
        item["candidate_sequence_token_ids_by_offset"] = [sequence]
        item["selected_candidate"] = predicted
        item["selected_candidate_token_ids_by_offset"] = selected_token_ids_by_offset(sequence)
        item["candidate_source"] = "selector_sidecar"
        return True
    return False


def selector_map(path):
    out = {}
    for idx, row in enumerate(load_jsonl(path)):
        key = case_key(row, idx)
        audit = row.get("selector_audit") or {}
        if key:
            out[key] = {
                "kind": audit.get("selector_kind"),
                "tool_call_index": audit.get("selector_tool_call_index", None),
                "predicted": audit.get("selector_predicted_value"),
                "json_key": audit.get("json_key"),
                "correct": audit.get("selector_correct"),
                "target_margin": audit.get("selector_target_margin"),
            }
            if out[key]["tool_call_index"] is None:
                # The projection audit records selector metadata without copying
                # every source field. Recover the index from the source rows when
                # present in top-level sidecar output.
                out[key]["tool_call_index"] = audit.get("tool_call_index")
    return out


def ranking_selector_map(path):
    out = {}
    for idx, row in enumerate(load_jsonl(path)):
        key = case_key(row, idx)
        out[key] = {
            "kind": row.get("kind"),
            "tool_call_index": row.get("tool_call_index"),
            "predicted": row.get("predicted_value"),
            "json_key": row.get("json_key"),
            "correct": row.get("correct"),
            "target_margin": row.get("target_margin"),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule-jsonl", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--selector-jsonl", type=Path, default=DEFAULT_SELECTOR)
    parser.add_argument("--selector-format", choices=["projection", "ranking"], default="ranking")
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    selectors = ranking_selector_map(args.selector_jsonl) if args.selector_format == "ranking" else selector_map(args.selector_jsonl)
    records = []
    totals = Counter()
    for idx, row in enumerate(load_jsonl(args.schedule_jsonl)):
        key = case_key(row, idx)
        selector = selectors.get(key)
        if not selector:
            totals["records_without_selector"] += 1
            records.append(row)
            continue
        kind = selector.get("kind")
        predicted = selector.get("predicted")
        try:
            tool_call_index = int(selector.get("tool_call_index"))
        except Exception:
            totals["selectors_without_tool_call_index"] += 1
            records.append(row)
            continue
        json_key = selector.get("json_key")
        restricted = 0
        candidates_missing = 0
        for item in row.get("schedule") or []:
            if item.get("kind") != kind:
                continue
            if int(item.get("tool_call_index", -1)) != tool_call_index:
                continue
            if kind == "argument_value" and item.get("json_key") != json_key:
                continue
            if kind == "tool_name" and item.get("json_key") not in {None, "name"}:
                continue
            if restrict_item(item, predicted):
                restricted += 1
            else:
                candidates_missing += 1
        row["selector_sidecar_schedule_choice"] = {
            "kind": kind,
            "tool_call_index": tool_call_index,
            "json_key": json_key,
            "predicted": predicted,
            "correct": selector.get("correct"),
            "target_margin": selector.get("target_margin"),
            "restricted_schedule_items": restricted,
            "candidate_missing_items": candidates_missing,
        }
        totals["records"] += 1
        totals[f"records:{kind}"] += 1
        totals["restricted_schedule_items"] += restricted
        totals["candidate_missing_items"] += candidates_missing
        totals["records_with_restrictions"] += int(restricted > 0)
        records.append(row)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "schedule_jsonl": str(args.schedule_jsonl),
        "selector_jsonl": str(args.selector_jsonl),
        "selector_format": args.selector_format,
        "out_jsonl": str(args.out_jsonl),
        "totals": dict(totals),
    }
    args.out_jsonl.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
