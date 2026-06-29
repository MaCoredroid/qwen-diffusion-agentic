#!/usr/bin/env python3
import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from build_qwen35_planner_selector_retention_mix import (
    ROOT,
    DEFAULT_MODEL,
    DEFAULT_RETENTION_DIR,
    DEFAULT_PLANNER_DIR,
    audit_rows,
    filter_eval_overlaps,
    load_dataset_dir,
    manifest_provenance,
    resolve_chat_template,
    strip_training_metadata,
    summarize_audit,
    write_jsonl,
)


DEFAULT_VALUE_DIR = ROOT / "data/qwen35_9b_candidate_value_span_public_train_curriculum"
DEFAULT_OUT_DIR = ROOT / "data/qwen35_9b_sequence_value_retention_mix_nooverlap_curriculum"
DEFAULT_RETENTION_INCLUDE_SOURCES = (
    "fastdllm_toolcall_train,"
    "synthetic_onecall_train,"
    "synthetic_toolresult_text_train,"
    "synthetic_toolresult_openai_train"
)


def parse_csv(value):
    return {item.strip() for item in str(value or "").split(",") if item.strip()}


def source_for(dataset, idx):
    if idx < len(dataset["audit_rows"]):
        audit = dataset["audit_rows"][idx]
        return (
            audit.get("source")
            or audit.get("source_dataset")
            or audit.get("mix_source")
            or audit.get("instance_source")
            or ""
        )
    return dataset["instances"][idx].get("source") or ""


def select_indices(dataset, cap, seed, include_sources=None, exclude_sources=None):
    include_sources = set(include_sources or [])
    exclude_sources = set(exclude_sources or [])
    indices = []
    skipped = Counter()
    for idx in range(len(dataset["instances"])):
        source = source_for(dataset, idx)
        if include_sources and source not in include_sources:
            skipped["source_not_included"] += 1
            continue
        if source in exclude_sources:
            skipped["source_excluded"] += 1
            continue
        indices.append(idx)
    random.Random(seed).shuffle(indices)
    if cap >= 0:
        indices = indices[:cap]
    return indices, skipped


def add_dataset(
    rows,
    dataset,
    mix_source,
    repeat,
    cap,
    seed,
    include_sources=None,
    exclude_sources=None,
):
    selected, skipped = select_indices(dataset, cap, seed, include_sources, exclude_sources)
    for repeat_idx in range(max(1, repeat)):
        for idx in selected:
            instance = copy.deepcopy(dataset["instances"][idx])
            rows.append(
                {
                    "instance": instance,
                    "dataset_dir": str(dataset["dir"]),
                    "source_dataset": mix_source,
                    "source_index": idx,
                    "repeat": repeat_idx,
                    "source": source_for(dataset, idx),
                }
            )
    return {"selected": len(selected), "skipped": dict(sorted(skipped.items()))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--conversation-template", default="fast_dllm_v2")
    parser.add_argument("--retention-dir", type=Path, default=DEFAULT_RETENTION_DIR)
    parser.add_argument("--value-dir", type=Path, default=DEFAULT_VALUE_DIR)
    parser.add_argument("--planner-dir", type=Path, default=DEFAULT_PLANNER_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--retention-cap", type=int, default=-1)
    parser.add_argument("--value-cap", type=int, default=-1)
    parser.add_argument("--planner-cap", type=int, default=-1)
    parser.add_argument("--retention-repeat", type=int, default=1)
    parser.add_argument("--value-repeat", type=int, default=1)
    parser.add_argument("--planner-repeat", type=int, default=4)
    parser.add_argument("--retention-include-sources", default=DEFAULT_RETENTION_INCLUDE_SOURCES)
    parser.add_argument("--retention-exclude-sources", default="")
    parser.add_argument("--exclude-eval-jsonl", type=Path, nargs="*", default=[])
    parser.add_argument("--block-size", type=int, default=1536)
    parser.add_argument("--truncation-side", choices=["left", "right"], default="left")
    parser.add_argument("--min-labels", type=int, default=1)
    parser.add_argument("--require-full-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=7028)
    args = parser.parse_args()

    datasets = {
        "retention": load_dataset_dir(args.retention_dir),
        "value": load_dataset_dir(args.value_dir),
        "planner": load_dataset_dir(args.planner_dir),
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    chat_template = resolve_chat_template(args.conversation_template)

    rows = []
    selection = {
        "retention": add_dataset(
            rows,
            datasets["retention"],
            "route_delta_retention",
            args.retention_repeat,
            args.retention_cap,
            args.seed + 1,
            include_sources=parse_csv(args.retention_include_sources),
            exclude_sources=parse_csv(args.retention_exclude_sources),
        ),
        "value": add_dataset(
            rows,
            datasets["value"],
            "candidate_value_span",
            args.value_repeat,
            args.value_cap,
            args.seed + 2,
        ),
        "planner": add_dataset(
            rows,
            datasets["planner"],
            "sequence_planner",
            args.planner_repeat,
            args.planner_cap,
            args.seed + 3,
        ),
    }
    rows, eval_overlap_removed = filter_eval_overlaps(rows, args.exclude_eval_jsonl)

    accepted, rejected = audit_rows(rows, tokenizer, chat_template, args)
    order = list(range(len(accepted)))
    random.Random(args.seed + 4).shuffle(order)
    accepted = [accepted[idx] for idx in order]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train_agentic_mix.json"
    audit_path = args.out_dir / "train_agentic_mix.audit.jsonl"
    rejected_path = args.out_dir / "train_agentic_mix.rejected.jsonl"
    overlap_removed_path = args.out_dir / "eval_overlap_removed.jsonl"
    train_path.write_text(
        json.dumps(
            {"type": "conversation", "instances": [strip_training_metadata(item["instance"]) for item in accepted]},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    write_jsonl(audit_path, [item["audit"] for item in accepted])
    write_jsonl(rejected_path, rejected)
    write_jsonl(overlap_removed_path, eval_overlap_removed)

    audits = [item["audit"] for item in accepted]
    source_manifests, provenance_blockers = manifest_provenance(datasets)
    source_counts = Counter(row["source_dataset"] for row in audits)
    detail_counts = Counter(f"{row['source_dataset']}:{row.get('source') or 'unknown'}" for row in audits)
    rejected_counts = Counter(row["source_dataset"] for row in rejected)
    manifest = {
        "train_path": str(train_path),
        "audit_path": str(audit_path),
        "rejected_path": str(rejected_path),
        "eval_overlap_removed_path": str(overlap_removed_path),
        "count": len(accepted),
        "candidate_count": len(rows) + len(eval_overlap_removed),
        "candidate_after_eval_filter_count": len(rows),
        "eval_overlap_removed_count": len(eval_overlap_removed),
        "eval_overlap_removed_source_counts": dict(
            sorted(Counter(row["source_dataset"] for row in eval_overlap_removed).items())
        ),
        "rejected_count": len(rejected),
        "source_counts": dict(sorted(source_counts.items())),
        "source_detail_counts": dict(sorted(detail_counts.items())),
        "rejected_source_counts": dict(sorted(rejected_counts.items())),
        "selection": selection,
        "source_manifests": source_manifests,
        "provenance_blockers": provenance_blockers,
        "contains_eval_slice": False if not provenance_blockers else None,
        "diagnostic_only": False if not provenance_blockers else None,
        "promotion_allowed": False if provenance_blockers else True,
        "promotion_note": (
            "Source manifests have no diagnostic/eval-slice blockers; overlap audit is still required before promotion."
            if not provenance_blockers
            else "Not promotion-eligible until provenance blockers are resolved."
        ),
        "caps": {
            "retention": args.retention_cap,
            "value": args.value_cap,
            "planner": args.planner_cap,
        },
        "repeats": {
            "retention": args.retention_repeat,
            "value": args.value_repeat,
            "planner": args.planner_repeat,
        },
        "retention_include_sources": sorted(parse_csv(args.retention_include_sources)),
        "retention_exclude_sources": sorted(parse_csv(args.retention_exclude_sources)),
        "exclude_eval_jsonl": [str(path) for path in args.exclude_eval_jsonl],
        "block_size": args.block_size,
        "truncation_side": args.truncation_side,
        "min_labels": args.min_labels,
        "require_full_labels": args.require_full_labels,
        "audit_summary": summarize_audit(audits),
        "rejected_summary": summarize_audit(rejected),
        "seed": args.seed,
    }
    manifest_path = args.out_dir / "train_agentic_mix.manifest"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
