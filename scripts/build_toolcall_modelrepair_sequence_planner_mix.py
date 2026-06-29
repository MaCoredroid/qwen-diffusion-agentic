#!/usr/bin/env python3
import argparse
import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_BASE_DIR = ROOT / "data/qwen35_9b_toolcall_model_repair_curriculum"
DEFAULT_PLANNER_DIR = ROOT / "data/qwen35_9b_toolcall_sequence_planner_distill_curriculum"
DEFAULT_OUT_DIR = ROOT / "data/qwen35_9b_toolcall_modelrepair_sequence_planner_mix_curriculum"


def load_dataset(dataset_dir):
    train_path = dataset_dir / "train_agentic_mix.json"
    audit_path = dataset_dir / "train_agentic_mix.audit.jsonl"
    payload = json.loads(train_path.read_text(encoding="utf-8"))
    if payload.get("type") != "conversation" or not isinstance(payload.get("instances"), list):
        raise ValueError(f"unsupported dataset format in {train_path}")
    audit_rows = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    instances = payload["instances"]
    if len(instances) != len(audit_rows):
        raise ValueError(
            f"{dataset_dir} has {len(instances)} instances but {len(audit_rows)} audit rows"
        )
    return instances, audit_rows


def source_family(source):
    if not source:
        return "unknown"
    if "sequence_planner_distill" in source:
        return "sequence_planner_distill"
    return source.split(":", 1)[0]


def select_indices(count, cap, seed):
    indices = list(range(count))
    rng = random.Random(seed)
    rng.shuffle(indices)
    if cap >= 0:
        indices = indices[:cap]
    return indices


def percentile_summary(values):
    if not values:
        return {}
    values = sorted(values)

    def at(frac):
        idx = min(len(values) - 1, max(0, round((len(values) - 1) * frac)))
        return values[idx]

    return {
        "min": values[0],
        "p50": at(0.50),
        "p90": at(0.90),
        "max": values[-1],
    }


def summarize_audit(rows):
    keys = ["length", "full_labels", "kept_labels", "assistant_label_count"]
    summary = {"count": len(rows)}
    for key in keys:
        values = [row.get(key) for row in rows if isinstance(row.get(key), (int, float))]
        if values:
            summary[key] = percentile_summary(values)
    return summary


def label_issue_counts(rows):
    zero = 0
    partial = 0
    full = 0
    for row in rows:
        zero += int(bool(row.get("zero_after_truncation")) or int(row.get("zero_label_count") or 0) > 0)
        partial += int(bool(row.get("partial_after_truncation")) or int(row.get("partial_label_count") or 0) > 0)
        full += int(bool(row.get("full_labels_kept")))
    return zero, partial, full


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--planner-dir", type=Path, default=DEFAULT_PLANNER_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--base-repeat", type=int, default=1)
    parser.add_argument("--planner-repeat", type=int, default=1)
    parser.add_argument("--planner-cap", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=733)
    args = parser.parse_args()

    if args.base_repeat < 1:
        raise ValueError("--base-repeat must be >= 1")
    if args.planner_repeat < 1:
        raise ValueError("--planner-repeat must be >= 1")

    base_instances, base_audit = load_dataset(args.base_dir)
    planner_instances, planner_audit = load_dataset(args.planner_dir)
    planner_indices = select_indices(len(planner_instances), args.planner_cap, args.seed)

    mixed = []
    mixed_audit = []
    for repeat_idx in range(args.base_repeat):
        for instance, audit in zip(base_instances, base_audit):
            mixed.append(instance)
            row = dict(audit)
            row["mix_source"] = "base_model_repair"
            row["mix_repeat"] = repeat_idx
            mixed_audit.append(row)
    for repeat_idx in range(args.planner_repeat):
        for idx in planner_indices:
            mixed.append(planner_instances[idx])
            row = dict(planner_audit[idx])
            row["mix_source"] = "sequence_planner_distill"
            row["mix_repeat"] = repeat_idx
            mixed_audit.append(row)

    rng = random.Random(args.seed + 1)
    order = list(range(len(mixed)))
    rng.shuffle(order)
    mixed = [mixed[idx] for idx in order]
    mixed_audit = [mixed_audit[idx] for idx in order]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train_agentic_mix.json"
    train_path.write_text(
        json.dumps({"type": "conversation", "instances": mixed}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    audit_path = args.out_dir / "train_agentic_mix.audit.jsonl"
    audit_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in mixed_audit),
        encoding="utf-8",
    )

    zero_label_rows, partial_label_rows, full_label_rows = label_issue_counts(mixed_audit)
    source_counts = Counter(str(row.get("source") or "unknown") for row in mixed_audit)
    source_family_counts = Counter(source_family(str(row.get("source") or "")) for row in mixed_audit)
    mix_source_counts = Counter(str(row.get("mix_source") or "unknown") for row in mixed_audit)
    manifest = {
        "train_path": str(train_path),
        "audit_path": str(audit_path),
        "count": len(mixed),
        "base_dir": str(args.base_dir),
        "planner_dir": str(args.planner_dir),
        "base_input_count": len(base_instances),
        "planner_input_count": len(planner_instances),
        "base_repeat": args.base_repeat,
        "planner_repeat": args.planner_repeat,
        "planner_cap": args.planner_cap,
        "planner_selected_count": len(planner_indices),
        "mix_source_counts": dict(sorted(mix_source_counts.items())),
        "source_family_counts": dict(sorted(source_family_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "zero_label_rows": zero_label_rows,
        "partial_label_rows": partial_label_rows,
        "full_label_rows": full_label_rows,
        "audit_summary": summarize_audit(mixed_audit),
        "seed": args.seed,
    }
    manifest_path = args.out_dir / "train_agentic_mix.manifest"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
