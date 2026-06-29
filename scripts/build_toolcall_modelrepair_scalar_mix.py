#!/usr/bin/env python3
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_BASE_DIR = ROOT / "data/qwen35_9b_toolcall_model_repair_curriculum"
DEFAULT_SCALAR_DIR = ROOT / "data/qwen35_9b_toolcall_multicall_scalar_curriculum"
DEFAULT_OUT_DIR = ROOT / "data/qwen35_9b_toolcall_modelrepair_scalar_mix_curriculum"


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


def scalar_variant(source):
    marker = ":scalar_"
    if marker not in source:
        return "unknown"
    tail = source.split(marker, 1)[1]
    return tail.split(":", 1)[0]


def source_family(source):
    if not source:
        return "unknown"
    if ":scalar_" in source:
        return "multicall_scalar"
    return source.split(":", 1)[0]


def select_scalar_indices(audit_rows, scalar_cap, seed):
    if scalar_cap < 0 or scalar_cap >= len(audit_rows):
        return list(range(len(audit_rows)))
    by_variant = defaultdict(list)
    for idx, row in enumerate(audit_rows):
        by_variant[scalar_variant(str(row.get("source") or ""))].append(idx)
    rng = random.Random(seed)
    for items in by_variant.values():
        rng.shuffle(items)

    variants = sorted(by_variant)
    selected = []
    cursor = 0
    while len(selected) < scalar_cap and any(by_variant.values()):
        variant = variants[cursor % len(variants)]
        cursor += 1
        if by_variant[variant]:
            selected.append(by_variant[variant].pop())
    rng.shuffle(selected)
    return selected


def summarize_audit(rows):
    if not rows:
        return {}
    keys = [
        "length",
        "full_labels",
        "kept_labels",
        "assistant_label_count",
    ]
    summary = {"count": len(rows)}
    for key in keys:
        values = [row.get(key) for row in rows if isinstance(row.get(key), (int, float))]
        if not values:
            continue
        values = sorted(values)
        summary[key] = {
            "min": values[0],
            "p50": values[len(values) // 2],
            "p90": values[min(len(values) - 1, int(len(values) * 0.9))],
            "max": values[-1],
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--scalar-dir", type=Path, default=DEFAULT_SCALAR_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--base-repeat", type=int, default=1)
    parser.add_argument("--scalar-cap", type=int, default=128)
    parser.add_argument("--seed", type=int, default=613)
    args = parser.parse_args()

    if args.base_repeat < 1:
        raise ValueError("--base-repeat must be >= 1")

    base_instances, base_audit = load_dataset(args.base_dir)
    scalar_instances, scalar_audit = load_dataset(args.scalar_dir)
    scalar_indices = select_scalar_indices(scalar_audit, args.scalar_cap, args.seed)

    mixed = []
    mixed_audit = []
    for repeat_idx in range(args.base_repeat):
        for instance, audit in zip(base_instances, base_audit):
            mixed.append(instance)
            row = dict(audit)
            row["mix_source"] = "base_model_repair"
            row["mix_repeat"] = repeat_idx
            mixed_audit.append(row)
    for idx in scalar_indices:
        mixed.append(scalar_instances[idx])
        row = dict(scalar_audit[idx])
        row["mix_source"] = "multicall_scalar"
        row["scalar_variant"] = scalar_variant(str(row.get("source") or ""))
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

    source_counts = Counter(str(row.get("source") or "unknown") for row in mixed_audit)
    source_family_counts = Counter(source_family(str(row.get("source") or "")) for row in mixed_audit)
    mix_source_counts = Counter(str(row.get("mix_source") or "unknown") for row in mixed_audit)
    scalar_variant_counts = Counter(
        str(row.get("scalar_variant") or "n/a")
        for row in mixed_audit
        if row.get("mix_source") == "multicall_scalar"
    )
    zero_label_rows = sum(
        1
        for row in mixed_audit
        if bool(row.get("zero_after_truncation")) or int(row.get("zero_label_count") or 0) > 0
    )
    partial_label_rows = sum(
        1
        for row in mixed_audit
        if bool(row.get("partial_after_truncation")) or int(row.get("partial_label_count") or 0) > 0
    )
    full_label_rows = sum(1 for row in mixed_audit if bool(row.get("full_labels_kept")))

    manifest = {
        "train_path": str(train_path),
        "audit_path": str(audit_path),
        "count": len(mixed),
        "base_dir": str(args.base_dir),
        "scalar_dir": str(args.scalar_dir),
        "base_input_count": len(base_instances),
        "scalar_input_count": len(scalar_instances),
        "base_repeat": args.base_repeat,
        "scalar_cap": args.scalar_cap,
        "scalar_selected_count": len(scalar_indices),
        "mix_source_counts": dict(sorted(mix_source_counts.items())),
        "scalar_variant_counts": dict(sorted(scalar_variant_counts.items())),
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
