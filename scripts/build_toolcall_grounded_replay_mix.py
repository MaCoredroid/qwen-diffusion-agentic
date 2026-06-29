#!/usr/bin/env python3
import argparse
import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_SYNTHETIC_DIR = (
    ROOT / "data/qwen35_9b_toolcall_grounded_spanfill_synthetic_onecall48_b1024_curriculum"
)
DEFAULT_REPLAY_DIR = (
    ROOT / "data/qwen35_9b_toolcall_grounded_spanfill_teacher_train_b1024_curriculum"
)
DEFAULT_OUT_DIR = (
    ROOT / "data/qwen35_9b_toolcall_grounded_spanfill_synth48_replay_teacher2_b1024_curriculum"
)


def load_dataset(dataset_dir: Path):
    train_path = dataset_dir / "train_agentic_mix.json"
    audit_path = dataset_dir / "train_agentic_mix.audit.jsonl"
    payload = json.loads(train_path.read_text(encoding="utf-8"))
    if payload.get("type") != "conversation" or not isinstance(payload.get("instances"), list):
        raise ValueError(f"unsupported dataset format in {train_path}")

    audits = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    instances = payload["instances"]
    if len(instances) != len(audits):
        raise ValueError(
            f"{dataset_dir} has {len(instances)} instances but {len(audits)} audit rows"
        )
    return instances, audits


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


def summarize_audits(rows):
    summary = {"count": len(rows)}
    for key in ("length", "full_labels", "kept_labels", "assistant_label_count"):
        values = [row.get(key) for row in rows if isinstance(row.get(key), (int, float))]
        if values:
            summary[key] = percentile_summary(values)
    return summary


def add_rows(instances, audits, mix_source, repeat, mixed, mixed_audits):
    for repeat_idx in range(repeat):
        for instance, audit in zip(instances, audits):
            mixed.append(instance)
            row = dict(audit)
            row["mix_source"] = mix_source
            row["mix_repeat"] = repeat_idx
            mixed_audits.append(row)


def label_issue_counts(rows):
    zero = 0
    partial = 0
    full = 0
    for row in rows:
        zero += int(bool(row.get("zero_after_truncation")) or int(row.get("zero_label_count") or 0) > 0)
        partial += int(
            bool(row.get("partial_after_truncation"))
            or int(row.get("partial_label_count") or 0) > 0
        )
        full += int(bool(row.get("full_labels_kept")))
    return zero, partial, full


def source_family(source):
    if not source:
        return "unknown"
    return source.split(":", 1)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-dir", type=Path, default=DEFAULT_SYNTHETIC_DIR)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--synthetic-repeat", type=int, default=1)
    parser.add_argument("--replay-repeat", type=int, default=2)
    parser.add_argument("--seed", type=int, default=971)
    args = parser.parse_args()

    if args.synthetic_repeat < 1:
        raise ValueError("--synthetic-repeat must be >= 1")
    if args.replay_repeat < 1:
        raise ValueError("--replay-repeat must be >= 1")

    synthetic_instances, synthetic_audits = load_dataset(args.synthetic_dir)
    replay_instances, replay_audits = load_dataset(args.replay_dir)

    mixed = []
    mixed_audits = []
    add_rows(
        synthetic_instances,
        synthetic_audits,
        "synthetic_grounded_spanfill",
        args.synthetic_repeat,
        mixed,
        mixed_audits,
    )
    add_rows(
        replay_instances,
        replay_audits,
        "teacher_train_grounded_replay",
        args.replay_repeat,
        mixed,
        mixed_audits,
    )

    order = list(range(len(mixed)))
    random.Random(args.seed).shuffle(order)
    mixed = [mixed[idx] for idx in order]
    mixed_audits = [mixed_audits[idx] for idx in order]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train_agentic_mix.json"
    train_path.write_text(
        json.dumps({"type": "conversation", "instances": mixed}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    audit_path = args.out_dir / "train_agentic_mix.audit.jsonl"
    audit_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in mixed_audits),
        encoding="utf-8",
    )

    zero_rows, partial_rows, full_rows = label_issue_counts(mixed_audits)
    mix_source_counts = Counter(str(row.get("mix_source") or "unknown") for row in mixed_audits)
    source_family_counts = Counter(
        source_family(str(row.get("source") or "")) for row in mixed_audits
    )
    manifest = {
        "train_path": str(train_path),
        "audit_path": str(audit_path),
        "count": len(mixed),
        "synthetic_dir": str(args.synthetic_dir),
        "replay_dir": str(args.replay_dir),
        "synthetic_input_count": len(synthetic_instances),
        "replay_input_count": len(replay_instances),
        "synthetic_repeat": args.synthetic_repeat,
        "replay_repeat": args.replay_repeat,
        "mix_source_counts": dict(sorted(mix_source_counts.items())),
        "source_family_counts": dict(sorted(source_family_counts.items())),
        "zero_label_rows": zero_rows,
        "partial_label_rows": partial_rows,
        "full_label_rows": full_rows,
        "audit_summary": summarize_audits(mixed_audits),
        "seed": args.seed,
    }
    manifest_path = args.out_dir / "train_agentic_mix.manifest"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
