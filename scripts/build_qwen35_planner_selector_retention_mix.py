#!/usr/bin/env python3
import argparse
import copy
import json
import random
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path("/home/mark/qwen_diffusion")
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from audit_toolcall_eval_overlap import eval_records, fingerprint, user_fingerprint, user_text, assistant_text  # noqa: E402

DEFAULT_MODEL = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_RETENTION_DIR = ROOT / "data/qwen35_9b_route_delta_trainonly_mix_curriculum"
DEFAULT_PLANNER_DIR = ROOT / "data/qwen35_9b_toolcall_sequence_planner_distill_no_public_multicall_smoke_curriculum"
DEFAULT_SELECTOR_DIR = ROOT / "data/qwen35_9b_public_train_pairwise_pathaware_phrase_argsketch_curriculum"
DEFAULT_OUT_DIR = ROOT / "data/qwen35_9b_planner_selector_retention_mix_curriculum"


def resolve_chat_template(name):
    third_party = ROOT / "fast-dllm/third_party"
    if str(third_party) not in sys.path:
        sys.path.insert(0, str(third_party))
    from lmflow.utils.conversation_template import PRESET_TEMPLATES

    if name not in PRESET_TEMPLATES:
        raise ValueError(f"unknown conversation template {name!r}")
    return PRESET_TEMPLATES[name]


def drop_none_fields(value):
    if isinstance(value, dict):
        return {key: drop_none_fields(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [drop_none_fields(item) for item in value if item is not None]
    return value


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_dataset_dir(dataset_dir):
    train_path = dataset_dir / "train_agentic_mix.json"
    audit_path = dataset_dir / "train_agentic_mix.audit.jsonl"
    manifest_path = dataset_dir / "train_agentic_mix.manifest"
    payload = load_json(train_path)
    instances = payload.get("instances")
    if payload.get("type") != "conversation" or not isinstance(instances, list):
        raise ValueError(f"{train_path} does not contain conversation instances")
    return {
        "dir": dataset_dir,
        "train_path": train_path,
        "audit_path": audit_path,
        "manifest_path": manifest_path,
        "instances": instances,
        "audit_rows": load_jsonl(audit_path),
        "manifest": load_json(manifest_path) if manifest_path.exists() else {},
    }


def conversation_for_template(instance):
    system = instance.get("system")
    messages = [{"role": "system", "content": system if system is not None else "You are a helpful assistant."}]
    messages.extend(copy.deepcopy(instance.get("messages") or []))
    return drop_none_fields(messages)


def token_stats(tokenizer, chat_template, instance, block_size, truncation_side):
    encoded = tokenizer.apply_chat_template(
        conversation=conversation_for_template(instance),
        tools=drop_none_fields(instance.get("tools") or None),
        chat_template=chat_template,
        return_assistant_tokens_mask=True,
        return_dict=True,
    )
    labels = [
        token if mask == 1 else -100
        for token, mask in zip(encoded["input_ids"], encoded["assistant_masks"])
    ]
    full_labels = sum(label != -100 for label in labels)
    if len(labels) <= block_size:
        kept = labels
    elif truncation_side == "right":
        kept = labels[:block_size]
    elif truncation_side == "left":
        kept = labels[-block_size:]
    else:
        raise ValueError(f"unsupported truncation side {truncation_side!r}")
    kept_labels = sum(label != -100 for label in kept)
    return {
        "length": len(labels),
        "full_labels": full_labels,
        "kept_labels": kept_labels,
        "full_labels_kept": full_labels > 0 and kept_labels == full_labels,
        "zero_after_truncation": kept_labels == 0,
        "partial_after_truncation": 0 < kept_labels < full_labels,
    }


def strip_training_metadata(instance):
    clone = copy.deepcopy(instance)
    clone.pop("source", None)
    return clone


def select_indices(count, cap, seed):
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
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

    return {"min": values[0], "p50": at(0.5), "p90": at(0.9), "max": values[-1]}


def summarize_audit(rows):
    out = {"count": len(rows)}
    for key in ("length", "full_labels", "kept_labels"):
        values = [row[key] for row in rows if isinstance(row.get(key), int)]
        if values:
            out[key] = percentile_summary(values)
    out["zero_after_truncation"] = sum(int(bool(row.get("zero_after_truncation"))) for row in rows)
    out["partial_after_truncation"] = sum(int(bool(row.get("partial_after_truncation"))) for row in rows)
    return out


def add_dataset(rows, dataset, mix_source, repeat, cap, seed):
    selected = select_indices(len(dataset["instances"]), cap, seed)
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
                    "source": instance.get("source") or "",
                }
            )


def eval_fingerprints(paths):
    records = eval_records(paths)
    return {
        "exact": {row["fingerprint"] for row in records},
        "user": {row["user_fingerprint"] for row in records},
        "records": records,
    }


def filter_eval_overlaps(rows, eval_paths):
    if not eval_paths:
        return rows, []
    fps = eval_fingerprints(eval_paths)
    kept = []
    removed = []
    for row in rows:
        instance = row["instance"]
        exact = fingerprint(user_text(instance), assistant_text(instance))
        user = user_fingerprint(user_text(instance))
        if exact in fps["exact"] or user in fps["user"]:
            removed.append(
                {
                    "source_dataset": row["source_dataset"],
                    "dataset_dir": row["dataset_dir"],
                    "source_index": row["source_index"],
                    "repeat": row["repeat"],
                    "source": row["source"],
                    "exact_overlap": exact in fps["exact"],
                    "user_overlap": user in fps["user"],
                    "user_excerpt": " ".join(user_text(instance).split())[:220],
                }
            )
        else:
            kept.append(row)
    return kept, removed


def audit_rows(rows, tokenizer, chat_template, args):
    accepted = []
    rejected = []
    for row in rows:
        stats = token_stats(tokenizer, chat_template, row["instance"], args.block_size, args.truncation_side)
        audit = {
            "source_dataset": row["source_dataset"],
            "dataset_dir": row["dataset_dir"],
            "source_index": row["source_index"],
            "repeat": row["repeat"],
            "source": row["source"],
            "tool_count": len(row["instance"].get("tools") or []),
            **stats,
        }
        if stats["kept_labels"] < args.min_labels:
            rejected.append({**audit, "reject_reason": "too_few_labels"})
            continue
        if args.require_full_labels and not stats["full_labels_kept"]:
            rejected.append({**audit, "reject_reason": "partial_labels_after_truncation"})
            continue
        accepted.append({"instance": row["instance"], "audit": audit})
    return accepted, rejected


def manifest_provenance(datasets):
    source_manifests = {}
    blocking = []
    for name, dataset in datasets.items():
        manifest = dataset["manifest"]
        source_manifests[name] = {
            "dir": str(dataset["dir"]),
            "count": manifest.get("count"),
            "promotion_allowed": manifest.get("promotion_allowed"),
            "diagnostic_only": manifest.get("diagnostic_only"),
            "contains_eval_slice": manifest.get("contains_eval_slice"),
            "no_eval_leakage": manifest.get("no_eval_leakage"),
            "train_path": manifest.get("train_path"),
        }
        if manifest.get("promotion_allowed") is False:
            blocking.append(f"{name}:promotion_allowed_false")
        if manifest.get("diagnostic_only") is True:
            blocking.append(f"{name}:diagnostic_only")
        if manifest.get("contains_eval_slice") is True:
            blocking.append(f"{name}:contains_eval_slice")
    return source_manifests, blocking


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--conversation-template", default="fast_dllm_v2")
    parser.add_argument("--retention-dir", type=Path, default=DEFAULT_RETENTION_DIR)
    parser.add_argument("--planner-dir", type=Path, default=DEFAULT_PLANNER_DIR)
    parser.add_argument("--selector-dir", type=Path, default=DEFAULT_SELECTOR_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--retention-cap", type=int, default=192)
    parser.add_argument("--planner-cap", type=int, default=-1)
    parser.add_argument("--selector-cap", type=int, default=160)
    parser.add_argument("--retention-repeat", type=int, default=1)
    parser.add_argument("--planner-repeat", type=int, default=2)
    parser.add_argument("--selector-repeat", type=int, default=1)
    parser.add_argument("--exclude-eval-jsonl", type=Path, nargs="*", default=[])
    parser.add_argument("--block-size", type=int, default=1536)
    parser.add_argument("--truncation-side", choices=["left", "right"], default="left")
    parser.add_argument("--min-labels", type=int, default=1)
    parser.add_argument("--require-full-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=6028)
    args = parser.parse_args()

    datasets = {
        "retention": load_dataset_dir(args.retention_dir),
        "planner": load_dataset_dir(args.planner_dir),
        "selector": load_dataset_dir(args.selector_dir),
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    chat_template = resolve_chat_template(args.conversation_template)

    rows = []
    add_dataset(rows, datasets["retention"], "route_delta_retention", args.retention_repeat, args.retention_cap, args.seed + 1)
    add_dataset(rows, datasets["planner"], "sequence_planner", args.planner_repeat, args.planner_cap, args.seed + 2)
    add_dataset(rows, datasets["selector"], "pairwise_selector", args.selector_repeat, args.selector_cap, args.seed + 3)
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
        "rejected_source_counts": dict(sorted(rejected_counts.items())),
        "source_manifests": source_manifests,
        "provenance_blockers": provenance_blockers,
        "contains_eval_slice": False if not provenance_blockers else None,
        "diagnostic_only": False if not provenance_blockers else None,
        "promotion_allowed": False if provenance_blockers else True,
        "promotion_note": (
            "Source manifests have no diagnostic/eval-slice blockers. Run overlap audit before promotion."
            if not provenance_blockers
            else "Not promotion-eligible until provenance blockers are resolved."
        ),
        "caps": {
            "retention": args.retention_cap,
            "planner": args.planner_cap,
            "selector": args.selector_cap,
        },
        "repeats": {
            "retention": args.retention_repeat,
            "planner": args.planner_repeat,
            "selector": args.selector_repeat,
        },
        "block_size": args.block_size,
        "exclude_eval_jsonl": [str(path) for path in args.exclude_eval_jsonl],
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
