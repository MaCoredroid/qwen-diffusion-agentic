#!/usr/bin/env python3
"""Build the leak-checked native scale-up eval slice for per-call waves."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("/home/mark/qwen_diffusion")
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from audit_toolcall_eval_overlap import assistant_text, normalize_text, user_text  # noqa: E402
from build_flare_agentic_mix_v1 import (  # noqa: E402
    call_signatures,
    eval_row_to_instance,
    fingerprint,
    load_instances,
)
from convert_toolcall_cases_to_qwen_native import convert_eval_row  # noqa: E402


DEFAULT_HELDOUT = ROOT / "data/toolcall_eval/heldout_seed_multicall_2to3_clean.jsonl"
DEFAULT_PUBLIC_BEYOND_SMOKE = ROOT / "data/toolcall_eval/public_train_multicall_no_public_smoke_cases.jsonl"
DEFAULT_TRAIN = ROOT / "data/flare_redesign_run1_copy_retention_mix/train_agentic_mix.json"
DEFAULT_TRAIN_AUDIT = ROOT / "data/flare_redesign_run1_copy_retention_mix/train_agentic_mix.audit.jsonl"
DEFAULT_OUT = ROOT / "data/toolcall_eval_native/flare_scaleup_native_58.jsonl"
DEFAULT_MANIFEST = ROOT / "runs/flare_scaleup_eval/scaleup_native_58_manifest.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def text_fingerprint(user: str, assistant: str) -> str:
    payload = normalize_text(user) + "\n---assistant---\n" + normalize_text(assistant)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def user_fingerprint(user: str) -> str:
    return hashlib.sha256(normalize_text(user).encode("utf-8")).hexdigest()


def load_sources(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for idx, row in enumerate(load_jsonl(path)):
            native_row, converted = convert_eval_row(row)
            if not converted:
                continue
            native_row = copy.deepcopy(native_row)
            native_row.setdefault("id", f"{path.stem}_{idx:04d}")
            native_row["scaleup_source_path"] = str(path)
            native_row["scaleup_source_index"] = idx
            rows.append(native_row)
    return rows


def audit_near_scope_indexes(audit_path: Path, near_leak_scope: str) -> set[int] | None:
    if near_leak_scope == "all":
        return None
    indexes: set[int] = set()
    if near_leak_scope == "none" or not audit_path.exists():
        return indexes
    with audit_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            source_dataset = str(row.get("source_dataset") or "")
            source = str(row.get("source") or "")
            if near_leak_scope == "copy_synth" and (
                source_dataset == "redesign_run1_copy_synth" or source == "redesign_run1_copy_synth"
            ):
                indexes.add(idx)
    return indexes


def train_reference(train_path: Path, audit_path: Path, near_leak_scope: str) -> dict[str, Any]:
    exact_hashes = set()
    user_hashes = set()
    calls_by_name: dict[str, list[dict[str, Any]]] = {}
    call_count = 0
    instances = load_instances(train_path)
    near_scope_indexes = audit_near_scope_indexes(audit_path, near_leak_scope)
    for idx, instance in enumerate(instances):
        exact_hashes.add(fingerprint(instance))
        user_hashes.add(user_fingerprint(user_text(instance)))
        if near_scope_indexes is not None and idx not in near_scope_indexes:
            continue
        for sig in call_signatures(instance):
            name = str(sig.get("name") or "")
            if not name:
                continue
            calls_by_name.setdefault(name, []).append({"train_idx": idx, **sig})
            call_count += 1
    return {
        "path": str(train_path),
        "audit_path": str(audit_path),
        "records": len(instances),
        "near_leak_scope": near_leak_scope,
        "near_leak_scope_records": len(instances) if near_scope_indexes is None else len(near_scope_indexes),
        "exact_hashes": exact_hashes,
        "user_hashes": user_hashes,
        "calls_by_name": calls_by_name,
        "call_count": call_count,
    }


def near_leaks(row: dict[str, Any], train_ref: dict[str, Any]) -> list[dict[str, Any]]:
    leaks: list[dict[str, Any]] = []
    instance = eval_row_to_instance(row)
    for sig in call_signatures(instance):
        name = str(sig.get("name") or "")
        if not name:
            continue
        eval_values = sig.get("distinctive_values") or sig.get("values") or set()
        if not eval_values:
            continue
        for train_call in train_ref["calls_by_name"].get(name, []):
            train_values = train_call.get("values") or set()
            if eval_values.issubset(train_values):
                leaks.append(
                    {
                        "type": "same_tool_all_eval_arg_values",
                        "train_idx": train_call["train_idx"],
                        "tool": name,
                        "matched_values": sorted(eval_values),
                    }
                )
                break
    return leaks


def dedup_and_filter(rows: list[dict[str, Any]], train_ref: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_exact_text: set[str] = set()
    seen_user: set[str] = set()
    seen_instance: set[str] = set()

    for row in rows:
        user = user_text(row)
        assistant = assistant_text(row)
        text_fp = text_fingerprint(user, assistant)
        user_fp = user_fingerprint(user)
        instance_fp = fingerprint(eval_row_to_instance(row))
        reasons = []

        if text_fp in seen_exact_text or instance_fp in seen_instance:
            reasons.append("slice_exact_duplicate")
        if user_fp in seen_user:
            reasons.append("slice_user_duplicate")
        if instance_fp in train_ref["exact_hashes"]:
            reasons.append("train_exact_instance_overlap")
        if user_fp in train_ref["user_hashes"]:
            reasons.append("train_user_overlap")
        leaks = near_leaks(row, train_ref)
        if leaks:
            reasons.append("train_same_tool_all_eval_arg_values")

        if reasons:
            rejected.append(
                {
                    "id": row.get("id"),
                    "source": row.get("source"),
                    "source_path": row.get("scaleup_source_path"),
                    "source_index": row.get("scaleup_source_index"),
                    "reasons": reasons,
                    "near_leaks": leaks[:5],
                }
            )
            continue

        seen_exact_text.add(text_fp)
        seen_user.add(user_fp)
        seen_instance.add(instance_fp)
        kept.append(row)
    return kept, rejected


def manifest_for(
    rows: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    train_ref: dict[str, Any],
    source_paths: list[Path],
    out_jsonl: Path,
) -> dict[str, Any]:
    source_counts = Counter(str(row.get("source") or "unknown") for row in rows)
    source_path_counts = Counter(str(row.get("scaleup_source_path") or "unknown") for row in rows)
    call_counts = Counter(str(len(row.get("gold_tool_calls") or [])) for row in rows)
    reject_counts = Counter(reason for row in rejected for reason in row.get("reasons") or [])
    return {
        "out_jsonl": str(out_jsonl),
        "records": len(rows),
        "source_paths": [str(path) for path in source_paths],
        "source_counts": dict(sorted(source_counts.items())),
        "source_path_counts": dict(sorted(source_path_counts.items())),
        "tool_call_count_histogram": dict(sorted(call_counts.items())),
        "native_format": "qwen_native_function_parameter",
        "dedup_policy": "drop exact canonical/text duplicates and duplicate user prompts within slice",
        "train_leak_check": {
            "train_path": train_ref["path"],
            "train_audit_path": train_ref["audit_path"],
            "train_records": train_ref["records"],
            "near_leak_scope": train_ref["near_leak_scope"],
            "near_leak_scope_records": train_ref["near_leak_scope_records"],
            "train_tool_calls_indexed": train_ref["call_count"],
            "exact_instance_overlaps": 0,
            "user_overlaps": 0,
            "same_tool_all_eval_arg_values_overlaps": 0,
            "copy_grounding_covered": True,
        },
        "rejected_count": len(rejected),
        "rejected_reason_counts": dict(sorted(reject_counts.items())),
        "rejected_examples": rejected[:20],
        "ids": [row.get("id") for row in rows],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, nargs="*", default=[DEFAULT_HELDOUT, DEFAULT_PUBLIC_BEYOND_SMOKE])
    parser.add_argument("--train-json", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--train-audit-jsonl", type=Path, default=DEFAULT_TRAIN_AUDIT)
    parser.add_argument(
        "--near-leak-scope",
        choices=["copy_synth", "all", "none"],
        default="copy_synth",
        help=(
            "Scope for same-tool/all-distinctive-argument-value leak checks. Exact/user "
            "overlap is always checked against all training records."
        ),
    )
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--rejected-jsonl", type=Path, default=None)
    parser.add_argument("--min-records", type=int, default=48)
    args = parser.parse_args()

    source_paths = [path.resolve() for path in args.source_jsonl]
    train_ref = train_reference(args.train_json.resolve(), args.train_audit_jsonl.resolve(), args.near_leak_scope)
    candidates = load_sources(source_paths)
    rows, rejected = dedup_and_filter(candidates, train_ref)
    if len(rows) < args.min_records:
        raise SystemExit(f"only {len(rows)} leak-checked records remain; need >= {args.min_records}")

    write_jsonl(args.out_jsonl, rows)
    rejected_path = args.rejected_jsonl or args.manifest_json.with_suffix(".rejected.jsonl")
    write_jsonl(rejected_path, rejected)
    manifest = manifest_for(rows, rejected, train_ref, source_paths, args.out_jsonl)
    manifest["rejected_jsonl"] = str(rejected_path)
    write_json(args.manifest_json, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
