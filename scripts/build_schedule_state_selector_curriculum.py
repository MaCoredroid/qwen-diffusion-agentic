#!/usr/bin/env python3
"""Build a schedule-state selector curriculum from skeleton value-infill slots.

The standalone value-infill objective teaches the model to emit the JSON value
span. This curriculum instead asks for the sampler-side decision used at
generation time: which candidate index to select and which protection policy to
apply for the active scheduled argument-value span.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


ROOT = Path("/home/mark/qwen_diffusion")
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_toolcall_labelaware_public_mix import (  # noqa: E402
    resolve_chat_template,
    summarize_audit,
    token_stats,
)


DEFAULT_MODEL = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_SLOT_JSONL = ROOT / "data/skeleton_value_infill/public_train_no_public_smoke/skeleton_value_slots.jsonl"
DEFAULT_BOUNDARY_JSONL = ROOT / "data/skeleton_value_infill/public_train_no_public_smoke/boundary_labels.jsonl"
DEFAULT_SUMMARY_JSON = ROOT / "data/skeleton_value_infill/public_train_no_public_smoke/summary.json"
DEFAULT_OUT_DIR = ROOT / "data/qwen35_9b_schedule_state_selector_no_public_smoke_curriculum"

SYSTEM = (
    "You choose the sampler schedule-state decision for behavior-preserving "
    "Qwen tool-call diffusion."
)


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def strip_source(instance: dict[str, Any]) -> dict[str, Any]:
    clone = copy.deepcopy(instance)
    clone.pop("source", None)
    clone.pop("_selector_meta", None)
    return clone


def compact_candidates(slot: dict[str, Any]) -> str:
    lines = []
    for idx, value in enumerate(slot.get("candidate_values") or []):
        lines.append(f"{idx}: {json.dumps(value, ensure_ascii=False)}")
    return "\n".join(lines)


def boundary_lookup(boundaries: list[dict[str, Any]]) -> dict[tuple[Any, Any, Any, Any], dict[str, Any]]:
    lookup = {}
    for row in boundaries:
        key = (
            row.get("id"),
            row.get("tool_call_index"),
            row.get("json_path"),
            row.get("target_text"),
        )
        if row.get("kind") == "argument_value" and key not in lookup:
            lookup[key] = row
    return lookup


def boundary_for_slot(slot: dict[str, Any], lookup: dict[tuple[Any, Any, Any, Any], dict[str, Any]]) -> dict[str, Any]:
    key = (
        slot.get("id"),
        slot.get("tool_call_index"),
        slot.get("json_path"),
        slot.get("target_text"),
    )
    row = lookup.get(key)
    if row:
        return row
    return {
        "kind": "argument_value",
        "recommended_block_size": min(8, max(1, len(slot.get("target_token_ids") or []))),
        "recommended_denoise_steps": 8,
        "must_shrink": True,
        "must_constrain": True,
        "must_be_json_completable": True,
    }


def selector_decision(slot: dict[str, Any], boundary: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_index": int(slot["target_index"]),
        "span_kind": "argument_value",
        "protection": "value_candidate_json_prefix_close_guard",
        "block_size": int(boundary.get("recommended_block_size") or 1),
        "denoise_steps": int(boundary.get("recommended_denoise_steps") or 8),
        "force_candidate_sequence": True,
        "require_json_prefix_safe": True,
        "close_tool_call_only_when_json_complete": True,
    }


def selector_prompt(slot: dict[str, Any], boundary: dict[str, Any]) -> str:
    local_peers = slot.get("local_peer_arguments") or []
    evidence = slot.get("context_evidence_matches") or []
    parts = [
        "Choose the schedule-state decision for the active argument-value span.",
        "Return only a minified JSON object with these exact keys:",
        "candidate_index, span_kind, protection, block_size, denoise_steps, force_candidate_sequence, require_json_prefix_safe, close_tool_call_only_when_json_complete.",
        "Do not emit the value itself and do not add prose.",
        "",
        "Decision context:",
        f"Case id: {slot.get('id')}",
        f"Tool call index: {slot.get('tool_call_index')}",
        f"JSON key: {slot.get('json_key')}",
        f"JSON path: {slot.get('json_path')}",
        f"Schema type: {slot.get('schema_type')}",
        f"Schedule token span: {slot.get('schedule_token_start')}..{slot.get('schedule_token_end')}",
        f"Target token count: {len(slot.get('target_token_ids') or [])}",
        f"Candidate count: {slot.get('candidate_count')}",
        "",
        "Boundary policy label:",
        json.dumps(
            {
                "kind": boundary.get("kind"),
                "recommended_block_size": boundary.get("recommended_block_size"),
                "recommended_denoise_steps": boundary.get("recommended_denoise_steps"),
                "must_shrink": boundary.get("must_shrink"),
                "must_constrain": boundary.get("must_constrain"),
                "must_be_json_completable": boundary.get("must_be_json_completable"),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "",
        "Full tool-call skeleton with all value slots:",
        json.dumps(slot.get("skeleton_calls_all_slots") or [], ensure_ascii=False, indent=2),
        "",
        "Focused skeleton for this active slot:",
        json.dumps(slot.get("skeleton_calls_focused_slot") or [], ensure_ascii=False, indent=2),
    ]
    if local_peers:
        parts.append("")
        parts.append("Nearby peer arguments:")
        for argument in local_peers[:12]:
            path = argument.get("json_path") or argument.get("argument_path") or argument.get("json_key")
            parts.append(f"- {path}: {json.dumps(argument.get('target'), ensure_ascii=False)}")
    if evidence:
        parts.append("")
        parts.append("Request evidence snippets:")
        for match in evidence[:4]:
            parts.append(f"- {match.get('excerpt')}")
    parts.extend(["", "Candidate values:", compact_candidates(slot)])
    return "\n".join(parts).strip()


def instance_for_slot(slot: dict[str, Any], boundary: dict[str, Any]) -> dict[str, Any]:
    decision = selector_decision(slot, boundary)
    source = (
        f"schedule_state_selector:{slot.get('id')}:call{slot.get('tool_call_index')}:"
        f"{slot.get('json_path') or slot.get('json_key')}:target{slot.get('target_index')}:"
        f"cands{slot.get('candidate_count')}"
    )
    return {
        "system": SYSTEM,
        "messages": [
            {"role": "user", "content": selector_prompt(slot, boundary)},
            {
                "role": "assistant",
                "content": json.dumps(decision, ensure_ascii=False, separators=(",", ":")),
            },
        ],
        "source": source,
        "_selector_meta": {
            "id": slot.get("id"),
            "slot_id": slot.get("slot_id"),
            "tool_call_index": slot.get("tool_call_index"),
            "json_path": slot.get("json_path"),
            "target_index": slot.get("target_index"),
            "candidate_count": slot.get("candidate_count"),
            "is_ambiguous": int(slot.get("candidate_count") or 0) > 1,
        },
    }


def repeat_count(slot: dict[str, Any], args: argparse.Namespace) -> int:
    candidate_count = int(slot.get("candidate_count") or 0)
    target_index = int(slot.get("target_index") or 0)
    if candidate_count <= 1:
        return args.singleton_repeat
    if target_index != 0:
        return args.nonzero_target_repeat
    return args.ambiguous_repeat


def build_instances(slots: list[dict[str, Any]], boundary_by_slot: dict[tuple[Any, Any, Any, Any], dict[str, Any]], args: argparse.Namespace):
    instances = []
    skipped = Counter()
    source_counts = Counter()
    examples = []
    for slot in slots:
        if not slot.get("usable_for_value_training"):
            skipped["not_usable"] += 1
            continue
        target_index = int(slot.get("target_index") if slot.get("target_index") is not None else -1)
        if target_index < 0:
            skipped["missing_target_index"] += 1
            continue
        candidate_count = int(slot.get("candidate_count") or 0)
        if candidate_count <= 1 and not args.include_singletons:
            skipped["singleton_excluded"] += 1
            continue
        if candidate_count > 1 and not args.include_ambiguous:
            skipped["ambiguous_excluded"] += 1
            continue
        boundary = boundary_for_slot(slot, boundary_by_slot)
        repeats = repeat_count(slot, args)
        if repeats <= 0:
            skipped["repeat_zero"] += 1
            continue
        base = instance_for_slot(slot, boundary)
        meta = base["_selector_meta"]
        examples.append(
            {
                **meta,
                "source": base["source"],
                "answer": base["messages"][-1]["content"],
                "candidate_values": slot.get("candidate_values"),
                "target": slot.get("target"),
                "boundary": {
                    "recommended_block_size": boundary.get("recommended_block_size"),
                    "recommended_denoise_steps": boundary.get("recommended_denoise_steps"),
                },
            }
        )
        family = "ambiguous_nonzero_target" if candidate_count > 1 and target_index != 0 else "ambiguous" if candidate_count > 1 else "singleton"
        for repeat_idx in range(repeats):
            item = copy.deepcopy(base)
            item["source"] = f"{base['source']}:repeat{repeat_idx}"
            instances.append(item)
            source_counts[family] += 1
    return instances, examples, skipped, source_counts


def audit_instances(tokenizer, chat_template, instances: list[dict[str, Any]], args: argparse.Namespace):
    accepted = []
    rejected = []
    audit_rows = []
    for instance in instances:
        stats = token_stats(tokenizer, chat_template, instance, args.block_size, args.truncation_side)
        row = {
            "source": instance.get("source") or "unknown",
            "id": (instance.get("_selector_meta") or {}).get("id"),
            "slot_id": (instance.get("_selector_meta") or {}).get("slot_id"),
            **stats,
        }
        audit_rows.append(row)
        if stats["kept_labels"] < args.min_labels:
            rejected.append({**row, "reason": "too_few_labels"})
            continue
        if args.require_full_labels and not stats["full_labels_kept"]:
            rejected.append({**row, "reason": "partial_labels"})
            continue
        accepted.append(instance)
    return accepted, rejected, audit_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--slots-jsonl", type=Path, default=DEFAULT_SLOT_JSONL)
    parser.add_argument("--boundary-jsonl", type=Path, default=DEFAULT_BOUNDARY_JSONL)
    parser.add_argument("--source-summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--conversation-template", default="fast_dllm_v2")
    parser.add_argument("--include-singletons", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-ambiguous", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--singleton-repeat", type=int, default=1)
    parser.add_argument("--ambiguous-repeat", type=int, default=2)
    parser.add_argument("--nonzero-target-repeat", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--truncation-side", choices=["left", "right"], default="left")
    parser.add_argument("--min-labels", type=int, default=1)
    parser.add_argument("--require-full-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diagnostic-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--contains-eval-slice", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=829)
    args = parser.parse_args()

    slots = list(load_jsonl(args.slots_jsonl))
    boundaries = list(load_jsonl(args.boundary_jsonl))
    boundary_by_slot = boundary_lookup(boundaries)
    raw_instances, examples, skipped, source_counts = build_instances(slots, boundary_by_slot, args)
    rng = random.Random(args.seed)
    rng.shuffle(raw_instances)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    chat_template = resolve_chat_template(args.conversation_template)
    accepted, rejected, audit_rows = audit_instances(tokenizer, chat_template, raw_instances, args)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train_agentic_mix.json"
    train_path.write_text(
        json.dumps(
            {"type": "conversation", "instances": [strip_source(instance) for instance in accepted]},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    audit_path = args.out_dir / "train_agentic_mix.audit.jsonl"
    rejected_path = args.out_dir / "train_agentic_mix.rejected.jsonl"
    examples_path = args.out_dir / "selector_examples.jsonl"
    write_jsonl(audit_path, audit_rows)
    write_jsonl(rejected_path, rejected)
    write_jsonl(examples_path, examples)

    raw_slot_counts = Counter("ambiguous" if int(slot.get("candidate_count") or 0) > 1 else "singleton" for slot in slots)
    accepted_meta = Counter()
    for instance in accepted:
        meta = instance.get("_selector_meta") or {}
        accepted_meta["ambiguous" if meta.get("is_ambiguous") else "singleton"] += 1
        if int(meta.get("target_index") or 0) != 0:
            accepted_meta["nonzero_target"] += 1
    source_summary = load_json(args.source_summary_json)
    manifest = {
        "train_path": str(train_path),
        "audit_path": str(audit_path),
        "rejected_path": str(rejected_path),
        "examples_path": str(examples_path),
        "slots_jsonl": str(args.slots_jsonl),
        "boundary_jsonl": str(args.boundary_jsonl),
        "source_summary_json": str(args.source_summary_json),
        "count": len(accepted),
        "raw_count": len(raw_instances),
        "rejected_count": len(rejected),
        "raw_slot_counts": dict(sorted(raw_slot_counts.items())),
        "accepted_counts": dict(sorted(accepted_meta.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "skipped_counts": dict(sorted(skipped.items())),
        "source_artifact_totals": source_summary.get("totals", {}),
        "tokenizer_model": str(args.model),
        "conversation_template": args.conversation_template,
        "objective": "schedule_state_candidate_index_and_policy_selection",
        "answer_schema": {
            "candidate_index": "zero-based integer candidate id",
            "span_kind": "argument_value",
            "protection": "value_candidate_json_prefix_close_guard",
            "block_size": "recommended local diffusion block size",
            "denoise_steps": "recommended local denoising steps",
            "force_candidate_sequence": True,
            "require_json_prefix_safe": True,
            "close_tool_call_only_when_json_complete": True,
        },
        "block_size": args.block_size,
        "truncation_side": args.truncation_side,
        "min_labels": args.min_labels,
        "require_full_labels": args.require_full_labels,
        "repeats": {
            "singleton": args.singleton_repeat,
            "ambiguous": args.ambiguous_repeat,
            "nonzero_target": args.nonzero_target_repeat,
        },
        "diagnostic_only": bool(args.diagnostic_only),
        "contains_eval_slice": bool(args.contains_eval_slice),
        "promotion_allowed": not bool(args.diagnostic_only) and not bool(args.contains_eval_slice),
        "promotion_note": "Promotion still requires separate public and heldout tool-call gates.",
        "chosen_audit_summary": summarize_audit(audit_rows, audit_rows),
        "seed": args.seed,
    }
    manifest_path = args.out_dir / "train_agentic_mix.manifest"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
