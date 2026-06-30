#!/usr/bin/env python3
import argparse
import copy
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from eval_toolcall_jsonl import extract_tool_calls


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_MODEL = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_OUT_DIR = ROOT / "data/flare_agentic_mix_v1"

EVAL_SLICES = [
    ("public_onecall_8", ROOT / "data/toolcall_eval/public_onecall_hermes_smoke.jsonl", 8),
    ("public_multicall_12", ROOT / "data/toolcall_eval/public_multicall_hermes_smoke.jsonl", 12),
    (
        "teacher_heldout_labelaware_8",
        ROOT / "data/toolcall_eval/public_onecall_teacher_heldout_labelaware_smoke.jsonl",
        8,
    ),
]

CATEGORY_SPECS = [
    {
        "name": "raw_public_real_toolcall",
        "target": 170,
        "paths": [ROOT / "data/fastdllm_toolcall_train/train_toolcall_no_public_multicall_smoke.json"],
        "allow_repeats": True,
        "purpose": "Real public one-call/multicall traces; repeated after leak filtering.",
    },
    {
        "name": "onecall_argument_grounding",
        "target": 220,
        "paths": [ROOT / "data/qwen35_9b_toolcall_argument_curriculum/train_agentic_mix.json"],
        "allow_repeats": False,
        "purpose": "One-call argument copy/context/key-value variants.",
    },
    {
        "name": "multicall_sequence_plan",
        "target": 90,
        "paths": [ROOT / "data/qwen35_9b_toolcall_multicall_curriculum/train_agentic_mix.json"],
        "allow_repeats": False,
        "purpose": "Multi-call full traces, continuation splits, and exact-plan examples.",
    },
    {
        "name": "multicall_gap_complex_extract",
        "target": 50,
        "paths": [ROOT / "data/qwen35_9b_toolcall_multicall_gap_curriculum/train_agentic_mix.json"],
        "allow_repeats": False,
        "purpose": "Missing-call recovery and complex argument extraction.",
    },
    {
        "name": "grounded_spanfill_value_copy",
        "target": 44,
        "paths": [ROOT / "data/qwen35_9b_toolcall_grounded_spanfill_synth48_replay_teacher2_b1024_curriculum/train_agentic_mix.json"],
        "allow_repeats": False,
        "purpose": "Grounded value-span copy into JSON arguments.",
    },
    {
        "name": "synthetic_format_toolresult",
        "target": 194,
        "paths": [
            ROOT / "data/synthetic_onecall_train/train_synthetic_onecall.json",
            ROOT / "data/synthetic_toolresult_train/train_synthetic_toolresult.json",
        ],
        "allow_repeats": False,
        "purpose": "Valid JSON/tool-call syntax and tool-result format stabilizers.",
    },
]


def resolve_chat_template(name: str):
    third_party = ROOT / "fast-dllm/third_party"
    if str(third_party) not in sys.path:
        sys.path.insert(0, str(third_party))
    from lmflow.utils.conversation_template import PRESET_TEMPLATES

    if name not in PRESET_TEMPLATES:
        raise ValueError(f"unknown conversation template {name!r}")
    return PRESET_TEMPLATES[name]


def drop_none_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: drop_none_fields(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [drop_none_fields(item) for item in value if item is not None]
    return value


def conversation_for_template(instance: dict[str, Any]) -> list[dict[str, Any]]:
    system = instance.get("system")
    messages = [{"role": "system", "content": system if system is not None else "You are a helpful assistant."}]
    messages.extend(copy.deepcopy(instance.get("messages") or []))
    return drop_none_fields(messages)


def token_stats(tokenizer, chat_template, instance: dict[str, Any], block_size: int, truncation_side: str) -> dict[str, Any]:
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


def load_instances(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    instances = payload.get("instances") if isinstance(payload, dict) else payload
    if not isinstance(instances, list):
        raise ValueError(f"{path} does not contain an instances list")
    return [item for item in instances if isinstance(item, dict)]


def load_eval_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def eval_row_to_instance(row: dict[str, Any]) -> dict[str, Any]:
    messages = copy.deepcopy(row.get("prompt_messages") or [])
    messages.append({"role": "assistant", "content": row.get("gold_assistant") or ""})
    system = ""
    filtered = []
    for message in messages:
        role = message.get("role")
        content = str(message.get("content") or "")
        if role == "system" and not system:
            system = content
            continue
        if role in {"user", "assistant", "tool"} and content:
            filtered.append({"role": role, "content": content})
    return {
        "system": system or "You are a helpful assistant.",
        "messages": filtered,
        "tools": copy.deepcopy(row.get("tools") or []),
    }


def assistant_text(instance: dict[str, Any]) -> str:
    return "\n".join(
        str(message.get("content") or "")
        for message in instance.get("messages") or []
        if message.get("role") == "assistant"
    )


def canonical_instance(instance: dict[str, Any]) -> str:
    kept = {
        "system": instance.get("system"),
        "messages": instance.get("messages") or [],
        "tools": instance.get("tools") or [],
    }
    return json.dumps(kept, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def fingerprint(instance: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_instance(instance).encode("utf-8")).hexdigest()


def normalize_scalar(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return f"bool:{str(value).lower()}"
    if isinstance(value, (int, float)):
        return f"num:{value}"
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return None
    return f"str:{text}"


def leaf_values(value: Any) -> list[str]:
    if isinstance(value, dict):
        out: list[str] = []
        for item in value.values():
            out.extend(leaf_values(item))
        return out
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(leaf_values(item))
        return out
    normalized = normalize_scalar(value)
    return [normalized] if normalized is not None else []


def distinctive_values(values: set[str]) -> set[str]:
    out = set()
    for value in values:
        _, _, raw = value.partition(":")
        if value.startswith("num:"):
            out.add(value)
        elif value.startswith("str:") and (len(raw) >= 3 or any(ch.isdigit() for ch in raw) or any(ch in raw for ch in "-_@:/.")):
            out.add(value)
    return out


def call_signatures(instance: dict[str, Any]) -> list[dict[str, Any]]:
    calls, invalid = extract_tool_calls(assistant_text(instance))
    signatures = []
    for call_index, call in enumerate(calls):
        name = str(call.get("name") or "")
        values = set(leaf_values(call.get("arguments") or {}))
        signatures.append(
            {
                "name": name,
                "values": values,
                "distinctive_values": distinctive_values(values),
                "call_index": call_index,
            }
        )
    return signatures


def build_eval_reference() -> dict[str, Any]:
    slices = {}
    exact_hashes = set()
    eval_calls = []
    for slice_name, path, limit in EVAL_SLICES:
        rows = load_eval_rows(path, limit)
        slice_calls = []
        for row_idx, row in enumerate(rows):
            instance = eval_row_to_instance(row)
            exact_hashes.add(fingerprint(instance))
            for sig in call_signatures(instance):
                item = {
                    "slice": slice_name,
                    "row_idx": row_idx,
                    "id": row.get("id"),
                    "name": sig["name"],
                    "values": sig["values"],
                    "distinctive_values": sig["distinctive_values"],
                }
                eval_calls.append(item)
                slice_calls.append(item)
        slices[slice_name] = {
            "path": str(path),
            "limit": limit,
            "records": len(rows),
            "calls": len(slice_calls),
        }
    return {"slices": slices, "exact_hashes": exact_hashes, "calls": eval_calls}


def find_near_leaks(instance: dict[str, Any], eval_ref: dict[str, Any]) -> list[dict[str, Any]]:
    leaks = []
    if fingerprint(instance) in eval_ref["exact_hashes"]:
        leaks.append({"type": "exact_content_hash"})
    train_calls = call_signatures(instance)
    for train_call in train_calls:
        if not train_call["name"]:
            continue
        for eval_call in eval_ref["calls"]:
            if train_call["name"] != eval_call["name"]:
                continue
            eval_values = eval_call["distinctive_values"] or eval_call["values"]
            if not eval_values:
                continue
            if eval_values.issubset(train_call["values"]):
                leaks.append(
                    {
                        "type": "same_tool_all_eval_arg_values",
                        "slice": eval_call["slice"],
                        "eval_id": eval_call["id"],
                        "tool": train_call["name"],
                        "matched_values": sorted(eval_values),
                    }
                )
    return leaks


def clean_instance(instance: dict[str, Any], source: str) -> dict[str, Any] | None:
    messages = []
    for message in instance.get("messages") or []:
        role = message.get("role")
        content = str(message.get("content") or "").strip()
        if role in {"user", "assistant", "tool"} and content:
            messages.append({"role": role, "content": content})
    if not messages:
        return None
    out = {"messages": messages, "source": source}
    original_source = instance.get("source")
    if original_source is not None and str(original_source) != source:
        out["source_detail"] = str(original_source)
    system = str(instance.get("system") or "").strip()
    if system:
        out["system"] = system
    tools = instance.get("tools") or []
    if tools:
        out["tools"] = copy.deepcopy(tools)
    return out


def retention_source(instance: dict[str, Any]) -> str:
    source = str(instance.get("source") or "")
    if "gsm8k" in source:
        return "general_retention_gsm8k"
    if "mbpp" in source:
        return "general_retention_mbpp"
    return "general_retention_other"


def percentile_summary(values: list[int]) -> dict[str, int]:
    if not values:
        return {"min": 0, "p50": 0, "p90": 0, "max": 0}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "p50": ordered[len(ordered) // 2],
        "p90": ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))],
        "max": ordered[-1],
    }


def candidate_pool(spec: dict[str, Any], tokenizer, chat_template, args, eval_ref: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = []
    for path in spec["paths"]:
        raw.extend(load_instances(path))
    cleaned = []
    rejects = Counter()
    reject_examples = []
    stats_rows = []
    for idx, instance in enumerate(raw):
        item = clean_instance(instance, spec["name"])
        if item is None:
            rejects["empty_or_invalid"] += 1
            continue
        leaks = find_near_leaks(item, eval_ref)
        if leaks:
            rejects["eval_near_duplicate"] += 1
            if len(reject_examples) < 20:
                reject_examples.append({"raw_idx": idx, "leaks": leaks[:3], "source": spec["name"]})
            continue
        stats = token_stats(tokenizer, chat_template, item, args.block_size, args.truncation_side)
        stats_rows.append(stats)
        if not stats["full_labels_kept"]:
            rejects["truncation_loses_labels"] += 1
            if len(reject_examples) < 20:
                reject_examples.append({"raw_idx": idx, "token_stats": stats, "source": spec["name"]})
            continue
        cleaned.append(item)
    summary = {
        "raw_count": len(raw),
        "accepted_count": len(cleaned),
        "rejected_count": sum(rejects.values()),
        "rejects": dict(sorted(rejects.items())),
        "reject_examples": reject_examples,
        "token_length": percentile_summary([row["length"] for row in stats_rows]),
        "full_labels": percentile_summary([row["full_labels"] for row in stats_rows]),
        "kept_labels": percentile_summary([row["kept_labels"] for row in stats_rows]),
        "purpose": spec["purpose"],
    }
    return cleaned, summary


def select_items(pool: list[dict[str, Any]], target: int, allow_repeats: bool, rng: random.Random) -> list[dict[str, Any]]:
    if not pool:
        raise ValueError("empty candidate pool")
    shuffled = [copy.deepcopy(item) for item in pool]
    rng.shuffle(shuffled)
    if len(shuffled) >= target:
        return shuffled[:target]
    if not allow_repeats:
        raise ValueError(f"pool has {len(shuffled)} examples but target is {target}")
    out = []
    cursor = 0
    while len(out) < target:
        if cursor % len(shuffled) == 0:
            rng.shuffle(shuffled)
        out.append(copy.deepcopy(shuffled[cursor % len(shuffled)]))
        cursor += 1
    return out


def leak_table(instances: list[dict[str, Any]], eval_ref: dict[str, Any]) -> dict[str, Any]:
    per_slice = {
        name: {
            "records": meta["records"],
            "eval_calls": meta["calls"],
            "exact_content_hash_matches": 0,
            "same_tool_all_eval_arg_values_matches": 0,
            "matched_training_examples": [],
        }
        for name, meta in eval_ref["slices"].items()
    }
    total_exact = 0
    total_near = 0
    for train_idx, instance in enumerate(instances):
        exact = fingerprint(instance) in eval_ref["exact_hashes"]
        leaks = find_near_leaks(instance, eval_ref)
        if exact:
            total_exact += 1
            for row in per_slice.values():
                row["exact_content_hash_matches"] += 1
        for leak in leaks:
            if leak.get("type") != "same_tool_all_eval_arg_values":
                continue
            total_near += 1
            row = per_slice[leak["slice"]]
            row["same_tool_all_eval_arg_values_matches"] += 1
            if len(row["matched_training_examples"]) < 10:
                row["matched_training_examples"].append(
                    {
                        "train_idx": train_idx,
                        "train_source": instance.get("source"),
                        "eval_id": leak.get("eval_id"),
                        "tool": leak.get("tool"),
                        "matched_values": leak.get("matched_values", [])[:12],
                    }
                )
    return {"total_exact_matches": total_exact, "total_near_matches": total_near, "per_slice": per_slice}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_leak_markdown(path: Path, table: dict[str, Any]) -> None:
    lines = [
        "# FLARE Agentic Mix V1 Leak Check",
        "",
        "| Eval slice | Records | Eval calls | Exact hash matches | Same tool + all eval arg values |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, row in table["per_slice"].items():
        lines.append(
            f"| {name} | {row['records']} | {row['eval_calls']} | "
            f"{row['exact_content_hash_matches']} | {row['same_tool_all_eval_arg_values_matches']} |"
        )
    lines.extend(["", f"Total exact matches: `{table['total_exact_matches']}`", f"Total near matches: `{table['total_near_matches']}`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--retention", type=Path, default=ROOT / "data/flare_stage1_ab_pilot_train/train_agentic_mix.json")
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--truncation-side", choices=["left", "right"], default="right")
    parser.add_argument("--conversation-template", default="fast_dllm_v2")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    eval_ref = build_eval_reference()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    chat_template = resolve_chat_template(args.conversation_template)

    selected = []
    category_manifest = {}
    for spec in CATEGORY_SPECS:
        pool, pool_summary = candidate_pool(spec, tokenizer, chat_template, args, eval_ref)
        items = select_items(pool, spec["target"], spec["allow_repeats"], rng)
        selected.extend(items)
        category_manifest[spec["name"]] = {
            **pool_summary,
            "target": spec["target"],
            "selected_count": len(items),
            "unique_selected_count": len({fingerprint(item) for item in items}),
            "allow_repeats": spec["allow_repeats"],
            "paths": [str(path) for path in spec["paths"]],
        }

    retention_raw = load_instances(args.retention)
    retention = []
    retention_stats = []
    for instance in retention_raw:
        item = clean_instance(instance, retention_source(instance))
        if item is None:
            continue
        stats = token_stats(tokenizer, chat_template, item, args.block_size, args.truncation_side)
        retention_stats.append(stats)
        if not stats["full_labels_kept"]:
            raise ValueError(f"retention example loses labels under block_size={args.block_size}: {stats}")
        retention.append(item)
    if len(retention) != 256:
        raise ValueError(f"expected 256 retention examples, got {len(retention)}")

    rng.shuffle(selected)
    rng.shuffle(retention)
    final_instances = selected + retention
    final_leak = leak_table(selected, eval_ref)
    if final_leak["total_exact_matches"] or final_leak["total_near_matches"]:
        raise ValueError(f"leak check failed: {final_leak}")

    source_counts = Counter(item.get("source") or "unknown" for item in final_instances)
    toolcall_source_counts = Counter(item.get("source") or "unknown" for item in selected)
    call_dist = Counter()
    for item in selected:
        call_dist[len(call_signatures(item))] += 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train_agentic_mix.json"
    manifest_path = args.out_dir / "manifest.json"
    leak_path = args.out_dir / "leak_check.json"
    leak_md_path = args.out_dir / "leak_check.md"

    write_json(train_path, {"type": "conversation", "instances": final_instances})
    manifest = {
        "train_path": str(train_path),
        "count": len(final_instances),
        "toolcall_count": len(selected),
        "retention_count": len(retention),
        "source_counts": dict(sorted(source_counts.items())),
        "toolcall_source_counts": dict(sorted(toolcall_source_counts.items())),
        "toolcall_call_count_distribution": dict(sorted((str(k), v) for k, v in call_dist.items())),
        "category_manifest": category_manifest,
        "retention_manifest": {
            "path": str(args.retention),
            "selected_count": len(retention),
            "source_counts": dict(sorted(Counter(item.get("source") or "unknown" for item in retention).items())),
            "token_length": percentile_summary([row["length"] for row in retention_stats]),
            "full_labels": percentile_summary([row["full_labels"] for row in retention_stats]),
            "kept_labels": percentile_summary([row["kept_labels"] for row in retention_stats]),
        },
        "eval_slices": eval_ref["slices"],
        "leak_check": final_leak,
        "seed": args.seed,
        "block_size": args.block_size,
        "truncation_side": args.truncation_side,
        "conversation_template": args.conversation_template,
        "success_bar": {
            "valid_json": "15+/28",
            "exact_args": "8+/24",
            "gsm8k_retention_floor": "strict accuracy >= 0.50",
        },
    }
    write_json(manifest_path, manifest)
    write_json(leak_path, final_leak)
    write_leak_markdown(leak_md_path, final_leak)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
