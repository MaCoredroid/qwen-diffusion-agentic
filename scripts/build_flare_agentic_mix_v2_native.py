#!/usr/bin/env python3
import argparse
import copy
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from build_flare_agentic_mix_v1 import (
    ROOT,
    call_signatures,
    clean_instance,
    eval_row_to_instance,
    find_near_leaks,
    fingerprint,
    leak_table,
    load_eval_rows,
    load_instances,
    percentile_summary,
    resolve_chat_template,
    retention_source,
    token_stats,
    write_json,
    write_leak_markdown,
)
from convert_toolcall_cases_to_qwen_native import convert_eval_row, convert_instance, native_instruction_text
from eval_toolcall_jsonl import qwen_native_tool_call_text


DEFAULT_MODEL = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_OUT_DIR = ROOT / "data/flare_agentic_mix_v2_native"

EVAL_SLICES = [
    ("public_onecall_8", ROOT / "data/toolcall_eval_native/public_onecall_qwen_native_smoke.jsonl", 8),
    ("public_multicall_12", ROOT / "data/toolcall_eval_native/public_multicall_qwen_native_smoke.jsonl", 12),
    (
        "teacher_heldout_labelaware_8",
        ROOT / "data/toolcall_eval_native/public_onecall_teacher_heldout_labelaware_qwen_native_smoke.jsonl",
        8,
    ),
]

TOOL_CATEGORY_SPECS = [
    {
        "name": "verified_teacher_native_exact",
        "target": 28,
        "kind": "teacher_verified_jsonl",
        "paths": [
            ROOT / "runs/flare_agentic_mix_v2_teacher_native_fresh30/teacher_q36_native_t07.jsonl",
            ROOT / "runs/flare_agentic_mix_v2_teacher_native_fresh30/teacher_q36_native_oldpilot_t07.jsonl",
        ],
        "case_paths": [
            ROOT / "data/flare_agentic_mix_v2_pilot/teacher_native_fresh30_cases.jsonl",
            ROOT / "data/flare_agentic_mix_v2_pilot/teacher_pilot_cases.jsonl",
        ],
        "allow_repeats": False,
        "purpose": "Qwen3.6 teacher traces, native format, filtered to exact-arguments-correct only.",
    },
    {
        "name": "raw_public_real_toolcall_native",
        "target": 96,
        "kind": "train_json",
        "paths": [ROOT / "data/fastdllm_toolcall_train/train_toolcall_no_public_multicall_smoke.json"],
        "allow_repeats": False,
        "purpose": "Diverse real one-call/multicall traces converted to Qwen-native targets.",
    },
    {
        "name": "public_multicall_gold_native",
        "target": 56,
        "kind": "eval_jsonl",
        "paths": [ROOT / "data/toolcall_eval/public_train_multicall_gold_cases.jsonl"],
        "allow_repeats": False,
        "purpose": "Gold public multicall traces converted to Qwen-native targets.",
    },
    {
        "name": "public_multicall_no_public_native",
        "target": 45,
        "kind": "eval_jsonl",
        "paths": [ROOT / "data/toolcall_eval/public_train_multicall_no_public_smoke_cases.jsonl"],
        "allow_repeats": False,
        "purpose": "Public-train-derived multicall traces excluding public eval smoke cases.",
    },
    {
        "name": "onecall_argument_grounding_native",
        "target": 96,
        "kind": "train_json",
        "paths": [ROOT / "data/qwen35_9b_toolcall_argument_curriculum/train_agentic_mix.json"],
        "allow_repeats": False,
        "purpose": "One-call argument copy/context/key-value curriculum converted to native.",
    },
    {
        "name": "multicall_sequence_plan_native",
        "target": 96,
        "kind": "train_json",
        "paths": [ROOT / "data/qwen35_9b_toolcall_multicall_curriculum/train_agentic_mix.json"],
        "allow_repeats": False,
        "purpose": "Multi-call full traces, continuation splits, and exact-plan examples converted to native.",
    },
    {
        "name": "multicall_gap_complex_extract_native",
        "target": 50,
        "kind": "train_json",
        "paths": [ROOT / "data/qwen35_9b_toolcall_multicall_gap_curriculum/train_agentic_mix.json"],
        "allow_repeats": False,
        "purpose": "Missing-call recovery and complex argument extraction converted to native.",
    },
    {
        "name": "grounded_spanfill_value_copy_native",
        "target": 45,
        "kind": "train_json",
        "paths": [
            ROOT
            / "data/qwen35_9b_toolcall_grounded_spanfill_synth48_replay_teacher2_b1024_curriculum/train_agentic_mix.json"
        ],
        "allow_repeats": False,
        "purpose": "Grounded value-span copy into native function parameters.",
    },
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def native_assistant_present(instance: dict[str, Any]) -> bool:
    for message in instance.get("messages") or []:
        if message.get("role") != "assistant":
            continue
        text = str(message.get("content") or "")
        if "<tool_call>" in text and "<function=" in text and "<parameter=" in text:
            return True
    return False


def build_eval_reference() -> dict[str, Any]:
    slices = {}
    exact_hashes = set()
    eval_calls = []
    for slice_name, path, limit in EVAL_SLICES:
        rows = load_eval_rows(path, limit)
        slice_calls = []
        for row_idx, row in enumerate(rows):
            native_row, _ = convert_eval_row(row)
            instance = eval_row_to_instance(native_row)
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


def prompt_instance_from_case(case: dict[str, Any], assistant_text: str, source: str) -> dict[str, Any]:
    system = ""
    messages = []
    for message in case.get("prompt_messages") or []:
        role = message.get("role")
        content = native_instruction_text(str(message.get("content") or "")).strip()
        if role == "system" and not system:
            system = content
            continue
        if role in {"user", "tool"} and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "assistant", "content": assistant_text})
    return {
        "system": system or "You are a helpful assistant.",
        "messages": messages,
        "tools": copy.deepcopy(case.get("tools") or []),
        "source": source,
    }


def teacher_verified_instances(spec: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases_by_id = {}
    for path in spec.get("case_paths") or []:
        for case in load_jsonl(path):
            if case.get("id"):
                cases_by_id[str(case["id"])] = case

    out = []
    rejects = Counter()
    raw_count = 0
    for path in spec["paths"]:
        for row in load_jsonl(path):
            raw_count += 1
            if row.get("status") != "ok":
                rejects["teacher_status_not_ok"] += 1
                continue
            if row.get("exact_arguments") is not True:
                rejects["teacher_not_exact_arguments"] += 1
                continue
            case = cases_by_id.get(str(row.get("id")))
            if case is None:
                rejects["missing_source_case"] += 1
                continue
            calls = row.get("teacher_calls") or []
            if not calls:
                rejects["missing_teacher_calls"] += 1
                continue
            assistant = qwen_native_tool_call_text(calls)
            out.append(prompt_instance_from_case(case, assistant, spec["name"]))
    summary = {
        "raw_count": raw_count,
        "accepted_before_leak_length": len(out),
        "teacher_exact_arguments_count": len(out),
        "rejects_pre_filter": dict(sorted(rejects.items())),
        "purpose": spec["purpose"],
    }
    return out, summary


def train_json_instances(spec: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out = []
    raw_count = 0
    changed = 0
    rejected = Counter()
    for path in spec["paths"]:
        for instance in load_instances(path):
            raw_count += 1
            converted, did_change = convert_instance(instance)
            changed += int(did_change)
            cleaned = clean_instance(converted, spec["name"])
            if cleaned is None:
                rejected["empty_or_invalid"] += 1
                continue
            if not native_assistant_present(cleaned):
                rejected["no_native_tool_assistant"] += 1
                continue
            out.append(cleaned)
    summary = {
        "raw_count": raw_count,
        "converted_count": changed,
        "accepted_before_leak_length": len(out),
        "rejects_pre_filter": dict(sorted(rejected.items())),
        "purpose": spec["purpose"],
    }
    return out, summary


def eval_jsonl_instances(spec: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out = []
    raw_count = 0
    converted_count = 0
    rejected = Counter()
    for path in spec["paths"]:
        for row in load_jsonl(path):
            raw_count += 1
            native_row, converted = convert_eval_row(row)
            converted_count += int(converted)
            instance = eval_row_to_instance(native_row)
            instance["source"] = spec["name"]
            if not native_assistant_present(instance):
                rejected["no_native_tool_assistant"] += 1
                continue
            out.append(instance)
    summary = {
        "raw_count": raw_count,
        "converted_count": converted_count,
        "accepted_before_leak_length": len(out),
        "rejects_pre_filter": dict(sorted(rejected.items())),
        "purpose": spec["purpose"],
    }
    return out, summary


def filter_pool(pool, spec, tokenizer, chat_template, args, eval_ref):
    cleaned = []
    rejects = Counter()
    reject_examples = []
    stats_rows = []
    for idx, item in enumerate(pool):
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
    return cleaned, {
        "accepted_count": len(cleaned),
        "rejected_count": sum(rejects.values()),
        "rejects": dict(sorted(rejects.items())),
        "reject_examples": reject_examples,
        "token_length": percentile_summary([row["length"] for row in stats_rows]),
        "full_labels": percentile_summary([row["full_labels"] for row in stats_rows]),
        "kept_labels": percentile_summary([row["kept_labels"] for row in stats_rows]),
    }


def raw_pool(spec: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if spec["kind"] == "teacher_verified_jsonl":
        return teacher_verified_instances(spec)
    if spec["kind"] == "train_json":
        return train_json_instances(spec)
    if spec["kind"] == "eval_jsonl":
        return eval_jsonl_instances(spec)
    raise ValueError(f"unsupported spec kind {spec['kind']!r}")


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


def write_manifest_markdown(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# FLARE Agentic Mix V2 Native",
        "",
        "All tool-call targets are Qwen-native `<function=...>/<parameter=...>` targets.",
        "Teacher rows are included only when the teacher output is exact-arguments-correct.",
        "",
        "## Breakdown",
        "",
        "| Source | Selected | Accepted | Raw | Purpose |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for name, row in manifest["category_manifest"].items():
        lines.append(
            f"| {name} | {row['selected_count']} | {row['accepted_count']} | "
            f"{row['raw_count']} | {row.get('purpose','')} |"
        )
    lines.extend(
        [
            "",
            "## Leak Check",
            "",
            f"Total exact matches: `{manifest['leak_check']['total_exact_matches']}`",
            f"Total near matches: `{manifest['leak_check']['total_near_matches']}`",
            "",
            "## Train Start Recommendation",
            "",
            manifest["training_recommendation"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--retention", type=Path, default=ROOT / "data/flare_stage1_ab_pilot_train/train_agentic_mix.json")
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--truncation-side", choices=["left", "right"], default="right")
    parser.add_argument("--conversation-template", default="fast_dllm_v2")
    parser.add_argument("--retention-target", type=int, default=512)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    eval_ref = build_eval_reference()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    chat_template = resolve_chat_template(args.conversation_template)

    selected = []
    category_manifest = {}
    for spec in TOOL_CATEGORY_SPECS:
        pool, pre_summary = raw_pool(spec)
        filtered, filter_summary = filter_pool(pool, spec, tokenizer, chat_template, args, eval_ref)
        items = select_items(filtered, spec["target"], spec["allow_repeats"], rng)
        selected.extend(items)
        category_manifest[spec["name"]] = {
            **pre_summary,
            **filter_summary,
            "target": spec["target"],
            "selected_count": len(items),
            "unique_selected_count": len({fingerprint(item) for item in items}),
            "allow_repeats": spec["allow_repeats"],
            "paths": [str(path) for path in spec["paths"]],
            "kind": spec["kind"],
        }

    retention_raw = load_instances(args.retention)
    retention_pool = []
    retention_stats = []
    for instance in retention_raw:
        item = clean_instance(instance, retention_source(instance))
        if item is None:
            continue
        stats = token_stats(tokenizer, chat_template, item, args.block_size, args.truncation_side)
        retention_stats.append(stats)
        if not stats["full_labels_kept"]:
            raise ValueError(f"retention example loses labels under block_size={args.block_size}: {stats}")
        retention_pool.append(item)
    if len(retention_pool) != 256:
        raise ValueError(f"expected 256 retention examples, got {len(retention_pool)}")

    retention = []
    retention_cycle = [copy.deepcopy(item) for item in retention_pool]
    cursor = 0
    while len(retention) < args.retention_target:
        if cursor % len(retention_cycle) == 0:
            rng.shuffle(retention_cycle)
        retention.append(copy.deepcopy(retention_cycle[cursor % len(retention_cycle)]))
        cursor += 1

    rng.shuffle(selected)
    rng.shuffle(retention)
    final_instances = selected + retention
    final_leak = leak_table(selected, eval_ref)
    if final_leak["total_exact_matches"] or final_leak["total_near_matches"]:
        raise ValueError(f"leak check failed: {final_leak}")

    source_counts = Counter(item.get("source") or "unknown" for item in final_instances)
    toolcall_source_counts = Counter(item.get("source") or "unknown" for item in selected)
    call_dist = Counter()
    unique_tools = set()
    for item in selected:
        signatures = call_signatures(item)
        call_dist[len(signatures)] += 1
        for sig in signatures:
            if sig["name"]:
                unique_tools.add(sig["name"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train_agentic_mix.json"
    manifest_path = args.out_dir / "manifest.json"
    leak_path = args.out_dir / "leak_check.json"
    leak_md_path = args.out_dir / "leak_check.md"
    manifest_md_path = args.out_dir / "manifest.md"

    write_json(train_path, {"type": "conversation", "instances": final_instances})
    training_recommendation = (
        "Run the native-everywhere pilot from the diffusion init, not from agentic-v1. "
        "B@1000 is still a useful transfer baseline, but the apples-to-apples native result "
        "should retrain on native targets with 50% retention rather than preserving old "
        "Hermes-format tool-call assumptions. If wall-clock pressure forces a shorter probe, "
        "B@1000 can be used as an ablation, but it is not the design anchor."
    )
    manifest = {
        "train_path": str(train_path),
        "count": len(final_instances),
        "toolcall_count": len(selected),
        "retention_count": len(retention),
        "retention_unique_count": len({fingerprint(item) for item in retention}),
        "native_format": "qwen_native_function_parameter",
        "source_counts": dict(sorted(source_counts.items())),
        "toolcall_source_counts": dict(sorted(toolcall_source_counts.items())),
        "toolcall_call_count_distribution": dict(sorted((str(k), v) for k, v in call_dist.items())),
        "toolcall_unique_tools": len(unique_tools),
        "category_manifest": category_manifest,
        "retention_manifest": {
            "path": str(args.retention),
            "selected_count": len(retention),
            "unique_selected_count": len({fingerprint(item) for item in retention}),
            "source_counts": dict(sorted(Counter(item.get("source") or "unknown" for item in retention).items())),
            "token_length": percentile_summary([row["length"] for row in retention_stats]),
            "full_labels": percentile_summary([row["full_labels"] for row in retention_stats]),
            "kept_labels": percentile_summary([row["kept_labels"] for row in retention_stats]),
            "repeat_policy": "256-row retention slice repeated/shuffled to reach 512; heldout NLL/eval data is not used.",
        },
        "eval_slices": eval_ref["slices"],
        "leak_check": final_leak,
        "seed": args.seed,
        "block_size": args.block_size,
        "truncation_side": args.truncation_side,
        "conversation_template": args.conversation_template,
        "training_recommendation": training_recommendation,
        "success_bar": {
            "native_live_decoder_exact_args": "beat transfer 19/28 and old post-hoc 9/28",
            "gsm8k_retention_floor": "strict accuracy >= 0.65 preferred; <0.50 fail",
        },
    }
    write_json(manifest_path, manifest)
    write_json(leak_path, final_leak)
    write_leak_markdown(leak_md_path, final_leak)
    write_manifest_markdown(manifest_md_path, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
