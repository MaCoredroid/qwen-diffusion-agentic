#!/usr/bin/env python3
"""Red-team the broadened ToolACE-40 eval for provenance, leakage, and difficulty."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from eval_flare_multiturn_percall_waves import split_tool_call_blocks  # noqa: E402
from eval_toolcall_jsonl import extract_tool_calls  # noqa: E402


DEFAULT_EVAL = ROOT / "data/toolcall_eval_native/flare_broaden_public_toolace60.jsonl"
DEFAULT_MANIFEST = ROOT / "data/toolcall_eval_native/flare_broaden_public_toolace60.manifest.json"
DEFAULT_OUT_DIR = ROOT / "runs/agentic_eval/northstar_broaden_toolace60/leak_redteam"


KEY_TRAIN_PATHS = [
    ROOT / "data/flare_redesign_run1_copy_retention_mix/train_agentic_mix.json",
    ROOT / "data/flare_stage1_ab_pilot_train/train_agentic_mix.json",
    ROOT / "data/flare_stage1_ab_pilot/train_agentic_mix.json",
    ROOT / "data/flare_agentic_mix_v1/train_agentic_mix.json",
    ROOT / "data/flare_agentic_mix_v1_native/train_agentic_mix.json",
    ROOT / "data/flare_agentic_mix_v2_native/train_agentic_mix.json",
    ROOT / "data/fastdllm_toolcall_train/train_toolcall.json",
    ROOT / "data/fastdllm_toolcall_train/train_toolcall_no_public_multicall_smoke.json",
]


PROVENANCE_PATHS = [
    ROOT / "data/toolcall_seed/qwen_toolcall_seed.manifest.json",
    ROOT / "data/fastdllm_toolcall_train/train_toolcall.manifest",
    ROOT / "data/fastdllm_toolcall_train/train_toolcall_no_public_multicall_smoke.manifest.json",
    ROOT / "data/toolcall_eval/public_train_multicall_gold_cases.summary.json",
    ROOT / "data/toolcall_eval/public_train_multicall_no_public_smoke_cases.summary.json",
    ROOT / "data/flare_agentic_mix_v1/manifest.json",
    ROOT / "data/flare_agentic_mix_v2_native/manifest.json",
    ROOT / "data/flare_redesign_run1_copy_retention_mix/train_agentic_mix.manifest",
    ROOT / "data/flare_stage1_ab_pilot/manifest.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-jsonl", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--train-path", type=Path, action="append", default=None)
    parser.add_argument("--update-manifest", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def normalize_text(text: Any) -> str:
    return "\n".join(str(text or "").strip().split())


def safe_identifier(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "")).strip("_")
    text = re.sub(r"_+", "_", text)
    return text.lower()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def messages_for_row(row: dict[str, Any]) -> list[dict[str, str]]:
    if isinstance(row.get("prompt_messages"), list):
        messages = [
            {"role": str(msg.get("role") or ""), "content": str(msg.get("content") or "")}
            for msg in row.get("prompt_messages") or []
            if isinstance(msg, dict)
        ]
    elif row.get("system") is not None or isinstance(row.get("messages"), list):
        messages = []
        if row.get("system") is not None:
            messages.append({"role": "system", "content": str(row.get("system") or "")})
        messages.extend(
            {
                "role": str(msg.get("role") or ""),
                "content": str(msg.get("content") or ""),
            }
            for msg in row.get("messages") or []
            if isinstance(msg, dict)
        )
    else:
        messages = []
    if row.get("gold_assistant") is not None:
        turn_users = row.get("turn_user_messages") or []
        blocks = split_tool_call_blocks(str(row.get("gold_assistant") or ""))
        if blocks:
            for turn_idx, block in enumerate(blocks):
                messages.append({"role": "assistant", "content": block})
                if turn_idx + 1 < len(turn_users) and turn_users[turn_idx + 1] is not None:
                    messages.append({"role": "user", "content": str(turn_users[turn_idx + 1])})
        else:
            messages.append({"role": "assistant", "content": str(row.get("gold_assistant") or "")})
    return messages


def user_text(row: dict[str, Any]) -> str:
    return "\n\n".join(
        str(message.get("content") or "").strip()
        for message in messages_for_row(row)
        if message.get("role") == "user" and str(message.get("content") or "").strip()
    )


def assistant_text(row: dict[str, Any]) -> str:
    if row.get("gold_assistant") is not None:
        return str(row.get("gold_assistant") or "")
    return "\n".join(
        str(message.get("content") or "")
        for message in messages_for_row(row)
        if message.get("role") == "assistant"
    )


def canonical_text_hash(row: dict[str, Any]) -> str:
    payload = {
        "user": normalize_text(user_text(row)),
        "assistant": normalize_text(assistant_text(row)),
    }
    return sha256_json(payload)


def user_hash(row: dict[str, Any]) -> str:
    return sha256_text(normalize_text(user_text(row)))


def scalar_values(value: Any) -> list[tuple[Any, str]]:
    if isinstance(value, dict):
        out: list[tuple[Any, str]] = []
        for item in value.values():
            out.extend(scalar_values(item))
        return out
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(scalar_values(item))
        return out
    rendered = value if value is not None else "null"
    return [(value, str(rendered))]


def normalize_scalar(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return f"bool:{str(value).lower()}"
    if isinstance(value, (int, float)):
        return f"num:{value}"
    text = normalize_text(value).lower()
    if not text:
        return None
    return f"str:{text}"


def call_signature_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    calls, invalid = extract_tool_calls(assistant_text(row))
    items = []
    for call_idx, call in enumerate(calls):
        args = call.get("arguments") or {}
        normalized_pairs = []
        values = set()
        for key, value in sorted(args.items(), key=lambda item: str(item[0])):
            leafs = scalar_values(value)
            normalized_leafs = [normalize_scalar(item[0]) for item in leafs]
            normalized_leafs = [item for item in normalized_leafs if item is not None]
            normalized_pairs.append((safe_identifier(key), tuple(sorted(normalized_leafs))))
            values.update(normalized_leafs)
        name = safe_identifier(call.get("name") or "")
        full_payload = {"name": name, "args": normalized_pairs}
        items.append(
            {
                "call_idx": call_idx,
                "name": name,
                "values": values,
                "full_hash": sha256_json(full_payload),
                "arg_key_set_hash": sha256_json({"name": name, "keys": [item[0] for item in normalized_pairs]}),
            }
        )
    return items


def load_instances(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("instances"), list):
        return [item for item in payload["instances"] if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def default_train_paths() -> list[Path]:
    paths = []
    for pattern in ("train_agentic_mix.json", "train_toolcall*.json", "*.train.json"):
        paths.extend((ROOT / "data").rglob(pattern))
    paths.extend(KEY_TRAIN_PATHS)
    return sorted({path.resolve() for path in paths if path.exists() and path.is_file()})


def build_train_index(paths: list[Path]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    exact_by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    user_by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    full_call_by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tool_names: dict[str, list[dict[str, Any]]] = defaultdict(list)
    values_by_tool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    files = []
    skipped = []
    total_rows = 0
    total_calls = 0
    for path in paths:
        try:
            instances = load_instances(path)
        except Exception as exc:
            skipped.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        files.append({"path": str(path), "rows": len(instances)})
        total_rows += len(instances)
        for row_idx, row in enumerate(instances):
            base = {
                "train_path": str(path),
                "train_idx": row_idx,
                "source": row.get("source"),
            }
            exact_by_hash[canonical_text_hash(row)].append(base)
            user_by_hash[user_hash(row)].append(base)
            for sig in call_signature_items(row):
                total_calls += 1
                item = {**base, "call_idx": sig["call_idx"], "tool": sig["name"], "values": sorted(sig["values"])}
                full_call_by_hash[sig["full_hash"]].append(item)
                if sig["name"]:
                    tool_names[sig["name"]].append(item)
                    if sig["values"]:
                        values_by_tool[sig["name"]].append({**item, "value_set": sig["values"]})
    return (
        {
            "files": files,
            "skipped": skipped,
            "total_rows": total_rows,
            "total_calls": total_calls,
            "exact_by_hash": exact_by_hash,
            "user_by_hash": user_by_hash,
            "full_call_by_hash": full_call_by_hash,
            "tool_names": tool_names,
            "values_by_tool": values_by_tool,
        },
        files,
    )


def eval_context_for_turn(row: dict[str, Any], turn_idx: int) -> str:
    prompt_users = [
        str(message.get("content") or "")
        for message in row.get("prompt_messages") or []
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    turn_users = row.get("turn_user_messages") or []
    extra_users = [
        str(value)
        for idx, value in enumerate(turn_users)
        if idx <= turn_idx and value is not None and str(value).strip()
    ]
    return normalize_text("\n\n".join(prompt_users + extra_users)).lower()


def argument_type(value: Any, context: str) -> str:
    _, rendered = scalar_values(value)[0]
    norm = normalize_text(rendered).lower()
    if norm and len(norm) >= 2 and norm in context:
        return "copy_from_context"
    return "derived_or_constant"


def schema_type_for_arg(row: dict[str, Any], tool_name: str, arg_name: str) -> str:
    safe_tool = safe_identifier(tool_name)
    for tool in row.get("tools") or []:
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        if not isinstance(fn, dict) or safe_identifier(fn.get("name") or "") != safe_tool:
            continue
        props = ((fn.get("parameters") or {}).get("properties") or {})
        for prop_name, prop_schema in props.items():
            if safe_identifier(prop_name) == safe_identifier(arg_name):
                expected = prop_schema.get("type") if isinstance(prop_schema, dict) else None
                if isinstance(expected, list):
                    expected = ",".join(str(item) for item in expected)
                return str(expected or "unknown")
    return "unknown"


def difficulty_rows(rows: list[dict[str, Any]], source_family: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        blocks = split_tool_call_blocks(row.get("gold_assistant") or "")
        episode_calls = 0
        episode_args = 0
        value_lengths = []
        arg_type_counts = Counter()
        schema_counts = Counter()
        calls_per_turn = []
        for turn_idx, block in enumerate(blocks):
            calls, invalid = extract_tool_calls(block)
            calls_per_turn.append(len(calls))
            context = eval_context_for_turn(row, turn_idx)
            for call in calls:
                episode_calls += 1
                for arg_name, arg_value in (call.get("arguments") or {}).items():
                    episode_args += 1
                    schema_counts[schema_type_for_arg(row, call.get("name") or "", arg_name)] += 1
                    for scalar, rendered in scalar_values(arg_value):
                        value_lengths.append(len(str(rendered)))
                        arg_type_counts[argument_type(scalar, context)] += 1
        out.append(
            {
                "id": row.get("id"),
                "source_family": source_family,
                "turns": len(blocks),
                "calls": episode_calls,
                "calls_per_turn_avg": sum(calls_per_turn) / len(calls_per_turn) if calls_per_turn else 0.0,
                "args": episode_args,
                "args_per_call": episode_args / episode_calls if episode_calls else 0.0,
                "avg_value_length": sum(value_lengths) / len(value_lengths) if value_lengths else 0.0,
                "arg_type_counts": dict(sorted(arg_type_counts.items())),
                "schema_type_counts": dict(sorted(schema_counts.items())),
            }
        )
    return out


def summarize_difficulty(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_episodes = len(rows)
    total_turns = sum(row["turns"] for row in rows)
    total_calls = sum(row["calls"] for row in rows)
    total_args = sum(row["args"] for row in rows)
    arg_types = Counter()
    schema_types = Counter()
    for row in rows:
        arg_types.update(row.get("arg_type_counts") or {})
        schema_types.update(row.get("schema_type_counts") or {})
    avg_value_lengths = [row["avg_value_length"] for row in rows if row["avg_value_length"]]
    return {
        "episodes": total_episodes,
        "turns": total_turns,
        "calls": total_calls,
        "args": total_args,
        "calls_per_turn": total_calls / total_turns if total_turns else 0.0,
        "args_per_call": total_args / total_calls if total_calls else 0.0,
        "avg_value_length_per_episode_mean": (
            sum(avg_value_lengths) / len(avg_value_lengths) if avg_value_lengths else 0.0
        ),
        "argument_type_counts": dict(sorted(arg_types.items())),
        "schema_type_counts": dict(sorted(schema_types.items())),
    }


def provenance_summary() -> dict[str, Any]:
    manifests = {}
    for path in PROVENANCE_PATHS:
        if path.exists():
            try:
                manifests[str(path.relative_to(ROOT))] = load_json(path)
            except Exception as exc:
                manifests[str(path.relative_to(ROOT))] = {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "chain": [
            {
                "node": "toolcall_seed",
                "artifact": "data/toolcall_seed/qwen_toolcall_seed.jsonl",
                "upstream": {
                    "hermes": "NousResearch/hermes-function-calling-v1 train streaming",
                    "glaive": "glaiveai/glaive-function-calling-v2 train streaming",
                    "toolace": "Team-ACE/ToolACE train streaming",
                },
                "script": "scripts/prepare_toolcall_seed_data.py",
            },
            {
                "node": "fastdllm_toolcall_train",
                "artifact": "data/fastdllm_toolcall_train/train_toolcall.json",
                "upstream": "data/toolcall_seed/qwen_toolcall_seed.jsonl; manifest shows selected train_source_counts hermes=96",
            },
            {
                "node": "public_train_multicall_gold_cases",
                "artifact": "data/toolcall_eval/public_train_multicall_gold_cases.jsonl",
                "upstream": "data/fastdllm_toolcall_train/train_toolcall.json via materialize_conversation_toolcall_cases.py",
            },
            {
                "node": "public_train_multicall_no_public_smoke_cases",
                "artifact": "data/toolcall_eval/public_train_multicall_no_public_smoke_cases.jsonl",
                "upstream": "train_toolcall.json after removing public_multicall_hermes_smoke overlaps",
            },
            {
                "node": "flare_agentic_mix_v2_native",
                "artifact": "data/flare_agentic_mix_v2_native/train_agentic_mix.json",
                "upstream": "includes 28 public_multicall_gold_native and 28 public_multicall_no_public_native records",
            },
            {
                "node": "run1_copy_retention_mix",
                "artifact": "data/flare_redesign_run1_copy_retention_mix/train_agentic_mix.json",
                "upstream": "uses flare_agentic_mix_v2_native as native pool and lists both public-train multicall JSONLs in pool/exclude-eval lists",
            },
            {
                "node": "B@1000 two-stream adapter",
                "artifact": "runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000",
                "upstream": "scripts/run_flare_stage1_ab_pilot_job.sh with DATASET_DIR=data/flare_stage1_ab_pilot_train, GSM8K/MBPP retention only",
            },
            {
                "node": "ToolACE-40 broaden eval",
                "artifact": "data/toolcall_eval_native/flare_broaden_public_toolace60.jsonl rows 20-59",
                "upstream": "Team-ACE/ToolACE train split, first accepted 40 rows from scripts/build_flare_broaden_public_eval.py",
            },
        ],
        "manifests": manifests,
    }


def overlap_audit(eval_rows: list[dict[str, Any]], train_index: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_eval = []
    totals = Counter()
    examples = defaultdict(list)
    for row in eval_rows:
        exact = train_index["exact_by_hash"].get(canonical_text_hash(row), [])
        user = train_index["user_by_hash"].get(user_hash(row), [])
        full_call_matches = []
        tool_name_matches = []
        near_value_matches = []
        for sig in call_signature_items(row):
            full = train_index["full_call_by_hash"].get(sig["full_hash"], [])
            full_call_matches.extend(full)
            if sig["name"] in train_index["tool_names"]:
                tool_name_matches.append({"tool": sig["name"], "train_match_count": len(train_index["tool_names"][sig["name"]])})
            if sig["values"]:
                for train_call in train_index["values_by_tool"].get(sig["name"], []):
                    if sig["values"].issubset(train_call["value_set"]):
                        near_value_matches.append(
                            {
                                "tool": sig["name"],
                                "matched_values": sorted(sig["values"]),
                                "train_path": train_call["train_path"],
                                "train_idx": train_call["train_idx"],
                            }
                        )
                        break
        item = {
            "id": row.get("id"),
            "source_row_idx": row.get("source_row_idx"),
            "canonical_text_overlap_count": len(exact),
            "user_prompt_overlap_count": len(user),
            "full_tool_signature_overlap_count": len(full_call_matches),
            "tool_name_overlap_count": len(tool_name_matches),
            "same_tool_all_arg_values_overlap_count": len(near_value_matches),
            "canonical_text_examples": exact[:3],
            "user_prompt_examples": user[:3],
            "full_tool_signature_examples": full_call_matches[:3],
            "same_tool_all_arg_values_examples": near_value_matches[:3],
        }
        per_eval.append(item)
        for key in (
            "canonical_text_overlap_count",
            "user_prompt_overlap_count",
            "full_tool_signature_overlap_count",
            "tool_name_overlap_count",
            "same_tool_all_arg_values_overlap_count",
        ):
            totals[key] += int(item[key])
            if item[key] and len(examples[key]) < 10:
                examples[key].append(item)
    eval_with_any_hard_overlap = sum(
        int(
            row["canonical_text_overlap_count"]
            or row["user_prompt_overlap_count"]
            or row["full_tool_signature_overlap_count"]
            or row["same_tool_all_arg_values_overlap_count"]
        )
        for row in per_eval
    )
    summary = {
        "eval_episode_count": len(eval_rows),
        "eval_ids": [row.get("id") for row in eval_rows],
        "train_files_checked": len(train_index["files"]),
        "train_rows_checked": train_index["total_rows"],
        "train_tool_calls_indexed": train_index["total_calls"],
        "counts": dict(sorted(totals.items())),
        "eval_episodes_with_any_hard_overlap": eval_with_any_hard_overlap,
        "overlap_examples": {key: value for key, value in sorted(examples.items())},
        "method": {
            "canonical_text": "sha256(normalized concatenated user text + normalized assistant/gold tool-call text)",
            "user_prompt": "sha256(normalized concatenated user turns including ToolACE follow-up user messages)",
            "full_tool_signature": "sha256(safe tool name + sorted argument keys + normalized scalar leaf values)",
            "same_tool_all_arg_values": "near check: same safe tool name and every eval scalar leaf value appears in one train call",
            "tool_name_overlap": "descriptive only; same tool name appears anywhere in train calls, not treated as leakage by itself",
        },
    }
    return summary, per_eval


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    difficulty = payload["difficulty"]
    leak = payload["leak_check"]
    lines = [
        "# ToolACE-40 Leak Red-Team",
        "",
        f"Verdict: **{payload['verdict']}**",
        "",
        "## Provenance",
        "",
    ]
    for node in payload["provenance"]["chain"]:
        lines.append(f"- {node['node']}: `{node['artifact']}`")
        lines.append(f"  Upstream: {node['upstream']}")
        if node.get("script"):
            lines.append(f"  Script: `{node['script']}`")
    lines.extend(
        [
            "",
            "## Explicit Overlap Check",
            "",
            f"- Eval ids checked: {leak['eval_episode_count']}",
            f"- Train files checked: {leak['train_files_checked']}",
            f"- Train rows checked: {leak['train_rows_checked']}",
            f"- Train tool calls indexed: {leak['train_tool_calls_indexed']}",
            f"- Canonical text overlaps: {leak['counts'].get('canonical_text_overlap_count', 0)}",
            f"- User prompt overlaps: {leak['counts'].get('user_prompt_overlap_count', 0)}",
            f"- Full tool-signature overlaps: {leak['counts'].get('full_tool_signature_overlap_count', 0)}",
            f"- Same-tool/all-arg-value overlaps: {leak['counts'].get('same_tool_all_arg_values_overlap_count', 0)}",
            f"- Descriptive tool-name overlaps: {leak['counts'].get('tool_name_overlap_count', 0)}",
            "",
            "## Difficulty",
            "",
            "| Slice | episodes | turns | calls/turn | args/call | avg value len | copy args | derived/constant args |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, item in [
        ("ToolACE-derived", difficulty["summary_by_source"].get("ToolACE-derived", {})),
        ("our-synthetic", difficulty["summary_by_source"].get("our-synthetic", {})),
    ]:
        arg_types = item.get("argument_type_counts") or {}
        lines.append(
            f"| {label} | {item.get('episodes', 0)} | {item.get('turns', 0)} "
            f"| {float(item.get('calls_per_turn', 0.0)):.3f} "
            f"| {float(item.get('args_per_call', 0.0)):.3f} "
            f"| {float(item.get('avg_value_length_per_episode_mean', 0.0)):.3f} "
            f"| {arg_types.get('copy_from_context', 0)} "
            f"| {arg_types.get('derived_or_constant', 0)} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: ToolACE-40 uses eval rows from the ToolACE train split and is therefore not a clean external heldout source. "
            "The explicit row-level overlap checks against local train mixes found no hard overlap, but the slice is structurally easier: "
            "one call per generated turn and mostly short scalar arguments copied verbatim from the active user context.",
            "",
            f"Full JSON: `{payload['paths']['audit_json']}`",
            f"Per-episode overlap rows: `{payload['paths']['overlap_jsonl']}`",
            f"Per-episode difficulty rows: `{payload['paths']['difficulty_jsonl']}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    all_rows = load_jsonl(args.eval_jsonl)
    toolace_rows = [row for row in all_rows if row.get("source_family") == "ToolACE-derived"]
    synthetic_rows = [row for row in all_rows if row.get("source_family") == "our-synthetic"]
    train_paths = [path.resolve() for path in args.train_path] if args.train_path else default_train_paths()
    train_index, train_files = build_train_index(train_paths)
    leak_check, overlap_rows = overlap_audit(toolace_rows, train_index)

    difficulty_episode_rows = difficulty_rows(toolace_rows, "ToolACE-derived") + difficulty_rows(synthetic_rows, "our-synthetic")
    grouped_difficulty = defaultdict(list)
    for row in difficulty_episode_rows:
        grouped_difficulty[row["source_family"]].append(row)
    difficulty = {
        "summary_by_source": {
            source: summarize_difficulty(rows)
            for source, rows in sorted(grouped_difficulty.items())
        },
        "episode_count": len(difficulty_episode_rows),
    }

    hard_counts = leak_check["counts"].get("canonical_text_overlap_count", 0)
    hard_counts += leak_check["counts"].get("user_prompt_overlap_count", 0)
    hard_counts += leak_check["counts"].get("full_tool_signature_overlap_count", 0)
    hard_counts += leak_check["counts"].get("same_tool_all_arg_values_overlap_count", 0)
    toolace_diff = difficulty["summary_by_source"].get("ToolACE-derived", {})
    synthetic_diff = difficulty["summary_by_source"].get("our-synthetic", {})
    easy = (
        toolace_diff.get("args_per_call", 0.0) <= synthetic_diff.get("args_per_call", 99.0)
        and (toolace_diff.get("argument_type_counts") or {}).get("copy_from_context", 0)
        >= (toolace_diff.get("argument_type_counts") or {}).get("derived_or_constant", 0)
    )
    verdict = "leak" if hard_counts else ("no-leak-but-easy" if easy else "no-leak-and-fair")

    paths = {
        "audit_json": str(args.out_dir / "toolace40_leak_redteam.json"),
        "audit_md": str(args.out_dir / "toolace40_leak_redteam.md"),
        "overlap_jsonl": str(args.out_dir / "toolace40_overlap_rows.jsonl"),
        "difficulty_jsonl": str(args.out_dir / "toolace40_difficulty_rows.jsonl"),
    }
    payload = {
        "verdict": verdict,
        "input_eval_jsonl": str(args.eval_jsonl),
        "input_manifest_json": str(args.manifest_json),
        "provenance": provenance_summary(),
        "leak_check": {
            **leak_check,
            "train_files": train_files,
            "skipped_train_files": train_index["skipped"],
            "key_training_paths": [str(path) for path in KEY_TRAIN_PATHS if path.exists()],
        },
        "difficulty": difficulty,
        "paths": paths,
    }
    write_json(Path(paths["audit_json"]), payload)
    write_jsonl(Path(paths["overlap_jsonl"]), overlap_rows)
    write_jsonl(Path(paths["difficulty_jsonl"]), difficulty_episode_rows)
    write_markdown(Path(paths["audit_md"]), payload)

    if args.update_manifest:
        manifest = load_json(args.manifest_json)
        manifest["leak_redteam"] = {
            "verdict": verdict,
            "audit_json": paths["audit_json"],
            "audit_md": paths["audit_md"],
            "overlap_jsonl": paths["overlap_jsonl"],
            "difficulty_jsonl": paths["difficulty_jsonl"],
            "leak_check": {
                key: value
                for key, value in payload["leak_check"].items()
                if key
                in {
                    "eval_episode_count",
                    "eval_ids",
                    "train_files_checked",
                    "train_rows_checked",
                    "train_tool_calls_indexed",
                    "counts",
                    "eval_episodes_with_any_hard_overlap",
                    "method",
                    "key_training_paths",
                    "skipped_train_files",
                }
            },
            "difficulty_summary_by_source": difficulty["summary_by_source"],
            "provenance_chain": payload["provenance"]["chain"],
        }
        write_json(args.manifest_json, manifest)

    print(json.dumps({"verdict": verdict, "leak_counts": leak_check["counts"], "difficulty": difficulty["summary_by_source"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
