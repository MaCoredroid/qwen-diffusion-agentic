#!/usr/bin/env python3
"""Build the leak-checked public episode pool for RL-pilot-v2."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_flare_broaden_public_eval import (  # noqa: E402
    load_chat_template,
    max_gold_path_tokens,
    row_user_text,
    sha256_json,
    split_tool_call_blocks,
    toolace_row_to_episode,
    user_fingerprint,
)
from rl_multiturn_tool_env import episode_fingerprint, read_jsonl, write_json, write_jsonl  # noqa: E402


DEFAULT_LOCAL_NATIVE = ROOT / "data/multiturn_sft_warmstart/public_train_multicall_native.jsonl"
DEFAULT_OUT = ROOT / "data/rl_multiturn_v2_public_pool/episodes.jsonl"
DEFAULT_MANIFEST = ROOT / "data/rl_multiturn_v2_public_pool/manifest.json"
DEFAULT_REJECTED = ROOT / "data/rl_multiturn_v2_public_pool/rejected.jsonl"
DEFAULT_MODEL = ROOT / "models/qwen3.5-9b-fastdllm-b1000-vllm-bf16"
DEFAULT_CHAT_TEMPLATE = Path("/home/mark/shared/lumoFlyWheel_codex_fork/docker/chat_templates/qwen3-openai-codex.jinja")
DEFAULT_MATCHED20_MANIFEST = ROOT / "runs/agentic_eval/northstar_matched_ar_vllm_vs_diffusion/fairness_manifest.json"
DEFAULT_EVAL_BATTERY = [
    ROOT / "data/toolcall_eval_native/flare_scaleup_native_58.jsonl",
    ROOT / "data/toolcall_eval_native/flare_nevertrain_bfcl_apibank.jsonl",
    ROOT / "data/toolcall_eval_native/flare_broaden_public_toolace60.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-native-jsonl", type=Path, default=DEFAULT_LOCAL_NATIVE)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--rejected-jsonl", type=Path, default=DEFAULT_REJECTED)
    parser.add_argument("--target-count", type=int, default=240)
    parser.add_argument("--min-count", type=int, default=150)
    parser.add_argument("--local-first-count", type=int, default=120)
    parser.add_argument("--min-turns", type=int, default=2)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--max-tools", type=int, default=12)
    parser.add_argument("--toolace-dataset", default="Team-ACE/ToolACE")
    parser.add_argument("--toolace-split", default="train")
    parser.add_argument("--max-toolace-stream-rows", type=int, default=50000)
    parser.add_argument("--prompt-tokenizer-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--chat-template-path", type=Path, default=DEFAULT_CHAT_TEMPLATE)
    parser.add_argument("--max-prompt-tokens", type=int, default=3500)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--skip-token-filter", action="store_true")
    parser.add_argument("--matched20-manifest", type=Path, default=DEFAULT_MATCHED20_MANIFEST)
    parser.add_argument(
        "--eval-battery-path",
        dest="eval_battery_paths",
        action="append",
        type=Path,
        default=list(DEFAULT_EVAL_BATTERY),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def public_eval_hash(row: dict[str, Any]) -> str:
    value = row.get("public_eval_hash")
    if value:
        return str(value)
    return sha256_json(
        {
            "id": row.get("id"),
            "prompt_messages": row.get("prompt_messages") or row.get("messages") or [],
            "turn_user_messages": row.get("turn_user_messages") or [],
            "tools": row.get("tools") or [],
            "gold_assistant": row.get("gold_assistant") or "",
        }
    )


def eval_reference(eval_paths: list[Path], matched20_manifest: Path) -> dict[str, Any]:
    ids: set[str] = set()
    fingerprints: set[str] = set()
    public_hashes: set[str] = set()
    user_hashes: set[str] = set()
    path_counts: dict[str, int] = {}
    matched20_ids: set[str] = set()
    if matched20_manifest.exists():
        manifest = load_json(matched20_manifest)
        for episode in manifest.get("episodes") or []:
            if episode.get("id"):
                matched20_ids.add(str(episode["id"]))
    else:
        manifest = {}
    for path in eval_paths:
        if not path.exists():
            path_counts[str(path)] = 0
            continue
        rows = read_jsonl(path)
        path_counts[str(path)] = len(rows)
        for row in rows:
            if row.get("id"):
                ids.add(str(row["id"]))
            fingerprints.add(episode_fingerprint(row))
            public_hashes.add(public_eval_hash(row))
            user_hashes.add(user_fingerprint(row_user_text(row)))
    ids.update(matched20_ids)
    return {
        "ids": ids,
        "fingerprints": fingerprints,
        "public_hashes": public_hashes,
        "user_hashes": user_hashes,
        "path_counts": path_counts,
        "matched20_manifest": str(matched20_manifest),
        "matched20_manifest_exists": matched20_manifest.exists(),
        "matched20_episode_ids": sorted(matched20_ids),
        "matched20_episode_count": len(matched20_ids),
        "matched20_input_jsonl": manifest.get("input_jsonl") if matched20_manifest.exists() else None,
    }


def token_stats(row: dict[str, Any], tokenizer, chat_template: str | None) -> dict[str, int]:
    max_prompt, max_gold = max_gold_path_tokens(tokenizer, chat_template, row)
    return {"max_gold_path_prompt_tokens": int(max_prompt), "max_gold_block_tokens": int(max_gold)}


def normalize_candidate(row: dict[str, Any], source_family: str, source_dataset: str, source_license: str) -> dict[str, Any]:
    item = copy.deepcopy(row)
    if not item.get("source_family"):
        item["source_family"] = source_family
    if not item.get("source_dataset"):
        item["source_dataset"] = source_dataset
    if not item.get("source_license"):
        item["source_license"] = source_license
    item.setdefault("turn_user_messages", [None for _ in split_tool_call_blocks(item.get("gold_assistant") or "")])
    item["public_eval_hash"] = public_eval_hash(item)
    return item


def reject_payload(row: dict[str, Any], source: str, reasons: list[str], token_info: dict[str, int] | None = None) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "source": source,
        "source_family": row.get("source_family") or row.get("source"),
        "turns": len(split_tool_call_blocks(row.get("gold_assistant") or "")),
        "tool_count": len(row.get("tools") or []),
        "reasons": reasons,
        "token_stats": token_info or {},
    }


def check_candidate(
    row: dict[str, Any],
    eval_ref: dict[str, Any],
    seen: dict[str, set[str]],
    args: argparse.Namespace,
    tokenizer,
    chat_template: str | None,
) -> tuple[bool, list[str], dict[str, int]]:
    reasons: list[str] = []
    blocks = split_tool_call_blocks(row.get("gold_assistant") or "")
    if len(blocks) < int(args.min_turns) or len(blocks) > int(args.max_turns):
        reasons.append("turn_count_out_of_range")
    if len(row.get("tools") or []) > int(args.max_tools):
        reasons.append("too_many_tools")
    row_id = str(row.get("id") or "")
    row_fp = episode_fingerprint(row)
    row_hash = public_eval_hash(row)
    user_hash = user_fingerprint(row_user_text(row))
    if row_id and row_id in eval_ref["ids"]:
        reasons.append("eval_id_overlap")
    if row_fp in eval_ref["fingerprints"]:
        reasons.append("eval_fingerprint_overlap")
    if row_hash in eval_ref["public_hashes"]:
        reasons.append("eval_public_hash_overlap")
    if user_hash in eval_ref["user_hashes"]:
        reasons.append("eval_user_overlap")
    if row_fp in seen["fingerprints"]:
        reasons.append("pool_fingerprint_duplicate")
    if user_hash in seen["user_hashes"]:
        reasons.append("pool_user_duplicate")
    info: dict[str, int] = {}
    if not args.skip_token_filter and not reasons:
        info = token_stats(row, tokenizer, chat_template)
        if info["max_gold_path_prompt_tokens"] > int(args.max_prompt_tokens) or info["max_gold_block_tokens"] > int(args.max_new_tokens):
            reasons.append("prompt_or_gold_too_long")
    return not reasons, reasons, info


def mark_seen(row: dict[str, Any], seen: dict[str, set[str]]) -> None:
    seen["fingerprints"].add(episode_fingerprint(row))
    seen["user_hashes"].add(user_fingerprint(row_user_text(row)))


def select_local_rows(
    args: argparse.Namespace,
    eval_ref: dict[str, Any],
    seen: dict[str, set[str]],
    tokenizer,
    chat_template: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    counts: Counter = Counter()
    for row in read_jsonl(args.local_native_jsonl):
        item = normalize_candidate(
            row,
            "public-native-warmstart",
            str(args.local_native_jsonl),
            "public-derived-training-pool",
        )
        ok, reasons, info = check_candidate(item, eval_ref, seen, args, tokenizer, chat_template)
        if not ok:
            counts.update(reasons)
            if len(rejected) < 500:
                rejected.append(reject_payload(item, "local_native", reasons, info))
            continue
        item["rl_v2_leak_check"] = {"eval_overlap": False, "pool_duplicate": False, "token_filter": info}
        mark_seen(item, seen)
        selected.append(item)
    return selected, rejected, counts


def select_toolace_rows(
    args: argparse.Namespace,
    eval_ref: dict[str, Any],
    seen: dict[str, set[str]],
    tokenizer,
    chat_template: str | None,
    current_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter, dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    counts: Counter = Counter()
    status = {"attempted": True, "dataset": args.toolace_dataset, "split": args.toolace_split, "error": None}
    try:
        stream = load_dataset(args.toolace_dataset, split=args.toolace_split, streaming=True)
        for row_idx, raw in enumerate(stream):
            if row_idx >= int(args.max_toolace_stream_rows):
                break
            candidate, reason = toolace_row_to_episode(raw, row_idx)
            if candidate is None:
                counts[reason] += 1
                continue
            item = normalize_candidate(candidate, "ToolACE-derived", args.toolace_dataset, "Apache-2.0")
            ok, reasons, info = check_candidate(item, eval_ref, seen, args, tokenizer, chat_template)
            if not ok:
                counts.update(reasons)
                if len(rejected) < 500:
                    rejected.append(reject_payload(item, "toolace_train", reasons, info))
                continue
            item["rl_v2_leak_check"] = {"eval_overlap": False, "pool_duplicate": False, "token_filter": info}
            mark_seen(item, seen)
            selected.append(item)
            if current_count + len(selected) >= int(args.target_count):
                break
    except Exception as exc:  # dataset access can be gated or network-limited
        status["error"] = f"{type(exc).__name__}: {exc}"
    status["selected"] = len(selected)
    return selected, rejected, counts, status


def selected_overlap_counts(rows: list[dict[str, Any]], eval_ref: dict[str, Any]) -> dict[str, int]:
    return {
        "id": sum(int(str(row.get("id") or "") in eval_ref["ids"]) for row in rows),
        "fingerprint": sum(int(episode_fingerprint(row) in eval_ref["fingerprints"]) for row in rows),
        "public_eval_hash": sum(int(public_eval_hash(row) in eval_ref["public_hashes"]) for row in rows),
        "user_text": sum(int(user_fingerprint(row_user_text(row)) in eval_ref["user_hashes"]) for row in rows),
    }


def manifest_for(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    reject_counts: Counter,
    eval_ref: dict[str, Any],
    toolace_status: dict[str, Any],
) -> dict[str, Any]:
    source_counts = Counter(row.get("source_family") or row.get("source") or "unknown" for row in rows)
    turn_counts = Counter(str(len(split_tool_call_blocks(row.get("gold_assistant") or ""))) for row in rows)
    row_manifest = [
        {
            "id": row.get("id"),
            "source_family": row.get("source_family"),
            "source_dataset": row.get("source_dataset"),
            "source_license": row.get("source_license"),
            "source_row_idx": row.get("source_row_idx"),
            "turns": len(split_tool_call_blocks(row.get("gold_assistant") or "")),
            "public_eval_hash": public_eval_hash(row),
            "episode_fingerprint": episode_fingerprint(row),
            "user_fingerprint": user_fingerprint(row_user_text(row)),
        }
        for row in rows
    ]
    return {
        "out_jsonl": str(args.out_jsonl),
        "records": len(rows),
        "turns": sum(item["turns"] for item in row_manifest),
        "target_count": int(args.target_count),
        "min_count": int(args.min_count),
        "episode_set_hash": sha256_json(row_manifest),
        "source_family_counts": dict(sorted(source_counts.items())),
        "turn_count_histogram": dict(sorted(turn_counts.items())),
        "rows": row_manifest,
        "selection": {
            "min_turns": int(args.min_turns),
            "max_turns": int(args.max_turns),
            "max_tools": int(args.max_tools),
            "local_first_count": int(args.local_first_count),
            "max_prompt_tokens": int(args.max_prompt_tokens),
            "max_new_tokens": int(args.max_new_tokens),
            "skip_token_filter": bool(args.skip_token_filter),
        },
        "sources": {
            "local_native": {
                "path": str(args.local_native_jsonl),
                "license": "public-derived-training-pool",
            },
            "ToolACE": toolace_status | {
                "license": "Apache-2.0",
                "format_transcode": "Pythonic bracket calls -> Qwen-native function/parameter XML",
            },
        },
        "frozen_eval_battery": {
            "paths": [str(path) for path in args.eval_battery_paths],
            "path_counts": eval_ref["path_counts"],
            "matched20_manifest": eval_ref["matched20_manifest"],
            "matched20_manifest_exists": eval_ref["matched20_manifest_exists"],
            "matched20_input_jsonl": eval_ref["matched20_input_jsonl"],
            "matched20_episode_count": eval_ref["matched20_episode_count"],
            "matched20_episode_ids": eval_ref["matched20_episode_ids"],
            "selected_overlap_counts": selected_overlap_counts(rows, eval_ref),
        },
        "rejected_count_recorded": len(rejected),
        "rejected_reason_counts": dict(sorted(reject_counts.items())),
        "rejected_jsonl": str(args.rejected_jsonl),
    }


def main() -> int:
    args = parse_args()
    eval_ref = eval_reference(args.eval_battery_paths, args.matched20_manifest)
    tokenizer = None
    chat_template = None
    if not args.skip_token_filter:
        tokenizer = AutoTokenizer.from_pretrained(str(args.prompt_tokenizer_path), trust_remote_code=True)
        chat_template = load_chat_template(args.chat_template_path)
    seen = {"fingerprints": set(), "user_hashes": set()}
    local_rows, rejected, reject_counts = select_local_rows(args, eval_ref, seen, tokenizer, chat_template)
    selected = local_rows[: int(args.local_first_count)]
    for row in selected:
        mark_seen(row, seen)
    toolace_rows, toolace_rejected, toolace_counts, toolace_status = select_toolace_rows(
        args,
        eval_ref,
        seen,
        tokenizer,
        chat_template,
        len(selected),
    )
    selected.extend(toolace_rows)
    if len(selected) < int(args.target_count):
        for row in local_rows[int(args.local_first_count) :]:
            if len(selected) >= int(args.target_count):
                break
            selected.append(row)
    rejected.extend(toolace_rejected)
    reject_counts.update(toolace_counts)
    if len(selected) < int(args.min_count):
        write_jsonl(args.rejected_jsonl, rejected)
        raise SystemExit(f"selected {len(selected)} episodes; need >= {args.min_count}")
    selected = selected[: int(args.target_count)]
    overlaps = selected_overlap_counts(selected, eval_ref)
    if any(overlaps.values()):
        write_jsonl(args.rejected_jsonl, rejected)
        raise SystemExit(f"selected rows overlap frozen eval battery: {overlaps}")
    write_jsonl(args.out_jsonl, selected)
    write_jsonl(args.rejected_jsonl, rejected)
    manifest = manifest_for(args, selected, rejected, reject_counts, eval_ref, toolace_status)
    write_json(args.manifest_json, manifest)
    print(
        json.dumps(
            {
                "records": manifest["records"],
                "turns": manifest["turns"],
                "source_family_counts": manifest["source_family_counts"],
                "selected_overlap_counts": manifest["frozen_eval_battery"]["selected_overlap_counts"],
                "toolace_error": toolace_status.get("error"),
                "episode_set_hash": manifest["episode_set_hash"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
