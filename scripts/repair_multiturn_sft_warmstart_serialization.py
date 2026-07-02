#!/usr/bin/env python3
"""Repair multi-turn SFT warm-start serialization.

The original warm-start corpus preserved prior assistant turns as assistant
messages, so the trainer labeled them in addition to the accepted current turn.
This script keeps the same audited examples but rewrites self-generated rows as
a compact single user -> assistant target. Retention rows are copied unchanged.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_multiturn_sft_warmstart import (  # noqa: E402
    compact_current_turn_prompt,
    compact_prior_assistant_calls,
    write_conversation_json,
)
from build_qwen35_planner_selector_retention_mix import resolve_chat_template, token_stats  # noqa: E402


DEFAULT_INPUT = ROOT / "data/multiturn_sft_warmstart/lmflow_dataset/train_agentic_mix.json"
DEFAULT_OUT_DIR = ROOT / "data/multiturn_sft_warmstart_repaired"
DEFAULT_MODEL = ROOT / "models/qwen3.5-9b-fastdllm-init"


def read_instances(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") != "conversation" or not isinstance(payload.get("instances"), list):
        raise ValueError(f"{path} is not a conversation dataset")
    return payload["instances"]


def assistant_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [message for message in messages if isinstance(message, dict) and message.get("role") == "assistant"]


def repair_instance(instance: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    source = instance.get("source")
    messages = [copy.deepcopy(item) for item in instance.get("messages") or [] if isinstance(item, dict)]
    assistants = assistant_messages(messages)
    if source != "selfgen_multiturn_exact_audited" or not assistants:
        return copy.deepcopy(instance), {"repaired": False, "reason": "retention_or_no_assistant"}

    final_assistant = str(assistants[-1].get("content") or "").strip()
    final_index = max(idx for idx, message in enumerate(messages) if message.get("role") == "assistant")
    history_messages = [
        {"role": str(message.get("role")), "content": str(message.get("content") or "")}
        for message in messages[:final_index]
        if message.get("role") in {"user", "assistant", "tool"}
    ]
    prior_label_spans = len(compact_prior_assistant_calls(history_messages))
    repaired = copy.deepcopy(instance)
    repaired["system"] = (
        "You are a tool-call formatter. Return exactly the requested Qwen-native "
        "<tool_call> block(s) using <function=...> and <parameter=...> tags with no prose."
    )
    repaired["messages"] = [
        {"role": "user", "content": compact_current_turn_prompt(history_messages, final_assistant)},
        {"role": "assistant", "content": final_assistant},
    ]
    repaired["tools"] = []
    repaired["sft_metadata"] = {
        **(instance.get("sft_metadata") or {}),
        "serialization": "single_target_compact_context",
        "prior_assistant_label_spans_removed": prior_label_spans,
        "original_message_count": len(messages),
        "original_assistant_message_count": len(assistants),
    }
    return repaired, {
        "repaired": True,
        "prior_assistant_label_spans_removed": prior_label_spans,
        "original_message_count": len(messages),
        "original_assistant_message_count": len(assistants),
    }


def summarize_token_stats(instances: list[dict[str, Any]], tokenizer, chat_template, block_size: int) -> dict[str, Any]:
    rows = [token_stats(tokenizer, chat_template, item, block_size, "left") for item in instances]
    out = {"rows": len(rows)}
    for key in ("length", "full_labels", "kept_labels"):
        values = sorted(int(row[key]) for row in rows)
        out[key] = {
            "min": values[0],
            "p50": values[len(values) // 2],
            "p90": values[int((len(values) - 1) * 0.9)],
            "max": values[-1],
        }
    out["over_block_size"] = sum(int(row["length"] > block_size) for row in rows)
    out["full_labels_kept"] = sum(int(bool(row["full_labels_kept"])) for row in rows)
    out["partial_after_truncation"] = sum(int(bool(row["partial_after_truncation"])) for row in rows)
    out["zero_after_truncation"] = sum(int(bool(row["zero_after_truncation"])) for row in rows)
    return out


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Repaired Multi-Turn SFT Warm-Start",
        "",
        "Self-generated rows are rewritten as one compact user prompt plus one assistant target.",
        "Prior assistant calls are retained only as user-visible context text, so the trainer labels only the final audited assistant target.",
        "",
        "## Counts",
        "",
        f"- Final rows: `{manifest['final_count']}`",
        f"- Repaired self-generated rows: `{manifest['repair_counts'].get('repaired', 0)}`",
        f"- Retention rows copied unchanged: `{manifest['repair_counts'].get('retention_or_no_assistant', 0)}`",
        f"- Source counts: `{manifest['source_counts']}`",
        "",
        "## Token Audit",
        "",
        f"- Original over 512 tokens: `{manifest['original_token_stats']['over_block_size']}`",
        f"- Repaired over 512 tokens: `{manifest['repaired_token_stats']['over_block_size']}`",
        f"- Original partial labels at 512-left: `{manifest['original_token_stats']['partial_after_truncation']}`",
        f"- Repaired partial labels at 512-left: `{manifest['repaired_token_stats']['partial_after_truncation']}`",
        f"- Repaired rows with full labels kept at 512-left: `{manifest['repaired_token_stats']['full_labels_kept']}/{manifest['final_count']}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--conversation-template", default="fast_dllm_v2_native")
    parser.add_argument("--block-size", type=int, default=512)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    instances = read_instances(args.input)
    repaired = []
    repair_counts = Counter()
    repair_details = []
    for idx, instance in enumerate(instances):
        item, detail = repair_instance(instance)
        repaired.append(item)
        repair_counts["repaired" if detail.get("repaired") else detail.get("reason", "unchanged")] += 1
        repair_details.append({"index": idx, "source": instance.get("source"), **detail})

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    chat_template = resolve_chat_template(args.conversation_template)
    original_stats = summarize_token_stats(instances, tokenizer, chat_template, args.block_size)
    repaired_stats = summarize_token_stats(repaired, tokenizer, chat_template, args.block_size)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train_agentic_mix.json"
    lmflow_path = args.out_dir / "lmflow_dataset" / "train_agentic_mix.json"
    write_conversation_json(train_path, repaired)
    write_conversation_json(lmflow_path, repaired)

    manifest = {
        "created_by": "scripts/repair_multiturn_sft_warmstart_serialization.py",
        "input": str(args.input),
        "train_path": str(train_path),
        "lmflow_dataset_dir": str(lmflow_path.parent),
        "lmflow_train_path": str(lmflow_path),
        "final_count": len(repaired),
        "source_counts": dict(sorted(Counter(item.get("source") or "UNKNOWN" for item in repaired).items())),
        "repair_counts": dict(sorted(repair_counts.items())),
        "original_token_stats": original_stats,
        "repaired_token_stats": repaired_stats,
        "block_size": args.block_size,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out_dir / "repair_details.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in repair_details) + "\n",
        encoding="utf-8",
    )
    write_report(args.out_dir / "report.md", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
