#!/usr/bin/env python3
"""Build a labeled copy-span eval slice from Run 1 copy-grounding examples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/home/mark/qwen_diffusion")
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from eval_toolcall_jsonl import extract_tool_calls  # noqa: E402


DEFAULT_INPUT = ROOT / "data/flare_redesign_run1_copy_retention_mix/train_agentic_mix.json"
DEFAULT_OUT = ROOT / "runs/flare_redesign_run1_redteam/copyspan_isolation/copyspan_eval_12.jsonl"


def load_instances(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("instances"), list):
        return payload["instances"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported payload shape in {path}")


def tool_names(tools: list[dict[str, Any]]) -> list[str]:
    names = []
    for tool in tools or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if name:
            names.append(str(name))
    return names


def convert_row(row: dict[str, Any], source_index: int, out_index: int) -> dict[str, Any] | None:
    messages = row.get("messages") or []
    assistant = next((item for item in messages if item.get("role") == "assistant"), None)
    if assistant is None:
        return None
    gold_assistant = assistant.get("content") or ""
    calls, invalid = extract_tool_calls(gold_assistant)
    if invalid or not calls:
        return None

    prompt_messages = []
    if row.get("system"):
        prompt_messages.append({"role": "system", "content": row["system"]})
    for item in messages:
        if item.get("role") != "assistant":
            prompt_messages.append({"role": item.get("role", "user"), "content": item.get("content", "")})

    tools = row.get("tools") or []
    return {
        "id": f"run1_copyspan_{out_index:04d}",
        "source": "run1_copy_grounding_labeled_train",
        "source_index": source_index,
        "task": "copy_from_context_tool_argument_values",
        "tools": tools,
        "prompt_messages": prompt_messages,
        "gold_assistant": gold_assistant,
        "gold_tool_names": [call["name"] for call in calls],
        "available_tool_names": tool_names(tools),
        "gold_tool_calls": calls,
        "copy_spans": row.get("copy_spans") or [],
        "isolation_note": (
            "Labeled Run 1 copy-grounding slice; tests C0 verbatim copy mechanics, "
            "not heldout generalization."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--min-copy-spans", type=int, default=1)
    args = parser.parse_args()

    rows = load_instances(args.input)
    converted = []
    skipped = 0
    for source_index, row in enumerate(rows):
        if len(row.get("copy_spans") or []) < args.min_copy_spans:
            continue
        item = convert_row(row, source_index, len(converted))
        if item is None:
            skipped += 1
            continue
        converted.append(item)
        if args.limit and len(converted) >= args.limit:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for item in converted:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "input": str(args.input),
        "out": str(args.out),
        "records": len(converted),
        "skipped_malformed": skipped,
        "copy_spans": sum(len(item.get("copy_spans") or []) for item in converted),
        "source_indices": [item["source_index"] for item in converted],
    }
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
