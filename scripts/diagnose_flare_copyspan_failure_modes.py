#!/usr/bin/env python3
"""Diagnose whether parallel copy failures are factorization or copy-circuit breaks."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("id")): row for row in rows}


def token_text(tokenizer, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def classify_span(
    *,
    tokenizer,
    block: dict[str, Any],
    parallel_ids: list[int],
    parallel_text: str,
    careful_ids: list[int],
) -> dict[str, Any]:
    start = int(block["token_start"])
    end = int(block["token_end"])
    gold_ids = [int(token_id) for token_id in block.get("token_ids") or []]
    parallel_span = parallel_ids[start:end]
    careful_span = careful_ids[start:end]
    target_text = str(block.get("target_text") or block.get("text") or "")
    token_count = len(gold_ids)

    exact_at_position = parallel_span == gold_ids
    careful_exact_at_position = careful_span == gold_ids
    target_text_anywhere = bool(target_text and target_text in parallel_text)
    first_ok = bool(parallel_span) and bool(gold_ids) and int(parallel_span[0]) == int(gold_ids[0])
    missing_first = bool(gold_ids) and not parallel_span

    if target_text_anywhere:
        failure_mode = "value_preserved_scaffold_failure"
        implication = "neither_value_factorization_nor_circuit"
    elif token_count == 1:
        failure_mode = "single_token_corrupt"
        implication = "B_circuit_disruption"
    elif missing_first or not first_ok:
        failure_mode = "multi_first_token_corrupt"
        implication = "B_circuit_disruption"
    else:
        failure_mode = "multi_first_ok_later_corrupt"
        implication = "A_factorization"

    return {
        "json_key": block.get("json_key"),
        "argument_path": block.get("argument_path"),
        "target_text": target_text,
        "token_count": token_count,
        "single_token": token_count == 1,
        "exact_at_position": exact_at_position,
        "careful_exact_at_position": careful_exact_at_position,
        "target_text_anywhere": target_text_anywhere,
        "first_token_ok": first_ok,
        "failure_mode": failure_mode,
        "implication": implication,
        "gold_token_ids": gold_ids,
        "parallel_token_ids": parallel_span,
        "careful_token_ids": careful_span,
        "gold_text": token_text(tokenizer, gold_ids),
        "parallel_text_at_position": token_text(tokenizer, parallel_span),
        "careful_text_at_position": token_text(tokenizer, careful_span),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-path", type=Path, default=Path("models/qwen3.5-9b-fastdllm-init"))
    parser.add_argument(
        "--blocks",
        type=Path,
        default=Path("runs/flare_redesign_run1_redteam/copyspan_isolation/copyspan_blocks_12.jsonl"),
    )
    parser.add_argument(
        "--parallel",
        type=Path,
        default=Path("runs/flare_redesign_run1_redteam/copyspan_isolation/arg32_tau050.jsonl"),
    )
    parser.add_argument(
        "--careful",
        type=Path,
        default=Path("runs/flare_redesign_run1_redteam/copyspan_isolation/arg8_tau099.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/flare_redesign_run1_redteam/copyspan_isolation/failure_mode_diagnostic.json"),
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    block_rows = read_jsonl(args.blocks)
    parallel_rows = by_id(read_jsonl(args.parallel))
    careful_rows = by_id(read_jsonl(args.careful))

    records = []
    totals: Counter[str] = Counter()
    for row in block_rows:
        record_id = str(row.get("id"))
        parallel = parallel_rows[record_id]
        careful = careful_rows[record_id]
        parallel_text = parallel.get("assistant") or ""
        careful_text = careful.get("assistant") or ""
        parallel_ids = tokenizer(parallel_text, add_special_tokens=False).input_ids
        careful_ids = tokenizer(careful_text, add_special_tokens=False).input_ids
        record = {
            "id": record_id,
            "parallel_generated_token_count": parallel.get("generated_token_count"),
            "parallel_reencoded_token_count": len(parallel_ids),
            "careful_generated_token_count": careful.get("generated_token_count"),
            "careful_reencoded_token_count": len(careful_ids),
            "parallel_valid_tool_json": bool(parallel.get("valid_tool_json")),
            "careful_valid_tool_json": bool(careful.get("valid_tool_json")),
            "spans": [],
        }
        for block in row.get("token_blocks") or []:
            if block.get("kind") != "argument_value":
                continue
            span = classify_span(
                tokenizer=tokenizer,
                block=block,
                parallel_ids=parallel_ids,
                parallel_text=parallel_text,
                careful_ids=careful_ids,
            )
            record["spans"].append(span)
            totals["spans"] += 1
            totals[f"mode:{span['failure_mode']}"] += 1
            totals[f"implication:{span['implication']}"] += 1
            totals["single_token" if span["single_token"] else "multi_token"] += 1
            totals["target_text_anywhere" if span["target_text_anywhere"] else "target_text_absent"] += 1
            totals["careful_exact_at_position" if span["careful_exact_at_position"] else "careful_not_exact_at_position"] += 1
        records.append(record)

    value_failure_count = (
        totals["mode:single_token_corrupt"]
        + totals["mode:multi_first_token_corrupt"]
        + totals["mode:multi_first_ok_later_corrupt"]
    )
    circuit_count = totals["implication:B_circuit_disruption"]
    factor_count = totals["implication:A_factorization"]
    summary = {
        "parallel": str(args.parallel),
        "careful": str(args.careful),
        "blocks": str(args.blocks),
        "totals": dict(totals),
        "fractions_all_parse_failed_spans": {
            "B_circuit_disruption": circuit_count / totals["spans"] if totals["spans"] else None,
            "A_factorization": factor_count / totals["spans"] if totals["spans"] else None,
            "value_preserved_scaffold_failure": (
                totals["mode:value_preserved_scaffold_failure"] / totals["spans"] if totals["spans"] else None
            ),
        },
        "fractions_value_token_failures_only": {
            "B_circuit_disruption": circuit_count / value_failure_count if value_failure_count else None,
            "A_factorization": factor_count / value_failure_count if value_failure_count else None,
            "value_failure_count": value_failure_count,
        },
    }
    payload = {"summary": summary, "records": records}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
