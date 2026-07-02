#!/usr/bin/env python3
"""Build the left-context-first copy-span probe report."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("/home/mark/qwen_diffusion")
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from analyze_flare_copyspan_outputs import canonical_value, predicted_arg  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("id")): row for row in rows}


def summary_for_output(path: Path) -> dict[str, Any]:
    return json.loads(path.with_suffix(".summary.json").read_text(encoding="utf-8"))


def format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def span_rows(
    cases_by_id: dict[str, dict[str, Any]],
    block_rows: list[dict[str, Any]],
    outputs_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    spans = []
    for block_row in block_rows:
        record_id = str(block_row.get("id"))
        case = cases_by_id[record_id]
        output = outputs_by_id[record_id]
        calls = case.get("gold_tool_calls") or []
        for block in block_row.get("token_blocks") or []:
            if block.get("kind") != "argument_value":
                continue
            call_index = int(block.get("tool_call_index") or 0)
            key = str(block.get("json_key") or "")
            gold_call = calls[call_index]
            function = str(gold_call.get("name") or "")
            arguments = gold_call.get("arguments") or {}
            gold_value = arguments.get(key)
            pred_value = predicted_arg(output, call_index, function, key)
            exact = pred_value is not None and canonical_value(pred_value) == canonical_value(gold_value)
            spans.append(
                {
                    "id": record_id,
                    "tool_call_index": call_index,
                    "function": function,
                    "json_key": key,
                    "target_text": block.get("target_text") or block.get("text") or "",
                    "token_count": int(block.get("token_count") or 0),
                    "exact": bool(exact),
                    "gold": gold_value,
                    "predicted": pred_value,
                }
            )
    return spans


def summarize_run(
    *,
    label: str,
    output_path: Path,
    cases_by_id: dict[str, dict[str, Any]],
    block_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    outputs = read_jsonl(output_path)
    outputs_by_id = by_id(outputs)
    summary = summary_for_output(output_path)
    totals = summary.get("totals") or {}
    spans = span_rows(cases_by_id, block_rows, outputs_by_id)

    counts = Counter()
    for span in spans:
        counts["copy_spans"] += 1
        counts["copy_exact"] += int(span["exact"])
        bucket = "single_token" if span["token_count"] == 1 else "multi_token"
        counts[f"{bucket}_spans"] += 1
        counts[f"{bucket}_exact"] += int(span["exact"])

    value_tokens = int(totals.get("sampler_parallel_commit_value_tokens") or 0)
    value_forwards = int(totals.get("sampler_parallel_commit_value_forward_visits") or 0)
    structural_tokens = int(totals.get("sampler_parallel_commit_structural_tokens") or 0)
    structural_forwards = int(totals.get("sampler_parallel_commit_structural_forward_visits") or 0)
    return {
        "label": label,
        "output_jsonl": str(output_path),
        "summary_json": str(output_path.with_suffix(".summary.json")),
        "parallel_commit_threshold": summary.get("parallel_commit_threshold"),
        "parallel_commit_kinds": summary.get("parallel_commit_kinds"),
        "record_valid_tool_json": int(totals.get("valid_tool_json") or 0),
        "record_exact_arguments": int(totals.get("exact_arguments") or 0),
        "record_count": int(totals.get("records") or len(outputs)),
        "copy_spans": int(counts["copy_spans"]),
        "copy_exact": int(counts["copy_exact"]),
        "copy_accuracy": counts["copy_exact"] / counts["copy_spans"] if counts["copy_spans"] else None,
        "single_token_spans": int(counts["single_token_spans"]),
        "single_token_exact": int(counts["single_token_exact"]),
        "single_token_accuracy": (
            counts["single_token_exact"] / counts["single_token_spans"]
            if counts["single_token_spans"]
            else None
        ),
        "multi_token_spans": int(counts["multi_token_spans"]),
        "multi_token_exact": int(counts["multi_token_exact"]),
        "multi_token_accuracy": (
            counts["multi_token_exact"] / counts["multi_token_spans"] if counts["multi_token_spans"] else None
        ),
        "value_tokens": value_tokens,
        "value_forward_visits": value_forwards,
        "value_tokens_per_forward": value_tokens / value_forwards if value_forwards else None,
        "structural_tokens": structural_tokens,
        "structural_forward_visits": structural_forwards,
        "structural_tokens_per_forward": structural_tokens / structural_forwards if structural_forwards else None,
        "single_token_details": [span for span in spans if span["token_count"] == 1],
        "failures": [span for span in spans if not span["exact"]],
    }


def markdown_report(payload: dict[str, Any]) -> str:
    rows = payload["runs"]
    lines = [
        "# FLARE Left-Context-First Copy Probe",
        "",
        "No new training. Reused the Run-1 copy-grounded checkpoint and the 12-record copy-span slice.",
        "The left-context-first condition decoded scaffold/key/tool-name/tag structure through the normal careful path,",
        "then allowed same-forward parallel commit only for `argument_value` schedule intervals.",
        "",
        "## Required Conditions",
        "",
        "| Condition | Run | Copy exact | Single-token exact | Value TPF | Value tokens/forwards | Valid tool JSON | Record exact |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    required = ["control_careful", "all_parallel_tau050", "leftctx_value_tau050"]
    for key in required:
        row = rows[key]
        lines.append(
            "| {condition} | `{label}` | {copy_exact}/{copy_spans} | {single_exact}/{single_spans} | "
            "{value_tpf} | {value_tokens}/{value_forwards} | {valid}/{records} | {exact_records}/{records} |".format(
                condition=payload["condition_names"][key],
                label=row["label"],
                copy_exact=row["copy_exact"],
                copy_spans=row["copy_spans"],
                single_exact=row["single_token_exact"],
                single_spans=row["single_token_spans"],
                value_tpf=format_float(row["value_tokens_per_forward"], 3),
                value_tokens=row["value_tokens"],
                value_forwards=row["value_forward_visits"],
                valid=row["record_valid_tool_json"],
                exact_records=row["record_exact_arguments"],
                records=row["record_count"],
            )
        )

    lines.extend(
        [
            "",
            "## Left-Context Threshold Check",
            "",
            "| Run | Copy exact | Single-token exact | Value TPF | Value tokens/forwards | Valid tool JSON | Record exact |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key in ["leftctx_value_tau050", "leftctx_value_tau070", "leftctx_value_tau080", "leftctx_value_tau085"]:
        row = rows[key]
        lines.append(
            "| `{label}` | {copy_exact}/{copy_spans} | {single_exact}/{single_spans} | "
            "{value_tpf} | {value_tokens}/{value_forwards} | {valid}/{records} | {exact_records}/{records} |".format(
                label=row["label"],
                copy_exact=row["copy_exact"],
                copy_spans=row["copy_spans"],
                single_exact=row["single_token_exact"],
                single_spans=row["single_token_spans"],
                value_tpf=format_float(row["value_tokens_per_forward"], 3),
                value_tokens=row["value_tokens"],
                value_forwards=row["value_forward_visits"],
                valid=row["record_valid_tool_json"],
                exact_records=row["record_exact_arguments"],
                records=row["record_count"],
            )
        )

    best = rows["leftctx_value_tau080"]
    single = best["single_token_details"][0] if best["single_token_details"] else None
    failure = best["failures"][0] if best["failures"] else None
    lines.extend(
        [
            "",
            "## Single-Token Subset",
            "",
            (
                f"Best speed/quality left-context point (`{best['label']}`): "
                f"{best['single_token_exact']}/{best['single_token_spans']} exact."
            ),
        ]
    )
    if single:
        lines.append(
            f"The single-token span is `{single['id']}` `{single['json_key']}` = `{single['target_text']}` "
            f"({single['token_count']} token), exact={single['exact']}."
        )
    lines.extend(["", "## Verdict", ""])
    lines.append(
        "H2 circuit disruption is not confirmed. Committing left context before value spans rescues copy quality from "
        "0/41 under all-parallel to 29/41 at matched tau 0.50 and to 40/41 at tau 0.80 while value TPF remains >1."
    )
    lines.append(
        "The single-token copy value is exact in the left-context-first condition, so the previous first/single-token "
        "corruption was primarily decode-order dependent, not an unavoidable masked-right-context copy-circuit break."
    )
    lines.append(
        "This is a strong H1 result, but not a full speed gate pass: the best >1 TPF point is still 40/41 rather than "
        "the 41/41 careful baseline."
    )
    if failure:
        lines.append(
            f"The remaining tau 0.80 miss is multi-token: `{failure['id']}` `{failure['json_key']}` target "
            f"`{failure['target_text']}` ({failure['token_count']} tokens)."
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--blocks", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    cases_by_id = by_id(read_jsonl(args.cases))
    block_rows = read_jsonl(args.blocks)
    run_specs = {
        "control_careful": "arg8_tau099",
        "all_parallel_tau050": "arg32_tau050",
        "leftctx_value_tau050": "leftctx_value_arg32_tau050",
        "leftctx_value_tau070": "leftctx_value_arg32_tau070",
        "leftctx_value_tau080": "leftctx_value_arg32_tau080",
        "leftctx_value_tau085": "leftctx_value_arg32_tau085",
    }
    condition_names = {
        "control_careful": "(0) CONTROL careful",
        "all_parallel_tau050": "(1) ALL-PARALLEL",
        "leftctx_value_tau050": "(2) LEFT-CONTEXT-FIRST, values-only parallel",
    }
    runs = {
        key: summarize_run(
            label=stem,
            output_path=args.root / f"{stem}.jsonl",
            cases_by_id=cases_by_id,
            block_rows=block_rows,
        )
        for key, stem in run_specs.items()
    }
    payload = {
        "cases": str(args.cases),
        "blocks": str(args.blocks),
        "condition_names": condition_names,
        "runs": runs,
        "verdict": {
            "h1_decode_order_fixable": True,
            "h2_circuit_disruption_confirmed": False,
            "speed_gate_full_pass": False,
            "summary": (
                "Left context first rescues copy at value_tpf > 1, including the single-token value, "
                "but the best >1 TPF point remains 40/41 instead of the 41/41 careful baseline."
            ),
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(markdown_report(payload), encoding="utf-8")
    print(json.dumps(payload["verdict"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
