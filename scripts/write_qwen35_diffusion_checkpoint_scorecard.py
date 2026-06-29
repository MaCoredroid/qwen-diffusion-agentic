#!/usr/bin/env python3
import argparse
import json
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SUMMARIES = [
    (
        "public one-call, max-1",
        ROOT
        / "runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_eval96_modelrepair_max1/public_onecall_8_complex_projection_v3.summary.json",
    ),
    (
        "Qwen3.6 teacher train one-call, max-1",
        ROOT
        / "runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_eval96_modelrepair_max1/teacher_train_labelaware_12_complex_projection_v3.summary.json",
    ),
    (
        "Qwen3.6 teacher heldout one-call, max-1",
        ROOT
        / "runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_eval96_modelrepair_max1/teacher_heldout_labelaware_8_complex_projection_v3.summary.json",
    ),
    (
        "public multi-call, sequence-preserving complex projection",
        ROOT
        / "runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_multicall_eval384_modelrepair/public_multicall_12_sequence_preserve_complex_v4.summary.json",
    ),
    (
        "public multi-call, complex + contextual projection",
        ROOT
        / "runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_multicall_eval384_modelrepair/public_multicall_12_sequence_preserve_complex_contextual_v4.summary.json",
    ),
    (
        "public multi-call, sequence-planner projection",
        ROOT
        / "runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_multicall_eval384_modelrepair/public_multicall_12_sequence_planner_segmentargs_v3.summary.json",
    ),
    (
        "public multi-call, scalar repair + contextual projection",
        ROOT
        / "runs/fastdllm_qwen35_9b_toolcall_multicall_scalar_argspanw1p5_b896_step300_eval/public_multicall_12_ckpt300_contextual_projection_v4.summary.json",
    ),
    (
        "synthetic tool-result, max-1",
        ROOT
        / "runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_toolresult_eval160_modelrepair_max1/synthetic_toolresult_10_complex_projection_v3.summary.json",
    ),
    (
        "OpenAI-style tool-result, max-1",
        ROOT
        / "runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_openai_toolresult_eval160_modelrepair_max1/synthetic_openai_toolresult_10_complex_projection_v3.summary.json",
    ),
]


def parse_labeled_path(value):
    if ":" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL:PATH")
    label, path = value.split(":", 1)
    if not label.strip():
        raise argparse.ArgumentTypeError("Summary label cannot be empty")
    return label.strip(), Path(path)


def load_summary(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def count(value, total):
    if value is None:
        return "n/a"
    return f"{value}/{total}"


def maybe_model_repair_count(summary, key, total):
    if not summary.get("model_repair_pass"):
        return "n/a"
    return count(summary.get("totals", {}).get(key), total)


def fmt_float(value):
    if value is None or value == "":
        return "n/a"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "n/a"


def is_projection_summary(summary):
    totals = summary.get("totals", {})
    return isinstance(totals.get("projected"), dict) or isinstance(totals.get("planned"), dict)


def summary_row(label, summary):
    totals = summary.get("totals", {})
    records = totals.get("records") or 0
    if is_projection_summary(summary):
        projected = totals.get("projected") or totals.get("planned") or {}
        return {
            "label": label,
            "records": records,
            "raw_valid": count(projected.get("valid_tool_json"), records),
            "raw_seq": count(projected.get("exact_tool_sequence"), records),
            "raw_args": count(projected.get("exact_arguments"), records),
            "constrained_seq": "n/a",
            "constrained_args": "n/a",
            "model_repair_seq": "n/a",
            "model_repair_args": "n/a",
            "extra_missing_repeated": (
                f"{projected.get('records_with_extra_calls', 0)} / "
                f"{projected.get('records_with_missing_calls', 0)} / "
                f"{projected.get('records_with_repeated_calls', 0)}"
            ),
            "unresolved_masks": "n/a",
            "tokens_per_second": "n/a",
            "out_jsonl": summary.get("out_jsonl") or summary.get("input_jsonl"),
            "projection": True,
            "replacement_counts": summary.get("replacement_counts") or {},
        }
    return {
        "label": label,
        "records": records,
        "raw_valid": count(totals.get("valid_tool_json"), records),
        "raw_seq": count(totals.get("exact_tool_sequence"), records),
        "raw_args": count(totals.get("exact_arguments"), records),
        "constrained_seq": count(totals.get("constrained_exact_tool_sequence"), records),
        "constrained_args": count(totals.get("constrained_exact_arguments"), records),
        "model_repair_seq": maybe_model_repair_count(summary, "model_repair_exact_tool_sequence", records),
        "model_repair_args": maybe_model_repair_count(summary, "model_repair_exact_arguments", records),
        "extra_missing_repeated": (
            f"{totals.get('records_with_extra_calls', 0)} / "
            f"{totals.get('records_with_missing_calls', 0)} / "
            f"{totals.get('records_with_repeated_calls', 0)}"
        ),
        "unresolved_masks": totals.get("unresolved_mask_examples", 0),
        "tokens_per_second": fmt_float(summary.get("generated_tokens_per_second")),
        "out_jsonl": summary.get("out_jsonl") or summary.get("input_jsonl"),
        "projection": False,
        "replacement_counts": {},
    }


def write_markdown(args, rows, loaded):
    best_public_onecall = next((row for row in rows if row["label"].startswith("public one-call")), None)
    public_multicall = next((row for row in rows if "sequence-preserving" in row["label"]), None)
    public_multicall_contextual = next((row for row in rows if "contextual projection" in row["label"]), None)
    public_multicall_sequence_planner = next(
        (row for row in rows if row["label"] == "public multi-call, sequence-planner projection"),
        None,
    )
    public_multicall_scalar_contextual = next(
        (row for row in rows if row["label"] == "public multi-call, scalar repair + contextual projection"),
        None,
    )
    tool_result = [row for row in rows if "tool-result" in row["label"]]

    lines = [
        f"# {args.title}",
        "",
        f"Date: {args.date}",
        "",
        "## Status",
        "",
        "This is the current promoted Qwen3.5-9B Fast-DLLM diffusion/QLoRA checkpoint scorecard.",
        "It consolidates the one-call, multi-call, and tool-result gates used by the roadmap.",
        "",
        "This checkpoint is not an agentic closeout model yet. It is the active 9B diffusion",
        "comparison point for the next data/training iteration.",
        "",
        "## Checkpoint",
        "",
        "```text",
        f"adapter: {args.adapter}",
        f"tokenizer: {args.tokenizer}",
        f"base model: {args.base_model}",
        "sampler: Fast-DLLM full-context sampling",
        "projection: deterministic constrained scalar/complex tool-call projection",
        "```",
        "",
        "## Results",
        "",
        "| Slice | Valid JSON | Raw seq | Raw args | Constrained seq | Constrained args | Model-repair seq | Model-repair args | Extra / missing / repeated | Tokens/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {label} | {raw_valid} | {raw_seq} | {raw_args} | {constrained_seq} | "
            "{constrained_args} | {model_repair_seq} | {model_repair_args} | "
            "{extra_missing_repeated} | {tokens_per_second} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Gate Readout",
            "",
        ]
    )
    if best_public_onecall:
        lines.append(
            f"- Public one-call has nonzero strict signal: raw `{best_public_onecall['raw_seq']}` sequence and "
            f"`{best_public_onecall['raw_args']}` arguments; constrained max-1 reaches "
            f"`{best_public_onecall['constrained_seq']}` / `{best_public_onecall['constrained_args']}`."
        )
    if public_multicall:
        lines.append(
            f"- Public multi-call remains the main gap: sequence-preserving complex constrained projection reaches "
            f"`{public_multicall['constrained_seq']}` sequence and `{public_multicall['constrained_args']}` arguments."
        )
    if public_multicall_contextual:
        scalar_clause = ""
        if public_multicall_scalar_contextual:
            scalar_clause = "; scalar repair plus contextual projection ties that score but is slower"
        lines.append(
            f"- The best postprocessed public multi-call path now reaches `{public_multicall_contextual['raw_seq']}` "
            f"sequence and `{public_multicall_contextual['raw_args']}` arguments with direct constrained complex/contextual "
            f"projection{scalar_clause}. This is a deterministic projection prototype, not a model-only metric."
        )
        lines.append(
            "- Cross-slice complex/contextual projection is neutral-to-positive on one-call and tool-result slices; "
            "see `qwen35_9b_contextual_projection_suite_result.md`."
        )
    if public_multicall_sequence_planner:
        lines.append(
            f"- A guarded request-evidence sequence planner raises public multi-call to "
            f"`{public_multicall_sequence_planner['raw_seq']}` sequence and "
            f"`{public_multicall_sequence_planner['raw_args']}` arguments. This is also a deterministic "
            f"projection prototype, not a model-only metric."
        )
    for row in tool_result:
        lines.append(
            f"- {row['label']} is strong under constrained max-1 projection: "
            f"`{row['constrained_seq']}` sequence and `{row['constrained_args']}` arguments."
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This checkpoint beats the 1.5B diffusion lab baseline on strict public tool-call metrics.",
            "- It is still far below the Qwen3.5 AR and Qwen3.6 teacher multi-call baselines.",
            "- The next training step should preserve the tool-result behavior while targeting missing-call recovery, raw complex-payload emission, and repeated-call-safe sequence control.",
            "- Continue reporting raw strict metrics beside constrained metrics; constrained projection is useful but not a substitute for the model learning valid tool calls.",
            "",
            "## Source Artifacts",
            "",
        ]
    )
    for label, summary in loaded:
        out_jsonl = summary.get("out_jsonl") or summary.get("input_jsonl") or ""
        lines.append(f"- {label}: `{out_jsonl}`")

    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="Qwen3.5-9B Diffusion Checkpoint-275 Agentic Scorecard")
    parser.add_argument("--date", default=str(date.today()))
    parser.add_argument(
        "--adapter",
        default="runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model",
    )
    parser.add_argument(
        "--tokenizer",
        default="runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300",
    )
    parser.add_argument("--base-model", default="models/qwen3.5-9b-fastdllm-init")
    parser.add_argument(
        "--summary",
        action="append",
        type=parse_labeled_path,
        default=[],
        help="Add LABEL:PATH summary JSON. Defaults to the active checkpoint-275 artifacts.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=ROOT / "qwen35_9b_diffusion_ckpt275_agentic_scorecard.md",
    )
    args = parser.parse_args()

    summary_specs = args.summary or DEFAULT_SUMMARIES
    loaded = []
    missing = []
    for label, path in summary_specs:
        if not path.exists():
            missing.append(str(path))
            continue
        loaded.append((label, load_summary(path)))
    if missing:
        raise SystemExit("Missing summary artifact(s):\n" + "\n".join(missing))
    if not loaded:
        raise SystemExit("No summaries to report")

    rows = [summary_row(label, summary) for label, summary in loaded]
    write_markdown(args, rows, loaded)
    print(args.out_md)


if __name__ == "__main__":
    main()
