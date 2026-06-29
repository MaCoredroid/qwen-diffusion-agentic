#!/usr/bin/env python3
import argparse
import copy
import json
from pathlib import Path


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_SOURCE = ROOT / "data/synthetic_onecall_train/train_synthetic_onecall.json"
DEFAULT_OUT_DIR = ROOT / "data/qwen35_9b_toolcall_format_curriculum"
DEFAULT_TRAIN_EVAL = ROOT / "data/toolcall_eval/toolcall_format_train_smoke.jsonl"
DEFAULT_HELDOUT_EVAL = ROOT / "data/toolcall_eval/toolcall_format_heldout_smoke.jsonl"
FORMAT_SYSTEM = (
    "You are a tool-call formatter. Return exactly one <tool_call> block with "
    "valid JSON and no prose."
)


def extract_call(text):
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"assistant text does not contain JSON object: {text[:80]!r}")
    payload = json.loads(text[start : end + 1])
    return str(payload["name"]), payload["arguments"]


def compact_tool_call(name, arguments):
    payload = {"name": name, "arguments": arguments}
    return "<tool_call>\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ": ")) + "\n</tool_call>"


def tool_name(tool):
    return str((tool.get("function") or tool).get("name"))


def eval_case(source, idx, instance, variant):
    assistant = instance["messages"][-1]["content"]
    name, _ = extract_call(assistant)
    return {
        "source": f"toolcall_format_{variant}",
        "id": f"toolcall-format-{idx:05d}-{variant}",
        "tools": instance.get("tools") or [],
        "prompt_messages": [
            {"role": "system", "content": instance.get("system") or FORMAT_SYSTEM},
            instance["messages"][0],
        ],
        "gold_assistant": assistant,
        "gold_tool_names": [name],
        "available_tool_names": sorted(tool_name(tool) for tool in instance.get("tools") or []),
    }


def original_variant(instance):
    clone = copy.deepcopy(instance)
    clone["messages"][-1]["content"] = compact_tool_call(*extract_call(clone["messages"][-1]["content"]))
    return clone


def single_tool_variant(instance):
    clone = original_variant(instance)
    name, _ = extract_call(clone["messages"][-1]["content"])
    clone["system"] = FORMAT_SYSTEM
    clone["tools"] = [tool for tool in clone.get("tools") or [] if tool_name(tool) == name]
    return clone


def explicit_format_variant(instance):
    clone = single_tool_variant(instance)
    name, arguments = extract_call(clone["messages"][-1]["content"])
    clone["messages"][0] = {
        "role": "user",
        "content": (
            "Format this function call exactly as a Qwen tool call.\n"
            f"Function name: {name}\n"
            "Arguments JSON: "
            + json.dumps(arguments, ensure_ascii=False, separators=(",", ": "))
        ),
    }
    return clone


def load_instances(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") != "conversation":
        raise ValueError(f"expected conversation dataset in {path}")
    return payload["instances"]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--train-eval-out", type=Path, default=DEFAULT_TRAIN_EVAL)
    parser.add_argument("--heldout-eval-out", type=Path, default=DEFAULT_HELDOUT_EVAL)
    parser.add_argument("--train-base-count", type=int, default=32)
    parser.add_argument("--heldout-base-count", type=int, default=16)
    args = parser.parse_args()

    source_instances = load_instances(args.source)
    if args.train_base_count + args.heldout_base_count > len(source_instances):
        raise ValueError("requested more base examples than the source contains")

    train_instances = []
    train_eval = []
    variants = [
        ("original", original_variant),
        ("single_tool", single_tool_variant),
        ("explicit_format", explicit_format_variant),
    ]
    for base_idx, source in enumerate(source_instances[: args.train_base_count]):
        for variant_name, make_variant in variants:
            instance = make_variant(source)
            train_instances.append(instance)
            train_eval.append(eval_case(source, base_idx, instance, variant_name))

    heldout_eval = []
    heldout_start = args.train_base_count
    heldout_end = heldout_start + args.heldout_base_count
    for base_idx, source in enumerate(source_instances[heldout_start:heldout_end], start=heldout_start):
        instance = original_variant(source)
        heldout_eval.append(eval_case(source, base_idx, instance, "heldout_original"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train_agentic_mix.json"
    train_path.write_text(
        json.dumps({"type": "conversation", "instances": train_instances}, indent=2) + "\n",
        encoding="utf-8",
    )
    write_jsonl(args.train_eval_out, train_eval)
    write_jsonl(args.heldout_eval_out, heldout_eval)

    manifest = {
        "source": str(args.source),
        "train_path": str(train_path),
        "train_eval_path": str(args.train_eval_out),
        "heldout_eval_path": str(args.heldout_eval_out),
        "train_base_count": args.train_base_count,
        "heldout_base_count": args.heldout_base_count,
        "train_instances": len(train_instances),
        "variants": [name for name, _ in variants],
    }
    manifest_path = args.out_dir / "train_agentic_mix.manifest"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
