#!/usr/bin/env python3
import argparse
import copy
import json
import random
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_MODEL = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_SYNTHETIC_DIR = (
    ROOT / "data/qwen35_9b_toolcall_grounded_spanfill_synthetic_onecall48_b1024_curriculum"
)
DEFAULT_RETENTION_DIR = (
    ROOT / "data/qwen35_9b_toolcall_grounded_spanfill_teacher_train_b1024_curriculum"
)
DEFAULT_SEQUENCE_PLANNER_DIR = (
    ROOT / "data/qwen35_9b_toolcall_sequence_planner_distill_compact_b1024_curriculum"
)
DEFAULT_TOOLRESULT_TEXT = ROOT / "data/synthetic_toolresult_train/train_synthetic_toolresult.json"
DEFAULT_TOOLRESULT_OPENAI = ROOT / "data/toolcall_eval/synthetic_toolresult_openai_smoke.jsonl"
DEFAULT_OUT_DIR = (
    ROOT / "data/qwen35_9b_toolcall_checkpoint24_antiregression_b1024_curriculum"
)


def resolve_chat_template(name):
    third_party = ROOT / "fast-dllm/third_party"
    if str(third_party) not in sys.path:
        sys.path.insert(0, str(third_party))
    from lmflow.utils.conversation_template import PRESET_TEMPLATES

    if name not in PRESET_TEMPLATES:
        raise ValueError(f"unknown conversation template {name!r}")
    return PRESET_TEMPLATES[name]


def drop_none_fields(value):
    if isinstance(value, dict):
        return {key: drop_none_fields(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [drop_none_fields(item) for item in value if item is not None]
    return value


def load_dataset_dir(dataset_dir):
    train_path = dataset_dir / "train_agentic_mix.json"
    payload = json.loads(train_path.read_text(encoding="utf-8"))
    instances = payload.get("instances")
    if not isinstance(instances, list):
        raise ValueError(f"{train_path} does not contain an instances list")
    return instances


def load_conversation_json(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    instances = payload.get("instances")
    if not isinstance(instances, list):
        raise ValueError(f"{path} does not contain an instances list")
    return instances


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def system_from_prompt_messages(case):
    for message in case.get("prompt_messages") or []:
        if message.get("role") == "system" and message.get("content"):
            return message["content"]
    return "You are a helpful assistant."


def openai_toolresult_instances(path):
    instances = []
    for case in load_jsonl(path):
        messages = copy.deepcopy(case.get("prompt_messages") or [])
        target = copy.deepcopy(case.get("gold_assistant_openai") or {})
        if not messages or not target.get("tool_calls"):
            continue
        system = system_from_prompt_messages(case)
        messages = [message for message in messages if message.get("role") != "system"]
        messages.append(target)
        instances.append(
            {
                "system": system,
                "tools": copy.deepcopy(case.get("tools") or []),
                "messages": messages,
            }
        )
    return instances


def conversation_for_template(instance):
    system = instance.get("system")
    messages = [{"role": "system", "content": system if system is not None else "You are a helpful assistant."}]
    messages.extend(copy.deepcopy(instance.get("messages") or []))
    return drop_none_fields(messages)


def token_stats(tokenizer, chat_template, instance, block_size, truncation_side):
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


def add_rows(rows, instances, mix_source, repeat):
    for repeat_idx in range(repeat):
        for instance in instances:
            rows.append((copy.deepcopy(instance), mix_source, repeat_idx))


def percentile_summary(values):
    if not values:
        return {}
    values = sorted(values)

    def at(frac):
        idx = min(len(values) - 1, max(0, round((len(values) - 1) * frac)))
        return values[idx]

    return {"min": values[0], "p50": at(0.5), "p90": at(0.9), "max": values[-1]}


def summarize_audits(rows):
    summary = {"count": len(rows)}
    for key in ("length", "full_labels", "kept_labels"):
        values = [row[key] for row in rows if isinstance(row.get(key), int)]
        if values:
            summary[key] = percentile_summary(values)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--conversation-template", default="fast_dllm_v2")
    parser.add_argument("--synthetic-dir", type=Path, default=DEFAULT_SYNTHETIC_DIR)
    parser.add_argument("--retention-dir", type=Path, default=DEFAULT_RETENTION_DIR)
    parser.add_argument("--sequence-planner-dir", type=Path, default=DEFAULT_SEQUENCE_PLANNER_DIR)
    parser.add_argument("--toolresult-text-json", type=Path, default=DEFAULT_TOOLRESULT_TEXT)
    parser.add_argument("--toolresult-openai-jsonl", type=Path, default=DEFAULT_TOOLRESULT_OPENAI)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--synthetic-repeat", type=int, default=1)
    parser.add_argument("--retention-repeat", type=int, default=2)
    parser.add_argument("--sequence-planner-repeat", type=int, default=1)
    parser.add_argument("--toolresult-text-repeat", type=int, default=1)
    parser.add_argument("--toolresult-openai-repeat", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--truncation-side", choices=["left", "right"], default="right")
    parser.add_argument("--require-full-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=1247)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    chat_template = resolve_chat_template(args.conversation_template)

    rows = []
    add_rows(rows, load_dataset_dir(args.synthetic_dir), "synthetic_grounded_spanfill", args.synthetic_repeat)
    add_rows(rows, load_dataset_dir(args.retention_dir), "teacher_train_grounded_retention", args.retention_repeat)
    add_rows(rows, load_dataset_dir(args.sequence_planner_dir), "sequence_planner_compact_retention", args.sequence_planner_repeat)
    add_rows(rows, load_conversation_json(args.toolresult_text_json), "synthetic_toolresult_text_retention", args.toolresult_text_repeat)
    add_rows(rows, openai_toolresult_instances(args.toolresult_openai_jsonl), "synthetic_toolresult_openai_retention", args.toolresult_openai_repeat)

    accepted = []
    rejected = []
    audit_rows = []
    for instance, mix_source, repeat_idx in rows:
        stats = token_stats(tokenizer, chat_template, instance, args.block_size, args.truncation_side)
        audit = {
            "mix_source": mix_source,
            "mix_repeat": repeat_idx,
            "tool_count": len(instance.get("tools") or []),
            **stats,
        }
        if stats["kept_labels"] <= 0 or (args.require_full_labels and not stats["full_labels_kept"]):
            rejected.append(audit)
            continue
        accepted.append(instance)
        audit_rows.append(audit)

    order = list(range(len(accepted)))
    random.Random(args.seed).shuffle(order)
    accepted = [accepted[idx] for idx in order]
    audit_rows = [audit_rows[idx] for idx in order]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train_agentic_mix.json"
    train_path.write_text(
        json.dumps({"type": "conversation", "instances": accepted}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    audit_path = args.out_dir / "train_agentic_mix.audit.jsonl"
    audit_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audit_rows),
        encoding="utf-8",
    )

    source_counts = Counter(row["mix_source"] for row in audit_rows)
    rejected_counts = Counter(row["mix_source"] for row in rejected)
    manifest = {
        "train_path": str(train_path),
        "audit_path": str(audit_path),
        "count": len(accepted),
        "candidate_count": len(rows),
        "rejected_count": len(rejected),
        "mix_source_counts": dict(sorted(source_counts.items())),
        "rejected_source_counts": dict(sorted(rejected_counts.items())),
        "inputs": {
            "synthetic_dir": str(args.synthetic_dir),
            "retention_dir": str(args.retention_dir),
            "sequence_planner_dir": str(args.sequence_planner_dir),
            "toolresult_text_json": str(args.toolresult_text_json),
            "toolresult_openai_jsonl": str(args.toolresult_openai_jsonl),
        },
        "repeats": {
            "synthetic": args.synthetic_repeat,
            "retention": args.retention_repeat,
            "sequence_planner": args.sequence_planner_repeat,
            "toolresult_text": args.toolresult_text_repeat,
            "toolresult_openai": args.toolresult_openai_repeat,
        },
        "block_size": args.block_size,
        "truncation_side": args.truncation_side,
        "require_full_labels": args.require_full_labels,
        "audit_summary": summarize_audits(audit_rows),
        "rejected_examples": rejected[:20],
        "seed": args.seed,
    }
    manifest_path = args.out_dir / "train_agentic_mix.manifest"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
