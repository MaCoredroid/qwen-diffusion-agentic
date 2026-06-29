#!/usr/bin/env python3
import argparse
import copy
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from eval_toolcall_jsonl import extract_tool_calls


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_MODEL = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_VALUE_SPAN_DIR = ROOT / "data/qwen35_9b_candidate_value_span_public_train_curriculum"
DEFAULT_TOOLCALL_TRAIN = ROOT / "data/fastdllm_toolcall_train/train_toolcall.json"
DEFAULT_SYNTHETIC_ONECALL = ROOT / "data/synthetic_onecall_train/train_synthetic_onecall.json"
DEFAULT_TOOLRESULT_TRAIN = ROOT / "data/synthetic_toolresult_train/train_synthetic_toolresult.json"
DEFAULT_OUT_DIR = ROOT / "data/qwen35_9b_route_delta_trainonly_mix_curriculum"


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


def load_conversation_json(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    instances = payload.get("instances")
    if not isinstance(instances, list):
        raise ValueError(f"{path} does not contain an instances list")
    return instances


def load_dataset_dir(dataset_dir):
    return load_conversation_json(dataset_dir / "train_agentic_mix.json")


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


def cap_instances(instances, cap):
    if cap < 0:
        return list(instances)
    return list(instances[:cap])


def add_instances(rows, instances, source, repeat=1, cap=-1):
    selected = cap_instances(instances, cap)
    for repeat_idx in range(repeat):
        for idx, instance in enumerate(selected):
            clone = copy.deepcopy(instance)
            clone["source"] = f"{source}:{idx}:repeat{repeat_idx}"
            rows.append((clone, source, repeat_idx))


def compact_tool_call_text(call):
    payload = {
        "name": call.get("name"),
        "arguments": call.get("arguments") if isinstance(call.get("arguments"), dict) else {},
    }
    return "<tool_call>\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ": ")) + "\n</tool_call>"


def tool_call_from_text(text):
    calls, invalid = extract_tool_calls(text or "")
    if invalid or len(calls) != 1:
        return None
    return calls[0]


def openai_tool_call(call, call_id):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": call.get("name"),
            "arguments": json.dumps(call.get("arguments") or {}, ensure_ascii=False, separators=(",", ":")),
        },
    }


def parse_tool_result_user_message(message):
    content = str(message.get("content") or "")
    match = re.search(r"Tool result for ([A-Za-z0-9_:-]+):\s*(\{.*?\})(?:\n|$)", content, flags=re.DOTALL)
    if not match:
        return None, None
    name = match.group(1)
    payload = match.group(2)
    try:
        parsed = json.loads(payload)
    except Exception:
        parsed = payload
    return name, json.dumps(parsed, ensure_ascii=False, sort_keys=True) if isinstance(parsed, dict) else str(parsed)


def openai_toolresult_train_instances(instances):
    out = []
    skipped = Counter()
    for idx, instance in enumerate(instances):
        messages = instance.get("messages") or []
        if len(messages) < 4:
            skipped["too_short"] += 1
            continue
        first_call = tool_call_from_text(messages[1].get("content") if len(messages) > 1 else "")
        next_call = tool_call_from_text(messages[-1].get("content") if messages else "")
        tool_name, tool_content = parse_tool_result_user_message(messages[2] if len(messages) > 2 else {})
        if not first_call or not next_call or not tool_name or tool_content is None:
            skipped["parse_failed"] += 1
            continue
        first_id = f"call_train_toolresult_{idx:05d}_first"
        next_id = f"call_train_toolresult_{idx:05d}_next"
        converted_messages = [
            copy.deepcopy(messages[0]),
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [openai_tool_call(first_call, first_id)],
            },
            {
                "role": "tool",
                "tool_call_id": first_id,
                "name": tool_name,
                "content": tool_content,
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [openai_tool_call(next_call, next_id)],
            },
        ]
        out.append(
            {
                "system": instance.get("system", "You are a helpful assistant."),
                "tools": copy.deepcopy(instance.get("tools") or []),
                "messages": converted_messages,
                "source": f"synthetic_toolresult_openai_train:{idx}",
            }
        )
    return out, skipped


def strip_training_metadata(instance):
    out = copy.deepcopy(instance)
    out.pop("source", None)
    return out


def accept_rows(rows, tokenizer, chat_template, args):
    accepted = []
    audits = []
    rejected = []
    for instance, source, repeat_idx in rows:
        stats = token_stats(tokenizer, chat_template, instance, args.block_size, args.truncation_side)
        audit = {
            "source": source,
            "instance_source": instance.get("source"),
            "repeat": repeat_idx,
            "tool_count": len(instance.get("tools") or []),
            **stats,
        }
        if stats["kept_labels"] < args.min_labels:
            rejected.append({**audit, "reject_reason": "too_few_labels"})
            continue
        if args.require_full_labels and not stats["full_labels_kept"]:
            rejected.append({**audit, "reject_reason": "partial_labels_after_truncation"})
            continue
        accepted.append(instance)
        audits.append(audit)
    return accepted, audits, rejected


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--conversation-template", default="fast_dllm_v2")
    parser.add_argument("--value-span-dir", type=Path, default=DEFAULT_VALUE_SPAN_DIR)
    parser.add_argument("--toolcall-train-json", type=Path, default=DEFAULT_TOOLCALL_TRAIN)
    parser.add_argument("--synthetic-onecall-json", type=Path, default=DEFAULT_SYNTHETIC_ONECALL)
    parser.add_argument("--toolresult-train-json", type=Path, default=DEFAULT_TOOLRESULT_TRAIN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--value-span-repeat", type=int, default=1)
    parser.add_argument("--toolcall-train-repeat", type=int, default=1)
    parser.add_argument("--synthetic-onecall-repeat", type=int, default=1)
    parser.add_argument("--toolresult-text-repeat", type=int, default=2)
    parser.add_argument("--toolresult-openai-repeat", type=int, default=3)
    parser.add_argument("--toolcall-train-cap", type=int, default=64)
    parser.add_argument("--synthetic-onecall-cap", type=int, default=48)
    parser.add_argument("--seed", type=int, default=2801)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--truncation-side", choices=["left", "right"], default="left")
    parser.add_argument("--min-labels", type=int, default=1)
    parser.add_argument("--require-full-labels", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    chat_template = resolve_chat_template(args.conversation_template)

    toolresult_train = load_conversation_json(args.toolresult_train_json)
    openai_toolresult, openai_skipped = openai_toolresult_train_instances(toolresult_train)

    rows = []
    add_instances(
        rows,
        load_dataset_dir(args.value_span_dir),
        "public_train_value_span",
        repeat=args.value_span_repeat,
    )
    add_instances(
        rows,
        load_conversation_json(args.toolcall_train_json),
        "fastdllm_toolcall_train",
        repeat=args.toolcall_train_repeat,
        cap=args.toolcall_train_cap,
    )
    add_instances(
        rows,
        load_conversation_json(args.synthetic_onecall_json),
        "synthetic_onecall_train",
        repeat=args.synthetic_onecall_repeat,
        cap=args.synthetic_onecall_cap,
    )
    add_instances(
        rows,
        toolresult_train,
        "synthetic_toolresult_text_train",
        repeat=args.toolresult_text_repeat,
    )
    add_instances(
        rows,
        openai_toolresult,
        "synthetic_toolresult_openai_train",
        repeat=args.toolresult_openai_repeat,
    )

    accepted, audits, rejected = accept_rows(rows, tokenizer, chat_template, args)
    order = list(range(len(accepted)))
    random.Random(args.seed).shuffle(order)
    accepted = [accepted[idx] for idx in order]
    audits = [audits[idx] for idx in order]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train_agentic_mix.json"
    audit_path = args.out_dir / "train_agentic_mix.audit.jsonl"
    rejected_path = args.out_dir / "train_agentic_mix.rejected.jsonl"
    train_path.write_text(
        json.dumps(
            {"type": "conversation", "instances": [strip_training_metadata(item) for item in accepted]},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    write_jsonl(audit_path, audits)
    write_jsonl(rejected_path, rejected)

    source_counts = Counter(row["source"] for row in audits)
    rejected_counts = Counter(row["source"] for row in rejected)
    manifest = {
        "train_path": str(train_path),
        "audit_path": str(audit_path),
        "rejected_path": str(rejected_path),
        "count": len(accepted),
        "candidate_count": len(rows),
        "rejected_count": len(rejected),
        "source_counts": dict(sorted(source_counts.items())),
        "rejected_source_counts": dict(sorted(rejected_counts.items())),
        "inputs": {
            "value_span_dir": str(args.value_span_dir),
            "toolcall_train_json": str(args.toolcall_train_json),
            "synthetic_onecall_json": str(args.synthetic_onecall_json),
            "toolresult_train_json": str(args.toolresult_train_json),
        },
        "repeats": {
            "value_span": args.value_span_repeat,
            "toolcall_train": args.toolcall_train_repeat,
            "synthetic_onecall": args.synthetic_onecall_repeat,
            "toolresult_text": args.toolresult_text_repeat,
            "toolresult_openai": args.toolresult_openai_repeat,
        },
        "caps": {
            "toolcall_train": args.toolcall_train_cap,
            "synthetic_onecall": args.synthetic_onecall_cap,
        },
        "openai_toolresult_converted": len(openai_toolresult),
        "openai_toolresult_skipped": dict(sorted(openai_skipped.items())),
        "block_size": args.block_size,
        "truncation_side": args.truncation_side,
        "min_labels": args.min_labels,
        "require_full_labels": args.require_full_labels,
        "contains_eval_slice": False,
        "diagnostic_only": False,
        "promotion_allowed": True,
        "route_delta_report": str(ROOT / "qwen35_public_train_candidate_value_span_route_delta.md"),
        "intended_failure_classes": [
            "one-call sequence retention",
            "scalar argument grounding",
            "text tool-result next-action retention",
            "OpenAI-style tool-result argument retention",
        ],
        "audit_summary": summarize_audits(audits),
        "rejected_examples": rejected[:20],
        "seed": args.seed,
    }
    manifest_path = args.out_dir / "train_agentic_mix.manifest"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
