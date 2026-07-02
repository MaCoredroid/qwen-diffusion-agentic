#!/usr/bin/env python3
"""Build audited self-generated multi-turn SFT warm-start data.

The builder uses public training episodes only, converts their gold references
to Qwen-native function/parameter form, runs the current careful diffusion
policy with live structure grammar, and keeps only turns whose generated tool
call is exact-arguments-correct under the audited scorer. The resulting
tool-call rows are oversampled for the miss classes identified by the careful
taxonomy, then mixed 50/50 with the existing retention corpus.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from audit_value_projection_tokens import audit_rows  # noqa: E402
from convert_toolcall_cases_to_qwen_native import convert_eval_row, convert_instance  # noqa: E402
from eval_toolcall_jsonl import extract_tool_calls, normalize_call_for_compare, tool_schema_by_name  # noqa: E402
from rl_multiturn_tool_env import (  # noqa: E402
    DEFAULT_DIFFUSION_ADAPTER,
    DEFAULT_DIFFUSION_BASE,
    EVAL_BATTERY_PATHS,
    MultiTurnToolRLEnv,
    filter_eval_battery_rows,
    read_jsonl,
    write_json,
    write_jsonl,
)


DEFAULT_JSONL_SOURCES = [
    ROOT / "data/flare_agentic_mix_v2_pilot/teacher_native_fresh30_cases.jsonl",
    ROOT / "data/flare_agentic_mix_v2_pilot/teacher_pilot_cases.jsonl",
]
DEFAULT_TRAIN_JSON_SOURCES = [
    ROOT / "data/flare_agentic_mix_v2_native_train_only/train_agentic_mix.json",
    ROOT / "data/qwen35_9b_toolcall_multicall_curriculum/train_agentic_mix.json",
]
DEFAULT_RETENTION = ROOT / "data/flare_stage1_ab_pilot_train/train_agentic_mix.json"
DEFAULT_OUT_DIR = ROOT / "data/multiturn_sft_warmstart"


def load_conversation_instances(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    instances = payload.get("instances") if isinstance(payload, dict) else None
    if not isinstance(instances, list):
        raise ValueError(f"{path} does not contain a conversation instances list")
    return [item for item in instances if isinstance(item, dict)]


def assistant_text_from_instance(instance: dict[str, Any]) -> str:
    return "\n".join(
        str(message.get("content") or "")
        for message in instance.get("messages") or []
        if isinstance(message, dict) and message.get("role") == "assistant"
    )


def instance_fingerprint(instance: dict[str, Any]) -> str:
    payload = {
        "system": instance.get("system"),
        "messages": instance.get("messages") or [],
        "tools": instance.get("tools") or [],
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def train_instance_to_episode_row(instance: dict[str, Any], source_path: Path, source_idx: int) -> dict[str, Any] | None:
    converted, _ = convert_instance(instance)
    messages = converted.get("messages") or []
    chosen_idx = None
    for idx, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = str(message.get("content") or "")
        calls, invalid = extract_tool_calls(content)
        if not invalid and len(calls) >= 2:
            chosen_idx = idx
    if chosen_idx is None:
        return None
    prompt_messages = []
    if converted.get("system"):
        prompt_messages.append({"role": "system", "content": str(converted["system"])})
    for message in messages[:chosen_idx]:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = str(message.get("content") or "").strip()
        if role in {"user", "assistant", "tool"} and content:
            prompt_messages.append({"role": role, "content": content})
    gold_assistant = str(messages[chosen_idx].get("content") or "").strip()
    calls, invalid = extract_tool_calls(gold_assistant)
    if invalid or len(calls) < 2:
        return None
    return {
        "id": f"{source_path.parent.name}_{source_idx:04d}",
        "source": converted.get("source") or source_path.parent.name,
        "prompt_messages": prompt_messages,
        "tools": converted.get("tools") or [],
        "gold_assistant": gold_assistant,
        "gold_assistant_format": "qwen_native_function_parameter",
        "gold_tool_calls": [
            {"name": call.get("name"), "arguments": call.get("arguments") or {}, "format": "qwen_native"}
            for call in calls
        ],
        "tool_call_count": len(calls),
        "train_source_path": str(source_path),
        "train_source_index": source_idx,
    }


def materialize_source_rows(
    input_jsonls: list[Path],
    input_train_jsons: list[Path],
    out_jsonl: Path,
) -> dict[str, Any]:
    rows = []
    source_manifest = []
    for input_jsonl in input_jsonls:
        source_rows = read_jsonl(input_jsonl)
        rows.extend(source_rows)
        source_manifest.append({"path": str(input_jsonl), "kind": "eval-jsonl", "input_rows": len(source_rows)})
    for input_train_json in input_train_jsons:
        source_instances = load_conversation_instances(input_train_json)
        converted_rows = []
        for idx, instance in enumerate(source_instances):
            row = train_instance_to_episode_row(instance, input_train_json, idx)
            if row is not None:
                converted_rows.append(row)
        rows.extend(converted_rows)
        source_manifest.append(
            {
                "path": str(input_train_json),
                "kind": "train-json",
                "input_rows": len(source_instances),
                "materialized_multicall_rows": len(converted_rows),
            }
        )
    converted_rows = []
    converted_count = 0
    rejected = Counter()
    for row in rows:
        native, converted = convert_eval_row(row)
        if not converted:
            rejected["convert_failed"] += 1
            continue
        converted_rows.append(native)
        converted_count += 1
    write_jsonl(out_jsonl, converted_rows)
    return {
        "sources": source_manifest,
        "out_jsonl": str(out_jsonl),
        "input_rows": len(rows),
        "converted_rows": converted_count,
        "rejected": dict(sorted(rejected.items())),
        "gold_assistant_format": "qwen_native_function_parameter",
    }


def split_prompt_messages(prompt_messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, str]]]:
    system = ""
    messages: list[dict[str, str]] = []
    for message in prompt_messages or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        if role == "system" and not system:
            system = content
            continue
        if role in {"user", "assistant", "tool"}:
            messages.append({"role": role, "content": content})
    return system or "You are a helpful assistant.", messages


def content_or_json(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def add_tool_response_context(messages: list[dict[str, str]], payload: Any, next_user: str | None) -> None:
    messages.append({"role": "tool", "content": content_or_json(payload)})
    if next_user is not None and str(next_user).strip():
        messages.append({"role": "user", "content": str(next_user).strip()})


def complex_paths_from_value(value: Any, schema: dict[str, Any], prefix: tuple[Any, ...] = ()) -> list[str]:
    paths: list[str] = []
    schema_type = schema.get("type") if isinstance(schema, dict) else None
    if isinstance(schema_type, list):
        schema_type = next((item for item in schema_type if item != "null"), schema_type[0] if schema_type else None)
    if schema_type in {"array", "object"} or isinstance(value, (list, dict)):
        paths.append(".".join(str(item) for item in prefix) or "$")
    if isinstance(value, dict):
        props = schema.get("properties") if isinstance(schema, dict) else {}
        props = props if isinstance(props, dict) else {}
        for key, item in value.items():
            paths.extend(complex_paths_from_value(item, props.get(key, {}), prefix + (key,)))
    elif isinstance(value, list):
        item_schema = schema.get("items") if isinstance(schema, dict) else {}
        item_schema = item_schema if isinstance(item_schema, dict) else {}
        for idx, item in enumerate(value):
            paths.extend(complex_paths_from_value(item, item_schema, prefix + (idx,)))
    return paths


def classify_gold_turn(gold_block: str, tools: list[dict[str, Any]], episode_turns: int, turn_idx: int) -> dict[str, Any]:
    calls, invalid = extract_tool_calls(gold_block or "")
    schemas = tool_schema_by_name(tools)
    normalized = [normalize_call_for_compare(call, schemas) for call in calls]
    complex_paths: list[str] = []
    for call in normalized:
        fn_schema = schemas.get(call.get("name"), {})
        props = fn_schema.get("properties") if isinstance(fn_schema, dict) else {}
        props = props if isinstance(props, dict) else {}
        arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        for key, value in arguments.items():
            complex_paths.extend(
                f"{call.get('name')}.{path}"
                for path in complex_paths_from_value(value, props.get(key, {}), (key,))
            )
    return {
        "gold_call_count": len(calls) + invalid,
        "gold_invalid_tool_call_count": invalid,
        "complex_value": bool(complex_paths),
        "complex_paths": sorted(set(complex_paths))[:20],
        "long_episode": episode_turns >= 4,
        "long_episode_final_stop": episode_turns >= 4 and turn_idx == episode_turns - 1,
    }


def build_sft_instance(
    episode: dict[str, Any],
    history_messages: list[dict[str, str]],
    assistant_text: str,
    *,
    source: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    system, _ = split_prompt_messages(episode.get("prompt_messages") or [])
    messages = [copy.deepcopy(item) for item in history_messages]
    messages.append({"role": "assistant", "content": assistant_text.strip()})
    instance = {
        "system": system,
        "messages": messages,
        "tools": copy.deepcopy(episode.get("tools") or []),
        "source": source,
        "sft_metadata": metadata,
    }
    return instance


def rollout_training_examples(args: argparse.Namespace, native_jsonl: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    env_args = argparse.Namespace(
        input_jsonl=native_jsonl,
        out_dir=args.out_dir / "rollouts",
        episode_limit=args.episode_limit,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        prompt_tokenizer_path=args.prompt_tokenizer_path,
        tokenizer_path=args.tokenizer_path,
        chat_template_path=args.chat_template_path,
        base_model=args.base_model,
        adapter=args.adapter,
        no_merge_adapter=args.no_merge_adapter,
        block_size=args.block_size,
        small_block_size=args.small_block_size,
        max_new_tokens=args.max_new_tokens,
        threshold=args.threshold,
        top_p=args.top_p,
        temperature=args.temperature,
        live_tool_json_topk=args.live_tool_json_topk,
        seed=args.seed,
        eval_battery_paths=args.eval_battery_paths,
    )
    torch.manual_seed(args.seed)
    env = MultiTurnToolRLEnv(env_args)
    accepted: list[dict[str, Any]] = []
    audit_rows_out: list[dict[str, Any]] = []
    rollout_summaries = []
    for episode in env.episodes:
        system, prompt_history = split_prompt_messages(episode.get("prompt_messages") or [])
        del system
        current_messages = [copy.deepcopy(item) for item in prompt_history]
        rollout = env.rollout_episode(episode)
        episode_summary = {
            "episode_id": episode["id"],
            "turns": len(rollout["turns"]),
            "accepted_turns": 0,
            "exact_turns": 0,
            "audit_clean_turns": 0,
        }
        for turn_idx, (turn_row, reward_row, gold_block) in enumerate(
            zip(rollout["turns"], rollout["rewards"], episode["gold_blocks"])
        ):
            turn_class = classify_gold_turn(gold_block, episode.get("tools") or [], len(episode["gold_blocks"]), turn_idx)
            exact = bool(reward_row.get("exact_args"))
            audit_clean = bool(reward_row.get("audit_clean")) and int(reward_row.get("projected_value_tokens_exact") or 0) == 0
            accepted_turn = exact and audit_clean
            episode_summary["exact_turns"] += int(exact)
            episode_summary["audit_clean_turns"] += int(audit_clean)
            episode_summary["accepted_turns"] += int(accepted_turn)
            metadata = {
                "episode_id": episode["id"],
                "turn_idx": turn_idx,
                "episode_turns": len(episode["gold_blocks"]),
                "reward": reward_row.get("reward"),
                "exact_args": exact,
                "audit_clean": audit_clean,
                "projected_value_tokens_exact": int(reward_row.get("projected_value_tokens_exact") or 0),
                "self_generated": True,
                "policy": "diffusion_careful_live_grammar_values_raw",
                "class": turn_class,
            }
            audit_rows_out.append({**metadata, "accepted_for_sft": accepted_turn})
            if accepted_turn:
                accepted.append(
                    build_sft_instance(
                        episode,
                        current_messages,
                        turn_row.get("assistant") or "",
                        source="selfgen_multiturn_exact_audited",
                        metadata=metadata,
                    )
                )
            current_messages.append({"role": "assistant", "content": str(turn_row.get("assistant") or "").strip()})
            next_user = episode.get("turn_user_messages", [])
            next_user_text = next_user[turn_idx + 1] if turn_idx + 1 < len(next_user) else None
            add_tool_response_context(current_messages, turn_row.get("tool_response_payload"), next_user_text)
        rollout_summaries.append(episode_summary)
    rollout_manifest = {
        "episodes": len(env.episodes),
        "turns": sum(item["turns"] for item in rollout_summaries),
        "accepted_turns": sum(item["accepted_turns"] for item in rollout_summaries),
        "exact_turns": sum(item["exact_turns"] for item in rollout_summaries),
        "audit_clean_turns": sum(item["audit_clean_turns"] for item in rollout_summaries),
        "episode_summaries": rollout_summaries,
        "leak_filter_manifest": env.leak_manifest,
    }
    return accepted, audit_rows_out, rollout_manifest


def repeat_to_count(items: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    if not items:
        raise ValueError("cannot repeat empty item list")
    pool = [copy.deepcopy(item) for item in items]
    out = []
    cursor = 0
    while len(out) < count:
        if cursor % len(pool) == 0:
            rng.shuffle(pool)
        out.append(copy.deepcopy(pool[cursor % len(pool)]))
        cursor += 1
    return out


def oversample_tool_examples(raw: list[dict[str, Any]], args: argparse.Namespace, rng: random.Random) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    repeat_hist = Counter()
    class_counts = Counter()
    for item in raw:
        meta = item.get("sft_metadata") or {}
        klass = meta.get("class") or {}
        repeats = 1
        reasons = ["base"]
        if klass.get("complex_value"):
            repeats += args.complex_extra_repeats
            reasons.append("complex_array_object_value")
        if klass.get("long_episode_final_stop"):
            repeats += args.long_stop_extra_repeats
            reasons.append("long_episode_final_stop")
        elif klass.get("long_episode"):
            repeats += args.long_context_extra_repeats
            reasons.append("long_episode_context")
        class_counts["complex_value"] += int(bool(klass.get("complex_value")))
        class_counts["long_episode"] += int(bool(klass.get("long_episode")))
        class_counts["long_episode_final_stop"] += int(bool(klass.get("long_episode_final_stop")))
        repeat_hist[repeats] += 1
        for copy_idx in range(repeats):
            cloned = copy.deepcopy(item)
            cloned["sft_metadata"] = dict(meta)
            cloned["sft_metadata"]["oversample_reasons"] = reasons
            cloned["sft_metadata"]["oversample_copy_idx"] = copy_idx
            expanded.append(cloned)
    rng.shuffle(expanded)
    if args.max_tool_examples and len(expanded) > args.max_tool_examples:
        expanded = expanded[: args.max_tool_examples]
    return expanded, {
        "raw_accepted_turns": len(raw),
        "oversampled_tool_examples": len(expanded),
        "class_counts_raw": dict(sorted(class_counts.items())),
        "repeat_histogram_raw": dict(sorted((str(k), v) for k, v in repeat_hist.items())),
        "oversample_policy": {
            "complex_extra_repeats": args.complex_extra_repeats,
            "long_stop_extra_repeats": args.long_stop_extra_repeats,
            "long_context_extra_repeats": args.long_context_extra_repeats,
            "max_tool_examples": args.max_tool_examples,
        },
    }


def select_retention(args: argparse.Namespace, target: int, rng: random.Random) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    retention_raw = load_conversation_instances(args.retention_json)
    if not retention_raw:
        raise ValueError(f"no retention instances in {args.retention_json}")
    selected = repeat_to_count(retention_raw, target, rng)
    for item in selected:
        item.setdefault("source", "retention")
        item["retention_mix"] = True
    return selected, {
        "path": str(args.retention_json),
        "raw_instances": len(retention_raw),
        "selected_instances": len(selected),
        "unique_selected_instances": len({instance_fingerprint(item) for item in selected}),
        "source_counts": dict(sorted(Counter(item.get("source") or "unknown" for item in selected).items())),
        "policy": "repeat/shuffle retention to match oversampled audited tool-call example count exactly",
    }


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Multi-Turn SFT Warm-Start",
        "",
        "This corpus is for the SFT warm-start stage only. Tool-call rows are self-generated on public training episodes, then kept only when the audited scorer verifies exact arguments and zero projected value tokens.",
        "",
        "## Counts",
        "",
        f"- Final instances: `{manifest['final_count']}`",
        f"- Tool-call instances: `{manifest['tool_count']}`",
        f"- Retention instances: `{manifest['retention_count']}`",
        f"- Raw accepted self-generated turns: `{manifest['oversampling']['raw_accepted_turns']}`",
        f"- Rollout exact turns: `{manifest['rollout']['exact_turns']}/{manifest['rollout']['turns']}`",
        "",
        "## Gates",
        "",
        f"- Eval-battery filter rejected rows: `{manifest['leak_filter']['rejected_rows']}`",
        f"- Value projection audit verified: `{manifest['projection_value_audit']['zero_projected_value_tokens_verified']}`",
        f"- Projected value tokens: `{manifest['projection_value_audit']['projected_value_tokens_exact']}`",
        "",
        "## SFT Settings",
        "",
        "- Train with `CONVERSATION_TEMPLATE=fast_dllm_v2_native` so the tool schema prompt and assistant targets use the native function/parameter contract.",
        "- The next gate is GSM8K retention accuracy `>=0.70`; stop before RL if it fails.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl",
        dest="input_jsonls",
        action="append",
        type=Path,
        default=None,
        help="Public training episode JSONL. Repeatable. Defaults to clean teacher-pilot public case pools.",
    )
    parser.add_argument(
        "--input-train-json",
        dest="input_train_jsons",
        action="append",
        type=Path,
        default=None,
        help="Conversation train JSON to materialize multi-call assistant targets from. Repeatable.",
    )
    parser.add_argument("--retention-json", type=Path, default=DEFAULT_RETENTION)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--episode-limit", type=int, default=18)
    parser.add_argument("--min-turns", type=int, default=2)
    parser.add_argument("--max-turns", type=int, default=7)
    parser.add_argument("--prompt-tokenizer-path", type=Path, default=ROOT / "models/qwen3.5-9b-fastdllm-b1000-vllm-bf16")
    parser.add_argument("--tokenizer-path", type=Path, default=ROOT / "models/qwen3.5-9b-fastdllm-b1000-vllm-bf16")
    parser.add_argument(
        "--chat-template-path",
        type=Path,
        default=Path("/home/mark/shared/lumoFlyWheel_codex_fork/docker/chat_templates/qwen3-openai-codex.jinja"),
    )
    parser.add_argument("--base-model", type=Path, default=DEFAULT_DIFFUSION_BASE)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_DIFFUSION_ADAPTER)
    parser.add_argument("--no-merge-adapter", action="store_true", default=True)
    parser.add_argument("--merge-adapter", dest="no_merge_adapter", action="store_false")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--small-block-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--live-tool-json-topk", type=int, default=128)
    parser.add_argument("--complex-extra-repeats", type=int, default=3)
    parser.add_argument("--long-stop-extra-repeats", type=int, default=3)
    parser.add_argument("--long-context-extra-repeats", type=int, default=1)
    parser.add_argument("--max-tool-examples", type=int, default=320)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument(
        "--eval-battery-path",
        dest="eval_battery_paths",
        action="append",
        type=Path,
        default=list(EVAL_BATTERY_PATHS),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    input_jsonls = args.input_jsonls if args.input_jsonls is not None else list(DEFAULT_JSONL_SOURCES)
    input_train_jsons = args.input_train_jsons if args.input_train_jsons is not None else list(DEFAULT_TRAIN_JSON_SOURCES)
    native_jsonl = args.out_dir / "public_train_multicall_native.jsonl"
    conversion_manifest = materialize_source_rows(input_jsonls, input_train_jsons, native_jsonl)
    native_rows = read_jsonl(native_jsonl)
    kept_rows, leak_filter = filter_eval_battery_rows(native_jsonl, native_rows, args.eval_battery_paths)
    filtered_native_jsonl = args.out_dir / "public_train_multicall_native.filtered.jsonl"
    write_jsonl(filtered_native_jsonl, kept_rows)

    tool_raw, rollout_audit_rows, rollout_manifest = rollout_training_examples(args, filtered_native_jsonl)
    tool_examples, oversampling = oversample_tool_examples(tool_raw, args, rng)
    if not tool_examples:
        raise SystemExit("no audited exact self-generated turns were accepted for SFT")

    retention, retention_manifest = select_retention(args, len(tool_examples), rng)
    final_instances = tool_examples + retention
    rng.shuffle(final_instances)

    train_path = args.out_dir / "train_agentic_mix.json"
    audit_path = args.out_dir / "train_agentic_mix.audit.jsonl"
    manifest_path = args.out_dir / "manifest.json"
    report_path = args.out_dir / "report.md"
    projection_audit_path = args.out_dir / "projection_value_audit.json"
    projection_audit_detail_path = args.out_dir / "projection_value_audit.jsonl"

    write_json(train_path, {"type": "conversation", "instances": final_instances})
    write_jsonl(audit_path, rollout_audit_rows)
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_path), trust_remote_code=True)
    tool_train_rows = [
        {
            "id": (item.get("sft_metadata") or {}).get("episode_id"),
            "assistant": assistant_text_from_instance(item),
            "gold_assistant": assistant_text_from_instance(item),
            "tools": item.get("tools") or [],
            "generated_token_ids": [],
            "backend_meta": {
                "projected_token_positions": [],
                "policy": "sft_warmstart_selfgen_exact_audited",
            },
        }
        for item in tool_examples
    ]
    audit_totals, audit_detail = audit_rows(tokenizer, tool_train_rows)
    write_json(projection_audit_path, {"totals": audit_totals})
    write_jsonl(projection_audit_detail_path, audit_detail)

    source_counts = Counter(item.get("source") or "unknown" for item in final_instances)
    tool_source_counts = Counter(item.get("source") or "unknown" for item in tool_examples)
    retention_source_counts = Counter(item.get("source") or "unknown" for item in retention)
    class_counts = Counter()
    for item in tool_examples:
        klass = ((item.get("sft_metadata") or {}).get("class") or {})
        class_counts["complex_value"] += int(bool(klass.get("complex_value")))
        class_counts["long_episode"] += int(bool(klass.get("long_episode")))
        class_counts["long_episode_final_stop"] += int(bool(klass.get("long_episode_final_stop")))

    manifest = {
        "created_by": "scripts/build_multiturn_sft_warmstart.py",
        "train_path": str(train_path),
        "audit_path": str(audit_path),
        "final_count": len(final_instances),
        "tool_count": len(tool_examples),
        "retention_count": len(retention),
        "mix": "50/50 tool self-generated audited turns and retention examples",
        "source_counts": dict(sorted(source_counts.items())),
        "tool_source_counts": dict(sorted(tool_source_counts.items())),
        "retention_source_counts": dict(sorted(retention_source_counts.items())),
        "tool_class_counts_after_oversampling": dict(sorted(class_counts.items())),
        "conversion": conversion_manifest,
        "leak_filter": leak_filter,
        "rollout": rollout_manifest,
        "oversampling": oversampling,
        "retention": retention_manifest,
        "projection_value_audit": audit_totals,
        "training_template": "fast_dllm_v2_native",
        "policy": {
            "source_pool": "public training rows only; eval battery paths are filtered by id/fingerprint",
            "decode": "diffusion careful + live Qwen-native structure grammar",
            "values": "raw generated model tokens; no value projection allowed",
            "keep_rule": "exact_args and audit_clean with projected_value_tokens_exact == 0",
            "oversample_focus": [
                "complex array/object-valued parameters",
                "long-episode final stop behavior",
            ],
        },
        "next_gate": {
            "sft_steps": "300-600 QLoRA steps",
            "gsm8k_retention": ">= 0.70 before RL",
            "matched20_start": "34/63 diffusion-careful",
            "matched20_promotion_bar": ">= 50/63 exact_args",
        },
        "seed": args.seed,
    }
    write_json(manifest_path, manifest)
    write_report(report_path, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
