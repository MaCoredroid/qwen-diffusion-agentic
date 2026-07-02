#!/usr/bin/env python3
"""Taxonomize diffusion-careful misses on the matched multi-turn slice.

The generated-history loop can turn an early wrong tool call into later misses.
For that bucket, this script optionally reruns each missed turn with gold prior
assistant/tool-response history and the same careful diffusion decoder. If the
turn becomes exact, it is counted as generated-history compounding.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from audit_value_projection_tokens import audit_rows  # noqa: E402
from eval_flare_multiturn_percall_waves import (  # noqa: E402
    build_episodes,
    build_schedule,
    make_gen_args,
    synthetic_tool_result,
)
from eval_flare_northstar_matched import (  # noqa: E402
    DEFAULT_AR_MODEL,
    DEFAULT_CHAT_TEMPLATE,
    DEFAULT_DIFFUSION_ADAPTER,
    DEFAULT_DIFFUSION_BASE,
    DEFAULT_INPUT,
    decode_text,
    load_chat_template,
    next_turn_user_message,
    render_matched_prompt,
    row_from_generation,
    tool_response_suffix,
    trim_scored_assistant,
)
from eval_fastdllm_toolcall_cases import full_context_sample, load_model, resolve_single_token_ids, resolve_token_ids  # noqa: E402
from eval_toolcall_jsonl import extract_tool_calls, parse_scalar_value, tool_schema_by_name  # noqa: E402
from flare_hf_cache import FlarePrefixCache  # noqa: E402


DEFAULT_MATCHED_DIR = ROOT / "runs/agentic_eval/northstar_matched_ar_vllm_vs_diffusion"
DEFAULT_OUT_DIR = DEFAULT_MATCHED_DIR / "diffusion_careful_failure_taxonomy"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def events(row: dict) -> dict:
    return ((row.get("backend_meta") or {}).get("sampler_schedule_events") or row.get("sampler_schedule_events") or {})


def value_shape_counts(calls: list[dict]) -> tuple[dict, int]:
    counts = Counter()
    max_chars = 0
    for call in calls:
        args = call.get("arguments") or {}
        if not isinstance(args, dict):
            args = {"arguments": args}
        for value in args.values():
            parsed = parse_scalar_value(value)
            if isinstance(parsed, dict):
                shape = "object"
            elif isinstance(parsed, list):
                shape = "array"
            elif isinstance(parsed, bool):
                shape = "bool"
            elif isinstance(parsed, (int, float)) and not isinstance(parsed, bool):
                shape = "number"
            else:
                shape = "string"
            counts[shape] += 1
            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            max_chars = max(max_chars, len(str(text)))
    return dict(sorted(counts.items())), max_chars


def first_mismatch(gold_calls: list[dict], pred_calls: list[dict]) -> dict:
    if len(gold_calls) != len(pred_calls):
        return {"type": "call_count", "gold": len(gold_calls), "pred": len(pred_calls)}
    for call_idx, (gold, pred) in enumerate(zip(gold_calls, pred_calls)):
        if gold.get("name") != pred.get("name"):
            return {
                "type": "tool_name",
                "call_idx": call_idx,
                "gold": gold.get("name"),
                "pred": pred.get("name"),
            }
        gold_args = gold.get("arguments") or {}
        pred_args = pred.get("arguments") or {}
        if not isinstance(gold_args, dict) or not isinstance(pred_args, dict):
            if gold_args != pred_args:
                return {"type": "arguments", "call_idx": call_idx, "gold": gold_args, "pred": pred_args}
            continue
        for key in sorted(set(gold_args) | set(pred_args)):
            if gold_args.get(key) != pred_args.get(key):
                return {
                    "type": "argument_value",
                    "call_idx": call_idx,
                    "tool": gold.get("name"),
                    "key": key,
                    "gold": gold_args.get(key),
                    "pred": pred_args.get(key),
                }
    return {"type": "unknown"}


def is_stop_or_truncation(row: dict) -> bool:
    assistant = str(row.get("assistant") or "")
    generated = int(row.get("generated_token_count") or 0)
    backend_meta = row.get("backend_meta") or {}
    max_new = int(backend_meta.get("max_new_tokens") or 384)
    return "</tool_call>" not in assistant or generated >= max_new


def classify_primary(row: dict, gold_history_exact: bool | None) -> str:
    if gold_history_exact:
        return "generated_history_compounding"
    if is_stop_or_truncation(row):
        return "stop_or_truncation"
    if not bool(row.get("valid_tool_json")) or not bool(row.get("all_schema_valid")) or not bool(row.get("all_required_args_present")):
        return "format_or_schema_error"
    if (
        not bool(row.get("exact_tool_sequence"))
        or int(row.get("extra_call_count") or 0) > 0
        or int(row.get("missing_call_count") or 0) > 0
        or int(row.get("repeated_call_count") or 0) > 0
    ):
        return "missing_extra_or_wrong_call"
    return "wrong_value_content"


def symptom_flags(row: dict) -> dict:
    return {
        "format_or_schema_error": (
            not bool(row.get("valid_tool_json"))
            or not bool(row.get("all_schema_valid"))
            or not bool(row.get("all_required_args_present"))
        ),
        "wrong_value_content": (
            bool(row.get("valid_tool_json"))
            and bool(row.get("all_schema_valid"))
            and bool(row.get("all_required_args_present"))
            and bool(row.get("exact_tool_sequence"))
            and not bool(row.get("exact_arguments"))
        ),
        "missing_extra_or_wrong_call": (
            not bool(row.get("exact_tool_sequence"))
            or int(row.get("extra_call_count") or 0) > 0
            or int(row.get("missing_call_count") or 0) > 0
            or int(row.get("repeated_call_count") or 0) > 0
        ),
        "stop_or_truncation": is_stop_or_truncation(row),
    }


def make_eval_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        input_jsonl=args.input_jsonl,
        episode_limit=args.episode_limit,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        prompt_tokenizer_path=args.prompt_tokenizer_path,
        chat_template_path=args.chat_template_path,
        ar_model_path=args.prompt_tokenizer_path,
        ar_served_model="unused",
        ar_base_url="unused",
        base_model=args.base_model,
        adapter=args.adapter,
        no_merge_adapter=args.no_merge_adapter,
        block_size=args.block_size,
        small_block_size=args.small_block_size,
        max_new_tokens=args.max_new_tokens,
        max_extra_tokens=12,
        threshold=args.threshold,
        top_p=args.top_p,
        temperature=args.temperature,
        diffusion_condition="baseline_careful",
        diffusion_structural_only=False,
        out_dir=args.out_dir,
        seed=args.seed,
    )


def build_gold_history_prompt(tokenizer, chat_template: str | None, episode: dict, target_turn_idx: int) -> str:
    prompt = render_matched_prompt(
        tokenizer,
        [dict(message) for message in episode["prompt_messages"]],
        episode["tools"],
        chat_template,
    )
    for prior_idx in range(target_turn_idx):
        gold_block = episode["gold_blocks"][prior_idx].strip()
        payload = synthetic_tool_result(gold_block, gold_block, episode["id"], prior_idx, episode["tools"])
        next_user = next_turn_user_message(episode, prior_idx + 1)
        prompt = prompt + gold_block + tool_response_suffix(payload, next_user)
    return prompt


def rerun_misses_with_gold_history(args: argparse.Namespace, misses: list[dict]) -> list[dict]:
    eval_args = make_eval_args(args)
    tokenizer = AutoTokenizer.from_pretrained(str(args.prompt_tokenizer_path), trust_remote_code=True)
    chat_template = load_chat_template(args.chat_template_path)
    episodes = build_episodes(eval_args)
    episode_by_id = {episode["id"]: episode for episode in episodes}

    model, model_tokenizer = load_model(
        args.base_model,
        args.adapter if args.adapter and args.adapter.exists() else None,
        merge_adapter=not args.no_merge_adapter,
        tokenizer_path=args.tokenizer_path,
    )
    model.eval()
    if hasattr(model, "config"):
        setattr(model.config, "bd_size", int(args.block_size))
    mask_id, stop_token_id, base_stop_token_ids = resolve_token_ids(model, model_tokenizer)
    tool_close_ids = model_tokenizer("</tool_call>", add_special_tokens=False).input_ids
    stop_token_ids = list(dict.fromkeys([int(item) for item in base_stop_token_ids + tool_close_ids]))
    argument_boundary_token_ids = resolve_single_token_ids(
        model_tokenizer, ["<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>"]
    )
    argument_newline_token_ids = resolve_single_token_ids(model_tokenizer, ["\n", "\n\n"])

    rows = []
    for miss in misses:
        episode = episode_by_id[miss["episode_id"]]
        turn_idx = int(miss["turn_idx"])
        prefix_cache = FlarePrefixCache()
        gen_args = make_gen_args(
            eval_args,
            condition="baseline_careful",
            prefix_cache=prefix_cache,
            mask_id=mask_id,
            stop_token_id=stop_token_id,
            stop_token_ids=stop_token_ids,
            argument_boundary_token_ids=argument_boundary_token_ids,
            argument_newline_token_ids=argument_newline_token_ids,
        )
        gen_args.record_projected_token_positions = True
        prompt = build_gold_history_prompt(model_tokenizer, chat_template, episode, turn_idx)
        schedule, _schedule_record, schedule_build_seconds = build_schedule(model_tokenizer, episode["gold_blocks"][turn_idx])
        prompt_input_ids = model_tokenizer([prompt], return_tensors="pt").input_ids.to("cuda")
        previous_live_tool_schemas = getattr(gen_args, "_live_tool_schemas", None)
        gen_args._live_tool_schemas = tool_schema_by_name(episode["tools"])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        try:
            with torch.no_grad():
                generated = full_context_sample(
                    model,
                    prompt_input_ids,
                    model_tokenizer,
                    gen_args,
                    sampler_schedule=schedule,
                    original_len_override=prompt_input_ids.shape[1],
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        finally:
            if previous_live_tool_schemas is None:
                try:
                    delattr(gen_args, "_live_tool_schemas")
                except AttributeError:
                    pass
            else:
                gen_args._live_tool_schemas = previous_live_tool_schemas
        new_ids = generated[prompt_input_ids.shape[1] :]
        history_text = decode_text(model_tokenizer, new_ids)
        assistant_text = trim_scored_assistant(history_text)
        row = row_from_generation(
            backend="diffusion_careful_gold_history",
            episode=episode,
            turn_idx=turn_idx,
            prompt=prompt,
            tools=episode["tools"],
            gold_block=episode["gold_blocks"][turn_idx],
            assistant_text=assistant_text,
            prompt_tokens=int(prompt_input_ids.shape[1]),
            generated_tokens=int((new_ids != gen_args.mask_id).sum().item()),
            turn_wall_seconds=time.time() - start,
            schedule_build_seconds=schedule_build_seconds,
            backend_meta={
                "sampler_schedule_events": getattr(gen_args, "_last_sampler_schedule_events", {}),
                "flare_cache_stats": getattr(gen_args, "_last_flare_cache_stats", {}),
                "flare_prefix_cache_stats": getattr(gen_args, "_last_flare_prefix_cache_stats", {}),
                "flare_timing_stats": getattr(gen_args, "_last_flare_timing_stats", {}),
                "max_new_tokens": gen_args.max_new_tokens,
                "history_mode": "gold_prior_history",
            },
        )
        row["generated_token_ids"] = [int(token_id) for token_id in new_ids.detach().cpu().tolist()]
        rows.append(row)
        print(
            f"gold-history episode={episode['id']} turn={turn_idx} exact_args={int(bool(row['exact_arguments']))} "
            f"wall={row['turn_wall_seconds']:.3f}s",
            flush=True,
        )
    return rows


def write_report(args: argparse.Namespace, taxonomy_rows: list[dict], summary: dict) -> None:
    lines = [
        "# Diffusion-Careful Matched-20 Failure Taxonomy",
        "",
        "Scope: 29 diffusion-careful misses from the matched-20 generated-history eval (`34/63` exact-args).",
        "A gold-prior-history counterfactual reruns each missed turn with the same careful decoder; if it becomes exact, the miss is counted as generated-history compounding.",
        "Never-train careful misses were skipped: no never-train diffusion-careful row exists yet, so this was not a quick taxonomy add-on.",
        "",
        "## Split",
        "",
        "| class | count |",
        "|---|---:|",
    ]
    for key, value in sorted(summary["primary_class_counts"].items()):
        lines.append(f"| {key} | {value} |")
    lines += [
        "",
        "## Symptom Flags",
        "",
        "| symptom | count |",
        "|---|---:|",
    ]
    for key, value in sorted(summary["symptom_counts"].items()):
        lines.append(f"| {key} | {value} |")
    lines += [
        "",
        "## Reward Implication",
        "",
        summary["reward_implication"],
        "",
        "## Rows",
        "",
        "| episode | turn | class | gold-history exact | symptom flags | first mismatch | value shapes |",
        "|---|---:|---|---:|---|---|---|",
    ]
    for row in taxonomy_rows:
        mismatch = row["first_mismatch"]
        mismatch_text = mismatch.get("type", "unknown")
        if mismatch.get("type") == "argument_value":
            mismatch_text += (
                f": {mismatch.get('tool')}.{mismatch.get('key')} "
                f"gold={str(mismatch.get('gold'))[:40]!r} pred={str(mismatch.get('pred'))[:40]!r}"
            )
        elif mismatch.get("type") in {"tool_name", "call_count"}:
            mismatch_text += f": gold={mismatch.get('gold')} pred={mismatch.get('pred')}"
        flags = ",".join(k for k, value in row["symptom_flags"].items() if value) or "-"
        lines.append(
            f"| {row['episode_id']} | {row['turn_idx']} | {row['primary_class']} | "
            f"{int(bool(row.get('gold_history_exact_arguments')))} | {flags} | {mismatch_text} | "
            f"{json.dumps(row['gold_value_shapes'], sort_keys=True)} |"
        )
    args.report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-dir", type=Path, default=DEFAULT_MATCHED_DIR)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--episode-limit", type=int, default=20)
    parser.add_argument("--min-turns", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--prompt-tokenizer-path", type=Path, default=DEFAULT_AR_MODEL)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_AR_MODEL)
    parser.add_argument("--chat-template-path", type=Path, default=DEFAULT_CHAT_TEMPLATE)
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
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--skip-gold-history-rerun", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_md = args.out_dir / "report.md"
    careful_rows = read_jsonl(args.matched_dir / "diffusion_careful" / "turns.jsonl")
    audit = json.loads((args.matched_dir / "diffusion_careful" / "projection_value_audit.json").read_text(encoding="utf-8"))
    if not audit.get("totals", {}).get("zero_projected_value_tokens_verified"):
        raise SystemExit("diffusion_careful audit is not clean; refusing taxonomy")
    misses = [row for row in careful_rows if not bool(row.get("exact_arguments"))]
    gold_rows = []
    if args.skip_gold_history_rerun:
        gold_rows_path = args.out_dir / "gold_history_rerun.jsonl"
        if gold_rows_path.exists():
            gold_rows = read_jsonl(gold_rows_path)
    else:
        torch.manual_seed(args.seed)
        gold_rows = rerun_misses_with_gold_history(args, misses)
        write_jsonl(args.out_dir / "gold_history_rerun.jsonl", gold_rows)
        gold_audit_totals, gold_audit_rows = audit_rows(
            AutoTokenizer.from_pretrained(str(args.tokenizer_path), trust_remote_code=True),
            gold_rows,
        )
        write_json(args.out_dir / "gold_history_projection_value_audit.json", {"totals": gold_audit_totals})
        write_jsonl(args.out_dir / "gold_history_projection_value_audit.jsonl", gold_audit_rows)
    gold_by_key = {(row["episode_id"], row["turn_idx"]): row for row in gold_rows}

    taxonomy_rows = []
    primary_counts = Counter()
    symptom_counts = Counter()
    for row in misses:
        key = (row["episode_id"], row["turn_idx"])
        gold_calls, _ = extract_tool_calls(row.get("gold_assistant") or "")
        pred_calls = row.get("calls") or []
        shapes, max_value_chars = value_shape_counts(gold_calls)
        gold_history_row = gold_by_key.get(key)
        gold_history_exact = bool(gold_history_row.get("exact_arguments")) if gold_history_row else None
        primary = classify_primary(row, gold_history_exact)
        flags = symptom_flags(row)
        for flag, value in flags.items():
            symptom_counts[flag] += int(bool(value))
        primary_counts[primary] += 1
        taxonomy_rows.append(
            {
                "episode_id": row["episode_id"],
                "turn_idx": int(row["turn_idx"]),
                "turns_in_episode": int(row.get("turns_in_episode") or 0),
                "primary_class": primary,
                "symptom_flags": flags,
                "gold_history_exact_arguments": gold_history_exact,
                "gold_history_valid_tool_json": bool(gold_history_row.get("valid_tool_json")) if gold_history_row else None,
                "gold_history_all_schema_valid": bool(gold_history_row.get("all_schema_valid")) if gold_history_row else None,
                "gold_history_assistant": gold_history_row.get("assistant") if gold_history_row else None,
                "valid_tool_json": bool(row.get("valid_tool_json")),
                "all_schema_valid": bool(row.get("all_schema_valid")),
                "all_required_args_present": bool(row.get("all_required_args_present")),
                "exact_tool_sequence": bool(row.get("exact_tool_sequence")),
                "called_names": row.get("called_names") or [],
                "first_mismatch": first_mismatch(gold_calls, pred_calls),
                "gold_value_shapes": shapes,
                "gold_max_value_chars": max_value_chars,
                "generated_token_count": int(row.get("generated_token_count") or 0),
                "denoise_forwards_total": int(events(row).get("denoise_forwards_total") or 0),
                "turn_wall_seconds": float(row.get("turn_wall_seconds") or 0.0),
            }
        )

    train_fixable = sum(primary_counts[key] for key in ("format_or_schema_error", "wrong_value_content", "missing_extra_or_wrong_call"))
    reward_implication = (
        f"Primary miss mass is {dict(sorted(primary_counts.items()))}. "
        f"Use a ToolRL-style graded reward with explicit format/schema/name/arg-name/value terms; "
        f"add episode-level credit for generated-history compounding ({primary_counts.get('generated_history_compounding', 0)} turns), "
        f"and keep exact-args as the promotion gate. Train-fixable direct current-turn misses: {train_fixable}/29."
    )
    summary = {
        "input_rows": str(args.matched_dir / "diffusion_careful" / "turns.jsonl"),
        "audit": audit.get("totals", {}),
        "misses": len(misses),
        "exact_args": sum(bool(row.get("exact_arguments")) for row in careful_rows),
        "total_turns": len(careful_rows),
        "primary_class_counts": dict(sorted(primary_counts.items())),
        "symptom_counts": dict(sorted(symptom_counts.items())),
        "gold_history_rerun_turns": len(gold_rows),
        "gold_history_exact_args": sum(bool(row.get("exact_arguments")) for row in gold_rows),
        "reward_implication": reward_implication,
    }
    write_jsonl(args.out_dir / "taxonomy.jsonl", taxonomy_rows)
    write_json(args.out_dir / "summary.json", summary)
    write_report(args, taxonomy_rows, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
