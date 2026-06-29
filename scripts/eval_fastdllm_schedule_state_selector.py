#!/usr/bin/env python3
"""Evaluate generated schedule-state selector JSON for agentic diffusion.

The schedule-state selector objective asks the diffusion model to emit a compact
sampler decision, not the tool argument value itself. This evaluator checks
whether a checkpoint can produce parseable decision JSON and whether the
candidate/protection fields match the curriculum label.
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch


ROOT = Path("/home/mark/qwen_diffusion")
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from eval_fastdllm_toolcall_cases import (  # noqa: E402
    apply_chat_template,
    load_model,
    resolve_chat_template,
    resolve_token_ids,
)


DEFAULT_BASE = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_DATASET = ROOT / "data/qwen35_9b_schedule_state_selector_no_public_smoke_curriculum/train_agentic_mix.json"
DEFAULT_OUT = ROOT / "runs/schedule_state_selector/no_public_smoke_selector_eval.jsonl"
DEFAULT_ADAPTER = ROOT / "runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model"

DECISION_KEYS = [
    "candidate_index",
    "span_kind",
    "protection",
    "block_size",
    "denoise_steps",
    "force_candidate_sequence",
    "require_json_prefix_safe",
    "close_tool_call_only_when_json_complete",
]
POLICY_KEYS = [key for key in DECISION_KEYS if key != "candidate_index"]


def load_instances(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("instances"), list):
        return payload["instances"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported dataset shape in {path}")


def messages_before_answer(instance: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    system = instance.get("system")
    if system:
        messages.append({"role": "system", "content": str(system)})
    for message in instance.get("messages") or []:
        role = message.get("role")
        if role == "assistant":
            break
        messages.append({"role": str(role), "content": str(message.get("content") or "")})
    return messages


def expected_decision(instance: dict[str, Any]) -> dict[str, Any]:
    for message in instance.get("messages") or []:
        if message.get("role") == "assistant":
            return json.loads(message.get("content") or "{}")
    raise ValueError("instance has no assistant label")


def user_text(instance: dict[str, Any]) -> str:
    for message in instance.get("messages") or []:
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def parse_int_field(pattern: str, text: str, flags: int = re.MULTILINE) -> int | None:
    match = re.search(pattern, text, flags=flags)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def parse_str_field(pattern: str, text: str, flags: int = re.MULTILINE) -> str | None:
    match = re.search(pattern, text, flags=flags)
    return match.group(1).strip() if match else None


def infer_metadata(instance: dict[str, Any], idx: int, expected: dict[str, Any]) -> dict[str, Any]:
    text = user_text(instance)
    return {
        "idx": idx,
        "case_id": parse_str_field(r"^Case id:\s*(.+)$", text),
        "tool_call_index": parse_int_field(r"^Tool call index:\s*(\d+)\s*$", text),
        "json_key": parse_str_field(r"^JSON key:\s*(.+)$", text),
        "json_path": parse_str_field(r"^JSON path:\s*(.+)$", text),
        "target_token_count": parse_int_field(r"^Target token count:\s*(\d+)\s*$", text),
        "candidate_count": parse_int_field(r"^Candidate count:\s*(\d+)\s*$", text),
        "target_index": expected.get("candidate_index"),
    }


def parse_decision(text: str) -> tuple[dict[str, Any] | None, str | None]:
    stripped = text.strip()
    attempts = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        attempts.append(stripped[start : end + 1])
    for candidate in attempts:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed, None

    fallback: dict[str, Any] = {}
    index_match = re.search(r'"?candidate_index"?\s*[:=]\s*(-?\d+)', stripped)
    if index_match:
        fallback["candidate_index"] = int(index_match.group(1))
    else:
        first_int = re.search(r"-?\d+", stripped)
        if first_int:
            fallback["candidate_index"] = int(first_int.group(0))
    for key in ("span_kind", "protection"):
        match = re.search(rf'"?{re.escape(key)}"?\s*[:=]\s*"([^"]+)"', stripped)
        if match:
            fallback[key] = match.group(1)
    for key in ("block_size", "denoise_steps"):
        match = re.search(rf'"?{re.escape(key)}"?\s*[:=]\s*(-?\d+)', stripped)
        if match:
            fallback[key] = int(match.group(1))
    for key in ("force_candidate_sequence", "require_json_prefix_safe", "close_tool_call_only_when_json_complete"):
        match = re.search(rf'"?{re.escape(key)}"?\s*[:=]\s*(true|false|True|False)', stripped)
        if match:
            fallback[key] = match.group(1).lower() == "true"
    if fallback:
        return fallback, "json_parse_failed_regex_fallback"
    return None, "json_parse_failed"


def normalized_decision(decision: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(decision, dict):
        return {}
    normalized: dict[str, Any] = {}
    for key in DECISION_KEYS:
        if key not in decision:
            continue
        value = decision[key]
        if key in ("candidate_index", "block_size", "denoise_steps"):
            try:
                normalized[key] = int(value)
            except (TypeError, ValueError):
                normalized[key] = value
        elif key in (
            "force_candidate_sequence",
            "require_json_prefix_safe",
            "close_tool_call_only_when_json_complete",
        ):
            if isinstance(value, str):
                normalized[key] = value.strip().lower() == "true"
            else:
                normalized[key] = bool(value)
        elif isinstance(value, str):
            normalized[key] = value.strip()
        else:
            normalized[key] = value
    return normalized


def field_metrics(expected: dict[str, Any], parsed: dict[str, Any], valid_json: bool) -> dict[str, bool]:
    metrics: dict[str, bool] = {"valid_json": valid_json}
    for key in DECISION_KEYS:
        metrics[f"{key}_exact"] = parsed.get(key) == expected.get(key)
    metrics["policy_exact"] = all(metrics[f"{key}_exact"] for key in POLICY_KEYS)
    metrics["decision_exact"] = all(metrics[f"{key}_exact"] for key in DECISION_KEYS)
    return metrics


def build_prompt(tokenizer, instance: dict[str, Any], chat_template):
    return apply_chat_template(tokenizer, messages_before_answer(instance), None, chat_template=chat_template)


def generate_decision(model, tokenizer, prompt: str, args, mask_id: int, stop_token_id: int):
    input_ids = tokenizer([prompt], return_tensors="pt").input_ids.to("cuda")
    seq_len = torch.tensor([input_ids.shape[1]], device="cuda")
    with torch.no_grad():
        generated = model.mdm_sample(
            input_ids,
            tokenizer=tokenizer,
            block_size=args.block_size,
            small_block_size=args.small_block_size,
            max_new_tokens=args.max_new_tokens,
            mask_id=mask_id,
            stop_token=stop_token_id,
            min_len=input_ids.shape[1],
            seq_len=seq_len,
            threshold=args.threshold,
            temperature=args.temperature,
            top_p=args.top_p,
            use_block_cache=args.use_block_cache,
        )[0]
    new_ids = generated[input_ids.shape[1] :]
    mask_count = int((new_ids == mask_id).sum().item())
    generated_token_count = int((new_ids != mask_id).sum().item())
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return text, mask_count, generated_token_count


def summarize_rate(totals: Counter, key: str) -> float:
    examples = totals["examples"]
    return totals[key] / examples if examples else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--dataset-json", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--conversation-template", default="fast_dllm_v2")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--only-ambiguous", action="store_true")
    parser.add_argument("--no-merge-adapter", action="store_true")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--small-block-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--use-block-cache", action="store_true")
    args = parser.parse_args()

    chat_template = resolve_chat_template(args.conversation_template)
    adapter_path = None if args.no_adapter else args.adapter
    model, tokenizer = load_model(
        str(args.base_model),
        str(adapter_path) if adapter_path else None,
        merge_adapter=not args.no_merge_adapter,
        tokenizer_path=str(args.tokenizer_path) if args.tokenizer_path else None,
    )
    mask_id, stop_token_id = resolve_token_ids(model, tokenizer)

    selected = []
    for source_idx, instance in enumerate(load_instances(args.dataset_json)):
        expected = expected_decision(instance)
        meta = infer_metadata(instance, source_idx, expected)
        if args.only_ambiguous and int(meta.get("candidate_count") or 0) <= 1:
            continue
        selected.append((source_idx, instance, expected, meta))
    if args.offset > 0:
        selected = selected[args.offset :]
    if args.limit and args.limit > 0:
        selected = selected[: args.limit]

    totals = Counter()
    start_time = time.time()
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for eval_idx, (source_idx, instance, expected_raw, meta) in enumerate(selected):
            expected = normalized_decision(expected_raw)
            row = {
                "eval_idx": eval_idx,
                "source_idx": source_idx,
                **meta,
                "expected": expected,
                "status": "ok",
            }
            try:
                prompt = build_prompt(tokenizer, instance, chat_template)
                text, mask_count, generated_token_count = generate_decision(
                    model,
                    tokenizer,
                    prompt,
                    args,
                    mask_id,
                    stop_token_id,
                )
                parsed_raw, parse_error = parse_decision(text)
                valid_json = parse_error is None
                parsed = normalized_decision(parsed_raw)
                metrics = field_metrics(expected, parsed, valid_json)
                row.update(
                    {
                        "generated_text": text,
                        "parsed": parsed,
                        "parse_error": parse_error,
                        "mask_count": mask_count,
                        "generated_token_count": generated_token_count,
                        **metrics,
                    }
                )
                totals["examples"] += 1
                for key, value in metrics.items():
                    totals[key] += int(value)
                if meta.get("candidate_count") is not None:
                    family = "ambiguous" if int(meta["candidate_count"]) > 1 else "singleton"
                    totals[f"examples:{family}"] += 1
                    totals[f"candidate_index_exact:{family}"] += int(metrics["candidate_index_exact"])
                    totals[f"decision_exact:{family}"] += int(metrics["decision_exact"])
            except Exception as exc:
                row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
                totals["errors"] += 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"selector-eval {eval_idx + 1}/{len(selected)} ok={totals['examples']} "
                f"idx={totals['candidate_index_exact']} json={totals['valid_json']} "
                f"decision={totals['decision_exact']} errors={totals['errors']}",
                flush=True,
            )

    elapsed = time.time() - start_time
    rates = {
        "valid_json_rate": summarize_rate(totals, "valid_json"),
        "candidate_index_accuracy": summarize_rate(totals, "candidate_index_exact"),
        "policy_exact_rate": summarize_rate(totals, "policy_exact"),
        "decision_exact_rate": summarize_rate(totals, "decision_exact"),
    }
    for key in DECISION_KEYS:
        rates[f"{key}_exact_rate"] = summarize_rate(totals, f"{key}_exact")
    summary = {
        "base_model": str(args.base_model),
        "adapter": str(adapter_path) if adapter_path else None,
        "merge_adapter": not args.no_merge_adapter,
        "tokenizer_path": str(args.tokenizer_path),
        "dataset_json": str(args.dataset_json),
        "out_jsonl": str(args.out_jsonl),
        "conversation_template": args.conversation_template,
        "limit": args.limit,
        "offset": args.offset,
        "only_ambiguous": args.only_ambiguous,
        "generation": {
            "block_size": args.block_size,
            "small_block_size": args.small_block_size,
            "max_new_tokens": args.max_new_tokens,
            "threshold": args.threshold,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "use_block_cache": args.use_block_cache,
        },
        "totals": dict(totals),
        "rates": rates,
        "elapsed_seconds": elapsed,
        "examples_per_second": totals["examples"] / elapsed if elapsed else 0.0,
        "mask_id": mask_id,
        "stop_token_id": stop_token_id,
    }
    if torch.cuda.is_available():
        summary["cuda_max_memory_allocated_gib"] = torch.cuda.max_memory_allocated() / (1024**3)
        summary["cuda_max_memory_reserved_gib"] = torch.cuda.max_memory_reserved() / (1024**3)
    args.out_jsonl.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
