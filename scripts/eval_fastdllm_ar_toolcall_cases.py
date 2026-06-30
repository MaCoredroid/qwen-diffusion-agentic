#!/usr/bin/env python3
"""Evaluate Fast-dLLM bridge weights in causal AR mode on tool-call cases."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path("/home/mark/qwen_diffusion")
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from eval_fastdllm_toolcall_cases import (  # noqa: E402
    contiguous_decoded_prefix,
    live_tool_json_top_token,
    make_prompt,
    native_tool_prefix_can_stop,
    resolve_chat_template,
    resolve_token_ids,
    tool_json_live_prefix_active,
)
from eval_toolcall_jsonl import score_tool_calls, tool_schema_by_name  # noqa: E402


DEFAULT_BASE = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_ADAPTER = ROOT / "runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000"
DEFAULT_INPUT = ROOT / "data/toolcall_eval_native/public_onecall_qwen_native_smoke.jsonl"
DEFAULT_OUT = ROOT / "runs/ar_vs_diffusion_native_baseline/ar_raw/public_onecall_24.jsonl"


def load_cases(path: Path, limit: int):
    cases = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                cases.append(json.loads(line))
            if limit and len(cases) >= limit:
                break
    return cases


def empty_totals():
    return {
        "records": 0,
        "ok": 0,
        "valid_tool_json": 0,
        "exact_tool_name_set": 0,
        "exact_tool_sequence": 0,
        "exact_tool_name_multiset": 0,
        "same_tool_call_count": 0,
        "exact_arguments": 0,
        "all_schema_valid": 0,
        "all_required_args_present": 0,
        "records_with_extra_calls": 0,
        "records_with_missing_calls": 0,
        "records_with_repeated_calls": 0,
        "total_extra_calls": 0,
        "total_missing_calls": 0,
        "total_repeated_calls": 0,
        "live_tool_json_grammar_active_steps": 0,
        "live_tool_json_grammar_replacement_steps": 0,
        "live_tool_json_grammar_unsafe_fallback_steps": 0,
        "errors": 0,
    }


def parse_eval_spec(spec: str):
    parts = spec.split(":")
    if len(parts) not in {3, 4}:
        raise argparse.ArgumentTypeError(
            "--eval must be name:input_jsonl:out_jsonl or name:input_jsonl:out_jsonl:limit"
        )
    name, input_jsonl, out_jsonl = parts[:3]
    limit = int(parts[3]) if len(parts) == 4 and parts[3] else 0
    if not name:
        raise argparse.ArgumentTypeError("--eval name cannot be empty")
    return name, Path(input_jsonl), Path(out_jsonl), limit


def add_metrics_to_totals(totals, row):
    totals["valid_tool_json"] += int(bool(row["valid_tool_json"]))
    totals["exact_tool_name_set"] += int(bool(row["exact_tool_name_set"]))
    totals["exact_tool_sequence"] += int(bool(row["exact_tool_sequence"]))
    totals["exact_tool_name_multiset"] += int(bool(row["exact_tool_name_multiset"]))
    totals["same_tool_call_count"] += int(bool(row["same_tool_call_count"]))
    totals["exact_arguments"] += int(bool(row["exact_arguments"]))
    totals["all_schema_valid"] += int(bool(row["all_schema_valid"]))
    totals["all_required_args_present"] += int(bool(row["all_required_args_present"]))
    totals["records_with_extra_calls"] += int((row["extra_call_count"] or 0) > 0)
    totals["records_with_missing_calls"] += int((row["missing_call_count"] or 0) > 0)
    totals["records_with_repeated_calls"] += int((row["repeated_call_count"] or 0) > 0)
    totals["total_extra_calls"] += int(row["extra_call_count"] or 0)
    totals["total_missing_calls"] += int(row["missing_call_count"] or 0)
    totals["total_repeated_calls"] += int(row["repeated_call_count"] or 0)
    events = row.get("live_tool_json_grammar_events") or {}
    totals["live_tool_json_grammar_active_steps"] += int(events.get("active_steps") or 0)
    totals["live_tool_json_grammar_replacement_steps"] += int(events.get("replacement_steps") or 0)
    totals["live_tool_json_grammar_unsafe_fallback_steps"] += int(events.get("unsafe_fallback_steps") or 0)


def load_model(args):
    tokenizer_path = str(args.tokenizer_path or args.adapter or args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        str(args.base_model),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(base, str(args.adapter))
        if not args.no_merge_adapter:
            model = model.merge_and_unload()
    else:
        model = base
    model.to("cuda").eval()
    return model, tokenizer


def greedy_ar_generate(model, tokenizer, prompt_input_ids, input_ids, case, args):
    stop_ids = set(int(token_id) for token_id in args.stop_token_ids)
    original_len = int(prompt_input_ids.shape[1])
    events = {"active_steps": 0, "replacement_steps": 0, "unsafe_fallback_steps": 0}
    schemas = tool_schema_by_name(case.get("tools") or [])

    for _ in range(args.max_new_tokens - max(0, input_ids.shape[1] - original_len)):
        with torch.no_grad():
            logits = model(input_ids=input_ids, use_cache=False).logits[:, -1, :].float()[0].clone()
        if args.ban_mask_token and args.mask_id is not None:
            logits[int(args.mask_id)] = -torch.inf

        token_id = int(torch.argmax(logits).item())
        if args.live_tool_native_grammar:
            generated_ids = input_ids[0, original_len:].detach().tolist()
            text = contiguous_decoded_prefix(tokenizer, generated_ids, args.mask_id)
            if tool_json_live_prefix_active(text):
                events["active_steps"] += 1
                sequence = torch.cat(
                    [
                        input_ids[0],
                        torch.tensor([args.mask_id], dtype=input_ids.dtype, device=input_ids.device),
                    ],
                    dim=0,
                )
                abs_idx = int(input_ids.shape[1])
                if token_id in stop_ids and native_tool_prefix_can_stop(text, schemas=schemas):
                    safe = True
                else:
                    selected, safe = live_tool_json_top_token(
                        tokenizer,
                        sequence,
                        logits,
                        original_len,
                        abs_idx,
                        args.mask_id,
                        args.live_tool_json_topk,
                        schemas=schemas,
                    )
                    if int(selected) != token_id:
                        events["replacement_steps"] += 1
                    token_id = int(selected)
                if not safe:
                    events["unsafe_fallback_steps"] += 1

        next_token = torch.tensor([[token_id]], dtype=input_ids.dtype, device=input_ids.device)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        if token_id in stop_ids:
            break
    return input_ids[0], events


def generate_case(model, tokenizer, case, args):
    generation_case = case
    if args.strip_gold_for_generation:
        generation_case = copy.deepcopy(case)
        for gold_key in ("gold_assistant", "gold_tool_names", "gold_tool_calls"):
            generation_case.pop(gold_key, None)
    prompt = make_prompt(tokenizer, generation_case, args.append_instruction, chat_template=args.chat_template)
    prompt_input_ids = tokenizer([prompt], return_tensors="pt").input_ids.to("cuda")
    input_ids = prompt_input_ids
    forced_prefix = args.forced_assistant_prefix or ""
    if args.force_tool_call_prefix:
        forced_prefix = "<tool_call>\n" + forced_prefix
    if forced_prefix:
        prefix_ids = tokenizer(forced_prefix, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
        input_ids = torch.cat([prompt_input_ids, prefix_ids], dim=1)
    generated, events = greedy_ar_generate(model, tokenizer, prompt_input_ids, input_ids, case, args)
    new_ids = generated[prompt_input_ids.shape[1] :]
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return text, int(new_ids.shape[0]), events


def run_eval(model, tokenizer, args, eval_name, input_jsonl, out_jsonl, limit):
    cases = load_cases(input_jsonl, limit)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    totals = empty_totals()
    generated_tokens = 0
    start = time.time()

    with out_jsonl.open("w", encoding="utf-8") as handle:
        for idx, case in enumerate(cases):
            row = {
                "idx": idx,
                "source": case.get("source"),
                "id": case.get("id"),
                "gold_tool_names": case.get("gold_tool_names") or [],
                "available_tool_names": case.get("available_tool_names") or [],
            }
            try:
                sample_start = time.time()
                text, token_count, events = generate_case(model, tokenizer, case, args)
                metrics = score_tool_calls(text, case.get("tools") or [], case.get("gold_assistant"))
                row.update(
                    {
                        "status": "ok",
                        "assistant": text,
                        "called_names": metrics["called_names"],
                        "calls": metrics["calls"],
                        "invalid_tool_json_count": metrics["invalid_tool_call_count"],
                        "valid_tool_json": metrics["valid_tool_call"],
                        "valid_tool_call": metrics["valid_tool_call"],
                        "exact_tool_name_set": metrics.get("exact_tool_name_set"),
                        "exact_tool_name_multiset": metrics.get("exact_tool_name_multiset"),
                        "exact_tool_sequence": metrics.get("exact_tool_sequence"),
                        "same_tool_call_count": metrics.get("same_tool_call_count"),
                        "exact_arguments": metrics.get("exact_arguments"),
                        "all_schema_valid": metrics["all_schema_valid"],
                        "all_required_args_present": metrics["all_required_args_present"],
                        "schema_valid_count": metrics["schema_valid_count"],
                        "required_args_count": metrics["required_args_count"],
                        "extra_call_count": metrics.get("extra_call_count"),
                        "missing_call_count": metrics.get("missing_call_count"),
                        "repeated_call_count": metrics.get("repeated_call_count"),
                        "extra_call_names": metrics.get("extra_call_names"),
                        "missing_call_names": metrics.get("missing_call_names"),
                        "repeated_call_names": metrics.get("repeated_call_names"),
                        "call_errors": metrics["call_errors"],
                        "generated_token_count": token_count,
                        "seconds": time.time() - sample_start,
                        "forced_assistant_prefix": (
                            ("<tool_call>\n" if args.force_tool_call_prefix else "")
                            + (args.forced_assistant_prefix or "")
                        ),
                        "live_tool_json_grammar_events": events,
                    }
                )
                totals["ok"] += 1
                add_metrics_to_totals(totals, row)
                generated_tokens += token_count
            except Exception as exc:
                row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
                totals["errors"] += 1
            totals["records"] += 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"{eval_name} {idx + 1}/{len(cases)} "
                f"seq={totals['exact_tool_sequence']} args={totals['exact_arguments']} "
                f"valid={totals['valid_tool_json']}",
                flush=True,
            )

    elapsed = time.time() - start
    summary = {
        "eval_name": eval_name,
        "input_jsonl": str(input_jsonl),
        "out_jsonl": str(out_jsonl),
        "base_model": str(args.base_model),
        "adapter": str(args.adapter) if args.adapter else None,
        "merge_adapter": not args.no_merge_adapter,
        "mode": "fastdllm_causal",
        "live_tool_native_grammar": args.live_tool_native_grammar,
        "live_tool_json_topk": args.live_tool_json_topk,
        "strip_gold_for_generation": args.strip_gold_for_generation,
        "force_tool_call_prefix": args.force_tool_call_prefix,
        "forced_assistant_prefix": args.forced_assistant_prefix,
        "append_instruction": args.append_instruction,
        "max_new_tokens": args.max_new_tokens,
        "ban_mask_token": args.ban_mask_token,
        "totals": totals,
        "elapsed_seconds": elapsed,
        "generated_tokens": generated_tokens,
        "generated_tokens_per_second": generated_tokens / elapsed if elapsed else 0.0,
        "mask_id": args.mask_id,
        "stop_token_ids": args.stop_token_ids,
        "conversation_template": args.conversation_template,
    }
    if torch.cuda.is_available():
        summary["cuda_max_memory_allocated_gib"] = torch.cuda.max_memory_allocated() / (1024**3)
        summary["cuda_max_memory_reserved_gib"] = torch.cuda.max_memory_reserved() / (1024**3)
    summary_path = out_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--eval", dest="eval_specs", action="append", type=parse_eval_spec, default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=640)
    parser.add_argument("--append-instruction", action="store_true")
    parser.add_argument("--conversation-template", default=None)
    parser.add_argument("--strip-gold-for-generation", action="store_true")
    parser.add_argument("--force-tool-call-prefix", action="store_true")
    parser.add_argument("--forced-assistant-prefix", default="")
    parser.add_argument("--live-tool-native-grammar", action="store_true")
    parser.add_argument("--live-tool-json-topk", type=int, default=1024)
    parser.add_argument("--ban-mask-token", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-merge-adapter", action="store_true")
    args = parser.parse_args()
    args.chat_template = resolve_chat_template(args.conversation_template)

    model, tokenizer = load_model(args)
    args.mask_id, args.stop_token_id, args.stop_token_ids = resolve_token_ids(model, tokenizer)
    print(
        "[token_ids] "
        + json.dumps({"mask_id": args.mask_id, "stop_token_ids": args.stop_token_ids}, sort_keys=True),
        flush=True,
    )
    eval_specs = args.eval_specs or [("default", args.input_jsonl, args.out_jsonl, args.limit)]
    summaries = [
        run_eval(model, tokenizer, args, eval_name, input_jsonl, out_jsonl, limit)
        for eval_name, input_jsonl, out_jsonl, limit in eval_specs
    ]
    if len(summaries) > 1:
        print(json.dumps({"suite": [item["out_jsonl"] for item in summaries]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
