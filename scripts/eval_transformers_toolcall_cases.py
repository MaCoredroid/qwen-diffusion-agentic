#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

from eval_toolcall_jsonl import score_tool_calls


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_INPUT = ROOT / "data/toolcall_eval/synthetic_toolresult_smoke.jsonl"
DEFAULT_OUT = ROOT / "data/toolcall_eval/qwen35_9b_transformers_toolresult.jsonl"


def load_cases(path, limit):
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
            if limit and len(cases) >= limit:
                break
    return cases


def generation_instruction(case):
    return case.get("teacher_instruction") or (
        "Return the necessary Qwen tool call or calls for the request above. "
        "Use only this format and no prose:\n"
        "<tool_call>\n"
        "{\"name\": \"tool_name\", \"arguments\": {}}\n"
        "</tool_call>"
    )


def apply_chat_template(tokenizer, messages, tools, enable_thinking):
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if tools:
        kwargs["tools"] = tools
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=enable_thinking,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def load_model(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    dtype = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.torch_dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": args.device_map,
        "torch_dtype": dtype,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        )

    errors = []
    for cls in (AutoModelForImageTextToText, AutoModelForCausalLM):
        try:
            model = cls.from_pretrained(args.model, **model_kwargs)
            model.eval()
            return tokenizer, model
        except Exception as exc:
            errors.append(f"{cls.__name__}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Could not load model:\n" + "\n".join(errors))


def first_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return "cuda"


def generate_case(tokenizer, model, case, args):
    import torch

    messages = list(case["prompt_messages"])
    messages.append({"role": "user", "content": generation_instruction(case)})
    prompt = apply_chat_template(
        tokenizer,
        messages,
        case.get("tools") or None,
        args.enable_thinking,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    device = first_device(model)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p
    with torch.inference_mode():
        output = model.generate(**inputs, **generation_kwargs)
    new_tokens = output[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


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
        "errors": 0,
    }


def parse_eval_spec(spec):
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


def run_eval(tokenizer, model, args, eval_name, input_jsonl, out_jsonl, limit):
    cases = load_cases(input_jsonl, limit)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    totals = empty_totals()
    start = time.time()
    with out_jsonl.open("w", encoding="utf-8") as f:
        for idx, case in enumerate(cases):
            row = {
                "idx": idx,
                "source": case.get("source"),
                "id": case.get("id"),
                "gold_tool_names": case.get("gold_tool_names") or [],
                "available_tool_names": case.get("available_tool_names") or [],
            }
            try:
                text = generate_case(tokenizer, model, case, args)
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
                    }
                )
                totals["ok"] += 1
                totals["valid_tool_json"] += int(row["valid_tool_json"])
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
            except Exception as exc:
                row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
                totals["errors"] += 1
            totals["records"] += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    summary = {
        "eval_name": eval_name,
        "input_jsonl": str(input_jsonl),
        "out_jsonl": str(out_jsonl),
        "model": args.model,
        "load_in_4bit": args.load_in_4bit,
        "torch_dtype": args.torch_dtype,
        "device_map": args.device_map,
        "attn_implementation": args.attn_implementation,
        "totals": totals,
        "elapsed_seconds": time.time() - start,
    }
    try:
        import torch

        if torch.cuda.is_available():
            summary["cuda_max_memory_allocated_gib"] = torch.cuda.max_memory_allocated() / (1024**3)
            summary["cuda_max_memory_reserved_gib"] = torch.cuda.max_memory_reserved() / (1024**3)
    except Exception:
        pass
    summary_path = out_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--eval",
        dest="eval_specs",
        action="append",
        type=parse_eval_spec,
        default=[],
        help="Run an eval after a single model load: name:input_jsonl:out_jsonl[:limit]",
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", default="nf4", choices=["nf4", "fp4"])
    parser.add_argument("--bnb-4bit-use-double-quant", action="store_true")
    parser.add_argument("--torch-dtype", default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="")
    args = parser.parse_args()

    tokenizer, model = load_model(args)
    eval_specs = args.eval_specs or [("default", args.input_jsonl, args.out_jsonl, args.limit)]
    summaries = [
        run_eval(tokenizer, model, args, eval_name, input_jsonl, out_jsonl, limit)
        for eval_name, input_jsonl, out_jsonl, limit in eval_specs
    ]
    if len(summaries) > 1:
        print(json.dumps({"suite": [item["out_jsonl"] for item in summaries]}, indent=2))


if __name__ == "__main__":
    main()
