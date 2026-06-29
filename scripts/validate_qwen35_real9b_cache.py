#!/usr/bin/env python3
"""Real 9B token-exact validation for the Qwen3.5 state-cache sampler."""

from __future__ import annotations

import argparse
import gc
import importlib
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT / "fast-dllm" / "v2") not in sys.path:
    sys.path.insert(0, str(ROOT / "fast-dllm" / "v2"))

from validate_qwen35_state_cache_sampler import (  # noqa: E402
    DummyTokenizer,
    cached_full_context_sample,
    cached_generation_functions_sample,
)


class DeviceProxy:
    def __init__(self, model, device):
        self._model = model
        self.device = torch.device(device)

    def __getattr__(self, name):
        return getattr(self._model, name)

    def forward(self, *args, **kwargs):
        return self._model.forward(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/qwen3.5-9b-fastdllm-init")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt-limit", type=int, default=3)
    parser.add_argument("--max-new-tokens", default="32,64")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--small-block-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--skip-generation-entry", action="store_true")
    parser.add_argument(
        "--strict-matmul-precision",
        action="store_true",
        help="Disable TF32/reduced reductions. Default leaves serving math unchanged.",
    )
    return parser.parse_args()


def make_args(model, tokenizer, max_new_tokens, cli_args):
    mask_id = getattr(model.config, "mask_token_id", None)
    if mask_id is None:
        mask_id = tokenizer.convert_tokens_to_ids("|<MASK>|")
    stop_token_id = getattr(model.config, "eos_token_id", None) or tokenizer.eos_token_id
    if isinstance(stop_token_id, (list, tuple)):
        stop_token_id = stop_token_id[0]
    return SimpleNamespace(
        max_new_tokens=int(max_new_tokens),
        block_size=int(cli_args.block_size),
        small_block_size=int(cli_args.small_block_size),
        mask_id=int(mask_id),
        stop_token_id=int(stop_token_id),
        threshold=float(cli_args.threshold),
        temperature=float(cli_args.temperature),
        top_p=float(cli_args.top_p),
        _last_sampler_schedule_events={},
        guard_tool_json_prefix=False,
        json_prefix_guard_kinds=set(),
    )


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def timed_call(device, fn, *args, **kwargs):
    sync(device)
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    sync(device)
    return result, time.perf_counter() - start


def full_context_golden(model, tokenizer, input_ids, args):
    toolcall_module = importlib.import_module("eval_fastdllm_toolcall_cases")
    return toolcall_module.full_context_sample(
        model,
        input_ids,
        tokenizer,
        args,
        sampler_schedule=None,
    )


def generation_golden(model, tokenizer, input_ids, args, device):
    generation_functions = importlib.import_module("generation_functions")
    proxy = DeviceProxy(model, device)
    return generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample(
        proxy,
        input_ids,
        tokenizer=tokenizer,
        block_size=args.block_size,
        small_block_size=args.small_block_size,
        max_new_tokens=args.max_new_tokens,
        mask_id=args.mask_id,
        min_len=input_ids.shape[1],
        seq_len=torch.tensor([input_ids.shape[1]], device=input_ids.device),
        use_block_cache=False,
        threshold=args.threshold,
        stop_token=args.stop_token_id,
        top_p=args.top_p,
        temperature=args.temperature,
    )[0]


def summarize_ids(tokenizer, ids, prompt_len):
    new_ids = ids[prompt_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip().replace("\n", "\\n")[:160]


def main() -> int:
    cli_args = parse_args()
    if cli_args.strict_matmul_precision:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.set_float32_matmul_precision("highest")
    device = torch.device(cli_args.device)
    model_dir = (ROOT / cli_args.model_dir).resolve()
    max_new_values = [int(item) for item in cli_args.max_new_tokens.replace(";", ",").split(",") if item.strip()]
    prompts = [
        "Answer in one sentence: what is 2 + 2?",
        "Write a JSON object with key city and value Paris.",
        "List two colors separated by a comma.",
    ][: cli_args.prompt_limit]

    print("Real 9B Qwen3.5 state-cache validation")
    print(f"model_dir={model_dir}")
    print(f"device={device} prompts={len(prompts)} max_new_tokens={max_new_values} atol={cli_args.atol}")
    print("loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": str(device)},
    )
    model.eval()
    modeling_module = importlib.import_module(type(model).__module__)
    print("model loaded")

    all_ok = True
    speedups = []
    generation_speedups = []
    with torch.inference_mode():
        for max_new in max_new_values:
            args = make_args(model, tokenizer, max_new, cli_args)
            for prompt_idx, prompt in enumerate(prompts):
                input_ids = tokenizer([prompt], return_tensors="pt").input_ids.to(device)
                print(f"\ncase prompt={prompt_idx} max_new_tokens={max_new} input_len={input_ids.shape[1]}")

                golden, full_seconds = timed_call(
                    device,
                    full_context_golden,
                    model,
                    tokenizer,
                    input_ids,
                    args,
                )
                torch.cuda.empty_cache()

                cached, cached_seconds = timed_call(
                    device,
                    cached_full_context_sample,
                    model,
                    input_ids,
                    args,
                    modeling_module,
                )
                torch.cuda.empty_cache()

                trace = []
                cached_trace = cached_full_context_sample(
                    model,
                    input_ids,
                    args,
                    modeling_module,
                    trace=trace,
                )
                max_logit_diff = max(trace) if trace else 0.0
                token_exact = torch.equal(golden, cached) and torch.equal(golden, cached_trace)
                speedup = full_seconds / cached_seconds if cached_seconds else float("inf")
                speedups.append(speedup)
                ok = token_exact and max_logit_diff <= cli_args.atol
                all_ok = all_ok and ok
                print(
                    "full_context_sample "
                    f"token_exact={token_exact} "
                    f"max_step_logit_abs_diff={max_logit_diff:.6g} "
                    f"full_seconds={full_seconds:.3f} cached_seconds={cached_seconds:.3f} "
                    f"speedup={speedup:.2f}x status={'MATCH' if ok else 'MISMATCH'}"
                )
                print(f"generated={summarize_ids(tokenizer, golden, input_ids.shape[1])!r}")

                if not cli_args.skip_generation_entry:
                    generation_args = make_args(model, tokenizer, max_new, cli_args)
                    gen_golden, gen_seconds = timed_call(
                        device,
                        generation_golden,
                        model,
                        tokenizer,
                        input_ids,
                        generation_args,
                        device,
                    )
                    torch.cuda.empty_cache()
                    gen_trace = []
                    gen_cached_dict, gen_cached_seconds = timed_call(
                        device,
                        cached_generation_functions_sample,
                        model,
                        input_ids,
                        generation_args,
                        modeling_module,
                        trace=None,
                    )
                    gen_cached = gen_cached_dict[0]
                    gen_trace_dict = cached_generation_functions_sample(
                        model,
                        input_ids,
                        generation_args,
                        modeling_module,
                        trace=gen_trace,
                    )
                    gen_cached_trace = gen_trace_dict[0]
                    gen_logit_diff = max(gen_trace) if gen_trace else 0.0
                    gen_token_exact = torch.equal(gen_golden, gen_cached) and torch.equal(gen_golden, gen_cached_trace)
                    gen_speedup = gen_seconds / gen_cached_seconds if gen_cached_seconds else float("inf")
                    generation_speedups.append(gen_speedup)
                    gen_ok = gen_token_exact and gen_logit_diff <= cli_args.atol
                    all_ok = all_ok and gen_ok
                    print(
                        "generation_functions_batch_sample "
                        f"token_exact={gen_token_exact} "
                        f"max_step_logit_abs_diff={gen_logit_diff:.6g} "
                        f"golden_seconds={gen_seconds:.3f} cached_seconds={gen_cached_seconds:.3f} "
                        f"speedup={gen_speedup:.2f}x status={'MATCH' if gen_ok else 'MISMATCH'}"
                    )

                gc.collect()
                torch.cuda.empty_cache()

    if speedups:
        print(f"\nfull_context_sample_speedup_mean={sum(speedups) / len(speedups):.2f}x")
        print(f"full_context_sample_speedup_min={min(speedups):.2f}x max={max(speedups):.2f}x")
    if generation_speedups:
        print(f"generation_functions_speedup_mean={sum(generation_speedups) / len(generation_speedups):.2f}x")
    print(f"FINAL: {'MATCH' if all_ok else 'MISMATCH'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
