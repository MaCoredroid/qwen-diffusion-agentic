#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]


CASES = {
    "diffusiongemma": {
        "model": "nvidia/diffusiongemma-26B-A4B-it-NVFP4",
        "prompt": "Write one short sentence about fast inference.",
        "max_tokens": 16,
        "llm_kwargs": {
            "quantization": "modelopt",
            "trust_remote_code": True,
            "max_model_len": 1024,
            "gpu_memory_utilization": 0.70,
            "max_num_seqs": 1,
            "max_num_batched_tokens": 1024,
            "attention_config": {"backend": "TRITON_ATTN"},
            "diffusion_config": {"canvas_length": 32, "max_denoising_steps": 4},
        },
    },
    "qwen-default": {
        "model": str(ROOT / "models/qwen3.5-9b-fastdllm-b1000-vllm-bf16"),
        "prompt": "Name one benefit of prefix caching.",
        "max_tokens": 12,
        "llm_kwargs": {
            "trust_remote_code": True,
            "max_model_len": 1024,
            "gpu_memory_utilization": 0.70,
            "max_num_seqs": 1,
            "max_num_batched_tokens": 1024,
        },
    },
    "qwen-align-apc": {
        "model": str(ROOT / "models/qwen3.5-9b-fastdllm-b1000-vllm-bf16"),
        "prompt": "Name one benefit of prefix caching.",
        "max_tokens": 12,
        "llm_kwargs": {
            "trust_remote_code": True,
            "max_model_len": 1024,
            "gpu_memory_utilization": 0.70,
            "max_num_seqs": 1,
            "max_num_batched_tokens": 1024,
            "enable_prefix_caching": True,
            "mamba_cache_mode": "align",
            "mamba_block_size": 1024,
            "mamba_ssm_cache_dtype": "float32",
        },
    },
}


def run_case(case_name: str, out_path: Path) -> int:
    case = CASES[case_name]
    # Env overrides so a box whose desktop already holds several GiB of VRAM can
    # fit the per-case gpu_memory_utilization budget under the free pool (the
    # engine hard-guards desired_budget <= free_memory at startup) and drop
    # CUDA-graph capture memory (the documented --enforce-eager run mode). Both
    # default to the per-case value so behavior is unchanged when unset.
    llm_kwargs = dict(case["llm_kwargs"])
    _util = os.environ.get("VLLM_SMOKE_GPU_UTIL")
    if _util:
        llm_kwargs["gpu_memory_utilization"] = float(_util)
    if os.environ.get("VLLM_SMOKE_ENFORCE_EAGER", "").lower() in ("1", "true", "yes"):
        llm_kwargs["enforce_eager"] = True
    # NvFP4-MoE backend override. The auto-selected FLASHINFER_CUTLASS backend
    # JIT-compiles an sm120 cutlass MoE module via nvcc; on a box without a
    # matching CUDA toolkit (or with a flashinfer/CUDA header-version clash) that
    # build fails. Forcing a precompiled/non-JIT backend (e.g. "marlin",
    # "cutlass" = VLLM_CUTLASS, or "emulation") lets the diffusiongemma smoke
    # exercise the dLLM decode path without the flashinfer JIT toolchain. Only
    # affects MoE models; the FLARE Qwen3.5-9B target is dense and never hits it.
    _moe = os.environ.get("VLLM_SMOKE_MOE_BACKEND")
    if _moe:
        llm_kwargs["moe_backend"] = _moe
    result = {
        "case": case_name,
        "model": case["model"],
        "env": {
            key: os.environ.get(key)
            for key in [
                "VLLM_USE_V2_MODEL_RUNNER",
                "CUDA_VISIBLE_DEVICES",
                "VLLM_WORKER_MULTIPROC_METHOD",
                "VLLM_SMOKE_GPU_UTIL",
                "VLLM_SMOKE_ENFORCE_EAGER",
                "VLLM_SMOKE_MOE_BACKEND",
            ]
        },
        "llm_kwargs": llm_kwargs,
    }
    try:
        from vllm import LLM, SamplingParams
        import torch
        import vllm

        result["vllm"] = vllm.__version__
        result["torch"] = torch.__version__
        result["torch_cuda"] = torch.version.cuda
        result["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            result["gpu_name"] = torch.cuda.get_device_name(0)
            result["gpu_capability"] = list(torch.cuda.get_device_capability(0))

        load_start = time.time()
        llm = LLM(model=case["model"], **llm_kwargs)
        result["load_seconds"] = time.time() - load_start

        gen_start = time.time()
        sp = SamplingParams(max_tokens=case["max_tokens"], temperature=0.0)
        # These are instruct ("-it") checkpoints: feed the prompt through the
        # model's chat template via llm.chat(), otherwise a raw continuation
        # degenerates (repeated punctuation) and the coherence check is
        # meaningless. Fall back to raw generate() for any base model that ships
        # no chat template.
        try:
            outputs = llm.chat(
                [{"role": "user", "content": case["prompt"]}], sp
            )
            result["prompt_mode"] = "chat_template"
        except Exception:
            outputs = llm.generate([case["prompt"]], sp)
            result["prompt_mode"] = "raw"
        result["generate_seconds"] = time.time() - gen_start
        result["output_text"] = outputs[0].outputs[0].text
        result["status"] = "PASS"
    except Exception as exc:
        result["status"] = "FAIL"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback_tail"] = traceback.format_exc()[-5000:]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, default=str), flush=True)
    return 0 if result["status"] == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=sorted(CASES))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    return run_case(args.case, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
