#!/usr/bin/env python3
"""OPT-4 forward-time breakdown: torch.profiler CUDA kernel attribution on the
fixed FLARE hybrid_clean engine (pin d2fccab, sync scheduler). Runs 3 turns
(short/medium/long), counts TRUE denoise forwards, and buckets CUDA self-time by
kernel family so the residual GPU cost of the CL-wide prefill-classed forward +
GDN chunk kernel (the OPT-4 target) is measured, not assumed.

Writes runs/p2_engine_battery_v3b/opt4_breakdown.json.
"""
import json, os, sys, time
from collections import defaultdict
from pathlib import Path

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("VLLM_USE_V1", "1")
import torch  # noqa: E402
from torch.profiler import profile, ProfilerActivity  # noqa: E402

ROOT = Path("/home/mark/qwen_diffusion")
sys.path.insert(0, str(ROOT / "scripts"))
VLLM_WS = Path("/home/mark/shared/vllm_p2_pr42406")
MODEL = ROOT / "models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16"
REF = ROOT / "runs/p2_engine_bench/matched20_ref.json"
OUT = ROOT / "runs/p2_engine_battery_v3b/opt4_breakdown.json"
GTS = [int(x) for x in os.environ.get("GTS", "7 25 35").split()]

import parity_audit_flare_engine as H  # noqa: E402
from vllm.v1.worker.gpu.model_states import qwen3_5_flare as QF  # noqa: E402

# count TRUE denoise forwards per turn
_CNT = {"fwd": 0}
_real_step = QF.Qwen3_5FlareSampler._hybrid_clean_step
def _step(self, shifted, block_logits, decode_slots, decode_idx, decode_indices_np,
          decode_slots_np, valid_len_np, is_committing, num_reqs, input_batch):
    if not bool(is_committing[0].item()):
        _CNT["fwd"] += 1
    return _real_step(self, shifted, block_logits, decode_slots, decode_idx,
                      decode_indices_np, decode_slots_np, valid_len_np,
                      is_committing, num_reqs, input_batch)
QF.Qwen3_5FlareSampler._hybrid_clean_step = _step

# kernel-name -> family bucket
def is_operator_row(name: str) -> bool:
    """True for CPU operator-dispatch rows that double-count their child device
    kernels (aten::mm wrapping cutlass gemm, autograd Function wrappers, cuda API
    calls). Excluded from the kernel-level breakdown to avoid double counting."""
    n = name
    if n.startswith("aten::") or n.startswith("cuda") or n.startswith("c10d"):
        return True
    if n.endswith("Function") or "autograd::" in n or "Backward" in n:
        return True
    return False

def bucket(name: str) -> str:
    n = name.lower()
    gdn = ("gated_delta", "gateddelta", "delta_rule", "deltarule", "chunk_gdn",
           "gdn", "causal_conv", "conv1d", "solve_tril", "l2norm", "l2_norm",
           "chunk_scan", "chunk_o", "chunk_a", "chunk_bwd", "chunk_fwd",
           "wy_fast", "fused_recurrent", "recurrent_gated")
    attn = ("attn", "flash", "_fwd_kernel", "attention")
    gemm = ("gemm", "cutlass", "sgemm", "cublas", "ampere", "sm90", "sm100",
            "sm120", "wgmma", "s16816", "tensorop")
    samp = ("topk", "radix", "sort", "argmax", "arange", "gather", "scatter")
    norm = ("rmsnorm", "rms_norm", "layernorm", "layer_norm", "norm_kernel",
            "silu", "swiglu", "act_and_mul", "rope", "rotary", "softmax")
    ew = ("elementwise", "vectorized", "copy", "cast", "fill", "index", "cat",
          "stack", "memset", "pad", "reduce", "mean", "add", "mul", "unrolled")
    if any(k in n for k in gdn):  return "GDN/linear-attn"
    if any(k in n for k in attn): return "full-attn"
    if any(k in n for k in samp): return "sampling/topk"
    if any(k in n for k in norm): return "norm/act/rope"
    if any(k in n for k in gemm): return "gemm(MLP+proj+lm_head)"
    if any(k in n for k in ew):   return "elementwise/copy"
    return "other"

def main():
    records = {r["global_turn"]: r for r in json.loads(REF.read_text())}
    os.environ["VLLM_QWEN3_5_FLARE_MASK"] = str(int(records[GTS[0]]["mask_id"]))
    adapter = H.build_engine_adapter("vllm", vllm_workspace=VLLM_WS, model_path=str(MODEL),
                                     canvas_length=32, decode_mode="hybrid_clean", seed=20260701)
    engine = adapter._build_engine()
    from vllm import SamplingParams
    results = []
    for gt in GTS:
        rec = records[gt]
        tools = [{"type": "function", "function": {"name": n, "parameters": p}}
                 for n, p in (rec["schemas"] or {}).items()]
        sp = SamplingParams(max_tokens=rec["n_ref"] + 16, temperature=0.0, top_p=1.0, seed=20260701,
                            stop_token_ids=sorted(int(x) for x in rec["stop_token_ids"]),
                            extra_args={"decode_policy": "hybrid_clean", "tools": tools,
                                        "grammar_topk": int(rec["grammar_topk"])})
        # warm (prefix cache + graph) once, untimed
        adapter._engine_generate(engine, list(rec["prompt_ids"]), sp)
        _CNT["fwd"] = 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            req = adapter._engine_generate(engine, list(rec["prompt_ids"]), sp)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        fwd = _CNT["fwd"]
        n_gen = len(req.outputs[0].token_ids)
        ka = prof.key_averages()
        fam = defaultdict(float)   # cuda self us, KERNEL-level only
        total_cuda = 0.0
        kernels = []
        for e in ka:
            cu = float(getattr(e, "self_device_time_total", 0.0) or getattr(e, "self_cuda_time_total", 0.0))
            if cu <= 0:
                continue
            if is_operator_row(e.key):   # skip double-counting operator dispatch rows
                continue
            total_cuda += cu
            fam[bucket(e.key)] += cu
            kernels.append((e.key, cu, int(e.count)))
        kernels.sort(key=lambda x: -x[1])
        fam_ms = {k: round(v/1000, 2) for k, v in sorted(fam.items(), key=lambda x: -x[1])}
        fam_pct = {k: round(100*v/total_cuda, 1) for k, v in sorted(fam.items(), key=lambda x: -x[1])}
        rr = {
            "gt": gt, "ep_t": "ep%d/t%d" % (rec["episode"], rec["turn"]),
            "n_gen": n_gen, "n_ref": rec["n_ref"], "denoise_forwards": fwd,
            "wall_s": round(wall, 3),
            "total_cuda_ms": round(total_cuda/1000, 2),
            "gpu_ms_per_forward": round((total_cuda/1000)/max(fwd, 1), 3),
            "host_overhead_ms_per_forward": round((wall*1000 - total_cuda/1000)/max(fwd, 1), 3),
            "family_ms": fam_ms, "family_pct": fam_pct,
            "n_kernel_names": len(kernels),
            "all_kernels": [{"name": k, "cuda_ms": round(v/1000, 2), "calls": c,
                             "family": bucket(k)} for k, v, c in kernels],
        }
        results.append(rr)
        print("gt%d %s n=%d fwd=%d wall=%.3fs cuda=%.1fms gpu_ms/fwd=%.3f host_ms/fwd=%.3f" % (
            gt, rr["ep_t"], n_gen, fwd, wall, rr["total_cuda_ms"], rr["gpu_ms_per_forward"],
            rr["host_overhead_ms_per_forward"]), flush=True)
        print("   families:", fam_pct, flush=True)
    OUT.write_text(json.dumps(results, indent=2) + "\n")
    print("WROTE", OUT, flush=True)

if __name__ == "__main__":
    main()
