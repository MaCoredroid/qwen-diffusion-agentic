#!/usr/bin/env python3
"""STOCK-AR vLLM rollout-throughput sweep (batch 1..8) -- the baseline for the
FLARE engine throughput curve.

Same never-train turn pool, same wave method, same samples/sec metric, same machine,
so the engine-vs-AR comparison is internally apples-to-apples on workload + harness.
Stock Qwen3.5-9B (snapshot c202236) on stock vLLM 0.23.0 (.venv-vllm), plain greedy
AR generation (no grammar guidance -- the FASTEST, therefore most conservative, AR
baseline for the engine to beat). Prompts are the exact FLARE turn prompts (prompt
token ids); AR generates the assistant tool-call rollout autoregressively, capped at
n_ref+MARGIN with the same stop tokens.

One heavy process; RAM cage.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/mark/qwen_diffusion")
REF = ROOT / "runs/p2_engine_nevertrain/nevertrain_ref.json"
OUTDIR = ROOT / "runs/p2_engine_batchgates"
MODEL = os.environ.get(
    "AR_MODEL",
    "/home/mark/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/"
    "snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a")

SEED = int(os.environ.get("BENCH_SEED", "20260701"))
MARGIN = int(os.environ.get("BENCH_MARGIN", "16"))
MAXSEQ = int(os.environ.get("BENCH_MAXSEQ", "8"))
BATCHES = [int(x) for x in os.environ.get("BENCH_BATCHES", "1 2 4 8").split()]
POOL_N = int(os.environ.get("BENCH_POOL_N", "32"))
PMIN = int(os.environ.get("BENCH_PMIN", "400"))
PMAX = int(os.environ.get("BENCH_PMAX", "1300"))


def gpu_mem_util():
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"]).decode().strip().splitlines()[0]
        u, m = out.split(",")
        return int(u), int(m)
    except Exception:
        return None, None


def main():
    records = json.loads(REF.read_text())
    pool = sorted((r for r in records if PMIN <= r["prompt_len"] <= PMAX),
                  key=lambda r: r["global_turn"])[:POOL_N]
    print(f"[ar] model={Path(MODEL).name} pool_n={len(pool)} "
          f"lens={min(r['prompt_len'] for r in pool)}..{max(r['prompt_len'] for r in pool)} "
          f"nref_mean={np.mean([r['n_ref'] for r in pool]):.1f} batches={BATCHES}", flush=True)

    from vllm import LLM, SamplingParams
    # Stock Qwen3.5-9B is a GDN-hybrid model; match the endgame-scoreboard stock-AR
    # serving config (mamba align / block 1024, triton GDN prefill, flashinfer
    # sampler off). enforce_eager per AR_EAGER (default: try cudagraph = fast path).
    eager = os.environ.get("AR_EAGER", "0") == "1"
    gmu = float(os.environ.get("AR_GMU", "0.66"))
    kw = dict(model=MODEL, trust_remote_code=True, max_model_len=4096,
              gpu_memory_utilization=gmu, max_num_seqs=MAXSEQ,
              max_num_batched_tokens=4096, enable_prefix_caching=True, seed=SEED,
              mamba_cache_mode="align", mamba_block_size=1024, enforce_eager=eager)
    t0 = time.time()
    engine = LLM(**kw)
    vc = engine.llm_engine.vllm_config
    print(f"[ar] booted boot_s={time.time()-t0:.1f} eager={vc.model_config.enforce_eager} "
          f"gmu={gmu} max_num_seqs={vc.scheduler_config.max_num_seqs}", flush=True)

    def make_sp(rec):
        return SamplingParams(
            max_tokens=rec["n_ref"] + MARGIN, temperature=0.0, top_p=1.0, seed=SEED,
            stop_token_ids=sorted(int(x) for x in rec["stop_token_ids"]))

    def run_wave(recs):
        try:
            engine.reset_prefix_cache()
        except Exception:
            pass
        prompts = [{"prompt_token_ids": list(r["prompt_ids"])} for r in recs]
        sps = [make_sp(r) for r in recs]
        outs = engine.generate(prompts, sps, use_tqdm=False)
        return sum(len(o.outputs[0].token_ids) for o in outs)

    for B in sorted(set(BATCHES)):
        run_wave(pool[:B])

    results = []
    for B in BATCHES:
        waves = [pool[i:i + B] for i in range(0, len(pool), B)]
        waves = [w for w in waves if len(w) == B]
        n_turns = sum(len(w) for w in waves)
        util = None
        t_start = time.time()
        total_tok = 0
        for wi, w in enumerate(waves):
            total_tok += run_wave(w)
            if wi == len(waves) // 2:
                u, _ = gpu_mem_util()
                util = u
        wall = time.time() - t_start
        _, m2 = gpu_mem_util()
        row = {
            "batch": B, "n_turns": n_turns, "n_waves": len(waves),
            "wall_s": round(wall, 3),
            "samples_per_sec": round(n_turns / wall, 3),
            "tokens_per_sec": round(total_tok / wall, 1),
            "total_gen_tokens": total_tok,
            "mean_gen_tokens_per_turn": round(total_tok / n_turns, 1),
            "gpu_util_mid_pct": util, "gpu_mem_used_mb": m2,
        }
        results.append(row)
        print(f"[ar] B={B:2d} n={n_turns} wall={wall:.2f}s "
              f"samples/s={row['samples_per_sec']:.2f} tok/s={row['tokens_per_sec']:.0f} "
              f"gen_tok/turn={row['mean_gen_tokens_per_turn']} "
              f"gpu_util~{util}% mem={m2}MB", flush=True)

    base = next((r for r in results if r["batch"] == 1), results[0])
    for r in results:
        r["speedup_samples_vs_b1"] = round(r["samples_per_sec"] / base["samples_per_sec"], 2)
    summ = {
        "engine": "stock_AR_vllm", "config": "stock vllm 0.23.0 (cudagraph+apc)",
        "model": MODEL, "pool_n": len(pool), "prompt_len_band": [PMIN, PMAX],
        "seed": SEED, "results": results,
    }
    (OUTDIR / "throughput_ar.json").write_text(json.dumps(summ, indent=2))
    print("[ar] SPEEDUP vs B=1: " + " | ".join(
        f"B{r['batch']}={r['speedup_samples_vs_b1']}x({r['samples_per_sec']}/s)"
        for r in results), flush=True)
    print("[ar] DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("[ar] ERROR:", repr(e), flush=True)
        traceback.print_exc()
        raise
