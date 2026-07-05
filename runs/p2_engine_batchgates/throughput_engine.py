#!/usr/bin/env python3
"""FLARE hybrid-clean ENGINE rollout-throughput sweep (batch 1..16).

Unblocked by the batch-correctness gates (no cross-request contamination). Measures
rollout samples/sec/GPU and tokens/sec as a function of concurrency, on never-train
turns, in the production serving config (CUDAGRAPH on, APC on, the v3b/nevertrain
engine). Tests the FLOP-reducing batch-robustness claim: hybrid should AMORTIZE the
bs=1 weight-stream floor (engine_build_status.md 0.H) as batch grows.

Method: one boot (max_num_seqs=16). For each B in BATCHES, process the same pool of
turns in WAVES of B concurrent requests (one engine.generate() call per wave, B
prompts submitted together). Sum wall across waves; samples/sec = n_turns/total_wall.
A warmup wave precedes timing. Greedy (throughput is temperature-insensitive).

One heavy process; RAM cage.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("VLLM_USE_V1", "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path("/home/mark/qwen_diffusion")
sys.path.insert(0, str(ROOT / "scripts"))
VLLM_WS = Path("/home/mark/shared/vllm_p2_pr42406")
MODEL = ROOT / "models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16"
REF = ROOT / "runs/p2_engine_nevertrain/nevertrain_ref.json"
OUTDIR = ROOT / "runs/p2_engine_batchgates"

os.environ["VLLM_FLARE_CUDAGRAPH"] = "1"
os.environ.setdefault("VLLM_FLARE_BIDIR_PROBE", "1")

import parity_audit_flare_engine as H  # noqa: E402
from vllm.v1.worker.gpu.model_states import qwen3_5_flare as QF  # noqa: E402

SEED = int(os.environ.get("BENCH_SEED", "20260701"))
MARGIN = int(os.environ.get("BENCH_MARGIN", "16"))
MAXSEQ = int(os.environ.get("BENCH_MAXSEQ", "16"))
BATCHES = [int(x) for x in os.environ.get("BENCH_BATCHES", "1 2 4 8 16").split()]
POOL_N = int(os.environ.get("BENCH_POOL_N", "32"))
PMIN = int(os.environ.get("BENCH_PMIN", "400"))
PMAX = int(os.environ.get("BENCH_PMAX", "1300"))

# ---- lightweight forward-count ledger (num_reqs per forward) ----
LED = {"fwds": 0, "nreq": []}
_real_step = QF.Qwen3_5FlareSampler._hybrid_clean_step


def _patched_step(self, shifted, block_logits, decode_slots, decode_idx,
                  decode_indices_np, decode_slots_np, valid_len_np,
                  is_committing, num_reqs, input_batch):
    LED["fwds"] += 1
    LED["nreq"].append(int(input_batch.num_reqs))
    return _real_step(self, shifted, block_logits, decode_slots, decode_idx,
                      decode_indices_np, decode_slots_np, valid_len_np,
                      is_committing, num_reqs, input_batch)


QF.Qwen3_5FlareSampler._hybrid_clean_step = _patched_step


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
    mask_id = int(records[0]["mask_id"])
    os.environ["VLLM_QWEN3_5_FLARE_MASK"] = str(mask_id)
    block_size = int(records[0]["block_size"])
    # pool: moderate-length never-train turns (avoid 16-way OOM on 2.6k prompts)
    pool = sorted((r for r in records if PMIN <= r["prompt_len"] <= PMAX),
                  key=lambda r: r["global_turn"])[:POOL_N]
    print(f"[tp] pool_n={len(pool)} prompt_len[{PMIN},{PMAX}] "
          f"lens={min(r['prompt_len'] for r in pool)}..{max(r['prompt_len'] for r in pool)} "
          f"nref_mean={np.mean([r['n_ref'] for r in pool]):.1f} batches={BATCHES}", flush=True)

    from vllm import SamplingParams
    adapter = H.build_engine_adapter(
        "vllm", vllm_workspace=VLLM_WS, model_path=str(MODEL),
        canvas_length=block_size, decode_mode="hybrid_clean", seed=SEED)
    adapter._engine_kwargs.update({
        "max_num_seqs": MAXSEQ, "max_num_batched_tokens": 4096,
        "enable_prefix_caching": True})
    t0 = time.time()
    engine = adapter._build_engine()
    print(f"[tp] booted boot_s={time.time()-t0:.1f}", flush=True)

    def make_sp(rec):
        tools = [{"type": "function", "function": {"name": n, "parameters": p}}
                 for n, p in (rec["schemas"] or {}).items()]
        return SamplingParams(
            max_tokens=rec["n_ref"] + MARGIN, temperature=0.0, top_p=1.0, seed=SEED,
            stop_token_ids=sorted(int(x) for x in rec["stop_token_ids"]),
            extra_args={"decode_policy": "hybrid_clean", "tools": tools,
                        "grammar_topk": int(rec["grammar_topk"])})

    def run_wave(recs):
        engine.reset_prefix_cache()
        prompts = [{"prompt_token_ids": list(r["prompt_ids"])} for r in recs]
        sps = [make_sp(r) for r in recs]
        outs = engine.generate(prompts, sps)
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        return ntok

    # warmup: one wave at each distinct batch size so cudagraph capture + first-call
    # overhead are excluded from timing.
    for B in sorted(set(BATCHES)):
        run_wave(pool[:B])
    torch.cuda.empty_cache()

    results = []
    for B in BATCHES:
        LED["fwds"] = 0
        LED["nreq"] = []
        util_samples = []
        waves = [pool[i:i + B] for i in range(0, len(pool), B)]
        # drop a trailing partial wave so every timed wave is exactly B-wide
        waves = [w for w in waves if len(w) == B]
        n_turns = sum(len(w) for w in waves)
        t_start = time.time()
        total_tok = 0
        for wi, w in enumerate(waves):
            total_tok += run_wave(w)
            if wi == len(waves) // 2:
                u, m = gpu_mem_util()
                if u is not None:
                    util_samples.append(u)
        wall = time.time() - t_start
        u2, m2 = gpu_mem_util()
        nreq = LED["nreq"]
        row = {
            "batch": B, "n_turns": n_turns, "n_waves": len(waves),
            "wall_s": round(wall, 3),
            "samples_per_sec": round(n_turns / wall, 3),
            "tokens_per_sec": round(total_tok / wall, 1),
            "total_gen_tokens": total_tok,
            "forwards": LED["fwds"],
            "forwards_per_sec": round(LED["fwds"] / wall, 1),
            "mean_forwards_per_turn": round(LED["fwds"] / n_turns, 2),
            "mean_batch_in_forward": round(float(np.mean(nreq)), 2) if nreq else 0,
            "max_batch_in_forward": int(max(nreq)) if nreq else 0,
            "gpu_util_mid_pct": util_samples[0] if util_samples else None,
            "gpu_mem_used_mb": m2,
        }
        results.append(row)
        print(f"[tp] B={B:2d} n={n_turns} wall={wall:.2f}s "
              f"samples/s={row['samples_per_sec']:.2f} tok/s={row['tokens_per_sec']:.0f} "
              f"fwd/turn={row['mean_forwards_per_turn']} "
              f"mean_batch_in_fwd={row['mean_batch_in_forward']} "
              f"gpu_util~{row['gpu_util_mid_pct']}% mem={m2}MB", flush=True)

    base = next((r for r in results if r["batch"] == 1), results[0])
    for r in results:
        r["speedup_samples_vs_b1"] = round(r["samples_per_sec"] / base["samples_per_sec"], 2)
        r["speedup_tokens_vs_b1"] = round(r["tokens_per_sec"] / base["tokens_per_sec"], 2)
    summ = {
        "engine": "FLARE_hybrid_clean", "config": "production(cudagraph+apc)",
        "model": str(MODEL.name), "pool_n": len(pool),
        "prompt_len_band": [PMIN, PMAX], "seed": SEED, "results": results,
    }
    (OUTDIR / "throughput_engine.json").write_text(json.dumps(summ, indent=2))
    print("[tp] SPEEDUP vs B=1: " + " | ".join(
        f"B{r['batch']}={r['speedup_samples_vs_b1']}x({r['samples_per_sec']}/s)"
        for r in results), flush=True)
    print("[tp] DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("[tp] ERROR:", repr(e), flush=True)
        traceback.print_exc()
        raise
