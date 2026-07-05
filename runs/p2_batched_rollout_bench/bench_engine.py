#!/usr/bin/env python3
"""FLARE hybrid-clean ENGINE batched-rollout THROUGHPUT sweep (batch 1..16).

RL-rollout regime: the diffusion twin generates on-policy RL signal at high
throughput. The batch-correctness gates (runs/p2_engine_batchgates) proved NO
cross-request contamination, so the batched path is safe to benchmark. This script
measures sustained rollout throughput vs concurrency in the production serving
config (PIECEWISE cudagraph ON, APC ON, the v3b/nevertrain certified engine).

PRIMARY MODE = temp=0.7 SEEDED (the RL sampling mode). One extra greedy bs=8 point
is added for reference (throughput is ~temperature-insensitive, we show it).

Method: one boot (max_num_seqs=16). For each B, process a fixed pool of never-train
turns in WAVES of B concurrent requests (one engine.generate() per wave -- the SYNC
scheduler co-batches the B requests). A warmup wave at each B precedes timing so
cudagraph capture + first-call overhead are excluded. samples/sec = n_turns/total_wall.

Metrics per point: turns/sec, generated tok/sec, per-forward ms, forwards/turn,
mean/effective batch-in-forward (co-batching occupancy -> sync-scheduler straggler
analysis), GPU util (sampled ACROSS the timed wall), GPU mem peak, host-RAM peak.

One heavy process; RAM cage; foreground.
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
HERE = ROOT / "runs/p2_batched_rollout_bench"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))
VLLM_WS = Path("/home/mark/shared/vllm_p2_pr42406")
MODEL = ROOT / "models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16"
REF = ROOT / "runs/p2_engine_nevertrain/nevertrain_ref.json"

os.environ["VLLM_FLARE_CUDAGRAPH"] = "1"
os.environ.setdefault("VLLM_FLARE_BIDIR_PROBE", "1")

from gpu_sampler import GpuSampler, host_ram_peak_gb, gpu_snapshot  # noqa: E402
import parity_audit_flare_engine as H  # noqa: E402
from vllm.v1.worker.gpu.model_states import qwen3_5_flare as QF  # noqa: E402

SEED = int(os.environ.get("BENCH_SEED", "20260701"))
MARGIN = int(os.environ.get("BENCH_MARGIN", "16"))
MAXSEQ = int(os.environ.get("BENCH_MAXSEQ", "16"))
BATCHES = [int(x) for x in os.environ.get("BENCH_BATCHES", "1 2 4 8 16").split()]
POOL_N = int(os.environ.get("BENCH_POOL_N", "48"))
PMIN = int(os.environ.get("BENCH_PMIN", "400"))
PMAX = int(os.environ.get("BENCH_PMAX", "1300"))
TEMP = float(os.environ.get("BENCH_TEMP", "0.7"))
GMU = float(os.environ.get("BENCH_GMU", "0.74"))
GREEDY_B = int(os.environ.get("BENCH_GREEDY_B", "8"))  # extra greedy reference point

# ---- forward ledger: num_reqs (co-batch occupancy) + committing flag per forward ----
LED = {"fwds": 0, "nreq": [], "committing": []}
_real_step = QF.Qwen3_5FlareSampler._hybrid_clean_step


def _patched_step(self, shifted, block_logits, decode_slots, decode_idx,
                  decode_indices_np, decode_slots_np, valid_len_np,
                  is_committing, num_reqs, input_batch):
    LED["fwds"] += 1
    LED["nreq"].append(int(input_batch.num_reqs))
    try:
        LED["committing"].append(bool(is_committing[0].item()))
    except Exception:
        LED["committing"].append(None)
    return _real_step(self, shifted, block_logits, decode_slots, decode_idx,
                      decode_indices_np, decode_slots_np, valid_len_np,
                      is_committing, num_reqs, input_batch)


QF.Qwen3_5FlareSampler._hybrid_clean_step = _patched_step


def occupancy_hist(nreq, B):
    h = {}
    for v in nreq:
        h[v] = h.get(v, 0) + 1
    # fraction of forwards that ran at the FULL requested width B
    full = sum(1 for v in nreq if v == B) / len(nreq) if nreq else 0.0
    return {str(k): h[k] for k in sorted(h)}, round(full, 3)


def main():
    records = json.loads(REF.read_text())
    mask_id = int(records[0]["mask_id"])
    os.environ["VLLM_QWEN3_5_FLARE_MASK"] = str(mask_id)
    block_size = int(records[0]["block_size"])
    pool = sorted((r for r in records if PMIN <= r["prompt_len"] <= PMAX),
                  key=lambda r: r["global_turn"])[:POOL_N]
    assert len(pool) >= 48, f"pool too small: {len(pool)}"
    print(f"[eng] pool_n={len(pool)} prompt_len[{PMIN},{PMAX}] "
          f"lens={min(r['prompt_len'] for r in pool)}..{max(r['prompt_len'] for r in pool)} "
          f"nref_mean={np.mean([r['n_ref'] for r in pool]):.1f} batches={BATCHES} "
          f"temp={TEMP} gmu={GMU} maxseq={MAXSEQ}", flush=True)

    from vllm import SamplingParams
    adapter = H.build_engine_adapter(
        "vllm", vllm_workspace=VLLM_WS, model_path=str(MODEL),
        canvas_length=block_size, decode_mode="hybrid_clean", seed=SEED,
        gpu_memory_utilization=GMU)
    adapter._engine_kwargs.update({
        "max_num_seqs": MAXSEQ, "max_num_batched_tokens": 4096,
        "enable_prefix_caching": True})
    t0 = time.time()
    engine = adapter._build_engine()
    boot_s = time.time() - t0
    u0, m0 = gpu_snapshot()
    print(f"[eng] booted boot_s={boot_s:.1f} idle_mem={m0}MB", flush=True)

    def make_sp(rec, temp):
        tools = [{"type": "function", "function": {"name": n, "parameters": p}}
                 for n, p in (rec["schemas"] or {}).items()]
        greedy = temp <= 0.0
        return SamplingParams(
            max_tokens=rec["n_ref"] + MARGIN, temperature=temp,
            top_p=1.0, seed=SEED + int(rec["global_turn"]),
            stop_token_ids=sorted(int(x) for x in rec["stop_token_ids"]),
            extra_args={"decode_policy": "hybrid_clean", "tools": tools,
                        "grammar_topk": int(rec["grammar_topk"])})

    def run_wave(recs, temp):
        engine.reset_prefix_cache()
        prompts = [{"prompt_token_ids": list(r["prompt_ids"])} for r in recs]
        sps = [make_sp(r, temp) for r in recs]
        outs = engine.generate(prompts, sps, use_tqdm=False)
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        fins = [o.outputs[0].finish_reason for o in outs]
        return ntok, fins

    # warmup: one wave per distinct batch width (excl. from timing)
    for B in sorted(set(BATCHES + ([GREEDY_B] if GREEDY_B > 0 else []))):
        run_wave(pool[:B], TEMP)
    torch.cuda.empty_cache()

    points = []  # per-point rows -> JSONL

    def bench_point(B, temp, variant):
        LED["fwds"] = 0
        LED["nreq"] = []
        LED["committing"] = []
        waves = [pool[i:i + B] for i in range(0, len(pool), B)]
        waves = [w for w in waves if len(w) == B]  # only full B-wide waves
        n_turns = sum(len(w) for w in waves)
        wave_walls = []
        fin_counts = {}
        sampler = GpuSampler(interval=0.2)
        sampler.start()
        t_start = time.time()
        total_tok = 0
        for w in waves:
            tw = time.time()
            ntok, fins = run_wave(w, temp)
            wave_walls.append(time.time() - tw)
            total_tok += ntok
            for f in fins:
                fin_counts[str(f)] = fin_counts.get(str(f), 0) + 1
        wall = time.time() - t_start
        sampler.stop()
        gs = sampler.summary()
        nreq = LED["nreq"]
        hist, full_frac = occupancy_hist(nreq, B)
        mixed = sum(1 for c in LED["committing"] if c)  # committing forwards
        row = {
            "engine": "FLARE_hybrid_clean", "config": "production(cudagraph+apc)",
            "variant": variant, "batch": B, "temp": temp,
            "n_turns": n_turns, "n_waves": len(waves),
            "wall_s": round(wall, 3),
            "samples_per_sec": round(n_turns / wall, 4),
            "tokens_per_sec": round(total_tok / wall, 2),
            "total_gen_tokens": total_tok,
            "mean_gen_tokens_per_turn": round(total_tok / n_turns, 1),
            "forwards": LED["fwds"],
            "committing_forwards": mixed,
            "forwards_per_sec": round(LED["fwds"] / wall, 1),
            "mean_forwards_per_turn": round(LED["fwds"] / n_turns, 2),
            "per_forward_ms": round(1000.0 * wall / LED["fwds"], 3) if LED["fwds"] else None,
            "mean_batch_in_forward": round(float(np.mean(nreq)), 3) if nreq else 0,
            "max_batch_in_forward": int(max(nreq)) if nreq else 0,
            "batch_occupancy_full_frac": full_frac,        # frac of forwards at width==B
            "batch_occupancy_efficiency": round(float(np.mean(nreq)) / B, 3) if nreq else 0,
            "batch_in_forward_hist": hist,
            "mean_wave_wall_s": round(float(np.mean(wave_walls)), 4),
            "p90_wave_wall_s": round(float(np.percentile(wave_walls, 90)), 4) if wave_walls else None,
            "finish_reasons": fin_counts,
            "gpu_mem_used_mb": gpu_snapshot()[1],
            "host_ram_peak_gb": host_ram_peak_gb(),
            "seed": SEED,
        }
        row.update(gs)
        points.append(row)
        print(f"[eng] {variant} B={B:2d} n={n_turns} wall={wall:.2f}s "
              f"samp/s={row['samples_per_sec']:.3f} tok/s={row['tokens_per_sec']:.0f} "
              f"fwd/turn={row['mean_forwards_per_turn']} pf_ms={row['per_forward_ms']} "
              f"occ={row['mean_batch_in_forward']}/{B}({row['batch_occupancy_efficiency']}) "
              f"util~{gs['gpu_util_mean_pct']}% mem={row['gpu_mem_used_mb']}MB "
              f"ram_peak={row['host_ram_peak_gb']}GB", flush=True)
        return row

    # extra reference: greedy at GREEDY_B FIRST (so a possible OOM/queue at the b16
    # tail of the temp sweep cannot cost us the greedy datapoint). GREEDY_B<=0 skips.
    if GREEDY_B > 0:
        try:
            bench_point(GREEDY_B, 0.0, "greedy")
        except Exception as e:  # noqa: BLE001
            torch.cuda.empty_cache()
            points.append({"variant": "greedy", "batch": GREEDY_B, "temp": 0.0,
                           "error": type(e).__name__, "detail": repr(e)[:200]})
            print(f"[eng] greedy B={GREEDY_B} ERROR: {repr(e)[:160]}", flush=True)

    # PRIMARY: temp=0.7 seeded sweep (ascending batch; b16 last)
    for B in BATCHES:
        try:
            bench_point(B, TEMP, "rl_temp0.7")
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            points.append({"variant": "rl_temp0.7", "batch": B, "temp": TEMP,
                           "error": "CUDA_OOM", "detail": repr(e)[:200]})
            print(f"[eng] rl_temp0.7 B={B} OOM: {repr(e)[:120]}", flush=True)
        except Exception as e:  # noqa: BLE001
            torch.cuda.empty_cache()
            points.append({"variant": "rl_temp0.7", "batch": B, "temp": TEMP,
                           "error": type(e).__name__, "detail": repr(e)[:200]})
            print(f"[eng] rl_temp0.7 B={B} ERROR: {repr(e)[:160]}", flush=True)

    # speedups vs B=1 (within the rl_temp0.7 variant)
    ok = [p for p in points if "error" not in p and p["variant"] == "rl_temp0.7"]
    base = next((p for p in ok if p["batch"] == 1), ok[0] if ok else None)
    if base:
        for p in ok:
            p["speedup_samples_vs_b1"] = round(p["samples_per_sec"] / base["samples_per_sec"], 3)
            p["speedup_tokens_vs_b1"] = round(p["tokens_per_sec"] / base["tokens_per_sec"], 3)

    # write per-point JSONL
    with (HERE / "engine_points.jsonl").open("w") as fh:
        for p in points:
            fh.write(json.dumps(p) + "\n")
    summ = {
        "engine": "FLARE_hybrid_clean", "config": "production(cudagraph+apc)",
        "model": str(MODEL.name), "vllm_pin": "95d8b47",
        "pool_n": len(pool), "prompt_len_band": [PMIN, PMAX], "seed": SEED,
        "gpu_memory_utilization": GMU, "max_num_seqs": MAXSEQ, "temp_primary": TEMP,
        "boot_s": round(boot_s, 1), "points": points,
    }
    (HERE / "engine_throughput.json").write_text(json.dumps(summ, indent=2))
    print("[eng] SPEEDUP vs B=1 (rl_temp0.7): " + " | ".join(
        f"B{p['batch']}={p.get('speedup_samples_vs_b1','?')}x({p['samples_per_sec']}/s)"
        for p in ok), flush=True)
    print("[eng] DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("[eng] FATAL:", repr(e), flush=True)
        traceback.print_exc()
        raise
