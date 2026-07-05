#!/usr/bin/env python3
"""LOSSLESS-APC END-GOAL BENCH — AR baseline arm (stock Qwen3.5-9B, vLLM, cudagraph).

The flywheel-class baseline: stock AR served by vLLM with automatic prefix caching
(APC) and cudagraph, batch=1 (matched to the diffusion engine). We REPLAY the exact
same per-turn prompt_ids as the engine arm (from nevertrain_ref.json) so the context
growth is identical token-for-token; AR generates its own greedy continuation capped
at n_ref+margin with the same stop tokens.

Cache modes (task c, at matched caching): APC stays ENABLED in both (the hybrid
model rejects mamba_block_size without prefix caching), matched to the engine arm;
cold is realized by dropping the cross-turn cache each turn:
  BENCH_RESET_APC=1 -> llm.reset_prefix_cache() before each turn : AR cache-COLD
  BENCH_RESET_APC=0 -> keep cache across turns                   : AR cache-ON

Prefill split: in COLD mode a max_tokens=1 probe (fresh, reset first) times the
prompt prefill; the full generate is also reset-first, so decode = full_wall -
probe_wall. In ON mode we record total wall (the serving number); APC benefit =
wall_cold - wall_on (AR KV reuse is exact, so greedy n_gen is identical cold vs on
-> the delta is pure prefill reuse).

NOTE: AR here is UN-guided greedy (no grammar mask). Grammar-guided decoding would
only ADD per-step overhead to AR, so omitting it is conservative (favorable to AR)
for the engine-vs-AR speed comparison.

Env: EP_START/EP_END, BENCH_MARGIN(16), BENCH_SEED(20260701), BENCH_APC_OFF,
     BENCH_OUT, BENCH_REF, AR_MODEL, GPU_MEM_UTIL(0.6), MAX_MODEL_LEN(4096).
One heavy process; run inside the RAM cage. Do NOT set VLLM_QWEN3_5_FLARE.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
# Make sure we do NOT route to the FLARE model state -- this is plain AR.
os.environ.pop("VLLM_QWEN3_5_FLARE", None)
os.environ.pop("VLLM_QWEN3_5_FLARE_DECODE", None)

ROOT = Path("/home/mark/qwen_diffusion")
REF = Path(os.environ.get("BENCH_REF", str(ROOT / "runs/p2_engine_nevertrain/nevertrain_ref.json")))
OUT = Path(os.environ.get("BENCH_OUT", str(ROOT / "runs/lossless_apc/bench/ar_on.jsonl")))
AR_MODEL = os.environ.get(
    "AR_MODEL",
    "/home/mark/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/"
    "c202236235762e1c871ad0ccb60c8ee5ba337b9a")


def main():
    records = json.loads(REF.read_text())
    ep_start = int(os.environ.get("EP_START", "0"))
    ep_end = int(os.environ.get("EP_END", "9"))
    margin = int(os.environ.get("BENCH_MARGIN", "16"))
    seed = int(os.environ.get("BENCH_SEED", "20260701"))
    reset_apc = os.environ.get("BENCH_RESET_APC", "0") == "1"
    gpu_util = float(os.environ.get("GPU_MEM_UTIL", "0.6"))
    max_model_len = int(os.environ.get("MAX_MODEL_LEN", "4096"))

    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                done.add(int(json.loads(line)["global_turn"]))
            except Exception:
                pass
    # IMPORTANT: keep global_turn (episode) order so APC reuse across turns is realistic.
    todo = [r for r in records if ep_start <= r["episode"] <= ep_end
            and r["global_turn"] not in done]
    todo.sort(key=lambda r: r["global_turn"])
    mode = "COLD(reset_apc)" if reset_apc else "ON(apc)"
    print(f"[bench-ar {mode}] ep{ep_start}..{ep_end} todo={len(todo)} done={len(done)} "
          f"model={Path(AR_MODEL).name} OUT={OUT.name}", flush=True)
    if not todo:
        print("[bench-ar] nothing to do", flush=True)
        return

    from vllm import LLM, SamplingParams

    t_boot = time.time()
    llm = LLM(
        model=AR_MODEL,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_util,
        max_num_seqs=1,
        enforce_eager=False,           # cudagraph ON (task requirement)
        enable_prefix_caching=True,    # stays on; cold realized via reset per turn
        mamba_cache_mode="align",
        mamba_block_size=1024,
        mamba_ssm_cache_dtype="float32",
        seed=seed,
    )
    boot_s = round(time.time() - t_boot, 1)
    apc_live = llm.llm_engine.vllm_config.cache_config.enable_prefix_caching
    enforce_eager = llm.llm_engine.vllm_config.model_config.enforce_eager
    print(f"[bench-ar] booted boot_s={boot_s} apc_live={apc_live} enforce_eager={enforce_eager}",
          flush=True)

    def full_sp(rec):
        return SamplingParams(
            max_tokens=rec["n_ref"] + margin, temperature=0.0, top_p=1.0, seed=seed,
            stop_token_ids=sorted(int(x) for x in rec["stop_token_ids"]))

    probe_sp = SamplingParams(max_tokens=1, temperature=0.0, top_p=1.0, seed=seed)

    def timed(prompt_ids, sp):
        t0 = time.time()
        outs = llm.generate({"prompt_token_ids": list(prompt_ids)}, sp, use_tqdm=False)
        wall = time.time() - t0
        o = outs[0].outputs[0]
        return wall, [int(x) for x in o.token_ids], getattr(o, "finish_reason", None)

    fh = OUT.open("a")
    for rec in todo:
        prompt_ids = rec["prompt_ids"]
        prefill = None
        if reset_apc:
            # cold prefill probe (fresh): reset, 1-token generate times prompt prefill
            llm.reset_prefix_cache()
            prefill, _, _ = timed(prompt_ids, probe_sp)
            prefill = round(prefill, 4)
            llm.reset_prefix_cache()  # full generate is also fresh (no reuse of probe)
        wall, ids, finish = timed(prompt_ids, full_sp(rec))
        wall = round(wall, 4)
        decode_s = round(wall - prefill, 4) if prefill is not None else None
        turn = {
            "arm": "ar", "mode": ("cold" if reset_apc else "on"),
            "global_turn": rec["global_turn"], "episode": rec["episode"], "turn": rec["turn"],
            "episode_id": rec["episode_id"], "prompt_len": rec["prompt_len"],
            "n_ref": rec["n_ref"], "n_gen": len(ids), "maxtok": rec["n_ref"] + margin,
            "finish_reason": finish, "wall_s": wall, "prefill_s": prefill,
            "decode_s": decode_s,
            "prefill_frac": (round(prefill / wall, 4) if (prefill is not None and wall > 0) else None),
            "reset_apc": reset_apc, "source_family": rec.get("source_family"),
        }
        fh.write(json.dumps(turn) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
        print(f"[bench-ar {mode}] gt{rec['global_turn']:3d} ep{rec['episode']}/t{rec['turn']} "
              f"plen={rec['prompt_len']:5d} n={len(ids)} wall={wall:.4f} prefill={prefill} "
              f"dec={decode_s} pf%={turn['prefill_frac']}", flush=True)
    fh.close()
    print("[bench-ar] DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("[bench-ar] ERROR:", repr(e), flush=True)
        traceback.print_exc()
        raise
