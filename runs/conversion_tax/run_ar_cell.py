#!/usr/bin/env python3
"""Conversion-tax AR cell runner (STOCK-AR or MERGED-AR), any capability class.

Offline vLLM, bf16, enforce_eager, mamba-cache align, gdn-prefill triton, plain
greedy (temp 0) at B=1 on pre-tokenized prompt_ids. IDENTICAL serving config for
both AR systems; only AR_MODEL differs. Captures the raw completion text per item;
authoritative scoring is done later by aggregate.py (scoring.py). Resumable
(append + skip-by-idx). Env: AR_MODEL, AR_REF, AR_OUT, AR_MAXTOK, BENCH_SEED,
AR_START, AR_END.
"""
import json
import os
import time
from pathlib import Path

os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

MODEL = os.environ["AR_MODEL"]
REF = Path(os.environ["AR_REF"])
OUT = Path(os.environ["AR_OUT"])
MAXTOK = int(os.environ.get("AR_MAXTOK", "384"))
SEED = int(os.environ.get("BENCH_SEED", "20260701"))
START = int(os.environ.get("AR_START", "0"))
END = int(os.environ.get("AR_END", "10000"))


def main():
    recs = [r for r in json.loads(REF.read_text()) if START <= r["idx"] <= END]
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                try:
                    done.add(int(json.loads(line)["idx"]))
                except Exception:
                    pass
    todo = [r for r in recs if r["idx"] not in done]
    print(f"[ar] model={MODEL} todo={len(todo)} done={len(done)} maxtok={MAXTOK}", flush=True)
    if not todo:
        print("[ar] nothing to do", flush=True)
        return

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    t_boot = time.time()
    llm = LLM(model=MODEL, trust_remote_code=True, dtype="bfloat16",
              max_model_len=4096, gpu_memory_utilization=0.66,
              enable_prefix_caching=True, enforce_eager=True,
              mamba_cache_mode="align", mamba_block_size=1024,
              gdn_prefill_backend="triton", seed=SEED)
    print(f"[ar] booted boot_s={round(time.time()-t_boot,1)}", flush=True)

    def gen_one(rec):
        sp = SamplingParams(max_tokens=MAXTOK, temperature=0.0, top_p=1.0, seed=SEED,
                            stop_token_ids=sorted(int(x) for x in rec["stop_token_ids"]))
        t0 = time.time()
        outs = llm.generate([{"prompt_token_ids": list(rec["prompt_ids"])}], sp, use_tqdm=False)
        wall = time.time() - t0
        o = outs[0].outputs[0]
        return wall, [int(x) for x in o.token_ids], getattr(o, "finish_reason", None)

    gen_one(todo[0])  # warmup (untimed); result still recorded below on real pass
    print("[ar] warmup done", flush=True)

    fh = OUT.open("a")
    for rec in todo:
        wall, ids, fin = gen_one(rec)
        text = tok.decode(ids, skip_special_tokens=True)
        turn = {"idx": rec["idx"], "prompt_len": rec["prompt_len"], "n_gen": len(ids),
                "maxtok": MAXTOK, "finish_reason": fin, "wall_s": round(wall, 3),
                "tok_per_s": round(len(ids) / wall, 2) if wall > 0 else None,
                "gen_text": text}
        fh.write(json.dumps(turn) + "\n"); fh.flush(); os.fsync(fh.fileno())
        print(f"[ar] idx{rec['idx']:2d} n={len(ids)} fin={fin} wall={round(wall,3)}", flush=True)
    fh.close()
    print("[ar] DONE", flush=True)


if __name__ == "__main__":
    main()
