#!/usr/bin/env python3
"""STOCK-AR (guided-decoding) vLLM batched-rollout THROUGHPUT sweep (batch 1..16).

The AR baseline for the FLARE-engine throughput curve. Matches the endgame
scoreboard's AR arm: stock Qwen3.5-9B (snapshot c202236) on stock vLLM 0.23.0,
GUIDED decoding via structured_outputs = regex_from_qwen_xml_tool_schema (the exact
`guided_tool_call_regex` the scoreboard used). Guided decoding is the fair baseline
because -- like the hybrid engine -- it emits ONE valid Qwen-native XML tool call
per turn (plain free-form AR is not a rollout you can score without parsing luck).

Same never-train pool, same wave method, same samples/sec metric, same machine, same
per-turn max_tokens budget (n_ref+MARGIN) as the engine bench -> apples-to-apples.
AR gets its FAST production path (cudagraph on) so the engine's ratio is NOT inflated
by handicapping AR (scoreboard server used enforce_eager for parity; here throughput
is the question, so AR runs its fastest config -- the conservative choice for the
engine thesis). temp=0.7 seeded = the RL rollout mode (AR throughput is ~temp-insensitive).

One heavy process; RAM cage; foreground.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("VLLM_USE_V1", "1")
import numpy as np  # noqa: E402

ROOT = Path("/home/mark/qwen_diffusion")
HERE = ROOT / "runs/p2_batched_rollout_bench"
sys.path.insert(0, str(HERE))
REF = ROOT / "runs/p2_engine_nevertrain/nevertrain_ref.json"
MODEL = os.environ.get(
    "AR_MODEL",
    "/home/mark/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/"
    "snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a")

from gpu_sampler import GpuSampler, host_ram_peak_gb, gpu_snapshot  # noqa: E402

SEED = int(os.environ.get("BENCH_SEED", "20260701"))
MARGIN = int(os.environ.get("BENCH_MARGIN", "16"))
MAXSEQ = int(os.environ.get("BENCH_MAXSEQ", "16"))
BATCHES = [int(x) for x in os.environ.get("BENCH_BATCHES", "1 2 4 8 16").split()]
POOL_N = int(os.environ.get("BENCH_POOL_N", "48"))
PMIN = int(os.environ.get("BENCH_PMIN", "400"))
PMAX = int(os.environ.get("BENCH_PMAX", "1300"))
TEMP = float(os.environ.get("BENCH_TEMP", "0.7"))
GMU = float(os.environ.get("AR_GMU", "0.66"))
EAGER = os.environ.get("AR_EAGER", "0") == "1"


# --- guided_tool_call_regex: copied verbatim from scripts/eval_flare_northstar_matched.py
#     (the scoreboard's regex_from_qwen_xml_tool_schema) to avoid importing the heavy
#     HF/torch eval module. Pure-python; byte-identical logic. ---
def regex_literal(text):
    return re.escape(str(text))


def schema_type(schema):
    expected = schema.get("type") if isinstance(schema, dict) else None
    if isinstance(expected, list):
        return next((i for i in expected if i != "null"), expected[0] if expected else None)
    return expected


def guided_value_regex(schema):
    if not isinstance(schema, dict):
        return "[^<]*"
    ev = schema.get("enum")
    if isinstance(ev, list) and ev:
        ch = []
        for v in ev:
            if isinstance(v, str):
                ch.append(regex_literal(v))
            elif isinstance(v, bool):
                ch.extend([str(v).lower(), str(v)])
            elif v is None:
                ch.append("null")
            else:
                ch.append(regex_literal(json.dumps(v, ensure_ascii=False)))
        return "(?:" + "|".join(dict.fromkeys(ch)) + ")"
    e = schema_type(schema)
    return {
        "integer": "-?[0-9]+",
        "number": "-?(?:[0-9]+(?:\\.[0-9]+)?|\\.[0-9]+)(?:[eE][+-]?[0-9]+)?",
        "boolean": "(?:true|false|True|False)",
        "array": "\\[[^<]*\\]",
        "object": "\\{[^<]*\\}",
    }.get(e, "[^<]*")


def guided_tool_call_regex(tools):
    alts = []
    for tool in tools or []:
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        name = str(fn["name"])
        schema = fn.get("parameters") or {}
        props = schema.get("properties") if isinstance(schema, dict) else {}
        props = props if isinstance(props, dict) else {}
        required = set(schema.get("required") or []) if isinstance(schema, dict) else set()
        params = []
        for pn, ps in props.items():
            body = (f"<parameter={regex_literal(pn)}>\\n"
                    f"{guided_value_regex(ps)}\\n</parameter>\\n")
            params.append(body if pn in required else f"(?:{body})?")
        alts.append("<tool_call>\\n" + f"<function={regex_literal(name)}>\\n"
                    + "".join(params) + "</function>\\n</tool_call>")
    if not alts:
        return ("<tool_call>\\n<function=[^>]+>\\n(?:<parameter=[^>]+>\\n[^<]*\\n"
                "</parameter>\\n)*</function>\\n</tool_call>")
    return "(?:" + "|".join(alts) + ")"


def main():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    records = json.loads(REF.read_text())
    pool = sorted((r for r in records if PMIN <= r["prompt_len"] <= PMAX),
                  key=lambda r: r["global_turn"])[:POOL_N]
    assert len(pool) >= 48, f"pool too small: {len(pool)}"
    # precompute per-turn guided regex (== scoreboard structured output)
    for r in pool:
        r["_regex"] = guided_tool_call_regex(r["tools"])
    print(f"[ar] model={Path(MODEL).name} pool_n={len(pool)} guided=regex_qwen_xml "
          f"lens={min(r['prompt_len'] for r in pool)}..{max(r['prompt_len'] for r in pool)} "
          f"nref_mean={np.mean([r['n_ref'] for r in pool]):.1f} batches={BATCHES} "
          f"temp={TEMP} gmu={GMU} eager={EAGER}", flush=True)

    kw = dict(model=MODEL, trust_remote_code=True, max_model_len=4096,
              gpu_memory_utilization=GMU, max_num_seqs=MAXSEQ,
              max_num_batched_tokens=4096, enable_prefix_caching=True, seed=SEED,
              mamba_cache_mode="align", mamba_block_size=1024, enforce_eager=EAGER)
    t0 = time.time()
    engine = LLM(**kw)
    boot_s = time.time() - t0
    vc = engine.llm_engine.vllm_config
    print(f"[ar] booted boot_s={boot_s:.1f} eager={vc.model_config.enforce_eager} "
          f"gmu={GMU} max_num_seqs={vc.scheduler_config.max_num_seqs}", flush=True)

    def make_sp(rec, temp):
        greedy = temp <= 0.0
        return SamplingParams(
            max_tokens=rec["n_ref"] + MARGIN, temperature=temp,
            top_p=1.0, seed=SEED + int(rec["global_turn"]),
            stop_token_ids=sorted(int(x) for x in rec["stop_token_ids"]),
            structured_outputs=StructuredOutputsParams(regex=rec["_regex"]))

    def run_wave(recs, temp):
        try:
            engine.reset_prefix_cache()
        except Exception:
            pass
        prompts = [{"prompt_token_ids": list(r["prompt_ids"])} for r in recs]
        sps = [make_sp(r, temp) for r in recs]
        outs = engine.generate(prompts, sps, use_tqdm=False)
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        fins = [o.outputs[0].finish_reason for o in outs]
        return ntok, fins

    for B in sorted(set(BATCHES)):
        run_wave(pool[:B], TEMP)

    points = []

    def bench_point(B, temp, variant):
        waves = [pool[i:i + B] for i in range(0, len(pool), B)]
        waves = [w for w in waves if len(w) == B]
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
        row = {
            "engine": "stock_AR_vllm_guided", "config": f"stock vllm 0.23.0 guided (eager={EAGER})",
            "variant": variant, "batch": B, "temp": temp,
            "n_turns": n_turns, "n_waves": len(waves),
            "wall_s": round(wall, 3),
            "samples_per_sec": round(n_turns / wall, 4),
            "tokens_per_sec": round(total_tok / wall, 2),
            "total_gen_tokens": total_tok,
            "mean_gen_tokens_per_turn": round(total_tok / n_turns, 1),
            # AR forward == 1 decode token; per-decode-step ms:
            "per_decode_step_ms": round(1000.0 * wall / total_tok, 3) if total_tok else None,
            "mean_wave_wall_s": round(float(np.mean(wave_walls)), 4),
            "p90_wave_wall_s": round(float(np.percentile(wave_walls, 90)), 4) if wave_walls else None,
            "finish_reasons": fin_counts,
            "gpu_mem_used_mb": gpu_snapshot()[1],
            "host_ram_peak_gb": host_ram_peak_gb(),
            "seed": SEED,
        }
        row.update(gs)
        points.append(row)
        print(f"[ar] {variant} B={B:2d} n={n_turns} wall={wall:.2f}s "
              f"samp/s={row['samples_per_sec']:.3f} tok/s={row['tokens_per_sec']:.0f} "
              f"gen_tok/turn={row['mean_gen_tokens_per_turn']} step_ms={row['per_decode_step_ms']} "
              f"util~{gs['gpu_util_mean_pct']}% mem={row['gpu_mem_used_mb']}MB "
              f"ram_peak={row['host_ram_peak_gb']}GB", flush=True)
        return row

    for B in BATCHES:
        try:
            bench_point(B, TEMP, "rl_temp0.7")
        except Exception as e:  # noqa: BLE001
            points.append({"variant": "rl_temp0.7", "batch": B, "temp": TEMP,
                           "error": type(e).__name__, "detail": repr(e)[:200]})
            print(f"[ar] rl_temp0.7 B={B} ERROR: {repr(e)[:160]}", flush=True)

    ok = [p for p in points if "error" not in p]
    base = next((p for p in ok if p["batch"] == 1), ok[0] if ok else None)
    if base:
        for p in ok:
            p["speedup_samples_vs_b1"] = round(p["samples_per_sec"] / base["samples_per_sec"], 3)
            p["speedup_tokens_vs_b1"] = round(p["tokens_per_sec"] / base["tokens_per_sec"], 3)

    with (HERE / "ar_points.jsonl").open("w") as fh:
        for p in points:
            fh.write(json.dumps(p) + "\n")
    summ = {
        "engine": "stock_AR_vllm_guided",
        "config": f"stock vllm 0.23.0 guided(regex_qwen_xml) eager={EAGER}",
        "model": MODEL, "pool_n": len(pool), "prompt_len_band": [PMIN, PMAX],
        "seed": SEED, "gpu_memory_utilization": GMU, "max_num_seqs": MAXSEQ,
        "temp_primary": TEMP, "boot_s": round(boot_s, 1), "points": points,
    }
    (HERE / "ar_throughput.json").write_text(json.dumps(summ, indent=2))
    print("[ar] SPEEDUP vs B=1: " + " | ".join(
        f"B{p['batch']}={p.get('speedup_samples_vs_b1','?')}x({p['samples_per_sec']}/s)"
        for p in ok), flush=True)
    print("[ar] DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("[ar] FATAL:", repr(e), flush=True)
        traceback.print_exc()
        raise
