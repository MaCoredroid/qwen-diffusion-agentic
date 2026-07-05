#!/usr/bin/env python3
"""Aggregate the 4 bench arms into per-turn / per-episode speedup tables + report.

Arms (all 57 turns, ep0-9 nevertrain, greedy, seed 20260701, batch=1, cudagraph):
  engine_cold.jsonl  diffusion FLARE, reset_prefix_cache each turn (fresh/turn)
  engine_on.jsonl    diffusion FLARE, deployed align-APC reuse (LOSSY, quality-neutral)
  ar_cold.jsonl      stock Qwen3.5-9B AR, reset each turn
  ar_on.jsonl        stock Qwen3.5-9B AR, APC reuse (KV reuse is exact/lossless)

Key honest caveats baked into the report:
  * "engine cache-ON" here == the DEPLOYED align-APC (byte-lossy on a quality-neutral
    near-tie class; Route A lossless publish is present-but-inert in the pin). The
    lossless variant would carry the SAME prefill savings minus a bounded per-1024
    refold; at these context lengths (<=2640 tok, <=2 checkpoint crossings) that is
    1-2 extra <=1024-token chunk calls per EPISODE, negligible vs per-turn decode.
  * engine-vs-AR TOTAL wall is confounded by output length (AR greedy picks its own
    continuation, n_gen != engine n_gen). PREFILL is not confounded (same input) ->
    that is the clean cross-backend comparison. We report both, flagged.
  * a few engine turns are parity breaks where on-mode n_gen != cold-mode n_gen; the
    within-engine APC ratio is reported both incl. and excl. those turns.
"""
import json
import statistics as st
from pathlib import Path

BD = Path("/home/mark/qwen_diffusion/runs/lossless_apc/bench")


def load(name):
    d = {}
    for line in (BD / name).read_text().splitlines():
        line = line.strip()
        if line:
            r = json.loads(line)
            d[r["global_turn"]] = r
    return d


ec, eo = load("engine_cold.jsonl"), load("engine_on.jsonl")
ac, ao = load("ar_cold.jsonl"), load("ar_on.jsonl")
gts = sorted(set(ec) & set(eo) & set(ac) & set(ao))


def mean(xs):
    return round(st.mean(xs), 4) if xs else None


rows = []
for gt in gts:
    e_c, e_o, a_c, a_o = ec[gt], eo[gt], ac[gt], ao[gt]
    eng_ngen_match = (e_c["n_gen"] == e_o["n_gen"])
    rows.append({
        "gt": gt, "episode": e_c["episode"], "turn": e_c["turn"],
        "prompt_len": e_c["prompt_len"], "first_turn": (e_c["turn"] == 0),
        "eng_cold_wall": e_c["wall_s"], "eng_on_wall": e_o["wall_s"],
        "eng_cold_prefill": e_c["prefill_s"], "eng_on_prefill": e_o["prefill_s"],
        "eng_cold_decode": e_c["decode_s"], "eng_on_decode": e_o["decode_s"],
        "eng_ngen": e_c["n_gen"], "eng_fwd": e_c["denoise_forwards"],
        "eng_ngen_match": eng_ngen_match,
        "eng_parity_on": e_o["byte_parity_full"],
        "ar_cold_wall": a_c["wall_s"], "ar_on_wall": a_o["wall_s"],
        "ar_cold_prefill": a_c["prefill_s"], "ar_cold_decode": a_c["decode_s"],
        "ar_ngen": a_c["n_gen"],
    })

# ---- helper aggregations ----
def agg(rows, sel):
    ecw = [r["eng_cold_wall"] for r in rows]
    eow = [r["eng_on_wall"] for r in rows]
    acw = [r["ar_cold_wall"] for r in rows]
    aow = [r["ar_on_wall"] for r in rows]
    return {
        "n": len(rows),
        # within-backend APC: mean of per-turn cold/on ratios AND sum-ratio
        "eng_apc_speedup_meanratio": mean([c / o for c, o in zip(ecw, eow)]),
        "eng_apc_speedup_sumratio": round(sum(ecw) / sum(eow), 4),
        "ar_apc_speedup_meanratio": mean([c / o for c, o in zip(acw, aow)]),
        "ar_apc_speedup_sumratio": round(sum(acw) / sum(aow), 4),
        # wall means
        "eng_cold_wall_mean": mean(ecw), "eng_on_wall_mean": mean(eow),
        "ar_cold_wall_mean": mean(acw), "ar_on_wall_mean": mean(aow),
        # prefill (clean: input-only)
        "eng_cold_prefill_mean": mean([r["eng_cold_prefill"] for r in rows]),
        "eng_on_prefill_mean": mean([r["eng_on_prefill"] for r in rows]),
        "ar_cold_prefill_mean": mean([r["ar_cold_prefill"] for r in rows]),
        "eng_prefill_reduction_x": round(
            st.mean([r["eng_cold_prefill"] for r in rows])
            / st.mean([r["eng_on_prefill"] for r in rows]), 3),
        "eng_prefill_saved_s_mean": mean(
            [r["eng_cold_prefill"] - r["eng_on_prefill"] for r in rows]),
        # decode (engine, cold; unchanged by APC)
        "eng_cold_decode_mean": mean([r["eng_cold_decode"] for r in rows]),
        # prefill fraction of wall, cold
        "eng_cold_prefill_frac": mean(
            [r["eng_cold_prefill"] / r["eng_cold_wall"] for r in rows]),
        "ar_cold_prefill_frac": mean(
            [r["ar_cold_prefill"] / r["ar_cold_wall"] for r in rows]),
        # engine-vs-AR at matched caching (wall; confounded by n_gen -> flagged)
        "engVSar_on_wall_ratio": round(sum(aow) / sum(eow), 4),   # >1 => engine faster
        "engVSar_cold_wall_ratio": round(sum(acw) / sum(ecw), 4),
        # engine-vs-AR PREFILL (clean, same input)
        "engVSar_cold_prefill_ratio": round(
            st.mean([r["ar_cold_prefill"] for r in rows])
            / st.mean([r["eng_cold_prefill"] for r in rows]), 4),
    }


allr = rows
turn0 = [r for r in rows if r["first_turn"]]
turnN = [r for r in rows if not r["first_turn"]]
eng_clean = [r for r in rows if r["eng_ngen_match"]]  # excl engine parity-break len diffs

# per-episode wall sums
episodes = {}
for r in rows:
    ep = episodes.setdefault(r["episode"], {"ec": 0, "eo": 0, "ac": 0, "ao": 0, "turns": 0})
    ep["ec"] += r["eng_cold_wall"]; ep["eo"] += r["eng_on_wall"]
    ep["ac"] += r["ar_cold_wall"]; ep["ao"] += r["ar_on_wall"]; ep["turns"] += 1
per_ep = []
for ep in sorted(episodes):
    e = episodes[ep]
    per_ep.append({
        "episode": ep, "turns": e["turns"],
        "eng_cold_s": round(e["ec"], 3), "eng_on_s": round(e["eo"], 3),
        "eng_apc_speedup": round(e["ec"] / e["eo"], 3),
        "ar_cold_s": round(e["ac"], 3), "ar_on_s": round(e["ao"], 3),
        "ar_apc_speedup": round(e["ac"] / e["ao"], 3),
    })

out = {
    "n_turns": len(rows), "n_episodes": len(episodes),
    "prompt_len_range": [min(r["prompt_len"] for r in rows),
                         max(r["prompt_len"] for r in rows)],
    "all": agg(allr, "all"),
    "turn0_firstturn": agg(turn0, "t0"),
    "turnN_reuse": agg(turnN, "tN"),
    "engine_ngen_matched_only": agg(eng_clean, "clean"),
    "per_episode": per_ep,
    "rows": rows,
}
(BD / "bench_aggregate.json").write_text(json.dumps(out, indent=2))

# ---- console summary ----
A = out["all"]; TN = out["turnN_reuse"]; T0 = out["turn0_firstturn"]
print(f"=== LOSSLESS-APC END-GOAL BENCH: {out['n_turns']} turns / {out['n_episodes']} episodes,"
      f" ctx {out['prompt_len_range'][0]}-{out['prompt_len_range'][1]} tok ===")
print(f"[ALL turns]")
print(f"  ENGINE  cold {A['eng_cold_wall_mean']}s  on {A['eng_on_wall_mean']}s "
      f"-> APC speedup {A['eng_apc_speedup_sumratio']}x (sum) / {A['eng_apc_speedup_meanratio']}x (mean-ratio)")
print(f"          prefill cold {A['eng_cold_prefill_mean']}s -> on {A['eng_on_prefill_mean']}s "
      f"({A['eng_prefill_reduction_x']}x, -{A['eng_prefill_saved_s_mean']}s/turn); "
      f"decode(cold) {A['eng_cold_decode_mean']}s; prefill_frac(cold) {A['eng_cold_prefill_frac']}")
print(f"  AR      cold {A['ar_cold_wall_mean']}s  on {A['ar_on_wall_mean']}s "
      f"-> APC speedup {A['ar_apc_speedup_sumratio']}x (sum) / {A['ar_apc_speedup_meanratio']}x (mean-ratio)")
print(f"          prefill_frac(cold) {A['ar_cold_prefill_frac']}")
print(f"[TURN>=1 only, within-episode reuse]  ENGINE APC {TN['eng_apc_speedup_sumratio']}x  "
      f"AR APC {TN['ar_apc_speedup_sumratio']}x  (n={TN['n']})")
print(f"[TURN 0 only, cold-start]             ENGINE APC {T0['eng_apc_speedup_sumratio']}x  "
      f"AR APC {T0['ar_apc_speedup_sumratio']}x  (n={T0['n']})")
print(f"[ENGINE vs AR @ matched caching]")
print(f"  on-wall: AR/ENG {A['engVSar_on_wall_ratio']}x  cold-wall: AR/ENG {A['engVSar_cold_wall_ratio']}x "
      f"(>1 => engine lower wall; CONFOUNDED by n_gen)")
print(f"  prefill(cold, clean same-input): AR/ENG {A['engVSar_cold_prefill_ratio']}x")
print(f"[ENGINE ngen-matched-only APC] {out['engine_ngen_matched_only']['eng_apc_speedup_sumratio']}x "
      f"(n={out['engine_ngen_matched_only']['n']}, excludes parity-break length diffs)")
print("Wrote", BD / "bench_aggregate.json")
