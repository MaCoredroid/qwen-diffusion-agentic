#!/usr/bin/env python3
"""Aggregate the v2 battery (bidir + cudagraph) into aggregate.json + a printed
summary. CPU-only. Cross-checks HF wall from the reference records."""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/mark/qwen_diffusion")
V2 = ROOT / "runs/p2_engine_battery_v2/matched20_turns.jsonl"
REF = ROOT / "runs/p2_engine_bench/matched20_ref.json"
BIDIR_EAGER = ROOT / "runs/p2_engine_bench/parity_bidir/battery_bidir.jsonl"
OUT = ROOT / "runs/p2_engine_battery_v2/aggregate.json"

rows = [json.loads(l) for l in open(V2) if l.strip()]
rows.sort(key=lambda r: r["global_turn"])
assert len(rows) == 63, f"expected 63 turns, got {len(rows)}"
ref = {r["global_turn"]: r for r in json.loads(REF.read_text())}

def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    import math
    k = (len(xs) - 1) * p
    f = math.floor(k); c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)

par = sum(1 for r in rows if r["byte_parity_full"])
eng_exact = sum(1 for r in rows if r["eng_exact_arguments"])
hf_exact = sum(1 for r in rows if r["hf_exact_arguments"])
eng_valid = sum(1 for r in rows if r["eng_valid_tool_call"])
verify_ok = sum(1 for r in rows if (r.get("verify") or {}).get("ok"))
proj_nonzero = sum(1 for r in rows if (r.get("counters") or {}).get("value_projection_events", 0) != 0)

ep = defaultdict(list)
for r in rows:
    ep[r["episode"]].append(r)
eng_epi = sum(1 for e, ts in ep.items() if all(t["eng_exact_arguments"] for t in ts))
hf_epi = sum(1 for e, ts in ep.items() if all(t["hf_exact_arguments"] for t in ts))

walls = [r["wall_s"] for r in rows]
fwds = [r["denoise_forwards"] for r in rows]
ngen = [r["n_gen"] for r in rows]
fwd_total = sum(fwds)
tok_total = sum(ngen)
wall_total = sum(walls)
worst = max(rows, key=lambda r: r["wall_s"])

# HF full-63 wall from the reference records
hf_walls = [ref[r["global_turn"]]["hf_turn_wall_seconds"] for r in rows]
hf_fwds = [ref[r["global_turn"]]["hf_denoise_forwards_total"] for r in rows]
hf_wall_mean = st.mean(hf_walls)

# per-forward ms
per_fwd_ms_amortized = 1000.0 * wall_total / fwd_total  # total wall / total forwards
per_fwd_ms_turnmean = st.mean([r["per_forward_ms"] for r in rows])
# long-turn (>=80 fwd) settled ms/f
settled = [r["per_forward_ms"] for r in rows if r["denoise_forwards"] >= 80]

cg_total = sum(r.get("cg_pw_dispatches", 0) for r in rows)
cg_turns = sum(1 for r in rows if r.get("cg_pw_dispatches", 0) > 0)

breaks = []
for r in rows:
    if not r["byte_parity_full"]:
        breaks.append({
            "gt": r["global_turn"], "ep": r["episode"], "t": r["turn"],
            "first_div": r["first_divergence"], "pos_mod32": (r["first_divergence"] % 32) if r["first_divergence"] is not None else None,
            "n_gen": r["n_gen"], "n_ref": r["n_ref"], "finish": r["finish_reason"],
            "eng_exact": int(r["eng_exact_arguments"]), "hf_exact": int(r["hf_exact_arguments"]),
            "eng_valid": int(r["eng_valid_tool_call"]), "proj": (r.get("counters") or {}).get("value_projection_events"),
            "engtok_fd": r.get("engine_tok_at_fd"), "reftok_fd": r.get("ref_tok_at_fd"),
        })
eng_ne_hf = [{"gt": r["global_turn"], "eng": int(r["eng_exact_arguments"]), "hf": int(r["hf_exact_arguments"])}
             for r in rows if r["eng_exact_arguments"] != r["hf_exact_arguments"]]

# compare to bidir-eager anchor turn-by-turn
anchor = {r["global_turn"]: r for r in (json.loads(l) for l in open(BIDIR_EAGER) if l.strip())}
parity_matches_anchor = all(
    rows_by_gt[gt]["byte_parity_full"] == anchor[gt]["byte_parity_full"]
    for gt in anchor
) if (rows_by_gt := {r["global_turn"]: r for r in rows}) else False
ngen_matches_anchor = all(rows_by_gt[gt]["n_gen"] == anchor[gt]["n_gen"] for gt in anchor)
fwd_matches_anchor = all(rows_by_gt[gt]["denoise_forwards"] == anchor[gt]["denoise_forwards"] for gt in anchor)

agg = {
    "config": "bidir(VLLM_FLARE_BIDIR_PROBE=1) + cudagraph(PIECEWISE, VLLM_FLARE_CUDAGRAPH=1)",
    "n_turns": len(rows),
    "byte_parity_full": par,
    "eng_exact_args": eng_exact, "hf_exact_args": hf_exact,
    "eng_valid": eng_valid, "eng_episode_exact": eng_epi, "hf_episode_exact": hf_epi,
    "verify_ok": verify_ok, "value_projection_nonzero": proj_nonzero,
    "wall_mean": round(st.mean(walls), 3), "wall_p50": round(pct(walls, 0.5), 3),
    "wall_p90": round(pct(walls, 0.9), 3), "wall_p10": round(pct(walls, 0.1), 3),
    "wall_min": round(min(walls), 3), "wall_max": round(max(walls), 3),
    "wall_std": round(st.pstdev(walls), 3), "wall_total": round(wall_total, 3),
    "worst_turn": {"gt": worst["global_turn"], "n_gen": worst["n_gen"],
                   "fwd": worst["denoise_forwards"], "wall_s": worst["wall_s"]},
    "fwd_mean": round(st.mean(fwds), 2), "fwd_total": fwd_total,
    "tokens_total": tok_total, "tokens_per_forward": round(tok_total / fwd_total, 3),
    "per_forward_ms_amortized": round(per_fwd_ms_amortized, 2),
    "per_forward_ms_turnmean": round(per_fwd_ms_turnmean, 2),
    "per_forward_ms_settled_longturns": round(st.mean(settled), 2),
    "ngen_mean": round(st.mean(ngen), 1), "ngen_median": int(st.median(ngen)), "ngen_max": max(ngen),
    "hf_wall_mean": round(hf_wall_mean, 3), "hf_fwd_mean": round(st.mean(hf_fwds), 2),
    "finish_stop": sum(1 for r in rows if r["finish_reason"] == "stop"),
    "finish_length": sum(1 for r in rows if r["finish_reason"] == "length"),
    "speedup_vs_hf": round(hf_wall_mean / st.mean(walls), 3),
    "cg_pw_dispatches_total": cg_total, "cg_pw_turns": cg_turns,
    "parity_breaks": breaks, "eng_ne_hf": eng_ne_hf,
    "parity_matches_bidir_eager_anchor": parity_matches_anchor,
    "ngen_matches_anchor": ngen_matches_anchor, "fwd_matches_anchor": fwd_matches_anchor,
    "anchor_wall_mean": round(st.mean([anchor[gt]["wall_s"] for gt in anchor]), 3),
    "per_turn": [{"gt": r["global_turn"], "wall_s": r["wall_s"], "fwd": r["denoise_forwards"],
                  "n_gen": r["n_gen"], "n_ref": r["n_ref"], "parity": r["byte_parity_full"],
                  "per_forward_ms": r["per_forward_ms"], "eng_exact": int(r["eng_exact_arguments"])}
                 for r in rows],
}
OUT.write_text(json.dumps(agg, indent=2) + "\n")

# bars
bars = {"HF": 3.904, "guided_AR": 1.213, "M2": 1.120, "stock_agg": 0.741}
print(json.dumps({k: v for k, v in agg.items() if k not in ("per_turn",)}, indent=2))
print("\n=== BAR ADJUDICATION (mean s/turn = %.3f) ===" % agg["wall_mean"])
for name, bar in bars.items():
    verdict = "UNDER (beat)" if agg["wall_mean"] < bar else "OVER (miss)"
    print(f"  {name:12s} bar={bar:.3f}  -> {verdict}  ratio={agg['wall_mean']/bar:.3f}x")
