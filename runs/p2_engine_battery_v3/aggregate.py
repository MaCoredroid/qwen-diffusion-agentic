#!/usr/bin/env python3
"""Aggregate the v3 promotion-attempt battery into aggregate.json + a printed
summary. CPU-only. Reproduces the v2 headline stats AND adds:
  (a) byte-determinism cross-check vs the v2 battery (same engine pin e5496cc),
  (b) the per-turn FRESH-CONTEXT parity certificate (fresh-boot per turn) vs the
      APC-on speed row -> the cache-path-invariant structural break set,
  (c) the residual-gap breakdown to the stock-agg 0.741 s/turn bar.
"""
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/mark/qwen_diffusion")
V3D = ROOT / "runs/p2_engine_battery_v3"
APC = V3D / "matched20_turns.jsonl"                 # APC-on full-63 (speed row = v2 protocol)
FRESH = V3D / "parity_cert_freshboot.jsonl"         # per-turn fresh-boot parity certificate
V2 = ROOT / "runs/p2_engine_battery_v2/matched20_turns.jsonl"
REF = ROOT / "runs/p2_engine_bench/matched20_ref.json"
BIDIR_EAGER = ROOT / "runs/p2_engine_bench/parity_bidir/battery_bidir.jsonl"
OUT = V3D / "aggregate.json"

# stock-agg per-forward target math
STOCK_AGG_BAR = 0.741
WEIGHT_STREAM_FLOOR_MS = 10.5  # measured on this card (per memory / OPT-4 notes)

rows = [json.loads(l) for l in open(APC) if l.strip()]
rows.sort(key=lambda r: r["global_turn"])
assert len(rows) == 63, f"expected 63 APC-on turns, got {len(rows)}"
ref = {r["global_turn"]: r for r in json.loads(REF.read_text())}
by = {r["global_turn"]: r for r in rows}


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
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
fwd_total = sum(fwds); tok_total = sum(ngen); wall_total = sum(walls)
worst = max(rows, key=lambda r: r["wall_s"])
hf_walls = [ref[r["global_turn"]]["hf_turn_wall_seconds"] for r in rows]
hf_fwds = [ref[r["global_turn"]]["hf_denoise_forwards_total"] for r in rows]
hf_wall_mean = st.mean(hf_walls)
per_fwd_ms_amortized = 1000.0 * wall_total / fwd_total
settled = [r["per_forward_ms"] for r in rows if r["denoise_forwards"] >= 80]
cg_total = sum(r.get("cg_pw_dispatches", 0) for r in rows)
cg_turns = sum(1 for r in rows if r.get("cg_pw_dispatches", 0) > 0)

apc_breaks = sorted(r["global_turn"] for r in rows if not r["byte_parity_full"])
eng_ne_hf = [{"gt": r["global_turn"], "eng": int(r["eng_exact_arguments"]), "hf": int(r["hf_exact_arguments"])}
             for r in rows if r["eng_exact_arguments"] != r["hf_exact_arguments"]]

# ---- byte-determinism cross-check vs v2 (same pin) ----
v2rows = {r["global_turn"]: r for r in (json.loads(l) for l in open(V2) if l.strip())}
v2_ngen_ident = all(by[g]["n_gen"] == v2rows[g]["n_gen"] for g in by)
v2_fwd_ident = all(by[g]["denoise_forwards"] == v2rows[g]["denoise_forwards"] for g in by)
v2_parity_ident = all(by[g]["byte_parity_full"] == v2rows[g]["byte_parity_full"] for g in by)
v2_exact_ident = all(by[g]["eng_exact_arguments"] == v2rows[g]["eng_exact_arguments"] for g in by)
v2_fd_ident = all(by[g]["first_divergence"] == v2rows[g]["first_divergence"] for g in by)

# ---- bidir-eager anchor cross-check ----
anchor = {r["global_turn"]: r for r in (json.loads(l) for l in open(BIDIR_EAGER) if l.strip())}
anchor_parity_ident = all(by[g]["byte_parity_full"] == anchor[g]["byte_parity_full"] for g in anchor)

# ---- fresh-context parity certificate ----
fresh_summary = None
if FRESH.exists():
    frows = {r["global_turn"]: r for r in (json.loads(l) for l in open(FRESH) if l.strip())}
    fresh_gts = sorted(frows)
    fresh_complete = (len(fresh_gts) == 63)
    fresh_par = sum(1 for g in fresh_gts if frows[g]["byte_parity_full"])
    fresh_exact = sum(1 for g in fresh_gts if frows[g]["eng_exact_arguments"])
    fresh_hf_exact = sum(1 for g in fresh_gts if frows[g]["hf_exact_arguments"])
    fresh_valid = sum(1 for g in fresh_gts if frows[g]["eng_valid_tool_call"])
    fresh_breaks = sorted(g for g in fresh_gts if not frows[g]["byte_parity_full"])
    # compare break sets on the overlap
    overlap = [g for g in fresh_gts]
    apc_break_set = set(apc_breaks)
    fresh_break_set = set(fresh_breaks)
    invariant = sorted(apc_break_set & fresh_break_set)                    # break in BOTH
    apc_only = sorted(apc_break_set - fresh_break_set)                      # break APC, parity fresh
    fresh_only = sorted(fresh_break_set - apc_break_set)                    # parity APC, break fresh
    # exact deltas fresh vs apc on overlap
    fresh_vs_apc_exact_flips = [
        {"gt": g, "apc": int(by[g]["eng_exact_arguments"]), "fresh": int(frows[g]["eng_exact_arguments"])}
        for g in fresh_gts if by[g]["eng_exact_arguments"] != frows[g]["eng_exact_arguments"]]
    fresh_summary = {
        "protocol": "per-turn fresh boot (cold prefix+mamba cache, single process/turn)",
        "n_measured": len(fresh_gts), "complete_63": fresh_complete,
        "missing": [g for g in range(63) if g not in frows],
        "parity": fresh_par, "exact": fresh_exact, "hf_exact_on_measured": fresh_hf_exact,
        "valid": fresh_valid, "breaks": fresh_breaks,
        "vs_apc": {
            "apc_break_set": apc_breaks, "fresh_break_set": fresh_breaks,
            "invariant_breaks_both_paths": invariant,
            "apc_only_breaks_resolve_fresh": apc_only,
            "fresh_only_breaks_hidden_by_apc": fresh_only,
            "exact_flips_fresh_vs_apc": fresh_vs_apc_exact_flips,
        },
    }

# ---- residual-gap breakdown to stock-agg 0.741 ----
mean_wall = st.mean(walls)
fwd_per_turn = fwd_total / len(rows)
per_fwd_needed_for_bar = 1000.0 * STOCK_AGG_BAR / fwd_per_turn  # ms/fwd to hit 0.741 at current fwd/turn
gap_breakdown = {
    "mean_s_per_turn": round(mean_wall, 4),
    "stock_agg_bar": STOCK_AGG_BAR,
    "ratio_over_bar": round(mean_wall / STOCK_AGG_BAR, 3),
    "fwd_per_turn": round(fwd_per_turn, 2),
    "per_forward_ms_amortized": round(per_fwd_ms_amortized, 2),
    "per_forward_ms_needed_for_bar_at_current_fwd_per_turn": round(per_fwd_needed_for_bar, 2),
    "weight_stream_floor_ms": WEIGHT_STREAM_FLOOR_MS,
    "reducible_overhead_ms_above_floor": round(per_fwd_ms_amortized - WEIGHT_STREAM_FLOOR_MS, 2),
    "ms_to_cut_per_forward_to_reach_bar": round(per_fwd_ms_amortized - per_fwd_needed_for_bar, 2),
    "headroom_between_floor_and_bar_target_ms": round(per_fwd_needed_for_bar - WEIGHT_STREAM_FLOOR_MS, 2),
    "note": ("bar reachable at current fwd/turn iff per-forward can fall from "
             f"{per_fwd_ms_amortized:.1f} to {per_fwd_needed_for_bar:.1f} ms; floor is "
             f"{WEIGHT_STREAM_FLOOR_MS} ms so the bar target sits "
             f"{per_fwd_needed_for_bar - WEIGHT_STREAM_FLOOR_MS:.1f} ms above the streaming floor "
             "(reachable). The extra cut is OPT-4 Part 1: fewer rows in the CL=32 gemm/attn "
             "(variable width) + width-1 GDN routed to fused_recurrent, which also lowers fwd/turn."),
}

agg = {
    "engine_pin": "e5496cc (qwen3_5-flare-modelstate), OPT-4 Part 1 UNLANDED (Task #37 pending)",
    "config_apc_on": "bidir(VLLM_FLARE_BIDIR_PROBE=1)+cudagraph(PIECEWISE,VLLM_FLARE_CUDAGRAPH=1), APC on",
    "n_turns": len(rows),
    "byte_parity_full_apc_on": par, "apc_breaks": apc_breaks,
    "eng_exact_args": eng_exact, "hf_exact_args": hf_exact,
    "eng_valid": eng_valid, "eng_episode_exact": eng_epi, "hf_episode_exact": hf_epi,
    "verify_ok": verify_ok, "value_projection_nonzero": proj_nonzero, "eng_ne_hf": eng_ne_hf,
    "wall_mean": round(mean_wall, 3), "wall_p50": round(pct(walls, 0.5), 3),
    "wall_p90": round(pct(walls, 0.9), 3), "wall_min": round(min(walls), 3),
    "wall_max": round(max(walls), 3), "wall_std": round(st.pstdev(walls), 3),
    "wall_total": round(wall_total, 3),
    "worst_turn": {"gt": worst["global_turn"], "n_gen": worst["n_gen"],
                   "fwd": worst["denoise_forwards"], "wall_s": worst["wall_s"]},
    "fwd_mean": round(st.mean(fwds), 2), "fwd_total": fwd_total,
    "tokens_total": tok_total, "tokens_per_forward": round(tok_total / fwd_total, 3),
    "per_forward_ms_amortized": round(per_fwd_ms_amortized, 2),
    "per_forward_ms_settled_longturns": round(st.mean(settled), 2) if settled else None,
    "hf_wall_mean": round(hf_wall_mean, 3), "hf_fwd_mean": round(st.mean(hf_fwds), 2),
    "speedup_vs_hf": round(hf_wall_mean / mean_wall, 3),
    "cg_pw_dispatches_total": cg_total, "cg_pw_turns": cg_turns,
    "determinism_vs_v2": {"ngen_identical": v2_ngen_ident, "fwd_identical": v2_fwd_ident,
                          "parity_identical": v2_parity_ident, "exact_identical": v2_exact_ident,
                          "first_div_identical": v2_fd_ident},
    "parity_matches_bidir_eager_anchor": anchor_parity_ident,
    "fresh_context_certificate": fresh_summary,
    "residual_gap_to_stock_agg": gap_breakdown,
    "per_turn": [{"gt": r["global_turn"], "wall_s": r["wall_s"], "fwd": r["denoise_forwards"],
                  "n_gen": r["n_gen"], "n_ref": r["n_ref"], "parity": r["byte_parity_full"],
                  "per_forward_ms": r["per_forward_ms"], "eng_exact": int(r["eng_exact_arguments"]),
                  "hf_exact": int(r["hf_exact_arguments"])} for r in rows],
}
OUT.write_text(json.dumps(agg, indent=2) + "\n")

bars = {"HF": 3.904, "guided_AR": 1.213, "M2": 1.120, "stock_agg": 0.741}
print(json.dumps({k: v for k, v in agg.items() if k != "per_turn"}, indent=2))
print("\n=== BAR ADJUDICATION (APC-on mean s/turn = %.3f) ===" % agg["wall_mean"])
for name, bar in bars.items():
    verdict = "UNDER (beat)" if agg["wall_mean"] < bar else "OVER (miss)"
    print(f"  {name:12s} bar={bar:.3f}  -> {verdict}  ratio={agg['wall_mean']/bar:.3f}x")
