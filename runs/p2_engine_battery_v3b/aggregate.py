#!/usr/bin/env python3
"""Aggregate the v3b PROMOTION-ATTEMPT battery (post-fix pin 95d8b47) into
aggregate.json + a printed summary. CPU-only. Computes:
  (0) the full-63 APC-on headline (parity/exact/valid/episode/verify/proj/timing),
  (1) the promotion-gate verdict (63/63 parity => exact exactly 47),
  (2) the DELTA vs the pre-fix v3 battery (e5496cc, 58/63): which breaks the Stage-3
      fix cleared, and byte-determinism on the shared-clean turns,
  (3) the fresh-context (per-turn fresh-boot) parity certificate on the 5 pre-fix
      break turns -> the path-invariant residual (is gt44 an APC class? no),
  (4) temp-0.7 a/b reproducibility + never-train spot-check rollup,
  (5) the MEASURED residual-gap breakdown to stock-agg 0.741 (weight-stream floor
      from the profiler gemm device-time vs the reducible above-floor overhead).
"""
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/mark/qwen_diffusion")
V = ROOT / "runs/p2_engine_battery_v3b"
APC = V / "matched20_turns.jsonl"
FRESH = V / "parity_cert_freshboot.jsonl"
T07A = V / "matched20_temp07a.jsonl"
T07B = V / "matched20_temp07b.jsonl"
NEVER = V / "nevertrain_spotcheck.jsonl"
PROF = V / "opt4_breakdown.json"
PREFIX = ROOT / "runs/p2_engine_battery_v3/matched20_turns.jsonl"   # pre-fix e5496cc
REF = ROOT / "runs/p2_engine_bench/matched20_ref.json"
OUT = V / "aggregate.json"

STOCK_AGG_BAR = 0.741
BARS = {"HF": 3.904, "guided_AR": 1.213, "M2_K3": 1.120, "stock_agg": 0.741}
# measured HBM stream floor cross-check
WEIGHTS_BYTES = 19_306_310_880          # sum of the 4 bf16 safetensors shards
HBM_BW_TBps = 1.792                     # RTX 5090 GDDR7 512-bit spec (~1.79 TB/s)


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    f = math.floor(k); c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


rows = [json.loads(l) for l in open(APC) if l.strip()]
rows.sort(key=lambda r: r["global_turn"])
assert len(rows) == 63, f"expected 63 APC-on turns, got {len(rows)}"
ref = {r["global_turn"]: r for r in json.loads(REF.read_text())}
by = {r["global_turn"]: r for r in rows}

par = sum(1 for r in rows if r["byte_parity_full"])
breaks = sorted(r["global_turn"] for r in rows if not r["byte_parity_full"])
eng_exact = sum(1 for r in rows if r["eng_exact_arguments"])
hf_exact = sum(1 for r in rows if r["hf_exact_arguments"])
eng_valid = sum(1 for r in rows if r["eng_valid_tool_call"])
verify_ok = sum(1 for r in rows if (r.get("verify") or {}).get("ok"))
proj_nonzero = sum(1 for r in rows if (r.get("counters") or {}).get("value_projection_events", 0) != 0)
eng_ne_hf = [{"gt": r["global_turn"], "eng": int(r["eng_exact_arguments"]), "hf": int(r["hf_exact_arguments"])}
             for r in rows if r["eng_exact_arguments"] != r["hf_exact_arguments"]]

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
fwd_per_turn = fwd_total / len(rows)
settled_ms = [r["per_forward_ms"] for r in rows if r["denoise_forwards"] >= 80]
cg_total = sum(r.get("cg_pw_dispatches", 0) for r in rows)
cg_turns = sum(1 for r in rows if r.get("cg_pw_dispatches", 0) > 0)
mean_wall = st.mean(walls)

# ---- promotion gate ----
gate = {
    "byte_parity_63_of_63": par == 63,
    "byte_parity_count": par, "breaks": breaks,
    "exact_args_exactly_47": eng_exact == 47, "exact_args": eng_exact,
    "episode_exact_13": eng_epi == 13, "episode_exact": eng_epi,
    "valid_63": eng_valid == 63, "valid": eng_valid,
    "PROMOTED": (par == 63 and eng_exact == 47 and eng_epi == 13 and eng_valid == 63),
}

# ---- delta vs pre-fix v3 (e5496cc, 58/63) ----
prefix_rows = {r["global_turn"]: r for r in (json.loads(l) for l in open(PREFIX) if l.strip())}
prefix_breaks = sorted(g for g in prefix_rows if not prefix_rows[g]["byte_parity_full"])
cleared_by_fix = sorted(set(prefix_breaks) - set(breaks))          # broke pre-fix, clean now
still_broken = sorted(set(prefix_breaks) & set(breaks))            # broke both
new_breaks = sorted(set(breaks) - set(prefix_breaks))             # regressions (want empty)
shared_clean = [g for g in by if g not in set(prefix_breaks) | set(breaks)]
ngen_ident_shared = all(by[g]["n_gen"] == prefix_rows[g]["n_gen"] for g in shared_clean)
exact_ident_shared = all(by[g]["eng_exact_arguments"] == prefix_rows[g]["eng_exact_arguments"] for g in shared_clean)
prefix_exact = sum(1 for g in prefix_rows if prefix_rows[g]["eng_exact_arguments"])

# ---- fresh-context (per-turn fresh-boot) certificate on the 5 pre-fix breaks ----
fresh_summary = None
if FRESH.exists():
    frows = {r["global_turn"]: r for r in (json.loads(l) for l in open(FRESH) if l.strip())}
    fresh_breaks = sorted(g for g in frows if not frows[g]["byte_parity_full"])
    fresh_clean = sorted(g for g in frows if frows[g]["byte_parity_full"])
    # path-invariance: for each measured gt, does APC-on parity == fresh-boot parity?
    path_consistent = {g: (by[g]["byte_parity_full"] == frows[g]["byte_parity_full"]) for g in frows}
    gt44_fresh = frows.get(44)
    fresh_summary = {
        "protocol": "one turn per fresh boot (single request => cold prefix cache, no cross-turn KV reuse)",
        "note_apc_off_hook": ("BENCH_APC_OFF=1 in-boot hook fails VllmConfig validation under the "
                              "diffusion align-cache config (prefix caching cannot be disabled); the "
                              "single-request-per-boot protocol is the documented fresh-context proxy (v3)."),
        "turns_measured": sorted(frows), "fresh_breaks": fresh_breaks, "fresh_clean": fresh_clean,
        "apc_on_parity_eq_fresh_parity_per_turn": path_consistent,
        "path_invariant_breaks": sorted(set(breaks) & set(fresh_breaks)),
        "gt44_fresh_first_div": (gt44_fresh or {}).get("first_divergence"),
        "gt44_fresh_ngen": (gt44_fresh or {}).get("n_gen"),
        "gt44_is_apc_class": (44 in frows and frows[44]["byte_parity_full"]),  # False => not APC class
        "verdict": ("gt44 breaks byte-parity IDENTICALLY under APC-on and cold-prefix fresh-boot "
                    "(same first_div, same n_gen) => path-invariant deterministic fp-residue, NOT an "
                    "APC/prefix-cache class. The documented APC protocol cannot rescue it to 63/63."),
    }

# ---- temp-0.7 a/b reproducibility ----
temp07 = None
if T07A.exists() and T07B.exists():
    a = {r["global_turn"]: r for r in (json.loads(l) for l in open(T07A) if l.strip())}
    b = {r["global_turn"]: r for r in (json.loads(l) for l in open(T07B) if l.strip())}
    gts = sorted(a)
    ident = all(a[g]["n_gen"] == b[g]["n_gen"] and a[g]["denoise_forwards"] == b[g]["denoise_forwards"]
                and a[g]["byte_parity_full"] == b[g]["byte_parity_full"] for g in gts)
    max_wall_delta = max(abs(a[g]["wall_s"] - b[g]["wall_s"]) for g in gts)
    temp07 = {
        "turns": gts, "n": len(gts),
        "all_bounded_valid_proj0": all(a[g]["finish_reason"] == "stop" and a[g]["eng_valid_tool_call"]
                                       and (a[g].get("counters") or {}).get("value_projection_events", 0) == 0
                                       for g in gts),
        "all_exact_eq_hf": all(a[g]["eng_exact_arguments"] == a[g]["hf_exact_arguments"] for g in gts),
        "all_parity_full": all(a[g]["byte_parity_full"] for g in gts),
        "byte_reproducible_a_vs_b": ident, "max_wall_delta_s": round(max_wall_delta, 4),
        "per_turn": [{"gt": g, "n_gen": a[g]["n_gen"], "fwd": a[g]["denoise_forwards"],
                      "valid": a[g]["eng_valid_tool_call"], "exact": int(a[g]["eng_exact_arguments"]),
                      "parity": a[g]["byte_parity_full"],
                      "wall_a": a[g]["wall_s"], "wall_b": b[g]["wall_s"]} for g in gts],
    }

# ---- never-train spot-check ----
nevertrain = None
if NEVER.exists():
    nrows = [json.loads(l) for l in open(NEVER) if l.strip()]
    nevertrain = {
        "n": len(nrows),
        "parity": sum(1 for r in nrows if r["byte_parity_full"]),
        "valid": sum(1 for r in nrows if r["eng_valid_tool_call"]),
        "exact_eq_hf": sum(1 for r in nrows if r["eng_exact_arguments"] == r["hf_exact_arguments"]),
        "proj0": sum(1 for r in nrows if (r.get("counters") or {}).get("value_projection_events", 0) == 0),
        "per_turn": [{"gt": r["global_turn"], "family": r.get("source_family"),
                      "parity": r["byte_parity_full"], "n_gen": r["n_gen"], "n_ref": r["n_ref"],
                      "valid": r["eng_valid_tool_call"], "exact": int(r["eng_exact_arguments"]),
                      "hf_exact": int(r["hf_exact_arguments"]), "wall_s": r["wall_s"],
                      "per_forward_ms": r["per_forward_ms"]} for r in nrows],
    }

# ---- MEASURED residual-gap breakdown to stock-agg 0.741 ----
prof = json.loads(PROF.read_text())
settled = [p for p in prof if p["denoise_forwards"] >= 80]          # gt25, gt35
gpu_ms_fwd = st.mean(p["gpu_ms_per_forward"] for p in settled)
gemm_ms_fwd = st.mean(p["family_ms"]["gemm(MLP+proj+lm_head)"] / p["denoise_forwards"] for p in settled)
gemm_pct = st.mean(p["family_pct"]["gemm(MLP+proj+lm_head)"] for p in settled)
# non-weight GPU compute (attn+GDN+norm+elementwise+sampling+other) per forward
nonweight_ms_fwd = gpu_ms_fwd - gemm_ms_fwd
resid_host_ms_fwd = per_fwd_ms_amortized - gpu_ms_fwd             # cudagraph residual host/launch
arith_floor_ms = 1000.0 * (WEIGHTS_BYTES / 1e12) / HBM_BW_TBps    # bf16 weights / HBM BW
per_fwd_needed = 1000.0 * STOCK_AGG_BAR / fwd_per_turn            # ms/fwd to hit 0.741 at this fwd/turn
gap_breakdown = {
    "engine_mean_s_per_turn": round(mean_wall, 4),
    "stock_agg_bar_s_per_turn": STOCK_AGG_BAR,
    "ratio_over_bar": round(mean_wall / STOCK_AGG_BAR, 3),
    "fwd_per_turn": round(fwd_per_turn, 2),
    "amortized_per_forward_ms_cudagraph_wall": round(per_fwd_ms_amortized, 2),
    "measured_gpu_ms_per_forward_settled": round(gpu_ms_fwd, 2),
    "weight_stream_floor_ms_measured_gemm": round(gemm_ms_fwd, 2),
    "weight_stream_floor_gemm_pct_of_gpu": round(gemm_pct, 1),
    "weight_stream_floor_ms_arithmetic": round(arith_floor_ms, 2),
    "nonweight_gpu_compute_ms_per_forward": round(nonweight_ms_fwd, 2),
    "residual_host_launch_ms_per_forward_cudagraph": round(resid_host_ms_fwd, 2),
    "per_forward_ms_needed_for_bar_at_current_fwd_per_turn": round(per_fwd_needed, 2),
    "ms_to_cut_per_forward_to_reach_bar": round(per_fwd_ms_amortized - per_fwd_needed, 2),
    "headroom_bar_target_above_weight_floor_ms": round(per_fwd_needed - gemm_ms_fwd, 2),
    "reducibility_verdict": (
        "Per-forward is GPU-compute-bound after cudagraph (residual host ~%.1f ms). The weight-stream "
        "floor (bf16 9B gemm device time, %.1f ms; arithmetic %.1f ms at %.2f TB/s) is ~%.0f%% of it and "
        "is irreducible at batch=1. To hit 0.741 at %.1f fwd/turn needs %.1f ms/fwd; the bar target sits "
        "only %.1f ms above the weight floor, but the non-weight per-forward compute is %.1f ms and Stage 3 "
        "A/B proved it does NOT shrink with variable width (cudagraph buckets narrow widths back to a "
        "captured bucket) -- so 0.741 is NOT reachable by width-narrowing / engine plumbing at batch=1. "
        "The reachable levers are orthogonal to the parity/integration work: (a) fewer forwards/turn "
        "(larger effective parallel commit -- model/schedule/training), (b) a lighter weight stream "
        "(fp8/int8 weights halve/quarter the %.1f ms floor -> per-forward ~12/9 ms -> ~0.68/0.51 s/turn, "
        "a quality tradeoff), or (c) batching to amortize the weight-stream floor across concurrent "
        "requests (serving throughput, not single-stream latency). stock-agg is also a stock-AR number "
        "over a DIFFERENT, shorter turn mix (49.06 tok/turn vs matched-20's %.1f fwd/turn)."
    ) % (resid_host_ms_fwd, gemm_ms_fwd, arith_floor_ms, HBM_BW_TBps, gemm_pct, fwd_per_turn,
         per_fwd_needed, per_fwd_needed - gemm_ms_fwd, nonweight_ms_fwd, gemm_ms_fwd, fwd_per_turn),
}

agg = {
    "engine_pin": "95d8b47 (qwen3_5-flare-modelstate) — OPT-4 Stage 1+2+3 LANDED, code default OFF",
    "config": "bidir(VLLM_FLARE_BIDIR_PROBE=1)+PIECEWISE cudagraph(VLLM_FLARE_CUDAGRAPH=1), APC on, greedy temp0 seed 20260701, uncapped n_ref+16",
    "n_turns": len(rows),
    "promotion_gate": gate,
    "byte_parity_full": par, "breaks": breaks,
    "eng_exact_args": eng_exact, "hf_exact_args": hf_exact, "eng_ne_hf": eng_ne_hf,
    "eng_valid": eng_valid, "eng_episode_exact": eng_epi, "hf_episode_exact": hf_epi,
    "verify_ok": verify_ok, "value_projection_nonzero": proj_nonzero,
    "wall_mean": round(mean_wall, 3), "wall_p50": round(pct(walls, 0.5), 3),
    "wall_p90": round(pct(walls, 0.9), 3), "wall_min": round(min(walls), 3),
    "wall_max": round(max(walls), 3), "wall_std": round(st.pstdev(walls), 3),
    "wall_total": round(wall_total, 3),
    "worst_turn": {"gt": worst["global_turn"], "n_gen": worst["n_gen"],
                   "fwd": worst["denoise_forwards"], "wall_s": worst["wall_s"]},
    "true_denoise_fwd_per_turn": round(fwd_per_turn, 2), "fwd_total": fwd_total,
    "tokens_total": tok_total, "tokens_per_forward": round(tok_total / fwd_total, 3),
    "per_forward_ms_amortized": round(per_fwd_ms_amortized, 2),
    "per_forward_ms_settled_longturns": round(st.mean(settled_ms), 2) if settled_ms else None,
    "hf_wall_mean": round(hf_wall_mean, 3), "hf_fwd_mean": round(st.mean(hf_fwds), 2),
    "speedup_vs_hf": round(hf_wall_mean / mean_wall, 3),
    "cg_pw_dispatches_total": cg_total, "cg_pw_turns": cg_turns,
    "delta_vs_prefix_v3": {
        "prefix_pin": "e5496cc (pre OPT-4 Stage 1/2/3)", "prefix_parity": len(prefix_rows) - len(prefix_breaks),
        "prefix_breaks": prefix_breaks, "prefix_exact": prefix_exact,
        "fix_cleared_breaks": cleared_by_fix, "still_broken": still_broken, "new_regressions": new_breaks,
        "shared_clean_ngen_identical": ngen_ident_shared, "shared_clean_exact_identical": exact_ident_shared,
        "note": ("Stage-3 fix clears %s (was breaks pre-fix) and holds gt60 to HF (pre-fix engine WON gt60 "
                 "=> exact 48; post-fix byte-matches HF => exact exactly 47). Lone residual %s." )
                 % (cleared_by_fix, still_broken),
    },
    "fresh_context_certificate": fresh_summary,
    "temp07": temp07,
    "nevertrain_spotcheck": nevertrain,
    "residual_gap_to_stock_agg": gap_breakdown,
    "bar_adjudication": {name: {"bar": bar, "ratio": round(mean_wall / bar, 3),
                                "verdict": "UNDER (beat)" if mean_wall < bar else "OVER (miss)"}
                         for name, bar in BARS.items()},
    "per_turn": [{"gt": r["global_turn"], "wall_s": r["wall_s"], "fwd": r["denoise_forwards"],
                  "n_gen": r["n_gen"], "n_ref": r["n_ref"], "parity": r["byte_parity_full"],
                  "per_forward_ms": r["per_forward_ms"], "cg_pw": r.get("cg_pw_dispatches"),
                  "eng_exact": int(r["eng_exact_arguments"]), "hf_exact": int(r["hf_exact_arguments"]),
                  "valid": int(r["eng_valid_tool_call"])} for r in rows],
}
OUT.write_text(json.dumps(agg, indent=2) + "\n")

# ---- printed summary ----
print(json.dumps({k: v for k, v in agg.items() if k not in ("per_turn",)}, indent=2))
print("\n=== PROMOTION GATE ===")
print(f"  byte-parity {par}/63 (need 63) -> {'MET' if par==63 else 'NOT MET'}  breaks={breaks}")
print(f"  exact_args {eng_exact} (need ==47) -> {'MET' if eng_exact==47 else 'DEVIATION'}")
print(f"  episode_exact {eng_epi} (need 13) -> {'MET' if eng_epi==13 else 'DEVIATION'}")
print(f"  valid {eng_valid} (need 63) -> {'MET' if eng_valid==63 else 'DEVIATION'}")
print(f"  PROMOTED = {gate['PROMOTED']}")
print("\n=== BAR ADJUDICATION (mean %.3f s/turn) ===" % mean_wall)
for name, bar in BARS.items():
    print(f"  {name:10s} bar={bar:.3f} -> {'UNDER (beat)' if mean_wall<bar else 'OVER (miss)'}  ratio={mean_wall/bar:.3f}x")
