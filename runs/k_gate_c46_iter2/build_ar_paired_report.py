#!/usr/bin/env python3
"""Tier1-C46 deficit-locus PAIRED read: twin@K1 diffusion vs AR-mode SWE-SFT.

Same 48 Tier1-C46 instances, IDENTICAL SWE-SFT weights, only the decode paradigm
differs (twin = FLARE hybrid_clean K=1 diffusion export; AR = mswe-S vLLM AR export).
Emits the per-instance paired table, McNemar exact (twin vs AR), the deficit-locus
verdict (A: conversion/decode-mode-specific | B: SFT-data-insufficiency | mixed), and
the AR-vs-stock-N=50 marginal comparison (DISJOINT pools — reported honestly).

usage: build_ar_paired_report.py [run_root]   (default runs/k_gate_c46)
Writes <root>/ar_paired_report.json and <root>/AR_PAIRED_READ.md ; prints the MD.
"""
import json, sys, glob, pathlib, collections, math

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "runs/k_gate_c46")

pool = json.load(open("data/swe_kraise_c46_pool/pool_manifest.json"))
pool_ids = list(pool["instance_ids"])
N = len(pool_ids)
repo_of = {x["instance_id"]: x["repo"] for x in pool["instances"]}


def load_scoring(subdir, tag):
    cands = glob.glob(str(root/subdir/"scoring"/f"*.{tag}.json"))
    if not cands:
        return None
    return json.load(open(sorted(cands)[0]))


def load_episodes(subdir):
    eps = {}
    for mp in glob.glob(str(root/subdir/"shard_*"/"verified"/"per_task"/"*"/"runner_metadata.json")):
        m = json.load(open(mp))
        eps[m["instance_id"]] = m
    return eps


def load_timing(subdir):
    p = root/subdir/"arm_timing.json"
    return json.load(open(p)) if p.is_file() else None


def covariates(eps, resolved):
    exit_hist = collections.Counter()
    tin = tout = ttot = 0
    loop_halts = budget = turnlimit = clean_exit = timeouts = empty = nonempty = 0
    ctx_overflow = 0
    turns = []
    halt_ids, empty_ids, nonempty_ids, ctx_overflow_ids = [], [], [], []
    for iid, m in eps.items():
        q = m.get("qwen") or {}
        u = q.get("usage") or {}
        tin += u.get("input_tokens") or 0
        tout += u.get("output_tokens") or 0
        ttot += u.get("total_tokens") or 0
        ec = q.get("exit_code")
        exit_hist[str(ec)] += 1
        # HARNESS TRUTH-TELLING (52ffcc2, active iter-2): a context-window cap-death
        # exits 0 with an empty patch but is env-limited — bucket it separately and
        # keep it OUT of clean_exit0 / empty_patches. Records predating the tag
        # (terminal_cause absent) -> is_ctx False -> old labeling preserved.
        is_ctx = (m.get("terminal_cause") == "ctx_overflow")
        if is_ctx: ctx_overflow += 1; ctx_overflow_ids.append(iid)
        if ec == 1: loop_halts += 1; halt_ids.append(iid)
        if ec == 55: budget += 1
        if ec == 53: turnlimit += 1
        if ec == 0 and not is_ctx: clean_exit += 1
        if q.get("timed_out"): timeouts += 1
        pb = m.get("patch_bytes") or 0
        if not pb:
            if not is_ctx: empty += 1; empty_ids.append(iid)
        else: nonempty += 1; nonempty_ids.append(iid)
        if q.get("num_turns"): turns.append(q["num_turns"])
        for rk in [k for k in m if k.startswith("qwen_retry")]:
            ru = (m[rk] or {}).get("usage") or {}
            tin += ru.get("input_tokens") or 0; tout += ru.get("output_tokens") or 0
            ttot += ru.get("total_tokens") or 0
    median_turns = (sorted(turns)[len(turns)//2] if turns else None)
    halt_set = set(halt_ids)
    return {
        "n_episodes": len(eps),
        "exit_histogram": dict(exit_hist),
        "loop_halts_exit1": loop_halts, "turnlimit_exit53": turnlimit,
        "clean_exit0": clean_exit, "budget_exit55": budget, "timeouts": timeouts,
        "empty_patches": empty, "edit_committed_nonempty": nonempty,
        "ctx_overflow_deaths": ctx_overflow, "ctx_overflow_ids": sorted(ctx_overflow_ids),
        "median_turns": median_turns,
        "input_tokens": tin, "output_tokens": tout, "total_tokens": ttot,
        "loop_halt_no_patch": len(halt_set & set(empty_ids)),
        "loop_halt_nonempty": len(halt_set & set(nonempty_ids)),
        "loop_halt_resolved": len(halt_set & resolved),
        "pre_resolve_halts": len(halt_set - resolved),
        "post_resolve_halts": len(halt_set & resolved),
        "resolved_with_edit_commitment": len(resolved & set(nonempty_ids)),
        "halt_ids": sorted(halt_ids), "empty_ids": sorted(empty_ids),
        "nonempty_ids": sorted(nonempty_ids),
    }


def mcnemar_exact_2sided(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


# ---- load both arms ------------------------------------------------------------
sc_ar = load_scoring("ar", "c46_ar")
sc_tw = load_scoring("diffusion", "c46_twinK1")
eps_ar = load_episodes("ar")
eps_tw = load_episodes("diffusion")
res_ar = set(sc_ar["resolved_ids"]) if sc_ar and sc_ar.get("resolved_ids") else set()
res_tw = set(sc_tw["resolved_ids"]) if sc_tw and sc_tw.get("resolved_ids") else set()
cov_ar = covariates(eps_ar, res_ar) if eps_ar else {}
cov_tw = covariates(eps_tw, res_tw) if eps_tw else {}
tm_ar = load_timing("ar")
tm_tw = load_timing("diffusion")

# ---- paired McNemar (twin vs AR) ----------------------------------------------
both = sorted(res_tw & res_ar)
tw_only = sorted(res_tw - res_ar)   # b: twin wins, AR loses
ar_only = sorted(res_ar - res_tw)   # c: AR wins, twin loses
neither = [i for i in pool_ids if i not in res_tw and i not in res_ar]
b, c = len(tw_only), len(ar_only)
net_tw_minus_ar = b - c
p = mcnemar_exact_2sided(b, c)

# ---- AR vs stock-N=50 marginal (DISJOINT pools) -------------------------------
n50 = {
    "pool": "runs/w2_n50 Tier1 N=50 (SWE-bench_Verified, pool fe1973937dfb500b…)",
    "arm": "STOCK Qwen3.5-9B AR (official snapshot, NOT SWE-SFT weights)",
    "resolved": 19, "n": 50, "frac": round(19/50, 4),
    "overlap_with_c46": 0,   # verified disjoint at build time
    "note": "DISJOINT instance sets (|C46 ∩ N50| = 0) AND different weights (stock vs SWE-SFT); "
            "population resolve-rate comparison only — NOT a paired/apples-to-apples claim.",
}

# throughput
def thr(tm, n):
    if tm and tm.get("wall_seconds"):
        h = tm["wall_seconds"]/3600.0
        return {"wall_seconds": tm["wall_seconds"], "concurrency": tm.get("concurrency"),
                "n": n, "eps_per_gpu_h": round(n/h, 2) if h > 0 else None}
    return None
thr_ar = thr(tm_ar, len(eps_ar))
thr_tw = thr(tm_tw, len(eps_tw))

# per-repo
tot_repo = collections.Counter(repo_of[i] for i in pool_ids)
rr_ar = collections.Counter(repo_of[i] for i in res_ar)
rr_tw = collections.Counter(repo_of[i] for i in res_tw)
per_repo = {r: {"ar": [rr_ar.get(r, 0), tot_repo[r]], "twin": [rr_tw.get(r, 0), tot_repo[r]]}
            for r in sorted(tot_repo)}

# ---- mechanical locus verdict --------------------------------------------------
data_ok = bool(sc_ar) and bool(sc_tw) and len(eps_ar) == N and len(eps_tw) == N
ar_res, tw_res = len(res_ar), len(res_tw)
# A: AR >> twin (AR meaningfully higher; McNemar leans AR with a real margin)
# B: AR ~ twin, both low (no material margin)
if not data_ok:
    locus = "RUN-NOT-COMPLETE"
elif (ar_res - tw_res) >= 6 and c >= b + 4:
    locus = "A: conversion/decode-mode-specific deficit"
elif abs(ar_res - tw_res) <= 3 and p >= 0.05:
    locus = "B: SFT-data-insufficiency"
else:
    locus = "MIXED"

report = {
    "read": "Tier1-C46 twin@K1 RE-GATE + deficit-locus paired read (iteration-2, K-raise step 5)",
    "pool_sha256": pool["pool_sha256"], "n": N,
    "arms": {
        "ar": {"weights": "models/qwen3.5-9b-fastdllm-mswe-S-iter2-vllm-bf16 (iter-2 arm-S SWE-SFT, AR export = the fold the KILL-T1 iter2 gate served)",
               "serve": "runs/stage_c_driver/runcage_ar.sh SNAP-override, stock vLLM 0.23 AR, gmu0.85 seqs4 ml32768",
               "decode": "AR native, temp 0.6/top_p 0.95/top_k 20 proxy-forced, NO pp, turn cap 75",
               "resolved": ar_res, "resolved_ids": sorted(res_ar)},
        "twin": {"weights": "models/qwen3.5-9b-fastdllm-mswe-S-iter2-merged + mswe2_S_twinK1 re-conv adapter (IDENTICAL iter-2 SFT), export models/qwen3.5-9b-fastdllm-mswe2-S-twinK1-vllm-bf16",
                 "serve": "FLARE hybrid_clean K=1 export, gmu0.74 seqs4, CERTIFIED read-clamp shim ACTIVE (limit=100, cert 7ae55d4)",
                 "decode": "diffusion hybrid_clean K=1, same envelope, turn cap 75",
                 "resolved": tw_res, "resolved_ids": sorted(res_tw)},
    },
    "PAIRED_mcnemar": {
        "ar_resolved": ar_res, "twin_resolved": tw_res,
        "both": len(both), "twin_only_b": b, "ar_only_c": c, "neither": len(neither),
        "net_twin_minus_ar": net_tw_minus_ar,
        "mcnemar_exact_p_2sided": round(p, 6),
        "both_ids": both, "twin_only_ids": tw_only, "ar_only_ids": ar_only,
    },
    "LOCUS_VERDICT": locus,
    "ar_vs_stock_n50": n50,
    "covariates": {"ar": cov_ar, "twin": cov_tw},
    "throughput": {"ar": thr_ar, "twin": thr_tw},
    "per_repo_resolved": per_repo,
    "data_sufficient": data_ok,
}
(root/"ar_paired_report.json").write_text(json.dumps(report, indent=2))

# ---- per-instance paired table -------------------------------------------------
def cell(iid, resset):
    return "✅" if iid in resset else "·"

exit_ar = {}
exit_tw = {}
pb_ar = {}
pb_tw = {}
for iid, m in eps_ar.items():
    exit_ar[iid] = (m.get("qwen") or {}).get("exit_code")
    pb_ar[iid] = m.get("patch_bytes") or 0
for iid, m in eps_tw.items():
    exit_tw[iid] = (m.get("qwen") or {}).get("exit_code")
    pb_tw[iid] = m.get("patch_bytes") or 0

L = []
L.append("# Tier1-C46 twin@K1 RE-GATE (iteration-2) — AR-mode PAIRED read + McNemar (K-raise step 5)\n")
if not data_ok:
    L.append(f"> **STATUS: RUN NOT COMPLETE.** ar_scoring={'ok' if sc_ar else 'MISSING'} "
             f"twin_scoring={'ok' if sc_tw else 'MISSING'} ar_eps={len(eps_ar)}/{N} twin_eps={len(eps_tw)}/{N}.\n")
L.append(f"pool_sha256 `{pool['pool_sha256'][:16]}…`  N={N}.  IDENTICAL iteration-2 arm-S SWE-SFT weights; "
         f"only the decode paradigm differs (AR native vs FLARE hybrid_clean K=1 diffusion WITH the "
         f"CERTIFIED read-clamp shim active, limit=100).\n")
L.append("## PRIMARY — paired resolve@1 (twin diffusion vs AR), McNemar exact")
L.append(f"- AR (SWE-SFT, AR decode) resolved: **{ar_res}/{N}**   twin@K1 (SWE-SFT, diffusion) resolved: **{tw_res}/{N}**")
L.append(f"- both=**{len(both)}**  twin-only(b)=**{b}**  AR-only(c)=**{c}**  neither=**{len(neither)}**")
L.append(f"- net (twin − AR) = **{net_tw_minus_ar}**   McNemar exact 2-sided p = **{p:.4f}**")
L.append(f"- **DEFICIT-LOCUS VERDICT: {locus}**")
if both: L.append(f"  - both-resolved ids: {both}")
if ar_only: L.append(f"  - AR-only ids (twin lost): {ar_only}")
if tw_only: L.append(f"  - twin-only ids (AR lost): {tw_only}")
L.append("\n## Per-instance paired table (48)")
L.append("| instance | repo | AR | twin | AR exit | tw exit | AR bytes | tw bytes |")
L.append("|---|---|:--:|:--:|--:|--:|--:|--:|")
for iid in pool_ids:
    L.append(f"| {iid} | {repo_of[iid].split('/')[-1]} | {cell(iid,res_ar)} | {cell(iid,res_tw)} | "
             f"{exit_ar.get(iid,'—')} | {exit_tw.get(iid,'—')} | {pb_ar.get(iid,0)} | {pb_tw.get(iid,0)} |")

L.append("\n## SECONDARY — loop-shape / edit-commitment covariates (paired)")
L.append("| metric | AR (SWE-SFT) | twin@K1 (SWE-SFT) |")
L.append("|---|---:|---:|")
def g(d, k, dflt="—"): return d.get(k, dflt)
L.append(f"| resolved | {ar_res}/{N} | {tw_res}/{N} |")
L.append(f"| loop-halts (exit 1) | {g(cov_ar,'loop_halts_exit1')} | {g(cov_tw,'loop_halts_exit1')} |")
L.append(f"| turn-limit (exit 53) | {g(cov_ar,'turnlimit_exit53')} | {g(cov_tw,'turnlimit_exit53')} |")
L.append(f"| clean (exit 0) | {g(cov_ar,'clean_exit0')} | {g(cov_tw,'clean_exit0')} |")
L.append(f"| budget (exit 55) | {g(cov_ar,'budget_exit55')} | {g(cov_tw,'budget_exit55')} |")
L.append(f"| ctx-overflow cap-deaths (env-limited) | {g(cov_ar,'ctx_overflow_deaths')} | {g(cov_tw,'ctx_overflow_deaths')} |")
L.append(f"| empty patches (honest miss) | {g(cov_ar,'empty_patches')} | {g(cov_tw,'empty_patches')} |")
L.append(f"| edit committed (non-empty) | {g(cov_ar,'edit_committed_nonempty')} | {g(cov_tw,'edit_committed_nonempty')} |")
L.append(f"| median turns | {g(cov_ar,'median_turns')} | {g(cov_tw,'median_turns')} |")
L.append(f"| loop-halt → no patch | {g(cov_ar,'loop_halt_no_patch')} | {g(cov_tw,'loop_halt_no_patch')} |")
L.append(f"| loop-halt → non-empty | {g(cov_ar,'loop_halt_nonempty')} | {g(cov_tw,'loop_halt_nonempty')} |")
L.append(f"| pre-resolve halts | {g(cov_ar,'pre_resolve_halts')} | {g(cov_tw,'pre_resolve_halts')} |")
L.append(f"| resolved WITH edit committed | {g(cov_ar,'resolved_with_edit_commitment')} | {g(cov_tw,'resolved_with_edit_commitment')} |")
L.append(f"| input tokens | {g(cov_ar,'input_tokens'):,} | {g(cov_tw,'input_tokens'):,} |" if cov_ar and cov_tw else "")
if thr_ar and thr_tw:
    L.append("\n### Throughput")
    L.append(f"- ar: **{thr_ar['eps_per_gpu_h']} eps/GPU-h** (n={thr_ar['n']}, wall={thr_ar['wall_seconds']}s, c={thr_ar['concurrency']})")
    L.append(f"- twin: **{thr_tw['eps_per_gpu_h']} eps/GPU-h** (n={thr_tw['n']}, wall={thr_tw['wall_seconds']}s, c={thr_tw['concurrency']})")
L.append("\n### Per-repo resolved (AR / twin, of total)")
L.append("| repo | AR | twin |")
L.append("|---|---|---|")
for r in sorted(tot_repo):
    a = per_repo[r]
    L.append(f"| {r} | {a['ar'][0]}/{a['ar'][1]} | {a['twin'][0]}/{a['twin'][1]} |")

L.append("\n## TERTIARY — AR arm vs STOCK N=50 baseline (DISJOINT pools; marginal only)")
L.append(f"- AR-mode SWE-SFT on Tier1-C46: **{ar_res}/{N} ({ar_res/N:.1%})**")
L.append(f"- stock-Qwen3.5 AR on Tier1-N=50: **19/50 (38.0%)** (runs/w2_n50)")
L.append(f"- **Overlap |C46 ∩ N50| = 0** — fully disjoint instance sets. Two confounds vs this "
         f"baseline: (1) different instances, (2) stock vs SWE-SFT weights. NO apples-to-apples "
         f"claim; population resolve-rate contrast only.")
md = "\n".join(x for x in L if x is not None) + "\n"
(root/"AR_PAIRED_READ.md").write_text(md)
print(md)
print("== mechanical locus verdict:", locus, "== (narrative + levers appended by hand)")
