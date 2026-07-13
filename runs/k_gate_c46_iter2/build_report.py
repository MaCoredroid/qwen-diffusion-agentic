#!/usr/bin/env python3
"""Tier1-C46 twin@K1 gate report (K-raise campaign step 5).

SINGLE ARM: the SWE-SFT+conversion twin@K1 diffusion, decoded K=1 (hybrid_clean,
FSM values), on the fresh Tier1-C46 slice. Emits resolve@1 vs the >=12 entry floor
(design §1.1), per-episode loop-halt / edit-commitment covariates, and the failure-
mode breakdown against the N=50 taxonomy (the pre-SFT RL-v2 twin: 2/50, 26 loop-halts,
35 empty patches, 25 pre-resolve halts).

usage: build_report.py [run_root]   (default runs/k_gate_c46)
Reads:
  <root>/diffusion/scoring/*.c46_twinK1.json                 official swebench report
  <root>/diffusion/shard_*/verified/per_task/*/runner_metadata.json  per-episode
  <root>/diffusion/arm_timing.json                            wall clock for throughput
  <root>/logs/diffusion_server.log                            serving-health covariate
Writes <root>/report.json and <root>/report.md ; prints the table.
"""
import json, sys, glob, pathlib, collections, re

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "runs/k_gate_c46")
ARM = "diffusion"
ENTRY_FLOOR = 12  # design §1.1 power floor

pool = json.load(open("data/swe_kraise_c46_pool/pool_manifest.json"))
pool_ids = list(pool["instance_ids"])
N = len(pool_ids)
repo_of = {x["instance_id"]: x["repo"] for x in pool["instances"]}

# ---- N=50 taxonomy baseline (pre-SFT RL-v2 twin; runs/w2_n50/report.md) --------
N50 = {
    "model": "RL-v2 twin (pre-SWE-SFT), w2_n50 diffusion arm",
    "n": 50, "resolved": 2, "diffusion_only_wins": 0,
    "exit_taxonomy": {"loop_exit1": 26, "turnlimit_exit53": 13, "clean_exit0": 10, "budget_exit55": 1},
    "empty_patches": 35, "loop_halts": 26,
    "loop_halt_no_patch": 18, "loop_halt_nonempty": 8, "loop_halt_resolved": 1,
    "pre_resolve_halts": 25, "post_resolve_halts": 1,
    "eps_per_gpu_h": 21.41, "median_turns": 25,
    "turn_cap": 50,
}


def load_scoring():
    cands = glob.glob(str(root/ARM/"scoring"/f"*.c46_twinK1.json"))
    if not cands:
        return None
    return json.load(open(sorted(cands)[0]))


def load_episodes():
    eps = {}
    for mp in glob.glob(str(root/ARM/"shard_*"/"verified"/"per_task"/"*"/"runner_metadata.json")):
        m = json.load(open(mp))
        eps[m["instance_id"]] = m
    return eps


def load_timing():
    p = root/ARM/"arm_timing.json"
    return json.load(open(p)) if p.is_file() else None


sc = load_scoring()
eps = load_episodes()
tm = load_timing()
resolved = set(sc["resolved_ids"]) if sc and sc.get("resolved_ids") else set()

# ---- per-episode covariates ----------------------------------------------------
exit_hist = collections.Counter()
tin = tout = ttot = 0
loop_halts = budget = turnlimit = clean_exit = timeouts = empty = nonempty = 0
ctx_overflow = 0
turns = []
halt_ids, empty_ids, nonempty_ids, timeout_ids, ctx_overflow_ids = [], [], [], [], []
missing = [i for i in pool_ids if i not in eps]
for iid, m in eps.items():
    q = m.get("qwen") or {}
    u = q.get("usage") or {}
    tin += u.get("input_tokens") or 0
    tout += u.get("output_tokens") or 0
    ttot += u.get("total_tokens") or 0
    ec = q.get("exit_code")
    exit_hist[str(ec)] += 1
    # HARNESS TRUTH-TELLING (2026-07-12): a context-window cap-death exits 0 with an
    # empty patch, but it is env-limited — NOT a clean exit-0 quit and NOT an honest
    # empty-patch miss. When the driver tagged terminal_cause=="ctx_overflow", bucket
    # it separately and keep it OUT of clean_exit0 / empty_patches. Historical C46
    # records predate the tag (terminal_cause absent) -> is_ctx False -> OLD labeling
    # preserved (this gate's numbers do not move retroactively).
    is_ctx = (m.get("terminal_cause") == "ctx_overflow")
    if is_ctx: ctx_overflow += 1; ctx_overflow_ids.append(iid)
    if ec == 1: loop_halts += 1; halt_ids.append(iid)
    if ec == 55: budget += 1
    if ec == 53: turnlimit += 1
    if ec == 0 and not is_ctx: clean_exit += 1
    if q.get("timed_out"): timeouts += 1; timeout_ids.append(iid)
    pb = m.get("patch_bytes") or 0
    if not pb:
        if is_ctx:
            pass  # counted under ctx_overflow (env-limited), not empty-as-honest-miss
        else:
            empty += 1; empty_ids.append(iid)
    else: nonempty += 1; nonempty_ids.append(iid)
    if q.get("num_turns"): turns.append(q["num_turns"])
    for rk in [k for k in m if k.startswith("qwen_retry")]:
        ru = (m[rk] or {}).get("usage") or {}
        tin += ru.get("input_tokens") or 0; tout += ru.get("output_tokens") or 0
        ttot += ru.get("total_tokens") or 0

median_turns = (sorted(turns)[len(turns)//2] if turns else None)

# loop-halt × edit cross-tab + pre/post-resolve halt split
halt_set = set(halt_ids)
halt_no_patch = len(halt_set & set(empty_ids))
halt_nonempty = len(halt_set & set(nonempty_ids))
halt_resolved = len(halt_set & resolved)
pre_resolve_halts = len(halt_set - resolved)
post_resolve_halts = len(halt_set & resolved)
# edit-commitment among resolved / unresolved
edit_committed = set(nonempty_ids)
resolved_with_edit = len(resolved & edit_committed)

# throughput
thr = None
if tm and tm.get("wall_seconds"):
    h = tm["wall_seconds"]/3600.0
    thr = {"wall_seconds": tm["wall_seconds"], "concurrency": tm.get("concurrency"),
           "n": len(eps), "eps_per_gpu_h": round(len(eps)/h, 2) if h > 0 else None}

# per-repo resolved
tot_repo = collections.Counter(repo_of[i] for i in pool_ids)
res_repo = collections.Counter(repo_of[i] for i in resolved)
per_repo = {r: [res_repo.get(r, 0), tot_repo[r]] for r in sorted(tot_repo)}

# ---- serving-health covariate (parse server log) -------------------------------
serving = {}
kill3 = {}
slog = root/"logs"/"diffusion_server.log"
if slog.is_file():
    txt = slog.read_text(errors="ignore")
    # KILL-3 zero-value-projection tripwire (design §2.5): the served engine logs
    # `projected_value_tokens_exact` (== decoder.stats.value_projection_events) per
    # request. It MUST be 0 on every request; nonzero = the grammar OVERWROTE a
    # model value token (a served-engine correctness regression). This is projection-
    # IMMUNE for the docker-scored resolve@1 PRIMARY (a phantom value token still
    # yields a real patch that real tests adjudicate), but is audited here for honesty.
    proj = [int(m) for m in re.findall(r"projected_value_tokens_exact=(\d+)", txt)]
    proj_nonzero = [v for v in proj if v > 0]
    kill3 = {
        "counter": "projected_value_tokens_exact (== value_projection_events)",
        "requests_audited": len(proj),
        "nonzero_request_count": len(proj_nonzero),
        "nonzero_values": sorted(proj_nonzero, reverse=True),
        "total_events": sum(proj_nonzero),
        "KILL3_CLEAN": len(proj_nonzero) == 0,
        "note": ("clean" if not proj_nonzero else
                 "tripwire fired: grammar overwrote value token(s) on "
                 f"{len(proj_nonzero)}/{len(proj)} requests "
                 f"({100.0*len(proj_nonzero)/len(proj):.2f}%). Does NOT contaminate "
                 "the docker-scored resolve@1 verdict (behavioral, projection-immune); "
                 "IS a served-engine note for any future K-track tok/fwd measurement."),
    }
    serving = {
        "decode_mode_hybrid_clean": ("decode_mode=hybrid_clean" in txt),
        "flare_gate_confirmed": ("Qwen3_5FlareModelState" in txt),
        "flare_hybrid_clean_req_lines": len(re.findall(r"FLARE hybrid_clean req", txt)),
        "diffusion_decoding_metrics_lines": len(re.findall(r"DiffusionDecoding metrics", txt)),
        "mask_suppression_248077": ("248077" in txt),
        "kill3_value_projection": kill3,
    }

# ---- data sufficiency + verdict ------------------------------------------------
scoring_ok = bool(sc)
eps_ok = len(eps) == N
data_sufficient = scoring_ok and eps_ok
nres = len(resolved)
if not data_sufficient:
    verdict = "RUN-NOT-COMPLETE"
elif nres >= ENTRY_FLOOR:
    verdict = "ENTRY-PASS"
else:
    verdict = "INCONCLUSIVE-BY-POWER"

report = {
    "gate": "Tier1-C46 twin@K1 RE-GATE (iteration-2, K-raise campaign step 5)",
    "pool_sha256": pool["pool_sha256"], "n": N,
    "entry_floor": ENTRY_FLOOR,
    "twin": {
        "base": "models/qwen3.5-9b-fastdllm-mswe-S-iter2-merged",
        "adapter": "runs/kraise_reconvert_iter2/mswe2_S_twinK1_run1recipe_step400_seed81101",
        "served_export": "models/qwen3.5-9b-fastdllm-mswe2-S-twinK1-vllm-bf16",
        "read_clamp": "CERTIFIED read-clamp shim ACTIVE (runs/k_gate_c46/proxy_readclamp.py, "
                      "cert 7ae55d4, LUMO_PROXY_READCLAMP_LIMIT=100)",
        "decode": "hybrid_clean K=1 (FSM values), canvas 32, temp 0.6/top_p 0.95/top_k 20, NO pp",
        "turn_cap": 75, "seed_base": 1234,
    },
    "PRIMARY_resolve_at_1": {
        "resolved": nres, "n": N, "frac": round(nres/N, 4) if N else None,
        "entry_floor": ENTRY_FLOOR, "verdict": verdict,
        "data_sufficient": data_sufficient,
        "resolved_ids": sorted(resolved),
    },
    "covariates": {
        "n_episodes": len(eps), "missing": missing,
        "exit_histogram": dict(exit_hist),
        "loop_halts_exit1": loop_halts, "turnlimit_exit53": turnlimit,
        "clean_exit0": clean_exit, "budget_exit55": budget, "timeouts": timeouts,
        "empty_patches": empty, "edit_committed_nonempty": nonempty,
        "ctx_overflow_deaths": ctx_overflow, "ctx_overflow_ids": sorted(ctx_overflow_ids),
        "median_turns": median_turns,
        "input_tokens": tin, "output_tokens": tout, "total_tokens": ttot,
        "loop_halt_edit_crosstab": {
            "loop_halt_no_patch": halt_no_patch, "loop_halt_nonempty": halt_nonempty,
            "loop_halt_resolved": halt_resolved,
            "pre_resolve_halts": pre_resolve_halts, "post_resolve_halts": post_resolve_halts,
        },
        "resolved_with_edit_commitment": resolved_with_edit,
        "halt_ids": sorted(halt_ids), "empty_ids": sorted(empty_ids),
    },
    "throughput": thr,
    "per_repo_resolved": per_repo,
    "serving_health": serving,
    "n50_taxonomy_baseline": N50,
    "scoring_present": scoring_ok,
}
(root/"report.json").write_text(json.dumps(report, indent=2))

# ---- markdown ------------------------------------------------------------------
L = []
L.append("# Tier1-C46 twin@K1 RE-GATE (iteration-2) — resolve@1 (K-raise campaign step 5)\n")
if not data_sufficient:
    L.append(f"> **STATUS: RUN NOT COMPLETE — no verdict.** scoring="
             f"{'ok' if scoring_ok else 'MISSING'} episodes={len(eps)}/{N}. "
             f"All cells below are partial.\n")
L.append(f"pool_sha256 `{pool['pool_sha256'][:16]}…`  N={N}  "
         f"iteration-2 twin@K1 (windowed-pool SWE-SFT + K-conversion), decoded K=1 hybrid_clean "
         f"WITH the CERTIFIED read-clamp shim active (limit=100, cert 7ae55d4).\n")
L.append("## PRIMARY — resolve@1 vs the entry floor")
L.append(f"- **resolved: {nres}/{N}  ({report['PRIMARY_resolve_at_1']['frac']:.1%})**   "
         f"entry floor = **{ENTRY_FLOOR}** (design §1.1 McNemar-power floor)")
L.append(f"- **VERDICT: {verdict}**  — "
         + ("≥ floor: the twin is off the 2/50 floor; K-track has power." if verdict == "ENTRY-PASS"
            else ("below the power floor: do NOT spend K rungs (SWE-SFT base too weak); escalate per USER_LEVER_BELT." if verdict == "INCONCLUSIVE-BY-POWER"
                  else "run incomplete.")))
if resolved:
    L.append(f"  - resolved ids: {sorted(resolved)}")
L.append("\n## SECONDARY — covariates")
if thr:
    L.append(f"### Throughput\n- **{thr['eps_per_gpu_h']} eps/GPU-h**  (n={thr['n']}, wall={thr['wall_seconds']}s, c={thr['concurrency']})  — N=50 baseline 21.41")
L.append("### Loop-halt / edit-commitment covariates (vs N=50 pre-SFT taxonomy)")
L.append("| metric | twin@K1 (C46, N=48) | pre-SFT RL-v2 (w2, N=50) |")
L.append("|---|---:|---:|")
L.append(f"| resolved | {nres}/{N} | 2/50 |")
L.append(f"| loop-halts (exit 1) | {loop_halts} | 26 |")
L.append(f"| turn-limit (exit 53) | {turnlimit} | 13* |")
L.append(f"| clean (exit 0) | {clean_exit} | 10 |")
L.append(f"| budget (exit 55) | {budget} | 1 |")
L.append(f"| ctx-overflow cap-deaths (env-limited) | {ctx_overflow} | — |")
L.append(f"| empty patches (honest miss) | {empty} | 35 |")
L.append(f"| edit committed (non-empty) | {nonempty} | 15 |")
L.append(f"| median turns | {median_turns} | 25 |")
L.append(f"| loop-halt → no patch | {halt_no_patch} | 18 |")
L.append(f"| loop-halt → non-empty | {halt_nonempty} | 8 |")
L.append(f"| pre-resolve halts | {pre_resolve_halts} | 25 |")
L.append(f"| post-resolve halts | {post_resolve_halts} | 1 |")
L.append(f"| resolved WITH an edit committed | {resolved_with_edit} | 2 |")
L.append("\n*N=50 used turn cap 50; this gate uses 75 (design §2.1), so exit-53 counts are not directly comparable.*")
L.append("### Per-repo resolved (resolved/total)")
L.append("| repo | twin@K1 |")
L.append("|---|---|")
for r in sorted(tot_repo):
    a = per_repo[r]
    L.append(f"| {r} | {a[0]}/{a[1]} |")
if serving:
    L.append("### Serving health (covariate)")
    L.append(f"- decode_mode=hybrid_clean: {serving.get('decode_mode_hybrid_clean')}  · "
             f"FLARE gate: {serving.get('flare_gate_confirmed')}  · "
             f"hybrid_clean req lines: {serving.get('flare_hybrid_clean_req_lines')}  · "
             f"DiffusionDecoding metrics lines: {serving.get('diffusion_decoding_metrics_lines')}  · "
             f"mask 248077 present: {serving.get('mask_suppression_248077')}")
    if kill3:
        L.append(f"- **KILL-3 zero-value-projection tripwire:** clean={kill3['KILL3_CLEAN']}  "
                 f"({kill3['nonzero_request_count']}/{kill3['requests_audited']} requests nonzero"
                 + (f", values {kill3['nonzero_values']}, {kill3['total_events']} events" if kill3['nonzero_request_count'] else "")
                 + f"). {kill3['note']}")
# anomalies
anom = []
if missing: anom.append(f"{len(missing)} episodes MISSING metadata: {missing[:6]}")
if len(eps) != N: anom.append(f"only {len(eps)}/{N} episodes present")
if not scoring_ok: anom.append("scoring report MISSING")
elif sc and sc.get("error_ids"): anom.append(f"scoring error_ids={sc['error_ids']}")
if kill3 and not kill3.get("KILL3_CLEAN", True):
    anom.append(f"KILL-3 tripwire fired on {kill3['nonzero_request_count']}/{kill3['requests_audited']} "
                f"served requests ({kill3['total_events']} value-projection events) — served-engine note; "
                f"projection-immune for docker-scored resolve@1 (verdict unaffected).")
if anom:
    L.append("\n## ANOMALIES"); [L.append(f"- {a}") for a in anom]
md = "\n".join(L) + "\n"
(root/"report.md").write_text(md)
print(md)
