#!/usr/bin/env python3
"""W2 N=50 report: paired resolve@1 McNemar (PRIMARY) + throughput, tokens,
loop-halt covariates, per-repo breakdown (SECONDARY). Pre-registered analysis.

usage: build_report.py <run_root>   (default runs/w2_n50)
Reads:
  <root>/<arm>/scoring/*.w2n50_<arm>.json     official swebench report
  <root>/<arm>/shard_*/verified/per_task/*/runner_metadata.json  per-episode
  <root>/<arm>/arm_timing.json                 wall clock for throughput
Writes <root>/report.json and <root>/report.md ; prints the table.
"""
import json, sys, glob, math, pathlib, collections
root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "runs/w2_n50")
ARMS = ["ar", "diffusion"]

def load_scoring(arm):
    cands = glob.glob(str(root/arm/"scoring"/f"*.w2n50_{arm}.json"))
    if not cands:
        return None
    return json.load(open(sorted(cands)[0]))

def load_episodes(arm):
    eps = {}
    for mp in glob.glob(str(root/arm/"shard_*"/"verified"/"per_task"/"*"/"runner_metadata.json")):
        m = json.load(open(mp))
        eps[m["instance_id"]] = m
    return eps

def load_timing(arm):
    p = root/arm/"arm_timing.json"
    return json.load(open(p)) if p.is_file() else None

pool = json.load(open("data/swe_w2_n50_pool/pool_manifest.json"))
pool_ids = list(pool["instance_ids"])
repo_of = {x["instance_id"]: x["repo"] for x in pool["instances"]}

data = {}
for arm in ARMS:
    sc = load_scoring(arm); eps = load_episodes(arm); tm = load_timing(arm)
    resolved = set(sc["resolved_ids"]) if sc else set()
    data[arm] = {"scoring": sc, "eps": eps, "timing": tm, "resolved": resolved}

def mcnemar_exact_p(b, c):
    # two-sided exact binomial on discordant pairs, p=0.5
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # P(X<=k) under Binom(n,0.5), doubled, capped at 1
    cdf = sum(math.comb(n, i) for i in range(0, k+1)) / (2**n)
    return min(1.0, 2*cdf)

# ---- PRIMARY: paired McNemar (diffusion vs AR) ----
ar_res = data["ar"]["resolved"]; df_res = data["diffusion"]["resolved"]
both = ar_res & df_res
ar_only = ar_res - df_res          # b: AR resolves, diffusion doesn't
df_only = df_res - ar_res          # c: diffusion resolves, AR doesn't
neither = set(pool_ids) - ar_res - df_res
b, c = len(ar_only), len(df_only)
net = c - b                        # diffusion - AR
p = mcnemar_exact_p(b, c)
# DATA-SUFFICIENCY GUARD: b=c=0 on EMPTY data gives net 0 / p 1.0 -> a spurious
# "parity YES". A parity claim is only valid when BOTH arms are fully scored with
# 50/50 episodes. Otherwise report None ("INSUFFICIENT DATA"), not a verdict.
_scoring_ok = bool(data["ar"]["scoring"]) and bool(data["diffusion"]["scoring"])
_eps_ok = len(data["ar"]["eps"]) == len(pool_ids) == len(data["diffusion"]["eps"])
data_sufficient = _scoring_ok and _eps_ok
parity = ((abs(net) <= 2) and (p >= 0.05)) if data_sufficient else None

def tok_and_halts(arm):
    eps = data[arm]["eps"]
    tin = tout = ttot = 0
    halts = budget = timeouts = empty = 0
    turns = []
    halt_ids, timeout_ids = [], []
    missing = [i for i in pool_ids if i not in eps]
    for iid, m in eps.items():
        q = m.get("qwen") or {}
        u = q.get("usage") or {}
        tin += u.get("input_tokens") or 0
        tout += u.get("output_tokens") or 0
        ttot += u.get("total_tokens") or 0
        ec = q.get("exit_code")
        if ec == 1: halts += 1; halt_ids.append(iid)
        if ec == 55: budget += 1
        if q.get("timed_out"): timeouts += 1; timeout_ids.append(iid)
        if not (m.get("patch_bytes") or 0): empty += 1
        if q.get("num_turns"): turns.append(q["num_turns"])
        # count retry token cost too
        for rk in [k for k in m if k.startswith("qwen_retry")]:
            ru = (m[rk] or {}).get("usage") or {}
            tin += ru.get("input_tokens") or 0; tout += ru.get("output_tokens") or 0
            ttot += ru.get("total_tokens") or 0
    return dict(input_tokens=tin, output_tokens=tout, total_tokens=ttot,
                loop_halts=halts, budget_halts=budget, timeouts=timeouts,
                empty_patches=empty, n_episodes=len(eps), missing=missing,
                median_turns=(sorted(turns)[len(turns)//2] if turns else None),
                halt_ids=halt_ids, timeout_ids=timeout_ids)

cov = {arm: tok_and_halts(arm) for arm in ARMS}

def throughput(arm):
    tm = data[arm]["timing"]; n = cov[arm]["n_episodes"]
    if not tm or not tm.get("wall_seconds"): return None
    h = tm["wall_seconds"]/3600.0
    return dict(wall_seconds=tm["wall_seconds"], concurrency=tm.get("concurrency"),
                n=n, eps_per_gpu_h=round(n/h, 2) if h > 0 else None)

thr = {arm: throughput(arm) for arm in ARMS}

# per-repo resolved
def per_repo(arm):
    res = data[arm]["resolved"]
    tot = collections.Counter(repo_of[i] for i in pool_ids)
    rr = collections.Counter(repo_of[i] for i in res)
    return {r: [rr.get(r,0), tot[r]] for r in sorted(tot)}
repo_tbl = {arm: per_repo(arm) for arm in ARMS}

# post-resolve vs pre-resolve halts (halt on an instance that ended resolved vs not)
def halt_split(arm):
    res = data[arm]["resolved"]
    hids = set(cov[arm]["halt_ids"])
    post = len(hids & res); pre = len(hids - res)
    return dict(post_resolve_halts=post, pre_resolve_halts=pre)
halt_sp = {arm: halt_split(arm) for arm in ARMS}

report = {
    "pool_sha256": pool["pool_sha256"], "n": len(pool_ids),
    "primary_resolve_at_1": {
        "ar_resolved": len(ar_res), "diffusion_resolved": len(df_res),
        "both": len(both), "ar_only_b": b, "diffusion_only_c": c, "neither": len(neither),
        "net_diffusion_minus_ar": net, "mcnemar_exact_p_2sided": round(p, 4),
        "parity_claim": parity, "data_sufficient": data_sufficient,
        "parity_rule": "|net|<=2 AND p>=0.05",
        "ar_only_ids": sorted(ar_only), "diffusion_only_ids": sorted(df_only),
    },
    "secondary": {
        "throughput": thr, "tokens_and_halts": cov,
        "halt_resolve_split": halt_sp, "per_repo_resolved": repo_tbl,
    },
    "scoring_present": {arm: bool(data[arm]["scoring"]) for arm in ARMS},
}
(root/"report.json").write_text(json.dumps(report, indent=2))

# ---- markdown + table ----
L = []
L.append(f"# W2 N=50 stock-AR vs diffusion — resolve@1 (paired)\n")
if not data_sufficient:
    _miss = [a for a in ARMS if not data[a]["scoring"] or len(data[a]["eps"]) != len(pool_ids)]
    L.append(f"> **STATUS: RUN NOT COMPLETE — no verdict.** Insufficient data (arms not fully "
             f"scored: {', '.join(_miss)}). All `0/50` cells below mean *not run*, NOT *failed*. "
             f"Launch: `setsid bash runs/w2_n50/run_all.sh &`. See ANOMALIES at bottom.\n")
L.append(f"pool_sha256 `{pool['pool_sha256'][:16]}…`  N={len(pool_ids)}  scoring: "
         f"AR={'ok' if data['ar']['scoring'] else 'MISSING'} diffusion={'ok' if data['diffusion']['scoring'] else 'MISSING'}\n")
L.append("## PRIMARY — paired resolve@1 McNemar")
L.append(f"- AR resolved: **{len(ar_res)}/50**   diffusion resolved: **{len(df_res)}/50**")
L.append(f"- both={len(both)}  AR-only(b)={b}  diffusion-only(c)={c}  neither={len(neither)}")
L.append(f"- net (diffusion−AR) = **{net:+d}**   McNemar exact 2-sided p = **{p:.4f}**")
_pv = "INSUFFICIENT DATA" if parity is None else ("YES" if parity else "NO")
L.append(f"- **PARITY CLAIM ({report['primary_resolve_at_1']['parity_rule']}): {_pv}**")
if ar_only: L.append(f"  - AR-only ids: {sorted(ar_only)}")
if df_only: L.append(f"  - diffusion-only ids: {sorted(df_only)}")
L.append("\n## SECONDARY")
L.append("### Throughput (episodes / GPU-h at run concurrency)")
for arm in ARMS:
    t = thr[arm]
    if t: L.append(f"- {arm}: **{t['eps_per_gpu_h']} eps/GPU-h**  (n={t['n']}, wall={t['wall_seconds']}s, c={t['concurrency']})")
    else: L.append(f"- {arm}: (timing missing)")
L.append("### Tokens + loop-halt covariates")
L.append("| arm | eps | resolved | input_tok | output_tok | loop_halts | budget | timeouts | empty | med_turns |")
L.append("|---|---|---|---|---|---|---|---|---|---|")
for arm in ARMS:
    c_ = cov[arm]
    L.append(f"| {arm} | {c_['n_episodes']} | {len(data[arm]['resolved'])} | {c_['input_tokens']:,} | "
             f"{c_['output_tokens']:,} | {c_['loop_halts']} | {c_['budget_halts']} | {c_['timeouts']} | "
             f"{c_['empty_patches']} | {c_['median_turns']} |")
L.append("### Loop-halt resolve split (covariate, pre-registered)")
for arm in ARMS:
    hs = halt_sp[arm]
    L.append(f"- {arm}: post-resolve halts={hs['post_resolve_halts']}  pre-resolve halts={hs['pre_resolve_halts']}")
L.append("### Per-repo resolved (arm: resolved/total)")
repos = sorted(set(repo_of.values()))
L.append("| repo | AR | diffusion |")
L.append("|---|---|---|")
for r in repos:
    a = repo_tbl['ar'][r]; d = repo_tbl['diffusion'][r]
    L.append(f"| {r} | {a[0]}/{a[1]} | {d[0]}/{d[1]} |")
# anomalies
anom = []
for arm in ARMS:
    if cov[arm]["missing"]: anom.append(f"{arm}: {len(cov[arm]['missing'])} episodes MISSING metadata: {cov[arm]['missing'][:6]}")
    if cov[arm]["n_episodes"] != 50: anom.append(f"{arm}: only {cov[arm]['n_episodes']}/50 episodes present")
    if not data[arm]["scoring"]: anom.append(f"{arm}: scoring report MISSING")
    elif data[arm]["scoring"].get("error_ids"): anom.append(f"{arm}: scoring error_ids={data[arm]['scoring']['error_ids']}")
if anom:
    L.append("\n## ANOMALIES"); [L.append(f"- {a}") for a in anom]
md = "\n".join(L) + "\n"
(root/"report.md").write_text(md)
print(md)
