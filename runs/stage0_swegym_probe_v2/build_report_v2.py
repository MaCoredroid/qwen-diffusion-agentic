#!/usr/bin/env python
"""Stage 0 phase 2 PROBE v2 report builder (ENVELOPE-CORRECTED).

Same measured schema as the greedy v1 builder, plus a greedy-vs-envelope block:
  * corrected yield resolve@1 (+ Wilson95) vs the greedy 0.15 baseline
  * GPU-min/attempt (this run's server-up window / 20)
  * exit-code distribution both runs -> the DIRECT degenerate-regime test
    (turn-limit exit 53 = the qwen greedy loop-halt / non-termination signature;
     loop-detect exit 1; clean exit 0), greedy-vs-envelope
  * D1 repricing at the corrected yield (>=0.20 => single-attempt GO bar).

Reads the greedy baseline from runs/stage0_swegym_probe/report.json and its
per-episode runner_metadata for the paired exit-code comparison. Env-acquisition
cost is carried over from v1 (same pre-pulled images; no pull stage in v2).
"""
import json, glob, os, statistics as st

RUN = "runs/stage0_swegym_probe_v2"
V1 = "runs/stage0_swegym_probe"
N_TARGET_LO, N_TARGET_HI = 600, 1000
GO_BAR = 0.20


def _wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return [round((c - h) / d, 4), round((c + h) / d, 4)]


def _pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1)))))
    return xs[k]


def load_gpu(run):
    fp = f"{run}/gen/gpu_util.csv"
    util, mem, pw, ts = [], [], [], []
    if os.path.exists(fp):
        for l in open(fp):
            l = l.strip()
            if not l:
                continue
            parts = l.split(",")
            if len(parts) < 4:
                continue
            ts.append(parts[0])
            try:
                util.append(float(parts[1])); mem.append(float(parts[2])); pw.append(float(parts[3]))
            except ValueError:
                pass
    return util, mem, pw, ts


def gen_wall_min(ts):
    if len(ts) < 2:
        return None
    import datetime as dt
    def p(x):
        return dt.datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ")
    return (p(ts[-1]) - p(ts[0])).total_seconds() / 60.0


def load_episodes(run):
    """instance_id -> {exit_code, patch_bytes, num_turns, agent_wall_s}."""
    eps = {}
    for meta in glob.glob(f"{run}/gen/*/verified/per_task/*/runner_metadata.json"):
        try:
            m = json.load(open(meta))
        except Exception:
            continue
        iid = m.get("instance_id") or os.path.basename(os.path.dirname(meta))
        q = m.get("qwen") or {}
        eps[iid] = {
            "instance_id": iid,
            "exit_code": q.get("exit_code"),
            "patch_bytes": m.get("patch_bytes"),
            "num_turns": q.get("num_turns"),
            "agent_wall_s": q.get("elapsed_s"),
            "empty_patch_retry": m.get("empty_patch_retry"),
        }
    return eps


def exit_hist(eps):
    """Categorize exit codes into the degenerate-regime buckets.
      0  -> clean terminal (agent produced a final answer / stopped itself)
      53 -> turn-limit exhaustion (the greedy loop/repeat non-termination signature)
      1  -> qwen-code loop-detector halt
      55 -> budget stop
      other/None -> other
    """
    b = {"clean": 0, "turn_limit": 0, "loop_detect": 0, "budget": 0, "other": 0}
    raw = {}
    for e in eps.values():
        c = e.get("exit_code")
        raw[c] = raw.get(c, 0) + 1
        if c == 0:
            b["clean"] += 1
        elif c == 53:
            b["turn_limit"] += 1
        elif c == 1:
            b["loop_detect"] += 1
        elif c == 55:
            b["budget"] += 1
        else:
            b["other"] += 1
    return b, {str(k): v for k, v in sorted(raw.items(), key=lambda kv: (kv[0] is None, kv[0]))}


def load_predictions(run):
    preds = {}
    for f in glob.glob(f"{run}/gen/*/verified/predictions.jsonl"):
        for l in open(f):
            l = l.strip()
            if l:
                r = json.loads(l); preds[r["instance_id"]] = r
    return preds


def load_score(run, rid):
    cands = glob.glob(f"{run}/score/*.{rid}.json")
    if not cands:
        cands = [c for c in glob.glob(f"{run}/score/*.json")
                 if os.path.basename(c) not in ("timing.json",)]
    rep = json.load(open(cands[0])) if cands else {}
    tp = f"{run}/score/timing.json"
    timing = json.load(open(tp)) if os.path.exists(tp) else {}
    return rep, timing


def main():
    rid = os.environ.get("RUN_ID", "probe20env")
    sub = json.load(open(f"{RUN}/artifacts/subset_probe20.json"))
    ids = sub["instance_ids"]
    n = len(ids)

    util, mem, pw, ts = load_gpu(RUN)
    wall_min = gen_wall_min(ts)
    eps = load_episodes(RUN)
    preds = load_predictions(RUN)
    patch_produced = sum(1 for r in preds.values() if (r.get("model_patch") or "").strip())
    ehist, eraw = exit_hist(eps)
    n_retry_fired = sum(1 for e in eps.values()
                        if isinstance(e.get("empty_patch_retry"), dict)
                        and (e["empty_patch_retry"] or {}).get("max_retries", 0) > 0
                        and (e["empty_patch_retry"] or {}).get("attempts", 0) > 0)

    rep, timing = load_score(RUN, rid)
    resolved = rep.get("resolved_instances", 0)
    completed = rep.get("completed_instances", 0)
    empty = rep.get("empty_patch_instances", 0)
    errors = rep.get("error_instances", 0)
    resolved_ids = rep.get("resolved_ids", [])
    score_wall_s = timing.get("score_wall_s")
    maxw = timing.get("maxw")

    ep_walls = [e["agent_wall_s"] for e in eps.values() if isinstance(e.get("agent_wall_s"), (int, float))]

    yield_rate = (resolved / n) if n else None
    gpu_min_per_attempt = (wall_min / n) if (wall_min and n) else None
    docker_min_per_eval = (score_wall_s / 60.0 / n) if score_wall_s else None

    # ---- greedy v1 baseline (paired) ----
    v1 = json.load(open(f"{V1}/report.json")) if os.path.exists(f"{V1}/report.json") else {}
    v1_eps = load_episodes(V1)
    v1_ehist, v1_eraw = exit_hist(v1_eps)
    v1_meas = v1.get("MEASURED", {})
    v1_env = v1.get("env_acquisition", {})
    v1_yield = v1_meas.get("yield_resolve_at_1")
    v1_resolved = v1.get("scoring", {}).get("resolved")
    v1_gpu_min = v1_meas.get("gpu_min_per_attempt")

    def price(y, gmin, dmin):
        if not y or y <= 0:
            return None
        alo, ahi = N_TARGET_LO / y, N_TARGET_HI / y
        return {
            "attempts_needed": [round(alo), round(ahi)],
            "serving_gpu_h": [round(alo * (gmin or 0) / 60.0, 1), round(ahi * (gmin or 0) / 60.0, 1)],
            "docker_eval_wall_h": [round(alo * (dmin or 0) / 60.0, 1), round(ahi * (dmin or 0) / 60.0, 1)],
        }

    decision = ("GO_single_attempt" if (yield_rate is not None and yield_rate >= GO_BAR)
                else "ADJUST_fallbacks")

    out = {
        "artifact": f"{RUN}/report.json",
        "run": "stage0_swegym_probe_v2 (ENVELOPE-CORRECTED)",
        "envelope": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed_base": 1234,
                     "empty_patch_retries": int(os.environ.get("SWE_EMPTY_PATCH_RETRIES", "1")),
                     "source": "reference envelope banked in runs/stage_c_n5v2/report.md; "
                               "forced proxy-side via LUMO_PROXY_FORCE_* (default-OFF passthrough otherwise)"},
        "n_attempts": n,
        "instances": ids,
        "env_acquisition": {**v1_env, "note": "carried over from greedy v1 (same pre-pulled images; no pull stage in v2)"},
        "generation": {
            "concurrency": int(os.environ.get("MAX_NUM_SEQS", 4)),
            "server_up_wall_min": round(wall_min, 1) if wall_min else None,
            "gpu_util_pct_mean": round(st.mean(util), 1) if util else None,
            "gpu_util_pct_median": round(st.median(util), 1) if util else None,
            "gpu_util_pct_p90": _pct(util, 90),
            "gpu_util_samples": len(util),
            "gpu_mem_mib_median": round(st.median(mem), 0) if mem else None,
            "gpu_power_w_median": round(st.median(pw), 0) if pw else None,
            "episode_wall_s_median": round(st.median(ep_walls), 1) if ep_walls else None,
            "episode_wall_s_p90": _pct(ep_walls, 90),
            "patch_produced": patch_produced,
            "empty_patch_retry_fired": n_retry_fired,
            "gpu_min_per_attempt": round(gpu_min_per_attempt, 2) if gpu_min_per_attempt else None,
        },
        "scoring": {
            "harness": "SWE-Gym/SWE-Bench-Fork @242429c (patched: reuse prebuilt instance image); reused from v1",
            "resolved": resolved, "completed": completed,
            "empty_patch": empty, "errors": errors,
            "resolved_ids": resolved_ids,
            "score_wall_s": score_wall_s, "max_workers": maxw,
            "docker_min_per_eval": round(docker_min_per_eval, 2) if docker_min_per_eval else None,
        },
        "MEASURED": {
            "yield_resolve_at_1": round(yield_rate, 4) if yield_rate is not None else None,
            "yield_wilson95": _wilson(resolved, n),
            "patch_produced_rate": round(patch_produced / n, 4) if n else None,
            "gpu_min_per_attempt": round(gpu_min_per_attempt, 2) if gpu_min_per_attempt else None,
            "docker_min_per_eval": round(docker_min_per_eval, 2) if docker_min_per_eval else None,
            "env_min_per_instance": v1_env.get("env_min_per_instance"),
        },
        "GREEDY_VS_ENVELOPE": {
            "yield_resolve_at_1": {"greedy": v1_yield, "envelope": round(yield_rate, 4) if yield_rate is not None else None},
            "resolved_of_20": {"greedy": v1_resolved, "envelope": resolved},
            "patch_produced_of_20": {"greedy": v1.get("generation", {}).get("patch_produced"), "envelope": patch_produced},
            "exit_code_buckets": {"greedy": v1_ehist, "envelope": ehist},
            "exit_code_raw": {"greedy": v1_eraw, "envelope": eraw},
            "turn_limit_rate": {"greedy": round(v1_ehist["turn_limit"] / n, 4),
                                 "envelope": round(ehist["turn_limit"] / n, 4)},
            "clean_terminal_rate": {"greedy": round(v1_ehist["clean"] / n, 4),
                                     "envelope": round(ehist["clean"] / n, 4)},
            "loop_or_turnlimit_rate": {"greedy": round((v1_ehist["turn_limit"] + v1_ehist["loop_detect"]) / n, 4),
                                        "envelope": round((ehist["turn_limit"] + ehist["loop_detect"]) / n, 4)},
            "gpu_min_per_attempt": {"greedy": v1_gpu_min, "envelope": round(gpu_min_per_attempt, 2) if gpu_min_per_attempt else None},
            "note": "turn-limit (exit 53) + loop-detect (exit 1) is the qwen greedy-repetition "
                    "degenerate-regime signature; a materially lower rate under the envelope confirms it.",
        },
        "D1_DECISION": {
            "go_bar_yield": GO_BAR,
            "corrected_yield": round(yield_rate, 4) if yield_rate is not None else None,
            "verdict": decision,
            "note": f"corrected yield {'>=' if (yield_rate or 0) >= GO_BAR else '<'} {GO_BAR} "
                    f"=> {'single-attempt GO' if decision.startswith('GO') else 'ADJUST fallbacks stand'}",
        },
        "PRICE_full_campaign": {
            "note": "attempts = target/yield; GPU-h = attempts*gpu_min_per_attempt/60; "
                    "docker wall = attempts*docker_min_per_eval/60. 1 attempt/instance (best-of-k not modeled).",
            "at_corrected_yield": price(yield_rate, gpu_min_per_attempt, docker_min_per_eval),
            "at_greedy_yield": price(v1_yield, v1_gpu_min, v1_meas.get("docker_min_per_eval")),
        },
    }
    json.dump(out, open(f"{RUN}/report.json", "w"), indent=2)

    # ---- table ----
    m = out["MEASURED"]; g = out["generation"]; s = out["scoring"]; gv = out["GREEDY_VS_ENVELOPE"]
    L = []
    L.append("STAGE 0 PHASE 2 PROBE v2 — ENVELOPE-CORRECTED yield (20 SWE-Gym instances, stock-AR @concurrency %s)" % g["concurrency"])
    L.append("envelope: temp=0.6 top_p=0.95 top_k=20 seed=1234 (proxy-forced) + empty-patch-retries=1")
    L.append("=" * 92)
    L.append(f"GENERATION (concurrency={g['concurrency']}):")
    L.append(f"  server_up_wall={g['server_up_wall_min']}min  GPU util%% mean={g['gpu_util_pct_mean']} "
             f"median={g['gpu_util_pct_median']} p90={g['gpu_util_pct_p90']} (n={g['gpu_util_samples']})")
    L.append(f"  GPU mem MiB median={g['gpu_mem_mib_median']}  power W median={g['gpu_power_w_median']}")
    L.append(f"  episode wall_s median={g['episode_wall_s_median']} p90={g['episode_wall_s_p90']}  "
             f"patch_produced={g['patch_produced']}/{n}  empty_patch_retry_fired={g['empty_patch_retry_fired']}")
    L.append(f"SCORING ({s['harness']}):")
    L.append(f"  resolved={s['resolved']}/{n}  completed={s['completed']}  empty_patch={s['empty_patch']} "
             f"errors={s['errors']}  score_wall={s['score_wall_s']}s maxw={s['max_workers']}")
    L.append(f"  resolved_ids={s['resolved_ids']}")
    L.append("-" * 92)
    L.append(f"CORRECTED yield resolve@1 = {m['yield_resolve_at_1']} (Wilson95 {m['yield_wilson95']})   "
             f"patch_produced_rate = {m['patch_produced_rate']}")
    L.append(f"CORRECTED GPU-min/attempt = {m['gpu_min_per_attempt']}   docker-min/eval = {m['docker_min_per_eval']}   "
             f"env-min/instance = {m['env_min_per_instance']}")
    L.append("=" * 92)
    L.append("YIELD TABLE — GREEDY (v1) vs ENVELOPE (v2), SAME 20 instances / SAME scorer:")
    L.append("  metric                     greedy      envelope")
    L.append(f"  resolve@1 yield            {gv['yield_resolve_at_1']['greedy']:<11} {gv['yield_resolve_at_1']['envelope']}")
    L.append(f"  resolved / 20              {gv['resolved_of_20']['greedy']:<11} {gv['resolved_of_20']['envelope']}")
    L.append(f"  patch_produced / 20        {gv['patch_produced_of_20']['greedy']:<11} {gv['patch_produced_of_20']['envelope']}")
    L.append(f"  turn-limit rate (exit53)   {gv['turn_limit_rate']['greedy']:<11} {gv['turn_limit_rate']['envelope']}")
    L.append(f"  loop+turnlimit rate        {gv['loop_or_turnlimit_rate']['greedy']:<11} {gv['loop_or_turnlimit_rate']['envelope']}")
    L.append(f"  clean-terminal rate        {gv['clean_terminal_rate']['greedy']:<11} {gv['clean_terminal_rate']['envelope']}")
    L.append(f"  GPU-min/attempt            {gv['gpu_min_per_attempt']['greedy']:<11} {gv['gpu_min_per_attempt']['envelope']}")
    L.append(f"  exit-code buckets greedy   = {gv['exit_code_buckets']['greedy']}")
    L.append(f"  exit-code buckets envelope = {gv['exit_code_buckets']['envelope']}")
    L.append("  (degenerate-regime test: turn-limit+loop = qwen greedy-repetition non-termination signature)")
    L.append("-" * 92)
    d1 = out["D1_DECISION"]
    L.append(f"D1 DECISION @ GO-bar {d1['go_bar_yield']}: corrected yield {d1['corrected_yield']} -> {d1['verdict']}")
    pc = out["PRICE_full_campaign"]["at_corrected_yield"]
    pg = out["PRICE_full_campaign"]["at_greedy_yield"]
    if pc:
        L.append(f"PRICE {N_TARGET_LO}-{N_TARGET_HI} keepers @ corrected yield {m['yield_resolve_at_1']}:")
        L.append(f"  attempts={pc['attempts_needed']}  serving GPU-h={pc['serving_gpu_h']}  docker-eval wall-h={pc['docker_eval_wall_h']}")
    if pg:
        L.append(f"  (greedy 0.15 was: attempts={pg['attempts_needed']}  serving GPU-h={pg['serving_gpu_h']}  docker-eval wall-h={pg['docker_eval_wall_h']})")
    open(f"{RUN}/report_table.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
