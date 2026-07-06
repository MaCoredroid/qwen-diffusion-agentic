#!/usr/bin/env python
"""Stage 0 phase 2 PROBE report builder. Combines:
  * pull/pull.jsonl            -> env acquisition cost (pull_s, size) per instance
  * gen/gpu_util.csv           -> GPU utilization during concurrent generation
  * gen/*/verified/campaign_summary.json + per_task -> episode wall/turns/verdicts
  * score/<model>.<rid>.json   -> official resolve yield
Emits report.json + report_table.txt with the MEASURED table + the PRICE recompute.
"""
import json, glob, os, statistics as st, sys

RUN = "runs/stage0_swegym_probe"
N_TARGET_LO, N_TARGET_HI = 600, 1000


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


def load_pull():
    rows = []
    fp = f"{RUN}/pull/pull.jsonl"
    if os.path.exists(fp):
        rows = [json.loads(l) for l in open(fp) if l.strip()]
    return rows


def load_gpu():
    fp = f"{RUN}/gen/gpu_util.csv"
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


def load_episodes():
    """Per-episode records from each shard's per_task/<iid>/metadata.json or the
    campaign_summary. Return list of dicts: instance_id, verdict, patch_bytes,
    agent_wall_s, turns."""
    eps = {}
    for meta in glob.glob(f"{RUN}/gen/*/verified/per_task/*/runner_metadata.json"):
        try:
            m = json.load(open(meta))
        except Exception:
            continue
        iid = m.get("instance_id") or os.path.basename(os.path.dirname(meta))
        q = m.get("qwen") or {}
        eps[iid] = {
            "instance_id": iid,
            "patch_bytes": m.get("patch_bytes"),
            "num_turns": q.get("num_turns"),
            "agent_wall_s": q.get("elapsed_s"),
            "timed_out_turnlimit": q.get("exit_code") == 53,
            "status": (m.get("eval_report") or {}).get("verdict"),
        }
    # fall back / augment with campaign summaries
    summaries = []
    for cs in glob.glob(f"{RUN}/gen/*/verified/campaign_summary.json"):
        try:
            summaries.append(json.load(open(cs)))
        except Exception:
            pass
    return eps, summaries


def load_predictions():
    preds = {}
    for f in glob.glob(f"{RUN}/gen/*/verified/predictions.jsonl"):
        for l in open(f):
            l = l.strip()
            if l:
                r = json.loads(l); preds[r["instance_id"]] = r
    return preds


def load_score():
    rid = os.environ.get("RUN_ID", "probe20")
    cands = glob.glob(f"{RUN}/score/*.{rid}.json")
    if not cands:
        cands = [c for c in glob.glob(f"{RUN}/score/*.json")
                 if os.path.basename(c) not in ("timing.json",)]
    rep = json.load(open(cands[0])) if cands else {}
    timing = json.load(open(f"{RUN}/score/timing.json")) if os.path.exists(f"{RUN}/score/timing.json") else {}
    return rep, timing


def main():
    sub = json.load(open(f"{RUN}/artifacts/subset_probe20.json"))
    ids = sub["instance_ids"]
    n = len(ids)

    pull = load_pull()
    pull_ok = [r for r in pull if r.get("status") in ("ok", "cached")]
    pull_s = [r["pull_s"] for r in pull if r.get("status") == "ok"]     # true pulls only
    sizes = [r["size_bytes"] for r in pull_ok]

    util, mem, pw, ts = load_gpu()
    # server-up wall from gpu sampler span
    gen_wall_min = None
    if len(ts) >= 2:
        import datetime as dt
        def p(x): return dt.datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ")
        gen_wall_min = (p(ts[-1]) - p(ts[0])).total_seconds() / 60.0

    eps, summaries = load_episodes()
    preds = load_predictions()
    patch_produced = sum(1 for r in preds.values() if (r.get("model_patch") or "").strip())

    rep, timing = load_score()
    resolved = rep.get("resolved_instances", 0)
    completed = rep.get("completed_instances", 0)
    empty = rep.get("empty_patch_instances", 0)
    errors = rep.get("error_instances", 0)
    resolved_ids = rep.get("resolved_ids", [])
    score_wall_s = timing.get("score_wall_s")
    maxw = timing.get("maxw")

    # episode wall
    ep_walls = [e["agent_wall_s"] for e in eps.values() if isinstance(e.get("agent_wall_s"), (int, float))]
    ep_turns = [e["num_turns"] for e in eps.values() if isinstance(e.get("num_turns"), (int, float))]

    yield_rate = (resolved / n) if n else None
    # GPU-min/attempt: whole GPU occupied during generation window / attempts
    gpu_min_per_attempt = (gen_wall_min / n) if (gen_wall_min and n) else None
    docker_min_per_eval = (score_wall_s / 60.0 / n) if score_wall_s else None
    env_pull_min_per_inst = (sum(pull_s) / 60.0 / len(pull_s)) if pull_s else 0.0

    def price(y):
        if not y or y <= 0:
            return None
        att_lo = N_TARGET_LO / y
        att_hi = N_TARGET_HI / y
        gpuh_lo = att_lo * (gpu_min_per_attempt or 0) / 60.0
        gpuh_hi = att_hi * (gpu_min_per_attempt or 0) / 60.0
        dockh_lo = att_lo * (docker_min_per_eval or 0) / 60.0
        dockh_hi = att_hi * (docker_min_per_eval or 0) / 60.0
        return {
            "attempts_needed": [round(att_lo), round(att_hi)],
            "serving_gpu_h": [round(gpuh_lo, 1), round(gpuh_hi, 1)],
            "docker_eval_wall_h": [round(dockh_lo, 1), round(dockh_hi, 1)],
        }

    out = {
        "artifact": f"{RUN}/report.json",
        "n_attempts": n,
        "instances": ids,
        "env_acquisition": {
            "mode": "docker pull xingyaoww prebuilt + re-tag (NO from-scratch build)",
            "n_pulled": len(pull_s), "n_cached": sum(1 for r in pull if r.get("status") == "cached"),
            "n_failed": sum(1 for r in pull if r.get("status") == "pull_failed"),
            "pull_s_mean": round(st.mean(pull_s), 1) if pull_s else None,
            "pull_s_median": round(st.median(pull_s), 1) if pull_s else None,
            "pull_s_p90": _pct(pull_s, 90),
            "pull_s_total": round(sum(pull_s), 1) if pull_s else None,
            "env_min_per_instance": round(env_pull_min_per_inst, 2),
            "size_gb_mean": round(st.mean(sizes) / 1e9, 2) if sizes else None,
            "size_gb_total": round(sum(sizes) / 1e9, 1) if sizes else None,
        },
        "generation": {
            "concurrency": int(os.environ.get("MAX_NUM_SEQS", 4)),
            "server_up_wall_min": round(gen_wall_min, 1) if gen_wall_min else None,
            "gpu_util_pct_mean": round(st.mean(util), 1) if util else None,
            "gpu_util_pct_median": round(st.median(util), 1) if util else None,
            "gpu_util_pct_p90": _pct(util, 90),
            "gpu_util_samples": len(util),
            "gpu_mem_mib_median": round(st.median(mem), 0) if mem else None,
            "gpu_power_w_median": round(st.median(pw), 0) if pw else None,
            "episode_wall_s_median": round(st.median(ep_walls), 1) if ep_walls else None,
            "episode_wall_s_p90": _pct(ep_walls, 90),
            "episode_turns_median": st.median(ep_turns) if ep_turns else None,
            "patch_produced": patch_produced,
            "gpu_min_per_attempt": round(gpu_min_per_attempt, 2) if gpu_min_per_attempt else None,
        },
        "scoring": {
            "harness": "SWE-Gym/SWE-Bench-Fork @242429c (patched: reuse prebuilt instance image)",
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
            "env_min_per_instance": round(env_pull_min_per_inst, 2),
        },
        "PRICE_full_campaign": {
            "note": "attempts = target/yield; GPU-h = attempts*gpu_min_per_attempt/60; "
                    "docker wall = attempts*docker_min_per_eval/60. best-of-k not modeled (1 attempt/instance here).",
            "at_measured_yield": price(yield_rate),
        },
    }
    json.dump(out, open(f"{RUN}/report.json", "w"), indent=2)

    # table
    L = []
    L.append("STAGE 0 PHASE 2 PROBE — MEASURED (20 SWE-Gym instances, stock-AR @concurrency %s)" % out["generation"]["concurrency"])
    L.append("=" * 78)
    ea = out["env_acquisition"]; g = out["generation"]; s = out["scoring"]; m = out["MEASURED"]
    L.append(f"ENV ACQUISITION (pull prebuilt xingyaoww + retag):")
    L.append(f"  pulled={ea['n_pulled']} cached={ea['n_cached']} failed={ea['n_failed']}  "
             f"pull_s mean={ea['pull_s_mean']} median={ea['pull_s_median']} p90={ea['pull_s_p90']}  "
             f"total={ea['pull_s_total']}s")
    L.append(f"  size GB mean={ea['size_gb_mean']} total={ea['size_gb_total']}  env_min/instance={ea['env_min_per_instance']}")
    L.append(f"GENERATION (concurrency={g['concurrency']}):")
    L.append(f"  server_up_wall={g['server_up_wall_min']}min  GPU util%% mean={g['gpu_util_pct_mean']} "
             f"median={g['gpu_util_pct_median']} p90={g['gpu_util_pct_p90']} (n={g['gpu_util_samples']})")
    L.append(f"  GPU mem MiB median={g['gpu_mem_mib_median']}  power W median={g['gpu_power_w_median']}")
    L.append(f"  episode wall_s median={g['episode_wall_s_median']} p90={g['episode_wall_s_p90']}  "
             f"turns median={g['episode_turns_median']}  patch_produced={g['patch_produced']}/{n}")
    L.append(f"SCORING ({s['harness']}):")
    L.append(f"  resolved={s['resolved']}/{n}  completed={s['completed']}  empty_patch={s['empty_patch']} "
             f"errors={s['errors']}  score_wall={s['score_wall_s']}s maxw={s['max_workers']}")
    L.append(f"  resolved_ids={s['resolved_ids']}")
    L.append("-" * 78)
    L.append(f"MEASURED yield resolve@1 = {m['yield_resolve_at_1']} (Wilson95 {m['yield_wilson95']})   "
             f"patch_produced_rate = {m['patch_produced_rate']}")
    L.append(f"MEASURED GPU-min/attempt = {m['gpu_min_per_attempt']}   docker-min/eval = {m['docker_min_per_eval']}   "
             f"env-min/instance = {m['env_min_per_instance']}")
    pr = out["PRICE_full_campaign"]["at_measured_yield"]
    if pr:
        L.append("-" * 78)
        L.append(f"PRICE for {N_TARGET_LO}-{N_TARGET_HI} keepers @ measured yield {m['yield_resolve_at_1']}:")
        L.append(f"  attempts needed = {pr['attempts_needed']}   serving GPU-h = {pr['serving_gpu_h']}   "
                 f"docker-eval wall-h = {pr['docker_eval_wall_h']}")
    open(f"{RUN}/report_table.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
