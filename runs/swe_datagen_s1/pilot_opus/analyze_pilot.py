#!/usr/bin/env python3
"""Per-episode + aggregate analysis for the Opus-teacher pilot.

Sources of truth:
  * per-episode boundaries/turns/patch: gen/<iid>/verified/per_task/<iid>/runner_metadata.json
  * cache breakdown: usage_adapter.jsonl windowed by [started_at, ended_at]
  * resolved status: batch/score/datagen-eval.batch.json (if present)

Pricing = PRIOR KNOWLEDGE (Claude API, claude-api skill, cached 2026-06-24), Opus 4.8:
  input $5.00/1M, output $25.00/1M, cache-read $0.50/1M (0.1x), cache-write(5m) $6.25/1M (1.25x)
"""
import datetime as dt
import glob
import json
from pathlib import Path

GEN = Path("runs/swe_datagen_s1/pilot_opus/gen")
ULOG = Path("runs/swe_datagen_s1/pilot_opus/usage_adapter.jsonl")
SCORE = Path("runs/swe_datagen_s1/pilot_opus/batch/score/datagen-eval.batch.json")
IDS = ["conan-io__conan-10213", "conan-io__conan-10408", "modin-project__modin-5507",
       "pydantic__pydantic-4882", "pydantic__pydantic-4911", "pandas-dev__pandas-47475",
       "pandas-dev__pandas-47493", "getmoto__moto-4867", "getmoto__moto-4874",
       "dask__dask-10342"]
NEAR_MISS = {"conan-io__conan-10213", "conan-io__conan-10408", "modin-project__modin-5507",
             "pydantic__pydantic-4882", "pydantic__pydantic-4911"}
P_IN, P_OUT, P_CR, P_CW = 5.0/1e6, 25.0/1e6, 0.50/1e6, 6.25/1e6


def iso(s):
    if not s:
        return None
    try:
        return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp()
    except Exception:
        return None


rows = [json.loads(l) for l in ULOG.read_text().splitlines() if l.strip()]

resolved = set()
if SCORE.exists():
    rep = json.loads(SCORE.read_text())
    resolved = set(rep.get("resolved_ids", []))

report_present = SCORE.exists()
episodes = []
for iid in IDS:
    mfs = glob.glob(f"{GEN}/{iid}/*/per_task/{iid}/runner_metadata.json")
    if not mfs:
        episodes.append({"instance_id": iid, "status": "NO_METADATA", "resolved": None})
        continue
    meta = json.loads(Path(mfs[0]).read_text())
    q = meta.get("qwen") or {}
    t0 = iso(meta.get("started_at")); t1 = iso(meta.get("ended_at"))
    win = [r for r in rows if t0 and t1 and (t0 - 3) <= r["ts"] <= (t1 + 3)]
    cr = sum(r.get("cache_read_input_tokens", 0) for r in win)
    cw = sum(r.get("cache_creation_input_tokens", 0) for r in win)
    un = sum(r.get("uncached_input_tokens", 0) for r in win)
    tot_in = cr + cw + un
    out = sum(r.get("completion_tokens", 0) for r in win)
    # cost
    cost_cached = un*P_IN + cr*P_CR + cw*P_CW + out*P_OUT
    cost_uncached = tot_in*P_IN + out*P_OUT
    episodes.append({
        "instance_id": iid,
        "family": "near_miss" if iid in NEAR_MISS else "fresh_cov",
        "resolved": (iid in resolved) if report_present else None,
        "patch_bytes": meta.get("patch_bytes"),
        "turns": q.get("num_turns"),
        "elapsed_s": round(q.get("elapsed_s") or 0, 1),
        "n_requests": len(win),
        "input_tokens_total": tot_in,
        "output_tokens": out,
        "uncached_input": un,
        "cache_read": cr,
        "cache_creation": cw,
        "cache_read_frac": round(cr/tot_in, 4) if tot_in else 0.0,
        "cost_cached_usd": round(cost_cached, 4),
        "cost_uncached_usd": round(cost_uncached, 4),
    })

n = len(IDS)
scored = [e for e in episodes if e.get("resolved") is not None]
n_resolved = sum(1 for e in episodes if e.get("resolved"))
tot_in = sum(e.get("input_tokens_total", 0) for e in episodes)
tot_out = sum(e.get("output_tokens", 0) for e in episodes)
tot_cr = sum(e.get("cache_read", 0) for e in episodes)
tot_cw = sum(e.get("cache_creation", 0) for e in episodes)
tot_un = sum(e.get("uncached_input", 0) for e in episodes)
cost_cached = sum(e.get("cost_cached_usd", 0) for e in episodes)
cost_uncached = sum(e.get("cost_uncached_usd", 0) for e in episodes)
yield_ = (n_resolved / n) if report_present else None

agg = {
    "n_episodes": n,
    "n_resolved": n_resolved if report_present else None,
    "yield": round(yield_, 4) if yield_ is not None else None,
    "report_present": report_present,
    "tokens": {"input_total": tot_in, "output_total": tot_out,
               "cache_read": tot_cr, "cache_creation": tot_cw, "uncached_input": tot_un,
               "cache_read_frac_of_input": round(tot_cr/tot_in, 4) if tot_in else 0.0},
    "cost_usd": {"cached_total": round(cost_cached, 2), "uncached_total": round(cost_uncached, 2),
                 "cached_per_episode": round(cost_cached/n, 3), "uncached_per_episode": round(cost_uncached/n, 3),
                 "cost_reduction_x": round(cost_uncached/cost_cached, 2) if cost_cached else None},
    "per_keeper_usd": ({"cached": round(cost_cached/n_resolved, 2),
                        "uncached": round(cost_uncached/n_resolved, 2)} if (report_present and n_resolved) else None),
}
print(json.dumps({"episodes": episodes, "aggregate": agg}, indent=1))
