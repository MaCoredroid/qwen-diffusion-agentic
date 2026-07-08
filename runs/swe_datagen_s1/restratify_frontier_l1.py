#!/usr/bin/env python3
"""L1 RE-STRATIFIER — rebuild frontier ORDER around the DEMONSTRATED per-family
yield with a NEAR-MISS EXPLOIT HEAD, through the KILL-D1 hash-asserted firewall.

WHY (post-rescue evidence, 2026-07-08): the officially-scorable head is exhausted;
the recoverable value is L1 = best-of-k re-draws of the 479 NEAR-MISSES (ids whose
best real attempt produced a non-empty FAILING patch) at their measured family
yields (0.07-0.83). The stock ledger draws COVERAGE-first ((attempt_count, idx)),
which buries count=1 near-misses behind 1660 count=0 fresh ids — so L1 never fires.
This rebuild:
  (1) puts the NEAR-MISS ids at the FRONT of `order`, grouped by family, families
      ranked by measured yield (zero-yield e.g. sphinx-doc LAST, not dropped);
  (2) then the UNATTEMPTED ids, grouped by family, ranked by yield (near-zero
      getmoto/facebookresearch LAST, not dropped — the kill bar governs);
  (3) then the empty-patch re-draws, then the terminal tail (resolved/env, which the
      ledger eligibility skips anyway);
and sets best_of_k.exploit_priority="frontier" so the (companion-patched) ledger
draws the frontier order authoritatively — near-misses FIRST. explore_rate=0.12
(the 88/12 convention); the explore POOL is data-empty because every family now has
>=1 resolved row (nothing left to "explore"), so the coverage work lives in exploit,
yield-ordered.

INVARIANT (KILL-D1, hash-asserted): this is a PERMUTATION of the existing id set
(no id added or dropped), so the eval-holdout disjointness is preserved; we RE-ASSERT
it anyway — recompute sha256(inner5 U tier0 U tier1), require == the pinned
.eval_holdout_sha256, and require new_order ∩ eval_holdout == 0.

usage: restratify_frontier_l1.py <out_frontier.json> [--explore-rate 0.12]
       [--attempts PATH]   (default: sibling attempts.jsonl; re-read at install)
"""
from __future__ import annotations
import json, sys, argparse, hashlib, time
from collections import defaultdict, Counter, OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path("/home/mark/qwen_diffusion")
FRONTIER = HERE / "frontier.json"
ATTEMPTS = HERE / "attempts.jsonl"
PIN = HERE / ".eval_holdout_sha256"
MANIFEST = REPO_ROOT / "data/swe_sft_pool/pool_manifest.json"

RING_SRC = {
    "tier0_20": REPO_ROOT / "runs/stage_c_driver/data/swe-bench-tier0-verified-instances-20260520.json",
    "tier1_100": Path("/home/mark/shared/lumoFlyWheel/docs/reports/auto_research/swe-bench-tier1-verified-instances-20260520.json"),
}
REAL_VERDICTS = {"resolved", "unresolved", "empty_patch", "error"}


def fam(iid: str) -> str:
    return iid.split("__")[0]


def repo_of(iid: str) -> str:
    owner, rest = iid.split("__", 1)
    return f"{owner}/{rest.rsplit('-', 1)[0]}"


def _ids(path: Path) -> set[str]:
    d = json.loads(path.read_text())
    if isinstance(d, dict):
        return set(d.get("instance_ids", []))
    if isinstance(d, list):
        if d and isinstance(d[0], str):
            return set(d)
        return {r.get("instance_id") for r in d if isinstance(r, dict) and r.get("instance_id")}
    return set()


def _sha(ids: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def _read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for l in p.read_text().splitlines():
        l = l.strip()
        if l:
            try:
                out.append(json.loads(l))
            except Exception:
                pass
    return out


def classify(order: list[str], rows: list[dict]):
    """Per-order-id class from VALID (non-infra_invalid) rows, and per-family yield."""
    byid = defaultdict(list)
    for r in rows:
        if r.get("infra_invalid"):
            continue
        iid = r.get("instance_id")
        if iid:
            byid[iid].append(r.get("verdict"))
    fam_res, fam_den = Counter(), Counter()
    for r in rows:
        if r.get("infra_invalid"):
            continue
        v = r.get("verdict")
        if v in REAL_VERDICTS:
            f = fam(r["instance_id"])
            fam_den[f] += 1
            if v == "resolved":
                fam_res[f] += 1
    fam_yield = {f: (fam_res[f] / fam_den[f]) for f in fam_den}

    cls = {}
    for iid in order:
        vs = byid.get(iid, [])
        if not vs:
            cls[iid] = "unattempted"
        elif "resolved" in vs:
            cls[iid] = "resolved"
        elif ("unresolved" in vs) or ("error" in vs):
            cls[iid] = "nearmiss"          # non-empty FAILING patch
        elif "empty_patch" in vs:
            cls[iid] = "empty"
        elif all(v == "env_unavailable" for v in vs):
            cls[iid] = "env_unavailable"
        else:
            cls[iid] = "other"
    return cls, fam_yield, fam_res, fam_den


def order_block(ids: list[str], fam_yield: dict) -> list[str]:
    """Group ids by family, families ranked by measured yield DESC (None/0 last),
    ties by more-ids-first then name; within a family, ids sorted (reproducible)."""
    by = defaultdict(list)
    for i in ids:
        by[fam(i)].append(i)
    for f in by:
        by[f].sort()
    fams = sorted(by, key=lambda f: (-(fam_yield.get(f) if fam_yield.get(f) is not None else -1.0),
                                     -len(by[f]), f))
    out = []
    for f in fams:
        out.extend(by[f])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--explore-rate", type=float, default=0.12)
    ap.add_argument("--attempts", default=str(ATTEMPTS))
    a = ap.parse_args()

    front = json.loads(FRONTIER.read_text())
    order = front["order"]
    rows = _read_jsonl(Path(a.attempts))
    cls, fam_yield, fam_res, fam_den = classify(order, rows)

    nearmiss = [i for i in order if cls[i] == "nearmiss"]
    unattempted = [i for i in order if cls[i] == "unattempted"]
    empty = [i for i in order if cls[i] == "empty"]
    tail = [i for i in order if cls[i] in ("resolved", "env_unavailable", "other")]

    nm_block = order_block(nearmiss, fam_yield)
    un_block = order_block(unattempted, fam_yield)
    em_block = order_block(empty, fam_yield)
    tail_block = sorted(tail)                      # terminal; ledger eligibility skips these
    new_order = nm_block + un_block + em_block + tail_block

    # ---- permutation invariant ------------------------------------------------
    assert sorted(new_order) == sorted(order), "restratify changed the id SET!"
    assert len(new_order) == len(set(new_order)) == len(order), "id count/uniqueness changed!"

    # ---- KILL-D1 hash-asserted firewall (recompute + assert == pin) ----------
    man = json.loads(MANIFEST.read_text())
    inner5 = set(man["held_out_rings"]["inner5"]["ids"])
    eval_holdout = inner5 | _ids(RING_SRC["tier0_20"]) | _ids(RING_SRC["tier1_100"])
    holdout_sha = _sha(eval_holdout)
    pinned = PIN.read_text().strip() if PIN.exists() else None
    if pinned is None:
        raise SystemExit("KILL-D1: no pinned .eval_holdout_sha256 to assert against.")
    if holdout_sha != pinned:
        raise SystemExit(f"KILL-D1 HASH MISMATCH: eval-holdout sha256={holdout_sha} != pinned {pinned}")
    leak = set(new_order) & eval_holdout
    if leak:
        raise SystemExit(f"KILL-D1: new order intersects eval holdout: {sorted(leak)[:10]}")

    # ---- best_of_k: near-miss-first exploit priority + refreshed yield map -----
    bok = dict(front.get("best_of_k", {}))
    bok["enabled"] = True
    bok["exploit_priority"] = "frontier"     # <- ledger draws frontier order authoritatively
    bok["explore_rate"] = a.explore_rate
    bok.setdefault("k_resolvable", 3)
    bok.setdefault("k_default", 1)
    ymap = {f: {"resolved": fam_res[f], "attempts": fam_den[f],
                "yield": (round(fam_yield[f], 4) if f in fam_yield else None)}
            for f in sorted(fam_den)}
    bok["yield_map_restrat"] = ymap
    bok["resolvable_head_len"] = len(nm_block)
    bok["near_miss_head_len"] = len(nm_block)
    bok["unattempted_block_len"] = len(un_block)
    bok["empty_redraw_len"] = len(em_block)
    bok["terminal_tail_len"] = len(tail_block)
    prov = dict(bok.get("provenance", {}))
    prov["l1_restratify"] = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "lever": "L1 near-miss best-of-k re-draw as the exploit head",
        "order_scheme": "[near-miss by family-yield desc] + [unattempted by family-yield desc, "
                        "near-zero last] + [empty re-draw by yield] + [terminal tail resolved/env]",
        "exploit_priority": "frontier (idx-primary; requires ledger.py exploit_priority support)",
        "explore_rate": a.explore_rate,
        "explore_note": "explore POOL data-empty (every family has >=1 resolved row -> all "
                        "resolvable); 88/12 convention kept in config, coverage lives in exploit.",
        "counts": {"near_miss": len(nm_block), "unattempted": len(un_block),
                   "empty_redraw": len(em_block), "terminal_tail": len(tail_block)},
        "evidence": ["attempts.jsonl (valid rows only)", "rescue_20260708/family_yield.json"],
        "family_yield_head": [f"{f}={round(fam_yield[f],3)}(nm={sum(1 for i in nm_block if fam(i)==f)})"
                              for f in sorted({fam(i) for i in nm_block},
                                              key=lambda f:(-fam_yield.get(f,-1), f))],
    }
    bok["provenance"] = prov

    by_repo = OrderedDict()
    for iid in new_order:
        r = repo_of(iid)
        by_repo[r] = by_repo.get(r, 0) + 1

    new_front = dict(front)
    new_front["order"] = new_order
    new_front["by_repo"] = by_repo
    new_front["n_frontier"] = len(new_order)
    new_front["best_of_k"] = bok
    new_front["kill_d1"] = "PASS"
    fw = dict(front.get("kill_d1_firewall", {}))
    fw["eval_holdout_sha256"] = holdout_sha
    fw["eval_holdout_sha256_pinned"] = pinned
    fw["hash_assert"] = "PASS"
    fw["frontier_intersect_eval_holdout"] = len(set(new_order) & eval_holdout)
    fw["restratified_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    fw["restratify_note"] = ("L1 near-miss-first permutation; id SET unchanged so "
                             "disjointness preserved and RE-ASSERTED (sha==pin, intersect==0).")
    new_front["kill_d1_firewall"] = fw

    Path(a.out).write_text(json.dumps(new_front, indent=1))
    summary = {
        "out": a.out,
        "n_frontier": len(new_order),
        "near_miss_head": len(nm_block),
        "unattempted_block": len(un_block),
        "empty_redraw": len(em_block),
        "terminal_tail": len(tail_block),
        "exploit_priority": "frontier",
        "explore_rate": a.explore_rate,
        "eval_holdout_sha256": holdout_sha,
        "hash_assert": "PASS",
        "frontier_intersect_eval_holdout": len(set(new_order) & eval_holdout),
        "permutation_of_old": sorted(new_order) == sorted(order),
        "nm_family_order": [f for f in sorted({fam(i) for i in nm_block},
                            key=lambda f: (-fam_yield.get(f, -1), f))],
    }
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
