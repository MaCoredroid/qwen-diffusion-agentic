#!/usr/bin/env python3
"""INTERVENTION re-stratifier — rebuild frontier ORDER around the DEMONSTRATED
per-family yield map, and bake the best-of-k config the patched ledger consumes.

WHY (cycle-1 evidence): full-frontier best-of-1 yields ~0.08 (4/50) — repo-uniform
round-robin spends 60%+ of attempts on families that have NEVER resolved
(pandas/moto/dask/modin/hydra/bokeh: 0-for-many), so the campaign cannot reach the
400-keeper floor. This reorders the frontier so DEMONSTRATED-RESOLVABLE families
(python/mypy, pydantic, conan, iterative/dvc) are attempted first, keeps a ~13%
exploration slice of the zero-yield families so the yield map keeps GROWING
(a family that starts resolving auto-promotes to best-of-k in the ledger), and
records the yield map + best-of-k knobs into frontier.json.

INPUTS (evidence for the map, read-only):
  * frontier.json                       — current order + all 2064 ids
  * attempts.jsonl                      — this campaign's own attempt ledger
  * ../stage0_swegym_probe/report.json  — the stage-0 20-instance probe

OUTPUT: a NEW frontier.json (same schema + a `best_of_k` block and `yield_map`),
written to the path given as argv[1] (use a scratch path to validate first, then
install). The OLD frontier is NOT mutated by this script; the caller backs it up.

family key == instance_id.split("__")[0] (unique per repo in this pool).
usage: restratify_frontier.py <out_frontier.json> [--explore-rate 0.13] [--k 3]
"""
from __future__ import annotations
import json, sys, argparse, time
from collections import defaultdict, OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
FRONTIER = HERE / "frontier.json"
ATTEMPTS = HERE / "attempts.jsonl"
PROBE = HERE.parent / "stage0_swegym_probe" / "report.json"

REAL_VERDICTS = {"resolved", "unresolved", "empty_patch", "error", "no_prediction"}


def fam(iid: str) -> str:
    return iid.split("__")[0]


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


def build_yield_map(order: list[str]) -> dict:
    """Combine the stage-0 probe + this campaign's attempts into a per-family
    (attempts, resolved, yield) map. Evidence only — used to CLASSIFY families
    (resolvable if resolved>0) and to rank the resolvable head."""
    stat = defaultdict(lambda: {"attempts": 0, "resolved": 0,
                                "resolved_ids": set(), "sources": set()})
    # stage-0 probe: 20 instances, resolved_ids in scoring
    if PROBE.exists():
        pr = json.loads(PROBE.read_text())
        resolved_ids = set(pr.get("scoring", {}).get("resolved_ids", []))
        for iid in pr.get("instances", []):
            s = stat[fam(iid)]
            s["attempts"] += 1
            s["sources"].add("probe")
            if iid in resolved_ids:
                s["resolved"] += 1
                s["resolved_ids"].add(iid)
    # this campaign's real attempts
    for r in _read_jsonl(ATTEMPTS):
        v = r.get("verdict")
        iid = r.get("instance_id")
        if not iid or v not in REAL_VERDICTS:
            continue
        s = stat[fam(iid)]
        s["attempts"] += 1
        s["sources"].add("attempts")
        if v == "resolved":
            s["resolved"] += 1
            s["resolved_ids"].add(iid)
    # ensure every family present in the order has an entry (even 0-attempt)
    for iid in order:
        _ = stat[fam(iid)]
    out = {}
    for f, s in stat.items():
        out[f] = {
            "attempts": s["attempts"],
            "resolved": s["resolved"],
            "yield": round(s["resolved"] / s["attempts"], 4) if s["attempts"] else None,
            "resolved_ids": sorted(s["resolved_ids"]),
            "sources": sorted(s["sources"]),
        }
    return out


def round_robin(fam_ids: dict[str, list[str]], fam_order: list[str]) -> list[str]:
    """Interleave ids across families in fam_order so every prefix is balanced."""
    idx = {f: 0 for f in fam_order}
    out = []
    remaining = sum(len(fam_ids[f]) for f in fam_order)
    while remaining:
        for f in fam_order:
            i = idx[f]
            if i < len(fam_ids[f]):
                out.append(fam_ids[f][i])
                idx[f] = i + 1
                remaining -= 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--explore-rate", type=float, default=0.13)
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()

    front = json.loads(FRONTIER.read_text())
    order = front["order"]
    ymap = build_yield_map(order)

    # ids per family, sorted (reproducible), from the current order
    fam_ids = defaultdict(list)
    for iid in order:
        fam_ids[fam(iid)].append(iid)
    for f in fam_ids:
        fam_ids[f].sort()

    # classify: resolvable == demonstrated >=1 resolve (probe OR attempts)
    resolvable = sorted([f for f, s in ymap.items() if s["resolved"] > 0],
                        key=lambda f: (-ymap[f]["yield"], f))  # highest yield first
    zero = sorted([f for f in fam_ids if f not in resolvable],
                  key=lambda f: (-len(fam_ids[f]), f))  # biggest pool first

    # HEAD: resolvable families round-robin (highest-yield-first tiebreak inside
    # each round via fam order). TAIL: zero families round-robin. The ledger draws
    # ~87% exploit (resolvable, best-of-k) + ~13% explore (fresh zero) per batch,
    # so intra-segment round-robin keeps each drawn slice repo-balanced.
    head = round_robin(fam_ids, resolvable)
    tail = round_robin(fam_ids, zero)
    new_order = head + tail
    assert sorted(new_order) == sorted(order), "re-stratify changed the id set!"
    assert len(new_order) == len(order), "re-stratify changed the id count!"

    by_repo = OrderedDict()
    for iid in new_order:
        r = front_repo_of(front, iid)
        by_repo[r] = by_repo.get(r, 0) + 1

    new_front = dict(front)  # preserve firewall/eligibility/provenance fields
    new_front["order"] = new_order
    new_front["by_repo"] = by_repo
    new_front["n_frontier"] = len(new_order)
    new_front["best_of_k"] = {
        "enabled": True,
        "seed_resolvable_families": resolvable,
        "k_resolvable": a.k,
        "k_default": 1,
        "explore_rate": a.explore_rate,
        "grows_from": "ledger recomputes resolvable = seed_resolvable_families "
                      "UNION {families with >=1 'resolved' row in attempts.jsonl}; "
                      "a zero-family that starts resolving auto-earns k_resolvable.",
        "yield_map": ymap,
        "resolvable_head_len": len(head),
        "explore_tail_len": len(tail),
        "provenance": {
            "restratified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "evidence": ["stage0_swegym_probe/report.json", "attempts.jsonl"],
            "note": "INTERVENTION cycle-2: cycle-1 yield 4/50=0.08 vs probe 0.25 "
                    "target; full-frontier @0.08-0.12 => max ~165-250 keepers < "
                    "400 floor. Re-stratify+best-of-3 to give the resolvable "
                    "families their shot; kill bar 0.10/200 unchanged (honest death).",
        },
    }
    Path(a.out).write_text(json.dumps(new_front, indent=1))
    # human summary to stderr
    print(f"[restratify] resolvable(head, k={a.k}): "
          + ", ".join(f"{f}({ymap[f]['resolved']}/{ymap[f]['attempts']}={ymap[f]['yield']}, "
                      f"n={len(fam_ids[f])})" for f in resolvable), file=sys.stderr)
    print(f"[restratify] zero(explore, k=1): "
          + ", ".join(f"{f}({ymap[f]['resolved']}/{ymap[f]['attempts']}, n={len(fam_ids[f])})"
                      for f in zero), file=sys.stderr)
    print(f"[restratify] head_len={len(head)} tail_len={len(tail)} "
          f"n={len(new_order)} explore_rate={a.explore_rate}", file=sys.stderr)
    print(json.dumps({"out": a.out, "resolvable": resolvable, "zero": zero,
                      "head_len": len(head), "tail_len": len(tail),
                      "n": len(new_order)}))
    return 0


def front_repo_of(front: dict, iid: str) -> str:
    """repo string for an id. Reuse the existing by_repo mapping by family prefix
    (owner__name-#### -> owner/name); falls back to prefix if unknown."""
    owner = iid.split("__")[0]
    name = iid.split("__")[1].rsplit("-", 1)[0]
    return f"{owner}/{name}"


if __name__ == "__main__":
    raise SystemExit(main())
