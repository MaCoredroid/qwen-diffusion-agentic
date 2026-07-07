#!/usr/bin/env python3
"""Stage-1 data-gen FRONTIER builder (CPU, one-time; idempotent).

Turns the 2,438-instance SWE-Gym SFT pool (data/swe_sft_pool/pool_manifest.json)
into an ORDERED attempt list the image-cycling orchestrator consumes batch by
batch. Three jobs, in order:

  1. KILL-D1 leakage firewall (HARD ASSERT). Re-verify the manifest's own
     kill_d1_check AND, wherever the held-out ring id lists are locally
     resolvable, re-derive `train_ids ∩ (verified_500 ∪ Tier0 ∪ Tier1) == ∅`.
     ANY nonzero overlap -> raise (do not build a frontier we could train a
     leaked model on). This is the §1.2 firewall enforced in code.

  2. Eligibility screen (best-effort, offline). Drop MONAI (absent from the
     SWE-Bench-Fork MAP_REPO_VERSION_TO_SPECS) and, when swebench is importable,
     drop rows whose version is not in the fork spec map or whose
     make_test_spec() raises. Prebuilt-image existence is NOT checked here
     (needs network); it is deferred to pull time — a pull miss is recorded by
     the orchestrator as `env_unavailable` (a non-yield skip), never a resolve
     denominator hit.

  3. Repo-stratified ordering. Interleave the eligible ids round-robin across
     repos (largest-repo-first tiebreak) so any prefix of the frontier is
     repo-balanced. The orchestrator prefers COVERAGE over repeats (task rule:
     distinct instances, stratify by repo), so batches drawn off the head of
     this order are automatically stratified.

Output: frontier.json {order:[ids], by_repo, excluded, eligibility} + a verbose
frontier_audit.json. Re-running is safe (pure function of the manifest); the
orchestrator's resume is driven by attempts.jsonl, NOT by mutating this file.

Run with the FORK venv so swebench + the spec map are importable:
  runs/stage0_swegym_probe/.venv-swegym/bin/python build_frontier.py
(falls back to a spec-map-less screen if swebench is absent — MONAI still dropped).

BELT-LEVER NOTE (2026-07-07): this builds the SWE-Gym BASE frontier only. The USER
belt-relax lever (drop `verified_500_tier2` from the holdout; add the 387
Verified-adjacent ids to the exploit head) is applied by `expand_frontier.py` as a
POST-step on top of this file's output (+ `restratify_frontier.py` for best_of_k).
firewall_assert here already enforces the RELAXED contract via the manifest: once
`verified_500_tier2` is retired to `manifest.relaxed_rings`, it no longer resolves
into `heldout_union`, so the HARD assert reduces to
`train_ids ∩ (inner5 ∪ tier0_20 ∪ tier1_100) == ∅`. The hash-assert on the eval
holdout lives in `expand_frontier.py`.
"""
from __future__ import annotations
import json, os, sys
from collections import defaultdict, OrderedDict
from pathlib import Path

REPO_ROOT = Path("/home/mark/qwen_diffusion")
MANIFEST = REPO_ROOT / "data/swe_sft_pool/pool_manifest.json"
HERE = Path(__file__).resolve().parent
OUT = HERE / "frontier.json"
AUDIT = HERE / "frontier_audit.json"

# Not in the SWE-Bench-Fork spec map -> unscorable by the official filter.
EXCLUDE_REPOS = {"Project-MONAI/MONAI"}

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def _load_ids(path_str: str) -> set[str] | None:
    """Best-effort load of a held-out ring id list from a local json file.
    Accepts a bare list, {instance_ids:[...]}, or a list of {instance_id:...}."""
    if not path_str:
        return None
    p = Path(path_str)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except Exception:
        return None
    if isinstance(d, dict) and "instance_ids" in d:
        return set(d["instance_ids"])
    if isinstance(d, list):
        if d and isinstance(d[0], str):
            return set(d)
        if d and isinstance(d[0], dict):
            return {r.get("instance_id") for r in d if r.get("instance_id")}
    return None


def firewall_assert(man: dict) -> dict:
    """HARD KILL-D1 re-assertion. Returns an audit dict; raises on any overlap."""
    train_ids = {r["instance_id"] for r in man["instances"]}
    rings = man.get("held_out_rings", {})
    checked = {}
    heldout_union: set[str] = set()

    # inner5 ids are inlined in the manifest -> always re-derivable.
    for name in ("inner5",):
        ids = set(rings.get(name, {}).get("ids", []))
        if ids:
            inter = train_ids & ids
            checked[name] = {"n": len(ids), "intersect": len(inter),
                             "overlap": sorted(inter)[:10]}
            heldout_union |= ids

    # tier0 / tier1 / verified_500 are referenced by source path -> load if present.
    for name in ("tier0_20", "tier1_100", "verified_500_tier2"):
        src = rings.get(name, {}).get("source", "")
        ids = _load_ids(src)
        if ids is not None:
            inter = train_ids & ids
            checked[name] = {"n": len(ids), "intersect": len(inter),
                             "overlap": sorted(inter)[:10], "source": src, "resolved": True}
            heldout_union |= ids
        else:
            checked[name] = {"source": src, "resolved": False,
                             "note": "id list not locally resolvable; relying on "
                                     "manifest kill_d1_check (recorded 0 overlap)"}

    manifest_kd1 = man.get("kill_d1_check", {})
    total_overlap = train_ids & heldout_union
    audit = {
        "train_ids_n": len(train_ids),
        "rechecked_rings": checked,
        "heldout_union_resolved_n": len(heldout_union),
        "recomputed_intersection_n": len(total_overlap),
        "recomputed_overlap_sample": sorted(total_overlap)[:20],
        "manifest_kill_d1_check": manifest_kd1,
    }
    # Two independent gates must both pass.
    if total_overlap:
        raise SystemExit(f"KILL-D1: {len(total_overlap)} train ids intersect held-out "
                         f"rings: {sorted(total_overlap)[:20]}")
    if manifest_kd1.get("result") != "PASS":
        raise SystemExit(f"KILL-D1: manifest kill_d1_check.result != PASS: {manifest_kd1}")
    audit["result"] = "PASS"
    return audit


def eligibility_screen(rows: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """Drop MONAI always; drop version-not-in-spec-map / make_test_spec failures
    when swebench is importable. Returns (kept_rows, dropped_audit, meta)."""
    specs = None
    make_test_spec = None
    rec = {}
    try:
        from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS as specs  # noqa
        from swebench.harness.test_spec import make_test_spec  # noqa
        swebench_ok = True
    except Exception as e:  # noqa: BLE001
        swebench_ok = False
        print(f"[frontier] swebench import unavailable ({type(e).__name__}); "
              f"spec-map/make_test_spec screen SKIPPED (MONAI still dropped, "
              f"version screen deferred to score-time harness).", file=sys.stderr)

    if swebench_ok:
        try:
            from datasets import load_dataset
            ds = load_dataset("SWE-Gym/SWE-Gym", split="train")
            rec = {ex["instance_id"]: dict(ex) for ex in ds}
        except Exception as e:  # noqa: BLE001
            print(f"[frontier] SWE-Gym load failed ({e}); version screen deferred.",
                  file=sys.stderr)
            swebench_ok = False

    # make_test_spec is a belt-and-suspenders check that is EXPENSIVE over 2,438
    # rows (it materializes eval scripts). It is OFF by default: the cheap
    # version-in-spec-map lookup is the real eligibility gate, and the official
    # harness re-runs make_test_spec at score time (any failure surfaces there as
    # a non-yield `error`, never a resolve). Enable with FRONTIER_MAKE_TEST_SPEC=1.
    do_mts = os.environ.get("FRONTIER_MAKE_TEST_SPEC", "0") == "1"

    kept, dropped = [], []
    for r in rows:
        iid, repo = r["instance_id"], r["repo"]
        reason = None
        if repo in EXCLUDE_REPOS:
            reason = "excluded_repo(not_in_fork_spec_map)"
        elif swebench_ok:
            ex = rec.get(iid)
            if ex is None:
                reason = "missing_from_swegym_dataset"
            else:
                ver = str(ex.get("version"))
                if repo not in specs or ver not in specs[repo]:
                    reason = f"version_{ver}_not_in_spec_map"
                elif do_mts:
                    try:
                        make_test_spec(ex)
                    except Exception as e:  # noqa: BLE001
                        reason = f"make_test_spec:{type(e).__name__}"
        if reason:
            dropped.append({"instance_id": iid, "repo": repo, "reason": reason})
        else:
            kept.append(r)
    meta = {"swebench_screen_applied": swebench_ok,
            "make_test_spec_applied": bool(swebench_ok and do_mts),
            "n_in": len(rows), "n_kept": len(kept), "n_dropped": len(dropped)}
    return kept, dropped, meta


def stratify_round_robin(rows: list[dict]) -> list[str]:
    """Interleave ids across repos so every prefix is repo-balanced. Repos with
    more instances go first within each round (deterministic); ids sorted within
    a repo for reproducibility."""
    by_repo: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        by_repo[r["repo"]].append(r["instance_id"])
    for k in by_repo:
        by_repo[k].sort()
    repos = sorted(by_repo, key=lambda k: (-len(by_repo[k]), k))
    order: list[str] = []
    idx = {k: 0 for k in repos}
    remaining = sum(len(v) for v in by_repo.values())
    while remaining:
        for repo in repos:
            i = idx[repo]
            if i < len(by_repo[repo]):
                order.append(by_repo[repo][i])
                idx[repo] = i + 1
                remaining -= 1
    return order


def main() -> int:
    man = json.loads(MANIFEST.read_text())
    fw = firewall_assert(man)
    print(f"[frontier] KILL-D1 firewall: {fw['result']} "
          f"(train={fw['train_ids_n']}, recomputed_overlap={fw['recomputed_intersection_n']})")

    rows = man["instances"]
    kept, dropped, meta = eligibility_screen(rows)
    order = stratify_round_robin(kept)

    by_repo = OrderedDict()
    for iid in order:
        repo = next(r["repo"] for r in kept if r["instance_id"] == iid)
        by_repo.setdefault(repo, 0)
        by_repo[repo] += 1

    frontier = {
        "artifact": str(OUT),
        "purpose": "Stage-1 data-gen ordered attempt frontier (repo-stratified, "
                   "KILL-D1-firewalled, eligibility-screened SWE-Gym SFT pool).",
        "source_manifest": str(MANIFEST),
        "manifest_built_at": man.get("built_at"),
        "kill_d1": fw["result"],
        "eligibility": meta,
        "excluded_repos": sorted(EXCLUDE_REPOS),
        "n_frontier": len(order),
        "by_repo": by_repo,
        "order": order,
    }
    OUT.write_text(json.dumps(frontier, indent=1))
    AUDIT.write_text(json.dumps({"firewall": fw, "eligibility": meta,
                                 "dropped": dropped}, indent=1))
    print(f"[frontier] wrote {OUT}: n={len(order)} across {len(by_repo)} repos "
          f"(dropped {len(dropped)}); by_repo={dict(by_repo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
