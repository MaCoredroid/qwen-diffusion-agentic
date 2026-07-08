#!/usr/bin/env python3
"""Apply the SWE-Gym rescue to the live ledger — RACE-SAFE.

(1) FLIP poisoned rows in attempts.jsonl from infra_invalid -> their TRUE rescue
    verdict (resolved|unresolved|empty_patch|error), preserving every row the live
    orch appended (re-read immediately before atomic rename; verify monotonicity).
    Only rows currently flagged infra_invalid with reason
    'fork_harness_no_report_env_setup_commit_none' AND scored by a rescue fork
    report are touched — nothing else.
(2) Report the MEASURED gym-tail family yields (family -> resolved/attempted) over
    exactly the episodes the rescue scored.

Keeper extraction is done separately (extract_keepers.py) so its own idempotent
instance_id dedup vs the live orch is preserved.

usage: apply_rescue.py [--apply]     (default: DRY-RUN, writes nothing)
"""
import json, sys, os, glob, time, collections
from pathlib import Path

HERE = Path("/home/mark/qwen_diffusion/runs/swe_datagen_s1")
RESCUE = HERE / "rescue_20260708"
ATTEMPTS = HERE / "attempts.jsonl"
POISON_REASON = "fork_harness_no_report_env_setup_commit_none"
APPLY = "--apply" in sys.argv

# Map original batch_id -> list of rescue fork reports to harvest verdicts from.
def rescue_reports():
    out = collections.defaultdict(list)
    for rep in glob.glob(str(RESCUE / "batch_*/score/parts/fork/*.json")):
        p = Path(rep)
        if "timing" in p.name:
            continue
        # batch dir name is 3 levels up: <batch>/score/parts/fork/<rep>
        batch_id = p.parents[3].name
        out[batch_id].append(p)
    return out

def load_report(p):
    d = json.loads(Path(p).read_text())
    return {
        "resolved": list(d.get("resolved_ids") or []),
        "unresolved": list(d.get("unresolved_ids") or []),
        "empty_patch": list(d.get("empty_patch_ids") or []),
        "error": list(d.get("error_ids") or []),
    }

def build_flip_map():
    """(instance_id, batch_id) -> verdict, plus per-family tallies."""
    flip = {}
    fam = collections.defaultdict(lambda: collections.Counter())
    per_batch = {}
    for batch_id, reps in sorted(rescue_reports().items()):
        agg = {"resolved": set(), "unresolved": set(), "empty_patch": set(), "error": set()}
        for rep in reps:
            r = load_report(rep)
            for v in agg:
                agg[v] |= set(r[v])
        # resolved wins over any other classification if a dup appears across reports
        seen = set()
        counts = collections.Counter()
        for v in ("resolved", "unresolved", "empty_patch", "error"):
            for iid in sorted(agg[v]):
                if iid in seen:
                    continue
                seen.add(iid)
                flip[(iid, batch_id)] = v
                fam[iid.split("__")[0]][v] += 1
                counts[v] += 1
        per_batch[batch_id] = dict(counts)
    return flip, fam, per_batch

def main():
    flip, fam, per_batch = build_flip_map()
    print("=== rescue reports harvested (original batch_id -> verdict counts) ===")
    for b, c in per_batch.items():
        print(f"  {b}: {c}")
    print(f"=== total scored episodes in flip map: {len(flip)} ===")

    # read attempts fresh
    lines = ATTEMPTS.read_text().splitlines()
    rows = [json.loads(l) for l in lines if l.strip()]
    n0 = len(rows)
    # which flip targets have an eligible poisoned row present?
    idx_by_key = {}
    for i, r in enumerate(rows):
        k = (r.get("instance_id"), r.get("batch_id"))
        if k in flip and r.get("infra_invalid") and r.get("infra_reason") == POISON_REASON:
            idx_by_key.setdefault(k, i)
    targets = list(idx_by_key)
    missing = [k for k in flip if k not in idx_by_key]
    print(f"=== eligible poisoned rows to flip: {len(targets)} / {len(flip)} "
          f"(missing/already-flipped/redrawn: {len(missing)}) ===")
    if missing:
        for k in missing[:20]:
            # explain why missing
            present = [r for r in rows if (r.get('instance_id'), r.get('batch_id')) == k]
            why = "no_row" if not present else f"verdict={present[0].get('verdict')},infra={present[0].get('infra_invalid')}"
            print(f"    MISS {k} -> {flip[k]}  ({why})")

    flip_verdict_counts = collections.Counter(flip[k] for k in targets)
    print(f"=== flip verdict breakdown: {dict(flip_verdict_counts)} ===")

    # family yields over scored episodes
    print("=== MEASURED gym-tail family yields (resolved/attempted) ===")
    fam_yield = {}
    for f in sorted(fam):
        c = fam[f]
        attempted = c["resolved"] + c["unresolved"] + c["empty_patch"] + c["error"]
        resolved = c["resolved"]
        fam_yield[f] = {"resolved": resolved, "attempted": attempted,
                        "unresolved": c["unresolved"], "empty_patch": c["empty_patch"],
                        "error": c["error"],
                        "yield": round(resolved / attempted, 3) if attempted else None}
        print(f"  {f:20s} resolved={resolved} attempted={attempted} "
              f"yield={fam_yield[f]['yield']}  (unres={c['unresolved']} empty={c['empty_patch']} err={c['error']})")
    (RESCUE / "family_yield.json").write_text(json.dumps(fam_yield, indent=1))

    if not APPLY:
        print("\n[DRY-RUN] no writes. re-run with --apply (in the quiet window after a fresh [status]).")
        return 0

    # ---- APPLY: race-safe flip ----
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    # RE-READ immediately before writing to capture any orch append since first read
    lines2 = ATTEMPTS.read_text().splitlines()
    rows2 = [json.loads(l) for l in lines2 if l.strip()]
    n1 = len(rows2)
    assert n1 >= n0, f"row count went backwards {n0}->{n1} (orch truncation?) ABORT"
    flipped = 0
    for r in rows2:
        k = (r.get("instance_id"), r.get("batch_id"))
        if k in flip and r.get("infra_invalid") and r.get("infra_reason") == POISON_REASON:
            r["verdict"] = flip[k]
            r.pop("infra_invalid", None)
            r.pop("infra_reason", None)
            r["rescued"] = True
            r["rescue_reason"] = "gym_fork_rescore_env_setup_commit_fix"
            r["rescored_at"] = ts
            flipped += 1
    tmp = ATTEMPTS.with_suffix(".jsonl.tmp_rescue")
    with tmp.open("w") as f:
        for r in rows2:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, ATTEMPTS)
    # verify
    lines3 = ATTEMPTS.read_text().splitlines()
    rows3 = [json.loads(l) for l in lines3 if l.strip()]
    n2 = len(rows3)
    final_flipped = sum(1 for r in rows3 if r.get("rescued"))
    print(f"[APPLY] flipped={flipped} rows; attempts count {n0}->{n1}->{n2} (monotonic={n2>=n0}); "
          f"rescued_rows_on_disk={final_flipped}")
    assert n2 == n1, f"post-write count {n2} != pre-write {n1}"
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
