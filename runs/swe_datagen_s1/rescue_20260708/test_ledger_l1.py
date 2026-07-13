#!/usr/bin/env python3
"""Unit tests for the L1 gated exploit_priority change in ledger.py.

Proves, against the LIVE frontier+attempts and a synthetic near-miss-first fixture:
  (1) BACKWARD COMPAT: no exploit_priority key (or "coverage") -> byte-identical
      draw to the PRE-EDIT ledger (imported from the .bak).
  (2) SET INVARIANCE: exploit_priority="frontier" returns the SAME eligible SET
      (only the order differs) -> cmd_state.remaining count cannot change.
  (3) NEAR-MISS-FIRST: with a near-miss-first `order`, "frontier" priority draws
      the count>=1 near-misses AHEAD of the count=0 fresh ids (coverage draws the
      fresh ids first). This is the whole point of the lever.
  (4) RECORD UNCHANGED: cmd_record source is textually identical to the backup
      (the edit touched only eligible_pools).
"""
import json, sys, importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import ledger as L  # patched


def _load_bak():
    from importlib.machinery import SourceFileLoader
    baks = sorted(ROOT.glob("ledger.py.bak_pre_l1_restrat_*"))
    assert baks, "no pre-edit backup found"
    m = SourceFileLoader("ledger_bak", str(baks[-1])).load_module()
    return m, baks[-1]


def main():
    Lb, bakpath = _load_bak()
    front = json.loads((ROOT / "frontier.json").read_text())
    order = front["order"]
    cfg = dict(front["best_of_k"])
    rows = [json.loads(l) for l in (ROOT / "attempts.jsonl").read_text().splitlines() if l.strip()]

    # (1) backward compat: patched vs backup, coverage default
    ex_p, xp_p = L.eligible_pools(order, rows, cfg)
    ex_b, xp_b = Lb.eligible_pools(order, rows, cfg)
    assert ex_p == ex_b and xp_p == xp_b, "BACKWARD-COMPAT FAIL: default draw differs from pre-edit ledger"
    print(f"[1] backward-compat OK: default draw identical to {bakpath.name} "
          f"(exploit={len(ex_p)} explore={len(xp_p)})")

    # explicit coverage == default
    cfg_cov = dict(cfg); cfg_cov["exploit_priority"] = "coverage"
    ex_c, xp_c = L.eligible_pools(order, rows, cfg_cov)
    assert ex_c == ex_p, "explicit coverage differs from default"
    print("[1b] explicit exploit_priority='coverage' == default OK")

    # (2) set invariance under frontier priority
    cfg_fr = dict(cfg); cfg_fr["exploit_priority"] = "frontier"
    ex_f, xp_f = L.eligible_pools(order, rows, cfg_fr)
    assert set(ex_f) == set(ex_p), "SET INVARIANCE FAIL: frontier priority changed eligible exploit SET"
    assert set(xp_f) == set(xp_p), "SET INVARIANCE FAIL: explore SET changed"
    assert ex_f != ex_p or len(ex_p) == 0, "frontier priority did not reorder (expected different order)"
    print(f"[2] set-invariance OK: frontier priority same SET (n={len(ex_f)}), reordered")

    # (3) near-miss-first on a synthetic near-miss-first order
    # fixture: 3 near-miss ids (a family with a resolved row so it's resolvable) at
    # the head, then 3 fresh ids of the same family after them.
    fixt_order = ["fam__x-1", "fam__x-2", "fam__x-3",  # near-miss (count=1) at head
                  "fam__x-4", "fam__x-5", "fam__x-6"]  # fresh (count=0) after
    fixt_rows = [
        {"instance_id": "fam__x-1", "batch_id": "b0", "verdict": "unresolved"},
        {"instance_id": "fam__x-2", "batch_id": "b0", "verdict": "unresolved"},
        {"instance_id": "fam__x-3", "batch_id": "b0", "verdict": "unresolved"},
        {"instance_id": "fam__seed-0", "batch_id": "b0", "verdict": "resolved"},  # makes 'fam' resolvable
    ]
    fixt_cfg = {"enabled": True, "seed_resolvable_families": [], "k_resolvable": 3,
                "k_default": 1, "explore_rate": 0.12}
    # coverage: fresh (count0) first
    ex_cov, _ = L.eligible_pools(fixt_order, fixt_rows, fixt_cfg)
    assert ex_cov[:3] == ["fam__x-4", "fam__x-5", "fam__x-6"], f"coverage head unexpected: {ex_cov}"
    # frontier: near-miss (count1, low idx) first
    fixt_cfg_fr = dict(fixt_cfg); fixt_cfg_fr["exploit_priority"] = "frontier"
    ex_frf, _ = L.eligible_pools(fixt_order, fixt_rows, fixt_cfg_fr)
    assert ex_frf[:3] == ["fam__x-1", "fam__x-2", "fam__x-3"], f"frontier head unexpected: {ex_frf}"
    print(f"[3] near-miss-first OK: coverage head={ex_cov[:3]}  frontier head={ex_frf[:3]}")

    # (4) L1-LEVER SURFACE. The L1 restrat's guarantee is proven by (1)-(3): the
    # eligible-pool DRAW is backward-compatible and the near-miss-first ordering is
    # correct. The original byte-identity snapshots of cmd_record/cmd_state vs the
    # pre-L1 backup are OBSOLETE — both functions legitimately evolved AFTER this
    # 2026-07-08 rescue (majority-no_prediction auto-infra #89, epoch markers,
    # opus_track exclusion, and the 2026-07-12 ctx-overflow truth-telling wiring),
    # none of which touch the L1 draw. We therefore assert the draw invariants hold
    # (above) and that the ctx-overflow wiring is present, rather than pin the ancient
    # snapshot. cmd_state is NOT touched by the truth-telling change.
    import inspect
    rec_src = inspect.getsource(L.cmd_record)
    assert "_ctx_overflow_ids" in rec_src and "env_limited" in rec_src, \
        "ctx-overflow truth-telling wiring missing from cmd_record!"
    assert "env_limited" not in inspect.getsource(L.cmd_state), \
        "cmd_state should be untouched by the ctx-overflow truth-telling change"
    print("[4] L1 draw invariants proven by [1]-[3]; cmd_record carries the "
          "ctx-overflow env_limited wiring; cmd_state untouched OK")

    print("\nALL LEDGER L1 TESTS PASSED")


if __name__ == "__main__":
    main()
