# L1_RESTRAT_NOTE.md — near-miss-first frontier re-stratification (live hot-swap)

**When:** 2026-07-08, applied live at the CYCLE 2 gen-phase quiet window (orch pid in
`orch.pid`, never touched). **Lever:** L1 (KILL_AUTOPSY §3) — *best-of-k re-draw of the
near-misses* — promoted to the exploit head, with an informed family-yield re-stratification
of the coverage tail. Ledger-brain change + frontier permutation + manifest re-assert, all
through the KILL-D1 hash-asserted firewall.

## What the evidence said (measured, `attempts.jsonl` valid rows, post-rescue)
- 791 valid rows: 232→**236 keepers** (resolved), **479 near-misses** (non-empty FAILING patch
  = `unresolved`/`error`), 63 empty-patch, 13 env_unavailable, 0 no_prediction (all now
  infra_invalid). attempts_real 828.
- The officially-scorable VA head is exhausted; the recoverable value is re-attempting the 479
  near-misses at their **measured family yields** — django 0.42, conan 0.32, scikit 0.64,
  sympy 0.50, pydata 0.53, pydantic 0.17, python 0.18, iterative 0.19, modin 0.20 … down to
  getmoto 0.071, facebookresearch 0.077, and sphinx-doc **0.000** (skip-per-autopsy, kept last).

## The blocker this had to solve (why frontier order alone was insufficient)
`ledger.eligible_pools` sorted the exploit pool **COVERAGE-first** `(attempt_count, idx)` — so the
546 count≥1 near-misses sat *behind* 1660 count=0 fresh ids. The live next-batch draw was 50 fresh
`python`/`iterative` ids; the L1 near-misses would not have been re-drawn for ~33 batches. No
frontier ordering can reorder count=1 ahead of count=0 when attempt_count is the sort's primary key.

## The change (two coupled edits + one re-assert)
1. **`ledger.py` (gated, backward-compatible):** `eligible_pools` now honors
   `best_of_k.exploit_priority`. Default `"coverage"` = unchanged `(attempt_count, idx, iid)`.
   New `"frontier"` = `(idx, attempt_count, iid)` → the frontier ORDER is authoritative, so a
   near-miss-first `order` re-draws near-misses ahead of fresh coverage. **Only reorders the
   returned lists; the eligible SET is byte-identical**, so `cmd_state.remaining` /
   `DONE_EXHAUSTED` are unchanged. `cmd_record` + `cmd_state` textually untouched.
   Unit-tested (`rescue_20260708/test_ledger_l1.py`): backward-compat vs the pre-edit `.bak`,
   set-invariance, near-miss-first, record/state identity — **all pass**.
2. **`frontier.json` (permutation of the same 2451-id universe):** new order =
   `[near-miss by family-yield desc]` (479, sphinx-doc 0.0 last) + `[unattempted by family-yield
   desc]` (1660, getmoto/facebookresearch near-zero last) + `[empty-patch re-draw by yield]` (63)
   + `[terminal tail resolved/env]` (249, skipped by eligibility). `exploit_priority="frontier"`,
   `explore_rate=0.12` (the 88/12 convention). Built by `restratify_frontier_l1.py`.
3. **KILL-D1 hash-assert (re-asserted, not weakened):** id SET unchanged ⇒ eval-holdout
   disjointness preserved; recompute `sha256(inner5 ∪ tier0_20 ∪ tier1_100)` =
   `c56f473…8d168e` == pinned `.eval_holdout_sha256`; `frontier ∩ eval_holdout == 0`;
   manifest `kill_d1_check` re-stamped PASS (train set unchanged; reorder only).

## 88/12 note (honest)
The **explore pool is data-empty**: every family now has ≥1 resolved row, so `resolvable` = all 22
families and there is no zero-yield family left to "explore." The 88/12 *convention* is kept in
config (`explore_rate=0.12`); the coverage work lives in the exploit pool, family-yield-ordered.

## Verified live effect (installed files, pre-CYCLE-3)
`ledger.py nextbatch frontier.json attempts.jsonl 50` → **50/50 near-miss re-draws**
(attempt_count all ≥1), families {psf 1, scikit-learn 9, pydata 6, sympy 19, pytest 6, django 9}
— highest-yield near-miss families first. `state` verdict CONTINUE, remaining 2202 (unchanged).

## Race-safety
`frontier.json` is READ-ONLY for the orch (drawn only at `nextbatch`); it embeds no attempts rows,
so there is nothing to preserve — atomic `mv` into place during the gen-phase quiet window, before
the CYCLE 3 draw. `attempts.jsonl` NOT touched (row count 1300 monotonic across the swap).
`ledger.py`/manifest atomic-written; `ledger.py` change is a no-op until the frontier carries the
flag. Backups: `frontier.json.bak_pre_l1_restrat_*`, `ledger.py.bak_pre_l1_restrat_*`.

## Corrected trajectory to the 400 floor
479 near-misses × best-of-3 (2 re-draws each) at measured family yields ⇒ E[+100–200] keepers
(optimistic per-attempt-independent bound ≈ +203; realize ~½ ⇒ +100). 236 + ~100–150 clears the
**400 floor** off L1 alone; the mid-yield unattempted coverage (python/iterative/modin/dask, ~0.18)
is the cushion; near-zero families (getmoto/facebookresearch/sphinx) are drawn LAST and the 0.10/200
kill bar governs honestly if the tail genuinely underperforms.
