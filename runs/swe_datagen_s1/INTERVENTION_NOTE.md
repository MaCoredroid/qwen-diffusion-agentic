# INTERVENTION NOTE — cycle-2 hot-patch (re-stratify + best-of-3 + distinct seeds)

**When:** mid-cycle-2, orchestrator PID 262097 ALIVE (never killed).
**Applied at:** the cycle boundary — all levers land at cycle-3 via files the loop
reads FRESH each cycle. No relaunch; `attempts.jsonl` / keepers / resume untouched.

## Why (evidence)
- Cycle-1 yield **4/50 = 0.08** vs the stage-0 probe's 0.15–0.25 target.
- Pool composition was the problem, not scoring: **0 scoring errors**, 38/50 were
  real-patch-but-unresolved (genuine difficulty). Per-family (probe ∪ cycle-1):

  | family (repo)            | attempts | resolved | yield | pool n | class      |
  |--------------------------|---------:|---------:|------:|-------:|------------|
  | python / mypy            | 7        | 3        | 0.43  | 257    | RESOLVABLE |
  | pydantic / pydantic      | 7        | 2        | 0.29  | 83     | RESOLVABLE |
  | iterative / dvc          | 7        | 1        | 0.14  | 225    | RESOLVABLE |
  | conan-io / conan         | 7        | 1        | 0.14  | 75     | RESOLVABLE |
  | pandas-dev / pandas      | 7        | 0        | 0.00  | 737    | zero       |
  | getmoto / moto           | 7        | 0        | 0.00  | 343    | zero       |
  | dask / dask              | 7        | 0        | 0.00  | 145    | zero       |
  | modin-project / modin    | 7        | 0        | 0.00  | 107    | zero       |
  | facebookresearch / hydra | 7        | 0        | 0.00  | 66     | zero       |
  | bokeh / bokeh            | 7        | 0        | 0.00  | 26     | zero       |

- **The math:** repo-uniform round-robin spends ~60 % of attempts on zero-yield
  families. Full 2064-frontier @0.08–0.12 ⇒ **~165–250 keepers < the 400 floor.**
- Seed variance is real and exploitable: dvc-10218 and pydantic-4911 **resolved
  under the probe seed but failed under the cycle-1 seed** → distinct seeds convert
  different instances → best-of-k is well-motivated.

## What changed (pre-authorized design levers)

1. **RE-STRATIFY** — `restratify_frontier.py` rebuilt `frontier.json::order`:
   demonstrated-resolvable families first (round-robin, highest-yield-first:
   python→pydantic→conan-io→iterative, head_len=640), zero-yield families in the
   tail (round-robin, biggest-pool-first, tail_len=1424). Id set unchanged (2064).
   Backup: `frontier_prestrat.json`.

2. **BEST-OF-K** — `ledger.py` now reads `frontier.json::best_of_k` and draws each
   batch as **~88 % exploit** (resolvable families, up to **k=3** distinct-seed
   attempts/instance, sorted `(attempt_count, index)` = coverage-before-depth,
   **keep-any-resolve**) + **~12 % explore** (fresh zero-yield instances, k=1) so the
   yield map keeps growing. `record` is now idempotent per **(instance_id, batch_id)**
   (was per instance_id) so an instance can be attempted once per cycle. **Each
   attempt is one attempts.jsonl row → counts in the rolling-yield denominator.**
   The resolvable set **GROWS**: `seed_resolvable_families ∪ {families with ≥1
   resolved row}` — a zero family that starts resolving auto-earns best-of-k.
   Backup: `ledger_prestrat.py`. Falls back to original best-of-1 if `best_of_k`
   absent (partial-install safe).

3. **DISTINCT SEEDS** — `datagen_gen.sh` now feeds each shard's per-cycle
   `base_seed` (`1234 + cycle·1e6 + k·1e5`, already written by
   `build_batch_dataset.py` but previously ignored) into the proxy's
   `LUMO_PROXY_FORCE_SEED`. A re-attempt in a later cycle now gets a
   **guaranteed-distinct** seed ⇒ a distinct rollout (was pinned at 1234 for all
   cycles). Backup: `datagen_gen_prestrat.sh`. Installed by atomic rename so the
   in-flight cycle-2 gen (old inode) finished cleanly; the fix is live from cycle-3.

4. **KILL LOGIC** — bar **unchanged at 0.10 over 200 real attempts**. best-of-k does
   NOT depress per-attempt yield (each seed is iid ~p), so the bar stays honest:
   `state`'s `remaining`/DONE_EXHAUSTED now tracks the SAME best-of-k eligibility
   nextbatch uses (no drift). **If the re-stratified regime cannot hold 0.10, the
   campaign SHOULD die honestly.**

5. **USER LEVER** — `USER_LEVER_BELT.md` documents the remaining, higher-yield lever
   (relax the leakage belt to a Tier0 ∪ Tier1 holdout, returning the 500-instance
   Verified-adjacent Tier-2 ring, expected 40–70 % yield). **USER decision; not
   enacted.**

## Verification (read-only, before exit)
- `ledger.py` behavior suite: 9/9 groups PASS (best-of-k budget, coverage-before-
  depth, per-(iid,batch) idempotent record, exhaustion, DONE_TARGET, kill honesty,
  growing map, backward-compat).
- LIVE `nextbatch` (cycle-3 preview): 50 distinct = 44 resolvable (11 each) + 6
  zero-family (12 %); **no resolved id re-drawn**.
- LIVE `state`: CONTINUE, keepers 4, best_of_k=True, remaining 2030
  (exploit_eligible 636 + explore_eligible 1394 — correctly re-adds the 16 unresolved
  resolvable instances for best-of-k). Probe used a scratch status file; the live
  `DATAGEN_STATUS.txt` was not touched.

## Projection
Re-stratify + best-of-3 over the 636 remaining resolvable instances ⇒ **~360
keepers** on tiny-sample yields (wide error bars) + exploration upside — **marginal
vs the 400 floor**. The honest kill decides; the belt-relax lever is the
high-confidence path to the floor if the USER pulls it.
