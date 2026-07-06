# USER LEVER BELT — the remaining lever (USER decision; NOT enacted)

Status at time of writing: the cycle-2 INTERVENTION (re-stratify + best-of-3 +
distinct per-cycle seeds) is LIVE and the campaign is running under it. This file
documents the ONE remaining, higher-yield lever that is **deliberately left for the
USER to decide**, because it trades away eval-holdout rigor. **Do not enact without
an explicit USER go.**

---

## The lever: relax the leakage belt to a Tier0 ∪ Tier1 holdout

### What the belt currently is
`build_frontier.py::firewall_assert` (the §1.2 KILL-D1 firewall) HARD-asserts that
the training / data-gen pool is disjoint from **all four** held-out rings recorded
in `data/swe_sft_pool/pool_manifest.json::held_out_rings`:

| ring                 |   n | role                                            |
|----------------------|----:|-------------------------------------------------|
| `inner5`             |   5 | inlined smoke holdout                           |
| `tier0_20`           |  20 | Tier-0 verified eval ring                       |
| `tier1_100`          | 100 | Tier-1 verified eval ring                       |
| `verified_500_tier2` | 500 | **SWE-Bench-Verified-adjacent ("Tier-2")**      |

Manifest `kill_d1_check` = PASS with `intersect_verified_500 = 0` and
`intersect_tier0_union_tier1 = 0`. So today the pool holds out **625 instances**,
including the 500-instance Verified-adjacent Tier-2 ring.

### The relaxation
Hold out **only `inner5 ∪ tier0_20 ∪ tier1_100`** (125 instances) and **return the
`verified_500_tier2` ring to the trainable frontier**. Those ~500 instances are
SWE-Bench-**Verified**-adjacent — curated to be genuinely solvable — so their
data-gen yield is expected to be **~40–70 %** (vs the SWE-Gym-wide 0.08–0.15 we
measure here). At 40–70 %, ~500 instances alone produce **~200–350 keepers**, which
clears the 400 keeper floor with margin even before the re-stratified SWE-Gym
resolvable families are counted.

### How to enact (when/if the USER says go)
1. In `data/swe_sft_pool/pool_manifest.json` (or a firewall override): drop
   `verified_500_tier2` from the enforced holdout union, keeping
   `inner5 ∪ tier0_20 ∪ tier1_100`. Re-run `build_frontier.py` so the firewall
   re-asserts the NEW (smaller) holdout and the 500 ids flow into the eligible pool.
2. Re-run `restratify_frontier.py` so the new instances land in the frontier with a
   yield map (they will start in the "explore" tail and auto-promote to best-of-k as
   they resolve — see `frontier.json::best_of_k.grows_from`).
3. The orchestrator hot-swaps at the next cycle boundary (frontier + ledger are read
   fresh each cycle); `attempts.jsonl` / keepers / resume state are preserved.

### Why it is a USER decision, not an automatic lever
This is **not** a tuning knob — it changes the **eval-holdout contract**. Returning
the Tier-2 Verified-adjacent ring to training:
- shrinks the protected eval holdout from 625 → 125 instances, reducing the
  never-trained surface the diffusion-vs-AR promotion decisions are measured on
  (see `[[diffusion-promotion-discipline]]`, `[[qwen-diffusion-experiment]]`);
- makes the SFT pool Verified-adjacent, which is exactly the population many public
  SWE eval sets are drawn from — a leakage-adjacency the USER may or may not accept
  for THIS SFT stage;
- is a one-way trade (once trained on those ids, they are burned for eval).

The intervention brief pre-authorized re-stratify + best-of-k as design levers but
explicitly reserved THIS one for the USER. It is documented here and left unenacted.

---

## Yield accounting: why this lever may be needed

- Cycle-1 (best-of-1, repo-uniform): **4/50 = 0.08**. Full 2064-frontier @0.08–0.12
  ⇒ **~165–250 keepers max < 400 floor.**
- Post-intervention projection (re-stratify to resolvable families + best-of-3),
  using the tiny-sample per-attempt family yields (python 0.43, pydantic 0.29,
  conan 0.14, dvc 0.14) as per-seed p over the 636 remaining resolvable instances:
  **~360 keepers** (P(≥1 resolve in 3) = 0.81 / 0.64 / 0.36 / 0.36 respectively),
  plus whatever the 12 % exploration slice promotes. **Marginal vs the 400 floor,
  on 7-attempt-per-family evidence (wide error bars).**
- The rolling-yield kill (0.10 over 200, **unchanged**) is the honest arbiter: if
  the re-stratified regime cannot hold 0.10, the campaign SHOULD die — and THIS
  belt-relax lever (expected 40–70 %) is the high-confidence path to the floor that
  the USER can then choose to pull.
