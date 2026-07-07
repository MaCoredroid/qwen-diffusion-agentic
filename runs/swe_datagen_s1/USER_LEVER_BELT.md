# USER LEVER BELT — ENACTED 2026-07-07 (USER greenlit)

> **STATUS: ENACTED.** The USER greenlit this belt-relax lever on **2026-07-07**.
> It is now LIVE — the enforced eval holdout is `inner5 ∪ tier0_20 ∪ tier1_100`
> (113 distinct ids, SHA-256-pinned), the `verified_500_tier2` ring is DROPPED from
> enforcement, and **387 SWE-bench_Verified-adjacent instances** (all 500 Verified
> test ids MINUS the 113 held eval ids) were added to the FRONT of the data-gen
> frontier (exploit head). Enactment record is at the bottom of this file
> (§ ENACTMENT 2026-07-07). The text below is the ORIGINAL pre-enactment proposal,
> kept verbatim for provenance.

---

# (original proposal) USER LEVER BELT — the remaining lever (USER decision; NOT enacted)

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

---

## ENACTMENT 2026-07-07 (USER greenlit)

**Decision.** The USER greenlit the belt lever on **2026-07-07**. Enacted live into
the running campaign (orchestrator PID 796779 never killed — same hot-swap pattern
as the f634b7b cycle-2 intervention: the loop reads `frontier.json` + `ledger` fresh
each cycle).

**What was enforced vs relaxed (the new leakage contract).**

| ring | n | role after enactment |
|------|--:|----------------------|
| `inner5` | 5 | **ENFORCED holdout** (inlined smoke) |
| `tier0_20` | 20 | **ENFORCED holdout** (Tier-0 eval ring) |
| `tier1_100` | 100 | **ENFORCED holdout** (Tier-1 eval ring; contains the W2 N=50 pool + the Tier1-C46 K-gate slice) |
| `verified_500_tier2` | 500 | **DROPPED from enforcement** — Verified ids outside every eval ring are now trainable |

The three enforced rings NEST/overlap (inner5 ⊂ tier0; tier0 ⊄ tier1), so the nominal
125 dedupes to **113 DISTINCT** held-out ids. KILL-D1 now hash-asserts the trainable
frontier is DISJOINT from those 113 (`sha256(sorted eval-holdout)` pinned to
`c56f473ad31e52bee0f85151562f4e2122e4815dfa3f1b776b15fe121e8d168e` in
`runs/swe_datagen_s1/.eval_holdout_sha256`; any ring-file drift trips the build).

**What was added.** `expand_frontier.py` prepended the **387 Verified-adjacent** ids
(= all 500 SWE-bench_Verified test ids − the 113 held eval ids) to the FRONT of
`frontier.json::order` (the exploit head), and added their 12 families
(astropy/django/matplotlib/mwaskom/pallets/psf/pydata/pylint-dev/pytest-dev/
scikit-learn/sphinx-doc/sympy) to `best_of_k.seed_resolvable_families`. Frontier
2064 → **2451**; manifest 2438 → **2825** train ids. All 387 official
`swebench/sweb.eval.x86_64.<slug_1776>` images confirmed to exist (387/387,
`verified_adjacent_image_check.json`).

**What it means (the caveat).** This shrinks the protected eval holdout from 625 → 113
never-trained instances and makes the SFT pool **repo/era-adjacent to
SWE-bench_Verified** — the population many public SWE eval sets are drawn from. The
firewall held is the **standard-practice** one: **NO evaluated instance ever trains**
(the 113-id eval rings — inner-5 smoke, Tier0-20, Tier1-100 including the N=50 pool
and the Tier1-C46 K-gate slice — stay hash-asserted out). It is a one-way trade: the
387 ids are now burned for eval. Promotion decisions
([[diffusion-promotion-discipline]]) must be read on the still-clean 113-id rings.

**Pipeline made dual-source** (so the Verified ids actually run end-to-end):
`build_batch_dataset.py` sources metadata from BOTH `SWE-Gym/SWE-Gym` and
`princeton-nlp/SWE-bench_Verified`; `pull_and_tag.sh` pulls the official
`swebench/…` image for Verified ids; `datagen_score.sh` routes Verified ids to the
OFFICIAL swebench-4.1.0 harness (the W2-proven path, matches the image provenance)
and SWE-Gym ids to the SWE-Bench-Fork harness, then MERGES both reports. All changes
are backward-compatible (a pre-belt batch with no `sources.json` scores exactly as
before — the backward-compat *single-source* path was verified live on the in-flight
cycle-6 batch).

**CAVEAT — the DUAL-SOURCE SCORING bug (found + fixed 2026-07-07, batch_0007).** The
belt's *dual-source* path was NOT actually exercised end-to-end by the cycle-6 check
(that batch was single-source, so it took the backward-compat branch). The FIRST
genuinely mixed batch — **batch_0007** (43 Verified + 6 SWE-Gym) — generated **49 real
patches** (44 non-empty; the AR server served for 33 min at 100 % GPU, gen rc=0) but
recorded **50/50 `no_prediction`, patch_bytes=0**. Root cause was purely in
`datagen_score.sh`: it handed the MERGED `all_predictions.jsonl` (both sources) to
*each* single-source harness, and swebench's `get_dataset_from_preds` validates that
**every** prediction id is in the (single-source) dataset **before** it applies the
`--instance_ids`/`-i` filter — so the gym harness rejected the 43 Verified ids and the
official harness rejected the 6 gym ids (`ValueError: Some prediction IDs not found in
dataset!`), both aborting before launching a single container → empty merged report →
the ledger scored every row `no_prediction`. **Fix:** `datagen_score.sh` now writes
per-source **filtered** prediction files (`pred_gym.jsonl` / `pred_ver.jsonl`) and
feeds each harness only its own source's rows (`prediction_ids ⊆ dataset_ids`
restored). Backward-compat preserved: no `sources.json` ⇒ `pred_gym == all_predictions`,
`pred_ver` empty. batch_0007 (and the orphaned partial batch_0008) were marked
`infra_invalid` in `attempts.jsonl` (excluded from yield/kill/coverage; ids
re-drawable) — the SCORING bug is our infra failure, never a teacher signal. A boot
race in `datagen_gen.sh::wait_ready` (a t≈0 `is-active` check mis-reading the
not-yet-registered scope as death) was hardened at the same time.

**MANDATORY GATE for any dual-source change (new standing rule).** Scratch/dry-run
validation MISSED this — the bug only appears when a *mixed* batch's real patches hit
the *real* single-source harnesses. Any future change to the dual-source pipeline must
pass a **both-sources LIVE gate**: ≥2 SWE-Gym + ≥2 Verified real episodes, each
producing a real patch through the FULL path (pull → gen → **score**), with a
non-empty merged report whose `_merged_from` lists BOTH sub-reports. The 2026-07-07
repair passed this gate (`runs/swe_datagen_s1/accept_dualsrc_*/`).

**Kill bar UNCHANGED** at 0.10 over 200 real attempts — the honest arbiter still
decides. Verified's expected 40–70 % yield (teacher resolved 19/50 on the Tier1-class
W2 pool) should clear the 400-keeper floor with margin.

**Backups:** `frontier_prebelt.json`, `data/swe_sft_pool/pool_manifest.json.bak_prebelt_*`,
`attempts.jsonl.bak_pre_dualsrc_scoring_correction_*`.
