# Opus-4.8 datagen TRANCHE-2 — iteration-2 coverage-targeted batch (pre-registered plan)

USER-greenlit iteration-2, budget cap **$230-equivalent**. Teacher of record:
**Claude-Opus-4.8** (qwen-code via the OAuth OpenAI adapter, native `qwen3_xml`), the
same stronger-teacher rig that produced tranche-1 (yield 0.40, $4.78/keeper cached).
This doc is the pre-registered selection + economics + leakage-firewall record. It
covers **collection only**; the shape-corrected (episode-windowing) retrain is a
strictly downstream step and does not change what we collect here.

Batch: `opus_tranche2/batch/` (built by `build_batch_dataset.py`, dual-source).
Episode order / pull order / gen order all read `opus_tranche2/target_ids.txt`.

---

## 0. Headline
- **116 target ids**, all present in the offline datasets (0 missing), **78 SWE-Gym +
  38 Verified**, 4 balanced shards of 29.
- **Zero leakage**: 116 ∩ {113-id eval holdout} = **∅**; 116 ∩ {334 existing keepers}
  = **∅**. Holdout SHA-256 hash-asserted == pinned (KILL-D1 PASS).
- **Wave-1 images (first 30) = 30/30 present.** Disk after pulls: 2.6 T avail / 25 % used.
- Expected **45–60 keepers**; projected spend **~$165–215** under the $230 cap.

---

## 1. The binding constraint this tranche attacks
The C46 deficit-locus paired read (`runs/k_gate_c46/AR_PAIRED_READ.md`) adjudicated the
campaign blocker as an **SFT-capability ceiling (finding B)**, not the decode mode: even
AR-decoded, the SFT weights resolve only 7/48 on Tier1-C46, and **seven repo families
resolve 0 in BOTH arms** — astropy, psf/requests, pydata/xarray, pylint-dev/pylint,
pytest-dev/pytest, scikit-learn, sphinx-doc/sphinx. The ranked lever #1 there is
**"lift the SFT ceiling: data scale-up (Opus tranche-2)"**. So tranche-2 is coverage-
targeted, with a strategic slice aimed squarely at those 7 families.

## 2. Selection doctrine and the priority-(i) finding
The directive priority was: (i) never-attempted frontier ids **in the 7 C46 zero-coverage
families** (train-side only); (ii) never-attempted fresh coverage elsewhere, coverage-first;
(iii) a minority near-miss (best-of-k) tail.

**Priority (i) is EMPTY.** Every train-side frontier id in all 7 zero-coverage families
has already been *validly* attempted (they are small families — astropy 18, sphinx-doc 33,
scikit-learn 25, pytest-dev 14, pylint-dev 7, psf 6, pydata 17 frontier ids — and the
prior orchestrator + tranche-1 swept them). There are **0** never-attempted ids to draw
there. The C46 instances themselves are eval-holdout and excluded regardless.

Consequence: the only available lever into those families is **best-of-k retry** — re-run
the Opus (stronger) teacher on the *unresolved-but-committed* near-misses (the model
produced a real, non-empty patch that failed tests; a stronger teacher has genuine signal
to fix it). This becomes **stratum A** below and is elevated in priority precisely because
it is the proxy for the (empty) priority (i).

Two dead SWE-Gym images (`facebookresearch__hydra-1540`, `pandas-dev__pandas-47900` —
xingyaoww repo "does not exist" on Docker Hub, permanent env_unavailable) were swapped for
live same-family fresh-coverage backfills (`hydra-1661`, `pandas-48050`) so no episode slot
is wasted.

## 3. Strata and per-family counts (116 total)

### Stratum A — zero-coverage best-of-k (36) — the priority-(i) proxy
Unresolved-only, non-keeper near-misses in the 7 zero-coverage families. Allocation is
weighted toward the **thinnest current training coverage** (keeper counts in parens):

| family | current keepers | tranche-2 A slots | near-miss pool |
|---|--:|--:|--:|
| sphinx-doc | **0** | 15 | 33 |
| astropy | 3 | 8 | 15 |
| pylint-dev | 2 | 4 | 4 (all) |
| scikit-learn | 18 | 3 | 7 |
| pytest-dev | 10 | 3 | 3 (all) |
| pydata/xarray | 10 | 2 | 2 (all) |
| psf/requests | 5 | 1 | 1 (all) |

sphinx-doc has **zero** keepers in the entire 334-keeper pool AND resolves 0 on C46 — the
single highest-value target; it gets the largest slice.

### Stratum B — fresh coverage elsewhere (70) — coverage-first, repo-balanced
Never-validly-attempted frontier ids, drawn by **repo-balanced round-robin** (the project's
`round_robin_by_repo`) over the 1,486-id never-attempted pool, to maximize coverage breadth
(the 188af68 lesson: coverage-first, NOT frontier-failure-first). Result is 8 productive
repos ~evenly:

`pandas-dev 9, getmoto 9, python 9, dask 9, iterative 9, modin-project 9, facebookresearch 8, bokeh 8`.

(Note: raw frontier-index order would have front-loaded 62 consecutive modin ids; the
round-robin is the faithful coverage-first reading and de-risks single-family yield collapse.)

### Stratum C — minority near-miss tail (10) — best-of-k on high-yield families
Unresolved-only, non-keeper near-misses in the most productive families (2 each):
`python 2, iterative 2, pydantic 2, django 2, conan-io 2`. Best-of-k where a second Opus
attempt on a committed-but-failed patch is efficient.

Episode order (`target_ids.txt`) is a **proportional interleave** of the three strata, so any
prefix/wave is representative: wave-1 (first 30) = 18 B + 9 A (sphinx-led) + 3 C.

## 4. Economics (pre-registered)
- **Cap:** $230-equivalent, accounted per `usage_adapter.jsonl` (same accounting as tranche-1).
- **Tranche-1 measured (Opus-4.8):** $3.52–4.78/keeper at 93–94 % cache-read; pooled yield
  **0.40** (fresh-coverage 0.60 / fresh-failure 0.32).
- **Per-episode cost** ≈ yield × $/keeper ≈ 0.40 × ~$4.15 ≈ **$1.66/episode**.
- **116 episodes → ~$165–215** projected spend (buffer under $230; stratum-A hard instances
  cost the same per episode but yield fewer keepers, which the budget — priced per episode —
  absorbs).
- **Expected keepers: 45–60** (strata weighting: 70 fresh @~0.6 dominate; 46 near-miss
  @~0.3, likely lower for the hardest zero-cov families).
- **Stop rule:** budget exhausted **OR** production keeper pool ≥ 400 (currently **334**;
  +66 hits the floor — so on a high-yield run the pool-floor could bind before budget, but
  budget is the more likely binding stop at the expected 45–60 yield).

## 5. Leakage firewall — evidence (KILL-D1)
- Enforced eval holdout = inner5 ∪ tier0_20 ∪ tier1_100 = **113 distinct ids**.
- `sha256(sorted(holdout))` = `c56f473ad31e52bee0f85151562f4e2122e4815dfa3f1b776b15fe121e8d168e`
  **==** pinned `.eval_holdout_sha256` → **hash-assert PASS** (recomputed from the live ring
  source files; any drift trips the assert).
- **116 targets ∩ 113 holdout = ∅** (zero-overlap proof; `holdout_clean: true` in
  `targets.json`).
- **116 targets ∩ 334 existing keeper instance_ids = ∅** (dedup; `overlap_with_keepers: 0`).
- Stratum B ids verified **never validly attempted** (attempts.jsonl, infra_invalid rows
  excluded); strata A/C are attempted-unresolved by construction (best-of-k).
- **Collision:** the live coverage-first orchestrator (pid 2888673) is **dead** (GPU handed
  to SFT training on 2026-07-09), so the never-attempted fresh-coverage picks cannot collide
  with a live production draw.

## 6. Runtime / build wiring
- `build_batch_dataset.py` routes by dataset membership → 78 gym (xingyaoww images, fork
  scorer) + 38 verified (official `swebench/sweb.eval.x86_64.<slug_1776>` images, official
  scorer). 38 verified rows carry `environment_setup_commit`; 78 gym rows correctly drop it
  (the present-None fork-harness fix).
- `pull_wave.sh` is dual-source (per-id source from `batch/sources.json`, ≤3 concurrent,
  disk-floor guarded, wave-bounded) — the tranche-1 pull_wave was pure-gym/hardcoded, this
  one must route both sources to the same driver key.
- `gen_next.sh` is the tranche-1 gen driver retargeted to `opus_tranche2` (bounded
  foreground chunks via the persistent Opus adapter on :30050). **Running episodes is the
  downstream RUN step, not this build.**

## 7. Downstream note (does NOT change this collection)
The shape-corrected retrain (episode **windowing** instead of front-truncation +
serve-exact retokenize) is a separate downstream task. It changes how keepers are packed
into SFT spans, not which instances we collect — so this tranche's id selection is
independent of it.
