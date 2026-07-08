# KILL_AUTOPSY.md — swe_datagen_s1

**When:** kill bar fired `2026-07-07T13:31:08Z` (`DATAGEN_KILL.txt`), orchestrator dead by design.
**Scope:** CPU-only forensics, no server, no GPU. Evidence = `attempts.jsonl` (1250 rows),
`batches/*/logs_score.txt`, `batches/*/score/`, `CYCLE23_INFRA_REPAIR_NOTE.md`, ledger state.
**Kill record:** `keepers=218/1000 (floor 400), attempts_real=937, lifetime_yield=0.2327,
rolling_yield=0.09 (w=200), kill_bar=0.10, remaining_frontier=2137(bok).`

---

## 0. VERDICT — **MIXED**

> The **kill trigger is a SILENT-INFRA artifact**; the **condition underneath it is a GENUINE
> scorable-frontier wall.** Both are true and neither cancels the other.

1. **The bar fired on infra, not on the teacher.** The rolling-200 window that crossed 0.10 was
   3/4 filled by batches 12/13/14 — **150 rows, 100% `no_prediction`, ZERO of them ever scored**
   because the deferred SWE-Bench-Fork scorer crashed deterministically (`TypeError` at
   `utils.py:180`) before any test ran. **136/150 of those were real, non-empty patches.** Had
   those rows been stamped `infra_invalid` — exactly as the identical-fingerprint `batch_0007`
   was — the rolling window over the last 200 **actually-scored** attempts is **0.415**, i.e.
   **4.6× above the kill bar. The campaign would not have been killed.**

2. **But the reason batches 12/13/14 were 100% unscorable-gym is a real wall.** The
   Verified-adjacent (officially-scorable) head — 387 ids — is **100% consumed, 0 remaining**.
   The entire remaining frontier (1494 unique ids / 2137 best-of-k slots) is non-verified gym
   families that (a) crash the fork scorer as-is and (b) even once scored carry **low** family
   yields (blended **0.111**). So continuing *as-is* yields ~0 keepers; continuing *even with the
   scorer fixed* clears the 400 floor only with best-of-k depth and **cannot reach 1000** off the
   current frontier.

**One-line adjudication:** *We did not hit a teacher-quality collapse. We ran the officially-
scorable head to exhaustion, and the safety net that is supposed to keep scorer outages out of
the kill window (`infra_invalid`) covered gen-side failures but not score-side harness crashes —
so a scorer outage on the unscorable gym tail masqueraded as a yield collapse and tripped the bar.*

---

## 1. Decisive evidence (the trigger is infra)

### 1a. Per-batch verdicts, time-ordered (post-correction `attempts.jsonl`)

| batch (relaunch era) | n | resolved | no_pred | empty | unres | yield | stamped infra_invalid? |
|---|--:|--:|--:|--:|--:|--:|:--:|
| …T075402Z (08) | 50 | 22 | 5 | 6 | 16 | 0.44 | no |
| …T084508Z (09) | 50 | 13 | 6 | 5 | 26 | 0.26 | no |
| …T093444Z (10) | 50 | 21 | 6 | 2 | 21 | 0.42 | no |
| …T101832Z (11) | 50 | 18 | **15** | 4 | 13 | 0.36 | no |
| **…T110240Z (12)** | 50 | **0** | **50** | 0 | 0 | **0.00** | **NO ← should be yes** |
| **…T115018Z (13)** | 50 | **0** | **49** | 0 | 0 | **0.00** | **NO ← should be yes** |
| **…T124034Z (14)** | 50 | **0** | **48** | 0 | 0 | **0.00** | **NO ← should be yes** |

Rolling(200) at kill = batches 11+12+13+14 → `18 resolved / 200 = 0.09`. The three 0.00 batches
contributed **0 scored patches** to that denominator.

### 1b. The three collapse batches are healthy-gen + crashed-score, NOT teacher failure

| batch | gen driver logs | gen preflight | score signature | predictions (nonempty) |
|---|--:|---|---|---|
| 12 | 4 | booted, no timeout | **fork `TypeError` → NO report** | 50 (44 nonempty) |
| 13 | 4 | booted, no timeout | **fork `TypeError` → NO report** | 49 (47 nonempty) |
| 14 | 4 | booted, no timeout | **fork `TypeError` → NO report** | 48 (45 nonempty) |

`batches/batch_0013_*/logs_score.txt`:
```
[score:gym] fork harness over 49 ids
  File ".../swebench/harness/utils.py", line 180, in get_environment_yml_by_commit
    reqs_url = os.path.join(SWE_BENCH_URL_RAW, repo, commit, req_path)
TypeError: join() argument must be str, bytes, or os.PathLike object, not 'NoneType'
[merge] resolved=0 unresolved=0 empty=0 error=0 from 0 sub-report(s)
[score] WARN: fork harness produced NO report for 49 gym ids -> record as no_prediction
```
Deterministic, pre-network, pre-container. **No test was run on any of the 136 real patches.**

### 1c. Precedent proves the correct disposition was `infra_invalid`

`batch_0007` (relaunch T00:49) has the **identical fingerprint** — gen booted, 44/49 real patches,
same `TypeError`, 100% `no_prediction` — and it **was** stamped `infra_invalid` and excluded from the
window (300 rows total stamped: batches 2,3,7,8 + relaunch 4,5). The `infra_invalid` machinery
(`CYCLE23_INFRA_REPAIR_NOTE.md`) was wired to **gen failure (`gen_rc≠0`) only**; the score-side fork
crash leaves `gen_rc=0`, so batches 12/13/14 slipped the net during the autonomous stretch.

### 1d. Contamination of the ledger denominator

| metric | ledger (as fired) | honest (scored-only) |
|---|--:|--:|
| lifetime_yield | 218 / 950 = **0.2295** | 218 / 728 = **0.2995** |
| rolling(200) | 18 / 200 = **0.09** | 83 / 200 = **0.415** |

**209 "loose" `no_prediction`** rows (fork crashes never stamped `infra_invalid`) sit in the
ledger's "real attempt" denominator. They never ran a test. They should not price the teacher.

---

## 2. The genuine wall (what EXACTLY drained)

### 2a. Families exhausted — yes, the scorable head is gone

- **Verified-adjacent head: 387 ids → 387 consumed, 0 remaining.** 158 unique VA ids resolved.
  This head routes to the **working official** harness (no fork crash).
- **Remaining frontier = 1494 unique ids / 2137 best-of-k slots, ALL non-verified gym:**
  pandas-dev 726, getmoto 329, dask 134, python 123, iterative 92, facebookresearch 53,
  modin-project 22, bokeh 15. Every one of these crashes the fork scorer as-is.

### 2b. Best-of-k retries spent? — **NONE.** The depth lever is fully untapped.

`attempts-per-id (non-infra-invalid) = {1: 950}`. **Every one of the 950 real attempts was a first
attempt; zero instances got a 2nd.** The whole run was coverage-first. Best-of-k depth was never
purchased — so all near-miss re-attempt value below is still on the table.

### 2c. Remaining frontier priced by family yield — expected keepers per 100 attempts

`scored_y` = resolved / (resolved+unresolved+empty), i.e. yield **conditional on the scorer running**:

| family | remaining ids | scored_y (lifetime) |
|---|--:|--:|
| pandas-dev | 726 | 0.111 |
| getmoto | 329 | 0.071 |
| dask | 134 | 0.167 |
| python (mypy) | 123 | 0.183 |
| iterative (dvc) | 92 | 0.130 |
| facebookresearch | 53 | 0.077 |
| modin-project | 22 | 0.000 |
| bokeh | 15 | 0.100 |

- **Blended expected yield of the remaining frontier = 0.111 → ~11 keepers per 100 _scored_ attempts.**
- **As-is (scorer unfixed): ~0 keepers per 100 attempts** — all become `no_prediction`, and worse,
  each poisons the kill window again.
- Coverage pass over all 1494 unique ids (scorer fixed) → **E[+166] keepers → ~384 total** — does
  **not** clear the 400 floor.
- All 2137 best-of-k slots (scorer fixed) → **E[+238] → ~456 total** — clears floor, **far short of 1000.**

> **Honesty note reconciling the LEDGER's "python 0/63, iterative 0/62 → 0.000":** that 0.000 is the
> *collapse-slice* number — those specific tail ids all `no_prediction`'d (fork crash), never scored.
> The **families' scored-subset lifetime yields are 0.13–0.18**, not zero. The "these families are
> worthless" reading is itself an infra artifact. Their real (low-but-nonzero) yield only materializes
> **if the scorer is fixed** — which is precisely why lever L2 below is worth its cost.

---

## 3. Recoverable levers — ranked, priced, with dependencies

`E[+keepers]` uses `scored_y` as the per-additional-attempt resolve probability (optimistic on small
samples; near-miss ids already showed a real failing patch, so the model was close).

| # | lever | E[+keepers] | infra cost | dep | note |
|---|---|--:|---|---|---|
| **L1** | **VA near-miss best-of-k** — re-attempt the **~156** VA ids that already produced a real failing patch (django 91, sympy 19, astropy 15, matplotlib 11, scikit-learn 9, pydata 6, pytest 6, pylint 5, psf 1). | **+69** | **NONE** (routes to the *working* official harness) | none | **Do first.** Zero infra, pure GPU re-draw. Skip sphinx-doc (33 near-miss, scored_y 0.000). |
| **L2** | **Fix fork scorer, then gym near-miss best-of-k + gym coverage** — unlocks 293 gym near-miss ids (pydantic 58, python 55, iterative 48, conan 41 …) **and** the 1494-id remaining frontier. | **+43** (near-miss) then blended **+166** (coverage) / up to **+238** (bok) | **medium** (§4B code fix, no rebuild if gym images already pulled) | L2A code fix | The only lever that reaches/clears the 400 floor. |
| **L3** | **Re-stratify frontier by fresh family yields** — reorder remaining draws by measured `scored_y` (python/dask/conan first, modin/getmoto/facebookresearch/bokeh last or dropped). | reallocates the L2 budget for **~1.5–2× keepers/attempt** early | low (`restratify_frontier.py` exists) | L2 | Efficiency multiplier on L2, not new keepers. |
| **L4** | **New scorable sources** (all need Docker-image builds + leakage gate vs `.eval_holdout_sha256`): **SWE-smith** ~50k instances / ~128 repos, synthetic (LM-injected + PR-derived), variable quality; **SWE-bench-Extra (Nebius)** ~6.4k verified-executable; **SWE-rebench** ~21k collected / verified subset, leakage-controlled; **SWE-Gym remainder** (already the tail — largely drained). | large but **unpriced until built** | **high** (image builds = the exact infra cost blocking us) | image build + `leakage_audit.py` | Honest sizes; none is free. SWE-bench Verified-500 stays held out (eval). |
| **L5** | **Accept 218 and SFT now** — 218 verified-resolved trajectories, 19 families (django 76 ≈ 35%, sympy 29, conan 22, sklearn 16, python 13, pydantic 11, matplotlib 10, iterative 9, pydata 9, pytest 7, +9 more). | n/a | none | none | Usable if django-heavy imbalance is acceptable / down-weighted. The safe floor if L1–L2 are declined. |

**Recommended sequence:** L1 (free, +69 → ~287) → L2A fix + L2 (→ clears 400 floor, ~456) with L3
ordering → decide L4 vs L5 for the 1000 target. L1 alone nearly recovers the yield the kill "showed."

---

## 4. The silent-infra defect + fix (two independent parts)

### 4A. Kill-honesty defect (this is the bug that killed the run)
- **Defect:** `datagen_score.sh` records a **fork-harness abort** (`NO report for N gym ids`) as
  **`no_prediction`**, explicitly "re-drawable under best-of-k" — which counts each row as a **real
  teacher attempt** in the rolling kill window. `infra_invalid` is only set on **gen** failure
  (`gen_rc≠0`); a score-side crash leaves `gen_rc=0` and slips through. Result: 150 unscored rows
  fired a 0.10 yield bar (§1a–1d).
- **Fix (one policy):** when the scorer produces **no report due to a harness exception** (traceback
  present / "produced NO report") for ids that **have a real non-empty patch**, stamp those rows
  **`infra_invalid`** (re-drawable, excluded from window/lifetime/coverage) — *identical* treatment to
  gen-side failures. Wire it in `datagen_score.sh`/`datagen_orch.sh`: parse the score log for the
  fork-abort WARN and pass `ledger.py record --infra-invalid "fork_scorer_abort"` for the affected ids.
  *"The kill judges the teacher, not our infra"* must cover score-side, not just gen-side.

### 4B. Scorer-capability defect (fix this to actually recover gym yield — enables L2)
- **Root cause (verified):** the `None` in `os.path.join(SWE_BENCH_URL_RAW, repo, commit, req_path)`
  is **`commit`** — a **present-but-`None` `environment_setup_commit`**. `get_environment_yml()` does
  `instance["environment_setup_commit"] if "environment_setup_commit" in instance else base_commit`;
  because the key **is present** (valued `None`) it never falls back to `base_commit`. **All 50/50**
  gym instances in each collapse batch carried `environment_setup_commit: None` — injected upstream by
  `build_batch_dataset.py`, which merged SWE-Gym rows (no env commit) into the superset schema as a
  literal `None`. The fork harness then routes them down the conda `environment.yml` branch and
  `os.path.join(..., None, ...)` crashes **before any container/network work** → 0 tests run.
- **Fix status:** the correct fix — **drop a present-`None` `environment_setup_commit`** so the
  harness falls back to `base_commit` — is **already coded at `build_batch_dataset.py:104-108**
  (`if rec.get("environment_setup_commit") is None: rec.pop(...)`), but it **postdates the collapse
  batches** (they were built by the buggy version; batch 13's `dataset_gym.json` still shows 50/50
  present-`None`). So the source defect is fixed for *future* draws; what remains is (a) re-scoring
  the poisoned gym patches with a corrected dataset, and (b) the durable guard: on a fork-harness
  abort, **do not record `no_prediction`** — record `infra_invalid` (§4A) so a scorer gap never
  prices the teacher again. Optionally route gym ids to **gym-native scoring against pre-built gym
  images** (the image *is* the environment) instead of reconstructing env from `environment.yml`.

---

## 5. Bottom line
Not a genuine yield collapse — an **exhaustion of the officially-scorable head** plus a
**score-side infra outage that the kill-honesty net failed to catch**. The teacher was still
resolving at ~0.42 on scored work up to the moment the bar fired on unscored crashes. Cheapest
recovery is **L1 (VA near-miss best-of-k, +69, zero infra)**; the durable path to the floor is
**L2 (fix the fork scorer)**; the 1000 target requires **L4 (new imaged sources)** or a decision to
**L5 (accept 218 and SFT)**.

---

## 6. RESOLUTION — 2026-07-08 (amends §4A/§4B; false kill reversed)

Root cause in **§4B confirmed independently** and now **actually fixed** — plus the durable
kill-honesty guard from §4A, implemented one layer deeper.

- **§4B correction of record:** §4B states the scorer fix was *"already coded at
  `build_batch_dataset.py:104-108`."* It was **not** — the file at kill time still had the bug
  (`records.append({k: ex.get(k) for k in FIELDS})`, no `pop`), which is why **every** dual-source
  `dataset_gym.json` (all 14 relaunch batches) carried `environment_setup_commit` present-`None`
  (verified: present-None=n_gym_rows, missing=0, populated=0 in every one). The fix is now **truly
  implemented**: `build_batch_dataset.py` drops `environment_setup_commit` when `None`, restoring
  true absence → fork `get_environment_yml` falls back to `base_commit`. **Verified end-to-end** by
  re-materializing the collapse batches through the real (offline) build path: rebuilt gym datasets
  → **50/50 key MISSING, 0 `os.path.join` TypeErrors** (was 50/50 crashes); rebuilt verified
  datasets → **44/44 key populated** (official harness untouched).

- **§4A guard, implemented at the LEDGER layer (more robust than log-parsing):** rather than parse
  `datagen_score.sh`'s fork-abort WARN, `ledger.py cmd_record` now **auto-stamps a batch
  `infra_invalid` when its REAL rows are MAJORITY `no_prediction`** — because a missing prediction
  is ALWAYS a pipeline gap (the agent at worst submits an empty patch), never teacher signal. This
  catches ANY score-side break (fork crash, official crash, merge gap) regardless of `gen_rc`, and
  is independent of log text. Unit-tested: healthy 9/10-scored batch → not flagged; empty-report
  10/10 no_pred → whole batch flagged; explicit `--infra-invalid` still wins; 6/50 leak-shape →
  correctly NOT flagged (sub-majority); idempotent re-record.

- **Ledger correction applied** (backup
  `attempts.jsonl.bak_pre_gym_scoring_correction_20260708T065904Z`): **209** poisoned rows re-marked
  `infra_invalid` (`reason=fork_harness_no_report_env_setup_commit_none`) — every `swe_gym`
  `no_prediction` row in a dual-source batch whose fork sub-report is absent (evidence-derived, not
  hardcoded). Kept real: resolved/unresolved/empty_patch/error, verified rows (official harness
  unaffected), and env_unavailable (genuine pull failures, confirmed in `pull.jsonl`). Untouched:
  relaunch batches 0004/0005 (already `infra_invalid` — gen never booted). Poisoned ids are now
  re-drawable under best-of-k.

- **Corrected ledger** (target=1000 floor=400 kill-yield=0.10 kill-window=200): verdict
  **KILL_YIELD_COLLAPSE → CONTINUE**; keepers **218** (unchanged); attempts_real **937→728**;
  attempts_infra **300→509**; rolling_yield **0.09→0.415**; lifetime_yield **0.2327→0.2995**. This
  matches §1d's "honest (scored-only)" column exactly — the kill was **false**. `DATAGEN_KILL.txt`
  removed; `DATAGEN_STATUS.txt` documents the reversal.

- **Unchanged strategic caveat (§0.2, §2):** reversing the false kill does **not** manufacture a
  path to 1000. The VA (officially-scorable) head is still exhausted and the remaining gym frontier
  is still low-yield (§2c). The **fix** now lets those gym ids actually score (unlocking L2); the
  levers L1–L5 remain the forward decision for the next phase. **Next phase must run an acceptance
  gate that exercises the FORK SCORING path over a fresh `dataset_gym.json`** (the 02:09Z acceptance
  was gen-only) before relaunching the orchestrator.
