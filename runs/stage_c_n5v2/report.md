# Stage-C N=5 v2 — CLEAN 3-arm, aligned runtime + OFFICIAL docker scoring

**Date:** 2026-07-05 (scored 2026-07-06T01:16Z). **Task:** #54/#66. **Supersedes** the DEPRECATED
`runs/stage_c_n5/` numbers (bare-checkout runtime confound; see `RUNTIME_ALIGNMENT_DIRECTIVE.md`).
**Runtime:** every episode ran *inside the official per-instance swebench docker image*
(`runner_metadata.json` `runtime=container`, `image=swebench/sweb.eval.x86_64.<inst>`), so the agent
could import the package and run the instance test command in-episode (acceptance gate 5/5, task #64).
**Scoring:** OFFICIAL `swebench.harness.run_evaluation` docker harness, pipeline `score rc=0`
(`logs/pipeline.log`); resolve verdicts are the official `scoring/*.json` reports, NOT a mock.

This is **the first real SWE-bench-Verified resolve table in the whole project.** Everything below is
verified against primary artifacts (official verdict JSON + per-instance `report.json` +
`usage.jsonl` exit-proof capture + applied `patch.diff` + official `test_output.txt`), not transcribed
from a rollup.

---

## 1. VERIFIED TABLE

### RESOLVE@1 (official docker)

| instance | stock-AR | merged-AR (RL-v2 as AR) | diffusion (same RL-v2 weights) |
|---|---|---|---|
| django-11119 | **RESOLVED** | **RESOLVED** | no-edit (loop-halt) |
| django-12754 | no-edit (empty) | edit, unresolved | edit, unresolved |
| django-13741 | **RESOLVED** | **RESOLVED** | edit, unresolved (loop-halt) |
| pytest-8399  | **RESOLVED** | edit, unresolved | **RESOLVED** |
| sympy-13757  | **RESOLVED** | no-edit (loop-halt) | no-edit (empty) |
| **resolve@1** | **4/5** | **2/5** | **1/5** |

Official verdict files (schema_version 2): `scoring/n5v2-stock-ar.n5v2_ar.json` resolved_ids = 4;
`scoring/n5v2-merged-ar.n5v2_mergedar.json` resolved_ids = [11119, 13741];
`scoring/n5v2-diffusion.n5v2_diffusion.json` resolved_ids = [8399]. All three confirm the table.

### turns | tokens | wall | exit  (exit-proof capture)

`turns`/`tokens` come from the qwen CLI stats when it exits cleanly; on turn-limit (exit 53) the CLI
writes nothing, so they are reconstructed from the proxy `usage.jsonl` bucketed into each instance's
`[started_at, ended_at]` window (`tok_src=proxy` on those rows). Loop-halt (exit 1) still emits qwen
stats.

**stock-AR** — resolve 0.8 / edit 0.8 / turns~44.8 / wall~91.1s / tok~836,673 / exits {ok:3, turn-limit:2} / loops 0

| inst | turns | tot_tok | wall | exit | verdict |
|---|--:|--:|--:|---|---|
| django-11119 | 43 | 867,976 | 103s | ok | RESOLVED |
| django-12754 | 52 | 969,570 | 98s | turn-limit | empty |
| django-13741 | 33 | 562,180 | 56s | ok | RESOLVED |
| pytest-8399  | 44 | 840,641 | 70s | ok | RESOLVED |
| sympy-13757  | 52 | 942,998 | 130s | turn-limit | RESOLVED (at the cap) |

**merged-AR** — resolve 0.4 / edit 0.8 / turns~32.2 / wall~59.2s / tok~520,270 / exits {ok:3, turn-limit:1, loop-halt:1} / loops 1

| inst | turns | tot_tok | wall | exit | verdict |
|---|--:|--:|--:|---|---|
| django-11119 | 34 | 580,230 | 65s | ok | RESOLVED |
| django-12754 | 51 | 898,207 | 107s | turn-limit | edit, unresolved |
| django-13741 | 16 | 199,387 | 18s | ok | RESOLVED |
| pytest-8399  | 39 | 693,879 | 85s | ok | edit, unresolved |
| sympy-13757  | 21 | 229,646 | 21s | **loop-halt** | empty |

**diffusion** — resolve 0.2 / edit 0.6 / turns~45.4 / wall~135.0s / tok~785,200 / exits {loop-halt:2, turn-limit:3} / loops 2

| inst | turns | tot_tok | wall | exit | verdict |
|---|--:|--:|--:|---|---|
| django-11119 | 29 | 378,817 | 55s | **loop-halt** | empty |
| django-12754 | 51 | 990,231 | 188s | turn-limit | edit, unresolved |
| django-13741 | 45 | 846,475 | 99s | **loop-halt** | edit, unresolved |
| pytest-8399  | 51 | 786,105 | 151s | turn-limit | RESOLVED |
| sympy-13757  | 51 | 924,374 | 181s | turn-limit | empty |

Note: diffusion has **zero clean exit-0** episodes (all 5 hit loop-halt or turn-limit); its one resolve
(pytest-8399) landed the correct patch and then ran to the 50-turn cap without a clean terminating turn.

### Spot-check: one resolved patch per arm is REAL (applied diff + official test pass)

| arm / instance | patch | applied? | FAIL_TO_PASS (official) | test_output |
|---|---|---|---|---|
| stock-AR / pytest-8399 | 549 B — `unittest_...fixture` -> `_unittest_...fixture` | yes | `test_fixtures_setup_setUpClass_issue8394` OK | 60 passed, 30 skipped |
| merged-AR / django-13741 | 465 B — `kwargs.setdefault("disabled", True)` | yes | `test_readonly_field_has_changed` OK | all PASS_TO_PASS green |
| diffusion / pytest-8399 | 540 B — fixture name -> `_{setup_name}_fixture_` | yes | `test_fixtures_setup_setUpClass_issue8394` OK | 60 passed, 30 skipped |

All three per-instance `report.json`: `patch_successfully_applied=true`, `resolved=true`, zero
FAIL_TO_PASS/PASS_TO_PASS failures. The two pytest patches are *different* real fixes to the same bug
(both make the xunit fixture name unique-per-class), independently resolving — not a copied patch.

---

## 2. ATTRIBUTION READING

Clean 2-way decomposition (all three arms share the aligned runtime + official scoring; only weights
and paradigm move):

- **Weights effect: stock-AR 4/5 -> merged-AR 2/5 = -2.** Swapping the stock Qwen3.5-9B for the merged
  **RL-v2 weights, both run as AR, same paradigm**, *loses* two resolves. The RL-v2 weights are a
  **worse AR payload for SWE than stock** — this is the dominant effect in the table.
- **Paradigm effect: merged-AR 2/5 -> diffusion 1/5 = -1.** Running the *same* RL-v2 weights as
  block-diffusion instead of AR loses one more. The paradigm adds a cost on top of the weights cost,
  but it is the smaller of the two moves.
- **Looping is substantially WEIGHTS-driven, not a pure diffusion artifact.** A loop-halt appears in
  **merged-AR** (sympy-13757, exit 1, `consecutive_identical_tool_calls`, empty patch) — i.e. the
  RL-v2 weights loop *even in AR mode*. Diffusion has two loop-halts (11119, 13741) and never a clean
  exit, so the paradigm *compounds* the looping, but the root cause is visible in AR. Stock-AR loops 0
  times. Loop provenance verified in `qwen_trace.json`: "Loop detection halted the run
  (consecutive_identical_tool_calls...)".
- **Old "env-artifact" contrast RETIRED.** The prior N=5 (`stage_c_n5/`) ran the agent in a bare
  checkout, so in-episode test/import failures were an arm-invariant confound and looping could be
  blamed on the broken env. With the runtime aligned (episode-in-official-container, imports + test
  cmd work), **looping persists and is now attributable to weights (+ paradigm), not the env.** The
  env-artifact reading is dead.

---

## 3. BINOMIAL HONESTY (n=5 — do not over-read)

Each arm is 5 Bernoulli trials. The confidence intervals are enormous and all overlap:

| arm | resolve@1 | Wilson 95% CI |
|---|---|---|
| stock-AR | 4/5 = 0.80 | [0.376, 0.964] |
| merged-AR | 2/5 = 0.40 | [0.118, 0.769] |
| diffusion | 1/5 = 0.20 | [0.036, 0.624] |

- **Individual per-instance deltas are within noise** — a single instance flipping moves any arm by
  0.2.
- **The 4/5-vs-1/5 gap (stock-AR vs diffusion) is SUGGESTIVE ONLY, not significant:** Fisher exact
  two-sided **p = 0.206**. stock-vs-merged p = 0.524; merged-vs-diffusion p = 1.00. **Nothing in this
  table reaches significance** — even the widest contrast fails at alpha = 0.05.
- **N required for significance:** to detect the plausible SWE-scale effect (~0.2–0.3 absolute resolve
  gap) at 80% power, alpha = 0.05 two-sided, a two-proportion test needs **~80–93 per arm**; only a very
  large gap (0.8 vs 0.2) is detectable at ~10/arm. **N = 25–50 is the pragmatic go/no-go tier** (it
  will surface large effects and rank the arms, but a small paradigm tax will stay inside the CIs).
  Report paired stats (McNemar on the shared instances), not just marginal rates.

**Bottom line: the direction (stock-AR > merged-AR > diffusion) is consistent and mechanistically
explained, but n=5 licenses a ranking, not a verdict.**

---

## 4. N=25–50 CHANGE LIST

1. **Add a 4TH ARM = diffusion-on-STOCK-conversion** (B@1000-class two-stream conversion of stock
   Qwen3.5-9B, **no RL-v2**). This is the decisive addition: it completes the 2x2
   {weights: stock, RL-v2} x {paradigm: AR, diffusion} and **separates paradigm-vs-weights at scale.**
   If diffusion-on-stock ~= stock-AR, the diffusion paradigm is clean and the RL-v2 payload owns the
   loss; if diffusion-on-stock still lags, the paradigm carries a real tax. **Cheap: ~2–3 h** per the
   certified conversion recipe (`REPRODUCE_V3.md §3.2`, B@1000), no new RL. This is the highest-value
   marginal GPU-hour in the whole plan.
2. **Turn-limit review — raise 50 -> 75.** VERIFIED: **6 of 15 episodes** hit the 50-turn
   `FatalTurnLimitedError` (exit 53) — not 5 (correction to the working note). One of them
   (stock-AR sympy-13757) **resolved *at* the cap**, so the limit is truncating real work, not just
   killing dead episodes. Cost note: turn-limited episodes already run 98–188 s and ~0.9–1.0 M tokens;
   75 turns raises the ceiling ~1.5x on the affected episodes (wall + tokens scale ~linearly with
   turns). Recommend **75** with that cost booked; do NOT go unbounded (the loop-halt episodes show the
   agent will spin).
3. **Keep the aligned runtime + official scoring** exactly as in this run (episode-in-official-container,
   `swebench` docker harness, `score rc=0`). No mock verdicts.

---

## 5. GO / NO-GO

**GO on N=25–50 — but NOT as a diffusion-vs-AR verdict on RL-v2 weights.** That specific comparison is
contaminated by the wrong-payload finding (RL-v2 was GRPO-trained on short structured tool-call
episodes, not SWE-style long-horizon repo edits). The N=25–50 run's PRIMARY job is the **4-arm 2x2**
to disambiguate paradigm-vs-weights with the cheap diffusion-on-stock arm as the pivot. Do not
over-invest GPU in the RL-v2 arms; the actionable next-cycle investment is **SWE-style RL data** via
the certified convert->RL->re-convert loop, so the weights payload matches the eval distribution.

## ARM 4 — diffusion-on-stock-conversion (init+B@1000, no RL-v2) — MONITOR VERDICT ADDENDUM (2026-07-05)
Official scoring (scoring/n5v2-diffstock.n5v2_diffstock.json): **0/5 resolved** — 2 completed-unresolved
(django-13741 446B, pytest-8399 466B real patches, tests not passing), 3 empty (11119 turn-limit no-edit;
12754 + sympy-13757 fast loop-halts ~40s). proj counter 3 events / ~10^6 tokens (contextualized vs the
3-arm diffusion total of 1 — nonzero class NEEDS a per-row explanation before any promotion claim).

### THE 4-ARM LADDER (official docker, aligned runtime, n=5)
stock-AR 4/5 > merged-AR(RL-v2) 2/5 > diffusion(RL-v2) 1/5 > diffusion(B@1000 stock-conversion) 0/5

### Honest adjudication (pre-registered branch: "~1-2/5 => paradigm struggles at SWE scale" — with a caveat)
The 4th arm did NOT cleanly isolate paradigm-vs-weights, because "stock in diffusion form" does not exist:
conversion requires training, and B@1000 is the PRE-RL foundation (34/63-class tool calls) — a weaker
agent than RL-v2 generally, which its two ~40s loop-halts confirm. What the ladder DOES establish:
(1) AR serving preserves SWE capability best; RL-v2 costs SWE capability even served AR (4/5->2/5 —
tool-call specialization tax); (2) diffusion serving costs additional SWE capability at every weight
level; (3) diffusion-twin SWE performance tracks the twin's general agentic capability level.
### CONSEQUENCE: the N=25-50 AR-vs-diffusion horse race is PREMATURE — it would measure this known gap
at higher significance. The path to a competitive diffusion twin on SWE is the methodology's own loop:
train toward SWE-style episodes (SFT/RL on SWE trajectories) and re-convert — the machinery certified by
#29. Recommendation: pause Stage-C scale-up; decision needed on the SWE-tuning campaign.
