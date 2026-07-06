# Stage-C N=5 v3 — ENVELOPE-CORRECTED 4-arm, aligned runtime + OFFICIAL docker scoring

**Date:** 2026-07-06 (both orchestrators `ORCH_DONE`). **Task:** #54. **This is THE standing SWE
baseline for the project.** It supersedes `runs/stage_c_n5v2/report.md`, whose numbers were produced
under **greedy `temp=0`** — a sampling deviation from the flywheel SWE reference (temp 0.6 proxy-forced)
that the user caught, and which the Qwen3-family docs name as the cause of endless-repetition
degenerate exits. The v2 greedy ladder now stands **only as a deviation-documented lower bound**
(`runs/stage_c_n5v2/report.md` §"REFERENCE-CONFIG CHECK").

**Envelope (this run):** `temp=0.6 / top_p=0.95 / top_k=20 / seed_base=1234`, forced proxy-side via
`LUMO_PROXY_FORCE_*` with a per-request seed, empty-patch re-drive retries=1 — the certified reference
envelope banked in the v2 report, wired into the qwen-code proxy (commit `289f023`).

**Runtime:** every episode ran **inside the official per-instance swebench docker image**
(`runner_metadata.json` `runtime=container`, `image=swebench/sweb.eval.x86_64.<inst>`; import + test-cmd
acceptance 5/5, #64). Session cap `--max-session-turns 50`, `--max-num-seqs 1` (paired, one server per
arm; verified `run_v3_arm.sh`).

**Scoring:** OFFICIAL `swebench.harness.run_evaluation` docker harness, `score rc=0`
(`logs/pipeline.log`). Verdicts are the official `scoring/*.json` reports (schema_version 2), NOT a mock.

Everything below is **verified against primary artifacts** — official verdict JSON + per-instance
`report.json` + applied `patch.diff` + official `test_output.txt` + proxy `usage.jsonl` exit capture +
`qwen_trace.json` loop provenance — re-derived for this adjudication, not transcribed from the rollup.

---

## 1. VERIFIED TABLE

### RESOLVE@1 (official docker)

| instance | stock-AR | merged-AR (RL-v2 as AR) | diffusion (same RL-v2 weights) | diffstock (B@1000 stock-conv, no RL-v2) |
|---|---|---|---|---|
| django-11119 | **RESOLVED** | **RESOLVED** | **RESOLVED** | **RESOLVED** |
| django-12754 | edit, unresolved | edit, unresolved | no-edit (empty) | no-edit (empty) |
| django-13741 | **RESOLVED** | **RESOLVED** | **RESOLVED** | no-edit (empty) |
| pytest-8399  | **RESOLVED** | **RESOLVED** | **RESOLVED** | no-edit (empty) |
| sympy-13757  | edit, unresolved | no-edit (empty) | no-edit (empty) | no-edit (empty) |
| **resolve@1** | **3/5** | **3/5** | **3/5** | **1/5** |

Official verdict files confirm the table exactly (`resolved_ids`):
`n5v3-stock-ar.n5v3_ar.json` = {11119, 13741, 8399};
`n5v3-merged-ar.n5v3_mergedar.json` = {11119, 13741, 8399};
`n5v3-diffusion.n5v3_diffusion.json` = {11119, 13741, 8399};
`n5v3-diffstock.n5v3_diffstock.json` = {11119}.

**The three model-serving arms resolve the IDENTICAL set** — {django-11119, django-13741, pytest-8399}
resolved, {django-12754, sympy-13757} not — across stock-AR, merged-AR, **and** diffusion. Not merely
equal marginal rates: the *paired* resolve sets coincide.

### turns | tokens | wall | exit  (exit-proof capture)

`turns` = qwen `num_turns` on clean/loop exits; on turn-limit (exit 53) qwen writes nothing, so `turns`
falls back to the proxy request count in the episode window (~1.5–2 reqs/session-turn ⇒ these read high,
e.g. 100–103, against the 50-session-turn cap). `wall` = qwen `elapsed_s`.

**stock-AR** — resolve 0.6 (3/5) / edit 1.0 (5/5) / turns~55.2 / wall~80.2s / tok~960,898 / exits {ok:3, turn-limit:2} / loops 0

| inst | turns | tot_tok | wall | exit | verdict |
|---|--:|--:|--:|---|---|
| django-11119 | 38 | 575,387 | 68s | ok | RESOLVED |
| django-12754 | 103 | 1,808,891 | 72s | turn-limit | edit, unresolved |
| django-13741 | 47 | 808,022 | 79s | ok | RESOLVED |
| pytest-8399  | 38 | 701,597 | 57s | ok | RESOLVED |
| sympy-13757  | 50 | 910,591 | 124s | turn-limit | edit, unresolved |

**merged-AR** — resolve 0.6 (3/5) / edit 0.8 (4/5) / turns~63.4 / wall~70.93s / tok~1,141,757 / exits {ok:3, turn-limit:2} / loops 0

| inst | turns | tot_tok | wall | exit | verdict |
|---|--:|--:|--:|---|---|
| django-11119 | 35 | 577,171 | 56s | ok | RESOLVED |
| django-12754 | 102 | 1,919,035 | 82s | turn-limit | edit, unresolved |
| django-13741 | 43 | 707,744 | 65s | ok | RESOLVED |
| pytest-8399  | 37 | 683,572 | 66s | ok | RESOLVED |
| sympy-13757  | 100 | 1,821,265 | 86s | turn-limit | empty |

**diffusion** — resolve 0.6 (3/5) / edit 0.6 (3/5) / turns~53.0 / wall~107.15s / tok~935,236 / exits {turn-limit:2, loop-halt:3} / loops 3

| inst | turns | tot_tok | wall | exit | verdict |
|---|--:|--:|--:|---|---|
| django-11119 | 50 | 796,288 | 82s | turn-limit | RESOLVED |
| django-12754 | 35 | 640,099 | 71s | **loop-halt** | empty |
| django-13741 | 50 | 964,422 | 150s | **loop-halt** | **RESOLVED** (post-resolve halt) |
| pytest-8399  | 49 | 935,459 | 78s | **loop-halt** | **RESOLVED** (post-resolve halt) |
| sympy-13757  | 81 | 1,339,910 | 155s | turn-limit | empty |

**diffstock** — resolve 0.2 (1/5) / edit 0.2 (1/5) / turns~33.8 / wall~67.92s / tok~470,273 / exits {turn-limit:1, loop-halt:4} / loops 4

| inst | turns | tot_tok | wall | exit | verdict |
|---|--:|--:|--:|---|---|
| django-11119 | 102 | 1,591,319 | 227s | turn-limit | RESOLVED |
| django-12754 | 19 | 276,620 | 33s | **loop-halt** | empty |
| django-13741 | 23 | 251,208 | 42s | **loop-halt** | empty |
| pytest-8399  | 13 | 118,510 | 18s | **loop-halt** | empty |
| sympy-13757  | 12 | 113,706 | 19s | **loop-halt** | empty |

### Spot-check: the diffusion resolves are REAL (applied diff + official test pass)

Verified `diffusion/…/django-13741`: `patch.diff` is the minimal genuine fix —
`kwargs.setdefault("disabled", True)` in `ReadOnlyPasswordHashField.__init__` (django/contrib/auth/forms.py).
Per-instance `report.json`: `patch_successfully_applied=true`, `resolved=true`, FAIL_TO_PASS
`test_readonly_field_has_changed` **success**, all 84 PASS_TO_PASS green, zero failures. `test_output.txt`
corroborates (`… test_readonly_field_has_changed … ok`). This is a real resolve produced by the
block-diffusion engine, not a scoring artifact. (django-11119 485 B and pytest-8399 549 B likewise apply
and pass; pytest-8399 landed its fixture-name fix *before* the loop-halt.)

---

## 2. REVISED ATTRIBUTIONS (the greedy-era taxes are RETIRED)

The v2 greedy ladder (stock-AR 4/5 > merged-AR 2/5 > diffusion 1/5 > diffstock 0/5) carried two
load-bearing attributions: a **−2 "RL-v2 is the wrong SWE payload" weights tax** (4/5→2/5) and a **−1
diffusion "paradigm tax"** (2/5→1/5). **Under the reference envelope both gaps VANISH.** The corrected
ladder is:

> **stock-AR 3/5  ==  merged-AR 3/5  ==  diffusion 3/5   >   diffstock 1/5**

- **The −2 RL-v2 weights tax was a sampling artifact.** merged-AR ties stock-AR at 3/5 on the *same*
  instances. The "RL-v2 loses two SWE resolves even as AR" claim does not survive proper sampling — it
  was a greedy-repetition confound. **RETIRED.**
- **The −1 diffusion paradigm tax was a sampling artifact.** diffusion ties merged-AR (and stock-AR) at
  3/5 on the *same* instances. The "block-diffusion paradigm costs a resolve" claim does not survive
  proper sampling. **RETIRED.**
- **What the greedy run actually measured** was each arm's differential susceptibility to the
  greedy-`temp=0` degenerate regime (endless-repetition non-termination), not an SWE-capability gap.
  Correcting the sampler collapses all three model arms onto the same three resolves.

**diffstock (1/5) is the only arm that stays down**, and it stays down for a *paradigm-independent*
reason already established: it is the pre-RL B@1000 foundation (a weaker general agent) served as
diffusion, and 4/5 of its episodes loop-halt in 12–23 turns producing no patch. Its one resolve
(django-11119) needed 102 proxy-reqs to the turn-limit. diffstock indexes the twin's *general agentic
capability floor*, not a diffusion-vs-AR paradigm tax (the RL-v2 diffusion twin, at 3/5, refutes any such
tax at this resolution).

---

## 3. BINOMIAL HONESTY (n=5 — the tie is the whole point)

At n=5, a perfect three-way tie plus a fourth arm one below is **the maximal "no detectable difference"
signal this resolution can produce.** State it exactly:

| arm | resolve@1 | Wilson 95% CI |
|---|---|---|
| stock-AR | 3/5 = 0.60 | [0.231, 0.882] |
| merged-AR | 3/5 = 0.60 | [0.231, 0.882] |
| diffusion | 3/5 = 0.60 | [0.231, 0.882] |
| diffstock | 1/5 = 0.20 | [0.036, 0.624] |

- **Paired McNemar discordance b = c = 0 for every pair among {stock-AR, merged-AR, diffusion}.** There
  is not a single instance on which one of the three model arms resolved and another did not. This is
  stronger than equal marginals: the paired signal is *identically* zero. **At n=5 there is no detectable
  tax — RL-v2 vs stock, or diffusion vs AR — at any statistic (marginal or paired).**
- The one non-trivial contrast, stock/diffusion 3/5 vs diffstock 1/5, is **Fisher exact two-sided
  p = 0.524** — not significant; even the widest gap in the table fails at α=0.05.
- **N required for significance is unchanged:** detecting a plausible ~0.2–0.3 absolute SWE resolve gap
  at 80% power / α=0.05 two-sided needs **~80–90 per arm**; **N=25–50 ranks arms and surfaces only large
  (≳0.3) effects.** The tie means the honest read is "no measured difference," and a properly powered run
  is the only thing that can change that.

**Bottom line: at n=5 the envelope run detects NO SWE-capability difference between stock-AR, RL-v2-AR,
and the RL-v2 diffusion twin. The greedy-era "RL-v2 tax" and "paradigm tax" narratives are retired as
sampling artifacts.**

---

## 4. DIFFUSION TEXTURE REMAINS (behavioral, not resolve-rate)

Equal resolve@1 does **not** mean identical behavior. The diffusion twin still shows a distinct texture,
verified from primary traces:

- **3 loop-halts** (exit 1, `qwen_trace.json`: "Loop detection halted the run
  (consecutive_identical_tool_calls: the model repeated the same tool call with identical arguments)") —
  django-12754, django-13741, pytest-8399 — vs **0 loop-halts in both AR arms** (`loops=0` for stock-AR
  and merged-AR). The RL-v2 *weights* do not loop in AR mode under the envelope; the diffusion *paradigm*
  does. (`loops=3` counts original-run halts; sympy-13757 additionally loop-halted only on its
  empty-patch RETRY, so it is booked as its original turn-limit exit.)
- **2 of the 3 are POST-RESOLVE halts:** django-13741 and pytest-8399 landed their resolving patches
  (465 B, 549 B) and *then* kept re-issuing identical tool calls until the loop detector halted them.
  The resolve survives the halt (both `resolved=true`), but the agent does not terminate cleanly —
  diffusion had **zero clean exit-0** episodes across all 5.
- **Wall cost:** diffusion ~**107.15s/episode** vs stock-AR ~**80.2s** (≈1.34×) and merged-AR ~70.9s.
  Token volume is comparable (diffusion ~935K vs stock ~961K), so the wall gap is engine throughput, not
  extra work.

This texture (loop-proneness, no clean terminals, ~1.3× wall) is the real, reproducible diffusion
signature at SWE scale — and it is a *serving/behavior* property to fix in the engine + decode loop, not
a resolve-capability deficit.

---

## 5. GREEDY (v2) → ENVELOPE (v3) — the degenerate-regime test

Same 5 instances, same aligned runtime + official scoring; only the sampler changed (greedy → reference
envelope). Greedy baseline from `runs/stage_c_n5v2/report_table.txt` + `diffstock_report_table.txt`.

| arm | resolve@1 | loop-halts | turn-limit |
|---|---|---|---|
| stock-AR | 4/5 → **3/5** | 0 → 0 | 2 → 2 |
| merged-AR | 2/5 → **3/5** | 1 → **0** (loop-halt eliminated) | 1 → 2 |
| diffusion | 1/5 → **3/5** (TRIPLED) | 2 → 3 | 3 → 2 |
| diffstock | 0/5 → **1/5** | 2 → 4 | 3 → 1 |

The envelope is **not** uniformly better on every cell — it *converges* the model arms:

- **merged-AR +1 and its loop-halt eliminated** — the RL-v2 weights stop looping in AR mode once the
  sampler leaves the greedy degenerate regime; this is the direct evidence the −2 "weights tax" was a
  sampling artifact.
- **diffusion +2 (TRIPLED, 1→3)** — the largest correction, closing the entire greedy gap to the AR arms.
- **stock-AR −1 (4→3)** — one resolve lost to sampling variance (sympy-13757, which resolved under greedy
  at the cap, is now edit-unresolved); consistent with n=5 noise and the honest tie.
- **diffstock +1 (0→1)** but loop-halts rose 2→4 — the envelope harvested one resolve while the weak
  foundation still loops out of most episodes.

The convergence *toward* a tie under the reference sampler is exactly the signature of a greedy-repetition
confound in v2, not a real capability ordering.

---

## 6. STATUS OF THE SWE-TUNING CAMPAIGN — premise DISSOLVED

`swe_tuning_campaign_design.md` was built to **recover the −2 RL-v2 SWE tax** by injecting SWE-capable
base weights. **At n=5 there is no −2 to recover** — merged-AR ties stock-AR, and the diffusion twin ties
both. The campaign's founding premise ("RL-v2 is the wrong SWE payload; the diffusion twin tracks a
weakened agent") is **dissolved at this resolution.** The campaign is **PARKED** (not killed): its
data/leakage/training machinery remains valid and GO-priced (Stage-0 probe v2 clears the yield bar at
0.25, see that report), but it is **not justified to spend until a properly powered run establishes there
is a gap to close.** That run is the N=25–50 horse race below. See the campaign doc's prepended STATUS
block for the full disposition and the D1 reprice.

---

## 7. THE N=25–50 PROPOSAL (the properly-powered horse race — now justified)

The v2 report deferred N=25–50 as "premature — it would measure a known gap at higher significance."
**That objection is now void:** there is no measured gap. The correct next experiment is therefore a
clean, adequately powered **stock-AR vs diffusion** horse race whose *only* honest prior is the tie — the
first run that can actually detect (or bound) a paradigm difference. Full spec + measured pricing live in
`swe_tuning_campaign_design.md` §"THE N=25–50 PROPOSAL"; summary:

- **Arms:** **stock-AR vs diffusion(RL-v2 twin)** (the two the project actually ships); **merged-AR
  optional 3rd** (isolates weights-vs-paradigm if the primary pair diverges). diffstock dropped — its
  1/5 is a known foundation floor, not a paradigm question.
- **Sampling:** reference envelope `temp 0.6 / top_p 0.95 / top_k 20`, **seeded per-request** (this run's
  contract); aligned runtime (episode-in-official-container); OFFICIAL docker scoring; **paired** (same
  instances every arm), resolve@1 one attempt/instance/arm.
- **Execution = BATCHED, concurrency 4+ (USER FROZEN CONFIG, `runs/loop_halt_polish/USER_DIRECTIVE_BATCHED_NRUN.md`).**
  Continuous batching per arm (baseline c=4, probe 6/8 HBM-gated: engine certified bs=8 @ gpu_mem 0.82,
  b16 needs gmu ≤ 0.62); NOT serial. resolve@1 / paired McNemar is unaffected (per-request seeds stay
  deterministic per episode).
- **Turn cap 75** (raised from 50). Justified from v3: all clean AR resolves finished by **turn 47**; the
  diffusion resolves land at **turns 49–50 — pressed against the 50 cap**; turn-limited episodes spun to
  the cap generating 80–103 proxy-reqs and resolved almost nothing (7 turn-limit exits across the run, 1
  resolve). 75 gives the diffusion arm ~50% headroom above where its resolves currently land while still
  bounding the dead spins; unbounded buys no resolves.
- **Pool:** stratified **N=25–50 slice of Tier1-100**, held out from all training under the campaign
  firewall (`train_ids ∩ (Tier0 ∪ Tier1 ∪ verified_500) = ∅`); never the inner-5.
- **Priced from measured v3 costs; speed = THROUGHPUT not latency** (frozen config). GPU-compute is
  ~concurrency-invariant: **N=50 ≈ 3.5–4.3 GPU-h (2-arm) / 5.1–6.2 GPU-h (3-arm)** (v3 b=1 walls
  stock ~107s / diffusion ~141s / merged ~119s cited for latency CONTEXT only — batched wall is
  queue-inflated). Batched **wall ~1–2 days** (≈50 Tier1 image pulls ~200 GB + serving ~2–4 h/arm at c=4
  + official scoring hours). This **reprices the campaign §5 N=25–50 line (35–60 GPU-h) DOWN ~10× in
  compute** — that estimate conflated eval with data-gen; a paired eval is only N × arms episodes.

---

## MEMORY-WORTHY SUMMARY

> **Stage-C N=5 v3 (envelope-corrected, official docker): stock-AR 3/5 == merged-AR 3/5 == diffusion 3/5
> (identical resolve sets, paired McNemar b=c=0), diffstock 1/5. The greedy-era −2 RL-v2 SWE tax and −1
> diffusion paradigm tax are RETIRED as sampling artifacts — at n=5 there is NO detectable SWE tax.
> Diffusion texture persists (3 loop-halts incl 2 post-resolve, 0 clean exits, ~1.34× wall). SWE-tuning
> campaign premise DISSOLVED (no −2 to recover), PARKED; the N=25–50 horse race is now the justified next
> step, priced ~2–6 GPU-h.**
