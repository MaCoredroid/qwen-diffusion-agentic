# K-Raise Campaign — lift K_max above 1.0 at SWE-Verified parity (behavioral gate, in-conversion K)

**Author:** design synthesis, 2026-07-06. **CPU-only design; no GPU run** (datagen owns the GPU). All GPU-hour
numbers are honest estimates flagged "measure, never assume." **User directive:** *raise K_max=1.0 while
MAINTAINING SWE-Verified on par — leakage attention explicit.*
**Owner discipline:** [[qwen-diffusion-commit-workflow]] (commit+push each step, narrate), [[diffusion-promotion-discipline]]
(promote only on raw/constrained model-only gains), [[native-function-format-rule]] (qwen3_xml native format across
gen/train/eval/decode), [[retrain-freely-rule]], [[gpu-utilization-standard]] (say NO to low util).
**Evidence base (read first):** `s2_pilot_design.md`+`s2_pilot_result.md` (the KILL), `l1_content_mix_result.md`
(K_max=1.0 measured, Amdahl), `goal_5x_rollout_b1.md` (retired 5×, K arithmetic), `runs/w2_n50/report.json` (the SWE
behavioral bar + failure signature), `swe_tuning_campaign_design.md` (the SFT campaign this rides on), `REPRODUCE_V3.md`
§3.2/§3.3/§3.9 (conversion recipes), `runs/swe_datagen_s1/{USER_LEVER_BELT,INTERVENTION_NOTE}.md` + `data/*/pool_manifest.json`
(the leakage rings), `scripts/eval_flare_freetext_cad.py` (the CAD sampler, byte-exact at k=1),
`engine_build_status.md` §0.G/§0.I (free-text overshoot + batch-occupancy).

---

## 0. TL;DR — the two reframes that make this NOT the S2 pilot again

S2 ran the cheapest decisive test of "train reasoning-span K 1→≥2" and **KILLed**: peak committed **1.053 tok/fwd**
(γ0.90), K=2-commit saturating at **5.3 %**, an order of magnitude short of 2.0. The failure was **K-non-engagement**
(McNemar p=1.000, no exactness loss) driven by the measured **0.238 top-1 conditional** on reasoning tokens — too little
low-entropy connective mass to average to 2.0 against a **token-exact GSM8K bar**. This campaign changes **two** things
that S2 held fixed, and keeps everything else S2 taught:

1. **The gate is BEHAVIORAL, not token-exact.** Success = the SWE-Verified patch **resolves**, not that the reasoning
   tokens match a K=1 twin byte-for-byte. Reasoning spans may diverge token-wise freely; only **value spans**
   (tool-call args, code identifiers, paths, line numbers, literals) must stay exact, and those are held K=1 by
   construction. This (a) **softens the kill** — a K that fails to engage costs *nothing* (the K=1 twin is untouched,
   ship it) rather than retiring a claim, and (b) **unlocks the one genuinely new speed lever S2 could not pull: a
   relaxed confidence threshold γ.** S2 needed γ∈{0.90,0.99} to preserve specific tokens; a behavioral gate lets γ drop
   toward 0.6–0.7 and commit runs of *good-enough* tokens, because a slightly-different-but-plausible connective token
   still yields a resolving patch. The S2 K-curve *climbed* as γ relaxed (0.99→0.95→0.90 gave 0%→1.4%→5.3%); the
   behavioral gate lets us extrapolate that curve into γ<0.9 territory S2's exactness bar forbade.

2. **K is trained IN THE CONVERSION, not grafted post-hoc.** The certified loop **already requires** a fresh two-stream
   re-conversion after SWE-SFT (`swe_tuning_campaign_design.md` §3; `REPRODUCE_V3.md` §3.9). S2's LoRA had to *fight* an
   already-converged K=1 denoiser and moved joint-commit only 1.5%→5.3%. Here the K-consistency pressure is folded into
   the **`L_diff` of that same re-conversion**, so the denoiser is *formed* with joint-commit propensity instead of
   bolted onto a K=1 fixed point. Marginal train cost ≈ 0 GPU-h (same 400-step conversion, re-weighted loss); only the
   adaptive-K eval sweeps are new spend.

**Honest headline:** the behavioral reframe makes the *downside* free and unlocks a real new lever, but it does **not**
guarantee K engages — the entropy wall is a property of the base distribution and code has its own value-density tax
(§8). This campaign is designed so that every rung either **ships a measured speed win at proven SWE-parity** or **stops
clean with the K=1 twin intact**. There is no rung whose failure costs a capability.

---

## 1. Object under test + lineage

| symbol | what it is | provenance |
|---|---|---|
| **M_swe** | AR base = SWE-SFT winner (S=merged-RL-v2+SWE-SFT, or T=stock+SWE-SFT) | `swe_tuning_campaign_design.md` §2, D2 |
| **twin@K1** (reference) | M_swe re-converted two-stream (Run-1 recipe, `L_diff` **plain**), decoded **K=1** | the certified-loop deliverable; §3.9 protocol |
| **twin@Kc** (candidate) | M_swe re-converted with **K-consistency `L_diff`** (§4), decoded adaptive-K | this campaign's trained arm |
| CAD sampler | `scripts/eval_flare_freetext_cad.py`, sha `e12364e7…6104b87` | byte-exact k=1 cert PASS (§5) |

The K-raise campaign takes the re-converted SWE twin and (a) turns on **entropy-gated adaptive-K decode** on twin@K1
(decode-only rungs, cheap) and (b) if decode alone stalls, trains **twin@Kc** (the in-conversion K objective). The
**par reference is always twin@K1** — its own K=1 self, paired — never AR. This isolates the K-raise effect from the
SWE-SFT effect and from the paradigm tax.

### 1.1 ENTRY PRECONDITION — the par gate needs a non-floor twin (do not skip this)

The current promoted twin (RL-v2, pre-SWE-SFT) resolves **2/50** on `w2_n50` vs AR 19/50 (`report.json`, net −17,
p=0.0), with **26 loop-halts / 35 empty-patches / 25 pre-resolve halts**. **A 2/50 twin cannot power a par gate:** paired
McNemar on a floored twin trivially "passes" (you cannot lose resolves you never had), which is *not evidence* that K
holds SWE. **Therefore this campaign ENTERS only after** the SWE-SFT re-conversion (`swe_tuning_campaign_design.md`,
currently PARKED; datagen live under the cycle-2 intervention) delivers a **twin@K1 resolving ≥ 12/46 (≈26 %) on the
fresh Tier1-C46 slice (§7)** — the floor at which McNemar has power to detect a 3–4 resolve loss. If SWE-SFT stays parked,
the K-track may still run its **infra + decode-only** rungs on the best available twin, but every SWE-par row on a floored
twin is reported **INCONCLUSIVE-BY-POWER**, not PASS. The infra track (§10) is **not** gated on this precondition and
proceeds immediately.

---

## 2. REFRAME 1 — the full BEHAVIORAL gate set (stats, thresholds, seeds)

Every K rung is adjudicated against **four** measurements. The primary is behavioral SWE-parity; the other three are the
S2 safety anchors carried over verbatim (they held in S2 and must keep holding). **Sampler-pinned per `REPRODUCE_V2 §0`:
every row records git commit, script sha256, sampler fn, base+adapter paths, dataset+manifest hash, decode flags (γ,
k_max), value-projection audit path.**

### 2.1 PRIMARY — SWE-Verified resolve@1, paired vs twin@K1, on the FRESH held-out slice

- **Slice:** `Tier1-C46` = **Tier1-100 ∖ (w2_n50-50 ∪ gate_ladder_5)** ≈ 46–48 instances (exact count asserted at build;
  §7). **This slice has never been used in ANY prior tuning or eval decision** — `w2_n50` used the *other* 50 Tier1
  instances for the AR-vs-diffusion horse race; its complement is pristine. It is firewalled from training (Tier1 is a
  held-out ring). Naming it explicitly is the point of the directive's "name which ring/slice."
- **Sampling contract (frozen to the v3/w2 standing baseline):** reference envelope **temp 0.6 / top_p 0.95 / top_k 20**,
  proxy-forced (`LUMO_PROXY_FORCE_*`), **per-request seeded**, empty-patch re-drive retries=1, episode-in-official-
  container (`swebench/sweb.eval.x86_64.<inst>`), native `qwen3_xml`, turn cap **75**. resolve@1 = **one seeded attempt
  per (arm, instance)**; the **same seed** is used for twin@K1 and twin@Kc on each instance so the *only* difference is
  decode-K (true pairing). Official `swebench.harness.run_evaluation` docker scoring, no mock.
- **Paired statistic:** on the shared instances, `b` = (twin@K1 resolves ∧ K>1 fails), `c` = (K>1 resolves ∧ K1 fails),
  net-loss = `b − c`; two-sided **exact-binomial McNemar** on `(b,c)`.
- **Seeds:** primary seed `1234`; **robustness seed** `20260701` (two seeds per [[retrain-freely-rule]] so a par call is
  never on one seed). Report both; the verdict is on the primary, confirmed-in-direction by the robustness seed.

| verdict | rule |
|---|---|
| **PAR-HOLD (pass)** | `net-loss ≤ 2` **AND** McNemar `p ≥ 0.05` (the `w2_n50` parity rule verbatim) — K did not detectably regress SWE |
| **PAR-KILL (hard)** | `net-loss ≥ 4` **AND** `p < 0.05` — K>1 significantly *regresses* SWE resolve ⇒ revert to twin@K1, rung unshippable |
| **INCONCLUSIVE** | `net-loss = 3` or `p` in the gap, or twin@K1 below the §1.1 power floor ⇒ one re-seed / report ranking, do not ship |

### 2.2 SECONDARY — the K-engagement measurement (is there any speed to gate?)

On the **same** Tier1-C46 diffusion episodes, the CAD sampler emits per-turn `denoise_forwards`, committed tokens, and
the commit-k histogram (§5). Report **blended avg committed tok/fwd** and **reasoning-span tok/fwd** (value-span
positions excluded via the audit counters). This is the "did K engage" signal — **it is not a capability gate**; it
decides SPEED-FAIL vs SPEED-PASS per rung (§6).

### 2.3 ANCHOR — tool-call matched-20 exact_args (the certified capability must not move)

`eval_flare_northstar_hybrid_clean.py` (sha pinned §6-table of `s2_pilot_design.md`), 20 matched turns, hybrid-clean
(K=1 FSM values, **adaptive-K OFF on this path** — the tool-call value spans are FSM-forced K=1 regardless of the
free-text K). Anchor = twin@K1's own matched-20 exact_args. **PASS:** McNemar net-loss vs anchor not significant
(p≥0.05) AND raw ≥ anchor − 3 (the #29 bar). **KILL-anchor:** significant net-loss ⇒ the free-text K-decode perturbed
the FSM tool-call path (should be impossible by construction — this gate catches a wiring regression).

### 2.4 RETENTION — GSM8K legacy full-context N=20 (general reasoning pinned)

`eval_flare_stage1_ab_diffusion.py::full_context_sample_one`, temp 0.0, N=20, base twin@K1/@Kc, seed 20260701.
Anchor = twin@K1 GSM8K (expected ≈13/20 = the RL-v2/conversion floor). **PASS:** ≥ anchor. **KILL-retention:** ≤ anchor−2
(fell below the conversion floor) OR in-training rolling KL-to-base > 0.05 unrecovered (twin@Kc only). Do **not** use
`measure_block_quality_curve.py` (disqualified mutable-remask diagnostic, `REPRODUCE_V3 §6.4`).

### 2.5 AUDITS — value-projection, zero-tolerance (KILL-3)

`audit_value_projection_tokens.py` on **every** diffusion turns.jsonl (SWE + retention + anchor). Required all-0:
`value_projection_events`, `parallel_commit_forced_tokens_counter`, `wave{1,2}_*`, `zero_forward_rows`;
`verification_mode == no_projection_events`. **Any nonzero ⇒ KILL-3** — the tok/fwd number is contaminated (this class
has produced every phantom win in the project). The CAD sampler already surfaces these counters (`_capture_stats`).

**RUNG PASS = PAR-HOLD ∧ anchor PASS ∧ retention PASS ∧ audits clean ∧ SPEED-PASS.**

---

## 3. Why the S2 failure mode does not transfer 1:1 — and the honest risk it does (constraint set)

The S2 KILL is the load-bearing prior. Stating what differs, honestly, including the transfer risk (directive (c)):

**What genuinely differs (why K has more room here):**
- **The token-exact wall is gone on reasoning content.** S2 died because the numeric/derived positions dominated the
  *average* tok/fwd and any joint-commit that changed a token broke GSM8K's strict `#### answer`. SWE resolve is a
  terminal 0/1 over a long trajectory; the connective/reasoning tokens can diverge arbitrarily. This is not a re-run of
  "manufacture the correct value" (SDTT's documented null) — it never was; but now it is *also* not "reproduce the
  correct connective token," which S2 *did* require.
- **γ can be relaxed (the new lever).** Behavioral tolerance lets γ drop to 0.6–0.7. S2 measured monotone K-growth as γ
  fell but hit the exactness floor at 0.90; this campaign sweeps the γ range S2 could not.
- **Code has more low-entropy structural mass than GSM8K arithmetic.** Indentation, closing brackets/parens, common
  boilerplate, and — critically — **copy-from-context spans** (re-emitting identifiers/paths already in the prompt) are
  exactly the `C≈0` copy spans ParallelBench says are trainable-parallel-safe (`training_redesign_10x_research`). GSM8K
  CoT has almost none of this; SWE tool-call+edit trajectories are full of it.
- **The denoiser is formed with the objective, not fighting it (reframe 2).** S2's post-hoc LoRA moved joint-commit 3.5×
  but off a converged K=1 base; the in-conversion objective (§4) never lets the denoiser converge to strict K=1.

**The honest risk it DOES transfer (do not hide this):**
- **The 0.238 entropy wall is a base-distribution property.** Code reasoning ("the bug is in function `_resolve`, at
  line 214, because `x` is `None`") is arguably *higher*-entropy than arithmetic in its *value* content — every
  identifier, path, and line number is a value that must be exact, and the entropy gate will block K>1 on each. If SWE
  reasoning is value-dense wall-to-wall, adaptive-K collapses to ≈K=1 **exactly as GSM8K did**, and the relaxed-γ lever
  cannot rescue it (relaxing γ on a value position produces a *wrong identifier* → broken patch → PAR loss, not a free
  speed gain). This is the central bet, and it is genuinely uncertain.
- **Code value-exactness is stricter in one axis than GSM8K.** A single wrong character in a path or identifier fails the
  patch. So the "value-blocked" mechanism must be **broader and more reliable** for code than the numeric-block that
  sufficed for GSM8K. If the entropy gate does not reliably fire on code value tokens (i.e. the model is *confidently
  wrong* on an identifier), relaxed-γ commit will silently corrupt values → PAR-KILL. §5 hardens the value-block for
  exactly this.

**Constraints inherited from S2 (non-negotiable):**
- **Values ALWAYS sequential/exact** — K=1 on every value span, by FSM (tool-call args) and by the contiguous-prefix
  entropy gate (free-text values). Never trained or decoded to joint-commit a value. This is the invariant S2 got right
  and the reason its KILL was clean (audits stayed 0).
- **Erosion cap ≤ 400–600 steps**; retrain-freely at a *different* step count in {300,400,500} rather than extending to
  rescue a weak number.
- **Byte-exact k=1 certificate before any K>1 row** (§5) — the sampler must be a pure extension of the promoted path or
  no row is comparable (the S2 R1 discipline; the CAD sampler already carries the certificate).
- **The K-curve saturates; do not chase a factor-of-two miss with more steps** (S2 lesson: the wall is
  entropy/architecture, not training dose).

---

## 4. REFRAME 2 — K trained IN THE CONVERSION: the `L_diff` objective

The re-conversion is the two-stream FLARE recipe (`FASTDLLM_FLARE_TWO_STREAM=1`; clean stream `L_AR` byte-identical to
the AR forward; denoise stream `L_diff` on masked answer positions; GDN `route_i`, state read-only during denoise,
advanced once at the committed block boundary — the 6/6 bit-identical snapshot-restore discipline). The **plain** recipe
(`REPRODUCE_V3 §3.3`: 400 steps, block 512, bd 32, lr 1e-5, LoRA r16/α32, targets q/k/v/o + in_proj_{qkv,z,b,a} +
out_proj, `VALUE_SPAN_LOSS_WEIGHT=2.0`, default mask schedule) trains the K=1 denoiser (twin@K1). twin@Kc adds
**K-consistency pressure** to `L_diff` via the options below. **Convention note (avoid the S2 sampler-mislabel trap,
`REPRODUCE_V3 §6.3`):** let `ρ` = fraction of a block's answer positions left MASKED (to-predict) per microbatch; the
plain conversion sees `ρ ~ U(0.30,0.80)`. "More multi-token prediction" = presenting the denoiser with more states where
**≥2 adjacent frontier positions are masked and must be committed from one forward.**

### 4.1 O1 — Frontier-adjacency mask schedule (train == serve) — PRIMARY

Bias the training mask pattern to **match the adaptive-K decode geometry**: instead of random-subset masking, stage `k`
**contiguous trailing masks on the block frontier** (exactly the states the CAD sampler feeds the engine, §5) and compute
`L_diff` **jointly over those k positions** from a single forward. Bias the effective **denoise-step count low** (few
refinement steps per block ⇒ each step commits 2–4 adjacent frontier tokens ⇒ the "mask-ratio biased low = more
multi-token prediction" the directive names). This is train==serve consistency: the denoiser sees the k-trailing-mask
frontier it will actually be decoded on, not isolated random infills. Directly generalizes S2's `r_S~U(0.50,0.90)`
higher-mask idea, but *contiguous-on-the-frontier* rather than random-subset, and folded into the conversion rather than
a post-hoc LoRA.

### 4.2 O2 — Span-level consistency weighting (reasoning vs value) — PRIMARY, and the value guard

Weight `L_diff` by **span class**, reusing the Run-1 span tagger that already drives `VALUE_SPAN_LOSS_WEIGHT=2.0`:
- **Reasoning / connective / structural spans:** high joint-consistency weight — the k-contiguous joint-CE of O1.
- **VALUE spans** (tool-call arg bodies, code identifiers, paths, line numbers, string/number literals): **standard K=1
  sequential CE only — never entered into a joint-commit target.** The denoiser is explicitly trained "parallelize
  connective, serialize values." This is the *training-time* analogue of the decode-time entropy gate and the primary
  defense against the §3 transfer risk (confidently-wrong identifiers). Values-always-sequential is the hard invariant.

### 4.3 O3 — K-curriculum (the schedule for O1) — RECOMMENDED

Anneal the joint-commit width during the 400-step conversion: `k=1` for the first ~⅓ (establish a stable base denoiser),
then curriculum `k→2` then `k→4` over the remainder, mirroring the staged decode targets (§6). Prevents early
destabilization (the erosion risk) while still forming wide-commit propensity. Pairs with the ≤400–600 cap; save every
100 steps for the sweep.

**Recommended objective = O1 (frontier-adjacency schedule) + O2 (span-class weighting), ramped by O3 (K-curriculum),
values-always-sequential invariant across all three.** Banked fallback = the DSCD nested-KL (`τ=2.0`) if the plain
cached-target joint-CE destabilizes retention (the S2 §11 fallback). **In-training safety kit** (RL kit, verbatim from
S2 §5 / `swe_tuning_campaign_design §2.5`): retention probe every 50 steps; rolling **KL-to-base early-stop at 0.05**;
report `max_KL_to_base`. **LoRA:** r16/α32 (the RL-v2/Run-1 envelope), attn+GDN targets (add MLP `gate_up/down` only if
matching the SWE-SFT target set, per §2.3 of the SFT design). Two seeds.

**Leakage inside the conversion (the sharp-test premise, preserved):** twin@Kc is trained on the **plain conversion mix
`data/flare_redesign_run1_copy_retention_mix`** — which **excludes both the RL-v2 pool and the SWE-SFT pool** — so the
re-conversion is *not trained on* the capabilities it must preserve. The SWE capability lives in the merged base weights
(M_swe), not in the conversion data. This is unchanged from `swe_tuning_campaign_design §3.1`.

---

## 5. Decode instrument — entropy-gated adaptive-K (k_max 2→4, value-blocked, native stop)

**The sampler exists and is validated:** `scripts/eval_flare_freetext_cad.py` (sha `e12364e7…6104b87`), a monkeypatch of
the promoted engine `_hybrid_clean_step`. Mechanism (verified): stage `k` trailing [MASK] probes, read the `k` `+1`-
shifted probe logits, commit the **leading contiguous run** with per-position confidence `c_i = max-softmax ≥ γ`, clip
`[1,k_max]`; **a sub-γ position blocks the run ⇒ values stay K=1**; **native EOS-stop mid-run**; **never remask** (GDN
state discipline / FR13 cache preserved). **R1 byte-exact certificate PASS** at k=1/γ1.0: reproduced the anchor 26/30 ·
0.8618 tok/fwd · gen_text 30/30 identical (`s2_pilot_result.md`). It is a pure extension of the promoted path.

**Extensions this campaign requires (all small, parity-gated):**
- **k_max = 4** (S2 was k_max=2). The staging loop already computes `k = min(k_max, room)`; extend the probe read and the
  contiguous-run commit to 4. Re-run the k=1 byte-exact cert after the change (mandatory).
- **γ range extended down** to `{0.6, 0.7, 0.8, 0.9}` (S2 used {0.90,0.95,0.99}) — the behavioral-gate lever.
- **Value-block hardened for code (the §3 defense).** Two layers: (i) the existing contiguous-prefix entropy block (a
  low-confidence position halts the run); (ii) on the **tool-call path**, values remain **FSM-forced K=1** regardless of
  γ (hybrid-clean, adaptive-K OFF) — unchanged. For **free-text** value tokens, keep the entropy block and additionally
  cap the committed run at any position whose top-2 margin is thin (a "confidently wrong identifier" guard); this is a
  decode-time invariant that does not need training.
- **Temperature interaction (spell it out).** Under the envelope temp 0.6, the entropy read uses the **pre-temperature
  logits** (max-softmax on raw logits) so the *commit decision is temperature-independent*, while the *sampled token*
  respects temperature. The k=1 certificate is **byte-exact at temp 0**; at temp 0.6 the K=1 reference is the temp-0.6
  twin@K1 itself (a *distributional-match*, not byte-match, certificate) — this is why the par gate pairs against the
  temp-0.6 twin@K1, same seed, not against the greedy anchor.

---

## 6. Staged K targets (1.5 → 2 → 4) — each with its own gate + dual-exit kill

Each rung has **two independent exits**, the crux of the behavioral reframe:
- **SPEED-FAIL (soft stop, costs nothing):** K does not reach the rung's tok/fwd target at par ⇒ the rung yields no
  speed ⇒ **stop the K-track at the last passing rung and ship it** with the K=1 twin intact. This is the S2 mode, but
  here it is a *dead-end-detected*, not a capability KILL.
- **PAR-KILL (hard):** K>1 significantly regresses SWE resolve (§2.1) OR breaks the tool-call anchor (§2.3) OR retention
  (§2.4) OR trips audits (§2.5) ⇒ **revert to twin@K1**, that rung is unshippable.

Each rung is run **decode-only first** (turn adaptive-K on over twin@K1 — cheap, no retrain); only if decode-only
SPEED-FAILs do we spend the trained twin@Kc. This is the promotion discipline: **credit training only if twin@Kc beats
decode-only-on-twin@K1 at par** ([[diffusion-promotion-discipline]]).

| rung | decode | target (SECONDARY §2.2) | γ regime | trained? | SPEED-PASS ⇒ | SPEED-FAIL ⇒ | PAR-KILL ⇒ |
|---|---|---|---|---|---|---|---|
| **K1.5** | k_max=2 | blended avg **≥1.5** tok/fwd (reasoning-span ≥1.5) | sweep {0.7,0.8,0.9} | decode-only first; twin@Kc if stalls | ship K1.5, go K2 | try twin@Kc; if still <1.5, **stop-ship K=1** | revert twin@K1 |
| **K2** | k_max=2 | blended avg **≥2.0** (reasoning ≥2) | push {0.6,0.7} | twin@Kc (O1+O2, curriculum to k=2) | ship K2, go K4 | ship K1.5 (last pass) | revert twin@K1 |
| **K4** | k_max=4 | blended avg **≥3.0** (reasoning ≥4) | {0.6,0.7} | twin@Kc (curriculum to k=4) | ship K4 | ship K2 (last pass) | revert twin@K1 |

**Target calibration (honest, Amdahl-anchored).** Blended avg tok/fwd = `1 / (f_value + (1−f_value)/K_reason)` with
grammar-scaffold folded out. GSM8K measured `f_value ≈ 2–5 %`; **code is value-denser** — estimate `f_value ≈ 0.25–0.40`
(identifiers/paths/numbers), **to be MEASURED by the L1-SWE census (§6.1) before the rungs run.** At `f_value = 0.30`:
K_reason 1.5 → blended **1.30**; K_reason 2 → **1.54**; K_reason 4 → **2.11**. So the nominal reasoning-span rungs
1.5/2/4 map to *blended* ≈1.3/1.5/2.1 — the blended targets in the table are set accordingly. **The higher code f_value
is the honest cap: even perfect reasoning-span K=∞ tops out at `1/f_value` ≈ 2.5–4× blended, so K4-blended-3.0 is the
optimistic edge, not the base case.** K1.5 is deliberately the *plausibly-reachable* first rung given relaxed γ (S2
already hit 1.05 decode-only at γ0.90 with the exactness bar *on*); K4 is the ambitious rung the entropy wall most likely
caps.

### 6.1 L1-SWE census (cheap pre-step, gates the whole projection)

Before any rung, run the CAD sampler's counters over the **datagen keeper trajectories** (`runs/swe_datagen_s1/keepers`)
and a decode-only pass on Tier1-C46: measure the **code content mix** (grammar-scaffold % / value % / structural %) and
the **top-1 conditional entropy on reasoning vs value spans** — the code analogues of `l1_content_mix_result.md`'s
0.238/54.2%/etc. This calibrates `f_value` and tells us *a priori* whether the entropy wall transfers (if reasoning-span
top-1 conditional ≈ GSM8K's 0.238 or worse, expect K to stall; if code copy-spans pull it lower, expect engagement).
**~0.5 GPU-h; do this first — it may pre-KILL the K-track on evidence before spending a rung.**

---

## 7. LEAKAGE firewall (explicit — the correctness spine)

**Training data = ONLY {datagen keepers} ∪ {original conversion mixes}.** Concretely:
- SWE-SFT stage: `runs/swe_datagen_s1/keepers` (SWE-Gym, repo-overlapping but **instance_id-disjoint** from Verified by
  construction; cycle-2 re-stratified + best-of-3).
- Re-conversion stage: `data/flare_redesign_run1_copy_retention_mix` (excludes RL-v2 pool **and** SWE-SFT pool).
- **No eval-ring instance ever enters training.** The existing `build_frontier.py::firewall_assert` (KILL-D1) HARD-asserts
  `train_ids ∩ (verified_500 ∪ Tier0 ∪ Tier1) = ∅`; manifest `data/swe_sft_pool/pool_manifest.json::kill_d1_check`
  currently PASS (`intersect_verified_500=0`, `intersect_tier0_union_tier1=0`, train_ids_n=2438). **Re-assert at
  train-launch AND at eval-launch.**

**Eval = held-out rings ONLY. The SWE-par gate runs on `Tier1-C46`, the FRESH slice never used in any prior tuning
decision:**
- `Tier1-C46` = **Tier1-100 ∖ (w2_n50_ids ∪ gate_ladder_5)**. `w2_n50` drew 50 of the 98 leakage-cleared Tier1
  candidates (`data/swe_w2_n50_pool/pool_manifest.json::sampling.source_subset = "Tier1-100 (seed=0)"`); its complement
  (~46–48 instances) was **never drawn**, never scored, never used to make a decision. `gate_ladder_5` (the stage_c
  baseline) and the 2 gate-ladder ids already removed from Tier1 candidates are excluded belt-and-suspenders.
- **Hash asserts (build-time + run-time, KILL-D1-class):**
  `Tier1_C46 ∩ train_ids = ∅` · `Tier1_C46 ∩ w2_n50_ids = ∅` · `Tier1_C46 ∩ gate_ladder_5 = ∅` ·
  `Tier1_C46 ⊂ tier1_100`. Emit `k_raise_pool_manifest.json` with the id list, each id's source ring, and a
  `pool_sha256`. Any nonzero intersection ⇒ **KILL-D1, do not eval.**
- **Anchor/retention sets:** tool-call matched-20 (never-train) and GSM8K test-split N=20 (disjoint from train by
  construction) — unchanged, hash-checked.

**Why NOT reuse w2_n50's 50:** those instances *were* used — the AR-vs-diffusion decision. Reusing them risks tuning γ /
the objective to that specific slice. The directive's "FRESH held-out slice never used in any prior tuning decision" is
satisfied only by the complement. If Tier1-C46 proves too small for power after the twin lifts off the floor, the
**escalation is the USER_LEVER_BELT decision** (relax to a larger held-out draw), not silently reusing w2_n50.

---

## 8. Budget per stage (GPU-h; RTX 5090 serving, docker scoring off-GPU)

Prices assume the datagen/SWE-SFT campaign has already delivered M_swe + twin@K1 (its budget is in
`swe_tuning_campaign_design §5`, not re-charged here). The re-conversion is **required by the certified loop regardless**,
so twin@Kc's train cost is the *marginal* re-weight of an already-budgeted 400-step conversion.

| stage | 5090 GPU-h | off-GPU docker | note |
|---|---:|---:|---|
| **L1-SWE census** (§6.1) | **~0.5** | ~0.3 h | content-mix + entropy on keepers + Tier1-C46 decode pass; may pre-KILL |
| **K1.5 decode-only** (twin@K1, γ-sweep ×3, Tier1-C46 ≈46 eps @ ~21 eps/GPU-h + anchor + retention) | **~3–4** | ~0.5 h | no retrain; cheapest rung |
| **K1.5/K2 trained** (twin@Kc re-conversion, 2 seeds ≈1.2 GPU-h + preservation battery ~1 + par-eval ~3) | **~5–7** | ~0.5 h | marginal train ≈ the plain re-conversion; +eval |
| **K4 trained** (curriculum-to-4 re-conversion 2 seeds + par-eval at k_max=4) | **~5–7** | ~0.5 h | ambitious rung |
| **per-rung docker scoring** (Tier1-C46 × 2 arms × ~0.6 min/eval) | — | ~1–2 h | official harness, parallelizable |
| **slack** (one re-seed / one re-sweep per rung, INCONCLUSIVE path) | +~3 | — | do not extend past 600 steps |
| **K-track total (through K4)** | **~20–28** | ~5–8 h | dominated by SWE par-eval occupancy (the twin's 21 eps/GPU-h) |

**Budget note (compounding with the directive's "~2–4 GPU-h class conversion"):** each *trained* rung's re-conversion is
the ~2–4 GPU-h class (Run-1 was `train_runtime=2068 s ≈ 0.57 GPU-h`/400 steps × 2 seeds + battery). The K-track's cost is
**mostly eval**, not train, because the SWE par-eval runs the slow (21 eps/GPU-h) diffusion twin over ~46 episodes ×
several K/γ configs. **The infra track (§10) directly attacks that eval cost** — B-P1 lifting the twin's eps/GPU-h is
what makes the later rungs affordable.

---

## 9. INFRA TRACK (runs alongside the K track; not gated on the §1.1 twin precondition)

Two engineering levers, both from the measured behavioral bar. **B-P1 is P0** (it attacks the twin's dominant
per-episode disadvantage and plausibly the loop-halt resolve gap); batch-occupancy is P1.

### 9.1 B-P1 — free-text decode policy: stop full-canvas re-denoise + EOS-aware canvas (P0)

**Evidence.** Free-text/reasoning per-forward is **25.8 ms vs 18.5 ms tool-call** even though both engage cudagraph,
because "pure block-diffusion free-gen **re-denoises the full canvas per committed token** with no FSM to
bulk-commit/prune" (`engine_build_status.md §0.G`); and emitted tok/fwd is **0.862 < 1.0** because "block-diffusion
**overshoots past EOS** and discards the tail" (§0.I). So free-text K=1 costs **29.9 ms/tok** (25.8/0.862) vs AR-cudagraph
10.72.
**Fix (two parts, both proven-in-mechanism by the CAD sampler):** (i) **EOS-aware canvas** — stop denoising the block the
instant EOS is committed; don't fill/discard the block tail (the CAD sampler's native-stop rule, productized into the
promoted engine); (ii) **frontier-only denoise** — denoise only the un-committed staged window, not the full 32-wide
canvas per committed token.
**Parity-gated:** must reproduce the K=1 byte-exact anchor (26/30 · 0.862→ now ≥1.0 emitted with the overshoot removed;
the *committed* tokens must be byte-identical, only the discarded-tail work is removed). Use the CAD R1 certificate as
the template.
**Effort: S–M** (mechanism proven; landing in-engine + the parity gate is the work).
**Expected recovery:** per-forward **25.8 → ~18.5 ms** (tool-call parity, ~1.4×) **and** emitted **0.862 → ~1.0**
tok/fwd (overshoot recovered) ⇒ free-text **29.9 → ~18.5 ms/tok (~1.6×)**, *before any K*. **Hypothesis to test, not
claim:** the EOS-aware canvas may also reduce the **loop-halt / empty-patch** signature (26/50 halts, 35/50 empty) if
those are partly "no clean terminal / overshoots the stop" — measure resolve@1 on Tier1-C46 before/after B-P1 as a
side-metric; a resolve lift here would be a behavioral bonus independent of K.

### 9.2 Batch-occupancy — recover the co-batch collapse (P1); two scoped options

**Evidence.** FLARE's **forced-sync scheduler** + per-request **variable draft widths** ⇒ effective batch **7.2/16 =
0.45 at b16**, util idles **84–88 %** (host-bound), head-of-line/straggler blocking; AR co-batches near-linearly at 100 %
(`engine_build_status.md §0.I`). This is why the twin is **21.4 eps/GPU-h vs AR 99.6** (4.65×) on `w2_n50`.

- **(A) Async-tolerant scheduling via the read-only snapshot machinery.** Forced-sync was mandated because the async
  scheduler produced an async-rollback divergence at pos-33 after a block boundary. The **read-only GDN snapshot**
  (state frozen during denoise, advanced once at commit, 6/6 bit-identical) is precisely the primitive that could make
  async safe: a per-request read-only snapshot means an async rollback cannot corrupt cross-request state.
  **Effort: L** (correctness-critical — must re-derive the pos-33 divergence and *prove* the snapshot closes it; a wrong
  fix silently corrupts). **Expected recovery:** toward AR's near-linear co-batch — effective batch **0.45 → ~0.7–0.9**,
  i.e. **~1.5–2× eps/GPU-h**. UNVERIFIED, correctness-gated.
- **(B) Width-bucketed co-batching.** Keep forced-sync; bucket requests by draft width so a synchronous wave co-batches
  members that commit the same #positions (no straggler *within* a bucket); re-bucket as widths drift. Scheduler-side
  only, does not touch the correctness-critical async path. **Effort: M.** **Expected recovery:** partial — recovers
  within-bucket straggler loss, not across-bucket; **0.45 → ~0.6–0.7**, **~1.3–1.5×**. **Honest caveat:** OPT-4-Part-1's
  variable-width work was *latency*-neutral (widths bucket back to the captured cudagraph bucket, `engine_optimization_plan
  §OPT-4`); this is a *throughput/occupancy* axis (different from latency), but the same bucketing physics bounds the
  ceiling — do not over-promise.

**Recommendation & sequencing:** **B-P1 first** (P0, unblocks the twin's per-episode cost and possibly resolve), then
**(B) width-bucket** (P1, moderate-risk occupancy), with **(A) async-tolerant** as a stretch (highest ceiling, correctness-
gated — only if (B) is insufficient and the eps/GPU-h cost is still the bottleneck). Both occupancy options help the
**batched eps/GPU-h** column of §11 but not the B=1 free-text ms/tok column (that is B-P1 + K).

---

## 10. Combined speed projection table (honest about compounding)

Baseline anchors (measured): free-text **K=1 = 29.9 ms/tok** (25.8 ms/fwd ÷ 0.862 emitted); batched diffusion **21.4
eps/GPU-h** at c=4 (AR 99.6). K applies **only to the parallelizable reasoning fraction**; blended via Amdahl at the
**estimated** `f_value = 0.30` (to be replaced by the §6.1 census — this is the dominant uncertainty). Factors are **NOT
cleanly multiplicative**: Amdahl caps K (`1/f_value ≈ 3.3×` blended ceiling at f_value 0.30), and occupancy vs
per-forward partly overlap. All non-baseline cells are **PROJECTED / UNVERIFIED**.

| configuration | free-text ms/tok (B=1) | vs AR-cudagraph 10.72 | batched eps/GPU-h (c=4) | vs AR 99.6 | notes |
|---|---:|---:|---:|---:|---|
| **twin@K1 today (baseline)** | 29.9 | 0.36× | 21.4 | 0.21× | measured (`§0.I`, `w2_n50`) |
| **+ B-P1 (infra-only)** | **~18.5** | 0.58× | **~28–32** | 0.30× | EOS-aware + frontier-denoise; ~1.6× ms/tok, ~1.3–1.5× eps |
| **+ B-P1 + batch-occ (B/A)** | ~18.5 | 0.58× | **~36–58** | 0.40–0.58× | width-bucket ~1.3× → async ~1.9× on eps only |
| **+ B-P1 + K1.5** | **~14.2** | 0.75× | ~35–42 | 0.40× | blended 1.30× on the 18.5 ms/tok floor |
| **+ B-P1 + K2** | **~12.0** | 0.89× | ~42–50 | 0.46× | blended 1.54× |
| **+ B-P1 + K4** | **~8.8** | **1.22×** | ~50–62 | 0.56× | blended 2.11× (optimistic edge; f_value-capped) |
| **+ B-P1 + batch-occ + K4** | ~8.8 | 1.22× | **~70–110** | 0.7–1.1× | the full stack, everything landing — UNVERIFIED |

**Reading this honestly:** (1) **B-P1 alone is the highest-confidence win** (mechanism proven by the CAD sampler) and
gets free-text from 0.36× to ~0.58× vs AR and eps from 0.21× to ~0.30×. (2) **K only reaches AR *latency* parity
(1.0×) around K4-blended**, and K4 is the rung the entropy wall most likely caps (§3) — so the realistic K contribution
is the K1.5–K2 band (0.75–0.89× vs AR), not parity. (3) **Batched eps/GPU-h is where the twin is furthest behind
(0.21×)** and where occupancy work matters most; only the full stack (B-P1 + async-occ + K4, all landing) projects
toward eps parity, and every factor there is unverified. (4) The factors **compound sub-linearly** — the free-text
ms/tok column already banks B-P1's per-forward gain, so K compounds on the *reduced* 18.5 ms floor, not the 29.9 ms
baseline; and the blended-K ceiling is hard-capped by code's f_value. **No single lever reaches AR parity; the honest
story is "B-P1 is a safe ~1.6× on free-text, K adds a further f_value-capped ~1.3–2.1× on reasoning if it engages at
all, occupancy is the batched-throughput lever, and none of it is a 5× story."**

---

## 11. Provenance + kill-gate summary (attach to the result report)

**Pre-registered KILLs:** KILL-D1 (leakage: any `Tier1_C46 ∩ {train, w2_n50, gate_ladder}` ≠ ∅) · PAR-KILL (SWE
net-loss ≥4 ∧ p<0.05 vs twin@K1) · KILL-anchor (tool-call matched-20 significant net-loss) · KILL-retention (GSM8K ≤
anchor−2 or KL>0.05 unrecovered) · KILL-3 (any value-projection counter nonzero) · KILL-0 (base/merge sanity:
mask_id≠248077 / bd_size≠32). **SPEED-FAIL** (K < rung target at par) is a *soft* stop, not a KILL — ship the last
passing rung with twin@K1 intact.
**Per-row provenance:** git commit; CAD sampler sha + the k=1 (temp-0 byte-exact) / temp-0.6 (distributional) certificate;
base+adapter paths; `k_raise_pool_manifest.json` sha + the three intersection asserts = 0; decode flags (γ, k_max);
value-projection audit JSON; the McNemar `(b,c)`+p at both seeds; the L1-SWE census `f_value`/entropy. Commit + push each
artifact to origin/main with narrated reasoning ([[qwen-diffusion-commit-workflow]]).
