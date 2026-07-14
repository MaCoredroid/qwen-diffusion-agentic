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

---

## STATUS(2026-07-10) — STEP 5 EXECUTED: Tier1-C46 twin@K1 ENTRY GATE — ENTRY BAR **NOT MET** (3/48 < 12)

Monitor-dispatched. Ran the §1.1 entry precondition: the SWE-SFT primary twin@K1 (M_swe_S = merged-RL-v2 +
SWE-SFT arm-1 + Run-1 two-stream re-conversion, decoded **K=1 hybrid_clean**, FSM values) over the **fresh
Tier1-C46** slice, 48 episodes through qwen-code on the FLARE vLLM engine, official swebench images + **official
`swebench.harness.run_evaluation` docker scoring**, frozen diffusion envelope (temp 0.6 / top_p 0.95 / top_k 20,
NO presence_penalty), turn cap 75, empty-patch re-drive retries=1, seed base 1234, c=4. **NO K>1 work** (K-curriculum
is a separate decision on this gate). Wall 12968 s (3.6 h), 13.33 eps/GPU-h. Heavy artifacts gitignored under
`runs/k_gate_c46/` (`report.json`/`report.md`/`build_report.py`, per-episode `diffusion/shard_*/verified/per_task/*/
runner_metadata.json`, merged `diffusion/predictions.jsonl`, official `diffusion/scoring/*.c46_twinK1.json`).

### PRIMARY — resolve@1 vs the ≥12 power floor
- **resolved = 3/48 (6.2 %)** — `resolved_ids = [django-11163, django-12193, django-13410]` (all django). **Entry
  floor = 12** (≈26 %, the level at which McNemar has power to detect a 3–4 resolve loss).
- **VERDICT = INCONCLUSIVE-BY-POWER — entry bar NOT met.** Per §1.1: a twin resolving 3/48 cannot power a paired
  par gate (you cannot lose resolves you never had), so **do NOT spend K rungs** on this base. This is **not a KILL**
  (reframe-1: a floored entry costs nothing — the K=1 twin is intact and ships as the deliverable of the certified
  loop; §4/step-4 anchor 50/63 + serving byte-cert already PASS). The K-track does **not** proceed; the SWE-SFT base
  must lift **edit-commitment** first (escalate per `runs/swe_datagen_s1/USER_LEVER_BELT`).

### Failure-mode breakdown vs the N=50 taxonomy (the question: did SWE-SFT move the loop-before-edit failure?)
Answer: it moved the *loop shape* but **not** the *edit-commitment outcome*. SWE-SFT converted "loop forever then
halt empty" into "cleanly quit early, still empty" — the empty-patch rate is essentially unchanged.

| signature | twin@K1 C46 (N=48) | pre-SFT RL-v2 w2 (N=50) | read |
|---|---:|---:|---|
| resolved | **3** (6.2 %) | 2 (4 %) | +1 — within noise, NOT off the floor |
| empty patches | **36** (75 %) | 35 (70 %) | **unchanged** — the gating deficit |
| edit committed (non-empty) | 12 | 15 | slightly fewer episodes edit at all |
| resolved-given-edit | **3/12 (25 %)** | 2/15 (13 %) | when it DOES edit, decent conditional resolve |
| loop-halts (exit 1) | **12** | 26 | halved — fewer pathological loops |
| clean exit-0 | **36** | 10 | **3.6× more** — model now cleanly "declares done" |
| clean-exit-0 → **no patch** | **29** | — | the new dominant failure: quit-before-edit |
| median turns | **9** | 25 | model gives up far earlier (not looping) |

Mechanism: 29/48 (60 %) episodes exit cleanly (exit 0) convinced they are done **without ever committing an edit**;
+7/48 loop-halt empty = 36/48 empty. SWE-SFT taught the model to **stop looping** but not to **commit an edit**. The
loop-before-edit failure became **give-up-before-edit**. Edit-commitment, not loop-halting, is the deficit that gates
SWE resolve — and it is what the K-track has no lever over (K raises decode speed, not edit propensity).

### Kill-gate + audit summary
- **KILL-D1 (leakage) — CLEAN.** `data/swe_kraise_c46_pool/pool_manifest.json` `KILL_D1_PASS=true`; all asserts pass
  (`Tier1_C46 ∩ {train, w2_n50, gate_ladder_5} = ∅`, `Tier1_C46 ⊂ Tier1-100`, no duplicates). pool_sha256
  `49d8f46d…`. n=48 (Tier1-100 ∖ (w2_n50 50 ∪ gate_ladder_5), the pristine complement).
- **KILL-3 (value-projection tripwire) — NOT CLEAN, but non-verdict-bearing.** The served engine logged
  `projected_value_tokens_exact` (== `value_projection_events`, "value tokens the grammar OVERWROTE, MUST be 0")
  nonzero on **4/1169 requests (0.34 %)** — values [10,1,1,1], 13 events total. This is **projection-immune for the
  docker-scored resolve@1 PRIMARY** (a phantom value token still yields a real patch that real tests adjudicate — no
  phantom-resolve mechanism exists on a behavioral gate), so the 3/48 verdict is uncontaminated. It **is** a
  served-engine correctness note that WOULD contaminate a K-track tok/fwd/exact-args measurement — flagged for repair
  before any adaptive-K rung (moot while the entry bar is unmet). Step-4's anchor/serving certs (83 turns) were 0/0;
  this larger 1169-request SWE surface exposed the 4-request edge.
- Scoring integrity: `error_instances=0`, `completed=12 / unresolved=9 / resolved=3 / empty=36 = 48`. Serving health:
  decode_mode=hybrid_clean True, FLARE gate True, mask 248077 present True, 1169 hybrid_clean reqs (K=1, ~0.95
  tok/denoise-step confirmed). Spot-checks: resolved django-11163 = genuine 570-byte diff to `forms/models.py`;
  empty django-11749 = genuine loop-halt-no-patch even after the re-drive (not a harness extraction bug).

**Bottom line (honest):** the SWE-SFT+conversion twin@K1 is byte-serving-certified and a strict superset of the AR
arm on the tool-call anchor, but end-to-end it resolves only **3/48** on the pristine Tier1-C46 slice — **below the
≥12 entry floor**. SWE-SFT did not close the empty-patch / edit-commitment gap; it reshaped the loop failure without
producing edits. **The K-raise campaign is gated OUT at entry** (INCONCLUSIVE-BY-POWER, not a KILL); the next lever is
the SWE-SFT/datagen edit-commitment escalation (USER_LEVER_BELT), not K.

## STATUS(2026-07-10) — STEP 5 FOLLOW-UP: DEFICIT-LOCUS AR-mode PAIRED read — verdict **MIXED (B-bound)**

Ran the **same 48 Tier1-C46 instances** through the **identical SWE-SFT weights served AR** (`models/qwen3.5-9b-
fastdllm-mswe-S-vllm-bf16` on the stage-c/w2_n50 AR path `runcage_ar.sh`: stock vLLM 0.23, gmu 0.85, seqs 4, ml 32768,
qwen3_xml tools, qwen3 reasoning), mirroring the twin gate **byte-for-byte except the decode paradigm** (same shard
plan, envelope temp 0.6/top_p 0.95/top_k 20 no-pp, re-drive=1, turn cap 75, seed base 1234, c=4, official docker
scoring). AR serving verified **pure AR** (0 FLARE/decode lines). Wall 1454 s, **118.84 eps/GPU-h (8.9× the twin's
13.33)**. Artifacts: `runs/k_gate_c46/AR_PAIRED_READ.md` + `ar_paired_report.json` + `ar/scoring/*.c46_ar.json` +
per-episode `ar/shard_*/verified/per_task/*/runner_metadata.json`.

### PRIMARY — paired resolve@1 (twin diffusion vs AR), McNemar exact
- **AR 7/48 (14.6 %) vs twin 3/48 (6.2 %)** — both=3, twin-only b=0, AR-only c=4, net −4, **McNemar exact p=0.125
  (NOT significant)**. Twin-resolved ⊂ AR-resolved (**b=0 → the diffusion decode never wins a task AR loses**).
- AR-only ids: django-16801, matplotlib-25122, sympy-13647, sympy-23262. Both: django-{11163,12193,13410}.

### The decisive covariate — edit-commitment is decode-mode-specific
| signature | AR (SWE-SFT) | twin@K1 (SWE-SFT) | read |
|---|---:|---:|---|
| resolved | **7** (14.6 %) | 3 (6.2 %) | +4, not significant (p=0.125) |
| edit committed (non-empty) | **46** (96 %) | 12 (25 %) | **A: decode collapses committal** |
| empty patches | **2** | 36 | same weights, opposite behavior |
| clean-quit → no patch | **≈1** | 29 | the twin's give-up-empty is a decode artifact |
| median turns | 14 | 9 | AR persists; twin gives up early |

### VERDICT = **MIXED, and the resolve gate is B-bound**
Two **stacked** deficits: (A) a large **decode-mode-specific edit-commitment collapse** — identical weights commit
46/48 edits AR-decoded vs 12/48 diffusion-decoded, so the twin's 75 %-empty failure is a FLARE-K=1 decode artifact,
**not** the SFT data; and (B) a **binding SFT-capability ceiling** — even with full committal, AR resolves only 7/48
(≪12 floor; 39/46 AR edits are wrong; astropy/sphinx/scikit/xarray/pytest/pylint/requests = 0 in BOTH arms). Because
PRIMARY resolve@1 is AR≈twin (both sub-floor, p=0.125), **the campaign blocker is capability/data (B), not decode.**
The decode deficit (A) is real and would roughly double the twin toward its AR twin, but that ceiling is 7/48 — still
sub-floor. In this SWE regime the K=1 twin is **both weaker and 8.9× slower** than AR (its only rationale, K>1 speed,
is unavailable until entry clears).

### RANKED LEVERS
1. **PRIMARY — lift the capability ceiling (B):** data scale-up (USER_LEVER_BELT / Opus tranche-2) + longer-seq
   training + trajectory-shape/front-truncation fix. Binding constraint; lifts BOTH decode modes; highest EV.
2. **SECONDARY (diffusion-track only) — recover committal (A):** decode policy on edit spans, two-stream SFT at
   8192-with-packing, K-curriculum on commitment. Necessary for a diffusion twin to match its AR self, **insufficient
   alone** (still sub-floor); do with/after lever 1, only if the diffusion serving path is still wanted.
3. **Re-run the entry gate after the ceiling lifts.** K stays gated-OUT at entry (McNemar powerless < 12).

**AR-vs-stock-N=50 (marginal only):** AR-SFT 7/48 (14.6 %) vs stock-AR 19/50 (38.0 %); **C46 ∩ N50 = ∅ (disjoint)** +
stock-vs-SFT weights — two confounds, no apples-to-apples claim.

## STATUS(2026-07-10) — STEP 5 FORENSICS: committal-collapse mechanism NAMED — **context-window exhaustion, not a quit**

Executed the **Forensics** phase (CPU-only, no GPU, no engine change) of the K1-committal method: turn-by-turn paired
diff of `qwen_trace.json` + `proxy.log` across the frozen C46 pool (N=48, identical SWE-SFT weights, AR-native vs FLARE
hybrid_clean K=1). Full write-up + scripts: `runs/k_gate_c46/K1_COMMITTAL_ANALYSIS.md`.

**Headline — the "clean quit" label was wrong.** The twin does not voluntarily stop and does not emit an early
EOS/end-of-turn instead of an edit. Terminal-cause tally over 48 twin episodes: **36/48 = CTX_OVERFLOW_400 at the frozen
32768 wall**, 12/48 loop-halt (exit 1), **0/48 voluntary quits**. The "36 empty patches" == the 36 context deaths. AR:
39/48 completed-with-edit, only 1/48 terminal overflow. Every death carries the byte-identical signature
`prompt at least 32516 + 253 output = 32769 = cap+1`.

**Proximate cause — argument-under-grounding on `read_file`.** Same weights, tools, envelope, harness; the twin fills
the window faster because it drops the `limit` argument. Population read-arg grounding: **AR emits `limit` on 243/348
(69%)** reads; **twin on only 38/251 (15%)** → **84% of twin reads run to EOF**, adding +5–7k tokens/turn and saturating
32768 in a **median 7.5 turns** (min 3) — before localize+edit. Cleanest paired frame (matplotlib-25122, AR-resolved):
turn-2 same file, **AR `read(off=410, limit=50)`** → survives to edit@T10 (resolved); **twin `read(off=423, no-limit)`**
→ 565-line EOF read → 32,516 → 400. This is the 0.238-top-1 argument-grounding crux made concrete: the twin won't
commit the numeric `limit`/`offset` it can't ground, so it defaults to the unbounded read.

**Terminal trigger + AR's escape.** The qwen-code proxy handles overflow by halving `max_tokens` (8127→…→253) and the
last rung lands one token over the cap (off-by-one). AR survives the same overflow because its windowed reads keep the
prompt low enough for the ladder to fit **and** qwen-code client-side history-compression fires near the cap
(AR input_tokens drops 32,263→21,461 mid-episode); the twin's prompt is already pinned at 32,516, above either escape.

**Hypothesis tally (vs the pre-registered H-space):** H1 (early EOS) **REJECTED 0/48**; H3 (thinking-exit) **REJECTED**;
H4 (harness reads output as completion) **REJECTED** (exit-0 is error-surfacing after the retry ladder, not a completion
read); H2 **REJECTED as literally stated but CONFIRMED in spirit** — the twin under-grounds the *arguments*, not the tool
call; **NEW dominant mechanism H5 = context exhaustion via arg-under-grounding × 32768 envelope, 36/48.**

**What this changes.** The **edit-commitment collapse (A) is real but was mislabelled** as a decode-time EOS/termination
bias. It is an infra/argument-grounding artifact, so the implied fix is **not** an EOS-penalty/min-turn decode lever.
The **B-bound capability-ceiling verdict is UNCHANGED** (AR commits 46/48 yet resolves 7/48; 39/46 AR edits wrong) — this
forensic does **not** reopen the entry gate. But it reprioritises the cheap diagnostic levers, ranked smallest-first for
the REPRO+CANDIDATE steps (not run here): (1) **fix the retry-ladder off-by-one + raise `max_model_len`** (arm-neutral,
zero model change — highest-EV, purely diagnostic); (2) **server-side read-window clamp / decode prior on `limit`**;
(3) **argument-grounding retrain** (the durable fix, folds into the §B trajectory-shape lever). Guardrail for CANDIDATE:
recovered committal that emits *wrong* patches is not a win — resolve@1 is the truth; expect levers 1–2 to reveal the
twin's true capability rather than lift resolve materially.

## STATUS(2026-07-10) — STEP 5 REPRO (GPU): committal collapse **causal-isolated to the diffusion decode**

Executed the **REPRO** phase (GPU): served the twin on the frozen C46 config (FLARE `hybrid_clean` K=1, mask 248077,
`max_model_len` 32768, gmu 0.74) and replayed **5 byte-exact divergence-point proxy dumps** (the requests that produced
the fatal unbounded read) at the frozen envelope (temp 0.6 / top_p 0.95 / top_k 20), `N=64`/prompt, + a temperature sweep,
a temp-0 determinism probe, and a **same-weights AR-decode control** (`careful_live_grammar`, `flare=0`). One GPU tenant,
server torn down (GPU→baseline). Full write-up + raw JSON: `runs/k_gate_c46/K1_COMMITTAL_ANALYSIS.md` (REPRO section).

**Headline — flip ONLY the decode paradigm, same weights/prompts/envelope:**

| pooled (5 prompts × N=64) | read_file | reads WITH `limit` | **unbounded reads** | EOS-quit |
|---|--:|--:|--:|--:|
| twin — hybrid_clean (diffusion) | 320/320 | 34/320 (10.6%) | **286/320 = 89.4%** | **0** |
| same weights — AR decode | 306/320 | 306/306 (100%) | **0/306 = 0.0%** | **0** |

The diffusion decode drops the `read_file` window args on **89%** of reads (corroborates the forensic pop. 84%); the
**identical weights AR-decoded drop them on 0%** and never overflow. Two sub-modes reproduced: offset-without-limit
(matplotlib-25122, 39/64) and whole-file (sympy/django-12273/matplotlib-20859, 64/64). When `limit` *is* emitted it is a
tight correct window; the greedy modal call grounds `offset=410,limit=30` — the same window AR uses. Divergence prompts
already sit at **28.9–30.2k / 32768** (residue of earlier unbounded reads), so one dropped arg overflows.

**H1/H3/H4 dead at the engine level (N>800).** read_file on 320/320 replays, 0 NO_TOOL/EOS; engine `stop_reason` over 818
`hybrid_clean` decodes = **806 complete_tool_call, 4 max_new_tokens, 0 EOS** (no forced `tool_choice` — the model was free
to quit and never did). **Temperature is not the lever:** unbounded rate ≈0.83–0.89 flat across temp {0,0.2,0.4,0.6}
(0.89 at greedy), and temp-0 "greedy" is **non-deterministic** on this path (drops `limit` 11/16 even at temp 0).

**Two serving-path findings:** (i) `hybrid_clean` does **not** expose per-token logprobs (HTTP 500 `list index out of
range` on output-logprobs and `prompt_logprobs`) — the per-position EOS-vs-toolcall logprob probe is unavailable on the
diffusion path; substituted engine `stop_reason` telemetry + resampling frequency. (ii) `hybrid_clean` is non-deterministic
at temp 0 (byte-parity cert regime not turn-reproducible on the served path).

**Verdict.** The empty-patch collapse (locus **A**) is a **diffusion-decode argument-under-grounding**, causal-isolated:
same weights AR-decoded ground the window 100%. It is **decode/conversion-side, not weights (B) and not temperature** — so
the smallest CANDIDATE levers are (1) **server-side read-window clamp when `limit` absent** (arm-neutral, zero model change,
directly kills the 89%), (2) retry-ladder off-by-one fix + raise `max_model_len` (both arms, diagnostic), (3) durable:
argument-grounding in the diffusion conversion / K-curriculum value guard, or AR-decode fallback for value spans. **Guardrail
unchanged:** committal recovery is B-ceiling-bounded (AR commits 46/48, resolves 7/48; 39/46 wrong) — REPRO explains/isolates
A, it does **not** lift resolve or reopen the entry gate. No promotion; the monitor takes the candidate to the cert path.

## STATUS(2026-07-11) — STEP 5 CANDIDATE (CPU/docker): read-window clamp — mechanism **CONFIRMED**, fix **PARTIAL**

Executed the **CANDIDATE** phase (CPU/docker only, no GPU, no engine change, no server booted for scoring). The 8
divergence instances were replayed through the twin with the experimental **read-window clamp shim** active
(`proxy_readclamp.py`, `LUMO_PROXY_READCLAMP_LIMIT=100` — inject `limit=100` on any `limit`-less `read_file`;
arm-neutral, zero model change) at the frozen C46 envelope + per-shard seeds. Clamp fired **50×**. Scored via the
**OFFICIAL swebench 4.1.0 harness** (all 8 ids are SWE-Verified → the `datagen_score.sh` official path, byte-identical
to the gate's `score_all.sh`; no gym-source id → SWE-Bench-Fork not exercised). **Independently re-scored → identical.**
Full write-up + per-instance table: `runs/k_gate_c46/K1_COMMITTAL_ANALYSIS.md` (CANDIDATE section).

**Delta vs the SAME 8 gate results (gate = 0/8 committal, 8/8 CTX_OVERFLOW, 0/8 resolve):**

| metric | gate (twin@K1) | candidate (clamp) | delta |
|---|--:|--:|--:|
| committal (non-empty scored patch) | 0/8 | **3/8** | +3 |
| CTX_OVERFLOW terminal (32768 wall) | 8/8 | **4/8** | −4 |
| **RESOLVE@1 (truth)** | 0/8 | **1/8** | +1 |

Committals: matplotlib-25122 (**RESOLVED** ✅), django-16256 (edited then hit wall, unresolved), matplotlib-20859
(unresolved). Candidate terminal tally: 4 CTX / 2 error_during_execution / 2 agent_gave_up(timeout) vs gate's 8/8 CTX.

**Verdict — mechanism CONFIRMED-causal, fix PARTIAL.** Neutralising the proximate cause (unbounded reads) turned the
gate's 0-committal/8-CTX-death cohort into 3 committals, 4 CTX deaths, 1 resolve — **direct causal confirmation of the
H5 read-arg-under-grounding → CTX-exhaustion mechanism** (bound the reads and episodes survive to commit). But PARTIAL:
(i) **4/8 still die of CTX** (clamp cuts per-read growth but many clamped reads still accrete to the cap; every death
still shows the `32516/253/32769` cap+1 signature); (ii) the shim **introduced new failure modes** — 2 exec-errors + 2
upstream **timeouts** — so 8→4 is not all clean recovery; (iii) **resolve barely moves and stays below AR** (AR resolves
**3/8** on these same ids — matplotlib-25122, sympy-13647, sympy-23262 — the clamp recovers only 1; the other 2
committals are committed-but-wrong). The **B-ceiling guardrail holds exactly as predicted**: lifting committal (A) lifts
resolve only marginally and cannot exceed the SFT ceiling. The clamp is a valid **diagnostic** (lets the twin's real
capability show instead of dying at the wall) and a serving-layer floor — **not** a resolve fix, and it does **not**
reopen the entry gate.

**Promotion recommendation.** Promote off the experimental shim onto the **standard cert path**, NOT straight to
production (it mutates tool-call args in the loop): **(1) matched-20 anchor** (§2.3, McNemar net-loss vs frozen anchor
not significant, with the clamp in the loop) **+ (2) A6 online==offline spot-cert** (served==offline under the shim).
Both PASS → production. Arm-neutral serving-side intervention ([[diffusion-promotion-discipline]]); the **durable** fix
stays argument-grounding in the conversion / K-curriculum value guard ([[retrain-freely-rule]]).

**SEPARATE MUST-FIX (restated, independent of the clamp): harness retry-ladder off-by-one.** The candidate run proves
it is still live — all 4 CTX deaths carry the byte-identical **`253 out / 32516 prompt / 32769 = cap+1`** signature.
Ladder bottoms at `max_tokens=253` while prompt is pinned at 32,516 and the harness prompt estimate reads 1 low, so the
final retry lands deterministically one token over. Fix both arms: floor the last rung at `cap − prompt − margin` (clamp
`max_tokens` so `prompt+max_tokens ≤ cap`) and correct the estimate. Orthogonal to the clamp (clamp reduces *how often*
the cap is reached; this fixes *what happens when it is*). Highest-EV, zero model change, symmetric.

### STEP 5 HARNESS TRUTH-TELLING — DONE (2026-07-12; labeling only, ladder behavior untouched)

Landed the *truth-telling half* of the retry-ladder finding (the *behavioral* off-by-one fix above stays open): a
cap-death (context-overflow retry ladder exhausts → vLLM's terminal 400 surfaced as the episode `result`) now produces a
**distinguishable, env-limited terminal record** instead of masquerading as an honest empty-patch miss / clean exit-0
quit — the exact mislabel the AR_PAIRED read flagged (29 "clean quits" that were really 36 cap-deaths). Wired end-to-end:

- **Driver** `scripts/run_swe_bench_qwen_code.py`: new `_classify_terminal_cause` writes
  `runner_metadata.terminal_cause="ctx_overflow"` from the terminal `[API Error: 400 … maximum context length …]`
  payload (API-error framing required; newest-attempt-first so an empty re-drive falls back to the real terminal —
  the 6/41 empty-retry C46 cases). Both host + container orchestration paths.
- **Ledger** `runs/swe_datagen_s1/ledger.py`: `record` scans per-task `runner_metadata` (`_ctx_overflow_ids`) and
  `_classify` reroutes an empty-patch cap-death `empty_patch → env_limited` (new verdict, in `REAL_VERDICTS` so
  rolling-yield + best-of-k coverage are byte-identical to the old empty_patch accounting — **only the lying label
  changes**), stamping `terminal_cause` on the attempts row.
- **Gate report builders** `runs/k_gate_c46/build_report.py` + `build_candidate_report.py`: consume the tag → a distinct
  `ctx_overflow_deaths` (env-limited) bucket, kept OUT of `clean_exit0` / `empty_patches (honest miss)`.
- **Untouched:** retry ladder (`build_context_retry_body`), clamp shim (`proxy_readclamp.py`), all decode. Labeling only.
- **Unit test** `scripts/test_terminal_cause_classification.py` (13 cases, green): mock cap-death record
  (32516/253/32769) → driver `ctx_overflow`, ledger `env_limited` (not empty_patch), scoreable outcomes unaffected,
  empty-retry fallback, and the no-retroactive-surgery path.

**Prospective only — no retroactive surgery.** The C46 gate arm (36 cap-deaths) and epoch-2 datagen batches predate the
tag → `_ctx_overflow_ids` returns ∅ → they keep the historical `empty_patch`/`clean_exit0` labels (re-running
`build_report.py` on the frozen arm still shows `ctx_overflow_deaths=0`, `empty_patches=36`). Detail + caveat banked in
`runs/k_gate_c46/K1_COMMITTAL_ANALYSIS.md` § "Harness TRUTH-TELLING fix — DONE".

---

## DIRECTIVE(2026-07-12) — END GOAL PINNED BY USER: K=5–10 AT THE GOLDEN NUMBER (stock-AR 19/50)

User directive (this session, verbatim intent): *after SWE-SFT, the end goal is to raise K to 5 or 10 (large
diffusion blocks) while MATCHING the pre-SFT anchor — stock-AR 19/50 vs twin 2/50 on N=50 SWE-Verified — as the
golden number; drive via SFT or post-training; no leakage.*

What this pins / changes vs the 2026-07-06 design above:

1. **GOLDEN NUMBER pinned.** The §2 behavioral parity bar is now anchored to the banked stock-AR **19/50** on the
   frozen `w2_n50` 50-id pool (f33fb6b run of record). Gate at EVERY rung: twin@K on the SAME frozen pool, frozen
   envelope (temp 0.6 / top_p 0.95 / top_k 20, NO presence penalty — the FLARE fragility rule), official docker
   scoring, ctx-overflow truth-telling labels active. PASS = not statistically below 19/50 (McNemar paired vs the
   banked stock-AR per-instance verdicts; α per §2). K is NEVER bought with quality below the golden number.
2. **LADDER EXTENDED.** §6 staged targets 1.5 → 2 → 4 now continue **→ 6 → 8–10** avg committed tok/fwd (engine
   counters, episode-weighted, measured over the gate run itself). Rungs >4 are new territory beyond the original
   design; same dual-exit kill discipline per rung (ship the last passing rung, stop clean, K=1 twin never at risk).
3. **TRAINING LEVERS licensed (user: "sft or post train"):** (a) in-conversion K-consistency `L_diff` (§4,
   unchanged, first lever); (b) SWE-content K-consistency SFT on keeper trajectories (train-side only); (c)
   **post-train on-policy RL** (S4-style speed-reward gated on resolve, flywheel-pattern harness) as the post-SFT
   lever if in-conversion stalls — design to be appended (see DESIGN-EXT task), honest about the caveat that
   diffusion-twin rollouts must serve on OUR engine (the flywheel GB10 host serves AR-only).
4. **LEAKAGE unchanged (user: "no leak"):** §7 firewall stands — 113-id eval holdout hash-asserted (KILL-D1);
   `w2_n50` ⊂ holdout is EVAL-ONLY forever; all training data from SWE-Gym + Verified-train-adjacent per
   USER_LEVER_BELT; per-tranche zero-overlap proof required before any promote.
5. **SEQUENCING unchanged:** iteration-2 (Opus tranche-2 datagen → windowed retrain → clamp cert → C46 re-gate,
   user-funded ~$230, IN FLIGHT) remains the entry precondition — the K ladder starts from the iteration-2 M_swe.

**Honest odds (registered now, before results):** the QUALITY half (twin 2/50 → 19/50-matched at K=1) is the
harder, unproven half — the AR-SFT arm itself sits at 7/48 on C46 and is untested on w2_n50; no measured lever yet
moves the capability ceiling except data scale/shape (iteration-2's bet). The SPEED half above K≈2–4 contends with
the measured entropy wall (S2 kill: 1.053 tok/fwd at token-exact bar; 0.238 top-1 conditional) — the behavioral
gate + γ relaxation + in-conversion training is the designed escape and is UNPROVEN above K=4. Code/edit content is
the most parallel-friendly content class in the dLLM literature, which is the honest reason to attempt the >4 rungs
at all. Each rung ships-or-stops-clean; a full-ladder miss still banks the best passing (K, 19/50-matched) twin.

---

# DESIGN-EXT(2026-07-12) — the three sections the DIRECTIVE commissioned

Appended to answer DIRECTIVE(2026-07-12) points 2 (ladder → 6 → 8–10), 3c (post-train on-policy speed-RL), and
1 (the golden-number gate as a stats spec). Same voice, same evidence discipline, same pre-registered-KILL structure
as §§0–11. **Nothing above this line changed.** These sections DO NOT relax the §1.1 entry precondition: the extended
ladder and the RL lever start from the **iteration-2 M_swe** (task #124, IN FLIGHT) once a twin@K1 clears the golden
number at K=1 — a K>1 rung is never spent on a floored twin (the campaign is gated OUT at entry today, 3/48 on C46).

## A. RUNGS ABOVE 4 — K6 and K8–10 (the entropy wall, re-derived for copy-heavy edit content)

### A.0 The reframe that makes 6–10 *arithmetically* possible: `f_blocked`, not `f_value`

The whole doc above (§6.1, §10, `l1_content_mix_result.md` §4) prices the blended ceiling at **`1/f_value`** — the
must-be-exact fraction. At the code estimate `f_value ≈ 0.25–0.40` that ceiling is **2.5–4×**, and **the directive's
6–10 avg committed tok/fwd sits ABOVE it.** Stating that head-on: *if the speed limiter is the must-be-exact fraction,
K6–K10 blended is unreachable on code and Section A is dead on arrival.* The only honest path to 6–10 is that the
limiter is **not** `f_value`. It is **`f_blocked`** — the fraction of positions the entropy gate actually *stops* (a
sub-γ / thin-top-2 position that forces K=1, §5). The two diverge exactly on **copy spans**:

- **GSM8K (why S2 walled at 1.05):** value tokens are **DERIVED** — `7×8=56`, the running sum, the `#### answer`. Derived
  ⇒ genuinely uncertain ⇒ top-1 conditional **0.238** (`s2_pilot_result.md`) ⇒ the gate blocks them ⇒ `f_blocked ≈ f_value`
  ⇒ K averages to ≈1. There is no gap to exploit.
- **SWE edit/diff (the bet):** a large fraction of must-be-exact tokens are **COPIED, not derived** — re-emitted verbatim
  from the file already in the prompt: unified-diff **context lines** (the 3 unchanged lines the format mandates around
  every hunk), the re-stated function signature / class body being edited, identifiers and paths that already appear in
  the retrieved source. A copied token is must-be-exact **and** low-entropy (top-1 → 1.0 because it is a literal copy),
  so the gate **does not block it** — it commits in the parallel run. On copy-heavy content **`f_blocked ≪ f_value`**, and
  the ceiling `1/f_blocked` can plausibly reach 6–10. This is the `C≈0` copy-span class ParallelBench certifies
  parallel-safe (`training_redesign_10x_research`, cited §3), instantiated on the one content class that is *made of* copies.

**So Section A's central, testable, UNPROVEN claim:** SWE edit content decomposes into (a) a small **derived-reasoning +
derived-value** core that stays K≈1 (the 0.238 wall, untouched), and (b) a large **copy + structural-boilerplate** mass
that is parallel-committable at high K — and (b) is large enough that the **blended** average clears 6, maybe 10. **The
§6.1 L1-SWE census is now the load-bearing pre-measurement:** it must report `f_blocked` (fraction gate-stopped on real
keeper edit trajectories), not just `f_value`. **If the census shows `f_blocked ≳ 0.15` on edit content, K6 is
pre-KILLed on evidence before a rung is spent** (ceiling < 6.7). This is the cheapest possible way to be wrong.

### A.1 (i) The training signal that could push committed-run length past the wall — on edit content specifically

The §4 objective (O1 frontier-adjacency + O2 span-class weighting, O3 curriculum) generalizes cleanly, with **one new
span class** the code setting demands:

- **COPY-SPAN class (new).** Extend the Run-1 span tagger (already tags VALUE for `VALUE_SPAN_LOSS_WEIGHT=2.0`) with a
  **copy detector**: a target token is COPY iff it is verbatim-alignable to a token in the retrieved context / the file
  span being edited (a decode-time-verifiable predicate, A.3). Put COPY spans in the **high-joint-commit** class with
  connectives; keep **DERIVED** values (computed literals, brand-new identifiers not in context, line numbers) in the
  **K=1** class. The trained propensity is literally *"parallelize copies and connectives, serialize what you must
  invent."* This is the specific new signal that GSM8K could not offer (arithmetic has no copy mass) and edit diffs
  offer in bulk.
- **What edit content actually gives the copy+structural class (cite the content, per the directive):**
  - **Unified-diff boilerplate** — `@@ -a,b +c,d @@` hunk headers, the ` `/`-`/`+ line-prefix column, closing
    brackets/parens, `EOF`/newline runs: deterministic given the language and the chosen hunk; near-zero entropy.
  - **Indentation / opener runs** — Python 4-space blocks, `def …(self` / `return ` / `self.` / `import ` openers: the
    low-entropy connective mass §3 already credited, denser on code than on GSM8K prose.
  - **Copy-from-context spans** — the diff's mandatory unchanged **context lines** and any re-emitted signature/identifier:
    verbatim in the prompt ⇒ top-1 ≈ 1.0 ⇒ committed in the run. This is the mass that does not exist in arithmetic.
  - **Edit-diff repetition structure** — a rename/refactor re-emits the same token many times; the `-`(old) and `+`(new)
    halves of a hunk share most of their tokens, so once the model commits the old line the new line is largely a copy of it.
- **Directive lever (b) SWE-content K-consistency SFT** rides the same signal on `runs/swe_datagen_s1/keepers` edit
  trajectories (train-side only, §7 firewall): a short K-consistency pass that presents the denoiser with contiguous
  copy+structural frontiers from real patches. Marginal cost ≈ the plain conversion (§8); **promotion-gated** — credited
  only if it beats decode-only-on-twin@K1 at the golden number ([[diffusion-promotion-discipline]]).
- **The honest adverse prior, restated for high K:** the derived core does **not** move (0.238 is a base-distribution
  property, §3). So the *reasoning-span* tok/fwd on the derived fraction stays ≈1 at every rung; **all** of K6–K10's
  headroom is bought from the copy+structural mass. If that mass is thinner than the census hopes, the rungs SPEED-FAIL
  and stop clean — they cannot manufacture parallelism the content does not contain (the S2 lesson, §3 last bullet).

### A.2 (ii) Block-size implications — canvas, the 12288 SFT block, the 32k window, APC

- **Canvas (bd) vs commit width K.** The commit run is clipped `[1, k_max]` inside the denoise canvas (`bd_size=32`).
  K6 needs `k_max=6`; K8–10 needs `k_max=10`. Geometrically **bd=32 already supports K≤10 with ~3× headroom** (a staged
  trailing-mask window of ≥ 2·k_max keeps the contiguous-run clip honest) — **no bd change is required for the numeric K
  target.** "Large diffusion blocks" (the directive's phrase) as a *bd bump* (bd 64/128 — wider parallel canvas, more
  copy-span opportunity per forward) is a **separate, optional re-conversion lever** at a different bd, [[retrain-freely-rule]];
  it is NOT on the K6/K8 critical path and is only pulled if k_max=10 within bd=32 SPEED-FAILs for lack of canvas.
- **The 12288 is the SFT AR block, not the conversion block.** `swe_tuning_campaign_design` AMENDMENT-B measured
  `block_size=12288` for the **single-stream SWE-SFT** (the 32768/24576 OOM on the fla `chunk_fwd_o`; 12288 = 24.8 GiB
  peak, ~6.5 margin). The **two-stream re-conversion** that trains twin@Kc runs at **block 512 / bd 32** (§4). So K6–K10
  live inside a 32-token canvas trained under a 512 block — **the 12288 memory wall does not bind the K rungs at all.**
  The 12288 front-truncation (69.88 % assistant-label retention) is an SFT-capability concern (it is why §1.1 gates on a
  non-floor twin), orthogonal to K.
- **32k serving window — the interaction that matters, and does NOT rescue it.** Higher K reaches EOS in fewer forwards,
  but it does **not** reduce prompt growth — and prompt growth is the measured killer: 36/48 C46 twin deaths are
  **CTX_OVERFLOW_400** at the 32768 wall from `read_file` **arg-under-grounding** (84 % forensic / 89.4 % REPRO unbounded
  reads, `K1_COMMITTAL_ANALYSIS.md`). **K speeds the decode, not the context economy** — a K10 twin that still drops
  `limit` still dies at 32,516 in a median 7.5 turns. **Therefore every rung >4 is co-gated on the read-window clamp cert
  (task #128) being live**; without it the golden-number gate is CTX-bound, not K-bound, and would mislabel a clamp
  failure as a K failure. (This is also why the §5 value-block must, if anything, get *tighter* at high K — A.3.)
- **APC (lossless prefix cache, `lossless_apc_design.md`) interaction.** The APC keys cached KV to the **committed
  prefix boundary**. Higher-K advances that boundary in larger jumps but the boundary is still a commit point, and the
  **never-remask** rule (§5, GDN FR13 discipline) guarantees committed = final, so the cache stays **lossless** at any K —
  no re-validation of a K10 commit is possible-or-needed because a committed run is immutable. The one caveat: a wrongly
  committed high-K copy is now *cached* and propagates; the copy-assert (A.3) is what keeps that from happening.

### A.3 (iii) Value-span protection at high K — the §5 rule must SURVIVE and TIGHTEN

The §5 value-blocked rule (values always K=1) is **non-negotiable and gets a third layer for K>4**, because the
K1-committal forensics prove the exact failure the naive high-K commit would amplify: **the twin is *confidently wrong*
on value args — it drops `limit`/`offset` at 84–89 % and never grounds them** (`K1_COMMITTAL_ANALYSIS.md`,
`runs/k_gate_c46`). A blind high-K run would parallel-commit those wrong-or-missing args faster. So:

1. **Tool-call args:** FSM-forced K=1, adaptive-K OFF on that path — **unchanged** (§2.3, §5). K is raised on free-text
   connective/copy spans ONLY; **args stay K=1 by construction.**
2. **Free-text derived values** (computed literals, line numbers, new identifiers not in context): the §5 contiguous-prefix
   entropy block + the top-2-margin "confidently-wrong-identifier" guard — **unchanged**, and now **load-bearing at K10**
   (one thin-margin position halts the whole run, capping the blast radius of a wrong commit).
3. **NEW — the COPY-ASSERT (K>4 only).** A position is eligible for a commit **wider than 4** ONLY if its top-1 token
   **exact-matches** the aligned token in the retrieved source span (a decode-time predicate, cheap: the source is in
   context). A high-K run is thus provably a **verified copy**, not a hopeful parallel guess. Any position that is not a
   verified copy caps the run at `k_max=4` (falls back to the §6 K4 regime). This converts "trust the model at K10" into
   "the model may only go past 4 where it is literally transcribing context" — the design-time answer to the directive's
   "any K>1 must not worsen value grounding." **The copy-assert never fires on args (FSM path) and never fires on derived
   values (not in context) — it is exactly the mechanism that keeps K high on copies and K=1 on everything that must be
   invented.** Audit `copy_assert_violations == 0` is added to the §2.5 KILL-3 counter set.

### A.4 (iv) Per-rung gate — golden number + committed-tok/fwd counter + kill + honest GPU-h

Both rungs are adjudicated by the **Section C golden-number gate** (twin@K vs banked stock-AR 19/50 on frozen `w2_n50`)
**AND** the §2.1 K-isolation gate (twin@K vs twin@K1, same-seed paired — isolates the K effect from SFT). Speed is the
episode-weighted **avg committed tok/fwd** from the engine counters over the gate run itself (directive point 2), with
the §2.5 value-projection audits clean (now including `copy_assert_violations`). Dual-exit per §6: SPEED-FAIL is a soft
stop (ship the last passing rung), PAR-KILL/golden-number-below is a hard revert.

| rung | k_max | copy-assert | speed target (episode-weighted avg committed tok/fwd) | golden-number gate | trained arm | SPEED-PASS ⇒ | SPEED-FAIL ⇒ | KILL ⇒ | GPU-h (5090, honest) |
|---|---|---|---|---|---|---|---|---|---|
| **K6** | 6 | ON (>4) | **≥ 6.0** blended | not below 19/50 (§C) | twin@K6 = re-conversion curriculum→6 (O1+O2+COPY class, O3), 2 seeds | ship K6, go K8 | try SWE-content K-cons. SFT; if still <6 **stop-ship K4** | golden-number PAR-KILL, or copy-assert audit nonzero, or retention/anchor KILL ⇒ revert twin@K1 | **~8–11** |
| **K8–10** | 10 | ON (>4) | **≥ 8.0** blended (10 is the stretch edge) | not below 19/50 (§C) | twin@K10 = curriculum→10, 2 seeds (+optional bd-64 re-conv if canvas-bound) | ship K8–10 (record max held K) | ship K6 (last pass) | as K6 | **~10–13** (+~3–5 if bd-64 pulled) |

**Kill criteria (pre-registered, per rung):** (a) **golden-number PAR-KILL** — twin@K statistically below stock-AR 19/50
by the §C McNemar rule ⇒ revert; (b) **SPEED-FAIL** — census-or-measured avg committed tok/fwd below the rung target at
the golden number ⇒ soft stop, ship last pass; (c) **copy-assert / value-projection audit nonzero** (KILL-3) ⇒ the
tok/fwd number is contaminated, revert; (d) **retention/anchor KILL** (§2.4/§2.3, GSM8K ≤ anchor−2 or matched-20
significant net-loss) ⇒ revert; (e) **CTX-bound INVALIDATION** — if the read-window clamp (#128) is not live, the golden
gate row is INCONCLUSIVE-BY-CTX, not a K verdict. **Census pre-KILL (A.0): `f_blocked ≳ 0.15` on edit content ⇒ K6 not
attempted.**

### A.5 (v) Stop-clean semantics — ship the last passing rung, explicitly

The extended ladder inherits §6's dual-exit and §0's downside-is-free reframe **verbatim**: the K=1 twin is the certified
deliverable of the loop and is **never at risk**. Concretely, the ship rule is a monotone staircase — **K1 → K1.5 → K2 →
K4 → K6 → K8–10** — and the shipped artifact is **the highest rung that held BOTH the golden number and its speed
target**. A K8 SPEED-FAIL ships K6; a K6 golden-number PAR-KILL ships K4; a K6 census pre-KILL ships K4 without a rung
spent. **A full miss above K4 costs nothing beyond the census + one decode-only pass** — the banked (K, 19/50-matched)
twin from §6 is the floor. There is no rung whose failure retires a capability or the golden number (that is the entire
point of the behavioral gate, §0).

## B. POST-TRAIN ON-POLICY SPEED-RL LEVER (S4-style, user-licensed — DIRECTIVE 3c)

**When this lever is pulled:** ONLY after the SFT/conversion ladder (§4 + Section A) stalls **below the K target at the
golden number** — i.e. in-conversion K-consistency delivered a golden-number-holding twin but its speed plateaued under
6 (the honest expected failure: the copy mass engaged but not enough). Speed-RL is the post-SFT lever to push committed
tok/fwd the last stretch **without** dropping below 19/50. It is NOT a capability lever (it cannot lift the SFT ceiling;
that is iteration-2's job) and is explicitly deferred until a golden-number-holding K=1/K≈2–4 base exists.

### B.1 Reward design — resolve-gated speed, never speed-gated resolve

- **Primary reward = terminal resolve (0/1), official docker.** `r_resolve ∈ {0,1}` = the official
  `swebench.harness.run_evaluation` verdict (patch applies ∧ all FAIL_TO_PASS + PASS_TO_PASS green) — the same
  ground-truth reward the SFT datagen rejection-samples on (`swe_tuning_campaign_design §1.1`). This is the gate.
- **Speed shaping — ONLY on resolving episodes.**
  `r = r_resolve · (1 + λ · clip((avg_committed_tok_per_fwd − K_ref)/K_ref, 0, 1))`, with `λ` small (≈0.2–0.3) and
  `K_ref` = the current shipped rung's tok/fwd. **On a FAILING episode `r_resolve = 0` ⇒ the speed term is multiplied
  out ⇒ zero reward for being fast-and-wrong.** This is the load-bearing asymmetry: the model can only earn the speed
  bonus by resolving *and* being fast, so it cannot game the reward by committing garbage quickly (the exact pathology a
  naive tok/fwd reward would train — a K10 twin that overshoots EOS and emits nothing, cf. the §0.I overshoot). The
  avg_committed_tok_per_fwd is the audited engine counter (value-projection + copy-assert clean, §2.5 / A.3), not a
  self-report.
- **Group-relative (GRPO) advantage:** rollout a group of `g` diffusion-decoded episodes per instance; advantage =
  reward − group mean. **A group with zero resolves is a zero-variance group ⇒ no gradient** — so speed-RL only produces
  signal on instances the base *sometimes* resolves. This is the sparse-terminal-reward problem `swe_tuning_campaign_design
  §2.1` flagged: **speed-RL is viable only on a base that already resolves a non-trivial fraction of the train-side
  curriculum** (post iteration-2), and must curriculum on resolvable instances or spend all its rollouts on zero-gradient
  groups.

### B.2 Rollout infra — the honest reality check (this is where the lever is expensive)

- **Diffusion-twin rollouts MUST serve on OUR vllm-pin engine on the 5090.** The reward being trained *is* the diffusion
  decode's committed tok/fwd — you **cannot** roll out AR and reward AR speed. So unlike the accelerated-RL methodology
  (`methodology_diffusion_accelerated_rl.md`, which correctly makes the *default* rollout path stock guided-AR because
  the twin is **not** a bulk-throughput multiplier — measured 0.73–0.94× AR at batch>1, best-of-N signal REFUTED), **this
  loop's rollouts are diffusion-decoded by necessity.** That is the tax the directive names.
- **The flywheel GB10 host serves AR-only — pattern-reuse its harness, NOT its server.** Reuse the flywheel's
  **verifiable-reward** machinery (official docker scoring, the ledger, coverage/best-of-k accounting,
  `runs/swe_datagen_s1` pattern) as the reward oracle — it is server-agnostic (docker scores a patch string). The
  **serving** side stays on the 5090 FLARE engine. Loop topology: **{5090 diffusion engine → rollouts} → {docker
  verifiable-reward, wherever docker runs} → {5090 → GRPO gradient step}**, serial on one GPU.
- **Throughput reality check — budget episodes/step honestly.** Measured diffusion rollout throughput is **21.4
  eps/GPU-h at c=4** (`runs/w2_n50/report.json` secondary; AR is 99.6, 4.65× faster — the twin's structural batch-occupancy
  collapse, §9.2, 0.45 effective batch at b16). So a GRPO step of `g=6 × p=8 = 48` rollouts costs **48 / 21.4 ≈ 2.24
  GPU-h of rollout alone**, before scoring and the gradient step. **This dominates everything:** a *minimal* 30-step run
  ≈ 30 × 2.24 ≈ **67 GPU-h of rollout**; a *real* 80–100-step run is **180–350+ GPU-h**. Two honest consequences: (1)
  **the §9 infra track (B-P1 EOS-aware canvas + batch-occupancy) is a hard prerequisite** — it is the only lever that
  moves 21.4 eps/GPU-h and therefore the RL budget; pull it before RL. (2) **Scope RL small and curriculum'd** — few
  steps, resolvable-instance curriculum, `g` as small as the group-variance tolerates — this is a *finishing* lever on a
  narrow speed gap, not a from-scratch trainer.

### B.3 On-policy / off-policy contamination rules

- **This is genuinely ON-policy for the diffusion decode.** The behavior policy (rollout) and the target policy (updated)
  are **both the diffusion twin@K** — we are training the diffusion decode to be faster, so target == behavior. This is
  the *opposite* of the methodology loop (target = AR, behavior = diffusion ⇒ needs importance correction, #30). **No
  cross-paradigm importance weight is needed** when rollouts are regenerated each step.
- **Bounded staleness if rollouts are reused.** If throughput forces reusing a rollout batch across `s` gradient steps
  (tempting at 21.4 eps/GPU-h), the reuse is off-policy in *version* only (same paradigm). Apply the standard clipped
  per-token importance weight **within-paradigm** `w_t = clip(exp(logp_θ − logp_θ_old), 1−ε, 1+ε)` on the
  reasoning/free/copy tokens, **masking FSM-forced structural tokens** (they are policy-independent), exactly the
  localized correction `methodology §off-policy` derives. Cap staleness `s ≤ 4` and trip on `max |logp_θ − logp_θ_old|`.
- **Value/arg tokens are excluded from the policy loss** (they are FSM-forced or K=1-serial) — so the RL gradient acts on
  the parallelizable spans only, which is also where the speed reward lives. Contamination guard: `hybrid_clean`
  does **not** expose per-token logprobs on the served path (HTTP 500, `K1_COMMITTAL_ANALYSIS.md` REPRO) — the RL loop
  must read logprobs from the **training-forward** (offline re-score), not the served engine; a served-vs-training logprob
  spot-cert (A6-style) is a pre-registered gate before any update.

### B.4 KL / retention safety kit — the S2 lesson, verbatim

- **KL-to-base early-stop at 0.05** (the S2 trip: `KL_TO_BASE_COEFF=0.05`, `kill_retention_tripped` fired at step 120 in
  `s2_pilot_result.md`; RL-v2 used the same on value/free tokens, structural masked). Rolling `max_KL_to_base` reported
  every step; trip ⇒ halt, keep the last pre-trip checkpoint.
- **Retention anchors every N steps** (the batteries §2.3/§2.4): GSM8K legacy N=20 (**anchor 13/20**, KILL ≤ anchor−2),
  tool-call **matched-20 exact_args** (anchor twin@K1's own; KILL on significant McNemar net-loss). These pin general
  reasoning and the certified tool-call spine against speed-RL erosion — speed-RL that resolves faster but forgets the
  tool-call args is a KILL, not a win.
- **Golden-number re-gate is the acceptance test** (§C): a speed-RL checkpoint ships ONLY if it holds 19/50 on `w2_n50`
  AND beats the pre-RL rung's committed tok/fwd. Speed bought below the golden number is reverted ([[diffusion-promotion-discipline]]).

### B.5 Leakage — train-side instances only, zero-overlap proof per batch

- **Train-side pool ONLY:** SWE-Gym keepers + the **387 Verified-adjacent** ids (all 500 Verified test − the 113-id eval
  holdout) per `runs/swe_datagen_s1/USER_LEVER_BELT.md` (ENACTED 2026-07-07). **Never** the 113-id holdout
  (`inner5 ∪ tier0_20 ∪ tier1_100`, sha `c56f473ad31e…d168e`) — which **contains `w2_n50` and Tier1-C46**, the eval rings.
- **Per-batch KILL-D1:** before every RL step, hash-assert `rollout_instance_ids ∩ eval_holdout_113 = ∅` against the
  pinned `.eval_holdout_sha256`; any ring-file drift or intersection ⇒ **halt the run** (not just the batch). This is
  stricter than the once-at-launch SFT assert because RL draws instances continuously — the proof is **per batch**,
  logged with each step's `pool_sha256`.
- **Verified-adjacent is a repo/era-adjacency caveat, not a leak** (USER_LEVER_BELT records it): the standard-practice
  firewall — *no evaluated instance ever trains* — holds; the golden number `w2_n50` stays never-trained forever.

## C. GOLDEN-NUMBER GATE PROTOCOL — the stats spec (DIRECTIVE point 1)

### C.1 The frozen reference — pinned to the exact file, hash, and per-instance vector

- **Pool:** the frozen **50-id `w2_n50`** pool. `pool_sha256 = fe1973937dfb500b5ced1f129648fbec712ee66c74bc357b4fd2b58d3057be4c`
  (`runs/w2_n50/report.json`). Subset list `runs/w2_n50/subset_n50.json`; per-instance envelope seeds
  `runs/w2_n50/inputs/seed_map.json` (base 1234, per-id offset — e.g. astropy-14182→1234, astropy-14539→2234).
- **Run of record:** commit **`f33fb6b`** ("W2 N=50 verdict banked"). **The banked file's own hash is pinned:**
  `sha256(runs/w2_n50/report.json) = 54a1b9373b3d6593a51bb191b3d2c39b83d0b931bc851385469d3e9da1bc22f6`. Any gate row
  MUST re-assert both hashes at eval-launch; a drift ⇒ the reference is not the run of record ⇒ do not adjudicate.
- **The banked stock-AR per-instance verdict vector (the golden number, n=50, 19 resolved):**
  `django-{12713,13315,13933,14089,14373,14855,15104,15561,16082,16429,16493}`,
  `matplotlib-{13989,24970}`, `pydata/xarray-4075`, `scikit-learn-{12585,14053}`, `sympy-{16886,20154,23824}`.
  The other 31 ids are stock-AR unresolved. **This 50-length 0/1 vector is the paired reference** — twin@K is scored
  per-id against it, not against a re-run of AR (the AR arm is banked, frozen, never re-served).

### C.2 The frozen envelope + scoring (identical to the run of record — the FLARE fragility rule)

- **Sampling envelope:** temp **0.6** / top_p **0.95** / top_k **20**, **NO presence penalty** (the FLARE fragility rule —
  presence penalty perturbs the grammar path; the run of record has none). Proxy-forced, native `qwen3_xml`, turn cap 75,
  empty-patch re-drive retries=1, episode-in-official-container `swebench/sweb.eval.x86_64.<inst>`.
- **Scoring:** official `swebench.harness.run_evaluation` docker, **no mock** — byte-identical to the banked AR path
  (`runs/w2_n50/score_all.sh`).
- **CTX-overflow truth-telling labels ACTIVE** (the 2026-07-12 fix, §STATUS): a cap-death is recorded
  `terminal_cause="ctx_overflow"` (env-limited), kept OUT of `clean_exit0`/`empty_patches` — so a twin that dies at the
  32768 wall is scored **env-limited, not an honest empty-patch miss**. A gate run whose twin CTX-death rate is nonzero
  is **CTX-bound** and its golden-number verdict is INCONCLUSIVE-BY-CTX until the read-window clamp (#128) is live
  (A.2) — the label makes this visible instead of silently deflating resolve.

### C.3 The paired statistic — TWO gates, and why both

The confound (banked AR = **stock** weights; twin@K = **SFT+conversion** weights + diffusion paradigm) means the
golden-number comparison is a **capability+paradigm bar** (the user's pinned absolute number), not an isolated K-effect.
So the protocol runs **two** paired McNemar gates and both must hold:

1. **GOLDEN-NUMBER gate (the directive's bar):** twin@K vs the **banked stock-AR vector** (C.1), paired on the 50 ids.
   `b` = (AR resolves ∧ twin@K fails), `c` = (twin@K resolves ∧ AR fails), net = `b − c`. Two-sided **exact-binomial
   McNemar** on `(b,c)`. **PASS = "not statistically below 19/50"**: `p ≥ 0.05` (twin not detectably worse than the
   golden number). The strict-parity reading `|net| ≤ 2 ∧ p ≥ 0.05` (`report.json::parity_rule`) is reported alongside
   but the directive's operational bar is *not-below*.
2. **K-ISOLATION gate (isolates K from SFT):** twin@K vs **twin@K1**, same-seed paired (§2.1) — the only gate that
   attributes a resolve change to K rather than to weights. **PASS = §2.1 PAR-HOLD** (`net-loss ≤ 2 ∧ p ≥ 0.05`).

**A rung ships only if BOTH pass.** Gate 1 answers "did we match the golden number"; gate 2 answers "did K cost us
resolves". Golden-hold with a K-isolation PAR-KILL means the SFT base is carrying a twin that K is degrading ⇒ revert K.

### C.4 Alpha + power honesty at n=50 — what deltas are actually detectable

At `α = 0.05` two-sided, McNemar's exact test on `n_d = b + c` discordant pairs can declare a regression **only when the
discordant split is steeply lopsided.** Computed exactly (rejection boundary = largest `min(b,c)` with `2·P(X ≤ min) ≤
0.05`, `X ~ Binom(n_d, ½)`):

| discordant pairs `n_d` | rejects when `min(b,c) ≤` | ⇒ minimum **detectable net-loss** `b−c` |
|---:|---:|---:|
| 10 | 1 | **8** |
| 12 | 2 | 8 |
| 14 | 2 | 10 |
| 16 | 3 | 10 |
| 17 (today's twin: b=17,c=0) | 4 | 9 |
| 20 | 5 | 10 |
| 24 | 6 | 12 |
| 30 | 9 | 12 |

**The honest reading:** at n=50 the gate detects only a **large** regression — a net-loss of roughly **8–12 resolves**
(≈16–24 pp) depending on the discordance rate. It **cannot** certify a tight match: a twin resolving, say, 14/50 (net-loss
~5 against 19) is **"not detectably below 19/50" and PASSES** — the non-inferiority band is wide. The golden number
itself is only pinned to **±6–7 resolves**: Wilson 95 % CI for 19/50 is **[12.9/50, 25.9/50]** (≈[25.9 %, 51.9 %]). So
"matching 19/50" at n=50 operationally means **"not catastrophically below,"** and a true equivalence-to-within-2-resolves
is **UNVERIFIABLE at n=50** (a TOST/equivalence test on a ±2 margin has near-zero power here). **Registered consequence:**
the gate is a coarse floor, not a fine parity certifier. Tightening requires more ids — which **burns holdout** (a
USER_LEVER_BELT decision, §7) — never silently reusing w2_n50 or a training-adjacent id. Today's twin (2/50, b=17/c=0,
p≈0.0) is decisively below and this is why the campaign is gated OUT at entry.

### C.5 Seed policy

- **twin@K** is served at **primary seed 1234** (the banked `seed_map.json` base — true per-id envelope pairing against
  the banked AR) and **confirmed-in-direction at robustness seed 20260701** ([[retrain-freely-rule]]: never a par call
  on one seed). The **verdict is on the primary; the robustness seed must not flip the direction.**
- **The banked AR is NOT re-seeded** — it is the frozen f33fb6b vector (C.1). The pairing is twin@K(seed 1234) vs
  banked-AR(its own recorded seeds); the envelope seeds match by construction (same `seed_map.json`), so the only
  difference from the AR arm is weights+paradigm — the exactly-pinned confound of C.3, not a seed artifact.

### C.6 The invariant — K is never bought below the golden number

**Pre-registered, non-negotiable:** no rung ships whose twin@K falls into golden-number PAR-KILL (C.3 gate 1 fails) or
whose golden-number verdict is INCONCLUSIVE-BY-CTX (C.2, clamp not live) or INCONCLUSIVE-BY-POWER (twin below the §1.1
entry floor). **Speed with quality below 19/50 is not a win — it is a revert.** The shipped artifact is always the
highest rung that held the golden number AND its speed target (A.5); a speed gain that costs the golden number reverts to
the last golden-number-holding rung, K=1 included. This is the §0 downside-is-free reframe applied to the directive's
absolute bar: the campaign can only ever *add* speed on top of a 19/50-matched twin, never trade the number away for it.

---

## DIRECTIVE-2(2026-07-12) — USER: ATTACK VALUE SPANS AT K>1 THROUGH TRAINING ("hard doesn't mean give up")

User directive (verbatim intent): *we DO need a lever to attack value spans at K>1 through training — it being
hard is not a reason to give up.* This licenses a **VALUE-SPAN K TRACK (V-track)** alongside the reasoning-span
ladder, and it re-opens the "values stay K=1 forever" rule AS A MEASURED QUESTION for the COPY subclass only.

Evidence honesty — why this is attackable NOW when it wasn't before: the historical neutral SFT variants
(value-span mask forcing, candidate-ranker, skeleton-value-infill) attacked raw EXACTNESS on a model that could
not ground arguments at K=1 at all. Today's object grounds at ~98% source-verbatim at K=1; the question is
PARALLELIZING an already-exact capability. The census (runs/k_census/) measures the copy/derived split: copy-class
value tokens (target string exists verbatim in context) are the majority of value mass with top1_exact 0.80 /
median conf 0.92 — a pointer/transcription structure with ONE retrievable joint completion. DERIVED values
(computed numbers, invented names) remain chain-rule-bound and stay K=1 — the V-track does NOT touch them.

Levers licensed (design to be appended as SECTION V):
- **V1 (training)**: copy-span joint-infill consistency in the conversion — mask ENTIRE copy spans, supervise all
  positions jointly conditioned on the context containing the source string; span-length curriculum; span
  identification reuses the census ngram-copy detector (census_content_mix.py).
- **V2 (constrained decode)**: pointer-constrained whole-span commit — candidate set mined from context n-grams;
  commit the full span in one forward when one candidate jointly dominates; optional 1-forward verify-then-accept.
  Promotable under the CONSTRAINED lane per [[diffusion-promotion-discipline]].
- **Gates (non-negotiable)**: value exactness held at the KILL-T1 anchor (exact_args, matched-20), golden number
  held (§C), derived-value spans measurably untouched, committed tok/fwd gain reported on value spans separately.

Ceiling arithmetic this unlocks (the reason to fund it): f_value ≈ 0.5 with copy-majority ⇒ if copy spans go
parallel, effective f_value ≈ 0.2 ⇒ blended ceiling 1/0.2 ≈ 5× — the user's K=5 becomes arithmetically reachable;
without the V-track the blended goal is likely capped ≈2×. The V-track is therefore NOT optional garnish; it is
load-bearing for DIRECTIVE(2026-07-12).

---

## STATUS(census) — 2026-07-12: L1-SWE CENSUS EXECUTED — §6.1 measured, DESIGN-EXT §A.0 pre-KILL **FIRES**

Full report `runs/k_census/CENSUS_REPORT.md`; headline JSON `runs/k_census/census_counters.json`. Object =
iteration-1 twin (HF two-stream form `…-mswe-S-merged`, identical weights). **~0.30 GPU-h, server down, GPU idle
at exit (383 MiB / 0%).** Three phases, all reusing validated instruments: A = content-mix over all 334 keepers
(offline, CAD-class token classification); B = teacher-forced top-1 conditional entropy via the FLARE two-stream
noisy probe on real keeper spans (decode-faithful clean==noisy suffix-mask, 3,390 probes / 80 turns); C = bounded
on-policy adaptive-K decode on 6 Tier1-C46 first-turns (the CAD sampler, k_max=4, γ∈{0.6,0.9}).

**MEASURED (replaces the §6.1 / §10 estimated `f_value ≈ 0.30`):**
- **Content mix** (all assistant tokens): grammar-scaffold **22.3%** / arg-value **49.7%** / free-text **28.0%**.
  Of model-chosen tokens: **arg-value 64.0% · free-text 36.0%**; **copy 57.3% · derived 42.7%**.
- **THE structural finding:** the keeper agent emits every edit as a **structured tool call** (`edit(old_string,
  new_string)`, `run_shell_command(<heredoc>)`), so the copy-heavy edit mass §A.0 bet on **lives inside tool-call
  arguments — the path §5/§A.3.1 hard-force to K=1.** The parallelizable copies are on the wrong side of the value
  firewall. The adaptive-K free-text domain (36%) is reasoning-dominated.
- **Span top-1 conditional** (code analogue of GSM8K 0.238): mean 0.60–0.77 (code carries more low-entropy copy
  mass than arithmetic; arg-copy median **0.92**), **but free-text-derived p10 = 0.246 ≈ 0.238** and the
  distribution is bimodal.
- **f_blocked (§A.0 limiter) = 0.335 pooled at γ0.6 (min over ALL classes 0.257, even arg-copy).** f_blocked ≥ 0.15
  on **every** span class at **every** γ ⇒ **the pre-registered A0 pre-KILL FIRES** (1/0.335 = 2.98 < 6.7).
- **On-policy (Phase C):** free-text adaptive-K engages **avg 1.20 tok/fwd (γ0.6) / 1.06 (γ0.9)**, k1_share 85–95%
  — the S2 GSM8K 1.05 wall reproduced on SWE reasoning. Blended-with-arg-lock **≈ 1.06× / 1.02×**.

**BLENDED-K CEILINGS (measured):** design-as-written (args K=1 + free-text gate) = **1.27× (γ0.6) / 1.13× (γ0.9)**;
most-aggressive (§5 relaxed, copy-assert on ALL positions, k_max→∞) = **2.98× (γ0.6) / 1.76× (γ0.9)**;
content-only f_value ceilings 1/f_value = 1.56× (arg-locked) / 1.16× (realistic) / 2.34× (derived-only).

**PER-RUNG VERDICT (pre-registered §6/§A.4 rules):**

| rung | blended target | design ceiling | §5-relaxed ceiling | on-policy | verdict |
|---|---:|---:|---:|---:|---|
| K1.5 | 1.3 | 1.27× | 2.98× | 1.06× | **SPEED-FAIL-EXPECTED** |
| K2 | 1.5 | 1.27× | 2.98× | 1.06× | **SPEED-FAIL-EXPECTED** |
| K4 | 3.0 | 1.27× | 2.98× | — | **PRE-KILL** |
| K6 | 6.0 | 1.27× | 2.98× | — | **PRE-KILL** (f_blocked 0.335≫0.15 AND 2.98<6) |
| K8–10 | 8.0 | 1.27× | 2.98× | — | **PRE-KILL** |

**Ship = twin@K1** (§A.5 staircase; no rung above K1 clears its speed target on this data). The K-ladder as written
is capped at ~1.1–1.3× blended on the current keeper data shape. The user's K=5–10 goal is unreachable here by ~4–8×.

**Reconciliation with SECTION V (V-track) ceiling arithmetic — HONEST CORRECTION.** SECTION V projects "effective
f_value ≈ 0.2 ⇒ blended ≈ 5×" if copy spans go parallel. The census does not support 5× on measured evidence:
(a) **derived (non-copy) mass is 42.7%, not 20%** — so even every copy going perfectly parallel yields effective
f_value 0.427 ⇒ ceiling **2.34×**, not 5×; (b) **copies do not cleanly parallelize** — arg-copy f_blocked = 0.257
at γ0.6 (a ≥25% per-position block rate ⇒ contiguous copy-runs of ~2.9, not ∞). The realistic upper bound with
copy-spans fully unlocked **and** §5 relaxed is **≈2.98× (γ0.6)**. **The V-track is worth pursuing to lift the ceiling
from ~1.3× toward ~2.3–3×, but the honest re-scope is that even a perfectly executed V-track tops out near ~3×
blended, not 5–10×** — the K=5–10 directive is not reachable on this data shape without also moving the 64% arg mass
onto the adaptive-K path (a harness/edit-format change). Re-run the census (0.3 GPU-h) after any such change before
spending a rung.

---

# SECTION V — VALUE-SPAN K TRACK: copy-span joint-infill (V1, train) + pointer-constrained whole-span commit (V2, decode)

Appended to answer DIRECTIVE-2(2026-07-12). The reasoning-span ladder (§4–6, Section A) explicitly **locks the
arg-value path at K=1** (§5 rule 1, A.3 rule 1). DIRECTIVE-2 re-opens that lock **for the COPY subclass only**: value
tokens whose exact string is already in context (paths, identifiers, re-emitted source blocks) are a *transcription/
pointer* problem with one retrievable joint completion, not an *invention* problem. This section designs the two
licensed levers — **V1** (train the denoiser to joint-infill whole copy spans) and **V2** (a decode-time constrained
whole-span commit) — and prices them against the census. **Same evidence discipline, same pre-registered-KILL
structure, same voice. Nothing above this section changes; V2 formally amends A.3 rule 1 for the copy subclass and
that amendment is documented in V.2.**

**Why this is a different problem than the historical value-SFT failures (the honest lineage).** The 2026-06 neutral
SFT variants named in DIRECTIVE-2 — value-span mask forcing (`qwen35_public_train_candidate_value_span_result.md`),
candidate-ranker (`qwen35_candidate_ranker_*`), skeleton-value-infill (`qwen35_skeleton_value_infill_*`) — attacked raw
**exactness** on a base that *could not ground arguments at K=1 at all*. They were pre-grounding capability patches and
they churned. Today's object is different in kind: it grounds copy args well at K=1 sequential (the certified matched-20
`exact_args` 49–51/63, `swe_tuning_campaign_design §STATUS`), and the V-track question is **parallelizing an
already-exact sequential capability**, not manufacturing exactness. That is why it is attackable now and was not then —
and it is why the V-track's entire risk surface is *"does parallel commit preserve the exactness K=1 already has,"* a
regression question with a zero-tolerance anchor (V.3), not a capability-lift question.

## V.0 The census verdict — read this before funding either lever (`runs/k_census/`)

**`CENSUS_REPORT.md` now exists** (aggregated verdict, `census_counters.json`) — this section uses its verdicts and
adds one artifact the census did not: `copy_runs.json` (`census_copy_runs.py`, this section), because the census emits
per-*token* copy fractions but not per-*span* run lengths, and the V2 whole-span projection needs the latter. All share
the same 4-gram detector (`census_content_mix.py::tok_ngrams`, n=4); counts drift <0.03 %, immaterial. **Read the census
verdict first, because it re-frames the whole V-track:** the census fired the §A.0 pre-KILL (`f_blocked` pooled 0.335 ≫
0.15 at γ0.6), pre-KILLed K4/K6/K8–10 (even the §5-relaxed ceiling 2.98× < K4's 3.0), SPEED-FAIL-flagged K1.5/K2
(ladder-as-written ceiling 1.27×, on-policy measured **1.06×**), and shipped **twin@K1** — *for the reasoning-span
ladder as written, with args locked at K=1.* **The census's own §1 structural finding is the V-track's mandate:** 64 %
of model-chosen mass is tool-call **arg values** the §5 firewall forces to K=1, and **57 % of model-chosen tokens are
verbatim copies — but the copy mass lives inside those locked args** ("the parallelizable copies are on the wrong side
of the value firewall," CENSUS §1). The census then names the V-track as one of two candidate unlocks (§8.2b: "push
copy-run length past the ~25 % per-position block wall *with* a hard copy-assert so §5's value protection is
preserved"). **SECTION V is the design of that census-recommended lever** — and it inherits the census's honest ceiling:
even perfectly executed it tops out near ~3× blended, not 5–10× (V.0(d)).

**(a) The mix — value mass is copy-majority, and it is the arg-value region that carries it** (`content_mix.json`, 334
keepers, model-chosen `mc = 992,727` tok, grammar folded out):

| region | frac of model-chosen | copy within region | derived within region |
|---|---:|---:|---:|
| **ARG_VALUE** | 0.6395 | **67.9 %** (431,048 tok) | 32.1 % (203,775 tok) |
| **FREETEXT** | 0.3605 | 38.4 % (137,417 tok) | 61.6 % (220,487 tok) |
| **pooled** | 1.000 | copy 0.5726 | derived 0.4274 |

So `f_value` candidates span **0.4274** (derived-only, the true must-invent floor) to **0.8616** (arg-locked +
freetext-derived, the K=1 domain *if only freetext copies parallelize* — i.e. today's §5-locked reality). The arg-value
region alone is 0.6395 of model-chosen mass, **two-thirds of it copy** — this is the mass DIRECTIVE-2 is buying.

**(b) The exactness gap — the load-bearing number V1 exists to close** (`entropy_probe.json`, FLARE two-stream noisy
probe on the iteration-1 twin `qwen3.5-9b-fastdllm-mswe-S-merged`, decode-faithful suffix-mask = exactly the condition a
parallel commit faces):

| class | mean top-1 conf | median | **top-1 exact** | f_blocked γ0.7 / γ0.8 |
|---|---:|---:|---:|---:|
| **ARG_VALUE / copy** | 0.767 | 0.919 | **0.804** | 0.294 / 0.355 |
| ARG_VALUE / derived | 0.599 | 0.607 | 0.612 | 0.577 / 0.651 |
| FREETEXT / copy | 0.748 | 0.875 | 0.784 | 0.340 / 0.423 |
| FREETEXT / derived | 0.634 | 0.647 | 0.635 | 0.532 / 0.637 |

The crux honesty: **copy args ground well at K=1 *sequential* (certified `exact_args`), but per-position top-1 under a
suffix-mask is only 0.804** — and whole-span exactness compounds that: an *independent* 6-token copy span at 0.804/pos
is `0.804^6 ≈ 0.27` byte-exact. **A naïve parallel commit of copy args at the iteration-1 twin's confidence corrupts
roughly ¾ of ≥6-token spans.** That single fact is (i) why V1 training is needed (lift *joint whole-span* exactness far
above the independent-position product by teaching pointer/transcription behavior), and (ii) why V2 needs verify-then-
accept as a floor even before V1 lands. It is also the exactness the KILL-T1 anchor (V.3) guards to zero regression.

**(c) The whole-span opportunity is real — and it lives in ARG_VALUE, not FREETEXT** (`copy_runs.json`, contiguous
copy-run segmentation, this section's new measurement):

| region | # copy runs | **mean run len** | max | copy mass in runs ≥6 / ≥8 / ≥16 |
|---|---:|---:|---:|---:|
| **ARG_VALUE** | 31,533 | **13.7 tok** | 1761 | **90.3 % / 85.9 % / 72.3 %** |
| FREETEXT | 85,231 | **1.6 tok** | 348 | 30.7 % / 25.5 % / 13.9 % |

This is the decisive shape fact and it **sharpens the V-track to an ARG_VALUE(copy) lever specifically**: arg-value
copies are **long contiguous runs** (72 % of their mass in spans ≥16 tokens — the re-emitted source blocks of
`str_replace`/`edit` old-string args, the verbatim file regions), so a single whole-span commit banks many tokens.
FREETEXT copies are **fragmented single tokens** (75,405 of 85,231 runs are length-1 — a shared word here and there,
not a transcribable span), so freetext-copy is *not* a whole-span lever and is left to the §4–6 reasoning-K entropy
gate where it already lives. **V1/V2 target ARG_VALUE copy runs; freetext stays with the A-ladder.**

**(d) The honest revision of DIRECTIVE-2's 5× arithmetic.** DIRECTIVE-2 sketched `f_value≈0.5 → effective 0.2 → 5×`.
That rested on the value/structural *heuristic* (fold ~50 % "structural" mass free). The span-length evidence does **not
support** it: the freetext mass that heuristic called "free structural connective" is exactly the fragmented,
mean-1.6-token, *derived* freetext — it has no long parallel runs. Priced on what the census actually measures:

- **Value-span (ARG_VALUE) ceiling if copy fully commits free** = `634k / 203k(derived) = 3.12×` (`copy_runs.json`).
- **Blended (model-chosen) hard ceiling** = `1 / derived_frac_of_mc = 1/0.4274 = 2.34×` — the copy/derived floor,
  identical to the census's S3 "derived-only most-generous" 2.34× (`census_counters.json`).
- **Census S2 entropy-gate ceiling** (pooled `1/f_blocked`, §5 relaxed + copy-assert everywhere): `1/0.335 = 2.98×` at
  γ0.6, `1.76×` at γ0.9 — the census's own most-aggressive scenario.
- **Census S1 ladder-as-written ceiling** (args locked): `1.27×` (γ0.6). On-policy **measured 1.06×**.

**So the measured, untrained, evidence-floor ceiling is ~2.3–3.0×, not 5×** — and this matches the census exactly. The
key reconciliation the census leaves open and V-track closes: **the census flags that reaching its S2 2.98× requires
"abandoning the §5 value-protection rule and letting the model parallel-commit tool arguments" — the exact behavior the
K1-committal forensics proved dangerous** (drops `limit`/`offset` at 84–89 %). **V-track is precisely the mechanism that
realizes the census's S2 ceiling WITHOUT abandoning §5:** the copy-assert + verify (V.2) permit parallel commit *only*
where the token is byte-verified against a context source — so it does not relax §5, it **splits §5** into a copy
subclass (unlockable, provably verified) and a derived subclass (`limit`/`offset` and all invented values — stays locked
forever, can never satisfy the copy-assert). S2's 2.98× becomes *safely* reachable, not recklessly.

**The quantitative V1 target, stated as a gap between two measured numbers:** the census's arg-copy `f_blocked` = 0.257
at γ0.6 implies the *untrained* twin, under the contiguous-run rule, commits a copy run of only `(1−0.257)/0.257 ≈ 2.9`
tokens before a sub-γ position halts it (CENSUS §3) — **but `copy_runs.json` shows the DATA carries arg-value copy runs
of mean 13.7 tokens (90 % of copy mass in runs ≥6).** V1's job is exactly to close the **2.9 (untrained-committable) →
13.7 (data-available)** gap — teach the denoiser to transcribe the whole run the data actually contains instead of
stalling at ~3 tokens. That gap is the measurable V1 lift (V.1 kill criterion), and V2's verify makes even the untrained
2.9-run safe to commit today.

Registered plainly: **5× is NOT arithmetically implied on this data shape.** ~2.3× is the copy/derived floor, ~3× the
entropy-gate ceiling, and the honest V-track contribution is lifting the census's shipped 1.06× toward ~1.8–2.3× — a
real ~2× improvement that is still ~2–3× short of the K5–10 directive. Reaching K5 needs *both* V-track *and* a data-
shape change that moves more mass onto the copy-parallel path (CENSUS §8.2a) — a *both-levers-succeed* target, absorbed
honestly, not a V-track-alone one.

**(e) Scope fence — what V-track does NOT touch.** The K1-committal forensics (`K1_COMMITTAL_ANALYSIS.md`) name the
twin's `limit`/`offset` drop (84–89 % unbounded reads → CTX death). Those are **DERIVED** numeric args (you *compute* a
line range; they are not in context to copy) with top-1 exact 0.612 (ARG_VALUE/derived above). **V-track does not touch
them** — they stay K=1 and their grounding is iteration-2's capability job (task #126/#127). Conflating "attack value
spans at K>1" with "fix limit/offset" would be a category error: V-track parallelizes the *copy* subclass that is
already exact; the derived-arg grounding gap is orthogonal and out of scope.

## V.1 DESIGN V1 — copy-span joint-infill consistency (training, folded into the conversion)

**Objective (exact).** Extend the §4 two-stream `L_diff` with a **whole-copy-span joint-infill target**. For a keeper
assistant turn, identify its arg-value copy spans (below); for a sampled span of length `L`, **mask ALL `L` positions
simultaneously** in the denoise stream (not a random ρ-subset), keep the entire prior context — which by construction
**contains the verbatim source string** — clean, and compute `L_diff` as the **joint cross-entropy over all `L` masked
positions from a single forward**: `L_copy = − Σ_{i=1..L} log p_θ(v_i | ctx, mask[1..L])`. The clean stream `L_AR` is
unchanged (byte-identical to the AR forward, §4). This trains the exact decode geometry V2 will run: "given the source
is in context and the whole value slot is blank, transcribe it in one shot." It generalizes O1 (frontier-adjacency
joint commit) from *contiguous-trailing* to *whole-slot* masking, on the one span class where the completion is a
retrievable pointer rather than a guess.

**Span identification at train time — reuse the census detector, and tighten it for precision.** The span tagger reuses
`census_content_mix.py`'s 4-gram copy predicate (`tok_ngrams`, n=4) that already produced the V.0 numbers, but with one
**precision fix that the training use demands and the census measurement did not**:

- *Census predicate (loose):* token is copy iff its trailing 4-gram ∈ `context_ngram_SET`. Honest detector accounting:
  **recall** misses copies shorter than 4 tokens and copies whose run boundary breaks a 4-gram at a re-tokenization
  seam (the length-1/2-3 buckets in `copy_runs.json` are partly this artifact); **precision** admits **false-positive
  runs** — a 4-gram (`def __init__(`, `        return `) that occurs *somewhere* in context but where the token is
  actually derived, not a pointer to a single source location. On a 1761-token "run" the loose predicate cannot tell a
  genuine re-emitted block from a coincidental multi-site overlap.
- *V1 span-selection predicate (tight):* a training copy span must be a **contiguous substring of a SINGLE context
  span** (align the candidate run to one source location by longest-match, not to the union n-gram set). This is
  strictly a subset of the loose predicate; it trades recall for precision, which is the correct trade for a *training
  target* (a mislabeled derived-as-copy span trains the confidently-wrong-identifier failure — the §3 transfer risk and
  a KILL-T1 regression). Report the tight-detector's kept-span count and the loose→tight shrinkage as a data-honesty
  line in the V1 manifest.

**Span-length curriculum (mirrors O3).** Anneal the sampled `L`: **`L ∈ {2,3,4}` for the first ~⅓** of the conversion
(establish stable short-transcription without destabilizing the K=1 base), then grow the sampling ceiling `L→8`, then
`L→16+` over the remainder — matched to the `copy_runs.json` mass (90 % of copy mass is in runs ≥6, so the curriculum
must reach ≥16 to bank the bulk, but only after the short-span base is stable). Cap span sampling so no single microbatch
is dominated by one 1761-token run (weight by `min(L, 32)` so canvas-scale spans do not swamp the loss).

**Loss composition vs plain `L_diff` and vs the §4 K-consistency `L_diff` — compose or conflict (explicit).** V1 does
**not** add a fourth objective; it **amends O2's span-class weighting**. O2 today has two buckets: reasoning/connective
(joint-commit) and VALUE (K=1 sequential, the "values-always-sequential" invariant). V1 **splits the VALUE bucket by the
copy predicate**:
- **ARG_VALUE / copy → the joint-commit bucket** (the new `L_copy` whole-span target). *This is the invariant change
  DIRECTIVE-2 licenses:* "values-always-sequential" becomes **"copies-parallel, derived-sequential."**
- **ARG_VALUE / derived and FREETEXT / derived → stay K=1 sequential CE** (unchanged; the 0.238-class must-invent mass).

So V1 and the §4 K-consistency objective **compose cleanly IFF the tight span tagger cleanly separates copy from
derived value tokens**, and **conflict IFF it does not** — a derived value trained under `L_copy` is exactly the
poisoned target O2 was built to exclude. **The composition's correctness is therefore identical to the tagger's
precision**, which is why V.1's tight predicate and the V.3 derived-byte-identical audit are non-negotiable. Weighting:
keep `VALUE_SPAN_LOSS_WEIGHT=2.0` on the derived-value K=1 CE (unchanged), and put `L_copy` at the O1 joint-commit
weight; the two never touch the same token (copy vs derived partition), so they are additive, not competing, on any
given position.

**What the denoiser must LEARN vs must NOT UNLEARN.**
- *Learn (the lift):* **pointer/transcription** — when the source is in context and the slot is masked, copy it jointly.
  The measurable target is *joint whole-span byte-exactness far above the independent-position product* (V.0(b): lift
  6-token joint exact from ~0.27 toward ~0.95).
- *Not unlearn (the guard):* **K=1 sequential exactness on DERIVED values** — the chain-rule capability the certified
  `exact_args` rests on. Derived values never enter `L_copy`; the retention/anchor kit (§4 in-training safety, KL-to-base
  0.05 early-stop, matched-20 `exact_args` probe every 50 steps) pins it. **V1 that lifts copy-span parallelism but
  drops derived `exact_args` is a KILL, not a win** (V.3).

**Train-data honesty.** `L_copy` supervises **keeper trajectories only** (`runs/swe_datagen_s1/keepers`, 334 →
iteration-2's expanded pool), **train-side, zero holdout** — the §7 firewall stands verbatim (the source string for
each copy span is the keeper's *own* prior context, e.g. its `read_file` result; no eval-ring instance, no
`w2_n50`/Tier1-C46 id, KILL-D1 re-asserted at train-launch). V1 is a **re-weight of the iteration-2 re-conversion**
(§4 two-stream, block 512 / bd 32), not a new training stage — marginal cost ≈ the plain conversion ([[retrain-freely-rule]]:
fold it in, do not bolt it on).

## V.2 DESIGN V2 — pointer-constrained whole-span commit (decode, CONSTRAINED lane)

Promotable under the CONSTRAINED lane of [[diffusion-promotion-discipline]] (credited only if it beats decode-only K=1
at the golden number). **V2 is A.3's copy-assert re-scoped to the arg-value path** — and it therefore formally **amends
A.3 rule 1**: "tool-call args stay FSM-forced K=1 by construction" now reads **"tool-call arg *derived* spans stay K=1;
arg *copy* spans are eligible for a whole-span commit under the dominance rule + copy-assert + verify below."** This is
the exact "re-open values-stay-K=1 as a measured question for the copy subclass only" DIRECTIVE-2 licenses; everything
A.3 said about derived values and freetext is unchanged.

**Candidate mining from live context.** Maintain a rolling **context n-gram index** over everything the runner has seen
this episode — `read_file` results, prior tool outputs, prior assistant turns — keyed for fast longest-match: paths,
identifiers, string/number literals, and (the big mass) verbatim source blocks. When the FSM enters a `<parameter=key>`
value body (it *knows* it is in a value slot — this is the grammar-masked path), and once ≥1 value token is emitted to
anchor position, query the index for **context substrings that start-align to the emitted value prefix**. The candidate
set `C` = maximal verbatim matches. Mining is CPU-cheap (the source is already in the prompt); it is the same index the
census 4-gram detector implies, run forward at decode time instead of offline.

**The joint-dominance commit rule (exact score + threshold).** Stage `L = min(candidate_remaining_len, k_max, canvas_room)`
trailing masks on the value frontier; one forward reads the `L` joint (`+1`-shifted) probe logits. Per masked position
`i`, `c_i = max-softmax(pre-temperature logits)` (temperature-independent commit decision, §5). Candidate score
`S(cand) = Σ_{i=1..L} log p_θ(cand_i)`. **Commit the whole `L`-span in one forward IFF all three hold:**
1. **single-candidate dominance:** `S(top1) − S(top2) ≥ δ·L` (margin scales with length; if `|C|=1`, auto-pass);
2. **contiguous confidence:** `min_{i} c_i ≥ γ_V` (every position clears the entropy gate — the §5 leading-run rule, no
   sub-γ hole inside the span);
3. **copy-assert (the A.3 predicate, verbatim):** `argmax_i == cand_i` byte-equal at every committed position.

Any position failing (2) or (3) **caps the run there** and the tail falls back to K=1 (§5 contiguous-prefix block).
Failing (1) → **no whole-span commit, fall back to K=1** for the slot (today's behavior). One sentence: *commit a whole
arg-value span in one forward iff a single context-mined candidate dominates the joint score by margin ≥ δ·L, every
position clears γ_V, and every committed token is byte-equal to the aligned candidate; else K=1.*

**Verify-then-accept (optional forward 2) + cost accounting.** The V.0(b) gap (0.804 top-1 under suffix-mask) means a
suffix-masked joint propose can be confidently wrong on a minority of positions even when dominance fires. Verify bounds
that: forward 2 re-presents the proposed span as **clean left-context with masks only AFTER it**, reads each proposed
position's conditional under *no* suffix-mask, and **accepts the leading run whose re-scored `argmax` still == `cand_i`**;
the first mismatch caps the accepted run (rest → K=1). **Cost:** a span of length `L` costs **1 forward** (dominance-only)
or **2 forwards** (verify) to commit `L` tokens, vs `L` forwards at K=1. Verify roughly *halves* the win but makes a
wrong whole-span commit **structurally impossible** (a mismatched position is never finalized → never-remask preserved →
APC stays lossless). **Recommendation: verify-ON until V1 lands and the reject rate is measured <5 %, then consider
verify-OFF only if the derived-byte-identical audit (V.3) stays clean at zero.**

**Failure fallback.** No dominant candidate, or sub-γ_V hole, or copy-assert miss → **K=1 exactly as today** (§5). V2
can only ever *add* whole-span commits on top of the K=1 path; its worst case is "never fires" = today's speed, never a
regression. This is the §0 downside-is-free reframe applied to decode.

**Engine integration honesty (where this hooks, and what it must respect).**
- *Decode loop:* hooks in `_hybrid_clean_step`'s value-body branch — the same monkeypatch surface the CAD sampler
  (`eval_flare_freetext_cad.py`, R1 byte-exact certified) already extends. It is a **new dominant-candidate staging
  path** *inside* the grammar-masked value slot, replacing the current forced-K=1 stepping there. Re-run the k=1
  byte-exact certificate after the change (mandatory, §5 rule).
- *Grammar masking:* the FSM still owns the envelope (`<parameter=..>`/`</parameter>` are GRAMMAR/K=0). V2 acts only on
  the **interior value tokens**; the candidate must not cross the closing `</parameter>` (clip `L` at the slot boundary
  the FSM knows). No grammar token is ever whole-span-committed by V2.
- *APC (lossless prefix cache):* a whole-span commit advances the committed prefix boundary by `L` in one step; the
  **never-remask + copy-assert** guarantee committed == verified-copy == final, so the cache stays **lossless** (A.2's
  argument, now with verify as the extra belt against a wrongly-cached copy).
- *Sync-scheduler / width (§9.2):* a whole-span commit is a **large variable-width draft** (`L` up to k_max). At **B=1
  (the goal per [[goal-5x-rollout-b1]]) this is pure win** — no co-batch. At B>1 it **worsens straggler/head-of-line** on
  the forced-sync scheduler unless §9.2 width-bucketing co-batches like-width value commits; register that V2's batched
  throughput benefit is gated on §9.2, while its B=1 benefit is not.

**Measured throughput projection (`copy_runs.json`, the census span-length distribution).** Value-span (ARG_VALUE)
committed tok/fwd if every copy run of length ≥ `Lmin` commits whole (`v` forwards/run; all other value tokens at 1/fwd):

| commit rule | Lmin≥2 | Lmin≥4 | Lmin≥6 | Lmin≥8 | copy-all-free ceiling |
|---|---:|---:|---:|---:|---:|
| **v=1 (dominance-only)** | **2.70×** | 2.60× | 2.45× | 2.31× | **3.12×** |
| **v=2 (verify-then-accept)** | 2.45× | 2.43× | 2.33× | 2.22× | 3.12× |

So V2 takes the value region from **1.0× → ~2.2–2.7×** (verify-ON, realistic Lmin≥4: **2.43×**). Because the top-X% of
copy spans by length carry almost all the mass (90 % in runs ≥6), committing **only** the long spans (Lmin≥6/8) still
banks **2.2–2.5×** — V2 does not need to fire on short fragments to win. **Blended over model-chosen**, priced against
the census's *measured* free-text on-policy (not an optimistic guess): value region 2.43× via V2 + free-text **1.196×**
(CENSUS §5, γ0.6 on-policy) → `992,727 / (634,823/2.43 + 357,904/1.196) = 1.77×`; if the A-ladder later lifts free-text
to its copy/derived ceiling (1.62×) and V2 reaches the value ceiling (3.12×), the blended is the `1/0.4274 = 2.34×`
floor of V.0(d). **This is the honest V-track contribution: it lifts the census's shipped on-policy 1.06× (both regions
serial) to ~1.8× (value region unlocked via V2, free-text still at its measured wall) and toward ~2.3× only if the
A-ladder free-text lever also engages — the doubling of the value region is the whole V-track story, and it is worth
~+0.7× blended on top of what the census shipped.**

## V.3 GATES + KILLS (non-negotiable — spell them out)

**KILL-T1 — `exact_args` matched-20 anchor, ZERO regression (the load-bearing gate).** V-track mutates the arg path, so
the certified argument capability is the primary tripwire. Anchor = **twin@K1's own matched-20 `exact_args`** (§2.3).
V-track tightens §2.3's `anchor−3` allowance to **zero tolerance: raw ≥ anchor AND McNemar net-loss vs anchor not
significant (p≥0.05) with net-loss = 0 on the paired arg-corruption reads.** A single whole-span commit that corrupts an
arg drops `exact_args` → **immediate KILL, revert the V-lever.** This is stricter than every other rung because a
value-copy corruption is exactly the failure V-track must prove it does not introduce.

**Golden-number gate (§C) at any rung that ships V-track.** The combined twin@K-with-V is adjudicated by **both** §C
gates: (1) GOLDEN — vs banked stock-AR **19/50** on frozen `w2_n50`, PASS = not statistically below (McNemar exact,
α per §C.4); (2) K-ISOLATION — vs twin@K1 same-seed paired (isolates the V-effect from SFT). **Ships only if both pass**,
frozen envelope + CTX-truth-telling labels active (§C.2); a CTX-bound row is INCONCLUSIVE-BY-CTX until the read-window
clamp (#128) is live, not a V verdict.

**Derived-value byte-identical audit (define it).** V-track must leave derived values *exactly* as K=1. Two-layer audit:
- *Structural (per-commit):* every V2 whole-span commit logs its positions; assert each is a **verified copy**
  (copy-assert holds) → **`copy_assert_violations == 0`, folded into the §2.5 KILL-3 counter set.** A derived position
  can never satisfy the copy-assert (not in context), so it can never be whole-span-committed by construction.
- *Paired byte-diff (per-eval):* run the shipping twin **V-ON vs V-OFF** on the same inputs, same seed, greedy; over
  every **derived** value token (census tight-detector label), assert **byte-identical output, zero divergences.** Any
  derived-span byte drift ⇒ the V-lever leaked past the copy subclass ⇒ **KILL.**

**Per-lever pre-registered kill criteria.**
- **V1 KILL (train):** held-out **whole-copy-span joint byte-exactness** (spans len 4–8, tight detector, decode-faithful
  suffix-mask) must reach **≥ 0.95 by ~step 300**; **KILL if < 0.90 after 400 steps** (the denoiser is not learning
  pointer behavior — no lift over the independent-position product ~0.27, so V2 will never dominate). AND **retention
  KILL:** derived `exact_args` (matched-20) regresses ⇒ V1 unlearned the must-invent capability ⇒ revert (= KILL-T1).
  Held-out copy spans are **train-side keeper spans withheld from the V1 loss**, not eval-ring (zero holdout burn).
- **V2 KILL / soft-stop (decode):** the dominance rule must fire on **≥ 40 % of arg-value copy-token mass** (below that,
  the projected value-span tok/fwd falls under ~1.6× and the decode complexity is not worth it) — firing < 40 % is a
  **SPEED-FAIL (soft stop, ship K=1 value path)**, not a KILL. Verify-reject-rate **> 20 %** (with verify ON) is also a
  **SPEED-FAIL** (propose quality too low, V1 not ready — verify never corrupts, it falls back to K=1). The only V2
  **hard KILL** is a nonzero derived-byte-diff or `copy_assert_violations > 0` (corruption).

**Honest GPU-h budget per lever (5090; the diffusion par-eval is the cost, not the train).**

| lever | GPU-h | what dominates |
|---|---:|---|
| **V2 decode-only probe** (iteration-1 twin, matched-20 + a few C46 edit turns; measures fire-rate + reject-rate + KILL-T1) | **~1–2** | one decode pass, no retrain |
| **V1 marginal train** (fold into iteration-2 re-conversion, 2 seeds ≈1.2) + preservation battery (~1) | **~2–3** | ≈ the plain conversion |
| **V-shipping rung eval** (golden + K-isolation on the slow diffusion twin, ~46 eps @ 21 eps/GPU-h × arms) | **~4–6** | the 21 eps/GPU-h twin tax (§8; §9 B-P1 reduces it) |
| **V-track total through one shipped rung** | **~8–11** | eval-dominated, same class as an A.4 rung |

## V.4 Sequencing — V2 decode-probe FIRST, then V1 folded into iteration-2

**Yes: the V2 decode-only probe should run BEFORE any V1 training spend.** It is prototypable on the **current
iteration-1 twin** (`models/qwen3.5-9b-fastdllm-mswe-S-merged`, the census model) with **zero retrain**, because it
measures the *mechanics* (dominance-fire-rate on arg-value copy mass, verify-reject-rate, and the KILL-T1 `exact_args`
anchor) which do not depend on the golden-number capability. Two decisive outcomes, both cheap:
- If dominance fires on **≥40 % of arg-value copy mass at verify-reject <20 % with `exact_args` held**, **V2 banks
  value-span speed with NO training** and quantifies exactly how much residual V1 must close — a genuine free lever on
  the copy mass the A-ladder locks.
- If verify-reject is **high** (the 0.804 top-1 gap biting), the probe **prices V1's job precisely** before spending on
  it — the cheapest possible way to size the training, in the exact spirit of the §6.1 census pre-KILL.

**vs the A-ladder:** V-track is **orthogonal** — it attacks the arg-value(copy) token class the A-ladder locks at K=1,
so the blended average is the two levers *combined* (V.2: value region ~2.4× via V2 + free-text at its **measured** 1.20×
→ **~1.8× blended today**, rising to ~2.3× only if the A-ladder later lifts free-text off its measured wall). Run the
**V2 probe as a sibling of the §6.1 census** (both are cheap iteration-1-twin measurements), before committing to any rung
above K4.

**vs iteration-2:** V1 **folds into the iteration-2 re-conversion** as a new O2 sub-bucket ([[retrain-freely-rule]] — do
not train a separate V1 checkpoint), so it is gated behind iteration-2's M_swe exactly like the rest of the ladder. The
recommended order: **(1) V2 decode probe on the iteration-1 twin (now, ~1–2 GPU-h) → (2) §6.1 census pre-KILL adjudication
→ (3) if V2 fires but reject is high, fold V1 into the iteration-2 re-conversion → (4) adjudicate the combined twin at
the golden number.** A V2-probe SPEED-FAIL (fires <40 %) stops the V-track before any training spend, banking the K=1
value path clean.

## V.5 — 10-line brief (the V-track in one screen)

1. **V2-dominance rule (one sentence):** commit a whole arg-value span in one forward iff a single context-mined
   candidate dominates the joint score by margin ≥ δ·L, every position clears γ_V, and every committed token is
   byte-equal to the aligned candidate (copy-assert); else fall back to K=1.
2. **Ceiling arithmetic (measured census):** value-region 1.0×→**2.43×** (v2 verify, Lmin≥4, `copy_runs.json`),
   copy-free ceiling **3.12×**; **blended lifts the census's shipped on-policy 1.06× → ~1.8×** (V2 on value + measured
   free-text 1.196×) **→ ~2.34× floor** only if the A-ladder free-text lever also engages; census S2 gate ceiling 2.98×.
3. **The honest revision (agrees with CENSUS_REPORT.md):** DIRECTIVE-2's 5× is **not** supported by measured evidence —
   the census fired the §A.0 pre-KILL and capped even §5-relaxed at 2.98×; V-track is the census-named (§8.2b) *safe*
   realization of that ~3× (copy-assert splits §5 instead of abandoning it), not a 5–10× lever.
4. **The load-bearing fact:** arg-value copy is **long runs** (mean 13.7 tok, 72 % mass in runs ≥16) — the real
   whole-span lever; freetext copy is **fragmented** (mean 1.6) — left to the A-ladder.
5. **The gap V1 closes:** copy args are exact at K=1 sequential (`exact_args` 49–51/63) but only **0.804** per-position
   under suffix-mask (≈0.27 for an independent 6-token span) — V1 lifts *joint* whole-span exactness to ≥0.95.
6. **V1 highest-risk assumption:** the tight copy/derived span tagger's **precision** — a derived value mislabeled copy
   trains the confidently-wrong-identifier failure (KILL-T1); composition with §4 is correct *iff* the tagger is.
7. **V2 highest-risk assumption:** **candidate mining picks the right source substring** when a block appears multiply in
   context — verify-then-accept + copy-assert bound it, but a mis-mined dominant candidate is the corruption path.
8. **Gates (zero-tolerance):** KILL-T1 `exact_args` (raw ≥ anchor, net-loss 0), golden-number §C at any V-rung,
   derived-byte-identical audit (V-ON vs V-OFF, 0 divergences), `copy_assert_violations == 0` (KILL-3).
9. **Total GPU-h:** ~**8–11** through one shipped rung (eval-dominated by the 21 eps/GPU-h twin tax); the V2 probe alone
   is **~1–2**.
10. **Recommended first probe:** V2 decode-only on the **current iteration-1 twin** (no retrain) — measure dominance-
    fire-rate on arg-value copy mass + verify-reject-rate + the KILL-T1 anchor; run it as a sibling of the §6.1 census,
    **before any V1 training spend**. Fires ≥40 % clean ⇒ free value-span speed and a priced V1; fires <40 % ⇒ stop the
    V-track clean.

## STATUS(v2-probe) — 2026-07-13: V2 DOMINANCE PROBE EXECUTED — pre-registered **SPEED-FAIL, V-track STOPS before V1 spend** (`runs/v2_probe/`)

The V.4 recommended-first probe ran, decode-only, on the current iteration-1 twin (`models/qwen3.5-9b-fastdllm-mswe-S-merged`,
HF form, teacher-forced) with **zero retrain**. Reused instruments verbatim: the census 4-gram copy detector
(`census_content_mix`) for ARG_VALUE copy-run segmentation (same runs as `copy_runs.json`), and the census FLARE
two-stream suffix-mask reader (`flare_two_stream_noisy_logits`, flare_shift) — extended from first-masked-position to
**whole-span (block-suffix) masking** + a live-context n-gram candidate miner + a clean-left verify forward. Data: **60
real edit turns** (40 keeper train-side edit turns from `probe_manifest.edit_turns` + 20 on-policy C46 edit turns mined
from the frozen twin@K1 diffusion dumps; C46 eval-only, decode-measurement not training). **889 arg-value copy spans**
(627 keeper + 262 C46), **594 source-in-window** (decode-faithful; 295 out-of-window = probe's 1792-tok window artifact,
excluded from headline), 1778 forwards, ~25 min GPU. Scripts + `V2_PROBE_REPORT.md`/`.json` + raw per-position dump in
`runs/v2_probe/` (gitignored; paths above). Grid: γ_V∈{0.6,0.7,0.8} (census entropy-gate values), δ∈{0.5,1.0,2.0} nats/tok
(**exploratory** — SECTION V gives no δ grid). Primary op-point (δ1.0, γ0.6, verify-ON, in-window).

**Headline — the KILL bar (fire on ≥40 % of arg-value copy-token mass) is MISSED by two orders of magnitude:**
- **Dominance fires on 0.22 % of arg-value copy-token mass** (in-window; 0.18 % over all spans) — strict whole-span fire
  0.45 % of mass. Across the **entire (δ,γ_V) grid the committed copy-mass frac stays in 0.04–0.28 %** — the verdict is
  δ/γ-insensitive. **< 40 % ⇒ SPEED-FAIL (V.4 soft-stop): ship the K=1 value path, do NOT spend V1 training now.**
- **Implied value-region committed tok/fwd = 1.00×** (verify) / 1.001× (dominance-only) — vs the design's 2.43× verify
  ceiling and 3.12× copy-free ceiling, and far under the ~1.6× "worth-it" floor. **Implied blended 1.063× vs the census's
  shipped on-policy 1.06×** — i.e. **zero lift** (design's 1.77×-if-V2-hit-2.43× is not reached; V2-on-current-twin adds
  nothing).
- **Failure-reason breakdown (mass-weighted, this is what V1 must train):** `byte_mismatch` **61.0 %**, `margin_short`
  27.5 %, `gamma_fail` 11.0 %, FIRE 0.45 %. Candidate mining is **healthy** (gold copy-string ∈ mined candidate set on
  **594/594** in-window spans) — the wall is **not** mining ambiguity, it is **per-position joint exactness**.

**The load-bearing mechanism (validated against the census anchor, cleanly separates "can't parallel" from "can't copy"):**
whole-span position-0 (clean left-context) argmax==gold **1.00** / conf 0.99 (≥ census 0.804), but **interior positions
(parallel-masked left neighbours) argmax==gold = 0.087**, while the **verify pass (clean-left full reveal) argmax==gold
= 0.998**. So the twin transcribes copy args **near-perfectly sequentially (K=1)** and **almost never in parallel** —
exactly V.0(b): a naïve whole-span commit corrupts essentially all ≥2-token spans, and 90 % of the mass lives in runs
≥16 which fire at **0 %**. This is the precise, priced V1 target: **lift interior joint whole-span exactness 0.087 →
≥0.95** (V.1 kill: ≥0.95 by step 300). V2 cannot dominate until V1 lands.

**Gates — only the SPEED gate fails; the safety gates are clean (no corruption):** verify-reject-rate 0.0–12.5 % across
the grid (**all < 20 %**; verify never finalizes a mismatch → APC stays lossless), copy-assert makes a wrong whole-span
commit structurally impossible, and derived values can never satisfy the copy-assert ⇒ `copy_assert_violations`/derived-
byte-diff = 0 by construction (KILL-T1 and the derived-byte audit are **not** at risk — this probe measures fire-rate, not
regression). The verdict is a **SPEED soft-stop, not a hard KILL of V-track's premise.**

**Decision (pre-registered, V.4):** **STOP the V-track before any V1 training spend on this evidence.** The probe did its
job — it **priced V1 precisely** (byte_mismatch-dominated: V1's job is *joint transcription*, not mining or thresholds).
V-track re-enters **only** folded into the iteration-2 re-conversion (V1 as the O2 copy sub-bucket, [[retrain-freely-rule]]),
where V1 can close the measured 0.087→0.95 joint-exactness gap; **re-run this exact probe on the iteration-2 twin before
any V2 decode ship.** Until then, the census verdict stands unchanged: ship twin@K1, value path K=1. GPU-h this probe:
~1.1 (server DOWN + GPU idle at exit).

---

## DIRECTIVE-3(2026-07-13) — USER ADJUDICATION OF THE V2-PROBE KILL: PIGGYBACK V1 ARM

The V2 probe fired its pre-registered kill (0.22% of copy mass vs the ≥40% bar; STATUS(v2-probe), d8e3519).
Presented options: (a) piggyback V1 arm on the required iteration-2 re-conversion, (b) full standalone V1
campaign, (c) honor the kill fully. **USER CHOSE (a) — piggyback.**

Terms (registered): the iteration-2 re-conversion step (#128) runs TWO arms — **twin@plain** (the standard
certified path; unchanged, remains the shipping candidate) and **twin@V1** (same conversion with the SECTION-V
copy-span joint-infill objective folded into L_diff; ~1.2 GPU-h marginal). Then the V2 DOMINANCE PROBE RE-RUNS
verbatim on twin@V1. Decision rule, pre-registered now: the V-track PROCEEDS only if (i) interior
parallel-masked copy exactness lifts 0.087 → ≥0.80 on the probe's span battery AND (ii) dominance fire rate
clears the original ≥40% copy-mass bar; anything less = V-track CLOSED with the census→probe→piggyback evidence
chain, twin@plain ships, no further V spend. twin@V1 must additionally pass the SAME preservation/KILL-T1
anchors as twin@plain before any probe result counts (a V1 arm that breaks exactness is dead regardless of speed).
No standalone V1 campaign is licensed by this directive.

---

## DIRECTIVE-4(2026-07-13) — USER: DESIGN GOAL-SPECIFIC LOSS + TRAINING, DO THE RESEARCH FIRST

The V1 piggyback probe measured interior parallel-masked copy exactness 0.107→0.133 vs the 0.80 DIRECTIVE-3 bar —
the conservative in-conversion dose is insufficient. USER DIRECTIVE: do NOT close the V-track on this evidence;
instead run a dedicated RESEARCH + DESIGN phase for a loss/training regime purpose-built for the goal (large-block
parallel span commit at SWE-Verified parity). Pre-registered inputs the research must confront:

1. **The verify asymmetry (measured, runs/iter2_cert_probe/v1_probe/)**: pos-0 argmax 0.99, sequential/verify
   argmax 0.9978 token-pooled, parallel-masked interior 0.087–0.133. The model KNOWS the content and can VERIFY a
   written-in span nearly perfectly in one forward; it cannot EMIT it parallel-masked. Hypothesis to adjudicate:
   a pointer-drafter + single-forward verify commit (spec-decode semantics, lossless acceptance under the frozen
   envelope) fires on most copy spans WITH NO TRAINING — i.e., V2's dominance-on-masked-canvas was the wrong
   commit rule, not a missing capability. In-repo prior art: task #9 lossless self-spec-decode acceptance.
2. **Dose-response honesty**: +2.6pp/400 in-conversion steps at weight 2.0 mixed with other objectives says little
   about an ISOLATED objective at higher dose (span-only batches, longer schedule, self-distillation from the
   model's own sequential emissions where the target is deterministic). Design a dose-response experiment with
   checkpointed kills, not a single big bet.
3. **Constraints**: conversion-preservation cert must survive (no architecture surgery beyond LoRA-scale),
   KILL-T1 exact_args + golden number remain non-negotiable, leakage firewall unchanged, single-5090 budgets.

Deliverable: SECTION W (research findings + prioritized experiment ladder with per-rung budget and kill rules),
then execute the top rung. The C46 re-gate / quality track proceeds in parallel and is unaffected.

---

## STATUS(2026-07-13) — V1-PROBE EXECUTED (#128 part 2): DIRECTIVE-3 rule output = V-TRACK CLOSED (superseded by DIRECTIVE-4)

The V2 dominance probe was **re-run verbatim on twin@V1** (`v2_dominance_probe.py`; same spans/grid/battery,
only the model object changed). Probe-loadable object = the fastdllm-HF form = re-conversion base
`mswe-S-iter2-merged` **+ the V1 copy-span-infill conversion adapter** (PEFT `merge_and_unload`, scale 2.0);
the exported vllm twin cannot be probed (official Qwen layout, mask stripped). Canonical battery **identical to
the iteration-1 baseline**: 40 keeper + 20 C46 = 60 turns, **889 spans, same turn_id set + span count**
(`--c46-turns 20 --span-cap 35`; C46 dumps unchanged since Jul 9), 1778 forwards, ~25 min GPU. Artifacts
(gitignored): `runs/iter2_cert_probe/v1_probe/` (`v1_probe_verdict.json`, `interior_exactness.json`,
`v2_probe_report_twinV1.json`; a first pass at the default `--span-cap 60` = 974 spans is kept as `*_cap60.*`).

**Preservation precondition — twin@V1 K=1 serving sanity: PASS.** 3 matched anchor turns on the twin@V1 FLARE
hybrid engine: valid 3/3, exact_args 3/3, **byte-identical to twin@plain online 3/3** → the V1 objective did
**not** break K=1 serving.

**DIRECTIVE-3 decision rule, applied verbatim (both conditions required):**
| condition | bar | iter-1 baseline | **twin@V1 (canonical, same battery)** | met? |
|---|---|---|---|---|
| (i) interior parallel-masked copy exactness | **≥0.80** | 0.107 (design-reported 0.087) | **0.117** (+0.010; span-mean 0.16→0.18; the cap-60 first pass read 0.133) | **NO** |
| (ii) dominance fire on copy-token mass | **≥0.40** | 0.0022 | **0.0037** (+0.0015; whole-span fire 0.45%→0.55%) | **NO** |

**Both gates MISS by ~two orders of magnitude.** The lift is **small and directionally correct** but the
failure structure is **unchanged**: `byte_mismatch` **57%** of mass (still the wall), verify-reject **0.0**
(safety intact), value-region tok/fwd **1.00×**, mining healthy (gold∈C **594/594**), pos-0/verify clean-left
≈0.99/0.998 (still transcribes near-perfectly **sequentially**, almost never **in parallel**). **The
pre-registered DIRECTIVE-3 rule therefore returns V-TRACK CLOSED** (a big-but-insufficient lift is still CLOSED).

**Disposition: SUPERSEDED by DIRECTIVE-4 (same day, above).** The user chose NOT to act on the pre-registered
closure and instead opened a dedicated research+design phase (SECTION W): the measured **verify asymmetry**
(near-perfect sequential/verify vs near-zero parallel emit) reframes the result as *the whole-span-on-masked-canvas
commit rule was wrong / the in-conversion dose too small*, not a proven capability ceiling. Net: the DIRECTIVE-3
**rule output is CLOSED**, but the **V-track stays open for the SECTION-W redesign** per DIRECTIVE-4. Either way
**twin@plain remains the shipping candidate**, the C46 re-gate (#129) runs twin@plain WITH the certified
read-clamp, and no V2-decode ship happens on current evidence. GPU-h this stage: ~1.4 (server DOWN + GPU idle at exit).

---

# SECTION W — VERIFY-ASYMMETRY ADJUDICATION + THE DRAFT-AND-VERIFY LADDER (DIRECTIVE-4 deliverable)

Appended 2026-07-13 to answer DIRECTIVE-4. Inputs: three commissioned research briefs (Angle A draft-and-verify,
Angle B dedicated training regimes, Angle C pointer/conditioning mechanisms) **plus an adversarial verification pass
that re-read and re-computed every load-bearing number from the raw records** —
`runs/iter2_cert_probe/v1_probe/v2_probe_raw_twinV1.json`, `runs/v2_probe/v2_probe_raw.json`, both
`interior_exactness*.json`, `v1_probe_verdict.json`, the probe script, and the model's attention-mask code
(`models/qwen3.5-9b-fastdllm-mswe-S-merged/modeling.py`). Where a brief and the raw artifacts disagreed, **the
artifact number wins and the correction is stated inline** — several headline claims (including one phrase inside
DIRECTIVE-4 itself) did not survive and are corrected below. Same discipline as SECTIONS A–C/V: pre-registered
kills, measured numbers only, honest ceilings. Nothing above this section changes: SECTION V's safety architecture
(copy-assert, derived-lock, KILL-T1 zero tolerance, §C golden protocol, §7 leakage firewall) is inherited wholesale.
**W replaces only two things: V2's commit rule (dominance-on-masked-canvas → pointer-draft + verify-accept) and
V1's training regime (mixed in-conversion dose → isolated dose-response ladder).**

## W.0 The adjudication — draft-and-verify IS the missing decode rule; it is NOT lossless by construction

### W.0.a What the raw probe records already prove (re-computed this session, both twins)

The V2/V1 probes logged per-position `verify_eq_gold`/`verify_eq_cand` arrays for all 594 in-window spans. Re-scored
under a **write-the-candidate-in + one-forward-verify** commit rule instead of dominance-on-masked-canvas:

| quantity (twin@V1 iter-2; iter-1 in parens) | measured |
|---|---|
| whole-span verify-clean, one forward, greedy accept | **570/594 = 0.960** (568/594 = 0.956) |
| …on the same spans where dominance fired | 0.2–0.4 % → **a ~480× commit-rule reversal** |
| error structure | 24 (26) spans with ANY error; 22 (24) exactly one token; **zero adjacent-error pairs** — point defects |
| whole-32 clean measured vs independence a³² | 0.941 vs 0.923 — the a^L extrapolation is slightly *pessimistic* at this conditional |
| multi-round spec-decode sim, gold-aligned candidate (teacher-forced oracle) | **17.58 (17.37) tok/fwd** on copy mass |
| deployable stubborn-top-1 candidate | **7.40 tok/fwd**; whole-clean 517/594 = 0.870 span-count / **0.83 by mass** |
| candidate mining | gold ∈ C on **594/594**; n_cand ≤4 for 82.5 %, ≤8 for 90.2 % (max 270) |
| long-run mass | capped spans (run>32): 431, mean run 249, total 107,395 tok; max 1902 — long runs sustain multi-round blocks |
| verify confidence proxy (pos-0) | mean 0.9773 / median 0.9992 / p10 0.9724 / 93.9 % ≥ 0.9 |

Both twins give the same numbers — **the verify capability is already saturated and did not need V1.** The
dominance rule was the wall, not a missing capability: DIRECTIVE-4 item 1's hypothesis is **CONFIRMED in
direction.** Task #9 (`qwen35_specdecode_acceptance_result.md`) reads post-hoc as the same asymmetry from the other
side: its *draft* used the broken parallel-masked capability (0.087) at the cost of a full 9B forward and its verify
was fine — evidence **for** inverting the roles (draft = free CPU pointer, verify = the intact forward). External
anchors: FreeDave (arXiv 2510.00294, training-free lossless dLLM draft-verify, up to **2.8×**), CopySpec/LLMA-class
copy-reference speculation (2–3× on editing workloads). The engine pin already carries the needed primitives
(variable-accept GDN state scatter `num_accepted_tokens` + `ssm_state_indices`, `dflash_drafter_plan.md`; APC
never-remask bitwise gates, `lossless_apc_design.md`).

### W.0.b The correction that changes the promotion path (adversarial pass, A1 — read before quoting any brief)

**KILLED: "lossless by construction / KILL-T1 + golden protected by construction."** The probe's "verify" is NOT the
serial K=1 commit conditional, so the Leviathan/Chen spec-decode theorem does not apply as invoked. Mechanism
(code-level): `flare_two_stream_bool_mask` (`modeling.py:198`) sets `noisy_to_noisy_mask = ~q_clean & ~kv_clean &
same_block` — noisy-stream queries attend **bidirectionally within the block**. The verify forward writes the whole
candidate span in and re-scores, so the logits "verifying" position i attend to drafted tokens at positions ≥ i
**including the token being verified**. Serial K=1 commits from the first-masked position with the suffix *masked*.
Different conditionals, demonstrably: at pos-0 — the serial conditional — argmax==gold is **0.9865 (twinV1) /
0.9916 (iter-1)**, while interior full-reveal verify is 0.9975/0.9973 (matched battery; the oft-quoted 0.9978 is the
unmatched cap-60 arm). **DIRECTIVE-4's phrase "sequential/verify argmax 0.9978" conflates two quantities:
sequential ≈ 0.987–0.992, full-reveal verify ≈ 0.997–0.998.** ("Clean-left re-score" in the briefs is a misnomer —
it is full-reveal-within-block.)

Consequence, registered as the new promotion rule: **pointer-draft + full-reveal-verify defines a NEW decode
envelope. KILL-T1 exact_args and golden 19/50 must be RE-CERTIFIED empirically (W-2), never claimed by
construction.** Mitigating measured fact: at pos-0 — the only like-for-like position — full-reveal changes argmax
accuracy by ≈0 (twinV1 0.9865 vs 0.9865; iter-1 0.9916 → 0.9899), so the leak inflation is plausibly small in
practice. That rehabilitates the *pricing*, not the losslessness theorem.

### W.0.c The corrected math (what survives, at the serial-faithful anchor)

- **Acceptance anchor = 0.987–0.992/token (serial conditional, greedy), not 0.9978.** Whole-32 span:
  0.9865³² = **0.65** / 0.9916³² = **0.76** (vs 0.93 claimed in the briefs).
- **Oracle multi-round re-priced: ~14.6–15.8 tok/fwd** on copies (vs 17.58 claimed); deployable stubborn-top-1
  7.40 stands. Value-region ≈ 2.4–2.7×; **blended estimate ~1.6–1.8×** (vs shipped 1.06×), under the census S2
  ceiling 2.98× (`992,727/(634,823/2.43+357,904/1.196) = 1.77×` reproduces exactly).
- **Both pre-registered bars still clear without training**: fire ≈ 96 % of in-window copy mass at the verify
  conditional (deployable whole-clean 0.87 span-count / 0.83 mass) vs the ≥40 % KILL bar; value-region ≥ the ~1.6×
  worth-it floor.
- **The one unmeasured quantity everything prices from:** every measured accept bit is *greedy argmax-match*. The
  frozen-envelope acceptance **E[p̃(gold)] under temp 0.6 → top_k 20 → top_p 0.95 is unmeasured** — and the
  multi-round sims are teacher-forced (post-rejection rounds reuse gold-prefix-conditioned bits), an oracle upper
  bound. Rung W-0 exists to read exactly these.
- **Dose-response re-baselined (A3):** the matched-battery V1 number is **0.1072 → 0.1168, +0.96 pp/400 steps**
  (span-mean 0.161 → 0.181; `v1_probe_verdict.json`). The "+2.6 pp → 0.133" figure came from the cap-60 first pass
  whose twinV1 arm used a *different span set* (670 spans) — not a matched comparison; struck. The strategic read
  (massively underdosed vs the 0.80 bar) is unchanged; every W.2 slope argument uses +0.96 pp from 0.117.

### W.0.d Verdict

**Draft-and-verify is adjudicated the missing decode rule** — the cheapest, best-evidenced lever on the 64 %
arg-value mass, recovering approximately what SECTION V priced V1 training to deliver (design 1.77×) for zero
training spend — **and it is rung 1 of the ladder.** But it ships only through an *empirical* new-envelope
certification (W-2), per [[diffusion-promotion-discipline]] (CONSTRAINED lane: copy-assert-gated commits, credited
only against decode-only K=1 at the golden number). Training (W.2) and conditioning (W.3) are demoted to
*acceptance-raisers and contingency*, entered on pre-registered triggers only.

## W.1 THE EXPERIMENT LADDER (cheapest-decisive first; every rung pre-registered before spend)

| rung | what | GPU-h | decides |
|---|---|---:|---|
| **W-0** | serial-faithful acceptance probe (training-free, HF-form, no engine change) | **1.5–2** | is the envelope tax small enough; is full-reveal verify a usable surrogate |
| **W-1** | decode-loop prototype (engine value-branch, no training) | 3–5 | on-policy value tok/fwd ≥1.6×; KILL-T1 anchor + K=1 byte-cert |
| **W-2** | new-envelope certification: golden 19/50 §C both gates + derived-byte audit | 4–6 | SHIP / revert |
| W-3 (trigger T-a/T-c) | training ladder W.2 (B3 gym → B1/B2 dose-response) | ≤ ~30 worst | is emission certainty trainable at 5090 dose |
| W-4 (trigger T-b) | conditioning boosters W.3 (C2/C3/C4 probes) | ~3–4 | can input-side alignment raise acceptance/fire |

Decode-only path to a shipped rung: **~9–13 GPU-h.** Grand worst-case (all triggers fire): **~45 GPU-h.** Earliest
decisive kill: **~2 GPU-h** (W-0). Triggers: **T-a** = W-0 kills on envelope tax → W.2 training becomes primary
(with W-0's softmax logs as calibration data). **T-b** = W-2 ships but blended plateaus < 1.6× on-policy → W.3
boosters. **T-c** = training also flat → W-B3 synthetic gym adjudicates trainable-vs-architectural, then stop clean.

### W.1.a Rung W-0 — serial-faithful acceptance probe (~1.5–2 GPU-h, no training, no engine change)

**Design (exact).** Extend `runs/iter2_cert_probe/v1_probe/v2_dominance_probe_twinV1.py` on the identical canonical
battery (60 turns, 889 spans, span-cap 35, same turn_id set) with three changes:

1. **Full 23k context window** — kills the 1792-token instrument artifact that excluded 4,859/15,656 = 31 % of copy
   mass from the headline; far-source verify quality gets measured for the first time.
2. **Three logged reads per span, full softmax (not binary match bits), warped by the exact engine sampler order
   (temp 0.6 → top_k 20 → top_p 0.95):**
   (a) *serial-faithful staged reads* — for position i: gold prefix clean, mask [i:L], read the first-masked
   position. This IS the K=1 decode conditional. Stratified 8 log-spaced offsets per span (~4.8k forwards), full
   per-position reads on any span whose stratified reads disagree with (b);
   (b) *full-reveal verify read* (the deploy candidate, today's probe verify) — logging per-position divergence vs
   (a): Δargmax rate and ΔE[p̃(gold)] at interior positions, not just pos-0. Free extra: log the same forward's
   **clean-stream causal logits** — if those track (a), the two-stream architecture gives a leak-free verify surface
   in one forward;
   (c) *false-accept battery* (Angle C) — perturbed candidates: off-by-one offset, single-token substitution,
   whitespace variant. Require P(full-span accept | wrong candidate) ≈ 0 under the chosen accept rule, else add a
   prob floor and re-simulate.
3. **Post-hoc rejection-sampling simulation at the frozen envelope** (accept d_i w.p. p̃_i(d_i); first rejection
   samples the residual; re-mine; multi-round): report committed copy-mass tok/fwd under (i) serial-faithful bits =
   ground truth and (ii) full-reveal bits = deploy rule. The (i)–(ii) gap **is the leak tax, measured.** Carry the
   caveat: multi-round remains teacher-forced post-rejection (oracle upper bound) until W-1.

**Pre-registered KILL/PROCEED.**
- **PROCEED to W-1 iff ALL:** serial-faithful simulated commit ≥ **8 tok/fwd** on in-window copy mass; implied
  blended ≥ **1.5×**; false-accept full-span rate **= 0** (post floor-tuning); serial-vs-full-reveal committed gap
  ≤ **15 % relative**.
- **KILL/re-price iff:** mean envelope acceptance ā = E[p̃(gold)] < **0.98/token** or implied blended < 1.5× — the
  envelope tax was the hidden wall; trigger T-a, training ladder becomes primary.
- **SURROGATE-FAIL iff:** the (i)–(ii) gap > 15 % — full-reveal verify is not a usable one-forward surrogate;
  price a staged/clean-stream verify (changes v, re-run the economics) before any W-1 engineering.

### W.1.b Rung W-1 — decode-loop prototype (~3–5 GPU-h incl. anchors, no training)

Implement pointer-draft + verify-accept in the engine's value-body branch (`_hybrid_clean_step`, the CAD
monkeypatch surface — R1 byte-exact certified): live n-gram miner (the probe's, run forward), draft written into
the noisy-stream span slots (clipped at the `</parameter>` boundary the FSM knows, L ≤ bd 32), accept rule at the
W-0-measured op-point with residual sampling, re-mask + re-mine after first rejection. Close the 67/594
top-1≠gold ranking gap with re-mine-after-divergence or ≤8-row batched verify (covers 90.2 % of spans; batch-row
width-32 latency on the GDN hybrid gets profiled here — per [[gpu-utilization-standard]], profile, don't guess).
**GDN discipline is the main engineering risk:** rejected suffixes must never fold into the recurrent state — use
the pin's variable-accept scatter; APC never-remask holds because only accepted==final tokens commit.
**Report:** on-policy committed tok/fwd on C46 edit turns (the teacher-forcing caveat dies here), matched-20
KILL-T1 anchor, K=1 byte-exact certificate re-run (feature-OFF path byte-identical — mandatory §5 rule).
**KILL/pre-registered:** on-policy value-region tok/fwd < 1.6× ⇒ SPEED-FAIL soft stop (ship K=1, bank the probe);
any KILL-T1 regression or accept-log violation ⇒ hard KILL. PROCEED ⇒ W-2.

### W.1.c Rung W-2 — new-envelope certification (~4–6 GPU-h, eval-dominated)

Full §C protocol on the W-1 artifact: (1) GOLDEN — vs banked stock-AR **19/50** on frozen `w2_n50` (McNemar exact,
§C.4 alphas); (2) K-ISOLATION — vs twin@K1 same-seed paired; plus the V.3 derived-byte-identical audit (V-ON vs
V-OFF, zero divergences over derived tokens) and exact_args matched-20 at zero tolerance. **Because W.0.b demoted
losslessness, this rung IS the promotion event, not paperwork.** KILL: statistically below golden, or any derived
byte drift ⇒ revert to K=1 serving; the probe evidence stays banked; escalate to W.2/W.3 as acceptance-raisers
rather than ship.

## W.2 The dedicated training regime (Angle B, re-baselined on the matched numbers) — rungs on trigger T-a/T-c

**Dose honesty first.** V1's 400 mixed in-conversion steps supervised on the order of ~10⁵ span tokens (order of
magnitude only — no counter persisted; trainer_state logs loss/lr) and moved the matched metric **+0.96 pp**
(0.1072→0.1168). Published capability injections: dParallel ≈ **140M** self-distilled tokens (92k×256×6, LoRA
r=32, LLaDA-8B → 8.5–10.5× step-compression at held accuracy; its critical ablation — certainty loss *without* the
correct-only guard: 10.4× but accuracy 76→58 — maps directly onto KILL-T1); Fast-dLLM v2 ~1B tokens for the whole
AR→BD adaptation. **~2–3 orders of magnitude underdosed — DIRECTIVE-4 item 2 corroborated.** (dParallel fine HPs
are unverified-as-cited; spot-check before copying exactly.) The leading indicator is the **span-length frontier**
(max L with whole-span exactness ≥0.9): currently **1** on both twins (whole-fire 0.917 @ len 1, ≤0.074 @ 2–3, 0 @
≥4). Induction-circuit formation is abrupt, so kills key on the frontier, not linear extrapolation of token-pooled
exactness. Per [[retrain-freely-rule]], the isolated phase below is licensed precisely because the mixed
in-conversion dose was measured insufficient — do not let the twin@V1 checkpoint constrain the design.

- **W-B3 — synthetic offset-copy gym (2–4 GPU-h, FIRST training rung — a decisive negative is available).**
  5–10M tokens of random-token strings (len 2–64, no semantic prior — forces a pure offset circuit) at varied
  distances (≤8k) inside realistic tool-call envelopes; whole-span parallel-masked joint CE; then 1–2M real
  keeper-span tokens to bind. **KILL: synthetic whole-span exactness cannot exceed 0.8 @ L=8 by 8M tokens ⇒ the
  failure is architectural (GDN two-stream), killing W-B1/W-B2 escalation for ~3 GPU-h.** Discard the adapter if
  the real-span probe doesn't transfer (≥0.30 at bind end).
- **W-B1 — flagship: isolated certainty-forcing self-distillation on deterministic copy-span targets.** Data:
  twin K=1 emissions over the train-keeper pool's edit turns (leakage firewall verbatim — train keepers only, zero
  eval-ring ids, KILL-D1 re-asserted). **Data-yield honesty (A6):** sequential-emission keep-rate is priced by the
  serial conditional, ~0.78–0.86 per 18-token span at greedy (0.987–0.992¹⁸) and lower at the engine's temp 0.6 —
  do NOT plan on "≈ all." Objective: decode-faithful structural mask (context clean, **whole span masked**, future
  masked — the serve canvas, unlike V1's random-noise mixture); loss = joint CE + β=2 certainty (entropy @ T=0.5)
  **only on argmax-correct positions** (the load-bearing guard); 10 % plain-conversion replay. LoRA r=32 **all
  modules** (attn+GDN+MLP), lr 2e-5, eff. batch 64. **Dose checkpoints (log-spaced): 2M / 8M / 32M / 128M span
  tokens ≈ 0.3 / 1.2 / 4.5 / 18 GPU-h** + 0.5 GPU-h probe each (the canonical 594-span battery, unchanged).
  Pre-registered dose-response from the corrected 0.117 start: ~0.25 @ 2M, ~0.45 @ 8M, ~0.70 @ 32M, ≥0.80 @ 128M;
  frontier 1 → 2–3 → 4–7 → 8–15 → 16+.
- **W-B2 — GLAT-style error-keyed glancing curriculum, run as W-B1's ablation arm to the 8M gate** (reveal
  N = ceil(λ·L·ê) scattered interior anchors, anneal to whole-span; +30–50 % step time). Decision rule: the arm
  with a ≥1.5× frontier lead at 8M absorbs the remaining budget. Both arms to the 8M gate ≈ 8 GPU-h combined.
- **KILL rules (all arms):** KR-1 at 8M: KILL if exactness < 0.30 AND frontier ≤ 2; at 32M: KILL if < 0.55 or
  frontier < 8. KR-2 every checkpoint: exact_args matched-20 zero-loss + retention — non-negotiable, stop on trip
  (the S2 KL-trip lesson, 0.0699 > 0.05 @ step 120, stands). KR-3: train loss improves but probe doesn't ⇒
  train/serve mismatch — debug, never escalate dose. KR-4: promote ONLY through W-1/W-2 (re-probe fire ≥ 40 %,
  verify-reject ≤ 0.20, value ≥ 1.6×, then golden §C).
- **Ceiling honesty:** no published result shows 0.80+ single-forward exactness on 16–32-token spans (dParallel
  commits ~8–10 tok/step). The 0.80 emission bar exceeds published evidence — which is exactly why the ladder makes
  draft-verify primary and training the *acceptance-raiser*: the promote surface is verified acceptance, not raw
  emission. Worst case ≈ 30 GPU-h; earliest training kill ≈ 3.

## W.3 Pointer/conditioning mechanisms (Angle C) — boosters on trigger T-b, or W-B1 format inputs

Mechanistic frame (consistent with every W.0 number): the transformer copy circuit is induction-style — attend from
the *previous token's* source occurrence to its successor. Parallel masking destroys the left neighbor, so interior
positions cannot compute span alignment; writing the candidate in (verify) restores it — hence 0.107 vs 0.997.
GDN linear layers are provably weak at exact long-range copy (fixed-size state); exact copy rides the sparse
full-attention retrieval heads — so every mechanism keeps alignment on attention layers or host-side.

- **W-C2 — evidence-conditioned transcription (~0.3 GPU-h):** duplicate the mined candidate sentinel-delimited
  immediately left of the block; re-run the interior-exactness battery. Bar: ≥ 0.5 zero-shot ⇒ decode-side win,
  compose with verify; < 0.5 ⇒ demote to W-B1 input format (converts free recall into transcription-at-short-offset
  and makes the training target deterministic). Scratch region mined from in-window context only (firewall clean),
  stripped before commit.
- **W-C3 — position-coupled alignment (0.5 zero-shot + 1.5 LoRA GPU-h):** alias masked slot i's position channel to
  `source_start+i` so a fixed-offset head solves the copy by position arithmetic. Pin-expressibility check FIRST
  (cudagraph/FLARE bookkeeping); GDN layers ignore RoPE — attention-only reach, GDN-confusion smoke mandatory. Bar:
  interior ≥ 0.5 by step 400 — must decisively beat the +0.96 pp/400 matched flat slope — else kill.
- **W-C4 — retrieval-head attention-bias steering (~0.5 GPU-h, offline HF-form only):** PASTA-style offset-aligned
  bias on top-k retrieval heads (head census via needle/copy battery first). Bar: interior ≥ 0.5 or ≥ 10× fire
  lift; highest pin-port risk of the zero-delta family — a negative port-feasibility read kills it before any
  engineering. Composes safely with verify (steering raises fire; verify gates commits).

## W.4 How every rung reports against the golden number + KILL-T1 (uniform block, pre-registered)

Every W rung's result report MUST carry, in this order:
1. **KILL-T1 surface:** matched-20 `exact_args` vs the twin@K1 anchor — raw ≥ anchor AND McNemar net-loss = 0.
   Hard KILL on trip, at every rung including probes that touch no weights (W-0 reports it as N/A-by-construction
   with the no-weight-delta hash; W-1+ measure it).
2. **K=1 byte-exact certificate** whenever the decode loop is touched (W-1+): feature-OFF path byte-identical to
   shipped serving (the twin@V1 3/3 sanity precedent).
3. **Envelope status line (new, from W.0.b):** does this rung's commit rule reproduce the serial K=1 envelope
   distribution? For pointer-draft + full-reveal verify the honest answer is **NO — new envelope**; therefore no
   ship without W-2's §C both-gates pass (GOLDEN vs 19/50 + K-ISOLATION vs twin@K1). "Lossless by construction"
   is banned phrasing in W reports.
4. **Corruption counters:** accept-log/copy-assert violations = 0; derived-byte-identical audit (V-ON/V-OFF) = 0
   divergences; false-accept battery result (W-0 rule carried forward).
5. **Leakage line:** training rungs consume train-keeper spans only (zero eval-ring/`w2_n50`/C46 ids, KILL-D1
   re-assert at launch); C46-based probe rungs are decode-measurement only, never training.
6. **GPU-h + server-DOWN/GPU-idle-at-exit**, per the standing hygiene rule.
7. **Promotion lane:** CONSTRAINED ([[diffusion-promotion-discipline]]) — draft-verify commits are copy-assert-gated
   decode mechanics, credited only if the W-2 artifact beats decode-only K=1 at the golden number.

Golden-cadence honesty: the §C golden gate costs ~4–6 GPU-h/arm — it runs at W-2 and at any training promote (KR-4)
ONLY; per-checkpoint safety is the cheap anchor set (exact_args + retention), never the golden eval.

## W.5 — 10-line brief (SECTION W in one screen)

1. **Adjudication:** the verify asymmetry is real and the commit rule was the wall — write-candidate-in +
   one-forward verify accepts **0.956–0.960** of whole spans where dominance fired 0.2 % (~480× reversal, both
   twins, zero training) — but the verify forward is **full-reveal-within-block, NOT the serial conditional**, so
   the draft-verify path is a **new decode envelope**, not lossless by construction.
2. **Corrected anchors:** serial-faithful acceptance 0.9865–0.9916/token (whole-32: 0.65–0.76); full-reveal 0.9973–
   0.9975; V1 dose-response +0.96 pp/400 matched (not +2.6 pp); oracle ~14.6–15.8 tok/fwd; **blended est.
   ~1.6–1.8×** vs shipped 1.06×, ceiling 2.98×.
3. **Rung W-0 (~1.5–2 GPU-h, training-free):** full-window probe re-run logging staged serial-faithful softmax
   (E[p̃(gold)] at the frozen envelope), full-reveal divergence, false-accept battery, + rejection-sampling sim.
4. **W-0 kill:** proceed iff serial-faithful sim ≥ 8 tok/fwd on copy mass, blended ≥ 1.5×, false-accept = 0,
   leak gap ≤ 15 %; ā < 0.98/token ⇒ envelope tax was the hidden wall ⇒ training becomes primary.
5. **Rung W-1 (3–5 GPU-h):** engine prototype in the CAD value branch (pin's variable-accept GDN scatter; APC
   never-remask); kill: on-policy value < 1.6× (soft) or any KILL-T1/corruption trip (hard).
6. **Rung W-2 (4–6 GPU-h):** THE promotion event — golden 19/50 §C both gates + derived-byte audit; no ship
   without it.
7. **Training (T-a/T-c, ≤ ~30 GPU-h):** W-B3 synthetic gym (2–4 h, decisive architectural negative available) →
   W-B1 certainty-forcing self-distillation vs W-B2 glancing arm to the 8M gate (~8 h) → winner to 128M;
   frontier-keyed kills, exact_args anchor every checkpoint.
8. **Conditioning (T-b, ~3–4 GPU-h):** C2 evidence-adjacency, C3 position-coupling, C4 retrieval-head steering —
   probe-first, pin-feasibility-gated, compose with verify.
9. **Budgets:** decode path ~9–13 GPU-h to a shipped rung; grand worst-case ~45; earliest decisive kill ~2.
10. **Highest-risk assumption:** frozen-envelope acceptance **E[p̃(gold)] ≥ ~0.98/token on copy positions** — every
    accept bit measured so far is greedy argmax at the *wrong (full-reveal) conditional*; the whole ~1.6–1.8×
    pricing and the rung ordering rest on that one unmeasured number, which W-0 reads first.

---

## DIRECTIVE-5(2026-07-14) — USER DESIGN REFINEMENTS TO THE W-LADDER DRAFT-AND-VERIFY RULE

Two user-directed refinements, binding on W-0 measurement and the W-1 prototype:

1. **RECENCY-FIRST candidate ordering.** Mine/verify candidates in REVERSAL order of context (most recent
   first). Rationale: agentic-SWE copy locality (old_string ≈ the last read; paths ≈ recent tool outputs);
   recency-first (a) resolves near-duplicate ambiguity in large contexts without extra verify cost — the primary
   mitigation for the large-window false-accept risk, and (b) drives expected candidates-verified-per-span toward
   1. W-0 MUST report: recency-hit-rate (fraction of gold spans whose gold candidate is the most-recent mined
   match) + the source-distance distribution (tokens between span and its source).
2. **CHEAP, BATCHABLE VERIFY.** Verification must be designed batched: (a) candidate-level — m candidate
   canvases for one span batch as sequences in ONE forward; recency-first keeps m small; a MOVING WINDOW over
   the candidate list caps batch size when mining returns many; (b) span-level (measure before trusting) —
   multiple drafted spans in the same block verified in ONE joint canvas/forward; W-0/W-1 must quantify the
   COUPLING RISK (wrong draft at span A perturbing verification of span B) before joint-verify is enabled;
   fallback = per-span verify. Verify cost model (forwards per committed token, batched) is a required W-0 output
   alongside the acceptance numbers.

---

## DIRECTIVE-5(2026-07-13) — USER: RECENCY-FIRST VERIFICATION + CHEAP/BATCHABLE VERIFY

Two user design rules for the SECTION-W draft-and-verify mechanism, binding on W-0 (measure) and W-1 (implement):

1. **RECENCY-FIRST candidate ordering (most-recent context first).** Mining ranks candidates by recency of their
   source occurrence (reverse context order); verification proceeds most-recent-first with EARLY EXIT on first
   accept. Rationale: in agentic SWE the correct copy source is near-always the most recent occurrence (the file
   just read, the path just emitted); stale near-duplicates deeper in context are exactly the false-accept
   distractors. Expected effects to MEASURE in W-0: (a) verify-rounds-to-accept distribution under recency order
   vs naive order; (b) false-accept reduction from recency priority (the perturbed/distractor battery re-scored
   with recency rank as tiebreak).
2. **VERIFY MUST BE CHEAP AND BATCHABLE — moving window if the candidate set exceeds the batch.** Multiple
   candidates for a span verify in ONE batched forward (m canvases, batch dim m); measured engine batch curve
   (18.7→35.3 ms/fwd b1→b16, p2_batched_rollout_bench) prices m=4–8 at ~1.5–2× a single forward, far below m
   sequential rounds. If candidates exceed the batch cap, verify in RECENCY-ORDERED CHUNKS (moving window) with
   early exit. W-0 must report the implied verify-cost model (expected forwards/span = f(candidate count, batch
   cap, recency-hit rate)); W-1 implements batched verify natively in the FLARE loop (sync-scheduler compatible —
   the m canvases are independent, no cross-request state).

These compose with the W-0 extra readouts already pinned (clamped-trace in-window coverage; ambiguity vs context
depth): recency-first is the designed MITIGATION for the large-context ambiguity risk — W-0 measures whether it
suffices (false-accept 0 bar unchanged).

---

## STATUS(2026-07-14) — C46 RE-GATE (ITER-2) EXECUTED: **SPLIT VERDICT — CAPABILITY PROVEN IN THE WEIGHTS, DECODE MODE IS NOW THE SOLE BINDING CONSTRAINT**

Run of record runs/k_gate_c46_iter2/ (48 ids, pool sha 49d8f46d…, frozen envelope, official scoring, truth-telling
labels ACTIVE, twin served WITH the certified read-clamp).

- **AR arm (iter-2 SFT fold, AR-decoded): 12/48** — up from 7/48 (iter-1). The iteration-2 quality levers (383-pool
  + windowing) MOVED the capability ceiling; the SFT weights now demonstrably carry entry-gate-level capability.
- **twin@K1+clamp: 1/48** (iter-1: 3/48) — ENTRY BAR ≥12/46 **NOT MET**; twin⊂AR, McNemar b=0/c=11, p=0.00098.
- **LOCUS_VERDICT: A — conversion/decode-mode-specific deficit.** Same weights: AR 12/48 vs diffusion-K=1 1/48.
- Truth-telling first outing: twin failure mass = **21/48 ctx_overflow_deaths** (honestly labeled, no longer
  clean-quit masquerade), 23 empty patches, 13 committed edits, 6 loop halts. The certified clamp (limit=100) did
  not suffice — non-read bloat routes still drive the context wall (consistent with the iter-1 PARTIAL finding).
- KILL-3 note: value-projection tripwire fired on 14/1201 served requests (1.17%) — projection-immune for the
  docker-scored verdict; flagged for any future served-engine tok/fwd measurement.

**ADJUDICATION (supersedes the pre-registered "principled stop"):** the stop's premise — "a second miss means the
ceiling is the 9B student" — is REFUTED by the AR arm's 12/48. The student is NOT the ceiling; the K=1 diffusion
decode is. The quality goal and the speed goal have therefore CONVERGED on the same object: the decode rule. The
W-ladder (draft-and-verify with DIRECTIVE-5 recency-first batched verification) is now load-bearing for BOTH goal
halves; W-0 launches immediately on the freed GPU. A W-1 engine prototype, if W-0 proceeds, gets gated on C46
under the NEW decode envelope (same 48-id pool, same scoring) — that is the next legitimate twin entry attempt.
