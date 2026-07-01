# GDN+Diffusion 10x Training-Redesign Research (workflow wfm2y1y2t, 2026-07-01)

Verified: 9/9 mechanisms confirmed real (1 verify agent errored on WeDLM). Papers: ParallelBench 2510.04767, CD4LM 2601.02236, D2F 2508.09192, Di4C 2410.08709, SDTT 2410.21035, Duo 2506.10892, CoDD 2603.00045, BD3-LM 2503.09573, B3D-RWKV 2605.25969.

# Training Redesign: Big-Block-at-Quality for GDN-Hybrid Block Diffusion

## 1. ROOT CAUSE — why big-block-at-quality fails on *our* stack specifically

Three failure sources compound, and our own numbers let us separate them:

**(a) The factorization barrier is real but it is NOT uniform across token types.** A single forward emits factorized per-position marginals, so it cannot commit token *k+1* conditioned on *k* in the same parallel step. Our measurements: `value_tpf 1.007–1.016` (pure sequential), `structural_tpf 1.132–1.175`, blended tool-call held-quality ceiling ~1.0–1.1x, Countdown ~1.7x. This is the wall — but ParallelBench (2510.04767) supplies the missing organizing variable: **conditional entropy `C(Y|X)`**, not surface difficulty. Spans where the output is *determined by the input* (`C(Y|X)≈0`) are parallel-**robust** and, critically, their fine-tuning ablation shows training can make `C≈0` spans parallel-*exact* (Replace-Index <50% → ~100%, stable under parallelism). Spans with `C(Y|X)>0` (mutual dependence between co-emitted tokens) can *never* be made parallel-exact by any amount of training (corroborated by 2602.00286).

This bisects our own crux. Our residual heldout misses under the causal-value-span test were **reasoning / convention / selection / cross-call** (`0.15 vs 15 percent`, `device_001 vs device_002`, `session_id` cross-call) — these are the *paired/correlated* `C>0` cases. But **single exact-argument copy-from-context is `C(Y|X)≈0`** — and our forced/protected ceiling (11–12/12) plus raw-given-structure (7/12 heldout, 10/12 public) proves the *capability* is present. So a large part of what we've been calling "the factorization barrier on values" is a **training/capacity problem on `C≈0` spans that ParallelBench says is fixable**, wrapped around a genuinely-hard `C>0` core that is not.

**(b) GDN causal-copy-circuit disruption — the novel, un-derisked axis.** Our copy circuit lives in the 1-in-4 full-attention layers organized causally (induction heads); FLARE keeps GDN causal-within-block and makes *those attention layers* bidirectional-within-block. Making the copy circuit bidirectional is precisely the mechanism that can break left-to-right verbatim copy (source token `<k` and its copy at `k` must be jointly/orderly committed). Our causal-value-span test came back **INERT (byte-identical to raw at temp 0)** — so this is **untested, not refuted**. This is our single biggest unknown, and **the literature is empty here**: CD4LM/D2F/Di4C/BD3-LM are all full-attention (LLaDA/Dream); the only recurrent-backbone analog is B3D-RWKV (2605.25969), and it *cratered exactly on the dependency-heavy tasks* (GSM8K −12.4, MATH −25).

**(c) Chronic free-gen undertraining + the GDN state-write corruption trap.** We run ~200–1000 LoRA steps vs FLARE's 9000 full-FT (2–11% of budget); two-stream A/B showed conversion works at scale (init 0.05 → 0.65–0.70 GSM8K) but B did not beat A on generation. Layered on top: the GDN recurrent state must be **read-only during denoise, advanced once at a committed boundary** — writing tentative/masked tokens into `S_t` corrupts it. Our current FLARE schedule handles this via clean-boundary seeding, but any redesign that increases parallelism *must not* leak tentative tokens into GDN state.

**Net root cause:** our present objective trains a single, position-marginal, fixed-schedule denoiser. It over-invests capacity in `C>0` spans it can never parallelize, under-invests in making `C≈0` copy spans parallel-*safe*, and leaves the bidirectional-copy-circuit disruption completely unmeasured.

---

## 2. TRAINING-REDESIGN DIRECTIONS (ranked by 10x-payoff × single-5090-QLoRA feasibility)

### #1 — Conditional-entropy-adaptive block schedule + explicit copy-from-context grounding supervision
**Backing:** ParallelBench (2510.04767) — the only paper with a *training* result on parallelizability; BD3-LM (2503.09573) for the schedule mechanics.
**Exact training change:** (i) Add explicit **copy-from-context supervision**: synthesize/label spans that are verbatim copies of prompt/context (arg values, IDs, paths) and train the two-stream objective to commit the *entire* `C≈0` span jointly at high mask ratio (force ≥2 tok/forward on those spans). (ii) Make the block schedule **adaptive per-span**, not fixed: wide parallel mask on low-`C` spans (copy, JSON scaffold), narrow/near-AR on high-`C` spans (reasoning, paired values, cross-call). (iii) Train the schedule *in* — ParallelBench shows training-free remasking (ReMDM/RCR) gave zero gain, only fine-tuned remasking helped.
**Expected block-at-quality gain:** This is where our real distribution lives. Copy + JSON scaffold are the bulk of tool-call tokens; if `value_tpf` moves 1.01 → ~2–3 on `C≈0` spans at held exactness, blended tool-call goes from ~1.05x toward ~2–3x. On reasoning (Countdown/GSM8K) the adaptive narrow-on-`C>0` policy protects the 7/16 we currently lose at the tau=0.50 cliff.
**GDN fit:** Fits well — it does *not* require GDN to be bidirectional; wide parallel commit on `C≈0` spans still routes copy through the causal full-attn circuit if we keep those spans left-to-right-within-span. **This directly tests whether the copy-circuit disruption (root cause b) is fatal or trainable.**
**Key risk:** If GDN's bidirectional-within-block copy circuit *is* the wall (our INERT test), copy-from-context supervision won't lift `value_tpf` and we'll have falsified ParallelBench-transfer-to-GDN — which is itself the single most valuable thing to learn. Also: adaptive boundaries need a cheap `C`-proxy at inference (entropy/confidence threshold, re-tuned on a GDN heldout — the tau=0.9 from LLaDA/Dream is mis-calibrated for us).

### #2 — Frozen-backbone HMM joint prior (CoDD-style) as a decode+light-train stage
**Backing:** CoDD (2603.00045) — freeze backbone, cache logits, train a small HMM-structured probabilistic circuit (hidden ~1024) as an inter-token joint prior; sample the product (frozen potentials × PC prior) jointly per step.
**Exact training change:** After the base block-diffusion QLoRA, **freeze it**, cache per-position logits over a reasoning/tool-call corpus, train the PC with the discrete-diffusion loss over the product distribution. No backbone backprop. ~3 GPU-hrs, trivially fits the 5090 (frozen forwards only).
**Expected gain:** Paper's own ceiling is ~2x fewer steps at held accuracy (Dream-7B GSM8K@64 +22pts, shrinking to +4.6 at 256 steps — pure collapse-prevention). Contributes a **~2x factor**, largest exactly in the aggressive-block regime we care about.
**GDN fit:** Excellent on *cost* — backbone untouched, composes on top of any QLoRA checkpoint, no GDN-state risk (it's a decode-time joint sampler). **Untested interaction:** a left-to-right HMM prior over a GDN linear-attention block schedule is novel (empty literature); the HMM's own left-to-right structure may actually *complement* our causal copy circuit.
**Key risk:** Demonstrated on masked full-attention diffusion, not semi-AR GDN. It lets token `k+1` condition on `k` *within one step* — the exact factorization fix — but validated on GSM8K accuracy, not strict paired-value exact-match. Must measure step-reduction on *our* decoder before banking.

### #3 — Diffusion-forcing schedule for inter-block pipeline parallelism (D2F-style)
**Backing:** D2F (2508.09192) — our *exact* target topology (bidirectional-within-block, causal-across-block, KV/state reuse); LoRA distillation, ~96 A100-h, 2.5x vs strong AR on GSM8K.
**Exact training change:** Stop training all blocks at one sampled noise level. Under block-causal attention, sample a **per-block monotonically-ascending mask-ratio profile** (later blocks noisier; their min 0.3 / max 0.7, block 16) so the student learns to advance future blocks while earlier blocks are only partially denoised. Pair with a threshold-gated pipelined decoder (tune quality/speed post-hoc).
**Expected gain:** The credible lever for raising *effective* block size via inter-block pipelining. Honest vs-AR number is **2.5x GSM8K / 1.6x code** — the 52.9x is vs a near-broken diffusion baseline on boilerplate and does not transfer.
**GDN fit:** Architecturally the closest match to what FLARE two-stream already builds, BUT the pipeline's win rests on cheap KV-cache reuse across blocks; our **GDN linear-attention state accounting does not transfer 1:1** — the 75% GDN layers re-scanning the prefix is our bottleneck, and pipelining partially-denoised future blocks risks writing tentative tokens into `S_t` (root cause c). Diffusion-forcing across blocks with *different simultaneous noise levels* is exactly the schedule most likely to trip the state-corruption trap.
**Key risk:** (1) D2F is **distillation from a pretrained bidirectional dLLM teacher** — we don't have one; we'd substitute self-distillation or use our converted base as a weak teacher (unproven). (2) The forcing schedule may stress the pure-torch GDN recurrence memory-wise at block 1024 on 32 GB. (3) Must solve GDN snapshot-at-committed-boundary before pipelining, or corruption dominates.

### #4 — Mixture-of-denoisers head for paired/correlated values (Di4C-style)
**Backing:** Di4C (2410.08709) — the *only* mechanism aimed squarely at the `C>0` paired-value barrier; replaces factorized marginals with a mixture (marginalized latent index) so one step emits a correlated joint.
**Exact training change:** Add a small marginalized latent `z`; condition the denoiser on `z`, marginalize at sampling. Keep marginal CE as one term, add a teacher-compose-then-match term + mixture-consistency term. Requires a frozen many-step teacher to distill from.
**Expected gain:** ~2x, and it's the honest attack on `start_time/end_time`, `INV-301/INV-302` co-commit — the thing #1 and #2 *cannot* fully fix because it's genuinely `C>0`.
**GDN fit:** Neutral-to-awkward. The mixture adds parameters/latent structure, which fights QLoRA's adapter-only regime (LoRA rank may underfit the latent). No specific GDN conflict, but no GDN validation either.
**Key risk:** Its "correlation" is *distributional* (validated on perplexity/FID), **never on strict exact-token joint correctness of paired values** — the exact thing we need. Highest chance of "improves perplexity, still 0/28 strict." Prototype only after #1–#2 land.

### #5 — Clipped per-block-size noise schedule (BD3-LM) — free quality lever, not a speed lever
**Backing:** BD3-LM (2503.09573).
**Exact training change:** Replace mask-rate `U[0,1]` with clipped `U[β,ω]`, grid-searched to minimize NELBO gradient-variance (starts: `U[0.45,0.95]` small blocks, `U[0.3,0.8]` large). Recovered ~2 PPL points for free in their runs.
**Expected gain:** No direct throughput; it *lowers the quality cost at a fixed block size*, which indirectly lets #1/#3 push block size further at the same held-quality bar. Zero extra compute, drop-in to our two-stream QLoRA.
**GDN fit:** No conflict; pure data/loss change. **Adopt as a default under all other directions**, not as a standalone bet.
**Key risk:** None material — it's the safe hygiene change. Ceiling is a couple PPL points, i.e. modest.

---

## 3. HONEST VERDICT

**~10x at strict grounding quality is not physically reachable on this architecture.** The evidence is convergent and adversarial:

- Every "10x–128x" headline in the literature is measured against the diffusion model's *own* 1024-step or near-broken baseline, on unconditional text / boilerplate code, at 110M–860M scale, scored by *generative perplexity* (gameable, and silent on exact-token correctness). The moment you demand held task quality vs a strong AR model, the lever collapses to the published **2.5x (D2F GSM8K) – 5.18x (CD4LM GSM8K, full-FT full-attention)** band.
- Our own measured ceilings agree: 1.7x hard tokens, 1.0–1.1x real tool calls. The `C(Y|X)>0` correlated-value barrier is architectural and scale-independent (2510.04767, 2602.00286) — RL trained directly on raw rollouts did not move it off 0/16.
- **Realistic ceiling for us:** ~**2–3x blended at held grounding** on the real tool-call distribution (copy + scaffold parallelizes, paired values do not), plausibly **3–4x on reasoning-heavy distributions** (Countdown/GSM8K-like, where more of the block is `C≈0`-ish) if the full stack (#1 + #2 + #5) lands. Not 10x. Anything claiming 5x+ at *strict* grounding would be measuring the wrong metric.

**The single design choice that matters most: stop chasing a fixed big block; make parallelism conditional-entropy-adaptive per span, and add explicit copy-from-context grounding supervision.** ParallelBench proves `C≈0` copy is trainable-parallel-safe and `C>0` is not — so the entire game is (a) harvest wide parallelism only on `C≈0` spans, (b) accept AR-speed on `C>0` spans, (c) keep the constrained grammar decoder as the permanent structural component. This is the one lever that turns our "1.0–1.1x blended" into a real multiple *without* betting on beating an un-trainable-away barrier.

**Novel-surface flag:** GDN linear-attention + diffusion is empty literature. Whether our bidirectional-within-block copy-circuit disruption (root cause b) blocks even `C≈0` parallel-copy is **the untested pivot** — our causal-value-span test was INERT, so we do not actually know if ParallelBench transfers to GDN. Experiment #1 is designed to resolve exactly this.

---

## 4. THE FIRST EXPERIMENT — smallest run to test Direction #1

**Hypothesis (falsifiable):** On our GDN-hybrid two-stream model, exact-argument *copy-from-context* is a `C(Y|X)≈0` span that fine-tuning can make **parallel-safe** — i.e. it can commit in ≪1 forward/token at held exactness — as ParallelBench's Replace-Index result (<50%→~100% under parallelism) predicts. If it cannot, GDN's causal-copy-circuit disruption is the wall and we have refuted the transfer.

**Setup (single 5090, QLoRA, ~1 short run):**
1. **Data:** synthesize ~2–4k copy-from-context examples where the assistant span is a *verbatim copy* of value tokens present in the prompt (arg values, IDs, paths). Reuse the 48 synthetic + public tool-call pool; label the value spans explicitly. Native chat_template format throughout (per the format-consistency rule; native-everywhere teacher recovered ~24pts).
2. **Objective:** existing FLARE two-stream, plus force the *entire value span* to be masked at high ratio and supervised to commit jointly (target ≥2 tok/forward on the value span), left-to-right *within span* (keep the causal copy circuit intact — do NOT make the value span bidirectional). Add BD3-LM clipped noise schedule (`U[0.3,0.8]`) as the default (Direction #5, free).
3. **Budget:** ~300–600 LoRA steps, block_size 512 (stay off the 1024 OOM edge), batch-1. One evening.

**Decode:** raw (no grammar decoder, no value-forcing, no protected sidecar) at a swept parallel-commit threshold, re-tuned on a GDN heldout (do not reuse LLaDA's tau=0.9).

**Success metric (block-size-at-held-quality, on the *value span* only):**
- **PASS:** raw value-span exact-match holds ≥ baseline (7/12 heldout, 10/12 public) while **`value_tpf` rises from 1.01 to ≥2.0** — i.e. the `C≈0` span commits in ≤half the forwards at held exactness. This confirms ParallelBench transfers to GDN and greenlights the full adaptive-schedule build (#1) + stacking CoDD (#2).
- **PARTIAL:** exactness holds only up to `value_tpf ~1.3–1.5` — the copy circuit tolerates *some* parallelism; proceed but cap expectations at the ~1.5–2x blended regime.
- **FAIL:** exactness drops (7→<5/12) at any `value_tpf>1.1`, byte-identical-to-baseline INERT recurs, or corruption (dup substrings / dropped chars) reappears on the value span. This **refutes ParallelBench-transfer-to-GDN**, localizes the wall to the bidirectional copy-circuit disruption (root cause b), and redirects effort to CoDD (#2, decode-time joint prior that sidesteps the circuit) rather than schedule changes.

**Why this experiment first:** it is the cheapest run that discriminates between our two live root causes — "undertrained `C≈0` copy" (fixable, ParallelBench-optimistic) vs "GDN copy-circuit disruption" (architectural, our novel empty-literature wall). Every downstream direction (#2–#5) is conditioned on that answer, and we currently do *not* have it (the prior causal-value-span test was INERT, not informative). No new infra, no vLLM, no teacher dependency — pure QLoRA + existing eval harness.