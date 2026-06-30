# FLARE Reproduction Plan (+ our optimization surfaces)

Date: 2026-06-28. Goal: reproduce FLARE (token-equal two-stream AR+diffusion conversion of Qwen3.5 GDN →
diffusion) on our setup, then push past it on the surfaces FLARE left open. Faithful paper digest:
`flare_paper_digest.md`. Roles: monitor (Claude) writes/steers/red-teams; Codex executes; no promotion
without raw/constrained evidence.

## Why reproduce even though FLARE exists
- FLARE code/weights are NOT released — there is nothing to download; reproduction is the only path.
- FLARE has **zero agentic / tool-calling eval** — our original frontier is unclaimed.
- FLARE's own #2 limitation is a DATA-distribution gap from using EXTERNAL teachers; we have the
  **same-family Qwen3.6-27B teacher** to harvest aligned "diffusion-friendly traces" → a concrete way to
  beat FLARE's residual gap (this is also our Phase-B OPDLM idea).
- We already have: a converted Qwen3.5-9B Fast-dLLM diffusion init (`option_a_causal_gdn`, GDN kept causal),
  the slow no-cache full-context sampler (a perfect bit-exact golden reference), the eval battery, FLA
  (flash-linear-attention) with `chunk_gated_delta_rule` + `initial_state`/`output_final_state`.

## Our optimization surfaces (where "we find something to optimize")
1. **Agentic/tool-call extension** — reproduce the recipe, then add tool-call/agentic transfer data + a BFCL
   eval (the thing FLARE skipped and our project was always about).
2. **Same-family teacher data** — FLARE blames the residual gap on external-teacher distribution mismatch;
   distill Qwen3.6-27B (same family) traces matched to the diffusion block structure → close that gap.
3. **Single-5090 serving** — FLARE serves on A100; a correct, cached single-5090 Diffusion-Trust path is its
   own optimization target.

## Staged roadmap
- **Stage 0 — GDN state-snapshot validation harness (FOUNDATION, do first).** Bit-exact test that snapshot
  clean GDN state at a block boundary + re-scan == full recompute. Gates everything (FLARE Route I training
  AND the inference cache). Spec below.
- **Stage 1 — Two-stream training, Route I (correctness-first).** Implement L_FLARE = L_AR + L_diff: clean
  causal stream + two complementary noisy block-bidirectional views + logit shift, GDN noisy block seeded
  from clean block-boundary state (reset per block + per document), document-packed mask. Use Route I
  (materialize clean boundary states in HBM; reuses standard FLA kernels) — defer the fused Route II kernel.
  Validate train loss + a short conversion run; measure retention vs AR on the Phase-A battery.
- **Stage 2 — Data mix (FLARE's #1 lever).** Build Long-CoT + Math + IF mix; screen mixes cheaply via
  AR-SFT proxy (FLARE's method). Add OUR tool-call/agentic traces + same-family 27B-teacher diffusion-friendly
  traces (our edge). This is where the biggest quality gains live, per FLARE.
- **Stage 3 — Inference (Diffusion-Trust + AR-Trust + strided-checkpoint cache).** Build on the Stage-0
  golden reference. Denoise reads state read-only; commit advances state once per finalized block; AR-Trust
  rewinds state on partial accept. This is the speedup ("cache project"); FLARE Alg 1-2 + §5.3 are the spec.
- **Stage 4 — Eval + push past FLARE.** Retention battery (vs AR) + BFCL/tool-call (the unclaimed frontier).

Route II (fused two-stream kernel, strided checkpoints in registers) is an OPTIMIZATION for small B / memory;
only needed if Route I's L/B memory is infeasible at our chosen B. Start with Route I.

## Stage 0 spec (the first concrete step — light GPU, runs alongside the Phase-A matrix)
Inference-only test, no training. Base: `models/qwen3.5-9b-fastdllm-init` (and/or a small Qwen3.5/Qwen3-Next
GDN model if faster to iterate). Write ONE script that, for a short prompt + 2 blocks:
1. **Full-recompute path** (current correctness path): run the GDN layers over [prompt + block1 + block2]
   from scratch; capture the GDN-layer outputs (and final state) for block2 positions.
2. **Snapshot path**: run [prompt + block1], capture each GDN layer's `final_state` at the block-1 boundary
   via FLA `chunk_gated_delta_rule(..., output_final_state=True)`; then run block2 ALONE seeded with that
   `initial_state`; capture block2 GDN-layer outputs.
3. `assert torch.allclose(full, snapshot, atol=1e-3)` per GDN layer; also verify the causal ShortConv lags
   across the boundary are handled (block2's first W-1 positions must read block1's tail, not zero).
Outcomes:
- **Match** → GDN is causal-within-block (FLARE Route I + the inference cache are valid for our bridge) →
  green-light Stage 1. Record per-layer max abs diff.
- **Mismatch** → either the bridge made GDN bidirectional-within-block (then we need forward-cache +
  within-block-backward, FLARE Route I/II nuance) OR a ShortConv/boundary bug. Diagnose before any kernel work.
Guardrails: no training; reuse `.venv-fastdllm`; keep the test tiny (1 prompt, small block, 1-2 GDN layers
first then all); document any FLA-version API differences. CHECKPOINT with the monitor on the per-layer diffs
before declaring the gate cleared.

## Note on execution
This is a multi-week project; the current Codex session is context-heavy from the grounding/Phase-A work. A
fresh Codex session pointed at this plan + `flare_paper_digest.md` is the clean vehicle for the bulk. Stage 0
is small and self-contained enough to start in either.

## Stage 0 — PASSED (2026-06-29)
Harness: `scripts/validate_gdn_state_snapshot.py` (local bridge GDN, random weights, CPU, all 24 GDN layers).
Result: MATCH on every GDN layer — snapshot-state + block re-scan == full recompute to ~1e-7 (fp32, atol 1e-3).
Controls that make the pass non-vacuous: native-vs-manual GDN diff = 0 (uses the real
`torch_chunk_gated_delta_rule`, not a re-impl); ShortConv zero-tail control large (~1.2 → test is sensitive to
a boundary bug); zero-after-tail ~6e-8 (causality holds). The local `torch_chunk_gated_delta_rule` natively
supports `initial_state`/`output_final_state` (cache API already exists).
Verdict: the GDN bridge is CAUSAL-WITHIN-BLOCK → the state-snapshot cache is PROVEN VALID (FLARE Route-I trap
#3 cleared). REPRIORITIZED: do **Stage 3 (inference cache)** next — now de-risked and it directly unblocks the
slow-eval problem — before Stage 1 training. (Did NOT use/install external FLA; the bridge has its own torch GDN.)

## Stage 3 (inference cache) outcome — 2026-06-29
Code: `scripts/validate_qwen35_state_cache_sampler.py`, `scripts/validate_qwen35_real9b_cache.py`,
`scripts/measure_qwen35_cache_gsm8k.py`. Design: snapshot committed-block GDN state + attn KV at block
boundaries, recompute only the active block per denoise step; read-only denoise + commit-once-on-finalize
(corruption guard); shifted-logits handling; follows the bridge's actual attention mask (currently CAUSAL
in inference, mdm_split_size=None).
- LOGIC correct: CPU/fp32 token-EXACT vs both golden samplers (~2e-7).
- bf16 9B: NOT bit-exact — GDN drift compounds from layer 0 (0.008 → ~0.5+ at logits), only 1/8 token-exact.
  bf16 GEMM/recurrent kernels are shape-dependent (active-block vs full-context shapes round differently).
  Expected; FLARE itself validates by eval-equivalence, not bit-exactness.
- Speedup: **~2.17× at 384 generated tokens** (real but modest; not the projected 3–8×). Short-seq 32/64 ≈ 1–1.6×.
- Quality-equivalence: **VACUOUS** — golden and cached both scored 0/3 GSM8K on the 3-doc slice → "same metric"
  = "both fail identically", NOT proof of quality preservation. Drift large (max 6.6).
- Side-signal (weak, N=3): DIFF_INIT scored 0/3 GSM8K vs AR ~0.9 → hints the conversion degraded math, as
  FLARE predicts (the Phase-A retention question peeking through).
Verdict: standalone cache = modest (~2×), quality-UNPROVEN; the easy-win thesis is dead. A trustworthy fast
model comes from FLARE training (model trained robust under the state schedule), which also recovers capability.
DECISION PENDING (escalated): (A) resolve cache quality on a solvable slice, or (B) pivot to FLARE Stage 1
two-stream training [monitor recommends B; banks the cache logic + ~2× for later].

## Stage 1 — two-stream FORWARD validated (2026-06-29)
Harness: `scripts/validate_flare_two_stream_forward.py` (validation-only, CPU fp32 random weights;
modeling.py, QLoRA, data pipeline, training services, GPU all UNTOUCHED — reversible).
KEY FINDING: the current training forward is **DIFFUSION-ONLY** (two noisy views, no clean AR stream, no
explicit logit shift) = the pure-block-diffusion FLARE shows degrades capability ~20 points. This is the
missing piece, and the likely cause of DIFF_INIT's degraded math (0/3 GSM8K).
5 tests, FINAL: PASS:
- mask rules: PASS (true_edges=44) — document-packed clean/noisy visibility rules.
- **clean logits vs AR: PASS, clean_logits_max_abs_diff=0** — the two-stream CLEAN stream is byte-identical
  to the AR/causal forward (the +14pt recovery path preserved). [the key result]
- GDN schedule/doc reset: PASS — seeded==full (2.98e-07), route_seeded=0, doc_reset_ref=0,
  wrong_cross_doc_sensitivity=0.88 (real control). (Codex caught + fixed a vacuous self-comparison first.)
- loss/logit-shift indexing: PASS — explicit FLARE shift ≈0 loss, wrong no-shift = 2.4 (shift validated).
- noisy finite/complementary: PASS — L_AR/L_diff finite, complementary views cover each target exactly once.
Verdict: the FLARE two-stream forward mechanism is CORRECT on this bridge. NEXT (the real, less-reversible
commitment — needs the A/B decision): integrate into the training path (modeling.py forward + data-pipeline
doc_id propagation) + a small two-stream training run. HOLDING here for A/B before the training commitment.

## A/B resolved → B, full autonomy, now in production (2026-06-29)
Lead granted full autonomy ("production do not wait, use best judgment") → proceeding with B (FLARE
two-stream conversion). Trainability smoke PASSED (`scripts/smoke_flare_two_stream_trainability.py`:
total/AR/diff losses overfit a tiny batch to ~0 in 50 CPU steps; separate AR/diff grad probes finite+nonzero).
All validation rungs now green (GDN snapshot → two-stream forward clean==AR → trainability).
PLAN: (1) production integration — port the validated two-stream objective into modeling.py TRAINING forward
behind a `FLARE_TWO_STREAM` flag (so diffusion-only stays available for comparison) + data-pipeline doc_id
propagation; (2) CONTROLLED PILOT — diffusion-only vs two-stream on the SAME small slice + budget, compare
retention vs AR; go/no-go = does the clean AR stream RECOVER capability (the FLARE +14pt effect); (3) SCALE
with the FLARE data mix (Long-CoT+Math+IF + tool/agentic + same-family Qwen3.6-27B-teacher traces — our edge
over FLARE's external-teacher residual gap). `flare` is the single mutator; checkpoint-first review on each big step.

## Route-II speed/mem fix — CPU validation outcome + GATE CORRECTION (2026-06-29)
User directive: "do the speed and mem fix according to the paper." `flare` implemented flag-gated Route-II
(`FASTDLLM_FLARE_GDN_ROUTE=route_ii`: strided clean-state checkpoints + `torch.utils.checkpoint` activation
ckpt; default stays route_i). CPU bit-exact validation result:
- **Forward loss + outputs BIT-EXACT (0.0)** vs Route-I → state reconstruction / clean→noisy seed / doc-reset
  are exactly right (the forward consumes the reconstructed states; a dropped term would diverge here).
- **Gradients NOT bit-exact: ~1e-7 (9.5e-7, ~8× fp32-eps) to 5e-5 worst; param grads 3–6e-8 (sub-eps).**
- flare's isolation (excellent): with `FASTDLLM_FLARE_ROUTE_II_CHECKPOINT=0` the diff PERSISTS and is
  **localized to the CLEAN recurrence grad** (noisy grad = 0.0 bit-exact). So NOT the checkpoint recompute —
  it's the strided **windowing** changing GEMM/chunk shapes → different fp32 reduction ORDER in the backward.
  Float non-associativity, the SAME physics as our documented bf16-cache shape-dependence. Inline Route-I
  reproduction is grad-identical → the test harness itself is valid (not broken).
- flare correctly HELD my `torch.equal`-on-grad gate and refused to promote / run the GPU gate on red.
**GATE WAS WRONG (my error), corrected:** `torch.equal` on gradients is incoherent for a reduction-order-
changing optimization that runs in **bf16** (per-elem grad rounding ~8e-3 rel = ~1000× larger than Route-II's
~1e-5 rel fp32 divergence). It also does NOT compound across steps (optimizer re-anchors on the bit-exact LOSS
each step; unlike the inference-recurrence drift). Corrected gate: (1) HARD forward-loss bit-exact (0.0);
(2) grad max RELATIVE diff ≤ 1e-3 (report actual; expect ~1e-5); (3) REQUIRED bug-injection controls
(zero boundary seed / off-by-one doc-reset / wrong stride offset / ShortConv zero-tail) must each drive fwd+grad
diff ≫1e-3 — proves the relaxed tolerance BITES (not vacuous, the recurring discipline lesson); (4) characterize
the 5e-5 as stride-monotonic (benign) vs a boundary-case outlier (investigate). IF all hold → GPU production
gate measuring ACTUAL peak-mem + step-time Route-II vs Route-I.

## Pilot (Route-I, superseded 10-step) FINISHED — confirms the speed/mem problem + an NLL-metric caveat
- **53× slowdown confirms Route-II is essential:** Route-I `two_stream` = 2002s/10steps (~200s/step) vs
  diffusion-only = 37.7s/10steps (~3.8s/step). Far worse than FLARE's theoretical ~2× → near-OOM thrash
  (31.9GB). Route-II target: kill OOM headroom + pull step time back toward ~2× of diffusion-only (single-digit s).
- **NLL-metric caveat for the NEXT recover-capability test (important):** ar_baseline NLL(all)=0.9366,
  init NLL(all)=0.9368 — essentially IDENTICAL. Because this heldout NLL is **causal teacher-forced**, it
  measures the CLEAN stream, which we PROVED is byte-identical to AR (`clean_vs_ar_logits=0`). So causal NLL
  is INSENSITIVE to diffusion-mode degradation and CANNOT be the recover-capability discriminator (it would
  give another vacuous "no difference," like the vacuous cache quality-equiv). The recover-capability test
  must measure **diffusion-mode** quality (actual block-diffusion generation, or heldout diffusion-loss),
  not causal NLL. (A/B adapters were not yet diffusion-eval'd; 10 steps is too tiny regardless.)

## Route-II GPU production gate — CORRECT but INERT (2026-06-29); pivot to the real bottleneck fix
CPU gate first PASSED decisively: combined `torch.allclose(atol=1e-3,rtol=1e-3)` — legit Route-II max margin
0.0104 (≈1% of tolerance), 4 injected-bug controls fail by **766×–8798×** (≈70,000× worst-legit↔weakest-broken
separation); forward scalar loss bit-exact at strides 2/4/8 (stride-1 single fp32-eps 9.54e-7 on partial-final).
Route-II math = CORRECT. THEN the GPU 9B 1-step measurement:
- Route-I two_stream: peak **31175 MiB**, 203.4s, loss 7.18882.
- Route-II stride-8:   peak **30877 MiB**, 204.8s, loss 7.18876.
- = **298 MiB saved (0.96%), ZERO speedup (1.4s slower).** Loss matches to 5e-5 → bit-exact in practice.
**Root-cause red-team:** Route-II reduces GDN boundary-STATE materialization (FLARE Fig 7: 18→0.45 GiB) — NOT
our bottleneck. Tell: Route-I and Route-II have ~identical step time despite Route-II materializing fewer states
⇒ the 200s is shared by both, not boundary-state compute. Our bottleneck is (a) the concatenated-2L clean+noisy
ACTIVATION memory at **31175/32607 = 95.6% (near-OOM)** and (b) 200s/step. FLARE's Route-II win comes from
**FUSED KERNELS (registers)**, which we deferred — the pure-PyTorch port has the logic without the kernel win
(paper: "kernels mitigate, don't eliminate"). Route-II kept flag-gated + banked (correct, validated); NOT promoted
as "the fix"; fused Triton kernel = separate deferred project.
**DECISION (autonomous, within "accept slow" + full-autonomy grant):** apply the standard high-leverage fix our
bottleneck actually needs — (1) WHOLE-MODEL gradient checkpointing on the two_stream path (recompute whole-block
activations in backward → attacks the 2L activation peak; needed for survivability regardless — a multi-hundred-
step pilot at 95.6% peak OOM-crashes partway), (2) `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. PLUS a
fwd-vs-bwd `torch.cuda.synchronize` timer split to localize the 200s (forward Python-block-loop vs backward
memory-peak). GOAL: peak < ~28 GiB (survivable). If speed also drops → bonus; if it stays ~200s but peak is SAFE
→ LAUNCH the meaningful A/B recover-capability pilot overnight at this config (the decisive test), kernel deferred.

## FORWARD OPTIMIZATION — DRAMATIC bit-exact win (2026-06-29); standing GPU-util rule set
Lead set a STANDING TEAM RULE: never accept ~10% GPU utilization — a starved GPU is an implementation defect to
profile + FIX, not band-aid (memory: `gpu-utilization-standard.md`). First application = the 187s two-stream
forward at 8% util. Profile-guided diagnosis (flare, following the data over hypotheses):
- torch.compile the GDN scan: NOT viable in the QLoRA/bnb-4bit path (Inductor sat CPU-bound, GPU 0%, never
  finished a forward). Aborted.
- GDN scans are only ~7.4s (my "redundant scans" guess was WRONG); gradient-checkpoint wrapper was a red herring
  (checkpoint-off changed forward 96.3→94.0s only, peak unchanged → ckpt ~neutral here, not the cost).
- **ACTUAL DEFECT: `.item()` loops in mask/position construction** — `local_position_ids_from_doc_ids`,
  `doc_causal_bool_mask`, `flare_two_stream_bool_mask` did per-element `.item()` in nested Python loops →
  thousands of host-device syncs → GPU starved. ~89s of the ~97s forward was pre-layer mask construction.
- **FIX: vectorized those helpers** (modeling.py). **Validation BIT-EXACT (byte-identical, not just allclose):**
  prod gate prod_vs_helper_logits=0 / loss=0 / clean_vs_ar_logits=0; old-mask vs vectorized-mask fp32 grad check
  loss_abs=0 logits_abs=0 grad_max_abs=0, ALL param grads torch.equal. (Vectorizing mask construction yields the
  SAME mask → identical math; cf. Route-II which changed reduction order → fp-eps.)
- **PERF:** 1536 forward **186.91s → 7.54s (24.8×)**, util 8%→47-77%, loss/grad unchanged (7.188815 / 4.515568);
  1024 forward 93.96→5.37s, util 8%→60%, peak 28859 MiB (safe). Step now ~28s (1536) / ~19.5s (1024) vs ~207s.
  → the pilot that would've been ~16h at 200s/step is now ~1.2h. Mask-vectorization PROMOTED; Route-II banked.

## DECISIVE A/B recover-capability pilot — LAUNCHED (2026-06-29)
`qwen-flare-stage1-ab-s1024-step200.service` (systemd transient, survives the tool shell — fixes the earlier
background-launch deaths). Controlled: A=diffusion-only vs B=two-stream, SAME data/seed/budget, BLOCK_SIZE=1024,
MAX_STEPS=200, GRADIENT_CHECKPOINTING=0, optimized vectorized-mask path. Arm A runs first (~2.4s/it), then B
(~19.5s/it); ~1.2h total. Artifacts under `runs/flare_stage1_ab_pilot/optimized_ab_s1024_step200/`.
**EVAL-VALIDITY GATE (locked with flare):** go/no-go metric MUST be DIFFUSION-MODE capability (block-diffusion
generation on heldout GSM8K/MBPP/IFEval, or diffusion-mode heldout loss), NOT causal teacher-forced NLL (which
measures the AR-identical clean stream → the earlier vacuous null ar 0.9366 vs init 0.9368). flare is preparing
the diffusion-mode eval OFFLINE (no GPU contention) and will CHECKPOINT the eval plan with me before running.
Interpretation guardrail: 200 steps ≈ 2% of FLARE's 9000-step budget → a null A==B is INCONCLUSIVE (too few
steps); a positive B>A is meaningful. Scale steps if null.

## DECISIVE A/B RESULT (2026-06-29) — weak-positive, INCONCLUSIVE on recover-capability
Denoising-NLL, fixed masks, 40 disjoint heldout examples (disjointness.json: 0 overlap), deterministic,
apples-to-apples (same masks init/A/B), eval GPU util 75-80% (rule satisfied):
| model | all NLL | GSM8K | MBPP |   | Δ | A-init | B-init | B-A |
| init  | 5.3748 | 4.9160 | 6.4715 |  | all | -0.3204 | -0.3812 | **-0.0608** |
| A diff-only | 5.0544 | 4.6172 | 6.0995 | | gsm8k | -0.2988 | -0.3478 | -0.0490 |
| B two-stream | 4.9936 | 4.5682 | 6.0105 | | mbpp | -0.3721 | -0.4611 | -0.0890 |
**Red-team verdict (honest):** B<A is REAL (deterministic) + CONSISTENT (all 3 favor B) but WEAK (-0.06..-0.09)
and **INCONCLUSIVE on recover-capability**, for two reasons: (1) BOTH A and B IMPROVED vs init → no degradation
visible — expected, since 200 steps is far too few AND denoising-NLL **structurally cannot show FLARE's
degradation** (both arms train on the denoising objective so both improve it; FLARE's −21.8 is a downstream
GENERATION-capability phenomenon that can worsen even as denoising-NLL improves). (2) The metric that WOULD test
capability — **generation — is NON-FUNCTIONAL** (~12 min, no first example; cacheless sampler too slow). My own
elevation of denoising-NLL to co-primary was a reasonable fallback but the deeper truth is it's an insufficient
proxy: denoising-NLL and generation-capability can diverge, and FLARE's claim is about the latter.
**NOT promoted.** Critical path is now the GENERATION eval (without it recover-capability is unmeasurable at any
scale). NEXT (steered): (1) diagnose generation speed — denoise steps/example, fwd-time/step, tokens-committed/
step at threshold 0.9 (likely threshold mis-calibrated for GDN → ~1 tok/step → ~256 forwards; or another
host-bound defect; or needs the banked GDN cache) → propose fix; (2) prep (not launch) a scaled A/B re-run
(~1000 steps each, ~overnight at ~19s/step) to test whether B-A WIDENS with scale + give the fixed gen eval a
more-trained model. Generation bs=1 produced no table; banked GDN cache may be needed for a usable gen eval.

## GENERATION SAMPLER BUG FOUND (2026-06-29) — likely confounds prior "degraded capability" evidence
flare instrumented one example: at threshold=0.9 the sampler took 10 denoise forwards but **126 of 127
"commits" selected mask_id (151665) ITSELF** — leaving those positions masked → no progress → the 12-min grind.
Only 1 actual non-mask commit. GPU 53–59% during the slow run → NOT a host-sync defect; a sampler **logic bug**
(model assigns high prob to the mask token; sampler never bans it as an output). **FIX (bit-safe):** ban mask_id
in denoise sampling (`logits[..., mask_id] = -inf` before top-p/argmax/threshold), applied identically to all
models. Estimate: full example 12 min → **~3.05s** (threshold 0.2, 21 forwards, 226 tokens) → gen eval viable
(~9 min for 60 ex × 3 models). Threshold 0.9 too strict after mask suppression → grid 0.2/0.3/0.5, tuned for
COHERENT output (not just speed; sample gens to be inspected). Secondary: `past_key_values` is INERT in the
bridge (layers don't read/update it) → full-context recompute for now (fine for short eval; banked GDN cache is
the scaled-serving path, separate).
**RE-FRAME (important, to verify):** this mask-self-prediction bug very likely CONFOUNDED prior diffusion-mode
generation evals — notably "DIFF_INIT 0/3 GSM8K" read as *the conversion degraded math*. That may have been a
broken sampler, not real degradation. init's FIXED-sampler generation score directly tests this (if init now
solves GSM8K, the "degraded capability" premise was partly a sampler artifact). If confirmed, update
[[qwen-diffusion-experiment]] memory (the 0/3-GSM8K-as-degradation claim).
**DECISION (autonomous):** GREEN-LIT the fix + threshold grid + fixed-sampler init/A/B generation table. HOLD the
scaled 1000-step run (config prepped: A ~40-45min, B ~5.3h, ~6h total) — the cheap fixed-sampler 200-step
generation may already show the B-vs-A signal AND whether init is degraded; decide scale AFTER seeing it.

## FIXED-SAMPLER GENERATION — FREE-GEN COLLAPSES for ALL models (2026-06-29)
Liveness fix WORKS (no mask stalls; init t03: 2420 tokens, 77 tok/s, 0 unresolved, ~31s/5-ex, GPU 57%). But
**capability is FLOORED: init/A/B all 0/5 GSM8K + 0/5 MBPP** at threshold 0.3 (and 0.5 similar). I inspected the
raw generations directly: **DEGENERATE GARBAGE** — a coherent START ("To find the best", "To solve") then
collapse into `\n\n` + "# но" (Russian) repetition. Not coherent-but-wrong; it's collapse.
**Red-team read:** classic **teacher-forced denoising works (NLL fine, B<A) but FREE generation collapses** =
severe **UNDERTRAINING for free generation** (200 LoRA steps vs FLARE's 9000 full-FT), amplified by error
accumulation (coherent start → collapses as it conditions on its own garbage). **init collapsing too** ⇒ the
whole pipeline is undertrained for free-gen, NOT the A/B arms specifically — and consistent with the prior
golden-sampler 0/3 GSM8K (so that was undertrained free-gen, not purely the sampler-liveness bug; the re-frame
is half-right: liveness bug was real, but the bridge genuinely can't free-generate coherently yet).
**Implication:** recover-capability CANNOT be measured by generation at this scale — everyone's undertrained.
The teacher-forced denoising-NLL (B<A, weak) is the only signal available at 200 steps. **NEXT (steered):**
(1) confirm it's undertraining/accumulation not a block-transition bug — 0.5 grid sample text + threshold-0.9
probe + block-1-vs-later collapse characterization; (2) if undertraining → LAUNCH the scaled 1000-step A/B run
(prepped; ~6h) as the real test (even 1000 may be too few vs 9000, but reveals the trajectory). Escalated to lead.

## CORRECTION — collapse is a PREFIX-BLIND SAMPLER BUG, not undertraining (2026-06-29)
The bug-vs-undertraining gate (which I required before scaling) PAID OFF. flare characterized the collapse:
threshold 0.9 + mask-ban STILL collapses (only 8 tokens cleared naturally, 210 forced argmax → not premature-
commit); block 0 starts coherent then collapses *within the first block* → template-fragment repetition.
**ROOT CAUSE: `fast-dllm/v2/batch_sample` denoises only `x_t[:, -block_size:]` and relies on `past_key_values`
— but the bridge's decoder layers NEVER read the cache (inert). So generation is PREFIX-BLIND** (each block
denoised ~without preceding context) → collapse. The golden sampler uses full-context `use_cache=False`.
**My undertraining hypothesis was WRONG/premature — flare overturned it with a concrete mechanism** (like the
redundant-scan and checkpoint-wrapper red herrings before). Key synthesis: earlier golden-sampler 0/3 had
full-context but NO mask-ban (mask-self-prediction stuck); today's batch_sample has mask-ban but NO full-context
(prefix-blind). **Never tried BOTH** → that's the fix.
**DECISION (autonomous):** scaled run HELD (do NOT scale off a broken sampler — would have wasted 6h). GREEN-LIT:
eval sampler = full-context recompute (`model(input_ids=x_t, use_cache=False)`) + mask_id ban, BOTH together;
rerun ONE-example collapse check (init GSM8K-0). GATE: coherent → run full init/A/B generation table (the real
recover-capability read); still collapses → then undertraining/deeper, discuss before scale. Speed fallback if
full-context too slow for 60 ex = the banked GDN state cache, but coherence answer on one example first.

## Full-context + mask-ban — BLOCK 0 COHERENT, collapses at block 1 (2026-06-29)
Both sampler fixes active (full-context use_cache=False + mask_id ban). One-example result (init, GSM8K-0,
Janet's-ducks): **block 0 is COHERENT and correctly comprehends the problem** ("Janet's ducks lay 16 eggs per
day. She ... eats 3 eggs for breakfast and uses 4 eggs for muffins, so") — prefix-blindness GONE, mask-loop GONE,
212/256 natural threshold commits (vs 8 before), GPU 97-100%, 0.144s/forward, ~34s/example. **But collapses into
"3 3 3" repetition starting EXACTLY at block 1.** Huge step (model CAN read prompt + start real reasoning); the
remaining failure is sustaining past block 0. The sharp cliff at block 0→1 is the clue.
**Diagnosis path (steered), cheap→deep:** (1) GREEDY-REPETITION test — "3 3 3" at temp=0 is classic greedy
degeneration (latches onto "3" from the problem); rerun with temp=0.7 (fixed seed) + repetition penalty ~1.2 →
if coherence extends past block 0, it was decoding, not structural; (2) BLOCK-TRANSITION audit — does generation
use the SAME per-block GDN-state seed + position_ids + block mask + logit shift as the VALIDATED two-stream
TRAINING forward? cliff-at-block-1 smells like gen diverging from the bit-exact-validated training semantics
(same class as prefix-blind); fix gen to match; (3) only if both fail → undertraining, discuss scale. Pattern of
this whole gen-debug arc: the generation sampler was never validated against the training path → bug whack-a-mole
(mask-loop → prefix-blind → block-cliff), each fix revealing the next. Scale still HELD.

## DECISIVE VERDICT — model HAS capability; collapse is OOD sampler semantics + accumulation (2026-06-29)
**FORCE-FED-BLOCK-0 = EXISTENCE PROOF.** Force-fed 33 gold tokens ("Janet sells 16-3-4=9 duck eggs a day. She
makes"), then full-context+mask-ban denoised the next block → **"9 * 2 = <<9*2=18>>$18 ... #### 18" — the CORRECT
GSM8K answer (gold=18)**, 48 forwards, 0 mask selections, 7s. So the model genuinely HAS diffusion-mode
capability; free-gen collapse is NOT a dead/undertrained-to-uselessness model — it's (a) error accumulation from
imperfect self-conditioning + (b) an audited STRUCTURAL DIVERGENCE.
**Audit (file:line):** sampler denoises SINGLE-STREAM CAUSAL (`diagnose...:475`; eval mask causal unless
training+mdm_split_size, `modeling.py:1201`) — but the model was TRAINED with the FLARE two-stream BLOCK-DIFFUSION
mask: noisy block bidirectional-within-block, attends preceding clean, clean/noisy position ids, clean-seeded
noisy GDN route (`modeling.py:1893`, `validate_flare_two_stream_forward.py:205`). Logit shift matches. So
GENERATION runs the model OUT-OF-DISTRIBUTION (causal) vs how it was trained (bidirectional block-diffusion).
**This reframes the prior "DIFF_INIT 0/3 = conversion degraded math":** largely a SAMPLER artifact (mask-loop →
prefix-blind → causal-OOD), not capability loss. Update [[qwen-diffusion-experiment]] memory once corrected-sampler
FREE-gen confirms.
**DECISION (autonomous):** GREEN-LIT the principled fix — run generation's active block through the TRAINED
block-diffusion semantics (mirror the validated training noisy-stream forward), NOT causal single-stream; VALIDATE
the gen-active-block-forward == training-noisy-stream-forward (the gen↔training tie never checked before); re-test
one-example free-gen. Coheres → run full init/A/B table (real recover-capability read, maybe without 9000-step
scale). Still accumulates → undertraining cleanly isolated → discuss scale. Scale HELD. Big lever: model
demonstrably works; just needs to run in the regime it was trained.

## CONSOLIDATED VERDICT → SCALE (2026-06-29)
The "run it bidirectional like training" hypothesis FAILED: corrected two-stream sampler →
"Janet's ducks lay ducks ducks ducks ... yas я я" GARBAGE, WORSE than causal (which gave a coherent block 0).
(Bidirectional parallel denoising of a fully-masked block collapses to a correlated high-freq token in an
undertrained model; causal commits left-to-right leveraging the preserved AR/clean stream, so it coheres longer.
Block trace `initial_masks=1` also flagged a possible diffusion-sampler masking bug — to verify in code.)
**Triangulated read (sufficient to decide):** (a) sampler bugs fixed (mask-loop, prefix-blind); (b) model HAS
capability — force-fed→correct "#### 18", AR preserved, denoising-NLL fine (B<A); (c) FREE diffusion generation
collapses at 200 steps regardless of sampler — the teacher-forced-vs-free-gen gap = **UNDERTRAINING for
bootstrapping**. Sampler bugs RULED OUT as sole cause. The real test is SCALE.
**DECISION (autonomous, per "production don't wait"):** (1) LAUNCH the scaled 1000-step A/B run (systemd, A
diffusion-only vs B two-stream, ~6h) — real recover-capability test via free-gen + trajectory probe; (2) in
PARALLEL (CPU code-only during training): audit/fix the diffusion sampler masking (`initial_masks=1`), note that
causal-full-context+mask-ban is the most coherent inference path so far; (3) post-scale: fixed-sampler free-gen
one-example check → full init/A/B generation table + denoising-NLL (does B-A WIDEN with scale?). No more
open-ended sampler whack-a-mole. This is the well-justified inflection: model works, needs scale + a correct
diffusion sampler.

## Scale run LAUNCHED + sampler masking bug FIXED in parallel (2026-06-29)
- **1000-step A/B run TRAINING:** systemd `qwen-flare-stage1-ab-s1024-step1000` (A diffusion-only first, then B
  two-stream), MAX_STEPS=1000 BLOCK_SIZE=1024 GRADIENT_CHECKPOINTING=0 expandable_segments, GPU **100% util**
  (rule satisfied), ~6h. Driver: `runs/flare_stage1_ab_pilot/optimized_ab_s1024_step1000/driver.log`.
- **Diffusion-sampler masking bug CONFIRMED + fixed (CPU, parallel, no GPU contention):** the `initial_masks=1`
  flag was real — the full-context diffusion path was NOT fresh-masking the active block (tail-fill mode).
  Fix: full-context path now defaults to **fresh 32-mask blocks** (`diagnose_flare_generation_speed.py:483`;
  `eval_fastdllm_toolcall_cases.py:1675` golden `full_context_sample` gets `--fresh-generation-blocks`;
  `--tail-fill-generation` kept as compat). py_compile clean. → This means the earlier bidirectional "ducks
  ducks" collapse was PARTLY this masking bug, not pure undertraining; the diffusion sampler will be correct
  (fresh-masked blocks + mask-ban + full-context) for the post-scale eval. The causal-sampler evidence
  (coherent block 0 → accumulation collapse; force-fed→correct) still supports undertraining for free-gen.
- POST-SCALE PLAN: corrected-sampler one-example free-gen check → full init/A/B generation table + denoising-NLL
  (does B−A widen with scale? does free-gen cohere with more training?).
- **Arm A (diffusion-only) DONE @1000 steps** (2026-06-29 13:14): train_loss 7.74→**3.4278** (substantial
  denoising-objective learning over 5× the steps), 40 min, peak 27889 MiB, adapter saved. **Arm B (two-stream)
  training** (~18.7s/it, ~5h). train_loss is the objective, NOT capability — the decisive read is the post-arm-B
  eval (free-gen + heldout denoising-NLL). No capability conclusion from train_loss.

## 4th SAMPLER BUG (tail-fill vs fresh-block) + scale run TRAINING (2026-06-29)
Scale run LAUNCHED + training: `qwen-flare-stage1-ab-s1024-step1000.service` active, arm A (diffusion_only) at
~171/1000 @ 2.38s/it, GPU 100%. Arm B (two-stream) follows; ~6h total. Committed + pushed (65495f2).
**Masking audit CONFIRMED the `initial_masks=1` red flag was a real bug:** the old sampler **tail-filled the
prompt remainder** instead of starting a **fresh 32-token masked active block** — GSM8K-0 (prompt len 799)
produced initial_masks=1. Existed in BOTH batch_sample AND full_context_sample. Fix (CPU-only, during training):
`fresh_generation_blocks` — active block starts with 32 masks (`fast-dllm/v2/generation_functions.py:38`,
gitignored→needs patch; `scripts/eval_flare_stage1_ab_diffusion.py:540`, `diagnose_flare_generation_speed.py:483`,
`eval_fastdllm_toolcall_cases.py:1675`). py_compile clean; no GPU eval during training.
**HONESTY CAVEAT on the "undertraining" conclusion:** the bidirectional "ducks ducks" garbage AND the
causal-collapse tests were ALL run with this tail-fill bug present → the free-gen evidence is CONFOUNDED. The
"undertraining" verdict is therefore PROVISIONAL. The clean test = post-scale free-gen with the FULLY-CORRECTED
fresh-block sampler. (Force-fed→correct "#### 18" still stands as the existence proof that capability is there.)
This is the 4th gen-sampler bug (mask-loop → prefix-blind → causal-OOD → tail-fill) — reinforces that the
generation path needs a validation harness vs the bit-exact training forward, like training had. Scale run
proceeds regardless (it's training data, independent of the eval sampler); post-scale we eval with the corrected
sampler — which may show coherent free-gen even pre-scale, or cleanly confirm undertraining.

## 1000-step run COMPLETE → post-scale eval launched (2026-06-29 18:30)
A diffusion-only loss 3.4278 (40min), B two-stream loss 3.9430 (5.25h) — NOT comparable (different objectives);
no capability read from train_loss. Both adapters saved, run inactive, GPU freed.
EVAL LAUNCHED (NLL-first ordering — learned from the 200-step round where slow generation gated the decisive
metric): (1) denoising-NLL init/A/B @1000 vs the same disjoint heldout-40 — **KEY READ: does B−A WIDEN vs the
200-step −0.0608?** (widen = recovery emerging with scale; flat/shrink = two-stream benefit marginal); (2)
corrected fresh-block sampler one-example free-gen coherence gate (cohere at 1000 vs the 200-step collapse?);
(3) if coherent → full init/A/B generation table (per-example GSM8K/MBPP + B−A + AR ceiling). Red-team each;
do not promote. (flare flagged the undertraining verdict as PROVISIONAL — correct, given the tail-fill was the
4th sampler bug; the corrected-sampler post-scale free-gen is the clean test.)

## POST-SCALE DENOISING-NLL — B−A WIDENS ~4× WITH SCALE (2026-06-29, red-teamed raw numbers)
Deterministic, disjoint heldout-40 (0 overlap). @1000 steps: init all=5.3748 (unchanged), A=3.7325, B=3.4806.
Deltas @1000: all A-init=-1.642 B-init=-1.894 **B-A=-0.2519**; gsm8k B-A=-0.2582; mbpp B-A=-0.2369.
**B-A widened from -0.0608 (@200) → -0.252 (@1000): ~4.1×, CONSISTENT across all 3 slices.** Real, clean signal:
the two-stream objective is increasingly better at diffusion-mode modeling than diffusion-only as scale grows.
**Mechanistically meaningful:** B beats A at DENOISING despite "spending" half its objective on L_AR — the clean
AR stream preserves language modeling that *transfers* to better denoising (the FLARE mechanism), and it compounds
with scale. Controlled: same data/seed/1000 steps.
**Honest caveats:** (1) still the PROXY — denoising-NLL can't show FLARE's *generation*-capability degradation;
both A and B improve vs init (no degradation pattern, structural). (2) NOT compute-matched: B's two-stream forward
is ~8× wall-clock/step (5.25h vs A's 40min) and ~2× FLOPs/step (the known two-stream cost) — but it IS the
intended FLARE token-equal comparison (objective differs, same data/steps; the cost is part of the objective).
VERDICT: strong right-direction signal for two-stream; the headline recover-capability claim still needs the
GENERATION result (coherence gate + table, running next). Not promoted on NLL alone.

## DECISIVE POST-SCALE GENERATION RESULT (2026-06-29) — conversion WORKS; A≈B on generation
1000-step diffusion generation, corrected sampler (fresh-block + mask-ban + full-context), N=20 GSM8K + 20 MBPP,
0 unresolved masks:
| model | GSM8K strict | MBPP |
| init  | 0.05 (1/20)  | 0 |
| A diffusion-only | **0.70 (14/20)** | 0 |
| B two-stream     | **0.65 (13/20)** | 0 |
**TWO findings, both honest:**
1. **CONVERSION WORKS AT SCALE (major positive):** both A and B free-generate COHERENT, CORRECT GSM8K (verified
   text: full step-by-step reasoning → "#### 18", "#### 3"), ~0.65-0.70 = **~78% of AR's ~0.90** in diffusion
   mode, up from the 200-step collapse and the init-baseline 0.05. **Undertraining hypothesis CONFIRMED — scale
   fixed free generation.** (init baseline still 0.05 → it's the training, not the sampler, that builds capability.)
2. **A-vs-B = TIE on generation (disciplined NULL on the headline):** B 13/20 vs A 14/20 = 1 example, within N=20
   noise; high per-example overlap (both solve same 12; A+2 unique, B+1 unique). **The NLL widening (B-A=-0.25)
   did NOT translate to a generation-accuracy advantage.** The FLARE two-stream>diffusion-only effect does NOT
   appear in generation at our scale — likely because diffusion-only A hasn't DEGRADED yet at 1000 steps (FLARE's
   degradation was ~9× more steps); the effect may need larger scale to manifest. NOT promoting B>A.
**Caveats / pending verification (flare):** (a) strict(0.70)>flex(0.55) inversion for BOTH arms → flex extractor
likely buggy; trust strict (matches the verified-correct text). (b) MBPP=0/20 for ALL (incl. A which solves GSM8K)
→ verify whether code is genuinely unrecovered or the MBPP test-harness/extraction is broken. N=20 is coarse.
**ROADMAP IMPLICATION:** the AR→block-diffusion CONVERSION is now demonstrably viable on our setup (Qwen3.5-9B
GDN, single 5090, QLoRA). Two-stream is the principled objective to carry forward (better NLL, AR-preserving, the
FLARE recipe) even though A≈B on generation at this scale. Next phase = leverage the working conversion: data-mix
+ the agentic/tool-call extension (the project's actual goal + our edge over FLARE), rather than over-investing in
proving a small A-vs-B generation effect that needs ~9× scale.

## Scoring verification (flare) + USER DECISION → agentic phase (2026-06-29)
- **GSM8K flex inversion:** flex extractor buggy; trust **strict** (matches verified-correct text). A 14/20 vs B
  13/20 confirmed tied; no B>A.
- **MBPP=0 was a HARNESS ARTIFACT:** model generates valid code then leaks role text ("…user\nWrite\nassistant");
  the harness executes ALL of it → NameError. Sanitized (cut at role markers) rescore: init 0/20, **A 6/20, B
  3/20**. Code partially recovered (weak but nonzero), masked by broken scoring. (Fix generation stop behavior +
  the eval harness in the agentic phase — clean termination matters for tool-call generation.)
- **Sharpened verdict:** A ≥ B on BOTH tasks (GSM8K 14v13, sanitized MBPP 6v3) → two-stream is NOT ahead on
  generation anywhere; B wins only on denoising-NLL. Conversion WORKS; B>A NOT supported. Firm.
- **USER DECISION:** next phase = **Agentic/data-mix extension** (the project's goal + FLARE's unclaimed frontier).
  Carry two-stream forward (principled, better NLL, AR-preserving). Plan: (1) tool-call/agentic transfer data +
  same-family Qwen3.6-27B-teacher diffusion-friendly traces into the FLARE-style mix (Long-CoT+Math+IF + agentic);
  (2) two-stream train on it; (3) a BFCL-style tool-call eval in DIFFUSION mode (the thing FLARE never measured) +
  the retention battery. Immediate: scope what agentic/tool-call data + teacher we have, fix generation stop
  behavior (the leaked-role-text issue), stand up a minimal diffusion-mode tool-call eval baseline on the current
  best checkpoint.

## AGENTIC PHASE — scoping done: stop-fix + inventory + tool-call baseline (2026-06-29)
- **Stop fix DONE:** model leaked role text past EOS; fixed → clean termination (prereq for tool-call JSON).
- **Asset inventory:** `train_toolcall.json` = **9644** tool-call train ex (+8645 no-multicall variant); eval
  slices: public one-call 8, public multicall 12, teacher-heldout label-aware 8, +smoke; **Qwen3.6-27B teacher
  servable** (`serve_sglang_qwen36_teacher.sh`; strong: one-call exact-args 18/24, multicall 10/12, synth 48/48);
  curricula under `data/qwen35_9b_*_curriculum` (format/label-aware/argument/grounded-span/multicall/planner/
  model-repair/candidate-ranking/route-delta/retention).
- **TOOL-CALL BASELINE (B@1000, raw diffusion-mode, corrected sampler+stop-fix, GPU 96% util):** public one-call
  valid-JSON 1/8 exact-args **0/8**; multicall 2/12 **0/12**; teacher-heldout 2/8 **1/8**. → valid JSON ~5/28,
  **exact args ~1/24**. The original argument-grounding wall, cleanly measured on the WORKING conversion. EXPECTED
  (B@1000 trained on the 256-sample GENERAL mix, not tool-calls). The phase tests whether two-stream training ON
  tool-call data recovers grounding (as GSM8K recovered with scale).
- **NEXT (steered, CHECKPOINT-before-launch — data = FLARE's #1 lever):** flare to propose the first agentic
  two-stream mix (tool-call subset of 9644 balanced + RETENTION GSM8K/math/IF vs forgetting + optional 27B-teacher
  traces), start-point (B@1000-continue vs init-fresh), eval = same tool-call slices vs the 0/24 baseline + a GSM8K
  retention check. Review the mix before the ~5h run.

## Agentic mix design APPROVED w/ tightenings (2026-06-29)
flare proposed: continue from **B_two_stream_s1024_step1000** (isolates "tool-call data adds grounding" from
"conversion recovers"; two-stream's L_AR is itself a retention mechanism), two-stream only, ~1000 steps, mix =
**768 tool-call / 256 retention (75/25)** (multicall 140 + grounded-span 80 + synthetic-format 158 + ~390 more),
eval on the exact baseline slices vs 0/24 + GSM8K/MBPP/NLL retention. APPROVED with 2 tightenings + a watch:
(1) **leak check must catch NEAR-DUPLICATES** not just exact hashes — per eval slice, count train examples sharing
the same tool name + same argument VALUES; bar = ZERO same-tool+same-arg-values overlap (else "same call, reworded
prompt" leaks). (2) show the FULL 768 breakdown — must attack BOTH baseline failure modes (valid-JSON/structure
AND argument-grounding). WATCH: 75/25 forgetting — GSM8K retention check is the guardrail; if GSM8K <0.5 post,
bump retention next iteration; report GSM8K pre(0.70)/post. SUCCESS BAR (N=28 coarse): valid JSON ~5→15+/28, exact
args ~1→8+/24 (a 1→3 move is noise). flare to BUILD mix+manifest+leak-table+breakdown → review → THEN green-light
the ~5h run. No launch before review.
