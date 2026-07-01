# Big-Block-at-Quality Redesign — Plan & Design Rationale

Date: 2026-07-01. Status: COMMITTED (user greenlit full stack). Companion to `training_redesign_10x_research.md`
(the verified literature synthesis) and the memory `qwen-diffusion-experiment.md`. This doc captures the DESIGN
REASONING we converged on, so the plan is self-explaining.

## 0. Goal + honest ceiling

Goal (user's framing): a block-diffusion model that emits a LARGE BLOCK per forward at MAINTAINED quality ("big
token"), faster than a WELL-OPTIMIZED AR. **~10x is NOT physically reachable at strict grounding quality on this arch.**
Why (first principles, not defeatism):
- The only speed lever vs same-arch AR is tokens committed per forward = block_size / denoise_steps.
- 10x needs ~10 correct tokens per forward. The factorization barrier forbids that for dependent tokens (below).
- Published block-diffusion at HELD quality vs strong AR tops out at **2.5x (D2F) – 5.18x (CD4LM)**; the 10-128x
  headlines are vs the model's own 1000-step baseline / on perplexity / at tiny scale. Our measured ceilings agree
  (~1.1-1.7x). Realistic target for us: **~2-3x blended tool-call, 3-4x reasoning** if the full stack lands.

## 1. The organizing principle: conditional entropy C(Y|X)

The factorization barrier is NOT uniform — it is organized by **C(Y|X)** = how determined the output block is by the
input (ParallelBench 2510.04767):
- **C(Y|X)≈0** — output DETERMINED by input: verbatim copy-from-context (arg values, IDs, paths), grammar-forced JSON
  scaffold. Parallel-ROBUST, and TRAINABLE to be parallel-EXACT.
- **C(Y|X)>0** — genuine internal dependence between co-emitted tokens: paired values (start/end time, INV-301/302),
  reasoning, cross-call ids. Can NEVER be made parallel-exact by any training (architectural, scale-independent,
  2602.00286) — UNLESS you model the joint (Sec 6).

## 2. Why C>0 tokens "have to be" sequential

Sequential decoding IS how you compute the exact joint, via the chain rule:
`P(y1..yk|x) = P(y1|x)·P(y2|x,y1)·P(y3|x,y1,y2)...` — each forward commits one token, the next conditions on it. Exact,
but N steps. A single diffusion forward only gives the INDEPENDENT marginals `∏ P(yi|x)` — the cross-token correlations
are thrown away. For C>0 tokens the product of marginals is the WRONG distribution → incoherent samples. So to make
C>0 faster you must put (some of) the joint back into one forward = joint modeling (Sec 6). That is the ONLY attack.

## 3. Per-span, not global, block sizing (the key structural decision)

Block-diffusion = sequential BETWEEN blocks (causal, handles cross-block deps correctly) + parallel WITHIN a block. The
barrier bites WITHIN the block. Consequence: **you cannot just make blocks large.**
- A bigger block puts MORE correlated (C>0) pairs into the same parallel step → MORE corruption. Large blocks make it
  WORSE, not better.
- Second sequentiality axis: within a block of B tokens over K denoise steps, tokens/forward = B/K. C>0 tokens force
  K≈B (each step conditions on prior commits = near-AR within block); pushing K≪B corrupts them.
- Free parallelism exists ONLY where within-block tokens are mutually independent given the prefix (C≈0).

**Therefore: choose block size PER SPAN, matched to local dependency** — large parallel blocks / few steps on C≈0
spans; small / near-AR / joint-modeled on C>0 spans. The 2-3x is the weighted average (cheap where C≈0, expensive
where C>0).

## 4. How we GET per-span sizing (mechanism) + overhead balance

Two places:
- **Training (labels available, reliable):** label C≈0 spans (copy-from-context, scaffold) and TRAIN them to commit
  jointly / parallel-EXACT (Run 1). We don't detect them — we make them genuinely parallel-safe.
- **Inference (no labels):** an adaptive commit rule decides how many tokens to commit per step. Three mechanisms; MIX
  them but BALANCE overhead — **net speedup = parallelism_gain − detection_overhead**, so the detector must RIDE ON
  compute already being done; NEVER add a forward just to decide.
  - **(c) grammar/structure — ~zero runtime cost:** precompute the schedule OFFLINE (scaffold vs value spans), one FSM
    transition/token. Best for our structured tool-call domain. (Run 1 uses the tool-sensitive planner: json_key→2
    steps, argument_value→8, prose→4.)
  - **(a) confidence/entropy — ~free:** elementwise max/entropy on logits the forward already produced. Fine adjustment
    within a span. First-order C-proxy.
  - **(b) learned boundary/commit-safety head — small:** one tiny matmul on existing hidden states (NOT a new forward).
    Optional upgrade, only if a+c leave parallelism on the table.
  - Order by overhead: c (static, free) + a (free) as the always-on base; b gated behind "only if needed."
- **The fine cut is carried by training + (a), for free:** grammar (c) is COARSE (all `argument_value` alike; can't tell
  a verbatim-copy C≈0 value from a derived C>0 value). Run 1 trains copy values to be SHARPLY, CORRECTLY confident
  under parallelism, so the free confidence signal (a) then separates C≈0 from C>0 inside a value span at zero cost.
- Honest failure of (a) alone: confidence is imperfect — a C>0 token can be confidently-but-independently WRONG
  (`end_time` marginal peaks at "17:00" regardless of `start_time`). That residual is exactly what the joint prior
  (Sec 6) exists for. This is why raw confidence-threshold only reached ~1.1-1.7x.

## 5. The joint prior for C>0 (Run 2 / CoDD)

Attack the C>0 residual by modeling (some of) the joint in one forward. CoDD (2603.00045): frozen backbone gives
per-position potentials φ_i(y_i); train a small TRACTABLE joint prior Q(y1..yk) (HMM/probabilistic-circuit, hidden
~1024); sample the PRODUCT `P(block) ∝ ∏φ_i(y_i) · Q(y1..yk)` so co-emitted tokens are CORRELATED. Lets k+1 condition
on k within one step = the exact factorization fix.
- **Why only ~2x (tractability wall):** tractable (samplable in one forward) ⟺ low-capacity. Language's true joint is
  intractable (that's why AR needs N steps). A small circuit captures SHORT-RANGE / low-order correlations (fixes
  AA-vs-BB, paired-value coherence) → drops N→~N/2 → ~2x. It cannot capture the full long-range reasoning joint.
- **Hopeful, specific to us:** the ~2x is measured on GENERAL reasoning (long-range). OUR C>0 is mostly SHORT-RANGE and
  structured (start↔end adjacent, INV-301↔INV-302 local, cross-call id = copy from a known source) = the BEST case for
  a tractable joint prior → could beat the general ~2x on tool-calls. Run 2 tests this directly.
- Alt if CoDD underdelivers: Di4C (2410.08709) mixture-of-denoisers (latent-mixture joint).

## 6. Committed redesign stack (build order forced by dependencies)

- **Run 1 (base, building on flare):** FLARE two-stream + #1 copy-from-context grounding (train C≈0 value spans to
  commit parallel-exact, LEFT-TO-RIGHT within span to keep the causal copy circuit intact) + conditional-entropy-
  adaptive schedule (c+a) + #5 BD3-LM clipped noise U[0.3,0.8] (free quality lever). QLoRA, ~300-600 steps, block 512,
  native format, 50/50 retention mix. GDN state read-only during denoise, advance once at committed boundary.
- **Run 2 (queued, the joint — user priority):** #2 CoDD frozen-backbone joint prior on Run-1's checkpoint. Attacks C>0.
- Directions #3 (D2F inter-block pipeline), #4 (Di4C), remain optional per research doc.

## 7. Pivotal gate + success metrics (Run 1)

Raw lane (no grammar decoder, no forcing), tau re-tuned on a GDN heldout (NOT LLaDA 0.9). Measure on the C≈0
copy-from-context value spans:
- **PASS:** raw value-span exact-match ≥ baseline (7/12 heldout, 10/12 public) while value_tpf 1.01 → ≥2.0 ⇒
  ParallelBench transfers to GDN, the bidirectional copy circuit tolerates parallel-copy → greenlight full stack.
- **PARTIAL:** holds only to value_tpf ~1.3-1.5 ⇒ some tolerance, cap expectations.
- **FAIL:** exactness drops at any value_tpf>1.1 / INERT / corruption ⇒ GDN copy-circuit disruption IS the wall → the
  schedule approach can't parallelize even C≈0 → pivot HARD to the joint prior (Run 2) as the primary lever.
- **Retention gate:** GSM8K ≥ ~0.60 (don't regress like mix-v1/v2).

## 8. The single most important open question

GDN linear-attention + diffusion is EMPTY literature. Whether GDN's bidirectional-within-block copy-circuit disruption
blocks even C≈0 parallel-copy is UNTESTED (our causal-value-span test was INERT). Run 1's gate resolves exactly this —
it is the cheapest experiment that discriminates "undertrained C≈0 (fixable)" from "GDN copy-circuit wall (architectural)".
