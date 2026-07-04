# Diffusion-Accelerated On-Policy RL — Methodology Reframe (user, 2026-07-03)

## The deliverable is a PROCESS, not a model

Final goal: **RL any AR model, using its diffusionized twin as the test-time-compute / rollout engine.** The loop:

1. **M_t (AR model)** → diffusionize cheaply (two-stream conversion, ~1000-step QLoRA — our validated recipe).
2. **Diffusion twin generates RL signal at high throughput**: hybrid-clean decode for valid rollouts; parallel best-of-N
   sample-and-decode for exploration/test-time compute (validated: pass@16 lifts hard); 4 tok/fwd native on
   reasoning-class content (CoT rollouts).
3. **RL-update the ORIGINAL AR model** from those rollouts. Train↔inference inconsistency (diffusion behavior policy ≠
   AR target policy) always exists — it is ABSORBED IN THE RL ALGORITHM (importance correction, trust region), not
   pretended away.
4. M_{t+1} → re-diffusionize → repeat. Each cycle converts a model that JUST GAINED something.

## Why hybrid-clean decode is the methodological core (not just a serving win)

Hybrid keeps VALUE tokens strictly sequential (chain rule) → diffusion rollouts are **nearly on-policy for the AR
model exactly where the learning signal lives**. The tokens where the policies diverge most (grammar-forced structure)
are already masked out of the policy loss. So the required off-policy correction is small and localized by
construction: clipped per-token importance weights `exp(logp_AR − logp_hybrid)` on value/free tokens only, computed
via the exact-re-score spine (AR log-probs on any sequence — built and validated).

## The evaluation reframe: capability preservation, not benchmark parity

Benchmark parity ("diffusion preserves some distribution") is the wrong certificate for the loop. The loop's viability
rests on: **conversion must preserve the NEWEST, most fragile capabilities** (the ones the previous cycle just
RL-acquired), else the flywheel erases its own gains. Therefore:

- **Per-capability conversion-tax table** (not one aggregate number): stock-AR vs converted-AR vs diffusion-hybrid on
  tool-call exactness, GSM8K, MBPP, instruction-following — the deltas per capability class. (First run queued:
  stock-vs-merged AR baseline.)
- **Preservation audit protocol (the sharp test)**: take a model with a FRESHLY ACQUIRED capability (our v2/v6 RL
  gains qualify), run the conversion ON it, verify the fresh capability survives (a) in diffusion mode, (b) in
  post-conversion AR mode. Convert-after-RL is the missing experiment; conversion-before-RL (our historical order)
  does not certify the loop.

## Assets already built that transfer as PROCESS components

Conversion recipe (two-stream, cheap, retention-characterized) · hybrid-clean decode (47/63 + 83/184, audited,
promoted) · exact-re-score spine (importance-weight primitive) · RL safety kit (KL-to-base + rolling KL early-stop +
retention probes + graded rewards + mapped tradeoff surface) · audited eval battery + contamination instrumentation
(generated-token audit, force counters, forwards/turn) · flywheel env (verifiable rewards, episode drivers) ·
best-of-N sample-and-decode machinery · P2 engine plan (rollout-throughput multiplier for step 2).

## Gaps = the new work items

1. Per-capability conversion-tax audit (extend the queued stock-vs-merged run to the full battery).
2. Convert-after-RL preservation experiment (convert on top of v2/v6; measure the RL-gained capability's survival).
3. Off-policy correction spec + a small validation run (AR updates from hybrid rollouts; stability + gain).
4. This methodology doc matured into the shippable artifact (loop spec, budgets, gates, audit protocol).
5. (Decision pending) P2 engine build — in this frame it is the ROLLOUT-THROUGHPUT multiplier, not just serving.

## Expert correction (user, from industry experience, 2026-07-03)
Published spec-decode speedups (EAGLE-3 2-2.5x, DFlash 3-6x) are benchmark-condition optimistic. On REAL agentic
workloads (SWE-Verified via a Claude-Code-style agent): expect <2x — acceptance dies on exact spans (paths/IDs/code),
tool-result interleaving keeps draft context cold, turns are prefill-heavy, and production batching kills the
idle-compute assumption. Converges with our measured 1.39 accept/round on agentic content. CONSEQUENCE: hybrid's
audited 1.71x on agentic content (no extra model, no verify overhead, batch-robust, unoptimized stack) is competitive
with real-world spec decode on this workload class — and the two compose. Use <2x as the honest spec-decode prior for
agentic comparisons in all future tables.

## ENDGAME SCOREBOARD (2026-07-04, runs/endgame_scoreboard/report.md, aggregate 247 audited turns)
stock-bf16-AR-guided 124/247 @ 0.741 s/turn (THE TRUE BAR) | stock-FP8 129/247 @ 0.910 (FP8 slower on 5090!) |
merged-AR-guided 127/247 (conversion tax = NEGATIVE, +3) | OUR hybrid-clean(v2) 130/247 @ 2.577.
QUALITY: diffusion-hybrid BEATS stock AR Qwen aggregate (+6); per-slice split honest: matched-20 47 vs stock's 51,
never-train ahead. CONVERSION TAX: none (slightly negative) on this battery. v6 NOT promoted (retention 0.70 held,
quality flat 47 + never-train drop) -> v2 remains best; RL plateau at v2 confirmed across v4/v5/v6 variants.
SPEED: 3.5x behind stock vLLM wall-clock — entirely the engine gap; P2 build in flight (parity harness committed).
