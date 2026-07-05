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

## MEASURED ROLLOUT-THROUGHPUT MULTIPLIER (2026-07-04, runs/p2_batched_rollout_bench/report.md)
Step 2 of the loop rests on "diffusion twin generates RL signal at HIGH THROUGHPUT by batching rollouts." That claim is
now **measured**, and the a-priori FLOP-reducing/batch-robust thesis is **DISCONFIRMED** on this hardware/workload.

**Batch-correctness precondition PASSED first** (`runs/p2_engine_batchgates`): NO cross-request contamination on the
batched path — the per-request GDN snapshot/restore state stays isolated, so the batched sweep was safe to run.

**The curve (samples/sec == rollouts/sec/GPU; engine vs stock guided-AR at its FASTEST offline+cudagraph path — the
conservative baseline).** 48 never-train tool-call turns/point, temp=0.7 seeded (RL mode), certified engine (pin
`95d8b47`, PIECEWISE cudagraph + APC), RTX 5090, RAM cage:

| batch | eng samp/s | AR samp/s | **eng/AR** | eng occ (eff) | eng util% / AR util% |
|---:|---:|---:|:---:|---:|:---:|
| 1 | 1.524 | 1.625 | **0.94×** | 1.0 (1.00) | 88 / 100 |
| 2 | 2.248 | 2.499 | **0.90×** | 1.5 (0.77) | 88 / 100 |
| 4 | 3.426 | 4.103 | **0.83×** | 2.7 (0.68) | 87 / 100 |
| 8 | 4.948 | 6.601 | **0.75×** | 4.8 (0.60) | 87 / 100 |
| 16 | 5.732 | 7.846 | **0.73×** | 7.2 (0.45) | 84 / 100 |

The engine is at rollout **parity at batch=1 (0.94×) and LOSES ground as batch grows (0.73× at b16)** — the ratio moves
the WRONG way for the thesis. **Mechanism (measured):** the hybrid genuinely does ~10× fewer forwards/turn (4.9 vs ~50
at b16), but (1) each forward is ~14× costlier (per_forward 18.7→35.3 ms vs AR per-step 12.2→2.53 ms — AR's cudagraph
amortizes weight load across the batch, the hybrid's widens), and (2) the FLARE **sync scheduler** + per-request
**variable draft widths** (3–18) give poor batch occupancy — **effective 7.2/16 = 0.45 at b16, never co-batching all
16** (structural). AR co-batches near-linearly at 100% util; the engine idles host-bound at 84–88%.

**Structural batch limitation:** head-of-line/straggler blocking under FLARE's *forced* sync scheduler (requests finish
at wildly different forward counts); continuous batching recovers *some* occupancy but is **bounded by the forced sync**.
And the per-request GDN state makes **b16 OOM at gmu 0.74** — the engine's rollout concurrency is capped tighter than
AR's on a 32 GB card (AR is flat ~22 GB).

**CONSEQUENCE for the loop (revise step 2's economics):** the diffusion twin is **NOT a rollout-throughput multiplier
vs fast guided-AR — it is ~0.7–0.9×.** A throughput-bound RL loop generates rollouts **faster with stock guided-AR
(1.1–1.4× at batch).** The twin's earned value in step 2 is **quality/parity at SAFE batch**, not samples/sec: safe
batching (proven), latency parity at low batch, 48/48 valid tool-call stops (guided-AR truncated 2/48), and the
certified quality edge (130 vs 124 exact_args). This is a floor, not a ceiling — 84–88% util is host-bound headroom;
OPT-4 part-1 fused_recurrent (task #37) is the lever to lift the curve. Engine build detail: `engine_build_status.md`
§0.I.

## ENDGAME SCOREBOARD (2026-07-04, runs/endgame_scoreboard/report.md, aggregate 247 audited turns)
stock-bf16-AR-guided 124/247 @ 0.741 s/turn (THE TRUE BAR) | stock-FP8 129/247 @ 0.910 (FP8 slower on 5090!) |
merged-AR-guided 127/247 (conversion tax = NEGATIVE, +3) | OUR hybrid-clean(v2) 130/247 @ 2.577.
QUALITY: diffusion-hybrid BEATS stock AR Qwen aggregate (+6); per-slice split honest: matched-20 47 vs stock's 51,
never-train ahead. CONVERSION TAX: none (slightly negative) on this battery. v6 NOT promoted (retention 0.70 held,
quality flat 47 + never-train drop) -> v2 remains best; RL plateau at v2 confirmed across v4/v5/v6 variants.
SPEED: 3.5x behind stock vLLM wall-clock — entirely the engine gap; P2 build in flight (parity harness committed).
