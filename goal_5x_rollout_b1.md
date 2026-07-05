# GOAL: 5× rollout speed vs guided-AR at B=1 (set by user, 2026-07-05)

## The reframe
Compare at **B=1** (latency-bound, the regime the twin already wins: 0.626 vs 0.741 agg), NOT B=16
(saturated GPU — same FLOPs/token, ratio physics-capped, and our tool-call bench there was the twin's
worst-case content). At B=1 both AR and the engine are weight-stream-bound, so:

    speed ratio ≈ (avg committed tokens per forward) × (AR per-step ms / engine per-forward ms)

Second factor → ~1.0 with engineering parity. **The goal is therefore one number:**

    NORTH STAR: avg ≥5 committed tokens/forward at HELD exactness on rollout content, per-forward at
    AR per-step parity → ≥5× rollout wall-clock at B=1.

## The Amdahl accounting (why this is feasible)
avg tok/fwd = 1 / (f_reason/K + f_value/1); grammar-forced tokens = 0 forwards (already live).
- Tool-call eval content (f_value≈15%): K→∞ caps avg at 6.7; K=8 gives 3.9. Hard but not the target mix.
- Rollout content (SWE/CoT/GSM8K-class, f_value≈2-5% exact spans): avg 5 needs K≈6-7 on reasoning spans.
- MEASURED TODAY: native 4 tok/fwd on GSM8K-class at held exactness (post drift-fix). Values stay K=1
  (chain rule, 0.238 top-1 measured — never revisit without new evidence).

## The three multiplicative levers (the campaign)
1. **L1 — content mix, measure first (M1, cheap, decisive):** B=1 bench on reasoning-heavy flywheel
   episodes (SWE/GSM8K-class), engine vs guided-AR: today's native avg tok/fwd + ratio. Expected 2-3×
   already. This baselines the campaign; the prior 0.73-0.94× number was tool-call-heavy B=16 — wrong
   content, wrong regime for this goal.
2. **L2 — per-forward parity (engineering, deterministic):** 18.5ms → ~12-13ms: #37 fused_recurrent GDN
   for 1-token probes + remaining host residue (6.54ms → 2-3ms). B=1 focus; occupancy work DEPRIORITIZED.
3. **L3 — raise K with training (the compounding lever):** un-park S2 consistency distillation + add
   entropy-gated adaptive K (commit easy spans at K=8-16, K=1 where uncertain; values always K=1).
   Retrain freely from the two-stream foundation; erosion cap + KL 0.05 + retention probes per the RL
   safety kit; promotion only on audited exactness-held gates.
- **L4 (absolute-latency bonus, ratio-neutral):** NVFP4 probe — cuts the 11.4ms weight-stream floor for
  BOTH sides (~2-3× absolute); measure on THIS card (FP8 was slower — trust nothing unmeasured on 5090).

## Gates
- M1 report: avg tok/fwd + ratio at B=1 on rollout content, audited (proj=0, exactness scored).
- L2 gate: per-forward ≤13ms with byte-parity maintained (the 233/247 certificate must not regress).
- L3 gate: avg tok/fwd ≥5 on rollout content at exactness within noise of K=1 careful (paired stats,
  episode bootstrap) + GSM8K retention ≥ anchor + full audit battery. Sampler-pinned per REPRODUCE_V2.
- Endgame: engine B=1 rollout wall-clock ≥5× guided-AR on the same episodes, same weights, audited.
