# GOAL: 5× rollout speed vs guided-AR at B=1 (set by user, 2026-07-05)

## The reframe
Compare at **B=1** (latency-bound, the regime the twin already wins: 0.626 vs 0.741 agg), NOT B=16
(saturated GPU — same FLOPs/token, ratio physics-capped, and our tool-call bench there was the twin's
worst-case content). At B=1 both AR and the engine are weight-stream-bound, so:

    speed ratio ≈ (avg committed tokens per forward) × (AR per-step ms / engine per-forward ms)

Second factor → ~1.0 with engineering parity. **The goal is therefore one number:**

    NORTH STAR: avg ≥5 committed tokens/forward at HELD exactness on rollout content, per-forward at
    AR per-step parity → ≥5× rollout wall-clock at B=1.

## ★ L1 MEASURED (2026-07-05, `l1_content_mix_result.md` + `l1_baseline_b1_result.md`)
Equation filled in today: **1.00 (reasoning tok/fwd, K=1) × 0.754 (14.1 AR ÷ 18.7 engine ms) = 0.75×**
(all-committed framing 1.03×). Both factors at/below parity → NO speed advantage on reasoning content.
- (1) Content mix (served, RLv2 hybrid_clean): grammar-forced 26.5 % (0-fwd) · value 54.2 % (K=1) ·
  structural 19.3 % (K=1); all 73.5 % model-chosen tokens are K=1 (732 fwd == 732 tokens).
- (2) **K_max(today) = 1.0 tok/fwd at held GSM8K exactness — NO parallel reasoning lane exists today.**
  ★ The "native 4 tok/fwd" claim below is REFUTED: legacy anchor sampler is ~1.02 tok/fwd at every block
  width (bdc8001 + l1_kcurve thr0.9 = 1.011); the fixed-K sampler that reaches nominal 4–8 tok/fwd is the
  disqualified mutable-remask diagnostic (0.25 at full denoise, fails anchor). Achievable-today avg tok/fwd
  = 1.36 served (all from the free scaffold) / 1.00 on reasoning tokens.
- (3) Ratio TODAY: reasoning-only 0.75× · all-committed 1.03× · per-forward 1.33× slower (AR
  enforce_eager-handicapped → true ratio ≤ reported).
- (4) L3 TARGET: for rollout f_value 2–5 %, avg=5 needs reasoning-span **K ≈ 5.4–6.3** (K→∞ ceiling
  1/f_value = 20–50). Distance from K_max(today)=1.0 → **~6× reasoning-span tok/fwd at held exactness**
  (K≈9 if L2 penalty left unclosed). S2 must (a) turn chain-rule value tokens into parallel reasoning
  spans and (b) erase the 15–45-pt exactness loss the only K≈4–8 sampler shows today.
- (5) L2 CONFIRMED: engine 18.7 ms/forward vs AR 14.1 ms/token = 1.33× per-forward penalty; L2 job
  18.7→~13 ms converts the 0.754 factor →1.0 (cannot change tok/fwd).

## ★★ L0 DONE + L2 MEASURED — FINAL-HEAD VERIFIED (2026-07-05, pin `0b44dcc`, `runs/l0l2_final_head_verify/`)
Re-ran the full no-regression set on the FINAL HEAD (L0 free-text fix `0b44dcc`); **zero source edits**,
byte-parity certificate intact. Two corrections change the honest equation:
- **AR-cudagraph fairness (L2):** the fair AR baseline is **10.72 ms/tok (cudagraph, 93.3 tok/s)**, not the
  14.1 ms eager used in L1. AR gets a 1.32× cudagraph speedup the engine already banks; using eager was a
  handicap. Honest AR per-step = **10.72 ms**.
- **Reasoning per-forward is 25.8 ms, not 18.7 (L0 revealed it).** L1's 18.7 ms was the engine emitting
  *unstoppable short tool-wraps* (0/24 scorable). The L0 fix makes free-text actually work (**30/30 clean
  stop, 26/30 GSM8K, 0 hangs**) — and the real free-CoT path costs **25.8 ms/forward** (cudagraph-engaged,
  K=1, only **0.86 emitted tok/fwd** because block-diffusion overshoots past EOS). So the honest
  reasoning-content per-forward is **25.8 ms**, a further ~1.4× above the 18.3 ms tool-call floor (measured
  clean this run; matches endgame 18.5). L2 job now: 25.8 → ~13 ms on the reasoning path.

**CORRECTED 5× EQUATION (reasoning content, final HEAD):**

    ratio = (committed tok/fwd) × (AR ms/tok ÷ engine ms/fwd)
          = 0.86 × (10.72 AR-cudagraph ÷ 25.8 engine-free-text) = **0.36× vs AR-cudagraph**   (0.47× vs AR-eager)
    model-chosen K = 1.00  (chain-rule wall)

The two honesty corrections move the L1-published **0.75×** → **0.36×** (fairness −0.17, working-free-CoT
per-forward −0.22). **Distance to the 5× north star ≈ 14×**, entirely in the K factor — L2 per-forward
parity (25.8→~13) buys at most ~2× and is still K-bound; only **L3 (S2 consistency-distillation +
entropy-gated adaptive K)** raises reasoning K above 1. No-regression evidence: 15-turn parity **14/15**
(lone break gt44 fp-residue), read-only fingerprint **6/6 bit-identical**, determinism **2× identical**,
value_projection **0**.

## The Amdahl accounting (why this is feasible)
avg tok/fwd = 1 / (f_reason/K + f_value/1); grammar-forced tokens = 0 forwards (already live).
- Tool-call eval content (f_value≈15%): K→∞ caps avg at 6.7; K=8 gives 3.9. Hard but not the target mix.
- Rollout content (SWE/CoT/GSM8K-class, f_value≈2-5% exact spans): avg 5 needs K≈6-7 on reasoning spans.
- ~~MEASURED TODAY: native 4 tok/fwd on GSM8K-class at held exactness (post drift-fix).~~ **REFUTED by L1
  (2026-07-05): the anchor sampler is ~1.0 tok/fwd at every block width; the 4-tok/fwd came from the
  disqualified fixed-K diagnostic. K_max(today) = 1.0 at held exactness.** Values stay K=1 (chain rule,
  0.238 top-1 measured — never revisit without new evidence).

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
