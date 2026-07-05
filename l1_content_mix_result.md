# L1 — content-mix + K-curve synthesis: the 5×-at-B=1 equation with today's numbers

_Assembled 2026-07-05. Campaign: `goal_5x_rollout_b1.md`. Regime: B=1, greedy (temp 0, seed 20260701)._
_Rolls up three L1 sub-measurements: CENSUS (content mix), KCURVE (K vs held quality), BASELINE
(engine-vs-AR ratio, `l1_baseline_b1_result.md`). Artifacts under `runs/l1_census`, `runs/l1_kcurve`,
`runs/l1_baseline_b1`, `runs/run1_legacy_block_quality_curve` (all gitignored; paths at bottom)._

## The equation (goal doc), filled in with today's measured numbers

    speed ratio(B=1) ≈ (avg committed tok/fwd) × (AR ms/step ÷ engine ms/forward)

| factor | TODAY (measured) | at 5× target |
|---|---:|---:|
| avg committed tok/fwd (reasoning tokens only, K=1) | **1.00** | **5.0** (needs L3) |
| avg committed tok/fwd (all-committed, credits free grammar) | 1.36 | — |
| AR ms/step ÷ engine ms/forward (14.1 ÷ 18.7) | **0.754** | **1.0** (needs L2) |
| **product = B=1 speed ratio (reasoning-tokens-only)** | **0.75×** | **5.0×** |
| product on all-committed-token basis | 1.03× | — |

**Today: 1.00 × 0.754 = 0.75× (engine SLOWER on useful reasoning tokens).** The entire 5× lives in two
multiplicative factors that are both ≈parity-or-worse today: L3 must raise reasoning-span tok/fwd 1→5,
and L2 must close the 1.33× per-forward penalty (0.754→1.0). Neither is done.

---

## (1) Content-mix fractions (CENSUS)

Measured token decomposition on the reasoning proxy (GSM8K first-30, 5-shot CoT), RLv2 hybrid_clean
serving path, 24 well-behaved turns (996 generated tokens):

| span class | tokens | fraction | forwards/token |
|---|---:|---:|---:|
| grammar scaffold (FSM force-committed) | 264 | **26.5 %** | **0** (free) |
| model-chosen VALUE span (arg/arith body) | 540 | **54.2 %** | 1 (K=1) |
| model-chosen STRUCTURAL | 192 | **19.3 %** | 1 (K=1) |
| → all model-chosen (value+structural) | 732 | **73.5 %** | 1 (K=1) |

`denoise_forwards == model_chosen_tokens` exactly (732 == 732): **every model-chosen token is one
forward.** The only tok/fwd > 1 is the 26.5 % zero-forward grammar scaffold. Tool-call content (184-turn
never-train contrast) shifts the mix to ~37 % grammar-forced → agg tok/fwd 1.60 (more free scaffold, same
K=1 on all model-chosen tokens). **Content caveat:** the RLv2 model tool-wraps its reasoning inside a
hallucinated `<function=think>` call, so its "reasoning" lands in the VALUE span (chain-rule bound, K=1) —
the parallelizable free-CoT fraction is effectively ~0 for this served model today.

## (2) K_max(today) from the K-curve + achievable-today avg tok/fwd

**K_max(today) at held GSM8K exactness = 1.0 tok/fwd on reasoning spans — there is NO validated parallel
speed lane on reasoning content today.** Two independent curves converge:

- **KCURVE today** (`l1_kcurve`, legacy confidence-threshold sampler @ thr 0.9, converted BASE model that
  free-CoTs, 8 fresh GSM8K prompts): agg **tok/fwd = 1.011**, 5/8 correct, 2 runaways. Re-confirms the
  anchor sampler commits ~1 token/forward.
- **Canonical Run-1 legacy block-quality curve** (`run1_legacy_block_quality_curve`, bdc8001, N=20):

  | small_block_size K | strict GSM8K | generated tok/fwd |
  |---:|---:|---:|
  | 32 | 14/20 (0.70) | 1.019 |
  | 16 | 14/20 (0.70) | 1.020 |
  | 8 | **15/20 (0.75)** | 1.020 |
  | 4 | 11/20 (0.55) | 1.024 |

  The validated legacy sampler is **~1.02 tok/fwd at EVERY block width** — this is a QUALITY curve, not a
  speed lane (report's own words). Quality holds at K=8/16/32, breaks at K=4.

**★ CORRECTION to the goal doc:** the "MEASURED native 4 tok/fwd on GSM8K-class at held exactness" figure
is **refuted**. It conflated `small_block_size=8` with `tok/fwd`. The only sampler that mechanically
reaches nominal 4–8 tok/fwd is the mutable-remask fixed-K **diagnostic** (`measure_block_quality_curve.py`),
which is **disqualified**: at full denoise (nominal 1 tok/fwd) it already scores only **5/20 = 0.25**,
failing the ≥0.60 anchor — so its higher-tpf points are meaningless. Achievable-today avg tok/fwd:
**1.36** on served mixed content (RLv2 hybrid_clean, driven entirely by the 26.5 % free scaffold), **1.00**
on reasoning tokens themselves.

## (3) Measured B=1 ratio today (BASELINE, 24 shared prompts, engine/AR, >1 = engine faster)

| framing | ratio |
|---|---:|
| **reasoning-token-only throughput** (honest useful-content rate) | **0.75×** |
| all-committed-token throughput (credits free grammar scaffold) | 1.03× |
| per-**forward** speed (engine 18.7 ms vs AR 14.1 ms/tok) | 0.754× (engine **1.33× slower/forward**) |

Engine 53.4 reasoning tok/s vs stock AR 70.8 tok/s. AR ran enforce_eager (scoreboard config, no
cudagraph) → **AR handicapped, so the true engine/AR ratio is ≤ reported.** Exactness NOT held: RLv2
tool-wraps single-turn GSM8K → 0/24 scorable vs AR free-CoT 29/30 (harness artifact — see
`l1_baseline_b1_result.md`).

## (4) THE L3 TARGET — stated precisely

Amdahl (grammar-forced = 0 forwards, folded out): **avg tok/fwd = 1 / (f_reason/K + f_value/1)**, solve
for the reasoning-span tok/fwd K that yields avg = 5:

| rollout f_value (exact spans) | K needed for avg = 5 | K→∞ ceiling (=1/f_value) |
|---:|---:|---:|
| 0.02 | **K ≈ 5.4** | 50 |
| 0.05 | **K ≈ 6.3** | 20 |
| 0.10 | K ≈ 9.0 | 10 |
| 0.15 (tool-call) | K ≈ 17 | 6.7 |
| 0.20 | ∞ (unreachable) | 5.0 |

**L3 target: on rollout reasoning content (f_value ≈ 2–5 %), commit K ≈ 5.5–6.5 tokens/forward on
reasoning spans at held exactness.** Distance from K_max(today)=1.0: a **~6× increase in reasoning-span
tok/fwd at held quality** (and, if L2's 1.33× per-forward penalty is left unclosed, K must reach ~9 to
still net 5×). **Therefore S2 consistency distillation + entropy-gated adaptive K must deliver BOTH:**
(a) reclassify reasoning tokens as parallel-committable spans (today they are chain-rule-bound value
tokens at K=1 — measured top-1 conditional 0.238), and (b) push held-exactness commits from 1 → ~6
tokens/forward on those spans. For reference, the only current sampler that mechanically reaches K≈4–8
loses 15–45 exactness points (0.75→0.55→0.25) — that entire gap is what S2 must erase. This is the sole
lever to >1× on reasoning content; L2 alone cannot move tok/fwd.

## (5) L2 confirmation — per-forward gap to AR (from baseline data)

- Engine **18.7 ms/forward** (median 18.84; matches the 18.5 ms bs=1 endgame floor) vs stock AR
  **14.1 ms/token** → **1.33× slower per forward** (0.754 as a ratio factor).
- This is the second multiplicative factor. **L2's deterministic job: 18.7 → ~13 ms** (#37 fused_recurrent
  GDN for 1-token probes + kill residual host overhead) to bring the factor 0.754 → 1.0.
- **L2 cannot change tok/fwd** (reasoning is K=1); it only converts the 0.75× reasoning ratio toward ~1.0×.
  Note AR was enforce_eager-handicapped here → the real per-forward gap to a cudagraph AR is **larger**
  than 1.33×, so L2 is necessary but not sufficient.

## Verdict for the 5×-at-B=1 campaign

The measured B=1 equation today is **1.00 × 0.754 = 0.75×** — the engine has no reasoning-content speed
advantage, exactly in the predicted 0.7–1.0× band. The goal-doc "expected 2–3× already" and "native 4
tok/fwd" are both **refuted** on the audited serving path. Both multiplicative factors sit at/below parity,
so 5× is entirely a forward-looking training bet: L2 (deterministic engineering) closes the 1.33× per-forward
penalty; L3 (S2 + adaptive-K, unexecuted) must raise reasoning-span tok/fwd from 1 to ~6 at held exactness —
a ~6× gain against a chain-rule wall where the only sampler reaching those K values today loses 15–45
exactness points. Feasibility is L3-gated and unproven; the campaign's honest status is "0.75× today, 5×
contingent on S2 delivering a parallel reasoning lane that does not exist yet."

## Artifacts (absolute paths)

- BASELINE (ratio, full analysis): `/home/mark/qwen_diffusion/l1_baseline_b1_result.md`,
  `/home/mark/qwen_diffusion/runs/l1_baseline_b1/summary.json`
- CENSUS (content mix): `/home/mark/qwen_diffusion/runs/l1_census/gsm8k_turns.jsonl`,
  `/home/mark/qwen_diffusion/runs/l1_census/gsm8k_prompts_clean.json`
- KCURVE (today, thr0.9): `/home/mark/qwen_diffusion/runs/l1_kcurve/kcurve.jsonl`
- Canonical legacy K-curve (bdc8001): `/home/mark/qwen_diffusion/runs/run1_legacy_block_quality_curve/report.md`
- Disqualified fixed-K diagnostic (0.25 anchor fail): `/home/mark/qwen_diffusion/qwen35_block_quality_curve_gate_result.md`
