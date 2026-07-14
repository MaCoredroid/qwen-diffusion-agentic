# PRODUCT DEFINITION — Diffusion-Qwen SWE Agent Serving (co-designed 2026-07-14)

**Co-designed by user + monitor. This document is the acceptance authority for "done".**

## The product (v1)

A single-GPU (RTX 5090-class) SWE-agent serving stack for Qwen3.5-9B whose **default and primary decode is the
pure block-diffusion twin** (FLARE hybrid_clean + W-2b draft-verify), OpenAI/qwen-code compatible, with the
lossless APC and the certified tool-call bridge.

### v1 ship requirements (ALL required)

1. **QUALITY — THE GOLDEN BAR**: the pure-diffusion twin matches **stock-AR 19/50 on the frozen w2_n50
   SWE-bench_Verified pool** (per SECTION C stats protocol: frozen envelope, official docker scoring, McNemar
   vs the banked per-instance verdicts, not-statistically-below). Entry milestone: C46 ≥12/46 first.
2. **SPEED**: draft-verify ON by default; shipped claim is the honest content-dependent envelope
   (measured: 3.3–3.9× forwards on write/copy-heavy, ~1.35×/1.12× wall on read-heavy episodes), bit-identical
   to K=1 sampling by construction (arej=0 release gate).
3. **SERVING CERTS as release CI**: KILL-T1 matched-20 exact-args (b=c=0 class), A6 online==offline, A7
   multi-turn APC, preservation cert (#29 protocol), FA battery 0, loop-halt canary, ctx-overflow truth-telling.
4. **dtype**: bf16.
5. **Opt-in flag (non-default)**: per-request AR decode policy + the AR-read router — shipped as a documented
   escape hatch, NOT the product's identity.

### The critical path to v1 (as of this writing)

X.2 pilot (in flight: KL-guarded AR-self-distillation of the read conditional) → **SECTION Y** (conversion as
long-context AR-self-distillation @12288 single-stream — the load-bearing training program) → C46 entry gate →
golden-number N=50 gate. Each stage pre-registered; kills stop the stage, not the mission ("hard doesn't mean
give up" — user directive, DIRECTIVE-4).

## v2 roadmap (post-v1)

- **NVFP4 serving** (original #54 directive) with its own cert pass (banked caveat: FP8 measured SLOWER on 5090).
- K-ladder / larger-block speed beyond draft-verify (SECTION W rungs, V-track revisit under the Y-trained twin).
- On-policy speed-RL (SECTION B) as the finishing lever once the twin resolves.

## Budget + provenance

Teacher-spend ceiling $500 (user, 2026-07-14; $151.02 spent). All training data leakage-firewalled vs the
113-id eval holdout (KILL-D1 hash-assert); w2_n50 and C46 are eval-only forever.
