# FLARE Paper Digest (arXiv 2606.01774)

"FLARE: Diffusion for Hybrid Language Model" — Adobe Research + Georgia Tech (Zhu, Shi, Ge, Tan, Xu,
Zhu, Kuen, Goswami, Jain, Chen, Tao, Gu). Submitted 2026-06-01. Project: tokflare.github.io.
**Converts Qwen3.5 hybrid (Gated DeltaNet + Gated Attention + causal ShortConv) AR checkpoints → diffusion.**
Code + weights "coming soon" (NOT released). License CC-BY-NC-SA. This is our exact architecture.

## Core method (token-equal two-stream objective)
`L_FLARE = L_AR + L_diff` (Eq 4-5), every token contributes ONE AR + ONE diffusion signal at unit weight:
- **L_AR** = standard next-token log-likelihood over the full sequence (the CLEAN, token-causal stream).
- **L_diff** = denoising log-likelihood over TWO COMPLEMENTARY masked "noisy" views per block (M_b and
  its complement), each bidirectional WITHIN the block, conditioned on the preceding CLEAN context.
- **Logit shift** is applied to the noisy-stream diffusion terms to align them with AR semantics (vs TiDAR
  which doesn't). Noisy stream kept block-bidirectional + random (vs I-DLM fully-masked) to preserve
  Diffusion-Trust decoding.
- Document-packed clean/noisy mask (Fig 3): clean stream causal + isolated; noisy block bidirectional +
  attends preceding clean; doc boundaries isolate packed samples.

## GDN handling = a STATE SCHEDULE, not a dense mask (Eq 6, §5.2, App A) — answers our key question
- The clean stream follows the **standard causal GDN recurrence** (S_t = α_t(I − β_t k_t k_t^T)S_{t−1} + β_t k_t v_t^T).
- Each **noisy block b is initialized from the preceding CLEAN block-boundary state S_{(b-1)B}** and updated
  ONLY with that block's noisy tokens; the noisy readout uses the shared block-end state (bidirectional
  within block); **the noisy state is reset to the clean boundary state per block** (no cross-noisy-block leak).
- The non-causal mask becomes "a schedule of writes, state propagation, and state resets." → GDN stays
  **causal-within-block** (our `option_a_causal_gdn` is the right direction; the 1-in-4 attention layers
  carry within-block bidirectionality). Document packing requires **noisy seed reset to ZERO at doc start**
  and ShortConv cross-doc lags masked to zero (App A, Challenge 2).
- Two training-kernel routes: **Route I** (chunk-then-refine: materialize clean block-boundary states in
  HBM, seed noisy blocks; correctness-first, reuses standard kernels, but L/B memory blowup at small B) and
  **Route II** (fused two-stream: store only STRIDED clean-state checkpoints, rebuild boundary state in
  registers, inject block gradients back into the clean recurrence). Route II is a STRUCTURAL requirement at
  small B (FLARE uses B=4); Fig 7: GDR at B=1, latency 135.10→37.69 ms, peak mem 18.14→0.45 GiB. Fig 8:
  Route II raises training MFU at B=4 from 13.80%→24.81%, matching pure-AR Qwen3.5-2B (24.04%). C(chunk)=64.

## Decoding (§3.3, Alg 1-2) — one checkpoint, two modes
- **Diffusion-Trust** (Alg 1): block iterative denoising; commit positions with confidence ≥ γ_s (noisy
  stream trusted, no clean verification). After block finalizes: "token-causal replay on finalized block →
  advance the live state." Denoise passes READ the recurrent state but DON'T write it back (intermediate
  denoise tokens may be revised → writing early corrupts the state — the corruption trap).
- **AR-Trust** (Alg 2): noisy samples = drafts; clean-stream logits verify left-to-right (speculative),
  accept with prob min{1, p_i(d_i)/q_i(d_i)}. Recurrent-state commit must rewind the GDN state to S^(r)
  on partial accept (fused gather-scatter, avoids replaying accepted tokens through every GDN layer).
- Serving: SGLang stack; 4 mechanisms (recurrent-state commit, native dLLM mask modes, fused verify/top-k,
  CUDA-graph replay safety). Fig 9: FLARE-2B 2087 tok/s GSM8K (2.2× LLaDA-2.1-mini, 4.8× SDAR-1.7B),
  1441 GPQA (3.6×), A100-80GB C=8.

## The #1 finding: transfer DATA QUALITY dominates objective/mask
- Controlled study (Qwen3-1.7B, ~10B tokens, 9000 steps, batch 256, seq 4096): pure block-diffusion
  conversion **DEGRADES capability −21.8 pts** (worst on Code/Knowledge); restoring the token-causal AR
  clean stream **recovers +14.0**; clean next-token loss + logit shift → near-AR.
- "Once the core loss-mask design is aligned, **transfer-data quality and distribution match dominate the
  residual gap**" — and "AR fine-tuning provides a low-cost proxy for screening data mixes." Mixes built
  from Long-CoT + Math + IF pools (Llama-Nemotron / Nemotron post-training). Final: Mix 4 (Long-CoT+Math+IF).

## Results (Table 1, FLARE-9B vs Qwen3.5-9B AR source)
Retention 95-99% on math/knowledge: MATH-500 95.20 vs 96.60 (98.6%); AIME-24 63.33 vs 65.56 (96.6%);
GSM8K 93.33 (AR-Trust) vs 89.16 (beats AR); MMLU 84.80 vs 88.21; GPQA 71.21 vs 80.30; HumanEval 92.07 vs
95.12; MBPP 91.05 vs 89.11 (beats AR); LiveCodeBench 49.71 = AR; IFEval 71.35 vs 91.31 (weakest, ~78%).
FLARE-9B matches/exceeds LLaDA-2.1-flash (100B) at ~1/10 params. NO tool-calling/agentic/function-calling eval.

## Limitations (§6) — the openings for us
1. ~2× training compute/memory (concatenated 2L clean+noisy input); kernels mitigate, don't eliminate.
2. **Residual gap to AR source is a DATA-distribution mismatch** — FLARE used EXTERNAL teachers (DeepSeek-R1,
   Qwen3, GPT-OSS-120B); more filtering doesn't close it; "future conversions may benefit from transfer data
   more tightly matched to the source AR model" / **"diffusion-friendly traces harvested from the same-family
   AR teacher."** (We have the Qwen3.6-27B same-family teacher → direct improvement surface; ties to OPDLM.)
3. Dense ≤10B, single SFT stage only; MoE + RL untested.
