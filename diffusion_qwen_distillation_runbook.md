# Diffusion-Qwen Distillation Runbook

Concrete train → verify → serve recipe for adapting **Qwen3.6-27B** (autoregressive) into a
**block-diffusion** model. The procedure is "fork an existing template and add the part it doesn't
cover."

- **Template:** Fast-dLLM v2 (NVlabs/Fast-dLLM, `v2/`) — adapts a pretrained AR model into a
  block-diffusion decoder with ~1B tokens of fine-tuning, already shown on Qwen2.5-Instruct with
  released 1.5B / 7B checkpoints. [R1, R2]
- **The gap you fill:** Qwen2.5 is pure attention; Qwen3.6-27B is GDN-heavy (3 of every 4 sublayers
  are Gated DeltaNet linear-attention). The template has no GDN handling — that is the novel work.
- **Target use:** agentic coding / SWE-bench Verified, via a patched coding-agent harness.

Architecture recap: 64 layers = 16 × (3×[Gated DeltaNet → FFN] + 1×[Gated Attention → FFN]); GDN
= gated linear-recurrence, causal, constant-size state, no KV cache; ships MTP (multi-step) heads. [R12]

---

## Phase 1 — TRAIN (adaptation / distillation)

1. **Fork + surgery.** Start from `Fast-dLLM/v2`. Load Qwen3.6-27B weights. Add an absorbing
   `[MASK]` token (reuse a reserved vocab id; init its embedding from the mean token embedding).
   Install the block-causal attention mask on the Gated Attention layers: **bidirectional within a
   block, causal across blocks** (clean previous blocks visible, future blocks masked). Switch the
   head to **shifted prediction** (Dream) so the AR next-token head maps onto fill-this-position
   with no new head weights. [R1, R3, R4]

2. **GDN handling — the only non-template step.**
   - **Option A (v0, cheapest):** leave the forward GDN chunk-scan kernel unchanged; use GDN as the
     *cross-block causal state carrier*. Snapshot the constant-size state `S` at each block boundary;
     re-scan only the current block each denoising pass. Within-block bidirectionality then comes
     entirely from the 1-in-4 Gated Attention layers.
   - **Option B (upgrade):** bidirectional GDN — forward scan (carries state across blocks +
     through block) plus a **backward scan within the block only** (resets at each block boundary so
     it can't leak future blocks); fuse the two outputs. New reversed-scan kernel + fusion, ~2×
     within-block GDN cost. Switch to B only if verification shows within-block under-mixing.

3. **Objective.** Block masked-diffusion: within the current block, mask a subset per a noise
   schedule, predict the originals, cross-entropy on masked positions only; previous blocks are
   clean context. The complementary mask preserves the AR objective so pretrained knowledge is not
   destroyed. [R1]

4. **Curriculum + scale.**
   - **Block-size warmup:** start at block size ≈1 (model is in-distribution, barely moves), grow
     1 → 4 → 16 → … → 256. This is what makes the transition from sequential to parallel cheap. [R5, R6]
   - **Low LR** (~1e-5, cosine) — Dream found LR critical to preserve AR-inherited knowledge. [R3]
   - **~1B tokens**, weighted toward code / repo edits / infill (target is SWE-bench), plus some
     general text to avoid forgetting. [R1]
   - 27B at ~1B tokens ≈ hundreds of GPU-hours, not a pretraining run. bf16 + fp32 master is the
     safe default; fp8 optional if your stack supports it.

## Phase 2 — VERIFY (cheap → expensive)

5. **Numerics.** Denoising loss / perplexity on held-out vs the AR teacher (expect a gap — it's a
   speed model). Sweep (denoising steps × block size × confidence threshold τ) to map the
   speed/quality Pareto.

6. **Diffusion-strength checks.** Infilling, code-in-the-middle, structured-format correctness.
   These should be *strengths*; if not, the conversion is broken.

7. **Capability test — read the caution first.** Wire into the patched codex; run a SWE-bench
   Verified slice (DiffusionGemma stand-in first, then Diffusion-Qwen), comparing vs AR Qwen3.6.
   Before this, read the agentic reality-check paper [R7] — agentic multi-step is where diffusion
   LMs have shown weakness, and it tells you what to watch. The GDN Option-A-vs-B ablation lives
   here: if large blocks smear, you need the bidirectional scan.

8. **Coverage check (RL-bridge prerequisite).** Measure the gap between Diffusion-Qwen samples and
   AR gold logprobs. Even if you only ship the product, this quantifies drift and gates the
   downstream calibrated RL bridge.

## Phase 3 — SERVE / INFERENCE

9. **Decode loop (block by block).** For block *b*: fix prefix `x_<b`, start from an all-`[MASK]`
   block, then iteratively refine — for each masked position compute confidence
   `c_i = max_v p(x_i = v | x_<b, x_b)`, unmask tokens with `c_i ≥ τ`, and always unmask at least the
   single highest-confidence token to guarantee progress. Repeat K passes (K = speed/quality dial),
   commit the block, advance. [R7]

10. **Caching (throughput win).** Fast-dLLM v2 ships hierarchical caching — a block-level cache for
    cross-block context plus a sub-block dual-cache for within-block parallel decoding. **Your
    addition:** snapshot the constant-size GDN state alongside the attention KV at each block
    boundary. That snapshot is what keeps long context cheap and is absent from the template. [R1]

11. **Engine + quant.** Extend vLLM's diffusion path (DiffusionGemma's day-0 vLLM support is a
    scaffold to build on) or SGLang, with a block scheduler + GDN state cache. Quantize to 4-bit to
    fit and to chase throughput on a 5090. Knobs exposed to the agent: denoising steps, τ, block
    size, per-position temperature. [R8]

12. **Harness glue.** Re-do tool-call boundary and stop-condition detection for block decoding (the
    block resolves all at once; you can't scan left-to-right for an end marker). Debug against the
    DiffusionGemma stand-in before Diffusion-Qwen exists.

## Knob cheat-sheet

| Knob | Start | Note |
|---|---|---|
| block size B | warmup 1 → 256 | larger = more parallelism, more smear risk |
| denoising steps K | sweep | fewer = faster, lower quality |
| confidence threshold τ | ~0.9 | higher = more passes, safer; always unmask top-1 |
| learning rate | ~1e-5, cosine | low; critical to preserve AR knowledge |
| fine-tune tokens | ~1B | more for quality |
| precision | bf16 (+fp32 master) | fp8 optional |
| GDN mode | Option A | switch to B if within-block under-mixes |

## What is novel vs the template

Everything except two things is standard Fast-dLLM v2: (1) GDN handling — as cross-block state
carrier (A) or bidirectional within-block scan (B), and the GDN state snapshot in the cache; and
(2) scale (27B vs the template's 1.5B/7B). The rest — block-causal mask, masked-diffusion objective,
block-size warmup, confidence-threshold decode, hierarchical cache — is the template.

---

## References

**Template & method**
- [R1] Fast-dLLM v2: Efficient Block-Diffusion LLM. arXiv:2509.26328 — https://arxiv.org/abs/2509.26328
- [R2] NVlabs/Fast-dLLM (code, v1 + v2; checkpoint `Efficient-Large-Model/Fast_dLLM_v2_1.5B`) —
  https://github.com/NVlabs/Fast-dLLM

**AR → diffusion adaptation**
- [R3] Dream 7B: Diffusion LLMs (AR-init from Qwen2.5, shifted prediction, noise rescheduling, LR
  sensitivity). arXiv:2508.15487 — https://arxiv.org/abs/2508.15487 ; blog: https://hkunlp.github.io/blog/2025/dream/
- [R4] Block Diffusion: Interpolating Between Autoregressive and Diffusion LMs (Arriola et al.).
  arXiv:2503.09573 — https://arxiv.org/abs/2503.09573
- [R5] Autoregressive-to-Diffusion VLMs / A2D-VL (progressive prediction-window curriculum) —
  https://runwayml.com/research/autoregressive-to-diffusion-vlms
- [R6] Stop Training for the Worst: Progressive Unmasking (PUMA; block-size warmup, AR-init).
  arXiv:2602.10314 — https://arxiv.org/abs/2602.10314
- [R9] Scaling Diffusion LMs via Adaptation from AR Models (DiffuGPT / DiffuLLaMA).
  arXiv:2410.17891 — https://arxiv.org/abs/2410.17891

**Inference / decoding + agentic caution**
- [R7] The Bitter Lesson of Diffusion Language Models for Agentic Workflows: A Comprehensive Reality
  Check (also documents block-diffusion confidence-threshold decoding). arXiv:2601.12979 —
  https://arxiv.org/abs/2601.12979

**Existence proof (throughput)**
- [R8] DiffusionGemma (26B MoE, ~4B active; >1,000 tok/s H100, ~700 tok/s RTX 5090; day-0 vLLM) —
  https://deepmind.google/models/gemma/diffusiongemma/ ; docs: https://ai.google.dev/gemma/docs/diffusiongemma

**Target model**
- [R12] Qwen/Qwen3.6-27B (64 layers, 3:1 Gated DeltaNet : Gated Attention, MTP, 262K ctx) —
  https://huggingface.co/Qwen/Qwen3.6-27B

**Method primitive**
- Gated DeltaNet (the GDN layer: gated delta-rule linear-attention recurrence). Underlying method
  behind Qwen3.6's linear-attention sublayers; see [R12] for Qwen's exact head/dim configuration.

> Note: the downstream **calibrated RL bridge** (using Diffusion-Qwen as a rollout sampler) is a
> separate document. Its references (off-policy correction, fp8 RL, on-policy distillation) live
> there, not here.
