# GDN Block-Diffusion Inference Speedup — Research Synthesis

Date: 2026-06-28. Source: multi-agent research workflow (5 angles + synthesis), verified citations.
Question: how to make cacheless block-diffusion inference over the Qwen3.5-9B Gated DeltaNet hybrid
tractable WITHOUT corrupting correctness, or find liftable implementations.

## Headline: your exact architecture is already published

- **FLARE (arXiv 2606.01774, "Diffusion for Hybrid Language Model")** — VERIFIED real. A conversion
  framework taking **hybrid-attention AR checkpoints** (project page: Qwen3.5 GDN + Gated Attention) to
  one model supporting AR-verified AND diffusion-parallel decoding, with a **strided-checkpoint GDN-state
  cache**, AR-Trust/Diffusion-Trust modes, B=4, ~1097 tok/s on A100. Identifies **transfer-data quality**
  as the dominant factor. Project: https://tokflare.github.io . **Code "coming soon" — not released yet.**
  This is the same-architecture blueprint; monitor for code drop.
- **Nemotron-TwoTower (arXiv 2606.26493)** — VERIFIED, **code + weights released**
  (`nvidia/Nemotron-TwoTower-30B-A3B-Base-BF16` ships `modeling_nemotron_twotower.py` + `inference.py`).
  Frozen AR context tower maintains the state cache, advanced block-by-block; denoiser seeded from cached
  state per block. 2.42× throughput at 98.7% AR quality. BUT Mamba-2 (not GDN), two-tower, 60B — port the
  pattern, not a drop-in.

## Two buckets (complementary, multiplicative)

### Bucket A — cheap valid wins NOW (cache-free, structurally cannot corrupt GDN state)
Why corruption-proof: in the no-cache regime every step rebuilds the full forward pass, so state is never
stale. These only change *how many tokens commit per step* → attack the `O(steps × blocks)` factor.
Residual risk is **quality/calibration only**, measurable on a heldout set.

| # | Option | Speedup | Notes |
|---|---|---|---|
| **A1** | **Confidence-threshold parallel decoding** (Fast-dLLM, arXiv 2505.22618) — commit ALL masked tokens above τ per step (≥1 always) | **2.5–5.8× alone, ~0.1% acc loss** | Already in Fast-dLLM (`threshold`). Re-tune τ for the converted GDN model (0.9 was calibrated on full-attention LLaDA/Dream; may be mis-calibrated). **Adopt first.** |
| A2 | Prompt-prefix state reuse — compute the prompt's GDN state + attn-KV once, reuse as fixed causal initial state | Large if prompts long | Valid iff prompt is condition-only (standard block-diffusion mask). Zero corruption risk. |
| A3 | SchED (2512.02892) / anchor early-commit AHD (2604.08964) | 2.3–4.0× / 70–80% step cut | Training-free plug-ins; stack on A1. |
| A4 | Sliding-window on the 1-in-4 FULL-ATTENTION layers only (keep GDN exact) | Modest | Do NOT window GDN layers. |

**Combined A: plausibly ~3–8× cache-free, no GDN state touched.**

**NON-win (corrects an earlier hypothesis):** "KV-cache only the 1-in-4 attention layers" does NOT help
alone — the 48 GDN layers re-scanning the full prefix every step are the bottleneck; caching only the 16
attention layers leaves the asymptotics unchanged. Attention-KV caching is a *companion* to the GDN
snapshot, never a standalone fix.

### Bucket B — the cache project (proper structural fix, ~1–3 weeks solo on one 5090)
- **B1: GDN-state snapshot cache.** Snapshot the committed-prefix causal GDN state at each block boundary
  (+ KV for attn layers); recompute ONLY the active block per denoise step from the frozen snapshot;
  advance state ONCE on commit. Per-step `O(full-context) → O(block)`, AR-like latency. **Multiplies with
  Bucket A.**
  - **The kernel primitive already exists** — flash-linear-attention's `chunk_gated_delta_rule` accepts
    `initial_state` and returns `final_state` (`output_final_state`), shape `(B,H,d_k,d_v)`. State
    threading per block is a supported API; **no new kernels needed.**
  - **Bit-exact-validatable** against your existing slow full-recompute path → test-driven build, not a
    research gamble. ~3–7 days for a correct prototype, +1–2 weeks to harden.
- B2: vLLM/SGLang integration — high risk, 4–10+ weeks, production serving only. Avoid unless needed.
- B3: retrain GDN bidirectional-within-block (DiffuMamba-style) — out of scope; forfeits the O(1) cache.

## Correctness traps (what silently corrupts GDN state)
1. **Snapshot ONLY at a committed block boundary, never mid-denoise.** (This is almost certainly the
   current bug: Fast-dLLM block cache advances/reuses state without snapshot+restore at commit.)
2. **Never advance state with masked/tentative tokens.** Hold committed-prefix state read-only during all
   within-block steps; advance once after the block finalizes.
3. **Bidirectionality gate — CHECK FIRST (sizes the whole project).** A single causal snapshot is exact
   IFF the conversion kept GDN layers causal-within-block (our `option_a_causal_gdn_v0` does). If GDN was
   made bidirectional within-block, one causal state is insufficient (need forward-cache + within-block
   backward re-scan, ~2× block cost).
4. **Off-by-one / block alignment** — align the diffusion block size to the GDN chunk size (else silent
   prefix-cache failure, cf. vLLM hybrid-cache bugs).
5. Fork state per branch for speculative/parallel candidates (copy-on-write).
6. Prompt-prefix reuse valid only if mask is strictly condition-only.
7. Confidence/entropy threshold *accuracy* gains were measured on full-attention dLLMs; re-tune on a GDN
   heldout set (quality, not corruption).

## THE recommended next step (this week)
Build a **one-file bit-exact validation harness against the existing slow path**, starting with the
causal-within-block check: for a short prompt + one block, compute GDN-layer outputs two ways —
(a) full-recompute (current path), (b) snapshot-`S_p` + active-block re-scan via FLA
`initial_state`/`output_final_state` — and `assert torch.allclose(a, b, atol=1e-3)`.
- If it matches → GDN is causal-within-block (trap #3 cleared) → the entire B1 cache project is proven
  valid and unblocked, as a test-driven build.
- If not → you've discovered you need the forward-cache + within-block-backward split *before* spending
  weeks.
In parallel (same day, hours, zero corruption risk): flip on confidence-threshold parallel decoding (A1)
for an immediate ~3–5× to relieve the eval pain while B1 lands.

(Note: this bit-exact harness is the same AR/state-equivalence de-risk recommended at project start —
independently re-derived here as the gating first step for the cache.)

## Verified-vs-inferred ledger
- VERIFIED: FLARE 2606.01774 exists (code not yet released); Nemotron-TwoTower 2606.26493 exists with
  released modeling+inference code; Fast-dLLM parallel 2.5–5.8×/~0.1% loss (2505.22618); SchED 2.3–4.0×
  (2512.02892); AHD 70–80% step cut (2604.08964); DiffuMamba mechanism (2511.15927, no code); Marconi
  SSM-checkpoint "exact-match-only" rule (2411.19379); FLA `initial_state`/`final_state` API.
- INFERRED (analysis, not measured on a converted GDN bridge): the per-step cost table; GDN-snapshot is
  load-bearing / attention-only caching insufficient alone; bit-exactness of snapshot+block-rescan;
  effort estimates; prompt-prefix reuse. FLARE's specific cache numbers are author-reported via the
  project page, not independently confirmed from the PDF.
