# FLA fused GDN kernel — feasibility (workflow ww7vf3mge, 2026-06-30)

**Decision-grade research (4 web agents + 2 codebase agents + adversarial synthesis). Banked.**

## Verdict: CONDITIONAL cheap drop-in — the whole bet rides on ONE 1–2h spike test
- **Math match PROVEN, not by analogy:** the official HF `qwen3_next` modeling calls
  `fla.ops.gated_delta_rule.chunk_gated_delta_rule` as the PRIMARY path and falls back to a torch fn **named
  `torch_chunk_gated_delta_rule`** only when FLA is absent. **OUR local scan IS that exact reference fallback**
  (same name, same math; our harness already validated it bit-exact). So FLA is, by construction, the kernel our
  function mirrors. This is also what FLARE's Route I uses ("reuses standard FLA kernels", C=64).
- **Call-site flag map:** q/k L2-normed in module → `use_qk_l2norm_in_kernel=False`; beta pre-sigmoided →
  `use_beta_sigmoid_in_kernel=False`; `g` per-token, **drop our cumsum** (FLA cumsums per-chunk); `scale=1/sqrt(d_k)`;
  `allow_neg_eigval=False`; ShortConv stays in the module (layer-separate, never in the swap zone); GVA native.
- **Backward + bf16 + per-block initial_state with gradient flow: YES.** `chunk_gated_delta_rule` is a
  `torch.autograd.Function` with working backward, autocast bf16, and returns **`dh0`** (grad wrt initial_state).
  With `cu_seqlens` it treats initial_state as per-segment and resets at boundaries = EXACTLY our FLARE two-stream
  schedule (seed noisy block from clean boundary, reset per block; keep caller-side `.detach()` at modeling.py:1199).
- **One real wrinkle (not a blocker):** FLA returns only `(o, final_state)`, no `output_chunk_states`. Fix: in
  `clean_gdn_docwise_with_boundaries`, per-block loop calling FLA `output_final_state=True`, thread final→initial.
  Noisy stream → pack the 288 per-block scans into one `cu_seqlens` call with per-segment initial_state (the direct
  cure for the residual idle).
- **SINGLE BIGGEST RISK:** the FLA GDN **backward** Triton kernel on **sm_120 Blackwell** — issue #607 (`tmem_store`
  bwd crash), #734 (cumsum crash). Those were on Triton 3.2–3.6/3.4; **we're on Triton 3.7.1 (newer than every
  report)** → may be fixed, UNVERIFIED → must test. #734 has a cheap `torch.cumsum` fallback; #607 has none → if it
  reproduces, STOP, keep 63%, do NOT hand-roll Triton.
- **Effort:** spike 1–2h → green = 0.5–1.5 days; #607 reproduces = 0 days, accept 63%.

## Integration plan (AFTER the v2 retrain + endgame eval)
- **Step 0 GATE (~1–2h, decides all):** `uv pip install "flash-linear-attention>=0.5.1"` (BARE, not `[cuda]` — the
  extra clobbers our torch 2.12.1+cu130 / triton 3.7.1). Run fwd+bwd on one bf16 block on the 5090 with non-zero
  initial_state. Check: no tmem_store/`no kernel image` throw; grads incl. dh0 finite; o/final_state match torch at
  fp-eps. Green→proceed; #734-only→swap that kernel for torch.cumsum→proceed; **#607 bwd crash, no fix→STOP, 63%.**
- **Call-site swap (~4 fns / ~120 lines):** `_torch_chunk_gated_delta_rule_impl` (472–547) → thin FLA adapter;
  wrapper (429–469) return-unpack; `clean_gdn_docwise_with_boundaries` (714–757) → per-block output_final_state loop;
  noisy stream → cu_seqlens packed call OR keep batched grouping; checkpointing keep use_reentrant=False, pass
  initial_state as explicit tensor into the checkpointed region.
- **Validation gate (our discipline):** parity (o/final_state + dq/dk/dv/dbeta/dg + dh0 vs torch at fp-eps; bf16
  ~1e-2 rel ok) → loss-overlay (few hundred two-stream steps, curves overlap) → no raw/constrained eval regression →
  re-measure util. **Success = util ~63% → ~90–100% with parity held.** Clamp gate slightly <1 (#389 instability).

## Lead explanation (the idea)
GDN = linear-attention RNN with a matrix state `S_t` (d_k×d_v associative memory):
`S_t = α_t(I − β_t k_t k_tᵀ)S_{t−1} + β_t k_t v_tᵀ`, `o_t = q_t S_t`. α=gated decay (Mamba2-style global forgetting),
β=delta-rule write strength; `(I−βkkᵀ)` is a Householder rank-1 correction (read current assoc, write targeted
error-correction). **Pure-torch chunking starves the GPU** because the recurrence is elementwise decay + tiny rank-1
GEMV on a d×d state (low arithmetic intensity, memory-bound, can't saturate tensor cores), parallel only over
batch×heads (we're batch 1), and an eager Python chunk loop host-launches hundreds of tiny CUDA kernels (~µs launch
overhead dominates). **The fused chunkwise-parallel Triton kernel** uses the **WY representation** (Bischof–Van
Loan): a product of Householder updates `∏(I−β_i k_i k_iᵀ)` collapses into `I − Σ w_i k_iᵀ` = matmul-shaped work. So
it (a) precomputes intra-chunk W,U via triangular solve, (b) computes chunk outputs with **dense attention-like
GEMMs**, (c) carries only the d×d boundary state across L/C chunks — fused into a few Triton kernels, state in
SRAM/registers, large dense GEMMs lighting up tensor cores → high util. Exact GDN math up to fp ordering (~16× vs
elementwise). Same kernel that trains Qwen3-Next.

**Sources:** GDN paper arxiv 2412.06464 · DeltaNet WY/chunkwise arxiv 2406.06484 · Songlin Yang "DeltaNet Explained
II" (sustcsonglin.github.io/blog/2024/deltanet-2/) · FLA kernel source (github fla-org/flash-linear-attention
fla/ops/gated_delta_rule/chunk.py) · HF Qwen3-Next modeling (FLA primary + torch fallback) · FLA Blackwell issues
#607/#734/#389/#759 · adjacent QwenLM/FlashQLA (faster but SM90/SM100 only, EXCLUDES sm_120).
