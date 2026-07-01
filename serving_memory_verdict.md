# Diffusion serving memory verdict (workflow wqldp8bz1, 2026-07-01) -- monitor-red-teamed

> 32GB is a SOFT ceiling (bf16 16.68GB weights + 4x-oversized KV cache), NOT hardware. Diffusion @ 7.6k B=1 = ~19-20GB (~12GB free). GREENLIT (user standing approval): the LOSSLESS fixes (expandable_segments + 4-head KV + drop commit clone), NOT 4-bit (4-bit deferred -- it costs T1/T2/T3 re-validation + AR re-baseline). The empirical 31.5GB was ENVIRONMENTAL confound (co-resident SGLang AR + fragmentation), not the diffusion.

Verified against the actual config, weight floor, and the three measured throughput/OOM artifacts. Here is the decision-grade verdict.

---

# VERDICT: 32 GB is NOT the limit for your 7.6k workload. You are ~70% optimized — one lossless fix + one env var reclaim the table; 4-bit weights are the ~11 GiB lever but the only one that costs you a re-baseline.

All numbers below are grounded in `config.json` (8 full-attn + 24 GDN layers, `num_kv_heads=4`, `head_dim=256`, `num_heads=16`, vocab 248,320, GDN 32 v-/16 k-heads × 128, conv kernel 4), the on-disk bf16 weight total (**16.68 GiB**), and the measured artifacts in `runs/flare_hf_cache/`.

## (1) Memory breakdown — B=1, prefix 7,600, bf16, block-cache serving path

| Term | Formula | Size | Scales with? |
|---|---|---:|---|
| Model weights (bf16, LoRA merged) | safetensors total | **16.68 GiB** | fixed floor |
| Attn KV — 8 layers, **as cached today (16 heads)** | 2·16·256·2 B ·8 ·7600 = 128 KiB/tok | **0.93 GiB** | linear in B×L |
| GDN recurrent state — 24 layers fp32 | 24 × [32,128,128]×4 B | **48 MiB** | **constant** |
| GDN conv tails — 24 layers | 24 × [1,3,8192] | ~2 MiB | constant |
| CUDA context + cuBLAS workspace | measured non-PyTorch | **~1.3 GiB** | fixed |
| Denoising activations (32-tok block only) | attn wts ∝L but 32 query rows + block logits [1,32,248320] ≈ 32 MiB fp32 | **<0.2 GiB** | mild |
| **TOTAL** | | **≈ 19.3 GiB** | **~12 GiB free** |

**The load-bearing structural fact:** the served path asserts `residual_full_context_model_calls == 0` (confirmed in every artifact). It denoises a **32-token block** seeded from cache and warms the 7.6k prompt block-by-block. So the two things the general dLLM-serving literature warns about **do not occur here**: there is **no `[B,L,V]` logit spike** (~2 GiB in LLaDA-style full-sequence diffusion → your block-only logits are ~32 MiB) and **no O(L²) full-context attention matrix**. Fast-dLLM v2 block diffusion is its own mitigation for the diffusion-specific memory crisis.

## (2) Are we optimized? What's on the table, ranked by size

- **bf16→4-bit weights: real, ~11 GiB. This is by far the biggest lever.** 16.68 GiB → ~5.5 GiB (nf4) or ~9 GiB (int8). *Caveat: it's the "~12 GiB" number minus the LoRA math, and it is the one change that is not free — see risk in (4).*
- **KV cached at 16 heads instead of 4: ~0.7 GiB @ B=1/7.6k, lossless, ~10 lines.** `attention_project` runs `repeat_kv` *before* caching (`flare_hf_cache.py:191-193`), storing K/V 4× too large. Repeat at SDPA time instead → 0.93→0.23 GiB @ 7.6k; **~1.5 GiB @ B=16/1024**. Bit-identical arithmetic.
- **`expandable_segments:True`: ~0 GiB but unblocks the OOM.** The B=16@1024 failure stranded 622 MiB reserved-but-unallocated; the error message itself recommends this flag.
- **`cat().clone()` at commit (`:308-309`): removes a transient 2–3× KV realloc spike.** Lossless.
- **KV-quant (fp8/int8): near-worthless here — ~0.1 GiB.** Only 8/32 layers carry KV. Skip until 30k+ context.
- **fp32 GDN state: leave it. 48 MiB total, context-independent, and it's your recurrence-numerics carrier.** Optimizing it is noise.
- **Does the GDN hybrid already help long-context? Yes — structurally, but the *absolute* 7.6k saving is only ~0.7 GiB.** 24/32 layers carry a **fixed 48 MiB** state instead of growing KV, so the hybrid pays ~4× less context-KV than a pure-attention 9B. That's decisive at 128k–1M (pure attention would be 16–125 GiB); at 7.6k it's rounding error against the 16.68 GiB weight floor. Bank it for scale-out, not for fixing 7.6k.

## (3) Hard limit or fixable — and at what B

**Fixable, and 7.6k isn't even close to the binding constraint.** At B=1 the constraint is context length, and at bf16/4-head KV you could push a *single* request to ~100k+ tokens before KV fills (~435k at 4-bit). No agentic prompt at 7.6k, or even 30k, is a wall at B=1.

The real ceiling is **batch × context on top of the weight floor.** Anchored to measured fits (usable ceiling ~31.3 GiB, and note the B=16@1024 OOM occurred with ~5.9 GiB held by *other processes* — partly environmental):

| Config | Concurrent 7.6k requests (est.) | Basis |
|---|---|---|
| bf16, today (16-head KV) | **B ≈ 8–10** | measured B=12@1024 fits at 23.4 GiB; KV scales ×7.4 to 7.6k |
| bf16 + 4-head KV + expandable_segments | **B ≈ 16–24** | 4× smaller KV; B=16@1024 clears trivially |
| **4-bit weights + 4-head KV** | **B ≈ 24–32** (activation/compute-bound before memory) | weight floor 16.68→5.5 GiB frees ~24 GiB for KV |

## (4) Ranked recommendations — cheapest win first

| # | Action | GB freed | Effort | Quality/parity risk | Re-validate T1/T2/T3? |
|---|---|---|---|---|---|
| 1 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | ~0 (unblocks OOM) | 1 line | none (allocator only) | **No** — bit-identical |
| 2 | Cache pre-`repeat_kv` K/V (4 heads), repeat at SDPA | 0.7 @B1/7.6k; ~1.5 @B16/1024 | ~10 lines | none | **No** — bit-identical; run T1 once as cheap regression guard |
| 3 | Preallocate KV, drop commit-time `cat().clone()` | removes transient 2–3× KV spike | medium | none | **No** — bit-identical; T1 regression guard |
| 4 | **4-bit nf4 (or int8) serving weights** | **~11 (nf4) / ~7.7 (int8)** | medium | small; changes numerics everywhere | **YES — full T1/T2/T3 + re-baseline AR at same precision.** Two specific hazards: (a) AR baseline was matched at bf16, so a fair diffusion-vs-AR comparison requires AR re-run at `PROFILE=bnb4`; (b) nf4 matmul is **not batch-shape-invariant** → the sample-vs-rescore parity contract must be re-verified on the 4-bit path |
| 5 | KV fp8/int8 | ~0.1 | low | low | only if adopted; skip at 7.6k |

**Do #1–#3 now** (free, lossless, no re-baseline) — they alone take you to B≈16–24 at 7.6k, which almost certainly covers the agentic eval. **Hold #4** until you actually need B>24 at 7.6k or want to push context past ~100k; it's the biggest lever but it's the only one that spends your just-established parity and forces an AR re-baseline, so gate it behind a concrete headroom need.

## (5) Honest note — is any agentic context just too long for 32 GB?

**Not for your stated workload, and not close.** 7.6k at B=1 sits at ~19 GiB with ~12 GiB free; even bf16/B=1 tolerates ~100k tokens, 4-bit ~700k. You would only hit a genuine wall when you need **long context AND real concurrency simultaneously** — e.g., 100k context at B=8 needs ~24 GiB of KV alone, which OOMs at bf16 but *fits* at 4-bit (~30 GiB total). The regime that no optimization saves on 32 GB is roughly **200k+ context at meaningful batch**, or extreme concurrency at long context. If you ever land there the levers are, in order: (a) 4-bit weights first, (b) prompt truncation/summarization of the agentic history, (c) accept small B and serialize, (d) a bigger box. For 7.6k tau2 prompts you need none of these — do #1–#3 and you're done.

**Bottom line:** 32 GB is a soft, self-imposed ceiling created by bf16 weights + a 4×-oversized KV cache, not a hardware wall. Two lossless fixes (expandable_segments + 4-KV-head caching) reclaim the batch headroom for the 7.6k eval with **zero** re-validation; 4-bit weights remain the ~11 GiB reserve lever for future long-context/high-batch work, spent only when a parity re-baseline is justified.

Key files: `/home/mark/qwen_diffusion/scripts/flare_hf_cache.py` (KV repeat-before-cache `:191-193`, commit `cat/clone` `:308-309`, GDN state `:151-176`), `/home/mark/qwen_diffusion/scripts/eval_fastdllm_toolcall_cases.py` (bf16 `load_model` `:795-818`, block dispatch `:94-104`), `/home/mark/qwen_diffusion/scripts/rl_pilot_countdown.py` (4-bit nf4 `:769-786`), `/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-init/config.json`, `/home/mark/qwen_diffusion/runs/flare_hf_cache/throughput_b1_b16_prefix1024.json` (B=16@1024 OOM, fragmentation message).