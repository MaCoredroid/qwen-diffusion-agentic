# Lossless prefix cache — DESIGN (APC-artifact class fix) — for monitor review

**Status:** design only, no implementation. CPU/code+artifact analysis; the S2 pilot owns the GPU.
**Depends on:** `runs/lossless_apc/rootcause.md` (committed `cc3a422`) — the mechanism, census, and
first-divergence file:line are established there; this doc designs the fix, its validation, and its cost.
**End goal (user):** SWE-class agentic multi-turn serving needs a **lossless** prefix cache on the
diffusion engine: **cache-on decode byte-identical to fresh-context decode.** Today's align-APC is
functional but lossy on the `{20,21,60}`+`{16,130}` artifact class (diverges only under cross-turn
reuse; restored under cold prefix).
**Pin:** `shared/vllm_p2_pr42406` @ `0b44dcc` (branch `qwen3_5-flare-modelstate`). Deployed export runs
canvas/commit block **32**, `mamba_block_size` (align-APC checkpoint stride) **1024**, `_GDN_CHUNK` **64**,
`--mamba-ssm-cache-dtype float32`, GDN state cache `[283, 32,128,128]` fp32.

---

## 0. The invariant (this is the whole design in one line)

> **LOSSLESS-APC INVARIANT.** For every reusable checkpoint `C` at a `MAMBA_BLOCK_SIZE`-aligned
> (1024) boundary, the durable published `(ssm_state, conv_tail)` at `C` must be **bit-identical** to
> the `(ssm_state, conv_tail)` a **fresh prefill of the exact token prefix `[0..C)` produces via the
> identical kernel path**, same boot. Equivalently: substituting the reused seed with a fresh
> recompute changes **zero output bits**. The gate is **bitwise identity**, never fp-closeness.

Everything below is machinery to make that invariant true *by construction* (produce the durable state
with the same computation fresh uses) and a validation plan that gates on **measured raw byte-parity**,
never on a scalar accept-rate or on `exact_args` (both are APC-invariant and provably blind to this
class — the flywheel's own 22/482 = 4.6% argmax drift sailed past a scalar accept gate,
`FR13_KERNEL_STATUS.md:49`, `FR13_SHARP_LOCALIZE_BIND.md`).

---

## 1. The lossless-by-construction mechanism

### 1.1 Why today's publish is lossy (one-paragraph recap; full trace in rootcause.md §2)

Attention KV reuse is exact (per-position K,V re-read by the same kernel). The GDN recurrent state is
the only **lossy summary**: its bits depend on the *order* tokens were folded. The cached state is
accumulated by the **commit path on 32-absolute canvas boundaries** (`block_commit_target`,
`qwen3_5_flare_ops.py:181-214`, `aligned=True`) using the **chunked** kernel `chunk_gated_delta_rule`
(`qwen_gdn_linear_attn.py:1518`), then copied into the 1024-aligned align row
(`run_fused_postprocess_align` via `postprocess_state`, `mamba_hybrid.py:290-329`; checkpoint advance
`new_state_idx = ceil(computed_after / MAMBA_BLOCK_SIZE) − 1`, `mamba_utils.py:302`). A **fresh** prefill
re-runs the same tokens as **one chunked call on the 64-grid anchored at absolute 0**. Because
`32 % 64 != 0`, half the commit boundaries **bisect a 64-chunk**; the chunk scan is fp-**non-associative**
across chunk boundaries, so the 32-grouped reduction ≠ the 64-grouped reduction. The bits diverge at the
first bisected boundary (durable when written at `qwen_gdn_linear_attn.py:1532`, reused verbatim next turn
at `:1513`) and surface as a ~fp-epsilon shift flipping the first value/name near-tie. The constructor
**already hazard-logs exactly this** (`qwen3_5_flare.py:248-261`), and the code already ships
`_DEFAULT_BLOCK = _GDN_CHUNK = 64` (`:116-119`) as the safe default the deployed config overrides to 32.

Two independent lossiness sources must both be closed:
- **(S1) grid mismatch** — 32-absolute fold boundaries vs the fresh 64-grid.
- **(S2) chunked-vs-recurrent** — even with matched boundaries, a *chunked* kernel is not bit-exact to a
  sequential recurrent scan (flywheel: WY chunked kernel **parked** for a ~6e-5 chunk-vs-recurrent gap,
  *"a different summation tree, never bit-exact to native"*, `FR13_KERNEL_STATUS.md:33-34`). This is why
  "cache only chunk-aligned states" is **necessary but not sufficient** (rootcause Refinement 1).

### 1.2 Key structural decision: DECOUPLE the GDN durable-publish grid from the canvas/commit block

The task constraint "32-absolute alignment stays" is honored by recognizing that the **32-canvas** and
the **GDN state fold grid** are two *separate* concerns that the current code accidentally couples:

- The **32-absolute canvas/commit** governs which tokens denoise/commit per step, the bidirectional
  block-attention mask, KV-append cadence, and the HF block-structure match (`block_commit_target`
  docstring, `qwen3_5_flare_ops.py:189-196`). **This stays exactly as-is.**
- The **GDN durable-state publish** governs the bits written into the reusable 1024-aligned row. **Only
  this changes.** It moves off the 32-fold running state onto a **canonical 64-grid recompute** so the
  published bits equal fresh-prefill bits.

Decoupling means the fix touches only the *durable publish*, not the hot denoise/commit path, KV, the
read-only-denoise discipline, or the HF-matching 32 alignment.

### 1.3 Route A (PRIMARY) — canonical-grid durable publish (unconditionally bit-exact, closes S1+S2 together)

At the moment a request crosses a 1024 checkpoint `C_k` (the only position vLLM ever *reuses*), do **not**
copy the 32-fold running state into the align row. Instead **replay the committed tokens of the window
`[C_{k-1}..C_k)` through the exact call a fresh prefill would make** — one `chunk_gated_delta_rule` over
those ≤1024 tokens on the 64-grid anchored at absolute 0, seeded from the previous *canonical* checkpoint
`C_{k-1}`'s state — and publish **that** result. Because it is byte-for-byte the same op, same grid, same
fp32 accumulation, same seed, same inputs as fresh prefill, its output is **bit-identical to fresh** on a
same-boot deterministic GPU (the RTX-5090 cold-prefix certificate proves that determinism holds:
`nevertrain_parity_cert_resetapc.jsonl`, v3b a==b across two boots).

This is the diffusion translation of the flywheel's shipped fix — **accept-only replay from h0 on a
recurrent path bit-exact to native** (`FR13_KERNEL_STATUS.md:13-24`). Our commit path already tracks
accepted tokens (`num_accepted_tokens` scatter, `postprocess_state`, `mamba_hybrid.py:300-329`); Route A
is "replay the accepted chain from the last canonical h0 and publish."

**What must be buffered.** The canonical replay needs the committed tokens' GDN inputs (q,k,v,g,beta) for
the window. Two implementation options (defer the choice to build, gate decides):
- **(A-buf)** a per-request ring buffer of the last ≤1024 committed tokens' per-layer GDN inputs, folded
  in one call at the 1024 publish. Simple, unconditionally = one fresh call; costs HBM
  (~1024 × per-layer q/k/v/g/beta × #GDN layers).
- **(A-reproj)** store only committed **hidden states** and re-run the GDN `in_proj` + the single chunk
  scan at publish. Cheaper storage, one extra projection pass per 1024 tokens.

Either way the extra compute is **one chunk scan per 1024 committed tokens** (amortized ≈ one extra GDN
pass per 32 commits) — negligible next to the denoise/commit work it rides behind.

### 1.4 Route A′ (optimization to *validate against A*, not to trust blind) — incremental 64-window accumulator

Maintain a second, canonical accumulator that advances **only on 64-boundaries**: whenever two
consecutive 32-commits complete a 64-window, fold those 64 tokens as one chunk call into it (seeded from
its own prior value). At 1024 it is published directly. This needs only a **64-token** buffer, not 1024.
It is bit-exact to Route A **iff** the GDN chunk kernel's cross-chunk carry is a strict left-to-right
sequential fold (so a chain of 64-calls == one 1024-call). The rootcause phrase *"non-associative across
chunk boundaries"* leaves open whether the cross-chunk `chunk_states` reduction is sequential or a
tree/parallel reduction; if tree, A′ ≠ A. **Therefore A′ ships only after the byte-parity gate (§3a)
shows A′ == A on the census turns.** Route A is the correctness baseline; A′ is a memory optimization the
gate must earn.

### 1.5 Route B (FALLBACK — simplest, but changes the HF-match) — block = 64

Set the canvas/commit block to a multiple of 64 (the code's own `_DEFAULT_BLOCK=64`, `qwen3_5_flare.py:119`).
Then commits never bisect a chunk, the running fold *is* canonical, and S1 vanishes with no replay
machinery. Cost: block=64 no longer matches HF's 32-token block structure, so the bidirectional denoise
mask changes and engine-vs-HF fresh output shifts. **This is acceptable for the user's stated target**
(cache-on == fresh is a self-consistency goal; engine == HF is a separate, deferred goal — rootcause §
"Not in scope"), but it **violates the task's "32-absolute alignment stays" constraint**, so it is the
fallback only if Route A's decoupling proves infeasible or the 32-canvas HF-match is later judged
non-load-bearing. Recorded for completeness; **not recommended as primary.**

### 1.6 The conv-tail carrier — isolate BEFORE assuming ssm-only (rootcause Refinement 2)

The commit publishes a **raw pre-conv conv tail** (`tail_after_append`, `qwen3_5_flare_ops.py:393-410`;
width `conv_kernel_size − 1`) alongside the fp32 ssm state (`FlareBoundarySnapshot`, `:360-382`). The
flywheel's APC-hit localizer found the **restored conv prior-window**, not the ssm state, to be the
**prime** AR carrier (ssm 47/48 clean; `scripts/fr13_apc_hit_first_divergence.py:6-19`). The conv tail is
**per-token exact data** (like KV), not a recurrent summary, so it *can* be made bit-exact — the flywheel
bug was a stale physical-block read, a storage/indexing defect, not a kernel non-associativity. **Design
requirement:** run the conv-tail-restore-vs-ssm-restore isolation probe (§3, ported from
`fr13_apc_hit_first_divergence.py`) on a `{16,130}`-style turn **before** finalizing; if the conv tail is
a carrier, fix its storage/read to the exact raw-qkv boundary window (byte-exact, no kernel change). Do
not build an ssm-only fix on the assumption that ssm is the sole carrier.

### 1.7 Determinism prerequisite (what "same computation → same bits" rests on)

Bit-identity requires the canonical replay to run under the **same reduction order / batch-invariant
settings** as fresh prefill. The flywheel's `FR13_BRANCH_FLIP_LOCALIZED_BIND.md` shows non-BI custom ops
(fp8 GEMM, tree-attn) *not* covered by `enable_batch_invariant_mode` can perturb bits under co-residency.
On the diffusion pin (greedy, bf16→fp32 GDN state, RTX-5090) the cold-prefix certificate already
demonstrates byte-reproducibility across boots, so BI holds for the GDN chunk kernel there. The design
**inherits** that regime and the §3 gate catches any residual. **Portability caveat:** on GB10/Spark,
fresh B=1 boots fork at tokens 11–71 (`FR13_SHARP_LOCALIZE_BIND.md`), so the cross-boot byte-gate is
invalid there — any future GB10 deployment must switch to the **same-boot in-process** oracle gate.

---

## 2. What changes in the pin, and what must stay

### 2.1 CHANGES (all confined to the durable-publish seam)

| # | file:function | change |
|---|---|---|
| C1 | `qwen3_5_flare.py` `postprocess_state` (:960-990) / `mamba_hybrid.py` `run_fused_postprocess_align` (:290-329) | at a 1024-checkpoint crossing, publish the **canonical replay** state (Route A) instead of the 32-fold running state. Only the *value written to the align row* changes; the advance logic (`mamba_utils.py:302`) is untouched. |
| C2 | new op in `qwen3_5_flare_ops.py` (CPU-testable, beside `block_commit_target`/`FlareBoundarySnapshot`) | `canonical_checkpoint_refold(...)`: the single-call 64-grid chunk scan over the window's buffered GDN inputs, seeded from `C_{k-1}`. Pure state-machine primitive with a CPU parity test vs a reference recurrent fold. |
| C3 | `Qwen3_5FlareRequestStates` (:124+) | per-request ring buffer for committed GDN inputs (A-buf) *or* committed hidden states (A-reproj); sized to `MAMBA_BLOCK_SIZE` (A) or `_GDN_CHUNK` (A′). |
| C4 | `qwen3_5_flare_ops.py` conv-tail path (:393-410) — **conditional on §1.6 probe** | if conv tail is a carrier: store/read the exact raw-qkv boundary window by logical position, not physical block_id. |

### 2.2 STAYS (must survive byte-for-byte; these are load-bearing correctness properties)

- **The af21dc8 read-only-denoise discipline** — `snapshot_readonly_rows`/`restore_readonly_rows`
  (`qwen3_5_flare_ops.py:321-352`) and restore-is-final-authority ordering
  (`qwen3_5_flare.py:965-979`): denoise advances GDN state by exactly 0. The fix touches only the
  **commit** publish, never the denoise rows.
- **32-absolute canvas/commit alignment** — `block_commit_target` `aligned=True`
  (`qwen3_5_flare_ops.py:211-214`): unchanged. KV-append cadence, bidirectional mask, HF block match
  all preserved. (Route A explicitly decouples the GDN grid from this so it need not move.)
- **fp32 boundary carrier + `assert_fp32_boundary`** (`qwen3_5_flare_ops.py:360-390`) — FR13 R3.
- **cudagraph capture of the hot denoise/commit path** — the canonical replay is an occasional
  (per-1024) publish-time op; it runs **outside** the captured region (eager), or is captured at a small
  bucketed set of window lengths. The hot path's captured shapes are unchanged. (Risk R2.)
- **1024 checkpoint stride / align-APC geometry** (`mamba_utils.py:302`, `[283, 32,128,128]` fp32) — the
  reuse granularity and cache slot geometry are unchanged; only the *contents* of the published row
  change.

---

## 3. Validation plan (gates — all gate on measured RAW byte-parity, same-boot oracle)

**Instrument.** The binding probe is a **per-token argmax-vs-fresh-context-oracle** teacher-forced
comparison (the only instrument that caught the flywheel's 4.6% drift, `FR13_SHARP_LOCALIZE_BIND.md`;
`FR13_GATE_BLINDSPOT`). Reuse the existing cold-prefix certificate machinery
(`nevertrain_parity_cert_resetapc.jsonl`) as the fresh reference. **Never** gate on `exact_args` (APC-
invariant, 130/247 either way) or a scalar accept rate.

**(a) Artifact-census byte-match — the decisive gate.** Under cache-on, turns **{gt20, gt21, gt60}**
(matched-20, pin e5496cc) and **{gt16, gt130}** (never-train, pin 95d8b47) must produce output
**byte-identical to their fresh-context (cold-prefix) decode**, at the named first-divergence positions
(`pos_mod32 ∈ {15,19,20}`). The `{16,130}` **mirror** is the sharpest single case: both turns share
`…<parameter=source>\ntest` on the `\n`(198)↔`_image`(4794) near-tie and today flip *opposite* ways under
reuse; lossless-APC must make **both** match their cold decode. PASS = 5/5 census turns byte-match fresh
under cache-on. Also run the **conv-vs-ssm isolation probe** (§1.6) here to confirm the carrier
attribution the fix assumed.

**(b) Full-battery cache-on == fresh certificate (upgrades the parity cert to cache-on).** Re-run the
whole 63-turn matched battery and the 184-turn never-train battery **with cache reuse ON**, and require
per-turn byte-parity vs the fresh-context certificate on **every APC-class turn**. This upgrades
`nevertrain_parity_cert_resetapc.jsonl` from a *cold-prefix* certificate to a *cache-on* certificate.
Expected residual: the **control** fp-residue class (`gt44` + the 10 never-train breaks, rootcause §1b)
still breaks — it breaks fresh too, so it is out of scope (engine-vs-HF, not cache-on-vs-fresh); the gate
asserts the APC-class turns match and the control set is **unchanged** (no new breaks introduced).

**(c) Multi-turn agentic speedup — the end-goal payoff number.** On 1000+-token growing-context episodes
(the SWE-class regime), measure **per-turn wall-clock speedup from APC** on the diffusion engine (cache-on
vs cache-reset) and compare to the **flywheel AR-with-APC** (FR13 replay route) on the same episodes.
Report: prefill tokens saved/turn, GDN-state reuse hit-rate at 1024 granularity, s/turn cache-on vs
cache-off, and the engine-vs-flywheel-AR per-turn delta. This is the number that justifies the fix for
the end goal; it must show a real per-turn win at **held byte-parity** (a) — speed with losslessness, not
instead of it.

**(d) No-regression set.** (i) CPU state-machine tests for the new `canonical_checkpoint_refold` +
existing `qwen3_5_flare_ops` suite green. (ii) The batched-bench correctness gates
(`runs/p2_engine_batchgates`) — gate-1 companion-invariance stays byte-identical 8/8; the "gt130
companion-INVARIANT near-tie flip" now resolves to the fresh value. (iii) Throughput/util unchanged on
the non-reuse path (the fix adds nothing to fresh prefill or denoise). (iv) A′==A byte-check before A′
ships (§1.4). (v) cudagraph capture succeeds and hot-path captured shapes are unchanged (§2.2 / R2).

---

## 4. Effort + GPU-hours

**Engineering (design → landed, on-GPU implementer):**
- C2 canonical refold op + CPU parity test: ~1.5 days.
- C1/C3 publish-seam rewire + per-request buffer: ~2–3 days.
- §1.6 conv-vs-ssm isolation probe (port `fr13_apc_hit_first_divergence.py`) + C4 if needed: ~1–2 days.
- Bring-up / byte-parity debugging (the real cost — chasing the last bits, BI/cudagraph interactions):
  ~3–5 days.
- **Total ≈ 1.5–2 weeks** of focused GPU-adjacent work (this doc + rootcause remove all the analysis).

**GPU-hours (RTX-5090, greedy, single-stream):**
- Census + isolation probes (a): capture-heavy, iterated — **~8–15 GPU-h** across debug cycles.
- Full-battery cache-on reruns (b): 63 + 184 turns × a few configs; a turn is ~1–6 s ⇒ **~2–4 GPU-h**
  per full pass, budget **~6–10 GPU-h** with iteration.
- Agentic speedup bench (c): 1000+-token episodes, engine + flywheel-AR arms — **~4–8 GPU-h**.
- No-regression (d): CPU tests are free; GPU gates **~2–3 GPU-h**.
- **Total ≈ 20–35 GPU-hours**, bring-up debugging dominant. No training; serving-correctness only.

---

## 5. Risks

- **R1 — fp-determinism limits (the hard floor).** Bit-identity holds only where the GDN chunk kernel is
  deterministic and batch-invariant. Route A is *by construction* the same op as fresh, so it inherits
  fresh's determinism — but if any non-BI custom op sits on the GDN input path under co-residency
  (`FR13_BRANCH_FLIP`), the published bits could still perturb vs a differently-batched fresh run. Mitigate:
  run the canonical replay under the same BI settings; gate (a) catches it; on GB10 switch to the same-boot
  in-process oracle (the cross-boot gate is invalid there).
- **R2 — cudagraph shape interaction.** The variable-length (≤1024) canonical replay must not force
  recapture or perturb the hot path's captured shapes. Mitigate: run it eager outside the captured region
  (its per-1024 cadence makes the eager cost negligible), or capture a small bucketed set; gate (d-v).
- **R3 — cache-hit-rate under 1024 granularity.** Mamba reuse only fires at ≥1024-matched prefixes
  (unchanged by this fix; set by `MAMBA_BLOCK_SIZE`). Sub-1024 shared prefixes get no GDN reuse and
  re-prefill — existing behavior, not a regression, but it bounds the (c) payoff on short contexts. The
  32-absolute canvas alignment does **not** change this granularity (it is not the reuse stride).
- **R4 — Route A′ over-trust.** A′ (64-window incremental) is only bit-exact if cross-chunk carry is
  strictly sequential; if it is a tree reduction, A′ ≠ A. Mitigate: A′ ships only after gating A′==A (§1.4);
  Route A is the correctness baseline regardless.
- **R5 — buffer memory (A-buf).** Buffering ≤1024 tokens × per-layer GDN inputs × #GDN layers costs HBM
  per in-flight request. Mitigate: A-reproj (store hidden states, re-project) or A′ (64-token buffer) if
  A-buf pressures the 5090's 32 GB; pick by measurement, not upfront.
- **R6 — mis-attribution (conv vs ssm).** If the fix assumes ssm-only and the conv tail is actually the
  prime carrier (as it was for the flywheel AR path), losslessness will not close. Mitigate: §1.6
  isolation probe is a **precondition**, not a follow-up.
- **R7 — scope creep to engine-vs-HF.** The fp-residue control class (`gt44` + 10) breaks fresh too and is
  explicitly out of scope; do not let a "still not matching HF" observation redirect the fix. The target is
  cache-on == fresh (achievable), not engine == HF (a separate deferred kernel-level goal).
