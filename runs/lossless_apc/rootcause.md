# Lossless prefix cache — ROOT CAUSE of the APC-artifact class (CPU/code+artifact analysis, no GPU)

**End goal (user):** SWE-class agentic multi-turn serving needs a **lossless** prefix cache on the
diffusion engine — cache-on byte-identical to fresh-context decode. Today's align-APC is functional
but lossy: the `{20,21,60}` (matched-20) + `{16,130}` (never-train) artifact class diverges from the
HF reference **only under cross-turn cache reuse** and resolves under cold prefix.

**Verdict (this analysis):**
1. The mechanism is a **GDN recurrent-state chunk/reduction-order fp divergence**: the GDN state a
   reused prefix carries across turns was folded by the **commit path on 32-absolute canvas-block
   boundaries** (which are **mid-chunk** relative to the `_GDN_CHUNK=64` grid a fresh prefill uses),
   so the cached state is bit-different from a fresh recompute. Attention KV reuse is exact; the
   recurrent GDN state is the lossy summary. **First-divergence point named below with file:line.**
2. The **canonical-boundary hypothesis SURVIVES** as the correct direction and correct localization,
   but the flywheel reference proves it is **necessary-but-under-specified**: "cache only chunk-aligned
   states" is not sufficient — the cached/published state must be produced by a kernel path that is
   **bit-exact to the fresh recurrent recompute**, and a chunked GDN kernel is *inherently* not
   (measured ~6e-5 chunk-vs-recurrent gap; the flywheel PARKED its chunked-WY kernel for exactly this).
3. The flywheel native+MTP AR APC is **also fp-lossy** (measured 22/482 = 4.6% per-token argmax flips
   that a scalar accept gate was blind to) — the diffusion project is not uniquely afflicted; it is the
   same GDN-APC fp class, and the flywheel's fix shape (sequential recurrent scan, bit-exact-to-native;
   accept-only replay from h0) is the template.

---

## Part 1 — Artifact census (every known APC-artifact instance)

Cross-turn APC artifacts are defined by the **cold-prefix certificate**: parity is *restored* when the
prefix cache is reset per turn (`reset_prefix_cache`), so the divergence is intrinsic to *cache reuse*,
not to the denoise math. All instances share: `value_projection_events == 0`, `verify == ok`,
`eng_exact == hf_exact` (both non-exact — the fp flip never changes the exact_arguments verdict), and a
first divergence at a **model-chosen value/name near-tie token** (structural scaffold `<tool_call>` /
`<function=` / `<parameter=…>` / `>` always matches). Sources:
`runs/p2_engine_battery_v2/aggregate.json`, `runs/p2_engine_nevertrain/{aggregate.json,
nevertrain_turns.jsonl, nevertrain_parity_cert_resetapc.jsonl}`, `engine_build_status.md`.

### 1a. The true cross-turn APC class (parity RESTORED under cold prefix)

| turn | slice | pin | first_div (pos_mod32) | ENG tok @ fd | HF tok @ fd | near-tie | cold-prefix | note |
|---|---|---|---|---|---|---|---|---|
| gt20 | matched-20 | e5496cc (pre-S3) | 20 (20) | 72435 | 55061 | value near-tie | RESTORED | cleared path-robustly by Stage-3 |
| gt21 | matched-20 | e5496cc (pre-S3) | 19 (19) | 2877 | 2173 | value near-tie | RESTORED | cleared path-robustly by Stage-3 |
| gt60 | matched-20 | e5496cc (pre-S3) | 19 (19) | 29408 | 198 (`\n`) | ENG *wins* | RESTORED | fresh byte-matches HF's 169-tok output **incl. HF's mistake**; the "+1 exact win" was an APC artifact |
| gt16 | never-train | 95d8b47 (post-S3) | 15 (15) | 198 (`\n`) | 4794 (`_image`) | `\n`↔`_image` | RESTORED | **mirror** of gt130 |
| gt130 | never-train | 95d8b47 (post-S3) | 15 (15) | 4794 (`_image`) | 198 (`\n`) | `_image`↔`\n` | RESTORED | **mirror** of gt16 |

**The `{16,130}` mirror is the sharpest single piece of evidence.** Both turns share the exact context
`…<parameter=source>\ntest` and sit on the same near-tie between `\n`(198) and `_image`(4794). Under
APC-on the engine picks **opposite** directions on the two turns (gt16 → `\n`, gt130 → `_image`), and HF
picks the opposite of the engine each time; **both resolve to HF under cold prefix.** A single
deterministic denoise bug could not flip *both* ways — but a *prefix-dependent fp perturbation of the
reused GDN state* does: each turn arrives with a different upstream prefix, so the accumulated
chunk-order fp error is a different perturbation vector, sometimes nudging the near-tie one way,
sometimes the other. This is the signature of a cache-state fp difference, not a logic error.

### 1b. Control — the fp-residue class (breaks IDENTICALLY under cold prefix; **NOT** APC)

The same fp mechanism exists **path-invariantly** at the *start* of generation (block#0), independent of
cache reuse — these break byte-identically fresh-boot, so a lossless APC will **not** fix them (they are
an engine-vs-HF gap, a different target than cache-on-vs-fresh):

| turn | slice | first_div (pos_mod32) | ENG↔HF | class |
|---|---|---|---|---|
| gt44 | matched-20 | 16 (16) | 18061↔77 | block#0 GDN-fold fp gap; path-invariant |
| gt10,32,55,70,74,91,108,124,131,179 | never-train (10) | 4–27 | e.g. `cd`↔`cp`, `.txt`↔`\n`, `diff`↔`result`, `.`↔`..`, `Get`↔`Add` | deterministic bf16 GDN-fold fp-residue |
| gt36 | never-train | 27 | `diff`↔`result` | fp-residue, path-sensitive tail |

Decomposition (never-train, 13 breaks): **10 deterministic fp-residue + 2 APC cross-turn {16,130} + 1
path-sensitive tail; 0 structural.** Matched-20 also had `gt45` (a Stage-3-fixed bidir-alignment
regression, not APC). Exact_args is **APC-invariant** (83/184 either way; 130/247 aggregate == HF).

### 1c. Batched-bench correctness gate — same class, deterministic near-tie

`runs/p2_engine_batchgates` gate-1 invariance is **byte-identical 8/8** (batching is safe; no
cross-request contamination). The "gt130 companion-INVARIANT deterministic near-tie flip" cited in the
task is the never-train gt130 instance above — a companion(sequence)-invariant, deterministic flip whose
only trigger is cross-turn prefix reuse: exactly the APC class, reproduced under the batch harness.

### Census pattern (load-bearing for Part 2)

- **First-divergence is always a value/name near-tie**, mid-block (`pos_mod32 ∈ {15,16,19,20}`). Not
  because the boundary is at 15–20, but because that is where the first fp-flippable near-tie *lives*
  (structural tokens have huge margins and never flip; argument-value tokens are the near-ties).
- **`proj==0` and `verify==ok` on every instance** → the denoise/commit state machine is clean; the
  perturbation enters through the *reused state*, not the sampler.
- **Cold-prefix restores parity on every APC instance** → the divergence is in the *cached* GDN state,
  and the recurrent GDN state is the only lossy (compressed, chunk-order-dependent) thing being reused —
  KV is stored/reloaded exactly.

---

## Part 2 — Code trace: cached GDN state vs fresh recompute, and the first bit-divergent point

Pin: `shared/vllm_p2_pr42406` @ `0b44dcc` (branch `qwen3_5-flare-modelstate`). Deployed export runs
**canvas/commit block = 32, mamba_block_size (align-APC checkpoint stride) = 1024, GDN chunk = 64,
`--mamba-ssm-cache-dtype float32`**.

### 2.1 The two geometries that must agree for losslessness

- **align-APC checkpoint stride** = `mamba_block_size` = 1024 (64-aligned). This is what is *reused
  across turns*: on a cache hit vLLM reuses the mamba state blocks for the matched prefix and prefills
  only the unmatched suffix. Checkpoint column advance:
  `new_state_idx = ceil(computed_after / MAMBA_BLOCK_SIZE) − 1` —
  `vllm/v1/worker/mamba_utils.py:302`.
- **GDN chunk** `_GDN_CHUNK = 64` — `qwen3_5_flare.py:115`. The prefill scan
  (`chunk_gated_delta_rule`) folds a sequence in 64-token chunks with cross-chunk state carry
  (`chunk_size = 64`, confirmed `qwen_gdn_linear_attn.py:1086`).
- **commit/canvas block** = 32 — `qwen3_5_flare.py:242-262`. Because `32 % 64 != 0`, the constructor
  **logs a hazard**: *"commit boundaries will land mid-chunk and the fp32 boundary snapshot will not be
  a clean recurrent checkpoint"* (`qwen3_5_flare.py:248-261`). **This is the deployed config**: block=32
  is required to match the HF reference's 32-token block structure (`block_commit_target` docstring,
  `qwen3_5_flare_ops.py:189-196`), yet 32 is HALF a GDN chunk.

### 2.2 How the CACHED state is produced (prior-turn commit path)

Per the FLARE header (`qwen3_5_flare.py:13-16`): *"commit → one clean causal pass … fp32 boundary
snapshot + raw conv tail published into the align-APC block-aligned row."* Concretely:

- Denoise forwards do **not** advance the GDN state — the read-only-denoise snapshot/restore holds the
  conv+ssm rows bit-fixed (`snapshot_readonly_rows`/`restore_readonly_rows`,
  `qwen3_5_flare_ops.py:321-352`; restore is the final authority in `postprocess_state`,
  `qwen3_5_flare.py:975-979`; scope widened to the align running-region in **af21dc8**).
- Only a **commit** advances state, and it commits on **32-absolute** boundaries: the first generated
  block commits `block_size − prompt_len % block_size` tokens, every later block a full 32
  (`block_commit_target`, `qwen3_5_flare_ops.py:181-214`). So the generated region is folded as a
  chain of ~32-token chunk-kernel calls, each starting from the previous boundary state.
- The align postprocess then checkpoints that running state into the 1024-aligned column
  (`run_fused_postprocess_align`, `mamba_hybrid.py:290-329`), kept fp32.

### 2.3 How the FRESH state is produced (cold-prefix recompute) — and the first divergent bit

On a cold prefix the same tokens are re-run as ordinary **prefill** through the one chunked scan:

```
qwen_gdn_linear_attn.py:1513   initial_state = ssm_state[prefill_state_indices]     # cache slot (or 0 if fresh)
qwen_gdn_linear_attn.py:1514   initial_state[~prefill_has_initial_state, ...] = 0
qwen_gdn_linear_attn.py:1518   core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
qwen_gdn_linear_attn.py:1524       initial_state=initial_state, output_final_state=True,
qwen_gdn_linear_attn.py:1526       cu_seqlens=attn_metadata.prefill_query_start_loc,
qwen_gdn_linear_attn.py:1527       chunk_indices=attn_metadata.chunk_indices,
qwen_gdn_linear_attn.py:1528       chunk_offsets=attn_metadata.chunk_offsets, ... )
qwen_gdn_linear_attn.py:1532   ssm_state[prefill_state_indices] = last_recurrent_state.to(ssm_state.dtype)  # the DURABLE state written to cache
```

Both paths reach the **same `chunk_gated_delta_rule` op** (line 1518), but with **different
`cu_seqlens` / `chunk_indices` / `chunk_offsets`** (lines 1526-1528) and a **different `initial_state`**
(line 1524):

- **Fresh:** one call over the whole prefix, chunk grid anchored at absolute position 0 → 64-token
  groups `[0,64,128,…,1024,…]`. Every 1024 checkpoint sits on a clean 64-boundary and is the exact
  cross-chunk reduction of the 64-grid.
- **Cache-reuse across a prior-turn generated region:** the state was folded as many `cu_seqlens=[0,32]`
  (or variable draft-width) calls landing on **32-absolute** boundaries — half of which bisect a
  64-chunk. Restarting the associative reduction at a 32-boundary is a **different grouping** of the
  same rank-1 updates. GDN's chunk scan is fp-**non-associative** across chunk boundaries, so a
  different grouping yields a different `chunk_states` cross-chunk reduction → **different bits** in
  `last_recurrent_state`.

**First point the two paths can produce different bits:** the cross-chunk state carry inside the GDN
chunk scan (`chunk_gated_delta_rule`, dispatched at `qwen_gdn_linear_attn.py:1518`; the `chunk_states[:,
-1]` boundary reduction the FLARE header and `qwen3_5_flare_ops.py:360-382` snapshot) **at the first
64-chunk boundary that a 32-absolute commit boundary bisects.** The divergence becomes durable the
moment it is written back to the cache at `qwen_gdn_linear_attn.py:1532`, and is then reused verbatim on
the next turn's cache hit (`initial_state = ssm_state[prefill_state_indices]`, line 1513). It surfaces
as a ~fp-epsilon shift in every downstream denoise logit, flipping the first value/name near-tie
(Part 1). This is the same fp mechanism the v3b/never-train reports name for the *fresh* block#0 fold
(*"HF folds 32 incl. `prompt%32` leftover; the aligned engine folds `32 − L%32` gen tokens from the L
checkpoint — fp-close, not bit-identical"*), here expressed **across turns** through the cache.

### 2.4 Why attention is exact but GDN is not

Attention KV is stored per-position (the exact K,V of each cached token) and re-read by the same kernel,
so a reused prefix's attention contribution is bit-identical — reuse is exact. The GDN state is a
**fixed-size recurrent summary** whose bits depend on the *order* the tokens were folded; the cache
stores the *summary*, not the tokens, so a summary folded on 32-absolute commit boundaries cannot equal
one folded on the fresh 64-grid. Losslessness therefore hinges entirely on the GDN state boundary/kernel
path.

---

## Part 3 — The flywheel native+MTP AR APC as the reference design (the precise comparison)

Repo: `shared/lumoFlyWheel_codex_fork`. The FR13 line of work is a *deep, adversarially-verified*
investigation of exactly this question for the **AR native + MTP (tree-spec)** GDN-APC path. Key docs:
`FR13_KERNEL_STATUS.md`, `FR13_SHARP_LOCALIZE_BIND.md`, `FR13_BRANCH_FLIP_LOCALIZED_BIND.md`,
`scripts/fr13_apc_hit_first_divergence.py`.

**Is the flywheel AR APC lossless, or lossy-but-unmeasured? — LOSSY, and eventually measured.**

1. **It is not bit-lossless.** FR13 found a **22/482 = 4.6% per-token argmax "lossless drift"** on the
   APC/MTP path (`FR13_SHARP_LOCALIZE_BIND.md:39`). The prior **scalar accept-rate gate was blind** to
   it (accept 3.198 > native; every scalar passes) — the "gate blindspot" (`FR13_GATE_BLINDSPOT`). The
   **only** instrument that catches it is a **per-token argmax-vs-clean-oracle** teacher-forced probe.
   This directly validates the diffusion project's own discipline of gating on **raw byte-parity**, not
   exact_args (which is APC-invariant here and would have hidden the entire artifact class).

2. **What makes the state they DO publish bit-exact — and the direct lesson for the diffusion fix.** The
   shipped GDN verify is a **sequential rank-1 per-ancestor recurrent scan** that is
   *"bit-exact-to-native-on-spine, batch-invariant by construction"* (`_tree_gdn_kernel`,
   `FR13_KERNEL_STATUS.md:6-11`), plus an **accept-only replay from h0** that re-executes only the
   accepted chain and publishes its durable state (`FR13_KERNEL_STATUS.md:13-24`). Crucially, the
   **chunked-WY GDN kernel is PARKED precisely because chunked math is a *different summation tree →
   ~6e-5 chunk-vs-recurrent gap, never bit-exact to native*** (`FR13_KERNEL_STATUS.md:33-34`). **That is
   the diffusion engine's exact situation**: the diffusion APC caches states from the *chunked*
   `chunk_gated_delta_rule` (and folds them on mid-64 32-boundaries) — the very kernel class the flywheel
   rejected for durable state. The AR reference's answer is: *publish/verify durable state on a
   sequential-recurrent path, not a chunked one.*

3. **The AR carrier is (partly) a DIFFERENT seam — a sharpening for the diffusion attribution.** The
   APC-hit first-divergence localizer pins the **prime suspect at the RESTORED CONV-state prior-window**
   (the K−1 conv window read back by physical block_id before `causal_conv1d` overwrites it), while the
   **restored ssm recurrent state is 47/48 faithful ("mostly clean")** —
   `scripts/fr13_apc_hit_first_divergence.py:6-19`. A second carrier is **co-residency batch-variance**
   (a tile-config / reduction-order artifact on custom ops **not covered by
   `enable_batch_invariant_mode`** — fp8 GEMM, TREE_ATTN — `FR13_BRANCH_FLIP_LOCALIZED_BIND.md`). The
   diffusion evidence lumps everything into "GDN-fold fp gap" and implicitly attributes it to the **ssm
   recurrent** state; the flywheel says **isolate the conv-tail restore separately**, because in the AR
   case *that* was the prime mover and ssm was mostly clean. The diffusion engine's read-only-denoise
   snapshots **both** conv and ssm rows (`qwen3_5_flare_ops.py:333-337`) and the commit publishes a
   **raw conv tail** (`tail_after_append`, `qwen3_5_flare_ops.py:393-410`), so the conv path is an
   untested-in-isolation suspect on the diffusion side too.

4. **Measurement-discipline caveat (hardware-dependent).** On the flywheel's **GB10**, fresh boots at
   B=1 **fork from any reference at tokens 11–71** due to boot-level autotune/kernel-selection, so
   "byte-identical to fresh" is only well-defined **same-boot, in-process** (`FR13_SHARP_LOCALIZE_BIND.md`:
   the cross-boot byte-gate is invalid). The diffusion runs are on **RTX 5090** and *do* reproduce
   byte-identically across boots (v3b temp-0.7 a==b across 2 boots; greedy deterministic), so the
   diffusion cold-prefix certificate is a *valid* fresh reference there — but any future GB10/Spark
   deployment must switch to the same-boot in-process gate before trusting a losslessness claim.

---

## Verdict — does the canonical-boundary hypothesis survive?

**YES — it survives and is corroborated, with two refinements the flywheel makes precise.**

- **Confirmed:** the artifact class is a **GDN recurrent-state chunk/reduction-order fp divergence**
  under cache reuse; attention KV reuse is exact; the first bit-divergent point is the GDN chunk scan's
  cross-chunk carry (`qwen_gdn_linear_attn.py:1518`, durable at `:1532`) whenever the commit path's
  **32-absolute** fold boundaries (`qwen3_5_flare_ops.py:181-214`) bisect the **64-chunk** grid
  (`_GDN_CHUNK=64`, hazard-logged at `qwen3_5_flare.py:248-261`) a fresh prefill uses. The
  cold-prefix-restored `{16,130}` mirror and the `proj==0/verify==ok` census are exactly this signature.

- **Refinement 1 (fix must be stronger than "chunk-aligned").** "Cache only chunk-aligned states" is
  necessary but not sufficient: the cached/published state must come from a **kernel path bit-exact to
  the fresh recurrent recompute**. A chunked GDN kernel is *inherently* not bit-exact (flywheel:
  ~6e-5 chunk-vs-recurrent, WY-parked). So bitwise losslessness by construction requires either
  (a) publishing the durable checkpoint via a **sequential-recurrent fold on the canonical 64-grid
  anchored at absolute 0** (equivalently: make `block_size` a multiple of `_GDN_CHUNK` so commits never
  bisect a chunk — the code's own `_DEFAULT_BLOCK=64` — *and* recompute the intra-checkpoint tail on the
  same grid on reuse), or (b) adopting the flywheel's **accept-only replay from h0 on a
  bit-exact-to-native recurrent scan**. Reuse only at 1024-aligned (already 64-aligned) checkpoints is
  the safe read side.

- **Refinement 2 (attribute conv vs ssm before building the fix).** The flywheel's APC-hit localizer
  found the **conv prior-window restore**, not the ssm state, to be the prime AR carrier (ssm 47/48
  clean). The diffusion attribution should be split the same way — an isolation probe on the diffusion
  side (conv-tail-restore-only vs ssm-restore-only across a `{16,130}`-style turn) before committing to
  an ssm-only canonical-boundary fix.

**Not in scope of a lossless APC:** the `gt44` / 10-turn fp-residue class breaks fresh too (Part 1b) —
that is an engine-vs-HF gap, not cache-on-vs-fresh. The user's target ("cache-on byte-identical to
fresh") is the **weaker, achievable** self-consistency goal, and the canonical-boundary/recurrent-fold
fix targets it exactly; matching HF bit-for-bit is a separate, deferred kernel-level goal.

---

## Evidence index

- Census: `runs/p2_engine_battery_v2/aggregate.json` (parity_breaks {20,21,44,45,60});
  `runs/p2_engine_nevertrain/aggregate.json` (13 breaks + decomposition),
  `nevertrain_parity_cert_resetapc.jsonl` (cold-prefix certificate, {16,130} restored),
  `nevertrain_turns.jsonl` (APC-on first_div); `runs/p2_engine_batchgates/` (gate-1 invariance).
  Narrative: `p2_engine_battery_v3b_result.md`, `p2_engine_nevertrain/report.md`,
  `engine_build_status.md`, `goal_5x_rollout_b1.md:110-118`.
- Engine (pin `0b44dcc`): `qwen3_5_flare.py` (:13-16 commit publish, :115/:242-262 block/chunk hazard,
  :906-990 pre/postprocess + read-only restore), `qwen3_5_flare_ops.py` (:181-214 block_commit_target,
  :321-352 snapshot/restore, :360-410 fp32 boundary + raw conv tail), `mamba_hybrid.py:166-329`
  (align pre/postcopy), `mamba_utils.py:263-306` (checkpoint stride), `qwen_gdn_linear_attn.py:1086`
  (chunk_size 64), `:1504-1558` (prefill-chunk vs decode-recurrent dispatch; **:1518 first-divergent
  op, :1532 durable write**). Snapshot-scope commit: **af21dc8**.
- Reference (flywheel): `FR13_KERNEL_STATUS.md` (sequential-recurrent verify bit-exact; chunked-WY
  parked, ~6e-5 gap; accept-only replay), `FR13_SHARP_LOCALIZE_BIND.md` (22/482 drift, gate blindspot,
  same-boot gate), `FR13_BRANCH_FLIP_LOCALIZED_BIND.md` (co-residency batch-variance; non-BI custom
  ops), `scripts/fr13_apc_hit_first_divergence.py` (conv prior-window = prime suspect; ssm 47/48 clean).
