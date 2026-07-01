# FLARE fast-serving foundation — architecture + build plan (design workflow wqav45xfp, 2026-06-30)

> Decision-grade output of an 11-agent design workflow (3 designs x 4 adversarial lenses + synthesis), code-verified on the box. Monitor-red-teamed. The SERVING-BASE recommendation (HF own-forward vs SGLang) reverses the prior SGLang greenlight -- escalated to user.

Both load-bearing facts confirmed. The F1 correction is real and decisive: modeling.py:2197 shows the noisy stream attends `torch.cat([clean_norm.repeat(2,1,1), noisy_norm])` — clean-stream KV, and GDN is seeded from `clean_gdn_docwise_with_boundaries` (causal clean scan, line 2160-2182). The boundary carriers are **clean-stream causal quantities**, so `advance()` must be a separate clean-causal forward, not a reuse of the noisy block. And the regime is settled: `gdn_mode` is read only in `_linear_attn_mdm` (line 1489, reachable only via the non-FLARE `mdm_split_size` path); the trained checkpoint ran `FASTDLLM_FLARE_TWO_STREAM=1` + `route_i`. Synthesizing now.

---

# FLARE Fast-Serving Foundation — Recommended Architecture + Staged Build Plan

**Scope:** the serving/caching foundation to build in `flare` (single mutator) BEFORE RL rollouts. Decision-grade. Grounded in the three designs, four red-team lenses, and two facts re-verified on the box just now.

---

## 0. Two facts that settle the design space (verified, not assumed)

1. **The trained target is route_i FLARE two-stream — the config `gdn_mode="option_a_causal_gdn_v0"` is a DEAD FIELD for this path.** `gdn_mode` is read only in `_linear_attn_mdm` (`modeling.py:1489-1504`), reachable only when `mdm_split_size is not None` (the non-FLARE MDM path, `2464-2549`). The FLARE forward (`_flare_two_stream_layer_forward:2142`) dispatches on `flare_gdn_route()` and ignores `gdn_mode`. The shipped adapter (`runs/flare_stage1_ab_pilot/two_stream_B_...`) was trained with `FASTDLLM_FLARE_TWO_STREAM=1` + `FASTDLLM_FLARE_GDN_ROUTE=route_i` (`run_flare_stage1_ab_pilot_job.sh:60`, `run_flare_stage1_ab_optimized_pair.sh:25`). **Decision: pin the served forward to route_i FLARE two-stream and treat any route/regime change as a cache-invalidating retrain event.** Several red-teams worried the config default contradicts the designs; it does not — the field is inert on this path.

2. **The boundary carriers are CLEAN-stream (causal) quantities, not noisy-block quantities.** Confirmed at `modeling.py:2160-2182` (GDN seeded from `clean_gdn_docwise_with_boundaries` + `clean_boundary_states`/`clean_raw_qkv`) and `2197` (noisy attention reads `clean_norm` KV, not committed-noisy KV; `clean_mask = doc_causal_bool_mask`, `2266`). **This kills the single most common bug across all three designs** ("on commit, append the now-clean block's KV, no recompute" — Design 1 §4; "final_state becomes new S" — Design 3 §6). Both reuse the *noisy, bidirectionally-contaminated* block output where the *clean causal* forward is required. The recommended architecture makes `advance()` an explicit incremental **clean-causal** forward.

---

## 1. Recommended architecture — graft the best pieces

**Verdict: build Design 1 / Design 3-Phase-A (the HF `.venv-fastdllm` seeded-block path), corrected by the F1 clean-stream fix, verified by a NEW full-stack instrument, with Design 2's exact-re-score as the RL parity spine (not its vLLM engine).** Do NOT port the flywheel CUDA or fork vLLM for the foundation (§4).

### The served forward (per denoise step, per request)

For the single in-flight block `b` at `block_start = b·32`, run the **route_i FLARE noisy forward restricted to the 32-token block**, seeded from a per-request pinned **clean-stream** checkpoint:

- **24 GDN layers:** `run_gdn_manual_route_i(gdn_layer, block_hidden, chunk_size=32, initial_state=S_cache[layer], conv_tail=tail_cache[layer])` (`modeling.py:787`). O(32), independent of prefix length. Final state **discarded** every step (block still noisy).
- **8 attention layers:** block Q (≤32 rows) attends `[clean-prefix K,V] ++ [block K,V]` under `flare_two_stream_bool_mask` restricted to one block = all-ones over clean prefix ⊕ 32×32 bidirectional block. One SDPA call. The clean-prefix KV is the **clean-stream causal** KV (from `clean_norm`), cached.
- **Head:** `lm_head(model.norm(noisy_hidden))` per position — **no `+1` right-shift** (the shift at `eval:2416` is a causal-AR artifact; the FLARE mask token at position `i` predicts `i` directly — `block_diff_mask:88` confirms within-block bidirectionality).

### The cache lifecycle — one genuinely new piece (`RequestDiffusionState`)

```
RequestDiffusionState (per request, pinned — NOT a cross-request content-hash pool):
  block_start : int
  S_cache     : list[24] [1,32,128,128] fp32     # GDN recurrent state (clean boundary)
  tail_cache  : list[24] [1,3,8192]              # GDN conv tail = RAW in_proj_qkv (pre-conv)
  kv_cache    : list[8]  (K,V) [1,4,block_start,256]  # CLEAN-STREAM causal prefix KV
```
- **reset()** — zero state, empty tail/KV; warm by one clean-causal scan over the prompt.
- **read** (every denoise step) — pure reads; seed the block forward above.
- **advance()** on block commit — **THE F1 FIX: run one incremental CLEAN-CAUSAL forward over the just-committed 32 tokens** (GDN `clean_gdn_docwise_with_boundaries`-equivalent single block with `output_chunk_states=True` to capture the boundary `S` + raw conv tail; attention clean-causal K,V projected from clean-stream `clean_norm` and appended). `block_start += 32`. **Never** roll the boundary from the noisy block's `final_state`/KV.
- **free()** on request end.

### Reuse ledger — flywheel FR13 APC

| From flywheel FR13 | Status | Why |
|---|---|---|
| **Checkpoint object** (fp32 `S` + last-`conv_kernel-1` RAW pre-conv tail) | **Reuse verbatim (concept)** | Identical carrier; `validate_gdn_state_snapshot.py` already models it |
| **R3 fp32 SSM cache** discipline | **Reuse verbatim** | `torch_chunk_gated_delta_rule` returns fp32 (`661-676`); cache as-is, never bf16 |
| **Conv-tail = RAW pre-conv discipline** (FR12) | **Reuse verbatim** | `gdn_project_and_conv:760` prepends raw `in_proj_qkv` tail; `zero_first_lags` check guards it |
| **Verification methodology** (matched-config cache-OFF A/B; per-token argmax not scalar norms; 5 failure modes: live-not-replay, `.detach().cpu().clone()`, file-based save, read the RESTORE frame) | **Reuse verbatim** | The proof style transfers even though the CUDA does not |
| **R1/R2/R4 invariants** | **Adapt** — hold by *construction* here (block==chunk==32, one torch realization, snapshot at exactly one boundary) BUT must be *engineered/asserted*, not assumed free (see risks) | AR earned these; we inherit them only if kernel flags + doc-anchored grid are pinned |
| **`_patch_gdn_linear` store/restore, EXACT_SEED chain** | **Adapt the pattern, not the code** | We use a per-request Python dict, not vLLM worker-state patching |
| **Tree kernels** (`fr10_gdn_tree_kernel`, `fr13_tree_conv_fused`, `fr13_replay_conv_remap`, node-bank, `SNAP_FIX`, LCP, rejection sampler) | **DROP entirely** | AR spec-decode tree infra; diffusion commits a whole block per schedule → no accept/reject tree → the entire IN-FLIGHT staleness class does not exist |
| **Forked-vLLM delivery vehicle** (0.19.0 patcher, launcher) | **DROP for foundation** | Our box has vLLM 0.23.0 stock; anchors won't match (§4) |

**From Design 2, keep exactly one thing: the exact-re-score parity spine (P2).** Not its engine. RL log-probs come from OUR training forward re-score, promoted to "trust the served forward" (P1) only per-layer once bit-exactness is proven (§3 contract).

---

## 2. Staged build plan (each stage gated on the prior)

### Stage 0 — Prerequisite decision A/B (cheap, decisive, do FIRST)
Before any caching work, answer "is FLARE serving even the right target for this checkpoint?"
- **FLARE-vs-causal task-accuracy A/B** on banked tool-call cases: serve today's causal `+1`-shift path vs the FLARE no-shift path (both cache-OFF) on the trained adapter. If FLARE-serving is not ≥ causal-serving on task score, the checkpoint isn't FLARE-serving-ready and this is a **train-side blocker** no cache can fix. (The adapter *was* trained two-stream/route_i, so this should pass — but it is the cheapest possible falsifier and must gate the rest.)

### Stage 1 — LOSSLESS FOUNDATION FIRST (no new kernel, no vLLM, HF path)
1. Add a serving method to `Fast_dLLM_Qwen3_5Model`: single-block route_i FLARE noisy forward from a passed-in checkpoint (GDN seeded scan + prefix-KV/32×32 SDPA), returning **un-shifted** per-position logits.
2. Wire `RequestDiffusionState` into `full_context_sample` (`eval:1975`): replace `model(input_ids=x_t, use_cache=False)` at `eval:2415` **and the two candidate-scoring recomputes at `eval:2168` and `eval:2302`** with the cached block forward. **Assert 0 residual full-context `model()` calls per committed block** (else speedup caps at ~3× — red-team F2/X5). Delete the `+1` shift (`eval:2416`).
3. Implement `advance()` as the **clean-causal** incremental forward (F1 fix). Hook at block-fill loop exit (~`eval:2067`).

### Stage 2 — BIT-EXACT + PARITY VERIFICATION PROTOCOL (the gate; must BUILD a new instrument)
**`validate_gdn_state_snapshot.py` as-is is INSUFFICIENT** — it is single-layer, random-weight, isolated, `atol=1e-3`, and **stream-blind** (feeds the same `hidden_states` to prefix and block; cannot see the clean/noisy distinction → cannot catch F1). Keep it as a per-layer unit check but **build the decisive instrument:**

**T1 — Full-stack real-weight multi-block A/B (the decisive test).** Reference = the actual `_flare_two_stream_training_forward` (clean+noisy) on the trained weights. Over ≥8 sequential block commits, diff **per-layer `S`, `conv_tail`, and clean-prefix KV** cache-ON (served `advance()`) vs cache-OFF (recompute clean boundaries from doc start). Pass = fp32 round-off AND **boundary-position-independent** (sweep where the boundary sits; diff must not grow — structural, not dilutional). *This is the only test that catches the F1 clean-stream advance bug and the R2 mis-seed.*

**T2 — Serving-vs-training-forward logit/logprob parity (real weights).** Teacher-force the identical `(committed prefix, block, mask realization t)` through (i) the cached served forward and (ii) `_flare_two_stream_training_forward` under the trained env flags. Diff per-position **logits and argmax** — never scalar norms; live, not replay; `.detach().cpu().clone()`; read the RESTORE frame. Argmax must never flip.

**T3 — Byte-identical trajectory canary (the astropy-12907 analog).** Greedy, temp=0, commit-one (lossy dial pinned lossless), grammar in the loop: cache-ON must emit a **byte-identical committed token stream + tool-call JSON** vs cache-OFF on a banked case. If not byte-identical, **downgrade the claim from "bit-exact" to "tight-tol lossless" honestly** (bf16 attention-KV reuse is ~1 ULP ≈ 0.008; near-tie argmax can flip).

**T4 — Negative controls (prove the instrument has teeth):** (a) deliberately mis-align the boundary (seed at `prompt_len` when `prompt_len % 32 ≠ 0`) → T1 diff MUST grow; (b) torch-vs-FLA kernel A/B → MUST diverge if flags differ; (c) feed a known non-symmetric `S` through the k-major/v-major layout → catches silent transpose (only relevant if Phase B ever touches the flywheel bank).

**Gate: T1+T2+T3 green on the trained adapter before RL wiring.**

### Stage 3 — RL-ROLLOUT LOG-PROB-CONSISTENCY INTEGRATION
- Rollout samples from the cached served forward; **log the per-block mask realization `t`** (which positions were masked at the step each token committed) so the diffu-GRPO re-score can replay it exactly.
- Ship **P2 (exact re-score)**: policy-gradient log-probs (old θ_old detached + new θ) both computed by the parity-exact FLARE forward, re-applying the **grammar FSM mask** (part of π — `eval:488/560`) and the **exact per-step commit-set**. Promote per-layer to **P1** (trust served logits directly) only as each layer passes T2.
- **Verify on the NF4 path, not bf16** (red-team X3): re-run T2 with rollout batch shape `[G,L]` vs re-score `[1,L]` on the 4-bit adapter; bitsandbytes NF4 matmul is not guaranteed batch-shape-invariant. If it fails, force identical batch shape/kernel between sample and score (or de-quantize for scoring).

### Stage 4 — LOSSY-KNOB PARETO SWEEP (only after Stage 2-3 green, cache held exact)
- The confidence-threshold parallel commit (`eval:2476`) is the ONLY lossy dial. Pin `threshold→1.0`/argmax-only/temp=0 for the foundation; sweep down afterward.
- **Critical coupling the designs get wrong (red-team X2): the lossy dial is separable from the CACHE but NOT from the LOG-PROB ESTIMATOR.** Committing K positions in one step means each token's training log-prob is a **mean-field conditional over still-masked siblings**. The re-score MUST replay the exact per-step commit-set (a naive full-block re-score is wrong under parallel commit). Sweep the payoff as a **3-D quantity: speed × task-accuracy × gradient-bias**, never as "fewer steps, cache untouched."

### Stage 5 — IN-HOUSE INCREMENTAL-GDN KERNEL (behind a hard scoping gate; likely DEFERRED)
See §5. Do not build on the current box.

---

## 3. The explicit train-serve-parity contract

RL is unbiased **iff** the token that gets sampled and the log-prob that gets differentiated pass through **ONE realization**. The contract, all clauses must hold:

| # | Clause | Enforcement |
|---|---|---|
| C1 | **Forward semantics** = route_i FLARE two-stream noisy forward; **no `+1` shift**; bidirectional-within-block attention; GDN clean-boundary-seeded | Hard guard at serve init: assert `flare_gdn_route()=="route_i"` and two-stream enabled; any route change invalidates the cache |
| C2 | **Boundary carriers are CLEAN-stream** (causal); `advance()` is a separate clean-causal forward | T1 |
| C3 | **One GDN kernel realization** — and replicate training's *internal* torch-boundary/FLA-block split exactly. `torch_chunk_gated_delta_rule:450` forces torch when `output_chunk_states=True` (boundary capture) and may use FLA when False (block scan). "Pin one kernel" is impossible verbatim; the contract is "replicate the exact split." | T4(b); dtype assert on cache tensors |
| C4 | `chunk_size == block_size == bd_size == 32`; boundary lands on a chunk boundary via **doc-anchored** grid (not prompt-anchored) | T4(a) |
| C5 | **fp32** recurrent state + conv cache; **raw pre-conv** conv tail | T1; `zero_first_lags` |
| C6 | q/k `l2norm` + `scale=1/√128`; GVA `repeat_interleave×2` (32 v / 16 k); gated RMSNorm fp32 `weight·norm·silu(z)` (`252`); partial-rope 0.25 / θ=1e7 interleaved; **per-doc local positions** | T2 |
| C7 | **Grammar FSM mask is part of π** — same processor object in serve AND re-score; assert identical `allowed_token_ids` per position before comparing log-probs | Shared module imported by both paths |
| C8 | **Same NF4 quant path + batch shape** between sample and re-score | Stage-3 NF4 T2 |
| C9 | Under parallel commit, re-score replays the **exact per-step commit-set** (mean-field conditional), not the filled block | Stage-4 keyed diff |

---

## 4. Decision: fastdllm HF path vs flywheel-forked-vLLM

**Build the foundation on the fastdllm HF path (`.venv-fastdllm`). Defer forked-vLLM until concurrency exists and its load gate is separately cleared.**

Rationale (feasibility red-team, verified on box):
- **The proven flywheel APC does NOT transfer to this box.** It lives in a container on **vLLM 0.19.0** on the **GB10 (117 GB unified)** serving an **fp8 27B**. Our box is a single **RTX 5090 sm_120, 32 GB**, and our local vLLM is **0.23.0 STOCK** — no EXACT_SEED/tree_gdn markers, and vLLM renamed `gdn_linear_attn.py`→`gdn_attention.py` between 0.19→0.23, so the flywheel's anchor-unique string-replace patcher will miss. The "GDN patches apply unchanged" claim is false here.
- **Missing kernel deps:** `flash_attn`, `causal_conv1d`, `fla` are all **absent** from `.venv-vllm`; building them for sm_120 + cu130 + torch 2.11 is a Blackwell-wheel gamble.
- **Custom arch doesn't load in vLLM** (`auto_map` `Fast_dLLM_Qwen3_5ForCausalLM`) — needs a `FastDLLMQwen3Next` registration + weight-remap loader + QLoRA→bf16 merge + the **entire net-new bidirectional diffusion forward** on the spec-decode path. Weeks, high risk, and exactly the code with no proven analog.
- **The payoff regime (batched/paged throughput, CUDA graphs) is what a single-5090 batch-1 rollout loop cannot yet exploit** — and the on-box FLA measurement shows fused GDN kernels *lose* to torch at batch-1 anyway (`fla_kernel_feasibility.md`).
- The HF path: stays in the working venv, no new kernel, **~100-150 MB per-request cache** (fits with room to spare), uses `run_gdn_manual_route_i` which already exists and is already partially validated. It directly attacks the real pain: the current no-cache path **cannot finish one sample in 240 s** (`phaseA_retention_snapshot_...note.md`).

**Forked-vLLM is a legitimate throughput follow-up** once batched RL concurrency is real; when that day comes, its load gate (patcher re-authored against 0.23.0, deps built for sm_120, remap loader, stock-causal-logits-match-HF milestone) is a separate project, and its exact-re-score spine (which we're already adopting) means it only ever produces *samples* — the parity log-probs stay on the HF forward regardless.

---

## 5. Honest risks + the kernel decision gate

**Correctness risks (must be closed before the "lossless" claim):**
- **R-F1 (highest): clean-stream advance.** Closed by the C2 clean-causal `advance()` + T1. Every design shipped this bug; do not.
- **R-overclaim: "bit-exact by construction" is asserted, not proven.** The only existing instrument is single-layer/random/1e-3/stream-blind. Until T1+T2+T3 exist and pass on real weights, the honest claim is "tight-tol lossless," not "byte-identical."
- **R-realization-split (C3):** the torch-boundary/FLA-block seam inside training means a naive "one kernel everywhere" serving path silently drifts. Replicate the split; T4(b).
- **R-parallel-commit-factorization (X2):** lossy dial couples to the log-prob estimator, not the cache. Handle in Stage 4.
- **R-NF4 nondeterminism (X3):** all losslessness proofs are on unquantized weights; RL runs NF4. Re-verify on the 4-bit path (Stage 3).
- **R-regime-drift (F5):** losslessness is a property of **route_i** (causal-seeded GDN), not the architecture. route_ii (256-token window) and dualpass (genuinely bidirectional GDN) break the per-block cache. C1 hard-guard.

**Speed risk (the honest one):** the O(prefix)→O(32) FLOP win is real only for **long agentic prefixes**. At short prefixes the 32-layer Python per-step orchestration (24 seeded scans + 8 SDPA) can eat the saving at batch-1 (65% util, host-launch-bound per the FLA note). **Measure the crossover prefix length** (256/1k/4k) with a cached-vs-uncached-FLARE wall-clock, same forward, cache the only variable. The RL target regime (long prefixes) is where it pays; confirm it.

**Kernel decision gate (Stage 5 — the in-house incremental-GDN kernel):**
> Build the flywheel-style recurrent `_tree_gdn_kernel` sub-block kernel ONLY if, at the serving batch size RL actually uses, a micro-benchmark shows the recurrent/fused scan **beats** `run_gdn_manual_route_i(chunk_size=32)` per 32-block, AND the parity cost (it is a **different fp realization**, ~0.0078 gap ≫ 1 bf16 ULP → requires a **retrain-to-recurrent**) is justified by that measured win.

On the **current box this gate FAILS**: the on-box FLA benchmark already shows fused/recurrent GDN is **slower and higher-memory than torch at batch-1** (5.78 vs 4.76 s/step), the fast FlashQLA-class kernels are **SM90/SM100 only (exclude sm_120)**, and it demands a retrain for a negative payoff. **Phase B is parked — revisit only if/when batched concurrency makes the batch-1 verdict obsolete** (which is also exactly when forked-vLLM starts to pay off). Do not retrain for a kernel that loses at the batch you serve.

---

## 6. One-paragraph bottom line

Build the **HF-path seeded-block route_i FLARE serving foundation** (`RequestDiffusionState` + seeded `run_gdn_manual_route_i` + clean-prefix-KV SDPA, no `+1` shift, all three recompute call-sites rerouted), with the **clean-stream `advance()`** as the load-bearing correction to every prior design. Reuse the flywheel **checkpoint object, fp32/raw-conv discipline, and verification methodology verbatim**; drop its tree CUDA and its vLLM fork. Gate "lossless" on a **new full-stack real-weight multi-block A/B (T1) + serving-vs-training-forward parity (T2) + byte-identical canary (T3)** — the existing single-layer validator cannot see the F1/parity defects. Ship diffu-GRPO on an **exact-re-score parity spine** verified on the **NF4** path with the mean-field commit-set replayed. Pin the lossy dial lossless for the foundation and sweep it later as a 3-D (speed × accuracy × gradient-bias) Pareto. Keep the in-house recurrent kernel and forked-vLLM strictly behind measurement gates that the current single-5090 batch-1 box does not pass.

Key files: `/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-init/modeling.py` (`:2142` layer forward, `:2160-2182` clean-seeded GDN, `:2197` clean-KV read, `:2266` clean_mask, `:787` `run_gdn_manual_route_i`, `:450` torch/FLA seam, `:1489` dead gdn_mode path), `/home/mark/qwen_diffusion/scripts/eval_fastdllm_toolcall_cases.py` (`:2168/:2302/:2415` three recomputes, `:2416` shift, `:2476` commit), `/home/mark/qwen_diffusion/scripts/validate_gdn_state_snapshot.py` (insufficient as-is), `/home/mark/qwen_diffusion/fla_kernel_feasibility.md` (batch-1 kernel loss, sm_120), `/home/mark/qwen_diffusion/machine_notes.md` (flywheel = vLLM 0.19.0 container on GB10).