# P2 Engine-Fast Diffusion Serving — Reuse Plan (workflow wc2mb4o4p, 2026-07-02)

16 agents, 0 errors, top candidates verified. User directive: REUSE (flywheel or external), build on top.

# P2 Decision Plan: Engine-Fast Diffusion Serving (flip the 1.29x deficit)

**Decision:** Adopt the **vLLM native dLLM path (Model Runner V2 `ModelState`, DiffusionGemma blueprint) on vllm main pinned at/after PR #42406 (2026-06-29)**, composed with our in-house FR13 align-mode APC discipline and `flare_hf_cache.py` as the port spec. This is a re-pin of the Lumo_FlyWheel fork target from 0.23 → post-v0.24.0 main, justified below; it is the maximum-reuse route, and it is reuse of *both* external engine work and our own flywheel/FR13 assets.

---

## 1. THE RECOMMENDED PATH

**Host:** Fork `vllm-project/vllm` main pinned at/just after the merge commit of PR #42406 (MRV2 align-mode APC for mamba, merged 2026-06-29, ~90 min after the v0.24.0 tag), branched inside the Lumo_FlyWheel fork lineage (`/home/mark/shared/lumoFlyWheel_codex_fork`, remote MaCoredroid/Lumo_FlyWheel-qwen-diffusion). Run under `VLLM_USE_V2_MODEL_RUNNER=1` with `VLLM_ATTENTION_BACKEND=TRITON_ATTN`, in a new venv alongside `.venv-vllm` (keep 0.23 intact as the AR baseline/fallback arm).

**Why re-pin off 0.23 (deviation from the stated requirement, flagged explicitly):** 0.23 has the MRV2 mechanism but NOT `diffusion_gemma.py`, NOT the diffusion structured-output guardrails, and NOT MRV2 align-APC. Building diffusion decode on 0.23 means hand-building exactly the scheduler/commit/attention machinery that v0.24+main already ships first-party. The retrain-freely rule generalizes: don't let an existing venv pin constrain the design. The 0.23 venv stays as the guided-AR baseline (eval-parity requires same-engine fairness — see risk R6).

### REUSE VERBATIM (external, in-tree vLLM)
- **`ModelState` ABC** (`vllm/v1/worker/gpu/model_states/interface.py`): `prepare_inputs / prepare_attn / custom_sampler / add_request / remove_request`, self-registration via `get_model_state_cls()` — zero runner/scheduler changes needed (verified claim).
- **DiffusionGemma commit machinery** (`diffusion_gemma.py`, 1363 lines): canvas-as-draft-tokens on the spec-decode path, `num_new_sampled_tokens_per_step=0`, KV advances **only on commit**, variable-accept in-kernel state scatter (`num_accepted_tokens` + `ssm_state_indices`) — this is the wave-2 parallel-commit primitive, already built.
- **Per-request causal/bidirectional attention**: Triton unified kernel `per_seq_causal_ptr` (`triton_unified_attention.py`) — sm_120-portable, no FA4 dependency. Denoise read = per-seq causal-off (all-ones over prefix+block for block queries, exactly the flare noisy mask); commit advance = causal-on.
- **GDN hybrid MRV2**: `MambaHybridModelState` (mamba_hybrid.py, landed v0.22, actively iterated) + `fla fused_recurrent(..., inplace_final_state=False)` (fused_recurrent.py:186) — the read-only-denoise flag already exists in-kernel.
- **Align-mode APC**: `--mamba-cache-mode align --mamba-block-size 1024 --mamba-ssm-cache-dtype float32` + `--enable-prefix-caching` (#30877/#27289/#42406) — exact-prefix-hash-keyed boundary checkpoints, restored by copy-into-slot (satisfies FlarePrefixCache's clone-on-restore invariant for free).

### REUSE VERBATIM (in-house flywheel)
- **Serving surface**: `inference_proxy.py` / `round_driver.py` / registry entry / `qwen3-openai-codex.jinja` (hash-pinned) — pure HTTP, engine-version-agnostic; the diffusion engine just keeps the OpenAI surface.
- **Model + gates**: `models/qwen3.5-9b-fastdllm-b1000-vllm-bf16` (loads as `Qwen3_5ForConditionalGeneration`, AR-parity PASS = built-in exactness verifier), `qwen35_9b_ar_parity_gate.py`, P1b speed probe, the hashed 20-ep/63-turn matched-eval harness.
- **Decode-mode hook**: `fr10_decode_modes.py` `SamplingParams.extra_args` pattern → add `block_diffusion` mode carrying wave params (tau, block_size, FSM id) per request; `fr10_equivalence_gate.py` as the raw-exactness promotion instrument.

### ADAPT (spec/pattern donors, not runnable code)
- **`scripts/flare_hf_cache.py`** — THE port spec: `RequestDiffusionState` semantics (fp32 GDN boundary snapshot validated ≤1e-3, conv_tail W-1 tokens, shifted-logits capture, advance-once, clean-only prefill advance) re-expressed as a `Qwen3_5FlareModelState`.
- **FR13 discipline corpus** — boundary-aligned-commit-verbatim invariant, off-boundary = poison; `fr13_replay_conv_remap.py` documents the conv/ssm shared-page `as_strided` traps the conv_tail write must respect.
- **dllm-plugin contracts** (read-only clone): commit-0 rollback of `num_computed_tokens`, grammar-shrinks-prefix-never-rewrites rule, documented dead ends (torch.compile removed, CUDA-graph blockers = host-side `tolist()` syncs).
- **dInfer graph recipe** (M3 only): control-flow-free commit logic, zero GPU→Python syncs, shape-bucketed graph capture with shared pool.
- **dLLM-Serve finding**: chunked `compute_logits` cap — our transient wave logits are block×151,936×4B fp32 (~0.6 GB at block 1024, bs=1) on a 0.745-util 32 GB card; budget it as an explicit line item.

### WRITE OURSELVES (the genuinely new code — deliberately small)
1. **`Qwen3_5FlareModelState`** (~medium, well-templated): composes `MambaHybridModelState` (prepare_attn/state handling) + DiffusionGemma's canvas/prepare_inputs. New logic: denoise-phase flag → `inplace_final_state=False` + skip the layer-level ssm write-back (`qwen_gdn_linear_attn.py:1532`) + conv-state write suppression; on commit → one clean causal pass, fp32 boundary snapshot + conv_tail published VERBATIM into the align-mode block row; shifted-logits capture (last noisy-stream logit before advance).
2. **Per-call wave scheduler inside `custom_sampler`**: wave-1 grammar-FSM scaffold projection (truly-forced-only), wave-2 tau-0.95 parallel commit of VALUE tokens with right-context-beyond-call-k masked, per-call advance. All-GPU bookkeeping (dInfer precondition) from day one.
3. **Offline token-level grammar FSM** compiled from the tool schemas in the qwen3_xml codex dialect (seed terminals from `schema_aware_drafter.py`; ship per-request via the existing `X-Lumo-Oracle` header or extra_args). Fully custom — both external tracks explicitly refuse this (plugin rejects it; v0.24 ships guardrails only, and hard-errors structured output for diffusion, so custom_sampler is the sanctioned home).
4. **Observability counters**: read_calls/advance_calls/residual-full-context, mamba-APC hit/miss/reused-tokens (upstream has NONE — mandatory, silent-0% is structural per #40696/#45238), force-counters (label-free audit: values must stay 0).

### Why this beats the alternatives
It's the only route where the four P2 pillars each land on an existing substrate: engine loop + commit path (in-tree, first-party, blogged, tagged release), bidirectional attention (sm_120-viable Triton), GDN state cache (align APC — already proven 1.87x on THIS 5090), serving surface (our proxy). Honest coverage ~50-60% of engine-side effort, and specifically the half hardest to build solo. Everything else either abandons the flywheel stack (SGLang, dInfer), downgrades to a pinned 0.20 personal fork (dllm-plugin), or rebuilds the whole loop by hand (0.23 injection). The one thing NOBODY has built — diffusion × GDN-hybrid × align-APC (#42792 still WIP) — we own regardless of route; here it's the *only* thing we own.

---

## 2. RANKED ALTERNATIVES

**A2 — vLLM 0.23 V1-runner hook injection (fallback, not first choice).** Stay on the proven `.venv-vllm`, port the dllm-plugin field-overloading contract (`--scheduler-cls/--worker-cls`, spec_token_ids = next block, sampled = committed, commit-0 rollback) + apply #36391's hook relaxation to our fork; align APC already works here. **Pro:** zero version churn, every launch flag already validated on this box, keeps eval fairness trivially same-engine. **Con:** no ModelState, no diffusion attention plumbing, no variable-accept scatter — we hand-build ~2-3x more engine code, re-deriving the 18.5k-line patcher's seam map on 0.23. This is the designated fallback if MRV2×GDN is broken (kill criterion K2).

**A3 — SGLang dLLM host.** Merged, maintained, the strongest external validation of chunked-prefill-as-denoise + radix cache; `DllmAlgorithm` plugin is the right wave abstraction. **Con:** 0% GDN×diffusion (their dLLM path assumes softmax KV; GDN state would be advanced by every denoise re-chunk — invasive novel surgery in their hybrid pool), abandons proxy/FR13/parity-gate/matched-eval assets, sm_120 gaps. Use as design reference only (already-extracted contract notes; optionally the 2-day SDAR-8B spike for a living reference number).

**A4 — dInfer / dllm-plugin / Fast-dLLM as hosts.** All rejected as hosts (dormant + double-pinned internals / 0.20 personal fork / HF-eager = the thing we measured losing). Recipe value only, folded into the recommended path above.

---

## 3. THE INTEGRATION SEAM

**Plug point:** `Qwen3_5FlareModelState` registered via `get_model_state_cls()` on the served `Qwen3_5ForConditionalGeneration`; per-request activation via `SamplingParams.extra_args["decode_mode"]="block_diffusion"` (fr10 pattern). One decode "step" = one denoise wave riding the spec-decode-shaped path:

| RequestDiffusionState (flare_hf_cache.py) | Engine mapping (vllm main, MRV2) |
|---|---|
| masked active block, B tokens | canvas as draft tokens (`spec_token_ids`-shaped buffer), `prepare_inputs` |
| noisy read mask (all-ones prefix+block, block queries only) | `prepare_attn` per-seq `causal=False` (Triton unified kernel) |
| GDN run from boundary `initial_state`, **no write-back** | `fused_recurrent(..., inplace_final_state=False)` + denoise-phase flag skipping the assignment at `qwen_gdn_linear_attn.py:1532` + conv-state write suppression |
| full-attn clean prefix KV, append-on-commit | paged KV advanced only on commit (DiffusionGemma pattern) |
| wave commits (tau ≥ 0.95, per-call) | `custom_sampler` decides; `sampled_token_ids` = 0..N committed, commit-0 → `num_computed_tokens` rollback |
| `advance()` once per block: clean causal pass, fp32 `chunk_states[:,-1]` + conv_tail snapshot; shifted-logits capture first | commit forward with per-seq `causal=True`; publish state VERBATIM into the align-mode block-aligned row; capture last noisy logits before advance |
| `advance_clean_only` (prompt prefill) | stock chunked prefill, `max-num-batched-tokens == mamba-block-size` (one boundary per step) |
| FlarePrefixCache (exact prefix + block_size key, clone-on-restore, 43/43 hits) | align APC exact-prefix-hash restore-by-copy + **our** hit/miss/reused counters; constraint: diffusion block size must divide `--mamba-block-size` (1024, multiple of FLA_CHUNK 64) so commit boundaries land on checkpoint rows |
| stats / force-counters | ModelState-owned counters surfaced through proxy metrics rows |

**Exactness verifier:** the model's clean stream is byte-identical to the AR forward, so every ModelState behavior is gateable against (a) the in-engine AR path and (b) `flare_hf_cache.py` step-by-step via `fr10_equivalence_gate.py`.

---

## 4. MILESTONES (one RTX 5090, gpu_util ≤0.745, solo)

**M1 — smallest end-to-end proof: one agentic turn served engine-fast.** (~2-2.5 weeks)
- Day 1-2 smoke gauntlet: (a) DiffusionGemma NVFP4 under TRITON_ATTN on sm_120 (proves dLLM path on this card); (b) Qwen3.5-9B export under MRV2, default then align+APC (tests whether #38041 is stale and #42406 holds); (c) 20-line read-only-denoise probe — forward same block twice from fixed initial_state, diff conv/ssm slots (go/no-go artifact).
- Then: `Qwen3_5FlareModelState` skeleton → denoise-read parity vs flare_hf_cache logits → advance-once + shifted-logits parity → greedy per-call waves (FSM stub = leftmost-forced only) on one matched-eval turn, temp 0, native template. **Gate:** turn output byte-matches the HF reference decode; forwards/turn ≈ 6-9; s/turn already < HF's 1.442.

**M2 — parity + speed on the matched eval.** (~1.5-2 weeks)
- Full offline FSM from tool schemas; wave-1/wave-2 wiring; cross-turn APC with counters (43/43-analog exact-hit audit; trigger-test #40696/#45238/#43587 on our prompt shapes); chunked lm_head cap.
- Rerun hashed slice (baf90863, 20 ep/63 turns) vs guided-AR-on-vLLM re-baselined **on the same engine build** (fairness). **Gate:** < 1.120 s/turn AND ≥55/63 exact-args, 15/20 episodes, 63/63 exact_seq, 63/63 valid_xml, force-counters 0 on values. Commit+push each step with narrated reasoning (standing workflow).

**M3 — engine-grade per-forward cost + batching.** (~1-2 weeks)
- dInfer recipe: control-flow-free wave logic (no `.item()`/`tolist()` — the plugin's documented graph blockers), shape-bucketed CUDA-graph capture bs=1 first (we currently run `--enforce-eager`; this is where the残り headroom lives), then multi-seq waves via UNIFORM_BATCH-style padding. **Gate:** GPU util healthy per standing rule (profile, no host-bound stalls); target the honest ~2-3x blended band vs AR at held quality.

Total: ~5-6.5 weeks solo. M1's day-1-2 gauntlet is deliberately front-loaded to fail fast.

---

## 5. RISKS + KILL CRITERIA

- **R1 MRV2×GDN broken (#38041)** — likely stale (predates hybrid PR) but unproven. *Kill K1:* if Qwen3.5 can't forward under MRV2 within 5 working days of fixes/upstream triage → drop to A2 (0.23 V1 injection), carrying the ModelState design as our own seam spec.
- **R2 sm_120 attention** — Triton unified per_seq_causal is portable but diffusion-on-sm_120 is publicly untested; FlashInfer JIT unusable here (no nvcc). *Kill K2:* DiffusionGemma smoke fails after 2 days of backend fallbacks → same A2 fallback (its bidirectional mask would be our Triton kernel either way).
- **R3 engine churn** — biweekly releases mid-MRV2 migration; #42792 (spec×align) WIP could land under us. Mitigate: hard commit pin, no tracking; budget exactly one rebase (post-M2). We own the diffusion×GDN×align intersection; treat upstream landings as free upgrades, not dependencies.
- **R4 align-APC pathologies** (#40696 short-prompt 0-hit, #45238 checkpoint-in-unique-tail, #43587 incremental turns) — all specifically hit GDN + our multi-turn shape. Mitigate: M2 trigger tests + mandatory in-house counters. Fallback: port the EXACT_SEED chunked-checkpoint mechanism (bit-exact on GB10) to the fork.
- **R5 semantic drift in the port** (shifted logits, conv_tail seam, decode-order invariant) — regression to the 0/41 corruption regime is silent without gates. Mitigate: equivalence-gate every stage against flare_hf_cache.py; value force-counters asserted 0.
- **R6 fairness** — moving engines invalidates the 1.120 s/turn number. Re-baseline guided AR on the pinned build (align APC flags identical) before claiming the win; quality caveat stands (N=20, single seed, synthetic tool results).
- **Kill K3 (thesis-level):** if at M2, with read-only O(block) denoise verified and (post-M3) graphs on, diffusion still misses 1.120 s/turn by >20% at healthy GPU util — the ~13x-fewer-forwards ⇒ wall-clock-win thesis fails on this hardware; stop, publish the profile, and re-scope (kernel-level work or accept the quality-only win). No sunk-cost continuation past that gate.

**Honestly not reusable, anywhere:** the wave-1 token-level grammar-FSM-over-parallel-positions (every external track explicitly refuses it), the fp32-boundary-snapshot/read-only-denoise/advance-once GDN-diffusion semantics (no precedent in any engine), and the GB10 patcher/tree-kernel code as runnable artifacts (seam-map and discipline value only). Those three, plus counters, are the entire write list — everything else is reuse.