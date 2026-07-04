# P2 Step-5 Engine↔Reference Byte-Parity (smoke export) — BLOCKED

**Date:** 2026-07-03 · **Card:** RTX 5090 (sm_120) · **Checkpoint (as tasked, checkpoint-agnostic):**
`models/qwen3.5-9b-fastdllm-b1000-vllm-bf16` (the SMOKE export).

## Verdict

**Step-5 byte-parity CANNOT be run as specified, and the block is structural /
implementation-level, not checkpoint-quality.** Three independent, code-verified
gaps each individually prevent a same-checkpoint token-for-token comparison
between the engine FLARE path and the HF hybrid_clean reference. This is *not* a
subtle logit-drift divergence to be reported with a first-divergent-token — the
two sides run **different decode algorithms on different on-disk model formats**,
and the harness seam that would drive the comparison is **not wired**. The
divergence, such as it is, is total (position 0) and expected.

This is the concrete, on-GPU confirmation of the "Orphaned FSM" open issue
already flagged in `engine_build_status.md` §2 (reviews 2 & 4) and debt item §4.1.

| Gate the task asked for | Status | Why |
|---|---|---|
| same checkpoint both sides | **impossible** | different architectures on disk (blocker B) |
| engine FLARE **hybrid_clean** path | **does not exist** | served path is a canvas denoiser; FSM is orphaned (blocker A) |
| greedy/argmax, identical grammar FSM | **N/A** | engine sampler has no grammar/FSM/greedy path (blocker A) |
| byte-match token-for-token | **not measurable** | harness vLLM turn-adapter unwired (blocker C) |
| `projected_value_tokens_exact==0`, live counters nonzero-sane | engine: **partial** | engine counters captured & sane; the projection/FSM counters live only on the (unrunnable-here) reference |
| first-divergent-token + both top-5 | **not producible** | reference decode is *undefined* on this export (no mask token / no bridge) — see blocker B |

---

## Blocker A — the engine's served decode path is a canvas/renoise denoiser, not hybrid_clean

`VLLM_QWEN3_5_FLARE=1` routes `Qwen3_5ForConditionalGeneration` ->
`Qwen3_5FlareModelState` -> **`Qwen3_5FlareSampler`**, which is a *canvas/commit
block denoiser* (structurally a slimmed `DiffusionSampler`):

- initializes a **random** canvas of `canvas_length` tokens (`init_canvas` ->
  `torch.randint`);
- each denoise step applies a temperature schedule -> +1 right-shift -> **Gumbel
  sample** -> entropy-bound accept / **random renoise** (`torch.rand`,
  `torch.randint`) -> stability+confidence convergence;
- commits a **whole block** of `argmax_canvas` tokens at once.

The HF reference (`scripts/eval_flare_northstar_hybrid_clean.py::sample_hybrid_clean`
+ `scripts/flare_hf_cache.py`) is a completely different algorithm:

- appends **one `[MASK]` sentinel** to the sequence, runs `shifted_active_logits`
  (the +1 right-shift), takes the **last-position** logit, suppresses `mask_id`
  to -inf, suppresses stop tokens when not `can_stop`;
- runs a **grammar FSM**: bulk-commits FSM-forced tokens (exactly-1 legal
  candidate), else greedy/unconstrained samples the single next token;
- emits **one token per step**.

These cannot byte-match by construction: block-parallel stochastic denoise vs.
single-token sequential greedy+FSM. Grep confirms the FSM is orphaned:

    $ grep -rn hybrid_clean vllm/ --include=*.py | grep -v vllm/v1/sample/hybrid_clean.py
    vllm/v1/worker/gpu/model_states/qwen3_5_flare.py:278:  # ...the orphaned hybrid_clean reference decoder...   (comment only)
    $ grep -rn "HybridCleanDecodePolicy|parse_hybrid_clean_request" vllm/ --include=*.py | grep -v .../hybrid_clean.py
    NONE

`vllm/v1/sample/hybrid_clean.py` (the FSM port) is imported by **nothing** on the
serving path. The "identical grammar FSM" the task requires is not present in the
engine at all.

### Empirical proof the served path is not even greedy/deterministic

Booted the real engine on the smoke export (`artifacts/step5_engine_side.py`,
in-process, greedy request `temperature=0`, `diffusion_config.canvas_length=32`
to match the export `bd_size=32`) and issued the **same** greedy request twice:

    run1 first tokens: [199975, 199975, 97978, 9493, 32071, 127681, ...]
    run2 first tokens: [ 22676,  22676, 46857, 131445,   13, 228956, ...]
    greedy_deterministic_across_two_identical_requests: FALSE

Two identical "greedy" requests produce **entirely different** sequences. The
canvas sampler runs its own temperature schedule + Gumbel + random renoise and
**ignores `SamplingParams.temperature`**, so it is not a greedy-argmax decoder
and is not even self-reproducible without a fixed seed. A greedy byte-parity
against it is undefined.

---

## Blocker B — "same checkpoint" is impossible: the two sides consume different on-disk formats

The task's export and the reference's model are **different architecture classes
on disk** (`artifacts/checkpoint_format_evidence.json`):

| | `-b1000-vllm-bf16` (engine loads) | `-fastdllm-init` (reference loads) |
|---|---|---|
| `architectures` | `Qwen3_5ForConditionalGeneration` | `Fast_dLLM_Qwen3_5ForCausalLM` |
| `model_type` | `qwen3_5` (stock, multimodal: has `vision_config`) | `Fast_dLLM_Qwen3_5` |
| `auto_map` | **None** | -> `modeling.py` / `configuration.py` bridge |
| `mask_token_id` | **None** | `248077` (`|<MASK>|`) |

The HF hybrid_clean reference **cannot load the vLLM export**: it needs the
`Fast_dLLM_Qwen3_5` bridge forward (route_i two-stream `shifted_active_logits`)
and a `mask_token_id` to append. The stock export has neither — the reference
decode is *undefined* on it, which is exactly why the "first-divergent-token +
both top-5" artifact cannot be produced (there is no reference side to read a
top-5 from on this checkpoint).

Additionally the reference default runs `-init` **+ the B@1000 adapter**
(`runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000`, present), i.e. base+adapter
weights — not the merged `-vllm-bf16` export. So even the weights are not the
same tensors. "Same checkpoint both sides" would require a single dual-format
export that is byte-equal in both the Fast_dLLM bridge form and the vLLM form;
that artifact does not exist.

This is checkpoint-format, not checkpoint-quality: a freshly trained export
(e.g. the RL-v2 export being built in a concurrent session) would still be a
stock `Qwen3_5ForConditionalGeneration` vLLM file with no bridge/mask token, so
all three blockers persist independent of training.

---

## Blocker C — the parity harness's vLLM turn-adapter is not wired

`scripts/parity_audit_flare_engine.py --mode turn --engine vllm` is the intended
byte-parity gate. Its `VllmFlareEngineAdapter.run_turn` **locates** the class but
then raises `EngineUnavailable` — the serving driver is not implemented
(`artifacts/step5_adapter_probe.py`, live):

    LOCATED_CLASS_FILE: .../vllm/v1/worker/gpu/model_states/qwen3_5_flare.py
    ENGINE_UNAVAILABLE_run_turn: Qwen3_5FlareModelState located ... but the vLLM
    serving driver is not wired into this harness. ... drive the turn via
    SamplingParams.extra_args decode_mode='block_diffusion', collect output token
    ids, and emit one StateSnapshot per committed block boundary via
    snapshot_from_vllm_modelstate() ...

So there is no code path today that drives one shared decode turn and reads token
ids + block-boundary snapshots out of the engine for comparison. `--engine self`
only re-runs the reference against itself (trivially byte-identical), and it too
requires the `-init` bridge + `peft` (absent from the engine venv
`.venv-vllm-p2-main`; the reference venv is `.venv-fastdllm`).

---

## What DID run — engine-side live capture (the reference side is unrunnable here)

`artifacts/step5_engine_side.json` — booted `Qwen3_5FlareModelState` on the smoke
export, greedy request, `canvas_length=32`:

- **FLARE live counters (nonzero-sane):** `read_calls=18`, `advance_calls=2`,
  `read_advance_ratio=9.0`, `residual_full_context_model_calls=0`, `block_size=32`.
  The read/advance ratio (~9 denoise reads per commit) is the fewer-forwards
  signal and is behaving, and the structural-zero tripwire is 0.
- **`route_verified=FALSE`** — the stock export exposes no `flare_gdn_route()`, so
  the engine cannot confirm it is running the route_i FLARE forward; the GROUND
  invariant is assumed from env, unverified (this is itself a parity risk).
- **Engine first-shift top-5 @ canvas pos 0** (the engine forward *does* produce a
  sane distribution): `[[149992, 10.125], [198, 9.56], [149373, 9.0],
  [148987, 9.0], [33, 8.94]]`. There is **no matching reference top-5** on this
  checkpoint (blocker B), so the paired first-divergence artifact the task asks
  for cannot be completed.
- Output is gibberish (`"AustrAustr... Special DataType..."`), consistent with the
  smoke export being non-diffusion-trained AND driven through a mismatched path —
  quality is irrelevant to this gate, as the task notes.

`projected_value_tokens_exact` is **not applicable** to the engine run: that
counter is produced by the audit battery over the reference's XML `<parameter>`
value tokens / two-wave projection channel, both of which live on the
hybrid_clean reference decode (unrunnable on this checkpoint), not the canvas
sampler. The audit machinery that would compute it is verified separately by
`--mode selftest` (15/15) and `--mode ops-parity` (18/18) per step-1.

---

## Unblock path (what must land before step-5 can be a real gate)

1. **Wire the hybrid_clean FSM decode onto the engine serving path** (or a
   forward-only seam): either integrate `HybridCleanDecodePolicy` into
   `Qwen3_5FlareSampler`, or expose an engine call that returns the same
   +1-shifted block logits for an externally supplied `x_t` so the shared
   `sample_hybrid_clean` driver can source logits from the engine. Only then are
   the two sides the *same algorithm* and a byte comparison is meaningful. (This
   is the true isolation the task wants: engine-forward == reference-forward at
   the logit level, before checkpoint quality.)
2. **Produce one dual-format checkpoint** that is byte-equal as both the
   `Fast_dLLM_Qwen3_5` bridge (with `mask_token_id`) and the vLLM
   `Qwen3_5ForConditionalGeneration` export — otherwise "same checkpoint both
   sides" is undefined.
3. **Implement the harness seam** `VllmFlareEngineAdapter.run_turn` +
   `snapshot_from_vllm_modelstate` (read `Qwen3_5FlareRequestStates.block_start /
   last_shift_logits` and `_gdn_caches()` rows post-commit) so
   `--mode turn --engine vllm` can drive one shared turn and emit the byte /
   state-snapshot report.
4. Only after 1-3: run 2-3 matched-20 turns, greedy, identical FSM, and assert
   token-for-token byte-identity + `projected_value_tokens_exact==0` on both sides.

Until then step-5 byte-parity remains **blocked**, and the honest KPI is: engine
substrate runs (counters sane), but the engine==reference equivalence is
**unmeasured** because the equivalent path does not yet exist.

## Reproduce

    # Blocker C (cheap, no model load):
    .venv-vllm-p2-main/bin/python artifacts/step5_adapter_probe.py
    # Engine-side live capture (in RAM cage; one heavy proc at a time):
    VLLM_QWEN3_5_FLARE=1 VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ATTENTION_BACKEND=TRITON_ATTN \
    VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    CUDA_HOME=.venv-vllm-p2-main/.../nvidia/cu13 \
    NVCC_APPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK MAX_JOBS=4 \
    systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=4G -- \
      .venv-vllm-p2-main/bin/python artifacts/step5_engine_side.py
