# P2 Engine Acceptance — Steps 5-6 on the REAL diffusion export (2026-07-03/04)

Acceptance re-run of `p2_engine_gauntlet_real_result.md` **after all three Step-5/6
structural blockers were wired**:

- **Blocker A** (vLLM pin `qwen3_5-flare-modelstate` `e38a9ea`): `hybrid_clean` made a
  selectable engine decode mode (`VLLM_QWEN3_5_FLARE_DECODE=hybrid_clean`) driving the
  FSM/greedy `HybridCleanBlockDecoder`.
- **Blocker B+C** (qwen_diffusion `ed479b3`): one dual-loadable checkpoint (HF-bridge
  loader over the vLLM export) + `VllmFlareEngineAdapter.run_turn` seam.

This acceptance ran the STEP-5 byte-parity and STEP-6 M2 gates against those wirings on
the real export `models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16` (block/canvas 32), RTX 5090
(sm_120), engine venv `.venv-vllm-p2-main`, every GPU process one-at-a-time in the
`systemd-run … MemoryMax=22G` cage (`free -g` ≥ 27 G available before each boot).

## Bottom line

| step | verdict | one-line |
|---|---|---|
| pre — CPU wiring intact | **PASS** | 61/61 CPU tests green (23 hybrid_clean + 17 hybrid_clean_flare_decode + 21 flare state-machine); no regression to the `af21dc8` read-only-denoise machinery. |
| 5 — turn byte-parity (engine hybrid_clean vs HF) | **FAIL (BLOCKED)** | TWO independent hard failures, both GPU-confirmed THIS run: (5A) the engine `hybrid_clean` sources **block-parallel** canvas logits while the HF reference is **sequential single-`[MASK]`** → algorithm-level divergence, byte-parity impossible by construction; (5B) real-length matched-20 turns hit a **deterministic CUDA illegal-memory-access at the first decode step** → the 3 parity turns cannot even be driven. |
| 6 — matched-20 M2 battery on the engine | **BLOCKED** | Gated on Step 5 (byte-parity ⇒ the ENGINE==HF-row quality gate). Byte-parity fails and real turns crash, so `exact_args` / `episode_exact` / TRUE forwards-per-turn / s-per-turn cannot be produced on the engine. No honest engine wall-clock. |

**Net:** the three blockers are *wired* (imports resolve, CPU tests green, the adapter
boots the real export and drives a short turn), but **the wiring does not make the engine
reproduce the trained algorithm, and it does not run at real prompt length.** Steps 5-6
still need net-new engineering: (1) close the per-token logit seam (run the reference's
sequential single-`[MASK]` forward schedule on the engine), and (2) fix a decode-at-scale
IMA in the shared FLARE canvas/commit spec-decode forward. Not a re-run. K3 remains
**unadjudicable on the engine path**.

---

## Pre-check — CPU wiring intact (PASS)

```
pytest tests/v1/sample/test_hybrid_clean.py \
       tests/v1/sample/test_hybrid_clean_flare_decode.py \
       tests/v1/worker/gpu/test_qwen3_5_flare_state_machine.py
=> 61 passed
```

Editable vLLM confirmed at `/home/mark/shared/vllm_p2_pr42406/vllm`; the `hybrid_clean`
FSM is now actually invoked at both the block-decoder and sampler seams (the assertion
the orphaned-FSM bug lacked). The `af21dc8` read-only-denoise fix is untouched.

---

## STEP 5 — turn byte-parity: FAIL

Per the acceptance rule ("run in order, stop at first hard failure"), Step 5 is the first
hard failure. There are two independent reasons, either of which alone blocks byte-parity.

### 5A — Algorithm-level divergence: block-parallel engine vs sequential single-`[MASK]` reference

This is a divergence of **algorithm**, not rounding, and it is **not an engine-side-and-small
fix**. Proven from both decoders' source:

- **HF reference** — `eval_flare_northstar_hybrid_clean.sample_hybrid_clean`
  (sha `a4c66751…`): decode is **sequential, one `[MASK]` at a time**. Each model-chosen
  token appends exactly one mask (`torch.cat([output_ids, mask], dim=1)`), forwards over
  `[committed_clean_prefix, MASK]`, and reads the **single last-position** shifted logit
  (`shifted_active_logits(...)[:, -1, :]`); one forward per non-forced token. Truly-forced
  structural tokens (`len(legal)==1`) are FSM bulk-committed with **zero** forwards. So the
  logit for output position *k* is conditioned on the **actual committed clean tokens
  0..k-1**.
- **Engine** — `qwen3_5_flare.Qwen3_5FlareSampler._hybrid_clean_step`: decode is
  **block-parallel**. One denoise forward runs over the whole 32-position canvas, then
  `_gather_block_logits` reshapes all 32 positions' logits `[num_decode, CL, vocab]` and
  `HybridCleanBlockDecoder.decode_block` walks them. Positions 1..31 are conditioned on the
  **noisy/random canvas**, not on the sequentially-committed clean prefix.

⇒ For any block containing >1 model-decoded token, the engine's per-position logits differ
from the reference's by construction; they cannot byte-match. The wiring commit itself
flags this as an **unclosed INTEGRATION SEAM** (`e38a9ea` `_hybrid_clean_step` docstring:
"byte-parity-exact with the reference **only once the per-token-vs-block-parallel logit
gap … is closed**").

**Empirical corroboration** (this export, no co-load needed):
- Engine `hybrid_clean` on a working short prompt emits **gibberish** —
  `assistant_text = "<tool_call>\n<function= .ер s ET    .  1. …"`
  (`runs/p2_engine_gauntlet_real/engine_smoke_adapter_short_hybrid_clean.json`), while its
  own zero-value-projection tripwire holds (`projected_value_tokens_exact=0`).
- The HF-bridge forward on the **same** export is coherent — top-1 `" Paris"`
  (`runs/p2_engine_gauntlet_real/blockerB_hf_bridge_forward.json`).

Same weights, same tokenizer, same mask id (248077): the difference is the decode
algorithm, exactly as the source shows.

**Divergence diagnosis:** ALGORITHM (not rounding). **First-divergent-position + top-5
logits could not be tabulated turn-for-turn** because the real-length parity turns crash
(5B); the short-prompt evidence already shows the engine's very first emitted structural
value token is off-distribution (gibberish) vs the reference's coherent stream.

**Fix scope:** NOT small / NOT engine-side-trivial. Byte-parity requires the engine to run
the reference's **sequential single-`[MASK]`** forward schedule (one forward per value
token over `[prefix, MASK]`), or to expose a **forward-only logit seam** feeding the shared
`sample_hybrid_clean` driver. Both are net-new engineering — and both remove the
block-parallel "fewer-forwards" mechanism for value tokens (see the strategic note below).

### 5B — Deterministic CUDA illegal-memory-access on real-length turns

The 3 matched-20 turns require the real agentic prompts. Turn-0 of episode 0 is **1041
tokens**; the engine crashes at the **first decode step**:

```
ERROR dump_input: scheduled_cached_reqs=CachedRequestData(req_ids=['0-84510935'],
  new_block_ids=[([9],[10],[11],[12])], num_computed_tokens=[1041], num_output_tokens=[1]),
  num_scheduled_tokens={0-…: 33}, scheduled_spec_decode_tokens={0-…: [-1 ×32]},
  num_common_prefix_blocks=[0,0,0,3], new_block_ids_to_zero=[12], num_spec_tokens_to_schedule=32
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

Prefill completes (`num_computed_tokens=1041`); the fault is the first decode
(1 real + 32 canvas draft tokens). Isolated across GPU boots this run
(`CUDA_LAUNCH_BLOCKING=1`, RAM cage):

| decode mode | read-only-denoise | mamba_block_size | prompt | result |
|---|---|---|---|---|
| — (short smoke, prior) | on | 1024 | 10 tok | **OK** |
| hybrid_clean | ON | 1024 | 1041 tok | **CRASH (IMA)** |
| canvas | **OFF** | 1024 | 1041 tok | **CRASH (IMA)** |
| canvas | on | **4096** (1041 in ONE mamba block) | 1041 tok | **CRASH (IMA)** |
| canvas | on | 1024, **APC off** | 1041 tok | config-rejected (align mandates `--enable-prefix-caching`) — not testable |

⇒ The IMA is **independent of decode mode** (canvas and hybrid_clean both), **independent
of the read-only-denoise snapshot** (crashes with it OFF — rules out the `af21dc8`
snapshot/restore rows), and **independent of the mamba-block-1024 boundary** (crashes with
the whole 1041-token state in one mamba block). It is in the **shared FLARE canvas/commit
spec-decode DECODE forward over a long, multi-KV-block prefix**. Under `CUDA_LAUNCH_BLOCKING`
the error surfaces at the async output-copy sync (`async_utils.get_output →
copy_event.synchronize()`); the exact faulting kernel needs `compute-sanitizer` (absent on
this box) or a `TORCH_USE_CUDA_DSA` rebuild — deferred.

**Diagnosis:** engine-side (vLLM `Qwen3_5FlareModelState` decode-at-scale), deterministic,
**NOT small**. This is the "block-32 mid-chunk / decode-at-scale" defect the `ed479b3`
smoke already flagged as an engine-owner handoff; this run pins it further (mode-,
readonly-, and mamba-block-independent). It blocks every real-length turn, so the 3
matched-20 byte-parity turns cannot be driven and `projected_value_tokens_exact` cannot be
compared turn-for-turn on real data.

**STEP 5 verdict: FAIL.** Stop here.

---

## STEP 6 — matched-20 M2 battery on the engine: BLOCKED

The correct quality gate is **ENGINE == HF row** (47/63, same weights + algorithm —
byte-parity implies it). Byte-parity fails (5A) and real turns crash (5B), so `exact_args`,
`valid`, `episode_exact`, TRUE forwards-per-turn and s-per-turn **cannot be produced on the
engine**. The only engine signal is short-prompt substrate liveness — `read_advance_ratio ≈
3.0`, `forced_grammar_tokens=5` (FSM bulk-commit, zero forwards), `zero_forward_rows=2`,
`projected_value_tokens_exact=0` — i.e. the fewer-forwards + zero-value-projection
mechanisms are live, but over **gibberish** output. That is substrate liveness, not the M2
KPI. No sunk-cost engine number was invented.

Reference rows (matched-20, `runs/endgame_scoreboard`, **NOT the engine**):

| row | exact_args | episode_exact | valid | s/turn | fwd-or-tok/turn |
|---|---:|---:|---:|---:|---:|
| OUR HF hybrid-clean (v2) | 47/63 | 13/20 | 63/63 | 3.904 | 56.83 denoise fwd/turn |
| stock-bf16-AR-guided | 51/63 | 14/20 | 63/63 | 1.213 | 82.24 tok/turn |
| stock-AR aggregate | 124/247 | 33/80 | 247/247 | 0.741 | 49.06 tok/turn |

**K3 (as written)** — PASS needs `< 1.120 s/turn AND ≥ 55/63 exact-args, 15/20 ep, 63/63
exact_seq, 63/63 valid_xml, value force-counters == 0`. **Speed verdict: cannot be
adjudicated on the engine path** — no honest engine wall-clock at real quality exists. The
only diffusion wall-clock remains the HF stack (3.904 s/turn, ≈3.5× the 1.120 target,
≈5.3× stock-AR aggregate), which is not the engine. Unchanged from the prior gauntlet, now
strengthened: **even with A/B/C wired, the block-parallel engine substrate cannot reproduce
the sequential reference and cannot run real-length turns.**

---

## Strategic note (surfaced by 5A)

The winning HF row's forward-savings (56.83 fwd/turn vs stock-AR's ~82 tok/turn) comes
**entirely from the grammar-FSM bulk-commit of truly-forced structural tokens with zero
forwards** — the reference decodes **every value token sequentially with one forward**. The
engine's block-parallel canvas is therefore a **different** algorithm, not a faithful
accelerator of the reference: on this checkpoint it is quality-dead (gibberish). A
byte-parity-and-quality engine path must run the sequential single-`[MASK]` value decode;
its only legitimate speed lever over guided-AR is the same FSM zero-forward bulk-commit,
not block-parallel value denoising. This should be reflected in the M-milestone plan.

---

## What remains before Steps 5-6 can pass (net-new engineering, not a re-run)

1. **Close the logit seam (Blocker 5A):** drive the engine with the reference's sequential
   single-`[MASK]` forward schedule, or expose a forward-only block-logit seam feeding the
   shared `sample_hybrid_clean` driver, so both sides are the same algorithm. Precondition
   for any byte-parity.
2. **Fix the decode-at-scale IMA (Blocker 5B):** localize with `compute-sanitizer` /
   device-side asserts (mode-, readonly-, mamba-block-independent; long multi-KV-block
   prefix, first decode), then fix in the FLARE canvas/commit spec-decode forward.
3. Then: 3-turn byte-parity (greedy, identical FSM, `projected_value_tokens_exact==0` both
   sides), then the matched-20 M2 A/B vs guided-AR **re-baselined on the same pinned build**.

---

## Artifacts (this acceptance) — `runs/p2_engine_acceptance/`
- `step5_ima_hybrid_clean_ro_on_mamba1024.log` — hybrid_clean, readonly ON, mamba 1024 → IMA.
- `step5_ima_canvas_ro_off_mamba1024.log` — canvas, readonly OFF, mamba 1024 → IMA (rules out readonly).
- `step5_ima_canvas_ro_on_mamba4096.{log,json}` — canvas, mamba 4096 → IMA (rules out mamba-block crossing).
- `step5_scheduler_dump_at_crash.txt` — the faulting decode step (num_computed_tokens=1041).
- `ima_mamba_block_probe.py` — the parametric IMA isolation probe.
- (prior, `runs/p2_engine_gauntlet_real/`) `engine_smoke_adapter_short_hybrid_clean.json`
  (engine gibberish + live counters), `blockerB_hf_bridge_forward.json` (HF-bridge coherent
  `" Paris"`).

## Reproduce (RAM cage; one heavy proc at a time)
```
VENV=/home/mark/qwen_diffusion/.venv-vllm-p2-main
systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=4G \
  -E CUDA_HOME=$VENV/lib/python3.12/site-packages/nvidia/cu13 \
  -E NVCC_APPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK \
  -E VLLM_USE_FLASHINFER_SAMPLER=0 -E VLLM_USE_V2_MODEL_RUNNER=1 \
  -E VLLM_ATTENTION_BACKEND=TRITON_ATTN -E VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  -E VLLM_QWEN3_5_FLARE=1 -E VLLM_QWEN3_5_FLARE_READONLY_DENOISE=1 -E MAX_JOBS=4 \
  -E CUDA_LAUNCH_BLOCKING=1 \
  -- $VENV/bin/python scripts/parity_audit_flare_engine.py \
     --mode engine-smoke --decode-mode hybrid_clean \
     --base-model models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16 \
     --tokenizer-path models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16 \
     --input-jsonl data/toolcall_eval_native/flare_scaleup_native_58.jsonl \
     --episode-index 0 --turn-index 0 --block-size 32 --max-new-tokens 32
# => CUDA illegal memory access at the first decode (num_computed_tokens=1041).
# mamba-block / readonly isolation: runs/p2_engine_acceptance/ima_mamba_block_probe.py
```
