# P2 Engine Gauntlet — Steps 4-6 on the REAL diffusion export (2026-07-03)

Re-run of `engine_build_status.md` §3 steps 4-6 on the **real diffusion-trained
vLLM export** produced by the export agent
(`models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16`, 19G; RL-v2 adapter merged onto the
init base — `real_diffusion_export_block_reconcile.md`), replacing the b1000
smoke/AR-parity export the earlier gauntlet was blocked on. Card: RTX 5090
(sm_120). Engine venv `.venv-vllm-p2-main` (editable vLLM
`/home/mark/shared/vllm_p2_pr42406`, branch `qwen3_5-flare-modelstate`). Every
GPU process ran one-at-a-time in the `systemd-run … MemoryMax=22G` RAM cage;
`free -g` >= 27G available before each boot. Block size pinned to
`canvas_length = 32` (`diffusion_config`) to match training (`bd_size=32`) and the
winning HF hybrid-clean eval (`--block-size 32`).

## Bottom line

| step | verdict | one-line |
|---|---|---|
| 4 — read-only-denoise probe (M1 go/no-go) | **PASS (core invariant), caveat** | after widening the restore scope, the whole GDN cache is **bit-identical** around every denoise forward (0 leaks, 10/10). A/B on/off is NOT identical because the engine sampler is off-paradigm. |
| 5 — turn byte-parity (engine vs HF) | **BLOCKED (FAIL)** | the three structural blockers from the smoke run **persist on the real export** (re-verified): orphaned FSM / format mismatch / unwired turn-adapter. |
| 6 — matched-20 M2 battery on engine | **BLOCKED (FAIL)** | engine turn-driver raises `EngineUnavailable`; canvas sampler non-deterministic + off-paradigm → exact_args / episode_exact / TRUE fwd-per-turn / s-per-turn cannot be produced. No engine-vs-HF wall-clock. K3 unadjudicable on the engine. |

**Net:** the sm_120 substrate and the read-only-denoise *mechanism* are validated
on the real checkpoint — the M1 GDN-state-discipline crux is now provably
enforceable (bit-identical). But the engine's **served decode path is a
canvas/renoise denoiser, a different algorithm than the `hybrid_clean` the model
was trained for**, so it emits gibberish and there is still **no engine-vs-HF turn
parity and no honest engine wall-clock**. Steps 5-6 need net-new engineering (wire
the FSM onto the engine + a dual-format checkpoint + the harness seam), not a
re-run.

---

## Step 4 — read-only-denoise probe (M1 go/no-go)

Probe `step4_real_probe.py`: boots FLARE in-process on the real export,
monkeypatches the GDN snapshot/restore ops, and per-row float32-fingerprints the
WHOLE conv/ssm cache around every denoise forward (negligible memory, no
full-cache clone). Invariant: "denoise advances GDN state by exactly 0" ⇒ after a
denoise forward + restore, NO row of the GDN cache may change.

### 4a. Widen the protection (the mandated fix)

Probe on the current committed engine (`22f660c`, "protect the block-table
checkpoint slot") **FAILED** on the real export, reproducing the task diagnosis:

```
protected rows = [1]     kernel-written rows = [1,2,3,4]
after restore: rows [2,3,4] still changed on 10/10 denoise forwards
max rowsum diff after restore = 5872.18
```

Root cause (GPU-measured, `step4_measure.json` / `step4_instrument.json`): this
ModelState runs with **num_spec == 0** (`spec_state_indices_tensor` is None), so
the collapsed `mamba_get_block_table_tensor` view names only checkpoint slot [1].
The forward writes the align **running-state** row (`non_spec_state_indices`
advances 1→2→3 across denoise sub-steps) and the fused align publish touches
neighbours — none named by the pre-forward block-table view. GDN cache: 24 layers,
`ssm_state [283,32,128,128]` fp32 (~566 MiB/layer for all rows → full freeze OOMs;
touched set is only a handful of slots).

**Fix (committed to the vLLM pin, `af21dc8`):** take the snapshot in `prepare_attn`
(post metadata-build, pre-forward) so it reads the ACTUAL `non_spec ∪ spec` state
indices the kernel is handed; protect `{non_spec ∪ spec} ∪ {block-table checkpoint
slots}` widened by a per-anchor contiguous guard band
(`VLLM_QWEN3_5_FLARE_READONLY_BAND`, default 4) to cover the align running-state
slots the pre-forward metadata cannot name. Per-anchor (not min..max) so batch>1
never freezes unrelated sequences; cheap (GDN recurrent cache, not KV). Mixed
commit+denoise batches keep the block-table-scoped set so commit rows advance.

### 4b. Re-run — whole-cache bit-identical: PASS

```
denoise forwards with restore        = 10
forwards restore was load-bearing    = 10/10
leak forwards (nonzero after restore)= 0
max rowsum diff after restore        = 0.0
per forward: changed [1,2,3,4] -> [] (protected [0..5])
STEP4_READONLY_ON: PASS
```

The entire conv/ssm cache is bit-identical around every denoise forward and the
restore is load-bearing. The M1 GDN read-only-denoise crux is achievable and now
enforced on the real export.

### 4c. A/B readonly on/off — NOT identical (root cause: off-paradigm sampler)

| run | committed-token sha256[:16] | output (all gibberish) |
|---|---|---|
| readonly OFF | `aa08de30f31df636` | `AustrAustr腔 Special musico…` |
| readonly ON, [1]-only (`22f660c`) | `f1a3a2982626becd` | ` allapot allapot腔 Special…` |
| readonly ON, banded (this fix) | `37509cc702a078e8` | `,,\n\n  11…` |

All three differ ⇒ the read-only scope is load-bearing (changes committed tokens);
and every output is gibberish. One root cause: the engine's served sampler is a
canvas/random-renoise denoiser, a different algorithm from the `hybrid_clean`
masked-diffusion the checkpoint was trained for. So (a) the engine path cannot
produce meaningful output on this checkpoint regardless of read-only scope, and
(b) "which scope is semantically correct" needs the HF `hybrid_clean` parity
reference, which is blocked (step 5). The A/B-identical-on-meaningful-output check
is not satisfiable on the engine path today — same blocker as steps 5-6.

**Step 4 verdict: PASS on the core M1 go/no-go artifact** (whole-cache bit-identical
read-only denoise, restore load-bearing, engine boots + forwards on the real
export). A/B-meaningful check N/A until the engine runs the trained decode
paradigm. Live counters (sane): `read_calls=10 advance_calls=1
read_advance_ratio=10.0 residual_full_context_model_calls=0 block_size=32
route_verified=False`.

---

## Step 5 — turn byte-parity (engine hybrid_clean vs HF): BLOCKED

The three blockers from `p2_engine_parity_smoke_result.md` were **re-verified on
the real export** and all persist (as that report predicted):

- **A — engine has no hybrid_clean path.** Served sampler is a canvas/renoise
  denoiser; the FSM (`vllm/v1/sample/hybrid_clean.py`) is orphaned. Re-observed:
  engine emits gibberish and gives different outputs for readonly on/off (§4c) —
  not the reference greedy+FSM single-token decode.
- **B — "same checkpoint" undefined.** Real-export `config.json`:
  `architectures=['Qwen3_5ForConditionalGeneration']`, `model_type=qwen3_5`,
  `auto_map=None`, `mask_token_id=None`, has `vision_config`. HF hybrid_clean needs
  the `Fast_dLLM_Qwen3_5` bridge + `mask_token_id` → cannot load this export.
- **C — harness turn-adapter unwired.** `VllmFlareEngineAdapter.run_turn` still
  raises `EngineUnavailable` on the real export (re-run of
  `runs/p2_engine_parity_smoke/artifacts/step5_adapter_probe.py`).

Different algorithms on different on-disk formats with no driving seam. BLOCKED /
FAIL. Unblock path unchanged (smoke report): wire `HybridCleanDecodePolicy` onto
the engine (or a forward-only logit seam), produce one dual-format checkpoint, and
implement `VllmFlareEngineAdapter.run_turn` + `snapshot_from_vllm_modelstate`.

---

## Step 6 — matched-20 M2 battery on the engine path: BLOCKED

Requires the missing seam (Blocker C): a driver running 20 episodes through the
engine with per-turn tool-result injection + committed-token/boundary readout.
`run_turn` raises `EngineUnavailable`, and the canvas sampler is non-deterministic
+ off-paradigm (§4c). So `exact_args`, `valid`, `episode_exact`, TRUE forwards/turn
and s/turn **cannot be produced on the engine** — no honest engine number, no
valid A/B vs re-baselined guided-AR.

Reference rows (matched-20, `runs/endgame_scoreboard`, unchanged):

| row | exact_args | episode_exact | valid | s/turn | fwd-or-tok/turn |
|---|---:|---:|---:|---:|---:|
| OUR HF hybrid-clean (v2) | 47/63 | 13/20 | 63/63 | 3.904 | 56.83 denoise fwd/turn |
| stock-bf16-AR-guided | 51/63 | 14/20 | 63/63 | 1.213 | 82.24 tok/turn |
| stock-AR aggregate | 124/247 | 33/80 | 247/247 | 0.741 | 49.06 tok/turn |

M2 gate / K3 (as written): PASS needs `< 1.120 s/turn AND >= 55/63 exact-args,
15/20 ep, 63/63 exact_seq, 63/63 valid_xml, value force-counters == 0`. K3 kills
the thesis if, at M2 with read-only O(block) denoise verified and graphs on,
diffusion misses 1.120 s/turn by > 20% at healthy GPU util.

**K3 verdict: cannot be adjudicated on the engine path** — the engine wall-clock
P2 exists to measure is unobtainable until steps 5-6 unblock. The only diffusion
wall-clock is the HF-stack reference (3.904 s/turn matched-20, ~3.5x the K3 target
and ~5.3x stock-AR aggregate), which is not the engine. The only honest engine
signal is read/advance ratio ~10 (fewer-forwards mechanism live) — substrate
liveness, not the M2 KPI. BLOCKED / FAIL — no sunk-cost engine number invented.

---

## Artifacts
- vLLM pin commit `af21dc8` (branch `qwen3_5-flare-modelstate`) — read-only restore widening.
- `step4_real_probe.py` — ON/OFF whole-cache probe.
- `step4_real_on.json` — pre-fix FAIL (leak [2,3,4], 5872.18).
- `step4_real_on_default.json` — post-fix PASS (0 leaks, 0.0).
- `step4_real_off.json` — readonly-OFF A/B run.
- `step4_measure.json` / `step4_instrument.json` — GPU measurement of row provenance.

## Reproduce (RAM cage; one heavy proc at a time)
```
VENV=/home/mark/qwen_diffusion/.venv-vllm-p2-main
systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=4G \
  -E CUDA_HOME=$VENV/lib/python3.12/site-packages/nvidia/cu13 \
  -E NVCC_APPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK \
  -E VLLM_USE_FLASHINFER_SAMPLER=0 -E VLLM_USE_V2_MODEL_RUNNER=1 \
  -E VLLM_ATTENTION_BACKEND=TRITON_ATTN -E VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  -E VLLM_QWEN3_5_FLARE=1 -E VLLM_QWEN3_5_FLARE_READONLY_DENOISE=1 -E MAX_JOBS=4 \
  -E READONLY=on -E OUT=runs/p2_engine_gauntlet_real/step4_real_on_default.json \
  -- $VENV/bin/python runs/p2_engine_gauntlet_real/step4_real_probe.py
# Set VLLM_QWEN3_5_FLARE_READONLY_BAND=0 to reproduce the pre-fix leak.
```
