# P2 Engine-Fast Diffusion Serving — Build Status & GPU Smoke Checklist

Workflow follow-on to `p2_serving_reuse_plan.md` (the reuse decision, milestones, kill criteria).
Date: 2026-07-04. Author: build+review sweep + real-export gauntlet + post-wiring acceptance +
IMA-fix / sequential-decode-rebuild acceptance.

**Bottom line (engine NOT promoted — one blocker left, now isolated to the forward):** the entire M1
write-list (`Qwen3_5FlareModelState` + ops + hybrid-clean FSM reference + parity harness + flywheel
serving surface) is implemented, CPU-tested, and committed. The gauntlet ran on the RTX 5090 (sm_120):
b1000 smoke export (§0), the REAL diffusion-trained export (§0.R — `qwen3.5-9b-fastdllm-rlv2-vllm-bf16`),
a post-wiring acceptance after blockers A/B/C were wired (§0.A), and then — after the **decode-at-scale
IMA (5B) was fixed and the sequential single-`[MASK]` decode was rebuilt on the engine (5A driver)** — a
final acceptance re-run of Steps 5-6 (§0.B). **Steps 1-4 PASS** (R1/R2/K1/K2 killed; §0.R Step-4
read-only-denoise bit-identical, 0 leaks 10/10, restore load-bearing under vLLM pin `af21dc8`). **The M1
substrate + GDN-state-discipline crux are PROVEN, and 5B is now closed** — the engine decodes real-length
turns (1041/1443/917-tok prompts, 300-tok canvas generations) without faulting. **Step 5 byte-parity
still FAILs**, now for a **single, precisely-isolated** reason:
- **5B — decode-at-scale CUDA IMA: FIXED** (vLLM pin `1e32dcd`). Root cause: `super().postprocess_state`
  (the MambaHybrid align spec-decode state copy) read the accepted draft token's intermediate GDN state
  from block-table column `src_col + (num_accepted−1)`, which assumes `num_accepted−1` **speculative**
  checkpoint columns exist — allocated only under a real `speculative_config`. The FLARE path drives the
  canvas as spec draft tokens **without** one (`num_speculative_blocks == 0`), so a commit crossing a
  mamba-block boundary indexed `src_col + (A−1)` far past the width-8 table ⇒ IMA (localized to
  `num_computed=1272` canvas / `1140` hybrid_clean, not the "first decode step" the pre-fix run guessed).
  Fix: feed the align state machine a neutral `num_accepted == 1` (a FLARE commit's final GDN state
  already lives in the running block; the real commit count is retained only for the counter). Verified:
  canvas runs the full 300-token cap with zero IMA (was faulting ~231); 66/66 CPU tests green.
- **5A — turn byte-parity: still FAIL, algorithmic, now moved into the FORWARD.** The sequential
  single-`[MASK]` **driver** was rebuilt (vLLM pin `5e2fb53`) and is correct — with 5B fixed, engine and
  HF bridge match **token-for-token through the first 12 grammar-forced tokens on the same dual-loadable
  export**, proving the FSM wiring is live. But at pos-12 (the first logit-dependent choice) the engine's
  **forward** output is wrong: top-5 logits diverge whole-distribution (ref `=num`=24.25 argmax; engine
  argmax `>`=18.25, `=num` ~16 lower — not a bf16 near-tie, and identical at shifted/raw-MASK/last-clean
  probes ⇒ not a +1-shift bug). Cause: the engine forward still runs over the **fixed 32-position
  spec-draft canvas** (`num_draft_tokens_per_req == num_spec == 32`), so the probe `[MASK]` at the tail
  attends to ~20 trailing `[MASK]`s — still a block-parallel read of a mostly-masked block, not the
  reference's exact `[tail + 1 MASK]`. The block-parallel-vs-sequential gap was moved from the driver
  into the forward, not closed.

Consequently **Step 6 is NOT ADJUDICABLE at parity** (gated on Step 5, the first hard failure): with 5A
open the engine's value logits are wrong, the grammar never sees `complete_tool_call`, and requests
over-generate to `max_new_tokens`, so a matched-20 battery is quality-meaningless and infeasibly slow.
**No engine wall-clock at real quality exists.** The only honest diffusion wall-clock remains the HF
stack (3.904 s/turn, ≈3.5× the K3 1.120 s/turn target, ≈5.3× stock-AR aggregate) — **not the engine.**
K3 remains **unadjudicable on the engine path.** No sunk-cost engine number was invented. **Remaining fix
(5A) is a scheduler / model-runner change**, not a driver change: drive a **variable single-`[MASK]`
forward width** (schedule `draft_len` spec tokens, not a fixed 32) so each probe forward is exactly
`[tail + 1 MASK]`. Details §0.B; full checklist §3.

---

## 0. GPU SMOKE GAUNTLET — RESULTS (2026-07-03, RTX 5090 / sm_120)

**Env pre-flight:** host ~30 G RAM / ~25 G avail; GPU free (gnome-shell only). Every torch/vLLM
process run one-at-a-time inside `systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=4G`
(the RAM cage killed exactly one host-RAM probe, never the session). `torch 2.11.0+cu130`,
`get_device_capability()=(12,0)` sm_120 confirmed in every smoke JSON. Editable vLLM from
`/home/mark/shared/vllm_p2_pr42406`.

**HEAD note:** the vLLM flare branch has advanced past this doc's §1 pin (`6482e1d`). Current
`qwen3_5-flare-modelstate` HEAD is **`22f660c`** — concurrent agents added `cd3ba35`
(zero-value-projection live tripwire) and `22f660c` (the M1-crux GDN read-only-denoise fix). The
qwen_diffusion fix committed alongside this update is **`1c69101`** (chat-template instruct prompts +
non-JIT NvFp4 MoE backend override in `scripts/p2_vllm_smoke.py`); other agents' uncommitted Group-4
audit-counter and harness diffs are preserved, none clobbered.

### Per-step verdict

**Step 1 — pin-venv sanity: PASS.** Editable vLLM confirmed, sm_120 detected, `VLLM_USE_V2_MODEL_RUNNER`
honored. CPU suites under the real torch: flare state-machine **21 passed** (doc said 17; concurrent
regression tests added), hybrid_clean **23 passed** (doc said 20), parity `--mode selftest` PASS,
`--mode ops-parity` **18/18** checks PASS, `--mode state-parity` PASS. (Bypassed a missing `tblib`
test-infra dep with `--noconftest`; no product code touched.)

**Step 2 — DiffusionGemma smoke on sm_120: PASS** (`logs/smoke_diffusiongemma.json`, `status=PASS`).
Coherent output `"Fast inference minimizes latency to provide real-time responses."` → **the
first-party dLLM decode path (canvas draft / per-seq-causal Triton / commit) runs correctly on this
card. R2 / K2 NOT triggered.** Two blockers fixed inline: (a) FlashInfer NvFp4-**MoE** JIT needs a
CUDA toolkit → forced a non-JIT MoE backend (`emulation`, reference-correct; `marlin` also works);
(b) the smoke fed a **raw prompt to an instruct model** → chat-template fix (raw prompt gave gibberish,
chat template gave the coherent sentence). Both fixes are in commit `1c69101`.

**Step 3 — Qwen3.5-9B under MRV2 (default + align+APC): PASS both** (`logs/smoke_qwen_default.json`,
`logs/smoke_qwen_align_apc.json`, both `status=PASS`). Both configs load and generate coherently
(`"Thinking Process:\n\n1.  **Analyze the Request"`); the align config trips **no** mamba-cache
assertion. → **#38041 (MRV2×GDN broken) is stale; #42406 (align-APC) holds. K1 NOT triggered.** Two
toolchain blockers fixed: (a) nvcc/CTK header skew — the cu13 wheel ships **nvcc 13.2** but cudart
headers report `CUDA_VERSION 13000` and cccl's `cuda_toolkit.h` hard-errors → bypassed with the
header's own sanctioned escape `NVCC_APPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK` (13.0↔13.2
ABI-compatible); (b) the FlashInfer **sampler** JIT failed to *link* (`-lcudart`/`-lcuda` absent: wheel
has `lib` not `lib64`, no `.so` dev symlink, no driver stub) → disabled it with
`VLLM_USE_FLASHINFER_SAMPLER=0` (the dense Qwen GDN path needs no FlashInfer, and native argmax is
better for step-5 byte-parity anyway). **Required env for the Qwen path:**
`CUDA_HOME=<venv>/lib/python3.12/site-packages/nvidia/cu13`,
`NVCC_APPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK`, `VLLM_USE_FLASHINFER_SAMPLER=0`, plus the
doc's `VLLM_USE_V2_MODEL_RUNNER=1` / `VLLM_ATTENTION_BACKEND=TRITON_ATTN`.

**Step 4 — read-only-denoise probe (M1 go/no-go): NOT A CLEAN GO — blocked by the checkpoint and by an
under-scoped restore, not by the substrate.** Probe: `scratchpad/step4_readonly_denoise_probe.py` —
boots the real FLARE engine in-process, monkeypatches the GDN snapshot/restore ops, and per-row
float32-fingerprints the whole conv/ssm cache around every denoise forward. Final run: 10 denoise
forwards, batch=1 (`scratchpad/step4f.log`).
- **Engine blocker fixed to even boot:** the FLARE path publishes the canvas into the spec-decode
  `draft_tokens` buffer, whose width = `num_speculative_tokens` = `diffusion_config.canvas_length`.
  With **no `diffusion_config`** (stock export ships none) the buffer is width-0 →
  `_finish_prefills` crashes (`shape mismatch: value [64] vs index [1,0]`, `qwen3_5_flare.py:742`).
  Passing `diffusion_config={canvas_length,…}` fixes it. **This is a real launcher gap:**
  `qwen35_9b_flare_hybrid_serve.sh` sets `VLLM_QWEN3_5_FLARE_BLOCK` but does **not** set
  `num_speculative_tokens`, so the launcher as-written cannot boot the FLARE path.
- **Findings (10/10 denoise forwards):** the restore makes the protected block-table boundary slot
  (**row 1**) bit-identical — the mechanism works — but every denoise forward mutates GDN rows
  **[1,2,3,4]** while `_denoise_state_rows` protects only **[1]**; rows **[2,3,4]** persist changed
  after restore (`max_rowsum_diff_after_restore = 6038.2`, nonzero on all 10 forwards). The probe's own
  bit-identical pass criterion therefore reports **STEP4: FAIL**. This empirically confirms the §2 crux
  worry ("snapshot must protect the exact physical rows the GDN kernel writes"): the current
  row-selection is **too narrow**.
- **Decisive A/B:** with `VLLM_QWEN3_5_FLARE_READONLY_DENOISE` **on** (`step4f.log`) vs **off**
  (`step4_off.log`) the committed outputs are **near-identical** (`"AustrAustr腔 Special … owl绘本征
  Sarah"` either way) → the leaked rows [2,3,4] do **not** determine the committed output on this
  checkpoint.
- **Why it's checkpoint-blocked:** the on-disk `qwen3.5-9b-fastdllm-b1000-vllm-bf16` is a **stock,
  non-diffusion-trained** export — it decodes gibberish through the canvas/denoise path regardless of
  the readonly flag. Step 4 cannot be a real quality go/no-go, and step 5's byte-parity cannot be
  meaningful, until a diffusion-trained export exists (with `canvas_length` a multiple of FLA_CHUNK 64).
  **Verdict: the sm_120 substrate is validated and the boundary-row protection mechanism works, but the
  restore's row scope must be widened to [1..4] and re-proven on a trained export before M1 can be
  called.** Not a K1/K3 kill — a fix-and-retest, gated on producing the trained checkpoint.

**Step 5 — turn byte-parity (engine vs HF): NOT RUN.** Gated behind a clean step-4 go per the §3
"do not proceed until the current step passes" rule. No `--mode turn` run was attempted; the R5
shifted-logit-capture divergence remains unmeasured.

**Step 6 — matched-20 battery / engine-vs-HF wall-clock: NOT RUN.** Consequently **there is no honest
engine-vs-HF wall-clock number** — the thesis KPI (s/turn, forwards-saved ratio) is still unmeasured on
the engine path. The only timings captured are single-prompt smoke latencies (e.g. Qwen default
`generate_seconds ≈ 0.51`, align+APC ≈ `0.48`), which are load/warm-up-dominated and **not** a valid
AR-vs-diffusion comparison.

### Remaining issues surfaced by the gauntlet (in addition to §2)
1. **Launcher cannot boot the FLARE path** — `qwen35_9b_flare_hybrid_serve.sh` must also set
   `num_speculative_tokens`/pass a `diffusion_config` with `canvas_length`, or `_finish_prefills`
   crashes at width-0. Real gap, blocks any real serve.
2. **Restore row scope is under-sized** — `_denoise_state_rows` protects only the boundary slot; the
   GDN kernel writes rows [1..4]. Widen the snapshot to the full written set and re-run the probe. (A/B
   says non-determinative on the stock export, but that is not evidence on a trained one.)
3. **No diffusion-trained vLLM export on disk** — the stock export decodes gibberish through the
   diffusion path, blocking steps 4-6 quality/parity/wall-clock gates. Produce a trained
   canvas_length-multiple-of-64 export before re-attempting M1. **→ RESOLVED in §0.R** (real export
   built); and **corrected**: the trained block is **32**, not a multiple of 64 — see §0.R.
4. **Toolchain workarounds must be baked into the launcher** — the cu13 CTK-skew flag and
   `VLLM_USE_FLASHINFER_SAMPLER=0` are required for the Qwen GDN path to build/run on this box.

---

## 0.R REAL-CHECKPOINT GAUNTLET — steps 4-6 re-run on the trained export (2026-07-03, RTX 5090)

The §0 gauntlet ran on the b1000 **smoke** export (stock, non-diffusion-trained → gibberish through the
canvas path). Steps 4-6 were re-run on the **real diffusion-trained export** to decide M1 for real.

### Export + block-config reconciliation (the pre-gauntlet decisions)
- **Real export produced:** `models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16` (19G, new dir; b1000 left
  untouched as tokenizer/AR-parity reference). It is the **RL-v2 adapter merged into init-materialized
  weights** (`W += (α/r)·B@A`, r16/α32/scale 2.0), mathematically identical to the promoted HF
  hybrid-clean eval's PEFT runtime application (`--base init --adapter …rl…v2/…step300 --no-merge`).
  The RL-v2 adapter continued FROM Run-1 so it **subsumes** Run-1; B@1000 (r8, attn-only) is a separate
  AR-parity lineage, not in the hybrid-clean delta. Export: `replacement_count=427`,
  `lora_merge_count=152` (24 GDN layers × 5 + 8 attn × 4 — **GDN in_proj/out_proj deltas merged too**,
  not just attention). Sanity gates PASS: (a) merge bit-exact vs `init + 2.0·(B@A)` (maxabs 0.0, incl.
  GDN); (b) one HF hybrid-clean episode is **coherent, not gibberish** — `exact_args=3/4`, valid/schema/
  sequence 4/4, right on the promoted 47/63 ≈ 74.6% rate. Full provenance/shas in the export's
  `conversion_manifest.json` (`real_diffusion_export_block_reconcile.md`).
- **Block size pinned `canvas_length = 32`** to match training (`bd_size=32`) and the winning HF eval
  (`set_block_size(model, 32)`) end-to-end. Engine default 64 would denoise two trained blocks per
  commit and break parity. **Stale-doc correction:** §0 said "engine default `_DEFAULT_BLOCK=32`" — the
  source is actually `_DEFAULT_BLOCK = _GDN_CHUNK = 64`. So `32 % 64 != 0` trips the engine's mid-chunk
  hazard (fp32 `chunk_states[:,-1]` boundary is a *partial* recurrent state) — an engine-side
  restore-scope concern to prove at the step-4 re-run, **not** a reason to change the trained block.
- **Launcher gap fixed:** a single `--diffusion-config '{"canvas_length":32,"max_denoising_steps":8}'`
  both sets the engine block AND (via `num_speculative_tokens` fallback) sizes the spec-decode
  `draft_tokens` buffer, fixing the width-0 `_finish_prefills` crash. `qwen35_9b_flare_hybrid_serve.sh`
  now defaults to the real export and emits the config only when the FLARE gate is on.

### Per-step verdict (real export)
| step | verdict | one-line |
|---|---|---|
| 4 — read-only-denoise probe (M1 go/no-go) | **PASS (core invariant)** | after widening restore scope, whole GDN cache **bit-identical** around every denoise forward — 0 leaks, 10/10, restore load-bearing. |
| 5 — turn byte-parity (engine vs HF) | **BLOCKED / FAIL** | 3 structural blockers persist on the real export (orphaned FSM / format mismatch / unwired turn-adapter). |
| 6 — matched-20 M2 battery / wall-clock | **BLOCKED / FAIL** | `run_turn` raises `EngineUnavailable`; canvas sampler off-paradigm + non-deterministic. **No engine-vs-HF wall-clock.** K3 unadjudicable on the engine. |

**Step 4 — PASS.** Pre-fix (committed `22f660c`, "protect slot [1] only") reproduced the leak on the
real export: `protected [1]` vs `kernel-written [1,2,3,4]`, rows [2,3,4] still changed 10/10,
`max_rowsum_diff = 5872.18`. Root cause (GPU-measured): this ModelState runs `num_spec == 0`, so the
pre-forward block-table view names only checkpoint slot [1] while the align **running-state** row
advances 1→2→3 across denoise sub-steps. **Fix (vLLM pin `af21dc8`):** snapshot in `prepare_attn`
(post-metadata, pre-forward) reading the ACTUAL `non_spec ∪ spec` state indices, protecting
`{non_spec ∪ spec} ∪ {block-table checkpoint slots}` widened by a per-anchor guard band
(`VLLM_QWEN3_5_FLARE_READONLY_BAND`, default 4). Re-run: `leak forwards = 0`, `max_rowsum_diff = 0.0`,
`changed [1,2,3,4] → [] (protected [0..5])`, `STEP4_READONLY_ON: PASS`. Live counters sane:
`read_calls=10 advance_calls=1 read_advance_ratio=10.0 residual_full_context_model_calls=0
block_size=32 route_verified=False`.
- **Caveat — A/B on/off NOT identical, and all outputs gibberish.** Committed-token sha differs across
  readonly OFF (`aa08de30…`) vs [1]-only (`f1a3a298…`) vs banded (`37509cc7…`) → the read-only scope IS
  load-bearing (changes committed tokens). But every output is gibberish because the **engine's served
  sampler is a canvas/random-renoise denoiser, a different algorithm than the trained `hybrid_clean`**.
  So "which scope is semantically correct" needs the HF `hybrid_clean` parity reference — which is
  blocked (step 5). Core M1 artifact (whole-cache bit-identical read-only denoise) PASSES; the
  A/B-on-meaningful-output check is N/A until the engine runs the trained decode paradigm.

**Step 5 — BLOCKED / FAIL** (three code-verified blockers, first found on the smoke export in
`p2_engine_parity_smoke_result.md`, **re-verified on the real export**):
- **A — engine has no hybrid_clean path.** `VLLM_QWEN3_5_FLARE=1` routes to `Qwen3_5FlareSampler`, a
  canvas/renoise **block** denoiser (random init → Gumbel sample → entropy accept / random renoise →
  commit whole block); the FSM (`vllm/v1/sample/hybrid_clean.py`) is imported by nothing on the serving
  path. It ignores `SamplingParams.temperature` and is **not even self-reproducible** — two identical
  greedy requests gave entirely different sequences (`greedy_deterministic: FALSE`). The reference is
  single-token greedy + grammar FSM with a `[MASK]` sentinel. Cannot byte-match by construction.
- **B — "same checkpoint" is undefined.** The vLLM export is stock
  `Qwen3_5ForConditionalGeneration` / `model_type=qwen3_5`, `auto_map=None`, `mask_token_id=None`,
  has `vision_config`. HF hybrid_clean needs the `Fast_dLLM_Qwen3_5` bridge + `mask_token_id` (248077)
  → **cannot load this export**; reference decode is undefined on it. A stock vLLM export always lacks
  the bridge/mask token, so this persists independent of training.
- **C — harness turn-adapter unwired.** `VllmFlareEngineAdapter.run_turn` locates the class then raises
  `EngineUnavailable` — no code path drives one shared turn and reads token-ids + block-boundary
  snapshots out of the engine.

**Step 6 — BLOCKED / FAIL, no engine wall-clock.** Requires the missing seam (blocker C); `run_turn`
raises `EngineUnavailable` and the canvas sampler is non-deterministic + off-paradigm, so `exact_args`,
`episode_exact`, TRUE forwards/turn and s/turn **cannot be produced on the engine**. The only honest
engine signal is the read/advance ratio ~10 (fewer-forwards mechanism live — substrate liveness, not
the KPI). Reference rows (matched-20, `runs/endgame_scoreboard`, NOT the engine):

| row | exact_args | episode_exact | valid | s/turn | fwd-or-tok/turn |
|---|---:|---:|---:|---:|---:|
| OUR HF hybrid-clean (v2) | 47/63 | 13/20 | 63/63 | **3.904** | 56.83 denoise fwd/turn |
| stock-bf16-AR-guided | 51/63 | 14/20 | 63/63 | **1.213** | 82.24 tok/turn |
| stock-AR aggregate | 124/247 | 33/80 | 247/247 | **0.741** | 49.06 tok/turn |

The only diffusion wall-clock is the **HF-stack** 3.904 s/turn (~3.5× the K3 1.120 target, ~5.3× stock
AR) — **not the engine**. **K3 cannot be adjudicated on the engine path** until steps 5-6 unblock. No
sunk-cost engine number was invented.

### What remains before the engine can be promoted (steps 5-6 need net-new engineering, not a re-run)
1. **Wire `hybrid_clean` onto the engine** — integrate `HybridCleanDecodePolicy` into
   `Qwen3_5FlareSampler`, or expose a forward-only logit seam so the shared `sample_hybrid_clean`
   driver sources +1-shifted block logits from the engine. Only then are both sides the same algorithm.
2. **Produce one dual-format checkpoint** byte-equal as both the `Fast_dLLM_Qwen3_5` bridge (with
   `mask_token_id`) and the vLLM `Qwen3_5ForConditionalGeneration` export.
3. **Implement the harness seam** `VllmFlareEngineAdapter.run_turn` + `snapshot_from_vllm_modelstate`
   so `--mode turn --engine vllm` drives one shared turn and emits the byte / state-snapshot report.
4. Then: matched-20 turn byte-parity (greedy, identical FSM, `projected_value_tokens_exact==0`) and the
   M2 wall-clock A/B vs guided-AR **re-baselined on the same pinned build** (R6 fairness).

### Artifacts (real gauntlet)
- vLLM pin `af21dc8` (branch `qwen3_5-flare-modelstate`) — read-only restore widening + guard band.
- `p2_engine_gauntlet_real_result.md` — steps 4-6 on the real export (this section's source).
- `real_diffusion_export_block_reconcile.md` — export build + `canvas_length=32` decision.
- `p2_engine_parity_smoke_result.md` — the smoke-export step-5 structural-blocker analysis (re-verified).
- `runs/p2_engine_gauntlet_real/` — `step4_real_probe.py`, `step4_real_on_default.json` (PASS, 0 leaks),
  `step4_real_on.json` (pre-fix FAIL), `step4_real_off.json`, `step4_measure/instrument.json`.

---

## 0.A POST-WIRING ACCEPTANCE — Steps 5-6 re-run after blockers A/B/C wired (2026-07-04, RTX 5090)

The §0.R gauntlet FAILed Steps 5-6 on three *unwired* structural blockers. Those three were then wired
and this acceptance re-ran Steps 5-6 against the wiring, on the real export
`models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16` (block/canvas 32), engine venv `.venv-vllm-p2-main`, each
GPU proc alone in the `systemd-run … MemoryMax=22G` cage. Source: `p2_engine_acceptance_result.md`.

### Wiring done (the three blockers)
- **Blocker A** (vLLM pin `qwen3_5-flare-modelstate` `e38a9ea`): `hybrid_clean` is now a **selectable
  engine decode mode** (`VLLM_QWEN3_5_FLARE_DECODE=hybrid_clean`) driving the FSM/greedy
  `HybridCleanBlockDecoder` — the previously-orphaned FSM is now invoked at both the block-decoder and
  sampler seams (the assertion the orphaned-FSM bug lacked).
- **Blocker B+C** (qwen_diffusion `ed479b3`): one **dual-loadable checkpoint** (HF-bridge loader over the
  vLLM export, so both the `Fast_dLLM_Qwen3_5` bridge and the vLLM export read the same weights) + the
  `VllmFlareEngineAdapter.run_turn` seam (the adapter now boots the real export and drives a short turn
  instead of raising `EngineUnavailable`).

### Verification (CPU wiring intact — PASS)
`pytest tests/v1/sample/test_hybrid_clean.py tests/v1/sample/test_hybrid_clean_flare_decode.py
tests/v1/worker/gpu/test_qwen3_5_flare_state_machine.py` → **61 passed** (23 hybrid_clean + 17
hybrid_clean_flare_decode + 21 flare state-machine). Editable vLLM confirmed; the `af21dc8`
read-only-denoise fix untouched; the `hybrid_clean` FSM now actually invoked on the decode path.

### Per-step verdict (post-wiring)
| step | verdict | one-line |
|---|---|---|
| pre — CPU wiring intact | **PASS** | 61/61 CPU tests green; no regression to the `af21dc8` machinery. |
| 5 — turn byte-parity (engine hybrid_clean vs HF) | **FAIL (BLOCKED)** | TWO independent GPU-confirmed hard failures: **5A** block-parallel engine logits vs sequential single-`[MASK]` reference ⇒ byte-parity impossible by construction, engine emits gibberish while HF bridge on identical weights is coherent (`" Paris"`); **5B** deterministic CUDA IMA at the first decode step on real-length (1041-tok) turns ⇒ the 3 parity turns cannot even be driven. |
| 6 — matched-20 M2 battery on the engine | **BLOCKED** | Gated on Step 5. Byte-parity fails and real turns crash, so `exact_args` / `episode_exact` / TRUE forwards-per-turn / s-per-turn **cannot be produced on the engine.** No honest engine wall-clock. |

**5A — algorithm divergence (proven from both decoders' source).** HF `sample_hybrid_clean` (sha
`a4c66751…`) decodes **sequentially, one `[MASK]` at a time**: append one mask, forward over
`[committed_clean_prefix, MASK]`, read the single last-position shifted logit — one forward per
non-forced value token; truly-forced structural tokens (`len(legal)==1`) are FSM bulk-committed with
**zero** forwards. So the logit for output position *k* is conditioned on the actual committed clean
tokens 0..k-1. The engine `Qwen3_5FlareSampler._hybrid_clean_step` instead runs **one denoise forward
over the whole 32-position canvas** and `HybridCleanBlockDecoder.decode_block` walks all positions;
positions 1..31 are conditioned on the **noisy canvas**, not the clean prefix. ⇒ For any block with >1
model-decoded token the per-position logits differ by construction. Empirically: engine `hybrid_clean`
on a working short prompt → gibberish (`"<tool_call>\n<function= .ер s ET …"`,
`engine_smoke_adapter_short_hybrid_clean.json`) with its zero-value-projection tripwire holding; the
HF-bridge forward on the **same** export → coherent top-1 `" Paris"` (`blockerB_hf_bridge_forward.json`).
Same weights, tokenizer, mask id (248077) — the difference is the decode algorithm. **Fix scope: NOT
small / NOT engine-side-trivial** — the engine must run the reference's sequential single-`[MASK]`
schedule (or expose a forward-only logit seam feeding the shared `sample_hybrid_clean` driver), which
also removes the block-parallel "fewer-forwards" mechanism for value tokens.

**5B — deterministic decode-at-scale CUDA IMA.** Turn-0/episode-0 (1041 tokens) prefills fine
(`num_computed_tokens=1041`) then faults at the first decode (1 real + 32 canvas draft tokens):
`torch.AcceleratorError: CUDA error: an illegal memory access was encountered`. Isolated across GPU
boots (`CUDA_LAUNCH_BLOCKING=1`, RAM cage):

| decode mode | read-only-denoise | mamba_block_size | prompt | result |
|---|---|---|---|---|
| — (short smoke, prior) | on | 1024 | 10 tok | **OK** |
| hybrid_clean | ON | 1024 | 1041 tok | **CRASH (IMA)** |
| canvas | **OFF** | 1024 | 1041 tok | **CRASH (IMA)** |
| canvas | on | **4096** (1041 in ONE mamba block) | 1041 tok | **CRASH (IMA)** |

⇒ The IMA is independent of decode mode, of the read-only snapshot (rules out `af21dc8`), and of the
mamba-block-1024 boundary. It lives in the **shared FLARE canvas/commit spec-decode DECODE forward over
a long multi-KV-block prefix**. Exact faulting kernel needs `compute-sanitizer` (absent) or a
`TORCH_USE_CUDA_DSA` rebuild — deferred. Engine-side, deterministic, **NOT small**.

### Step 6 — no engine wall-clock; the only honest wall-clock is the HF/stock reference
Step 6 **did not run on the engine** (blocked by Step 5): byte-parity fails and real turns crash, so no
`exact_args`, `episode_exact`, TRUE forwards/turn or s/turn exist on the engine. The only engine signal
is short-prompt substrate liveness (`read_advance_ratio ≈ 3.0`, `forced_grammar_tokens=5` FSM
bulk-commit with zero forwards, `zero_forward_rows=2`, `projected_value_tokens_exact=0`) — i.e. the
fewer-forwards + zero-value-projection mechanisms are live, but over **gibberish**. That is liveness,
not the KPI. **No sunk-cost engine number was invented.** The honest wall-clock table below is the
matched-20 reference (`runs/endgame_scoreboard`, **NOT the engine**):

| row | exact_args | episode_exact | valid | s/turn | fwd-or-tok/turn |
|---|---:|---:|---:|---:|---:|
| OUR HF hybrid-clean (v2) — diffusion, **not the engine** | 47/63 | 13/20 | 63/63 | **3.904** | 56.83 denoise fwd/turn |
| stock-bf16-AR-guided (same build) | 51/63 | 14/20 | 63/63 | **1.213** | 82.24 tok/turn |
| stock-AR aggregate | 124/247 | 33/80 | 247/247 | **0.741** | 49.06 tok/turn |

The winning HF row's forward-savings (56.83 fwd/turn vs stock-AR ~82 tok/turn) come **entirely from the
grammar-FSM bulk-commit of truly-forced structural tokens with zero forwards** — every value token is
still decoded sequentially with one forward. So the engine's block-parallel canvas is a **different
algorithm**, not a faithful accelerator of the reference; on this checkpoint it is quality-dead. A
byte-parity-and-quality engine path must run the sequential single-`[MASK]` value decode; its only
legitimate speed lever over guided-AR is the same FSM zero-forward bulk-commit, not block-parallel value
denoising. This should be reflected in the M-milestone plan.

### What remains before Steps 5-6 can pass (net-new engineering, not a re-run)
1. **Close the logit seam (5A):** drive the engine with the reference's sequential single-`[MASK]`
   forward schedule, or expose a forward-only block-logit seam feeding the shared `sample_hybrid_clean`
   driver, so both sides run the same algorithm. Precondition for any byte-parity.
2. **Fix the decode-at-scale IMA (5B):** localize with `compute-sanitizer` / device-side asserts
   (mode-, readonly-, mamba-block-independent; long multi-KV-block prefix, first decode), then fix in
   the FLARE canvas/commit spec-decode forward.
3. Then: 3-turn byte-parity (greedy, identical FSM, `projected_value_tokens_exact==0` both sides), then
   the matched-20 M2 A/B vs guided-AR **re-baselined on the same pinned build** (R6 fairness).

### Artifacts (acceptance) — `runs/p2_engine_acceptance/`
- `step5_ima_hybrid_clean_ro_on_mamba1024.log`, `step5_ima_canvas_ro_off_mamba1024.log`,
  `step5_ima_canvas_ro_on_mamba4096.{log,json}` — the three IMA isolation boots.
- `step5_scheduler_dump_at_crash.txt` — faulting decode step (`num_computed_tokens=1041`).
- `ima_mamba_block_probe.py` — parametric IMA isolation probe.
- (prior, `runs/p2_engine_gauntlet_real/`) `engine_smoke_adapter_short_hybrid_clean.json` (engine
  gibberish + live counters), `blockerB_hf_bridge_forward.json` (HF-bridge coherent `" Paris"`).
- vLLM pin `e38a9ea` (blocker A), qwen_diffusion `ed479b3` (blockers B+C), acceptance commit `589a0dd`.

---

## 0.B IMA-FIX + SEQUENTIAL-DECODE-REBUILD ACCEPTANCE — Steps 5-6 (2026-07-04, RTX 5090)

The §0.A acceptance left Steps 5-6 FAIL for two *unclosed* reasons: 5A (block-parallel vs sequential)
and 5B (decode-at-scale IMA). Since then **5B was root-caused and FIXED** (vLLM pin `1e32dcd`) and **the
sequential single-`[MASK]` decode was rebuilt on the engine** (vLLM pin `5e2fb53`, GAP-5A driver). This
section is the final acceptance re-run of Steps 5-6 on those fixes, real export
`models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16` (block/canvas 32, mamba 1024, align+APC), each GPU proc alone
in the `systemd-run … MemoryMax=22G` cage. Source: `p2_engine_acceptance_result.md` (superseding update);
qwen_diffusion commit `237fdcf`.

### Per-step verdict (post-5B-fix + 5A-rebuild)
| gap / step | verdict | one-line |
|---|---|---|
| 5B — decode-at-scale CUDA IMA | **FIXED** | align spec-decode state copy indexed non-existent speculative block-table columns; feed the align state machine a neutral `num_accepted==1`. Real turns now decode without faulting. |
| 5A — turn byte-parity (engine hybrid_clean vs HF) | **FAIL — algorithmic (not numeric)** | driver rebuilt correct (12-token forced-prefix byte-match) but the engine **forward** still reads the fixed 32-position spec-draft canvas instead of `[tail + 1 MASK]`; the probe MASK attends ~20 trailing MASKs ⇒ forward logits diverge whole-distribution at the first real choice. |
| 6 — matched-20 battery at parity | **NOT ADJUDICABLE** | gated on Step 5. 5B robustness demonstrated (engine now runs real-length turns without crashing); with 5A open the engine over-generates on wrong values, so no quality / wall-clock at parity exists. |

### 5B — root cause + fix (GPU-localized, verified)
The IMA is **not** at the first decode step (the §0.A guess): canvas mode decodes ~231 tokens then faults
at `num_computed=1272`; hybrid_clean at `1140`. `_flare_bounds_check` on every named slot/block/GDN index
tensor **passes** — the OOB is *inside a kernel*. An env-gated phase synchronize
(`VLLM_FLARE_SYNC_DEBUG=1`) pinned the last clean phase before the fault to
**`postprocess pre-super (align state copy)`** — i.e. `super().postprocess_state` (the MambaHybrid align
spec-decode state copy), firing regardless of read-only-denoise. An align-kernel input dump made it exact:
```
A=32 N=1074 src_idx=2 bs=528 needs_copy=True token_bias=13 dest_col=1
  src+bias=15  bt_stride(width)=8      <-- gather col 15 into a width-8 block table
```
- **Mechanism:** `postprocess_mamba_fused_kernel`'s temporal copy reads the accepted draft token's
  intermediate GDN state from block-table column `src_col + (num_accepted−1)`. That assumes `num_accepted−1`
  **speculative** checkpoint columns exist — allocated only when a real `speculative_config` sets
  `num_speculative_blocks`. FLARE drives the canvas as spec draft tokens **without** a `speculative_config`
  (`num_speculative_blocks == 0`), so the mamba block table has no such columns; a commit of `A` tokens
  crossing a mamba-block boundary indexes `src_col + (A−1)` (2+13=15) past the width-8 table ⇒ IMA.
- **Fix (`1e32dcd`, `Qwen3_5FlareModelState.postprocess_state`):** a FLARE commit is a single causal pass
  whose final GDN state already lives in the running block — there are **no** per-token intermediate states
  to select. `num_computed_tokens` is already advanced by `post_update` (consuming the real `num_sampled`)
  *before* `postprocess_state`, so `num_sampled` here feeds ONLY the num_accepted scatter. Feed the align
  state machine a neutral `num_accepted == 1` ⇒ the boundary migration is a plain running-block copy
  (`token_bias == 0`, in-bounds); the real commit count is retained only for the commit counter.
- **Verified:** canvas decode runs the full 300-token cap with zero IMA (was faulting ~231); the only
  `needs_copy=True` is the clean prefill boundary at `token_bias=0`. **66/66 CPU tests green** (no
  regression to the `af21dc8` read-only-denoise machinery).

### 5A — byte-parity FAIL, diagnosed ALGORITHMIC (top-5 logits, both sides)
With 5B fixed and the sequential single-`[MASK]` **driver** rebuilt (`5e2fb53`), both sides run the SAME
dual-loadable export (HF `Fast_dLLM` bridge over the vLLM export, blocker B; mask id **248077** passed to
the engine via `VLLM_QWEN3_5_FLARE_MASK`), and the engine adapter now wires tool schemas + `grammar_topk`
to the engine FSM via `SamplingParams.extra_args`. Reference produces coherent bounded tool calls
(ep0/turn0 42 tok `stop=complete_tool_call`). **Turn-0 (greedy, identical prompt/schemas/mask):** engine
matches the reference **token-for-token for the first 12 tokens, then diverges at position 12** and
degenerates. Those 12 are the tool-call scaffolding + tool name — all **grammar-forced**, so the match
proves the FSM wiring is live, NOT that the forward is correct. Pos-12 is the first logit-dependent choice:
```
decoded:  ref  "<tool_call>\n<function=initialize_qubits>\n<parameter=num_qubits>\n2\n</parameter>\n<"
          eng  "<tool_call>\n<function=initialize_qubits>\n<parameter=num_qubits>\n\n\n00\n\n..."
```
| token | text | REFERENCE (HF bridge) | ENGINE (vLLM) |
|---|---|---:|---:|
| 29 | `>` | 17.625 | **18.25 (argmax)** |
| 45334 | `=num` | **24.25 (argmax)** | 8.625 |
| 28 | `=` | 19.75 | 9.625 |
| 2334 | `num` | 12.5 | 16.125 |

This is **not** a bf16 near-tie flip (whole distribution differs) and **not** a +1-shift/position bug
(the shifted, raw-MASK, and last-clean probe positions all give the same wrong logit) — **the engine's
forward output itself is wrong.**
- **Root cause (algorithmic):** the `5e2fb53` rebuild fixed the *driver* (reads one probe logit, drives
  the chain-rule schedule) but the engine *forward* still processes the **fixed 32-position spec-draft
  canvas**. The scheduler sets `num_draft_tokens_per_req = num_spec == 32` (no variable-width spec
  schedule), so every probe forward runs over `[clean tail, MASK, MASK×(31−tail_len)]`. The FLARE denoise
  read is bidirectional, so the probe `[MASK]` at position `tail_len` attends to ~20 trailing `[MASK]`s —
  still a partial **block-parallel** read of a mostly-masked block, not the reference's exact
  `[tail + 1 MASK]`. **The block-parallel-vs-sequential gap was moved from the driver into the forward,
  not closed.**
- **Fix needed (NOT a driver change):** drive the diffusion decode with a **variable single-`[MASK]`
  forward width** — schedule `draft_len` spec tokens, not a fixed 32 — so each probe forward is exactly
  `[tail + 1 MASK]`. This is a **scheduler / model-runner** change (dynamic per-step spec-token count for
  the diffusion path), plumbed through `num_draft_tokens_per_req` / `num_spec_tokens_to_schedule` (the
  same lever standard spec-decode `dynamic_sd_lookup` uses). Until then byte-parity is impossible by
  construction.

### Step 6 — not adjudicable at parity; 5B robustness demonstrated; honest wall-clock unchanged
Per "stop at first hard failure," Step 5 is the first hard failure, so the matched-20 quality/wall-clock
battery is **not run at parity**. What IS newly true post-5B: the engine decodes **real-length turns
(prompts 1041/1443/917 tok; 300-tok canvas generations) without crashing** — the substrate is live at
scale. But with 5A open the engine's value logits are wrong, the grammar never observes a
`complete_tool_call`, and the request over-generates to `max_new_tokens` (grammar cost grows with
`committed`), so a full 63-turn battery is both quality-meaningless AND infeasibly slow. **No sunk-cost
engine KPI was invented.** The only honest diffusion wall-clock remains the HF/stock reference below
(`runs/endgame_scoreboard`, **NOT the engine** — there is still no engine row):

| row | exact_args | episode_exact | valid | s/turn | fwd-or-tok/turn |
|---|---:|---:|---:|---:|---:|
| OUR HF hybrid-clean (v2) — diffusion, **not the engine** | 47/63 | 13/20 | 63/63 | **3.904** | 56.83 denoise fwd/turn |
| stock-bf16-AR-guided (same build) | 51/63 | 14/20 | 63/63 | **1.213** | 82.24 tok/turn |
| stock-AR aggregate | 124/247 | 33/80 | 247/247 | **0.741** | 49.06 tok/turn |

**K3 cannot be adjudicated on the engine path.** The HF row's forward-savings (56.83 fwd/turn vs stock-AR
~82 tok/turn) come **entirely from grammar-FSM bulk-commit of truly-forced structural tokens with zero
forwards** — every value token is still decoded sequentially, one forward each. So the engine's
block-parallel canvas is a **different algorithm**, not a faithful accelerator; its only legitimate speed
lever over guided-AR is the same FSM zero-forward bulk-commit, not block-parallel value denoising. Reflect
in the M-milestone plan.

### Artifacts (this acceptance) — `runs/p2_engine_acceptance/`
- `byte_parity_2proc.py` — two-process byte-parity driver (one 9B per process; reference vs engine on
  identical prompt/schemas/mask; token + byte + top-k-logit diff at first divergence).
- `ima_mamba_block_probe.py`, `step5_scheduler_dump_at_crash.txt`, and the three prior §0.A IMA isolation
  boots (now superseded by the 5B fix).
- vLLM pins: `5e2fb53` (sequential single-`[MASK]` driver, GAP 5A), `1e32dcd` (5B IMA fix +
  `VLLM_QWEN3_5_FLARE_MASK` + `VLLM_FLARE_SYNC_DEBUG`). qwen_diffusion `237fdcf` (adapter passes tool
  schemas via `extra_args`; this doc). Repro: engine env of §0.A + `VLLM_FLARE_SYNC_DEBUG=1` for the
  decode-fault phase, `VLLM_FLARE_BOUNDS_CHECK=1` for the (passing) index-tensor checks.

---

## 1. What was built (paths + local commits)

### Repo A — vLLM pin `/home/mark/shared/vllm_p2_pr42406`
Editable-installed into `/home/mark/qwen_diffusion/.venv-vllm-p2-main`. Upstream base pinned at
`2665ed7` (PR #46838, i.e. at/after the MRV2 align-APC PR #42406 merge). **Not pushed** (no upstream
on the branch). Branch layout:
- `main` → `2665ed7` (clean upstream pin)
- `hybrid-clean-decode-policy` → `397fc98`
- `qwen3_5-flare-modelstate` → `6482e1d` (current HEAD; contains everything below)

Local commits on the flare branch:
| commit | what |
|---|---|
| `397fc98` | `[v1][sample]` Add hybrid-clean tool-call decode policy |
| `edb4d05` | `[v1][diffusion]` Qwen3_5FlareModelState: GDN-hybrid block-diffusion serving |
| `3ff71a8` | `[v1][diffusion]` Fix FLARE read_calls double-count + mamba-block-size in denoise-row derivation |
| `6482e1d` | `[v1][sample]` hybrid_clean: suppress [MASK] sentinel on value/structural tokens |

Files:
- `vllm/v1/worker/gpu/model_states/qwen3_5_flare_ops.py` — pure torch-only state-machine primitives
  (import-light, CPU-testable): `right_shift_block_logits`/`capture_shift_logit`,
  `per_seq_causal_flags`, `flare_step_and_phase`/`flare_commit_num_sampled`, `commit_num_accepted`,
  `snapshot_readonly_rows`/`restore_readonly_rows`, `FlareBoundarySnapshot` + `assert_fp32_boundary`
  + `tail_after_append`.
- `vllm/v1/worker/gpu/model_states/qwen3_5_flare.py` — `Qwen3_5FlareModelState(MambaHybridModelState)`
  + `Qwen3_5FlareRequestStates` + `Qwen3_5FlareSampler` (canvas denoise/commit). Subclasses
  MambaHybrid to inherit align-APC pre/postcopy + `num_accepted_tokens` scatter + GDN attn metadata;
  grafts the DiffusionGemma canvas/commit path.
- `vllm/model_executor/models/qwen3_5.py` — registration: `get_model_state_cls()` returns the FLARE
  state under env `VLLM_QWEN3_5_FLARE=1`, else `MambaHybridModelState`.
- `vllm/v1/sample/hybrid_clean.py` — the standalone tool-call decode-policy REFERENCE (FSM,
  forced-token bulk-commit, value/structural split, audit counters), ported verbatim from
  `scripts/eval_fastdllm_toolcall_cases.py` + `diagnose_toolcall_json_completability.py` +
  `sample_hybrid_clean`. **See §2: this is NOT on the serving path.**
- Tests: `tests/v1/worker/gpu/test_qwen3_5_flare_state_machine.py` (17 pure-CPU ops tests),
  `tests/v1/sample/test_hybrid_clean.py` (20 pure-CPU tests incl. the [MASK]-sentinel regression).

### Repo B — qwen_diffusion `/home/mark/qwen_diffusion`
Remote `origin` = `MaCoredroid/qwen-diffusion-agentic`. On `main`, **2 commits ahead of origin/main**
(the harness base `ddfa895` is pushed; the audit-counter fixes are not):
| commit | pushed? | what |
|---|---|---|
| `ddfa895` | yes | P2 parity+audit harness: HF hybrid-clean reference vs new FLARE engine |
| `782b441` | **no** | P2 parity harness: close audit-counter tautology + gate FSM value-projection leak channel |
| `1e73790` | **no** | Add REPRODUCE_V2 (dual-mode AR+diffusion recipe; adjacent) |

Files:
- `scripts/parity_audit_flare_engine.py` — the M1/M2 promotion-gate instrument. Four modes:
  `selftest` (15/15, tokenizer-only), `ops-parity` (15/15, imports the REAL engine ops and proves
  byte/numeric equivalence to `flare_hf_cache`), `state-parity` (tiny route_i model, 4 gates), `turn`
  (GPU; one matched-20 turn, HF reference vs vLLM engine, 6 gates).
- `scripts/p2_vllm_smoke.py` — **untracked** smoke driver with three cases (`diffusiongemma`,
  `qwen-default`, `qwen-align-apc`); drives the §3 steps 2-3. Will be committed alongside this doc.

### Repo C — flywheel fork `/home/mark/shared/lumoFlyWheel_codex_fork`
Remote `fork` = `MaCoredroid/Lumo_FlyWheel-qwen-diffusion`. HEAD `b91184d0`. **Not pushed.**
| commit | what |
|---|---|
| `00b72352` | Wire P2 hybrid-diffusion serving surface for Qwen3.5-9B |
| `b91184d0` | flare-hybrid launcher: set the real engine gate `VLLM_QWEN3_5_FLARE` |

Files:
- `scripts/qwen35_9b_flare_hybrid_serve.sh` — host vLLM launcher pointed at the pin venv
  `.venv-vllm-p2-main/bin/vllm`, serving `models/qwen3.5-9b-fastdllm-b1000-vllm-bf16` as
  `qwen3.5-9b-flare-hybrid-clean`. Now derives+exports `VLLM_QWEN3_5_FLARE` from `DECODE_POLICY`.
- `model_registry.yaml` — entry `qwen3.5-9b-flare-hybrid-clean` (validated through `load_registry`).
- `docs/qwen3.5-9b-flare-hybrid-serving-note.md` — MTP coexistence + shared align prefix cache note.

---

## 2. Review verdicts + open issues

**All four reviews: `fix-needed`.** What was fixed in-loop, then the open issues that remain — the
GPU-only-unvalidatable crux, the architectural gaps, and the perf/util debt.

### Fixed in-loop (already committed)
- **read_calls double-count** + **mamba-block-size in denoise-row derivation** (`3ff71a8`): non-align
  branch was indexing the MAMBA block table with the attention `block_size`, pointing snapshot/restore
  at the wrong state rows. Now uses `mamba_spec.block_size`. read_calls no longer counted in the
  sampler's `_apply_shift`.
- **[MASK]-sentinel suppression** in hybrid_clean (`6482e1d`): reference decoder unconditionally does
  `logits[:, mask_id] = -inf` every forward; the port had dropped it, so a parameter VALUE token could
  argmax to the mask id and diverge. Restored + regression test (20/20).
- **Gate-#3 tautology** in the parity harness (`782b441`): `two_wave_wave1_projected_tokens` /
  `parallel_commit_forced_tokens` were hard-coded to 0, so the "ZERO projected values" gate could
  never fail on a projecting engine. Real counters now plumbed through; added `grammar_value_projection`
  gate + 5 selftest regression guards (15/15), incl. one that would have wrongly passed before the fix.
- **Launcher silent-AR gate mismatch** (`b91184d0`): launcher exported only `FLARE_DECODE_POLICY`
  (zero consumers in the pin) → it silently served the AR `MambaHybridModelState` under the diffusion
  model name. Now derives+exports `VLLM_QWEN3_5_FLARE`.

### Open — THE CRUX (GPU-only, unvalidatable off-GPU; this IS the M1 go/no-go)
- **GDN read-only-denoise state discipline** (`qwen3_5_flare.py` `_denoise_state_rows`/`_gdn_caches`,
  ~L389-481). Flagged by **all four** reviews. Correctness of the read-only-denoise restore hinges on
  this reconstructing EXACTLY the physical conv+ssm rows the GDN kernel writes:
  - It reconstructs rows independently (`_mamba_state_idx_gpu[denoise_slots]`) rather than reading the
    gdn_attn backend's actual `non_spec/spec_state_indices_tensor`. If the running-row vs
    block-table-slot or spec-decode index layout diverges, the snapshot protects the WRONG rows and
    denoise permanently corrupts the boundary `S_t`/conv state = the fatal, silent failure.
  - `_gdn_caches()` classifies ANY module whose `.kv_cache` is a 2-tuple of nonempty tensors as GDN.
    A full-attention layer whose kv_cache is also 2-length would be misclassified and its K/V
    snapshot/restored as conv/ssm (silent). No assertion pins layer count/identity/shape.
  - **Silent-fatal path:** `preprocess_state` does `if not caches: return` (no snapshot) and
    postprocess then skips restore, so an in-place kernel write-back of tentative denoise tokens leaks
    into `S_t`/conv with NO error. Should hard-fail when readonly is enabled AND denoise rows exist AND
    caches is empty.
  - **Postprocess ordering:** restore runs FIRST, then `super().postprocess_state` scatters
    `num_accepted=max(num_sampled,1)=1` and runs the align block-row publish over ALL rows including
    denoise. "Denoise advances GDN state by 0" relies on `num_computed` not advancing so the align copy
    is a no-op; if a denoise row is ever non-block-aligned the align kernel re-copies and clobbers the
    just-restored boundary.
  - NOT proven by the 17 CPU tests (which cover only the unused pure ops). This is exactly the M1
    day-1-2 read-only-denoise probe (§3 step 4).

### Open — architectural / semantic-drift (turn-parity risk)
- **Orphaned FSM** (reviews 2 & 4): `vllm/v1/sample/hybrid_clean.py` is imported by NOTHING in `vllm/`
  (grep-confirmed) and referenced nowhere in the FLARE ModelState/Sampler. The served path is
  `Qwen3_5FlareModelState.custom_sampler → Qwen3_5FlareSampler`, a plain canvas denoiser with zero
  grammar/tool/FSM/value logic. `decode_policy=hybrid_clean` is a name collision between two unrelated
  mechanisms (hybrid_clean = HF masked-diffusion with a `[MASK]` token; FLARE = canvas/random-renoise,
  no mask token — cannot be trivially merged). Every value-projection / FSM / "zero value projection"
  guarantee the serving note advertises is **off the actual serving path**. Wiring
  `parse_hybrid_clean_request`/`HybridCleanDecodePolicy` into a real decode scheduler is unimplemented.
- **Shifted-logit capture uses the wrong stream** (reviews 1, 2, 3): engine `capture_shift_logit`
  (`qwen3_5_flare.py` ~L762-773) captures `block_logits[:,-1:]` from the converging DENOISE step (a
  forward over the pre-freeze/renoised canvas), whereas the reference `flare_hf_cache.advance()`
  re-runs `cached_noisy_block_logits` over the COMMITTED (argmax) clean block. The carried
  position-0 logit can drift. The +1 right-shift DIRECTION itself is correct. state-parity/ops-parity
  cannot detect this off-GPU — only `--mode turn` on the real engine can. This is R5 semantic drift:
  regression to the 0/41 corruption regime is silent without the gate.
- **Bidirectional-denoise-on-GDN is the unproven premise** (review 1): per-seq `causal=False` only
  affects full-attention layers; GDN linear-attn ignores causal entirely — its "bidirectional block
  read" is realized ONLY by the snapshot/restore recurrence. Whether that reproduces training's
  bidirectional-block GDN semantics is the novel unvalidated claim. `rswa_prefix_lens` passed together
  with per-seq `causal=False` is also unvalidated.
- **Harness self-contradictions vs the real engine** (review 3): `compare_snapshot_sequences` pass
  condition requires `not only_ref and not only_eng`, but the reference records prefill boundaries
  (fresh FlarePrefixCache each turn) while the real engine exposes commit-only boundaries, so
  `state_snapshot_equality` will ALWAYS fail via `only_ref`. Gate #2's `reported_model_value_tokens`
  clause: the engine has no such counter, so a byte-identical engine yields 0 vs N and gate #2
  spuriously FAILS. Both are harness design decisions to resolve on-GPU (see §3 step 5 kill note).

### Open — audit theater (counters that can't catch a regression)
- `force_projected_value_tokens` (`# must stay 0`) and `residual_full_context_model_calls`: initialized
  to 0, NEVER incremented, NEVER asserted → dead.
- `advance_calls` increments on EVERY tensor postprocess (denoise AND commit), not commit-only. Since
  denoise dominates, read_calls ≈ advance_calls, so the read/advance ratio — **the ~13x-fewer-forwards
  ⇒ wall-clock-win thesis metric and the M2 gate** — collapses to ~1. Left unfixed: the commit signal
  is only cheaply available in the sampler, not `postprocess_state`; needs a design decision.
- `hybrid_clean.py` `value_projection_events` is never incremented on the live path, so
  `verify_invariants()`'s `assert value_projection_events==0` is tautological. The other two invariants
  are guaranteed by the loop structure. Reporting "0 value projections" as evidence is theater.
- **Dead ops vs commit-message claims:** `commit_num_accepted`, `FlareBoundarySnapshot`,
  `assert_fp32_boundary`, `tail_after_append` are defined + unit-tested but called NOWHERE in the live
  path (0 uses each). The fp32-boundary + raw-conv-tail publish is delegated wholesale to inherited
  align `postprocess_state` + `--mamba-ssm-cache-dtype float32`; the load-bearing integration is
  untested and the "17/17" covers only the unused pure ops.
- `route_i` guard is declarative-only: stock Qwen3.5 never exposes `flare_gdn_route`, so `_assert_route`
  always falls to the env default and the GROUND invariant is never checked against served semantics.

### Open — perf / GPU-utilization debt (violates the standing util rule)
- `Qwen3_5FlareSampler._gather_block_logits` (L639) and `_apply_shift` (L656) run per-decode-row Python
  loops with `.tolist()`/`bool(...)` host syncs + per-call `async_copy_to_gpu` allocations on the hot
  path every step. The proven DiffusionGemma path vectorizes the identical gather/pad sync-free
  (`diffusion_gemma.py` L1269-1274). Caps GPU util under batching; a CUDA-graph blocker. Not
  incorrectness — deferred to P2.2+/M3 (§4).
- If hybrid_clean were ever wired: `HybridCleanGrammar._keeps_prefix` does a full `tokenizer.decode` of
  the ENTIRE prefix for EVERY candidate EVERY step (O(prefix × candidates)) — not viable at serving
  latency.

### Open — config / docs hazards
- **Block/chunk misalignment:** engine default `_DEFAULT_BLOCK=32` is HALF a GDN chunk (FLA_CHUNK 64).
  A stock export has no `diffusion_config`, so 32 is what runs; commit boundaries land mid-chunk on
  every other block, and the fp32 boundary snapshot `chunk_states[:,-1]` mid-chunk is not a clean
  recurrent checkpoint. Trained `canvas_length` must be a multiple of 64; set
  `VLLM_QWEN3_5_FLARE_BLOCK` accordingly.
- **Per-request mode switching not wired:** registration is process-global via `VLLM_QWEN3_5_FLARE=1`
  in `get_model_state_cls`, not per-request `extra_args["decode_mode"]` (fr10). AR and block-diffusion
  cannot coexist in one server; every request is forced through the diffusion ModelState.
- **Stale docs:** serving note says `FLARE_DECODE_POLICY` selects the sampler; the real selector is
  `VLLM_QWEN3_5_FLARE`. The note also conflates KV `block_size` (~16) with `mamba_block_size` (1024) in
  the `max-num-batched-tokens` reasoning.

### Verified-clean (for the record)
- The **+1 right-shift is mathematically correct** (`right_shift_block_logits`/`capture_shift_logit`
  mirror the reference `shifted_active_logits`/`advance`; block_start==0 self-prepends noisy[:,:1]).
  Confirmed by 17 CPU state-machine tests + 20 hybrid_clean tests (all pass).
- All vLLM-pin CLI/env used by the launcher are valid in the pinned build (`--attention-backend
  TRITON_ATTN`, `--gdn-prefill-backend triton`, `--mamba-cache-mode align`, `--mamba-block-size`,
  `--mamba-ssm-cache-dtype float32`, `VLLM_USE_V2_MODEL_RUNNER`, `SamplingParams.extra_args`,
  `get_model_state_cls`). No API misuse found.
- **Caveat on test coverage:** torch was NOT installed in the review environment, so the 17 FLARE
  state-machine tests and ALL GPU-path behavior could not be executed there — only the 20 pure-Python
  hybrid_clean tests ran. FLARE ops correctness currently rests on code reading. Re-run under the pin
  venv on the GPU box (§3 step 1).

---

## 3. THE GPU SMOKE CHECKLIST (run in order, the moment the GPU frees)

This is the M1 day-1-2 fail-fast gauntlet, front-loaded exactly as `p2_serving_reuse_plan.md` §4
prescribes, extended through the M1 turn gate (step 5) and the M2 matched-20 battery (step 6). Each
step lists its **pass criterion** and its **kill criterion** (from plan §5). Do not proceed to the
next step until the current one passes.

Environment for every step: `VLLM_USE_V2_MODEL_RUNNER=1`, `VLLM_ATTENTION_BACKEND=TRITON_ATTN`,
python/vllm from `/home/mark/qwen_diffusion/.venv-vllm-p2-main`.

### Step 1 — pin-venv sanity (precursor)
- **Do:** confirm `.venv-vllm-p2-main/bin/vllm` imports vLLM from `/home/mark/shared/vllm_p2_pr42406`
  (editable), the flare branch is checked out (HEAD `6482e1d`), `torch.cuda.get_device_capability()`
  reports **sm_120 = [12, 0]**, and `VLLM_USE_V2_MODEL_RUNNER` is honored. Then re-run the CPU test
  suites under this venv (they could not run in the review env — no torch there):
  `pytest tests/v1/worker/gpu/test_qwen3_5_flare_state_machine.py` (17) and
  `pytest tests/v1/sample/test_hybrid_clean.py` (20), plus
  `scripts/parity_audit_flare_engine.py --mode selftest` (15), `--mode ops-parity` (15),
  `--mode state-parity` (4 gates).
- **Pass:** editable install confirmed, sm_120 detected, all 17+20 unit tests + selftest/ops-parity/
  state-parity green under the real torch.
- **Kill:** none — this is setup. If it fails, fix before proceeding; do NOT burn the fail-fast budget
  on a broken venv.

### Step 2 — DiffusionGemma smoke on sm_120  (plan §4 M1 (a))
- **Do:** `python scripts/p2_vllm_smoke.py diffusiongemma --out logs/smoke_diffusiongemma.json`
  (NVFP4 DiffusionGemma-26B, `quantization=modelopt`, `attention_config.backend=TRITON_ATTN`,
  `diffusion_config` canvas 32 / 4 steps).
- **Pass:** loads + generates a coherent short sentence, `status=PASS`. Proves the first-party dLLM
  decode path (canvas draft tokens, per-seq causal Triton kernel, commit machinery) runs on THIS card.
- **Kill K2 (R2 sm_120 attention):** if the DiffusionGemma smoke fails after **2 days of backend
  fallbacks** → drop to **A2 (0.23 V1-runner hook injection)**; its bidirectional mask would be our
  Triton kernel either way. Carry the ModelState design as our own seam spec.

### Step 3 — our export under MRV2, default then align+APC  (plan §4 M1 (b))
- **Do:** `python scripts/p2_vllm_smoke.py qwen-default --out logs/smoke_qwen_default.json` then
  `python scripts/p2_vllm_smoke.py qwen-align-apc --out logs/smoke_qwen_align_apc.json`
  (`models/qwen3.5-9b-fastdllm-b1000-vllm-bf16`; align case adds `enable_prefix_caching`,
  `mamba_cache_mode=align`, `mamba_block_size=1024`, `mamba_ssm_cache_dtype=float32`). Tests whether
  #38041 (MRV2×GDN broken) is stale and whether #42406 (align-APC for mamba) holds.
- **Pass:** Qwen3.5-9B forwards under MRV2 in BOTH configs, coherent generation, `status=PASS` both;
  align+APC config loads without the mamba-cache assertion tripping.
- **Kill K1 (R1 MRV2×GDN broken):** if Qwen3.5 cannot forward under MRV2 within **5 working days** of
  fixes/upstream triage → drop to **A2 (0.23 V1 injection)**, carrying the ModelState design as our
  seam spec.

### Step 4 — read-only-denoise probe  (plan §4 M1 (c) — the go/no-go artifact)
- **Do:** the ~20-line probe: forward the SAME block twice from a fixed `initial_state` with the
  denoise-phase flag on (`inplace_final_state=False` + `_denoise_state_rows` snapshot/restore + conv
  write suppression), diffing the conv/ssm slots and the logits between the two forwards. This is the
  live-fire test of the §2 GDN-state-discipline crux — it validates `_gdn_caches` classification (no
  full-attn layer misclassified), that the snapshot protects the exact physical conv+ssm rows the fla
  kernel writes, and that denoise leaks NOTHING into the boundary `S_t`/conv.
- **Pass (go):** for denoise rows, `conv_state` and `ssm_state` slots are **bit-identical** before vs
  after the denoise forward, and the two forwards produce **identical logits** — i.e. denoise advances
  GDN state by exactly 0. Emit the go/no-go artifact (diff report).
- **Kill:** this probe is the M1 go/no-go and a **precondition of the K3 thesis-level gate** ("with
  read-only O(block) denoise verified"). If denoise corrupts `S_t`/conv and it cannot be fixed within
  the R1 window → the MRV2×GDN read-only-denoise premise is unworkable on this substrate → drop to
  **A2 (K1)**. Also gate against **R5**: if the snapshot protects the wrong rows, the fix must be
  proven here, not deferred — this failure is silent downstream.

### Step 5 — parity harness turn: engine vs HF byte-match  (plan §4 M1 gate)
- **Do:** `python scripts/parity_audit_flare_engine.py --mode turn --engine vllm` on one matched-eval
  turn (temp 0, native chat_template, greedy per-call waves, FSM stub = leftmost-forced only). Boots
  the vLLM V2 runner with `VLLM_QWEN3_5_FLARE=1` and drives (a) the HF hybrid-clean reference and
  (b) the FLARE engine over the same turn.
- **Pass:** all six turn-mode gates green — `byte_identical` (token AND byte exact vs the HF
  reference), `value_token_counts_equal`, `reference_zero_projected_values`,
  `engine_zero_projected_values`, `no_grammar_value_projection`, `state_snapshot_equality`. Plus the
  plan M1 targets: forwards/turn ≈ **6-9** and s/turn already **< HF's 1.442**. This is where the
  §2 shifted-logit-capture divergence surfaces if real — off-GPU modes cannot detect it.
- **Harness caveats to resolve here (review 3, design decisions, not silent relaxations):**
  `state_snapshot_equality` will fail via `only_ref` because the reference records prefill boundaries
  while the real engine exposes commit-only boundaries — decide whether to compare the shared set only.
  Gate #2's `reported_model_value_tokens` clause spuriously fails a byte-identical engine that lacks
  the counter — either emit `model_value_tokens` from the engine adapter or drop that redundant clause
  (the XML-derived `value_token_count` already covers it).
- **Kill (R5 semantic drift):** if turn output cannot be made byte-identical to the HF reference, the
  port has drifted (shifted-logit capture or conv_tail seam) — equivalence-gate every stage against
  `flare_hf_cache.py`, assert value force-counters 0. Regression to the 0/41 corruption regime is
  silent without this gate. If unfixable → re-scope; do not advance to step 6.

### Step 6 — matched-20 battery on the engine path  (plan §4 M2 gate)
- **Do:** rerun the hashed slice (`baf90863`, 20 ep / 63 turns) on the FLARE engine path with the full
  offline FSM + wave-1/wave-2 wiring + cross-turn APC counters, **against guided-AR re-baselined on the
  SAME pinned build** (identical align-APC flags — R6 fairness). Trigger-test the align-APC
  pathologies (#40696 / #45238 / #43587) on our multi-turn prompt shapes; apply the chunked lm_head cap.
- **Pass (M2 engine-promotion gate, all of):** the **engine quality gate is PARITY with the HF
  hybrid-clean row** — the engine runs the same weights + same algorithm, and Step-5 byte-parity
  *implies* the same score, so the target is **ENGINE == HF row (47/63 exact-args, 13/20 episodes,
  63/63 exact_seq, 63/63 valid_xml, value force-counters == 0)**, NOT a higher model-quality number.
  Plus the speed target **< 1.120 s/turn** and **force-counters == 0 on values**. (Prerequisite: fix
  the `advance_calls` counter so the read/advance ratio is real before quoting the forwards-saved
  metric.) **Note:** the **55/63 / 15/20** figure is the *K3 thesis aspiration* for the diffusion
  model's raw quality — a stronger, model-training target that gates the overall thesis, **not** the
  engine's parity gate. Promoting the engine only requires reproducing the promoted HF row byte-for-byte
  at `< 1.120 s/turn`; lifting the model to 55/63 is a separate, training-side milestone.
- **Kill K3 (thesis-level):** if at M2 — with read-only O(block) denoise verified (step 4) and, after
  M3, graphs on — diffusion still **misses 1.120 s/turn by > 20% at healthy GPU util**, the
  ~13x-fewer-forwards ⇒ wall-clock-win thesis fails on this hardware. **Stop, publish the profile,
  re-scope** (kernel-level work, or accept the quality-only win). No sunk-cost continuation past K3.

---

## 4. What remains for P2.2+ (after M1 passes)

### Wall-clock measurement vs re-baselined guided-AR (R6 fairness — do this before any speed claim)
Moving engines (0.23 → post-0.24 main pin) **invalidates the existing 1.120 s/turn number**. Before
claiming a diffusion win, re-baseline **guided-AR on the pinned build** with the identical align-APC
flag set, same hashed slice, same engine. The quality caveat stands (N=20, single seed, synthetic tool
results). Only a same-engine A/B is admissible under the promotion discipline.

### Batching + engine-grade per-forward cost (plan §4 M3)
- **Remove the host-bound hot path** (standing GPU-util rule): vectorize `_gather_block_logits` /
  `_apply_shift` sync-free like `diffusion_gemma.py` L1269-1274 — eliminate the per-row `.tolist()` /
  `bool(...)` syncs and per-call `async_copy_to_gpu` allocations. These are CUDA-graph blockers.
- **CUDA-graph capture:** we currently run `--enforce-eager` — this is where the remaining headroom
  lives. dInfer recipe: control-flow-free wave logic (no `.item()`/`tolist()`), shape-bucketed graph
  capture bs=1 first, then multi-seq waves via UNIFORM_BATCH-style padding.
- **Gate:** GPU util healthy per the standing rule (profile, no host-bound stalls); target the honest
  ~2-3x blended band vs AR at held quality.

### Engineering debt to clear at/before P2.2 (from the reviews)
1. **Wire the FSM onto the serving path** (or explicitly drop the "hybrid_clean on the engine" framing):
   today the value-projection / FSM / "zero value projection" guarantees live only in the standalone
   `hybrid_clean.py` reference, not in `Qwen3_5FlareSampler`. Either integrate
   `HybridCleanDecodePolicy` into the FLARE custom_sampler or reconcile the two diffusion paradigms
   (masked-diffusion vs canvas/renoise).
2. **Fix `advance_calls`** to increment commit-only (needs the commit signal surfaced from the sampler)
   so the forwards-saved metric — the whole thesis KPI and the M2 gate — is real.
3. **Wire or delete the dead audit machinery:** `force_projected_value_tokens`,
   `residual_full_context_model_calls`, `commit_num_accepted`, `FlareBoundarySnapshot`,
   `assert_fp32_boundary`, `tail_after_append`. If the fp32-boundary/conv-tail publish is truly
   delegated to inherited align postprocess + `--mamba-ssm-cache-dtype float32`, prove that integration
   (it is currently untested) or route it through the primitives.
4. **Hard-fail the silent-fatal GDN paths:** assert layer count/identity/shape in `_gdn_caches`; error
   (not `return`) when readonly is enabled with denoise rows but no caches found.
5. **Enforce block/chunk alignment:** engine default block 32 → multiple of FLA_CHUNK 64; set/validate
   `VLLM_QWEN3_5_FLARE_BLOCK`; ensure trained `canvas_length` is a multiple of 64 so boundary snapshots
   land on clean recurrent checkpoints.
6. **Per-request mode switching:** honor `extra_args["decode_mode"]` so AR and block-diffusion coexist
   in one server, instead of the process-global `VLLM_QWEN3_5_FLARE=1` switch.
7. **Fix stale docs** in `docs/qwen3.5-9b-flare-hybrid-serving-note.md` (`FLARE_DECODE_POLICY` →
   `VLLM_QWEN3_5_FLARE`; the KV-block vs mamba-block conflation).

### Push discipline
Nothing is pushed to the shared forks yet. Per the standing commit workflow, once step 5 (M1 turn
gate) passes on-GPU, push the flare branch to the vLLM pin fork and the serving surface to the flywheel
fork, each with narrated reasoning. This status doc is committed+pushed to `qwen-diffusion-agentic`
now as the pre-GPU checkpoint.
