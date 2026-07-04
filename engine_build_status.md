# P2 Engine-Fast Diffusion Serving — Build Status & GPU Smoke Checklist

Workflow follow-on to `p2_serving_reuse_plan.md` (the reuse decision, milestones, kill criteria).
Date: 2026-07-03. Author: build+review sweep, four parallel agents.

**Bottom line:** the entire M1 write-list (`Qwen3_5FlareModelState` + ops + hybrid-clean FSM
reference + parity harness + flywheel serving surface) is implemented, CPU-tested, and committed
locally in three repos. The GPU smoke gauntlet has now been **run on the RTX 5090 (sm_120) through
step 4** (see §0). **Substrate PASS, checkpoint blocked:** steps 1-3 all pass — the first-party dLLM
decode path, the sm_120 attention/GDN kernels, and MRV2×GDN both default and align+APC all run
correctly on this card, killing the R1/R2/K1/K2 substrate risks. Step 4 (the M1 read-only-denoise
go/no-go) is **not a clean go: the substrate mechanism works but is under-scoped, and there is no
diffusion-trained export to decide it** — so steps 5-6 (turn byte-parity, matched-20 wall-clock) did
**not** run and there is **no engine-vs-HF wall-clock number yet.** All four reviews returned
**fix-needed**; the crux (GDN read-only-denoise state discipline) is now empirically characterised in
§0 step 4. Full checklist with pass/kill criteria in §3.

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
   canvas_length-multiple-of-64 export before re-attempting M1.
4. **Toolchain workarounds must be baked into the launcher** — the cu13 CTK-skew flag and
   `VLLM_USE_FLASHINFER_SAMPLER=0` are required for the Qwen GDN path to build/run on this box.

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
- **Pass (M2 gate, all of):** **< 1.120 s/turn** AND **≥ 55/63 exact-args**, **15/20 episodes**,
  **63/63 exact_seq**, **63/63 valid_xml**, and **force-counters == 0 on values**. (Prerequisite: fix
  the `advance_calls` counter so the read/advance ratio is real before quoting the forwards-saved
  metric.)
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
