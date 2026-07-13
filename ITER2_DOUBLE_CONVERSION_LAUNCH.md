# ITERATION-2 DOUBLE CONVERSION — LAUNCH NOTE (#128 part 1)

**Launched:** 2026-07-13 (detached, caged). Runner `runs/iter2_conversion_runner.sh`,
pidfile `runs/iter2_conversion.pid`, STOP-file `runs/iter2_conversion.STOP`,
log `runs/kraise_reconvert_iter2/runner.log`. Sequential: **ARM A (plain) → to completion → ARM B (V1)**.

Object (CONVERSION_READY_iter2.md PRIMARY): **M_swe_S (iter2)** = init+RL-v2 + windowed SWE-SFT.
Shared STEP 0 merges `runs/swe_sft_arm1_iter2/Aswe_S_step400_seed71101/checkpoint-400` into
`models/qwen3.5-9b-fastdllm-mtplus1-merged` (W += 2.0·B@A, maxabs bit-exact gate, mask 248077 / bd 32)
→ re-conversion base `models/qwen3.5-9b-fastdllm-mswe-S-iter2-merged`. Both arms re-convert from it.
Seed convention: **81101** (same as the iteration-1 mswe_S re-conversion; distinct from RL-v2 80101/80102).

## ARM A — twin@plain (shipping candidate, byte-reproducible)
The #29 convert-after-RL protocol **verbatim as iteration-1 ran it** — the exact env chain of
`scripts/kraise_reconvert_mswe_S_driver.sh` (→ `scripts/run_flare_redesign_run1.sh`): single continuous
400-step cosine, block 512 / bd 32, lr 1e-5, LoRA r16/α32 dropout0.05, 9 targets, VALUE_SPAN_LOSS_WEIGHT=2.0,
VALUE_SPAN_MASK_PROB=1.0, data `data/flare_redesign_run1_copy_retention_mix` (5055 rows, **excludes** the
SWE/RL pool = leakage firewall). NO V1 code enters this path (`FASTDLLM_V1_COPY_SPAN` unset).
Output: re-conv adapter `runs/kraise_reconvert_iter2/mswe2_S_twinK1_run1recipe_step400_seed81101`;
clean-stream vLLM export **`models/qwen3.5-9b-fastdllm-mswe2-S-twinK1-vllm-bf16`** (= merged base + adapter,
the HF merged AR form), export script sha `6d507ec9…` (pinned).

## ARM B — twin@V1 (SECTION-V V1 copy-span joint-infill, DIRECTIVE-3 piggyback)
IDENTICAL conversion **+** the V1 objective folded into L_diff. V1 code is a NEW file
`scripts/v1_copy_span_infill.py`; it is activated only through an **env-gated, strict-no-op** hook in the
shared finetuner (`FASTDLLM_V1_COPY_SPAN=1`) — arm A never enters it, so the plain path is untouched and
byte-reproducible. Output adapter `…/mswe2_S_twinV1_run1recipe_step400_seed81101`; export
**`models/qwen3.5-9b-fastdllm-mswe2-S-twinV1-vllm-bf16`**; V1 manifest `…/v1_copy_span_manifest.json`.

### V1 implementation (exactly what SECTION V.1 prescribes)
1. **Whole-copy-span joint masking.** Per training window, arg-value COPY spans are masked in the denoise
   stream **as whole spans** (all L positions at once), prior context (which contains the verbatim source)
   kept clean; the two-stream forward supervises all L masked positions **jointly from one forward** — this
   is `L_copy = −Σ log p(v_i | ctx, mask[1..L])`. Realized by injecting a per-window `flare_mask_indices`
   into the model's EXISTING two-stream forward hook (no modeling.py edit): view0 masks
   {random block noise} ∪ {whole copy spans} ∪ {derived-value forced}; view1 is the complement.
2. **Census 4-gram detector + V1 tight precision fix.** Copy = trailing 4-gram present in the earlier
   same-doc context (census `tok_ngrams` n=4, reused predicate). TIGHTENED: a training copy span must be a
   contiguous substring of a SINGLE earlier source location (longest verbatim match), not the union n-gram
   set — trades recall for precision so no derived value is mislabeled copy (the KILL-T1 regression risk).
   Loose→tight shrinkage is logged to the V1 manifest.
3. **Span-length curriculum.** L∈{2,3,4} for the first ~⅓, ceiling→8, then →32 over the remainder; per-span
   masked length capped at the ceiling (≤32) so no single microbatch is dominated by one canvas-scale run
   (the min(L,32) weighting realized as a mask cap).
4. **Copy/derived partition.** Copy value tokens → whole-span joint-infill bucket (curriculum). Derived
   ARG_VALUE tokens → forced-masked prob 1.0 every step (unchanged from plain). Copy and derived never
   overlap. `VALUE_SPAN_MASK_PROB=0` for arm B hands value masking to the collator; `VALUE_SPAN_LOSS_WEIGHT`
   stays 2.0.

### Conservative choices (design left open; picked conservatively, LOGGED)
- **[C1] copy-span weight** kept at the plain **2.0** (not down-weighted to the O1 joint-commit 1.0). This
  changes nothing on the derived path (zero regression risk to the certified `exact_args` / KILL-T1 guard)
  and keeps arm B a strict superset perturbation of the plain recipe. Down-weight-to-1.0 is a clean follow-up.
- **[C2] curriculum step** = collator call counter / MAX_STEPS (== optimizer global_step at GRAD_ACCUM=1, batch 1).
- **[C3] per-span mask** = contiguous `min(Ls, ceiling)` window anchored at the span start.
- **[C4] random-block component** regenerated (U[0.3,0.8], adaptive low-rate U[0.02,0.12] on non-value
  blocks) so L_diff on non-copy positions is preserved when the model's own sampler is bypassed by the
  provided mask; distributional (not bit-identical) fidelity, sufficient for the experimental arm.

**Firewall:** V1 detects copy spans in the SAME re-conversion mix as the plain arm (copy/retention mix,
excludes the SWE/eval pool); no keeper/eval-ring instance is trained on. KILL-D1 firewall stands verbatim.

**Verification this turn (launch-only):** V1 CPU unit smoke PASS (whole-span selection, contiguous no-holes,
copy/derived partition, curriculum ceilings, finite L_copy); collator emits a valid bool `flare_mask_indices`;
merge maxabs gate PASS; ARM A first steps observed with finite loss. ARM B runs second, autonomously; on any
crash the runner emits `[state] ARM_FAILED` and writes the STOP-file.
