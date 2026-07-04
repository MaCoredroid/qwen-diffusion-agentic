# Real Diffusion Export + Block-Config Reconciliation (2026-07-03)

Follow-on to `engine_build_status.md` §0 step-4 and its open items 1, 3, and the
config/docs "block/chunk misalignment" hazard. Produces the **real
diffusion-trained vLLM export** (the b1000 on-disk export is a smoke/AR-parity
checkpoint that decodes gibberish through the canvas/denoise path) and pins the
FLARE canvas block size with evidence.

## 1. Block-config reconciliation — DECISION: `canvas_length = 32`

Three numbers were in tension:

| source | value | where |
| --- | ---: | --- |
| trained block-diffusion size (`bd_size`) | **32** | `models/qwen3.5-9b-fastdllm-init/conversion_manifest.json` |
| engine default block (`_DEFAULT_BLOCK = _GDN_CHUNK`) | **64** | `vllm_p2_pr42406/.../qwen3_5_flare.py:95,99` |
| GDN chunk (`_GDN_CHUNK`) | **64** | same file, L95 |

**Correction to a stale doc:** `engine_build_status.md` §0 ("engine default
`_DEFAULT_BLOCK=32`") is wrong. The engine source is `_DEFAULT_BLOCK =
_GDN_CHUNK = 64`. Verified in code, not from the doc.

**Which block size did the winning HF hybrid-clean eval (47/63) run?** `32`. The
whole chain is block-32 end to end:

- `init` `bd_size=32`; B@1000 `TRAIN_BD_SIZE=32`; Run-1 `TRAIN_BD_SIZE=32`
  (REPRODUCE_V2 §3–5).
- RL-v2 matched-20 and the promoted hybrid-clean matched-20/never-train evals all
  pass `--block-size 32` (REPRODUCE_V2 §6–7).
- The HF sampler hard-sets it: `eval_flare_northstar_hybrid_clean.py:311`
  `set_block_size(model, 32)`; `FlarePrefixCache`/`RequestDiffusionState.advance`
  step one 32-token block at a time (`sample_hybrid_clean` `_maybe_advance_cache`).

**Decision: pin the engine canvas to 32 to match training/eval.** Using the
engine default 64 would denoise **two trained blocks per commit** and break parity
with the checkpoint — the model never saw 64-token denoise blocks. `32 %
_GDN_CHUNK(64) != 0` trips the engine's mid-chunk hazard warning
(`qwen3_5_flare.py:212`): a commit boundary at 32 lands mid-GDN-chunk, so the fp32
`chunk_states[:,-1]` boundary snapshot is a partial (non-checkpoint) recurrent
state. That is an **engine-side restore-scope concern** (the same one as
`engine_build_status.md` open items 2/5 — widen `_denoise_state_rows` and verify
the read-only-denoise restore at a mid-chunk boundary on the trained export), **not
a reason to change the trained block size.** The step-4 probe already boots at
`canvas_length=32`; this reconciliation makes that the pinned, documented choice.

**Wiring:** `num_speculative_tokens` falls back to `diffusion_config.canvas_length`
when no `speculative_config` is set (`vllm/config/vllm.py:508-519`), so a single
`--diffusion-config '{"canvas_length":32,...}'` both (a) sets the engine block
(L207) and (b) sizes the spec-decode `draft_tokens` buffer to width 32 — fixing the
`_finish_prefills` width-0 crash without a separate `--speculative-config`. This
mirrors the proven offline boot recipe in `step4_readonly_denoise_probe.py`.

## 2. The real export

**Output:** `models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16` (19G, new dir; the b1000
export is left untouched — it is still the tokenizer reference + AR-parity model).

**What is actually the serving delta.** The three-stage campaign is a
**continued-training lineage, not a runtime adapter stack.** The winning HF
hybrid-clean path is `--base-model models/qwen3.5-9b-fastdllm-init --adapter
runs/rl_multiturn_grpo_v2/from_selected_base_g4_step300/adapter_model
--no-merge-adapter` (`eval_flare_northstar_hybrid_clean.py`). Only the **single
RL-v2 adapter** is applied at serve/eval time. It was continued FROM the Run-1
adapter (identical r16/α32/targets, byte-identical file size) so it **subsumes
Run-1**; B@1000 (r8, attention-only) is a separate conversion-foundation/AR-parity
lineage and is **not** in the hybrid-clean delta. Exporting = merging the RL-v2
adapter into the init-materialized weights (`W += (α/r)·B@A`), which is
mathematically identical to the eval's PEFT runtime application.

**Provenance (full chain + shas in `conversion_manifest.json`):**

- base `Qwen/Qwen3.5-9B @ c202236…b9a`, materialized to `qwen3.5-9b-fastdllm-init`
  (mask token `|<MASK>|` id **248077 < vocab 248320** → reserved slot, **no embed
  resize**, export shapes match the official conditional wrapper).
- RL-v2 adapter `adapter_model.safetensors` sha256
  `c67f0a16…60c841` (r16, α32, scale 2.0; targets q/k/v/o + in_proj_qkv/z/a/b +
  out_proj).
- Run-1 `d4ee095c…d692d87`; B@1000 `f06b78c2…dda8f4` (lineage record).
- exporter `$FLYWHEEL/scripts/export_qwen35_9b_fastdllm_vllm.py` sha256
  `6d507ec9…a7119d5` (matches REPRODUCE_V2 pin).

**Export result:** `replacement_count=427`, `mapped_text_tensors=427`,
`lora_merge_count=152`, `lora_scale=2.0`. The 152 = 24 GDN layers × 5 modules + 8
attn layers × 4 — i.e. **the GDN `in_proj_*`/`out_proj` LoRA deltas merged too**,
not just attention (B@1000 merged only 32 = 8×4).

## 3. Sanity (two gates, both PASS)

**(a) Merge correctness — bit-exact.** For a GDN `in_proj_qkv` tensor and an attn
`o_proj` tensor, the export weight equals `init + 2.0·(B@A)` **exactly** (maxabs
diff 0.0), the LoRA delta is nonzero, and the weight moved off `init`. Proves the
merge (incl. GDN) is real and complete.

**(b) Decode quality — coherent, exact-args plausible.** One HF hybrid-clean
episode (block 32) on init+RL-v2 (the bit-identical weights the export encodes;
the export itself is the official conditional-wrapper arch, which the Fast-dLLM
hybrid sampler cannot load, so the identical-weights source is the valid proxy):

```
episodes=1 turns=4 exact_args=3/4 valid_tool_call=4/4 exact_tool_sequence=4/4
schema_ok=4/4 sec_per_turn=2.45 forwards_per_turn=32.5 value_tokens=95
```

3/4 exact-args (75%) is right on the promoted 47/63 ≈ 74.6% matched-20 rate;
valid/schema/sequence all 4/4 → **not gibberish** (contrast the b1000 smoke export,
which decodes gibberish through this path). Artifact:
`runs/reproduce_v2/rlv2_export_sanity_hybrid/summary.json`.

## 4. Launcher fix

`$FLYWHEEL/scripts/qwen35_9b_flare_hybrid_serve.sh`:

- `MODEL_DIR` default → the real `qwen3.5-9b-fastdllm-rlv2-vllm-bf16` export.
- Reads `canvas_length` from the export `conversion_manifest.json` (fallback 32),
  builds `--diffusion-config '{"canvas_length":32,"max_denoising_steps":8}'`, and
  passes it to `vllm serve` **only when the FLARE gate is on**. This is the
  M1-launcher-gap fix (`engine_build_status.md` open item 1): the FLARE path could
  not boot before because no `diffusion_config`/`num_speculative_tokens` was wired,
  so the draft buffer was width-0. Also exports `VLLM_QWEN3_5_FLARE_BLOCK=$canvas`
  as an env belt-and-suspenders. `bash -n` clean; canvas parse verified against the
  real manifest.

## 5. Open items NOT closed here (engine-side, gated on GPU serve)

- Read-only-denoise restore row scope (`[1]` vs written `[1..4]`) + mid-chunk
  boundary-snapshot at block 32 — must be re-proven on THIS export at the step-4
  re-run before M1 can be called.
- Steps 5–6 (turn byte-parity, matched-20 wall-clock) remain unrun.
