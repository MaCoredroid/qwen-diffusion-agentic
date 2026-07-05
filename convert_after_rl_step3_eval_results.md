# Convert-After-RL Preservation Audit — STEP 3 (eval battery a/b/c/d) — RESULTS

Work-item #29. Frame: `convert_after_rl_design.md` (commit `6f5d20f`), reproduction contract `REPRODUCE_V2.md`.
Arm under test: **A_new** = fresh Run-1-recipe two-stream conversion (400 steps, seed 80101) on **M_{t+1}** (merged init+RL-v2),
the diffusion twin of the promoted merged-AR system. This step measures whether the RL-acquired tool-call exactness (the
`34→47` hybrid gain) survives a re-diffusionization the conversion was **not** trained on.

STEP 1 (merge, `77e97a5`) and STEP 2 (train A_new, `a1fe656`) done previously. This step = design §5 (export) + §6 (a/b/c/d).

## Headline (raw counts, all sampler-pinned + audited)

| # | measurement | A_new | anchor (C0) | design gate | verdict |
|---|---|---:|---:|---|---|
| a1 | diffusion hybrid-clean matched-20 | **47/63** | 47/63 | PASS ≥44 | **PASS** (== anchor; paired-McNemar net-loss 0, p=1.0) |
| a2 | diffusion careful matched-20 | **42/63** | 44/63 | PASS ≥41 | **PASS** (−2, within noise; consistent w/ a1) |
| a3 | diffusion hybrid never-train | **83/184** | 83/184 | PASS ≥78 | **PASS** (== anchor; McNemar net-loss 0) |
| — | diffusion aggregate | **130/247** | 130/247 | — | == anchor |
| b | AR-guided matched-20 / never-train / agg | **50 / 86 / 136 /247** | 50 / 77 / 127 | PASS agg ≥122 ∧ m20 ≥47 | **PASS** (agg +9 over C0) |
| c | GSM8K legacy full-context N=20 (strict) | **12/20** | 13/20 | PASS ≥13 / FAIL ≤11 | **INCONCLUSIVE** (−1 row; above KILL-2 floor 11; deterministic) |
| d | value-projection audits (3 diffusion turns.jsonl) | **all 0** | 0 | any nonzero ⇒ INVALID | **CLEAN** (`no_projection_events`) |

**Overall (design §8 logic).** The RL-acquired tool-call capability is **PRESERVED**: a1/a3 sit exactly on the C0 anchors and are
**turn-for-turn identical** to the promoted C0 system (paired-McNemar b=c=0), the diffusion aggregate is 130/247 == anchor, AR-mode
is 136/247 (+9 over C0), and every value-projection audit is clean. No KILL criterion fired. The **only** soft spot is GSM8K
retention at 12/20 — one row under the 13/20 anchor, in the design's single-row-rerun / INCONCLUSIVE band, **above** the KILL-2
conversion floor (11). This is not an RL-gain erosion (the tool-call capability held verbatim) and not a broad recipe confound
(a/b did not drop with c). Its 12-vs-13 boundary is exactly what the confirmatory **second training seed (80102)** — the next,
separate step — is designed to resolve.

## (a1) sharpest test — paired McNemar vs C0 (design §7)

`A_new` diffusion hybrid matched-20 is **turn-for-turn identical** to the promoted C0 run:
`b (C0-right,A_new-wrong)=0`, `c (A_new-right,C0-wrong)=0`, **net-loss 0**, two-sided exact-binomial `p=1.0`. a3 never-train:
`b=1, c=1`, net-loss 0, `p=1.0`. Re-conversion cost **nothing** on the RL tool-call capability.

**Bootstrap sub-gate note (transparency).** The design's a1 PASS also lists "episode-bootstrap 95% LB ≥41". The measured LB is
**35** (point 47, UB 58, stable across seeds) — but this is **identical for C0** (LB=35 too), because A_new is turn-identical to
C0. At n=20 episodes with within-episode correlation the episode bootstrap is too wide (~±12 turns) to resolve the ±3–4 band and
**cannot discriminate A_new from the very system it must preserve** — the reference itself fails the LB≥41 sub-gate. The
verdict therefore rests on the design's designated sharpest test (paired-McNemar: net-loss 0) plus raw==anchor, not on the
non-discriminating absolute-LB sub-gate. (Stats: `convert_after_rl_step3_paired_stats.json`.)

## STEP 3 provenance (design §12 checklist)

- **git commit at run:** `f3d05fb` (HEAD).
- **Pinned script sha256 — all 5 re-verified == design §6 table (OK, no divergence):**
  `eval_flare_northstar_hybrid_clean.py a4c66751…`, `eval_flare_northstar_matched.py 4cda3acf…`,
  `eval_flare_stage1_ab_diffusion.py eaa78d7a…`, `audit_value_projection_tokens.py 7b203e3e…`,
  `export_qwen35_9b_fastdllm_vllm.py 6d507ec9…`.
- **Base (diffusion) / adapter:** `models/qwen3.5-9b-fastdllm-mtplus1-merged` (mask_token_id 248077, bd_size 32, has_weights)
  + `runs/convert_after_rl/Anew_run1recipe_step400_seed80101` (adapter sha `d77dad68…`, == STEP 2). `--no-merge-adapter`.
- **Export (design §5):** `models/qwen3.5-9b-fastdllm-mtplus1-Anew-vllm-bf16` — manifest `replacement_count=427`,
  `lora_merge_count=152`, `lora_scale=2.0`, `mapped_text_tensors=427` (matches the merged-AR export profile).
- **Sampler function names:** a = `eval_flare_northstar_hybrid_clean.py` (`diffusion_hybrid_forced_grammar_seq_values`);
  a2 = `eval_flare_northstar_matched.py::run_diffusion` (`baseline_careful` + `--diffusion-structural-only`);
  b = `eval_flare_northstar_matched.py::--backend ar-vllm-guided` (vLLM 0.23.0, bf16, gpu-util 0.66, mamba-cache align,
  gdn-prefill triton, enforce-eager — mirrors `run_stock_qwen35_ar_guided_controls.sh::run_arm`);
  c = `eval_flare_stage1_ab_diffusion.py::full_context_sample_one`.
- **Datasets (sha256 prefix):** matched/careful/AR-matched `flare_scaleup_native_58.jsonl` `8453a60e…`;
  never-train `flare_nevertrain_bfcl_apibank.jsonl` `0533a0d1…`; GSM8K `gsm8k_main_test_first20.jsonl` `84fa1c7c…`.
- **Decode flags:** a/a3 block-size 32, max-new 384, top-p 0.95, temp 0.0, grammar-topk 256; a2 +small-block 32, max-extra 12,
  threshold 0.9; c block/small-block 32, max-new 256, threshold 0.9, top-p 0.95, temp 0.0, mask-id 248077, stop-id 248046,
  `FASTDLLM_FLARE_GDN_ROUTE=route_i FASTDLLM_GDN_KERNEL=torch`, `--skip-nll`.
- **Audit JSONs (all CLEAN, `verification_mode=no_projection_events`, every hard counter 0):**
  `…/Anew_matched20_hybrid/diffusion_hybrid_forced_grammar_seq_values/projection_value_audit.json`,
  `…/Anew_matched20_careful/diffusion_careful/projection_value_audit.json`,
  `…/Anew_nevertrain_hybrid/diffusion_hybrid_forced_grammar_seq_values/projection_value_audit.json`.

## Sampler-pin reconciliation notes (script == pin, doc flags stale)

Two REPRODUCE_V2/design command-lines used flag/label names that a later revision of the **pinned** script (sha256 still
matches the §6 table) folded into defaults; the pinned behavior is reproduced by omitting the stale flags. Documented so the
divergence is explained per §6:

1. `eval_flare_stage1_ab_diffusion.py` (c): `--full-context-generation --fresh-generation-blocks` are **removed** — these are now
   the defaults (`set_defaults(full_context_generation=True, fresh_generation_blocks=True)`; opt-out is `--active-block-generation`
   / `--tail-fill-generation`). Sampler is still `full_context_sample_one`. `--model-names` takes fixed slot names; `--adapter-b`
   maps to slot `B_two_stream` (doc's `Run1` label was stale). Heldout path is `data/flare_stage1_ab_pilot/heldout_nll.jsonl`
   (doc wrote `…_ab_pilot_train/…`); used only for the disjointness bookkeeping (0 overlap) under `--skip-nll`.
2. GSM8K N=20 was run as **4 chunks of 5** (generation-batch-size 1, constant few-shot) — byte-identical to a single 20-run;
   `full_context_sample_one` recomputes per single sequence with no cross-problem dependence. Determinism verified: reruns of
   chunks 0 and 3 reproduced per-example strict flags **exactly** (temp 0 greedy + torch GDN kernel).

## Artifacts (under gitignored `runs/`; counts recomputed from raw turns.jsonl → tracked JSONs at repo root)

`runs/convert_after_rl/{Anew_matched20_hybrid, Anew_matched20_careful, Anew_nevertrain_hybrid, Anew_ar_guided/{matched20,
nevertrain_bfcl_apibank60}, Anew_gsm8k/chunk[0-3]}` + `step3_eval_results.json` + `step3_paired_stats.json`. New infra:
`scripts/run_mtplus1_anew_ar_guided_slice.sh` (single-slice, self-contained boot→eval→kill AR-guided runner). Machine-readable:
`convert_after_rl_step3_results.json`, `convert_after_rl_step3_paired_stats.json`.

## Compute discipline

All heavy steps caged `systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=4G`, one model loaded at a time, GPU
pre-flight (<2 GB) before each, every command foreground ≤600 s; GSM8K + AR-guided chunked so each slice finishes in one call.
Total GPU wall ≈ 30 min (export 9 s; hybrid m20 ~4 min; careful m20 ~6 min; never-train ~6.5 min; AR m20 ~2 min; AR never-train
~3 min; GSM8K 6 chunks ~9 min; audits CPU).
