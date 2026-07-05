# Convert-After-RL Preservation Audit — RESULT (work-item #29, the loop's sharp test)

**Verdict: FLYWHEEL PRESERVES — the conversion step does not erase freshly-RL'd capability; the loop does not eat its own gains.**

Frame: `convert_after_rl_design.md` (commit `6f5d20f`), reproduction contract `REPRODUCE_V2.md`, anchors
`endgame_table_final.md` / `runs/endgame_scoreboard`. This doc consolidates STEP 1 (merge, `77e97a5`), STEP 2 (train
A_new, `a1fe656`), STEP 3 (eval battery seed 80101, `16087b5`), and STEP 4 (statistics + the confirmatory second seed
80102 + adjudication — this commit). Two independent re-conversions of the RL'd model were run; both preserve.

## 0. The question and the answer

The flywheel re-runs conversion on a model that **just gained** capability every cycle, and the conversion step is
**not trained on** that fresh capability. Historical order was always convert-*then*-RL, which never tests this. We ran
the missing direction: a **fresh** two-stream conversion on top of the merged RL-v2 weights (`M_{t+1}`), using the
**original Run-1 conversion mix** (deliberately NOT the RL episodes), and measured whether the RL-acquired tool-call
exactness (`34→47` hybrid gain) survives. **It survives, verbatim, across two seeds — losing exactly zero tool-call
turns vs the promoted no-reconvert system across 126 paired diffusion turns, and gaining 3.**

## 1. Two-seed result table (all raw counts, all sampler-pinned + audited)

| # | measurement | seed-1 (80101) | seed-2 (80102) | combined | anchor (C0) | design gate | verdict |
|---|---|---:|---:|---:|---:|---|---|
| a1 | diffusion hybrid-clean matched-20 | **47/63** | **50/63** | pooled 97/126 (≡48.5/63) | 47/63 | PASS ≥44 | **PASS** (both ≥ anchor) |
| a2 | diffusion careful matched-20 (2nd) | **42/63** | **43/63** | — | 44/63 | PASS ≥41 | **PASS** (−1/−2, moves with a1) |
| a3 | diffusion hybrid never-train | **83/184** | **83/184** | — | 83/184 | PASS ≥78 | **PASS** (== anchor both) |
| — | diffusion aggregate (hybrid) | **130/247** | **133/247** | — | 130/247 | — | **≥ anchor** both |
| b | AR-guided m20 / never-train / **agg** | 50 / 86 / **136/247** | — (seed-1 certifies AR mode) | — | 50 / 77 / 127 | PASS agg ≥122 ∧ m20 ≥47 | **PASS** (+9 over C0) |
| c | GSM8K legacy full-ctx N=20 (strict) | **12/20** | **14/20** | **26/40 = 0.65** | 13/20 | ≥13 PASS / ≤11 FAIL | **PASS** (combined == anchor) |
| d | value-projection audits (all diffusion turns.jsonl) | all 0, clean | all 0, clean | — | 0 | any nonzero ⇒ INVALID | **CLEAN** both seeds |

`b` (AR-mode preservation) was certified on the primary seed in STEP 3; per the design's confirmatory-seed scope the
second seed re-ran the diffusion battery **(a)** + retention **(c)** (the rows that carry the preservation signal and the
one soft spot). The seed-2 diffusion aggregate is **133/247** (+3 over the 130 anchor), i.e. re-conversion #2 is turn-for-
turn *stronger* than the promoted system, not weaker.

## 2. Statistics (design §7 — both estimators, per prior gates). File: `convert_after_rl_step4_stats.json`

### Paired-turn McNemar vs C0 (the designated SHARPEST "preserved vs eroded" test)

Each A_new run paired turn-by-turn against the promoted **C0 = init+RL-v2** hybrid-clean run (matched-20 47/63:
`runs/hybrid_forced_grammar_seq_values_v2/matched20/…`; never-train 83/184:
`runs/hybrid_broaden_nevertrain_v2/nevertrain_bfcl_apibank60/…diffusion_hybrid…`). Pairing validated: 63/184 turns
aligned 1:1, **gold_sha256 mismatch = 0**. `b` = C0-right & A_new-wrong; `c` = A_new-right & C0-wrong; net-loss `= b−c`;
two-sided exact-binomial `p`.

| run | b (C0-right, A_new-wrong) | c (A_new-right, C0-wrong) | net-loss b−c | p (two-sided) |
|---|---:|---:|---:|---:|
| a1 seed-1 vs C0 | 0 | 0 | **0** | 1.0 |
| a1 seed-2 vs C0 | **0** | **3** | **−3** (a *gain*) | 0.25 |
| **a1 pooled (126 turns) vs C0** | **0** | **3** | **−3** | 0.25 |
| a3 seed-1 vs C0 | 1 | 1 | **0** | 1.0 |
| a3 seed-2 vs C0 | 1 | 1 | **0** | 1.0 |

**The decisive line: `b = 0` for a1 in BOTH seeds** — across two independent re-conversions and 126 paired diffusion
tool-call turns, A_new got **every** turn C0 got right, and re-conversion cost **zero** tool-call turns. Seed-2 added 3
(strict superset of C0). Net-loss is 0 or negative everywhere; never a significant loss. This is the sharpest possible
form of "the RL gain the conversion was not trained on survived."

### Episode-level bootstrap (B = 10000 percentile CI; absolute level vs the pre-RL floor 34 / half-gain line 40)

| run | point | 95% LB | 95% UB |
|---|---:|---:|---:|
| a1 seed-1 | 47 | 35 | 58 |
| a1 seed-2 | 50 | 39 | 60 |
| a1 C0 (reference) | 47 | 35 | 58 |
| **a1 pooled 40-ep (per-63 scale)** | **48.5** | **40.5** | 56.0 |

The design's a1 absolute sub-gate "bootstrap LB ≥ 41" is **non-discriminating at n=20** (documented in STEP 3): the
episode bootstrap with within-episode error compounding is ~±12 turns wide, and **C0 itself — the very system A_new must
preserve — has LB = 35**, failing the same sub-gate. It therefore cannot separate A_new from the reference and is not
load-bearing. What the bootstrap *does* show is decisive: every CI **clears the pre-RL floor (34)** with margin, and the
pooled two-seed LB (40.5) clears the **half-gain-lost line (40)**. The verdict rests on the design's designated sharpest
test (McNemar net-loss ≤ 0, never significant) + raw ≥ anchor, exactly as the STEP-3 reconciliation established.

### Retention (c) — the only STEP-3 soft spot, RESOLVED by the second seed

Seed-1 12/20 sat in the design's single-row-rerun / INCONCLUSIVE band (one row under the 13 anchor, above the KILL-2
floor 11). The design mandates "rerun once and report both seeds." Seed-2 = **14/20 ≥ 13 → PASS**; **combined 26/40 =
0.6500 = the 0.65 anchor exactly.** The 12-vs-13 boundary was seed noise; the two-seed mean lands on the anchor. Both
seeds are well above the B@1000 conversion floor (11). Retention is preserved, not eroded.

## 3. Adjudication (design §8 overall verdict logic)

- **a1 PASS** — raw ≥ anchor both seeds (47, 50; both ≥ 44 and ≥ the 47 C0 anchor); McNemar net-loss ≤ 0 both seeds
  (0, −3), never significant; CIs clear the pre-RL/half-gain region.
- **b PASS** — AR-guided aggregate 136/247 ≥ 122 (+9 over C0), matched-20 50 ≥ 47.
- **c PASS** — combined 0.65 == anchor; seed-2 resolves the inconclusive; both above KILL-2.
- **d CLEAN** — every value-projection audit counter 0, `no_projection_events`, on all six diffusion turns.jsonl (3 per
  seed). No phantom-win contamination.

**⇒ FLYWHEEL PRESERVES (loop viable): a1 PASS ∧ b PASS ∧ c PASS ∧ d clean.** (a2/a3 consistency: both hold; a2 −1/−2 in
the rawer careful lane while hybrid holds/gains is inside observed paradigm noise, not an erosion signal.)

This is **not** ERODES (a1 did not fail — it held and gained), **not** RECIPE-CONFOUND/KILL-4 (nothing dropped together;
AR-mode *rose* +9 and retention resolved to the anchor — the recipe did not broadly damage the base), and **not**
INCONCLUSIVE (a1 is 47/50, above the anchor, not in the 39–43 gray band after two seeds).

**No kill criterion fired.** KILL-1 merge gate PASS (STEP 1, bit-exact `init+2.0·B@A`, mask 248077 / bd_size 32).
KILL-2 GSM8K ≥ 11 both seeds (12, 14). KILL-3 audits clean both seeds. KILL-4 not triggered (b = 136 ≥ 118 and c > 11).

## 4. What this certifies for the loop

The conversion step (`M_t → diffusionize`) applied to a model carrying a fresh, un-replayed RL capability **preserves that
capability** — in diffusion mode (the lane the loop serves), in AR mode (the clean stream), and without collateral
retention damage — reproducibly across two seeds. The step-1 preservation mechanisms hypothesized as *possibly needed*
(KL-to-pre-conversion, capability replay, convert-and-RL-jointly) are **not required** by this evidence: a plain fresh
Run-1-recipe conversion on the merged RL weights, using data that excludes the RL pool, already holds the gain. The
flywheel `M_t → diffusionize → RL-update → M_{t+1} → re-diffusionize → repeat` does not erase its own gains at the
conversion boundary.

## 5. Provenance (design §12 checklist)

- **git commit at run:** `16087b5` (STEP-4 work committed on top). **All 4 eval-script sha256 re-verified == design §6
  pins** (no divergence): hybrid_clean `a4c66751…`, matched `4cda3acf…`, stage1_ab `eaa78d7a…`, audit `7b203e3e…`.
- **Merged diffusion base (both seeds):** `models/qwen3.5-9b-fastdllm-mtplus1-merged` (mask_token_id 248077, bd_size 32,
  has_weights). **Seed-2 adapter:** `runs/convert_after_rl/Anew_run1recipe_step400_seed80102` (sha
  `34cfa1346ffe1d50…`), 400 steps, Run-1 recipe, r16/α32, targets q/k/v/o + in_proj_{qkv,z,b,a} + out_proj; `--no-merge-adapter`.
- **Seed-2 training health (KILL check) PASS:** 80 logged points, zero NaN/Inf, loss 2.207/5.288/3.547 (min/max/mean ≈
  seed-1's 2.208/5.395/3.483); **LR at every 100-step boundary matches the reference cosine to 4 s.f.** (8.810e-6 /
  5.283e-6 / 1.581e-6 / 1.639e-10) — the 4 resumable chunks reproduce a single continuous 400-step horizon; adapter 304
  tensors, all finite, 152/152 lora_B nonzero. Dataset `data/flare_redesign_run1_copy_retention_mix` (5055; excludes the
  RL-v2 pool), same as seed-1.
- **Samplers:** a1/a3 `eval_flare_northstar_hybrid_clean.py` (`diffusion_hybrid_forced_grammar_seq_values`); a2
  `eval_flare_northstar_matched.py` (`baseline_careful` + `--diffusion-structural-only`); c
  `eval_flare_stage1_ab_diffusion.py::full_context_sample_one` (defaults full-context + fresh-blocks; `--adapter-b`→slot
  `B_two_stream`; GSM8K = 4×5 chunks, batch 1, constant few-shot; reuses the identical 20-problem input as seed-1).
- **C0 reference runs paired against:** matched-20 `runs/hybrid_forced_grammar_seq_values_v2/matched20/…` (47/63);
  never-train `runs/hybrid_broaden_nevertrain_v2/nevertrain_bfcl_apibank60/diffusion_hybrid_forced_grammar_seq_values/…`
  (83/184). Seed-1 McNemar (b=0,c=0 / b=1,c=1) reproduced here bit-for-bit → C0 reference + pairing validated.
- **Audit JSONs (all CLEAN):** `runs/convert_after_rl/Anew_{matched20_hybrid,matched20_careful,nevertrain_hybrid}_seed80102/…/projection_value_audit.json`.
- **Machine-readable:** `convert_after_rl_step4_stats.json` (McNemar + bootstrap, both seeds), `convert_after_rl_seed80102_results.json`
  (seed-2 battery + audits), plus STEP-3 `convert_after_rl_step3_{results,paired_stats}.json`.

## 6. Compute discipline

Every heavy step caged `systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=4G`; one model loaded at a time;
GPU pre-flight (<2 GB wait-loop) before each; every command foreground ≤ 600 s; training chunked into 4 resumable
segments and GSM8K into 4×5 chunks so each slice finished in one call. Seed-2 total GPU wall ≈ 34 min train + ~20 min
evals; within the ~2 GPU-h confirmatory-seed budget; no extension past the designed 400 steps.
