# Convert-After-RL Preservation Audit — Experiment Design (work-item #29, the loop's sharp test)

Author: design sweep, 2026-07-04. Status: **DESIGN ONLY — monitor review before any GPU run.** CPU-only produced.
Frame: `methodology_diffusion_accelerated_rl.md` (the preservation-audit reframe, step-2 @`0475152`), reproduction contract
`REPRODUCE_V2.md`, anchors `runs/endgame_scoreboard/report.md` + `endgame_table_final.md`.

---

## 0. The question, in one paragraph

The flywheel is a PROCESS: `M_t (AR) → diffusionize → RL-update the AR model → M_{t+1} → re-diffusionize → repeat`. Every
cycle re-runs the conversion step on a model that **just gained** capability. Benchmark parity is the wrong certificate; the
loop lives or dies on whether **conversion preserves the newest, most fragile capability — the one the last RL cycle acquired,
which the conversion step is NOT trained on.** Our entire historical order was convert-*then*-RL, which never tests this. This
experiment runs the missing direction: take the merged weights that carry the RL-v2 gain, run a **fresh** two-stream conversion
on top of them using the **original conversion data** (deliberately NOT the RL episodes), and measure whether the RL-acquired
tool-call exactness survives — in diffusion mode, in AR mode, and without collateral retention damage.

**Verdict axis.** PRESERVED ⇒ the conversion step does not erase freshly-RL'd capability ⇒ the flywheel does not eat its own
gains. ERODED ⇒ each cycle's conversion resets the model toward its pre-RL diffusion behavior ⇒ the loop is not viable as drawn
and step-1 needs a preservation mechanism (KL-to-pre-conversion, capability replay, or convert-and-RL-jointly).

---

## 1. The object under test — lineage and exact artifacts

The three-stage campaign is a **continued-training lineage, not a runtime adapter stack** (`real_diffusion_export_block_reconcile.md`).
The single serving delta is the RL-v2 adapter (it continued FROM Run-1, so it **subsumes Run-1**; B@1000 is a separate
attention-only AR-parity lineage, **not** in the hybrid-clean delta).

| symbol | what it is | on disk |
|---|---|---|
| base | `Qwen/Qwen3.5-9B @ c202236…b9a` | HF cache snapshot |
| init | materialized Fast-dLLM candidate, mask `\|<MASK>\|` id **248077**, `bd_size=32` | `models/qwen3.5-9b-fastdllm-init` |
| RL-v2 adapter | diffu-GRPO adapter (r16/α32, q/k/v/o + in_proj_{qkv,z,a,b} + out_proj), sha `c67f0a16…60c841` | `runs/rl_multiturn_grpo_v2/from_selected_base_g4_step300/adapter_model` |
| **M_{t+1} (AR, merged)** | init + RL-v2 folded (`W += (α/r)·B@A`, scale 2.0); the scoreboard's **merged-AR** model | `models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16` (vLLM export; `lora_merge_count=152`, `replacement_count=427`) |

M_{t+1} is an **AR model** that carries the RL gain in AR mode (merged-AR-guided **127/247** aggregate). The vLLM export strips
the diffusion bridge/mask token (`mask_token_id=None`), so it is **not** diffusion-loadable as-is. The experiment therefore
builds M_{t+1} once more **inside the Fast-dLLM HF stack** (bridge + mask token intact) so a fresh conversion can be trained on
top of it (Step 1 below).

### Anchors (all measured; these are the numbers the audit is judged against)

| lane | metric | value | source |
|---|---|---|---|
| diffusion hybrid-clean (constrained) | matched-20 exact_args | **47/63** | REPRODUCE_V2 §7, scoreboard |
| diffusion hybrid-clean | never-train exact_args | **83/184** | REPRODUCE_V2 §7 |
| diffusion hybrid-clean | aggregate | **130/247** | `endgame_table_final.md` |
| diffusion careful (rawer lane) | matched-20 exact_args | **44/63** | REPRODUCE_V2 §6 (RL-v2 gate) |
| AR mode (merged-AR-guided) | matched-20 / never-train / agg | **50/63 · 77/184 · 127/247** | scoreboard |
| GSM8K retention (RL-v2) | legacy full-context strict, N=20 | **13/20 = 0.65** | REPRODUCE_V2 §6 |
| **pre-RL diffusion (Run-1, careful)** | matched-20 exact_args | **34/63** | memory: "44/63 (+10 vs Run-1's 34)" |
| pre-RL retention (Run-1) | GSM8K legacy N=20 | **14–15/20 = 0.70–0.75** | REPRODUCE_V2 §5 |
| conversion floor (B@1000) | GSM8K legacy N=20 | **11/20 = 0.55** | REPRODUCE_V2 §4 |

**The thing being preserved = the RL gain: `34 → 47` (+13 turns) in the hybrid-clean lane, `34 → 44` (+10) careful.** If a fresh
conversion on top of the RL weights drops matched-20 back toward 34, conversion erased the RL gain.

**Promotion-discipline note (raw vs constrained vs protected).** Raw diffusion tool-calls are ~0 (a known structural fact, not
an artifact to fix here), so there is no meaningful RAW tool-call lane. Preservation is judged in the **constrained hybrid-clean
lane** — legitimate because that is exactly the lane in which RL-v2's capability was certified and in which the loop serves. The
**careful** lane (rawer, fewer projections) is carried as a secondary consistency check so the constrained number cannot mask
model-side erosion. No protected/sidecar number is used as evidence anywhere.

---

## 2. Design: one primary arm, two required controls, one optional sensitivity arm

| arm | base weights | conversion data | recipe | what it isolates |
|---|---|---|---|---|
| **A_new (PRIMARY)** | M_{t+1} merged (init+RL-v2) | ORIGINAL Run-1 copy/retention/public mix (`data/flare_redesign_run1_copy_retention_mix`, 5055, **not** the RL pool) | Run-1 two-stream copy-grounding, **400 steps**, r16/α32, GDN+attn | Does re-diffusionizing an RL'd model preserve the RL gain? |
| **C0 — no-reconvert (control)** | init | — (existing RL-v2 adapter) | none (current promoted system) | The target level A_new must hold: 47/63 · 130/247 · 127/247 AR · 0.65 |
| **C1 — convert-before-RL (control)** | init | — (historical Run-1 then RL-v2) | historical order | Symmetric mirror: conversion produced 34, RL added +13. Shows the gain at stake and the erosion floor (34). |
| A_new-strict (OPTIONAL) | M_{t+1} merged | B@1000 GSM8K/MBPP-only mix (`data/flare_stage1_ab_pilot_train`, **zero tool-call data**) | B@1000 recipe, r8 attn-only, **≤500 steps** | Strictest "trained on none of the tool-call distribution" probe |

Why Run-1 recipe/mix is the primary (not literal B@1000): it is the **actual conversion foundation of the promoted diffusion
system**, so its diffusion tool-call anchor (34 pre-RL → 47 post-RL) is calibrated to exactly this recipe; it is **400 steps**
(inside the extended-training-erosion cap of ~400–600, `MEMORY`); and it runs on the **diverse 5055-sample mix** that retained
GSM8K 0.75 historically (tiny mixes under many steps are the documented retention killer). Its data does not include the RL v2
pool (`data/rl_multiturn_v2_public_pool`, a different, leak-filtered ToolACE pool), so the RL-acquired exactness remains a
capability A_new **was not trained on** — the whole point. The optional strict arm removes even Run-1's SFT tool-call data.

**Retrain-freely discipline:** A_new is disposable. If the conversion looks over- or under-trained, retrain at a different step
count in {300, 400, 500, 600} rather than reasoning around a bad checkpoint. Budget explicitly includes a second seed.

---

## 3. Step 1 — Rebuild M_{t+1} as a diffusion-loadable base (the merge)

Merge the RL-v2 adapter into the `init` Fast-dLLM candidate **in the HF stack** so the mask token / bridge / `bd_size=32` survive
(the vLLM export does not). This produces `M_{t+1}` as a concrete diffusion-loadable AR checkpoint — the diffusion twin of the
scoreboard's merged-AR model.

Write `scripts/merge_adapter_into_fastdllm_candidate.py` (PEFT `merge_and_unload`; ~40 lines):

```
# pseudocode-exact spec
base = load Fast_dLLM_Qwen3_5(models/qwen3.5-9b-fastdllm-init)     # bridge loader, trust_remote_code
peft = PeftModel.from_pretrained(base, runs/rl_multiturn_grpo_v2/from_selected_base_g4_step300/adapter_model)
merged = peft.merge_and_unload()                                   # W += (α/r)·B@A, scale 2.0
merged.save_pretrained(models/qwen3.5-9b-fastdllm-mtplus1-merged)
copy: conversion_manifest.json (append lineage note), tokenizer, added_tokens.json, mask-token config
```

**Merge sanity gate (mirror the export's gate, `real_diffusion_export_block_reconcile.md` §3a):** for one GDN `in_proj_qkv`
tensor and one attn `o_proj` tensor, assert `merged == init + 2.0·(B@A)` to **maxabs diff 0.0**, LoRA delta nonzero, weight moved
off init; assert `conversion_manifest.json: mask_token_id==248077, bd_size==32, has_weights==true`. **Any failure ⇒ KILL-1
(stop; the base is wrong, do not train).**

Equivalence check (cheap, decisive): one HF hybrid-clean episode on `models/qwen3.5-9b-fastdllm-mtplus1-merged` with the **A_new
adapter absent** must reproduce the init+RL-v2 hybrid-clean behavior (≈3/4 exact on the sanity episode, coherent, not gibberish),
confirming the merged base == the serving weights before any re-conversion.

---

## 4. Step 2 — Fresh two-stream conversion → A_new (PRIMARY arm)

Run-1 recipe, capped at 400 steps, base = the merged candidate, data = the existing original conversion mix. Adapted from
REPRODUCE_V2 §5 (only `MODEL_PATH`, `OUTPUT_DIR`, `MAX_STEPS`, seeds change):

```bash
cd /home/mark/qwen_diffusion
ENV_PY="$PWD/.venv-fastdllm/bin/python" \
DATASET_DIR="$PWD/data/flare_redesign_run1_copy_retention_mix" \
OUTPUT_DIR="$PWD/runs/convert_after_rl/Anew_run1recipe_step400_seed80101" \
MODEL_PATH="$PWD/models/qwen3.5-9b-fastdllm-mtplus1-merged" \
MAX_STEPS=400 MAX_TRAIN_SAMPLES=5055 BLOCK_SIZE=512 TRAIN_BD_SIZE=32 GRAD_ACCUM=1 \
LEARNING_RATE=1e-5 SAVE_STEPS=100 SAVE_TOTAL_LIMIT=4 LOGGING_STEPS=5 \
LORA_R=16 LORA_ALPHA=32 LORA_DROPOUT=0.05 \
LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj \
DISABLE_GROUP_TEXTS=0 TRUNCATION_SIDE=left CONVERSATION_TEMPLATE=fast_dllm_v2_native \
VALUE_SPAN_LOSS_WEIGHT=2.0 VALUE_SPAN_MASK_PROB=1.0 \
SEED=80101 DATA_SEED=80101 \
scripts/run_flare_redesign_run1.sh
```

Runner exports (unchanged, the two-stream/copy-grounding schedule): `FASTDLLM_FLARE_TWO_STREAM=1`, `FLARE_TWO_STREAM=1`,
`FASTDLLM_FLARE_GDN_ROUTE=route_i`, mask-rate 0.3–0.8, adaptive-copy schedule on, `FASTDLLM_GDN_KERNEL=fla`, cosine LR.
Second seed for the confirmatory pass: `SEED=80102 DATA_SEED=80102`, `OUTPUT_DIR=…_seed80102`.

Optional strict arm (only if greenlit): REPRODUCE_V2 §4 B@1000 command with `MODEL_PATH=…mtplus1-merged`, `MAX_STEPS=500`, r8
attn-only, `DATASET_DIR=data/flare_stage1_ab_pilot_train`.

**In-conversion guardrails (retention probes):** the runner already probes GSM8K during training. Watch max KL and the block-mode
anchor; the erosion lesson says steps beyond ~400–600 on this mix erode capability — do **not** extend past 600 to "rescue" a
weak preservation number.

---

## 5. Step 3 — Export A_new's clean stream for AR-mode eval

The two-stream clean stream is byte-identical to the AR forward; exporting it merges the clean-stream LoRA onto the merged base →
`merged + A_new` as a vLLM AR model. Adapt REPRODUCE_V2 §4 export (converted-model = the **merged** candidate, not init):

```bash
cd /home/mark/qwen_diffusion
.venv-fastdllm/bin/python "/home/mark/shared/lumoFlyWheel_codex_fork/scripts/export_qwen35_9b_fastdllm_vllm.py" \
  --official-model "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a" \
  --converted-model "$PWD/models/qwen3.5-9b-fastdllm-mtplus1-merged" \
  --adapter "$PWD/runs/convert_after_rl/Anew_run1recipe_step400_seed80101" \
  --output "$PWD/models/qwen3.5-9b-fastdllm-mtplus1-Anew-vllm-bf16" \
  --overwrite
```

---

## 6. Step 4 — Evaluation battery (a/b/c/d), all sampler-pinned

**Sampler-pinning rule (REPRODUCE_V2 §0) is mandatory for every row:** record git commit, script sha256, sampler function name,
adapter path, dataset path + manifest hash, decode flags, and the value-projection audit file. Re-verify each sha256 against the
table below at run time; if any differs, do not compare to the anchors until the divergence is explained.

| script | pinned sha256 |
|---|---|
| `eval_flare_northstar_hybrid_clean.py` | `a4c66751008390ec44ff4fbb7d025352dc71ba21a005948411883818b908b1f3` |
| `eval_flare_northstar_matched.py` | `4cda3acf752c093a0ee3d3e1208c2cdc5deb064b027d984cdf54e8fa93b6203f` |
| `eval_flare_stage1_ab_diffusion.py` | `eaa78d7a9abfb32b7ab73c7753cf87026741e372ef13a1c0f8e44ead79b5e503` |
| `audit_value_projection_tokens.py` | `7b203e3e8e2a7a7bbfa6f831be295543c728b08d9228bd241f0f07e35a620b40` |
| `export_qwen35_9b_fastdllm_vllm.py` | `6d507ec9ba3308ff7e0f600bc0b5ec7c4ff96f66eff4e4e92175d42af7a119d5` |

### (a) RL-gain survival in DIFFUSION mode — the primary signal

Run the promoted hybrid-clean matched-20 (REPRODUCE_V2 §7) against **base = merged candidate, adapter = A_new**:

```bash
.venv-fastdllm/bin/python scripts/eval_flare_northstar_hybrid_clean.py \
  --input-jsonl data/toolcall_eval_native/flare_scaleup_native_58.jsonl \
  --out-dir runs/convert_after_rl/Anew_matched20_hybrid \
  --episode-limit 20 --min-turns 3 --max-turns 6 \
  --prompt-tokenizer-path models/qwen3.5-9b-fastdllm-b1000-vllm-bf16 \
  --chat-template-path "/home/mark/shared/lumoFlyWheel_codex_fork/docker/chat_templates/qwen3-openai-codex.jinja" \
  --base-model models/qwen3.5-9b-fastdllm-mtplus1-merged \
  --adapter runs/convert_after_rl/Anew_run1recipe_step400_seed80101 \
  --tokenizer-path models/qwen3.5-9b-fastdllm-b1000-vllm-bf16 \
  --no-merge-adapter --block-size 32 --max-new-tokens 384 --top-p 0.95 --temperature 0.0 --grammar-topk 256
```

Also run: never-train hybrid-clean (`flare_nevertrain_bfcl_apibank.jsonl`, `--episode-limit 60 --min-turns 1 --max-turns 8`,
anchor 83/184) for the full diffusion aggregate; and the **careful** matched-20 secondary check (`eval_flare_northstar_matched.py
--backend diffusion --diffusion-condition baseline_careful --diffusion-structural-only`, anchor 44/63). Each followed by its
value-projection audit.

### (b) AR-mode preservation — merged + A_new, guided

Serve `models/qwen3.5-9b-fastdllm-mtplus1-Anew-vllm-bf16` and evaluate with the same guided harness that produced the
merged-AR-guided **127/247** row (`scripts/run_stock_qwen35_ar_guided_controls.sh` → `eval_flare_northstar_matched.py --backend
ar-vllm-guided`), matched-20 + never-train:

```bash
STOCK_MODEL="$PWD/models/qwen3.5-9b-fastdllm-mtplus1-Anew-vllm-bf16" \
OUT_ROOT="$PWD/runs/convert_after_rl/Anew_ar_guided" \
scripts/run_stock_qwen35_ar_guided_controls.sh    # bf16 arm; served-model-name a diffusion-mtplus1-Anew
```

### (c) Retention — GSM8K legacy full-context (the pinned continuity sampler)

REPRODUCE_V2 §4/§5 exact command, `--adapter-b runs/convert_after_rl/Anew_run1recipe_step400_seed80101`, base = the merged
candidate, `--generation-tasks gsm8k --generation-limit 20 --full-context-generation --fresh-generation-blocks --block-size 32
--temperature 0.0 --mask-id 248077 --stop-token-id 248046`. Sampler = `eval_flare_stage1_ab_diffusion.py::full_context_sample_one`.
**Do NOT substitute `measure_block_quality_curve.py` (mutable-remask diagnostic, disqualified).**

### (d) Audits — hard, zero-tolerance invariants (REPRODUCE_V2 §8)

`audit_value_projection_tokens.py` on every diffusion turns.jsonl. Required on both slices:
`projected_value_tokens_exact==0`, `parallel_commit_forced_tokens_counter==0`, `wave1_projected_tokens==0`,
`wave1_value_tokens_counter==0`, `wave2_forced_tokens_counter==0`, `zero_forward_rows==0`,
`verification_mode==no_projection_events`. **Any nonzero ⇒ KILL-3 (the measurement is invalid; the exact_args number cannot be
read as capability).**

---

## 7. Measurement + statistics (both methods, per prior gates)

matched-20 = 20 episodes / 63 turns; never-train = 60 episodes / 184 turns. exact_args is a per-turn binary; turns within an
episode are correlated (error compounding). Two complementary estimators:

1. **Episode-level bootstrap (absolute level + CI vs the pre-RL floor).** Resample the 20 (resp. 60) episodes with replacement,
   B = 10000, recompute exact_args each draw; report A_new's exact_args with a 95% percentile CI. Used to check the CI clears the
   pre-RL region (34) and the half-gain-lost line (40).
2. **Paired-turn McNemar (did re-conversion cost anything vs the no-reconvert baseline C0).** Pair the 63 turns of A_new against
   C0's 47/63 run; discordant counts `b` = C0-correct & A_new-wrong, `c` = A_new-correct & C0-wrong; two-sided exact-binomial
   test on `(b, c)`. Net loss `= b − c`. This is the sharpest "preserved vs eroded" test and matches how hybrid-vs-careful was
   reported historically (+4/−1 paired). Apply the same pairing for AR-mode (b) vs merged-AR-50 and for never-train vs 83.

Single-row rerun rule (REPRODUCE_V2 §10): if GSM8K first20 moves by one row, rerun once and report both seeds.

---

## 8. PASS/FAIL thresholds with justification

Gain preserved = +13 hybrid / +10 careful (34→47/44). Eval noise on 63 turns ≈ ±3–4 turns paired (observed). Thresholds:

| # | measurement | anchor | PRESERVED (PASS) | ERODED (FAIL) | INCONCLUSIVE |
|---|---|---|---|---|---|
| a1 | diffusion hybrid-clean matched-20 | 47/63 | **≥44** AND McNemar net-loss vs C0-47 not sig. (p≥0.05) AND bootstrap 95% LB **≥41** | **≤38** OR bootstrap LB **≤35** OR McNemar sig. net-loss ≥7 | 39–43 → retrain (steps∈{300..600}) + 2nd seed |
| a2 | diffusion careful matched-20 (2nd) | 44/63 | ≥41 and moves consistently with a1 | ≤35 | 36–40 |
| a3 | diffusion hybrid never-train (breadth) | 83/184 | ≥78 AND McNemar vs 83 not sig. | ≤72 | 73–77 |
| b | merged+A_new AR-guided aggregate | 127/247 | **≥122** AND matched-20 ≥47 (McNemar vs 50 not sig.) | **<118** OR matched-20 <44 | 118–121 |
| c | GSM8K legacy full-context (N=20) | 13/20 | **≥13/20** | **≤11/20** (falls to B@1000 floor) | 12/20 → rerun once |
| d | value-projection audits | 0 | all counters 0 | any nonzero (run invalid) | — |

**Justification.** (a1) 44 retains ≥10/13 of the gain (≥ careful RL-v2 level); a ≤3-turn paired drop is inside observed
paradigm/run noise; the bootstrap LB≥41 keeps the CI clear of the 34–40 pre-RL/half-lost region. 38 loses >half the gain; a
CI touching ≤35 is statistically indistinguishable from the pre-RL floor ⇒ erosion. (b) two-stream's premise is a preserved
clean AR stream, so AR mode should barely move; ≥122 (≤5-turn agg loss, ~within paired noise) is required and a matched-20 drop
<44 indicts the recipe. (c) 13/20 is the REPRODUCE_V2 "minimum to proceed"; ≤11 means retention fell to the B@1000 conversion
floor. (d) projection contamination has repeatedly manufactured phantom wins here — it is a run-invalidating hard gate, not a
metric.

**Overall verdict logic.**
- **FLYWHEEL PRESERVES (loop viable):** a1 PASS ∧ b PASS ∧ c PASS ∧ d clean. (a2/a3 as consistency.)
- **FLYWHEEL ERODES (loop eats its gains):** a1 FAIL with d clean and b/c NOT both failing — i.e. the drop is **specifically the
  RL tool-call capability**, while AR-mode and general retention hold. This is the true negative result for step-1.
- **RECIPE CONFOUND (not a loop verdict):** a1 ∧ b ∧ c all drop together ⇒ the conversion recipe damaged the base broadly (over-
  training / wrong rank), not a preservation-specific failure ⇒ KILL-4: retune conversion (fewer steps / rank) and re-judge; do
  not report as a loop finding.

---

## 9. The two required control rows (report side-by-side with A_new)

| row | matched-20 hybrid | matched-20 careful | never-train hybrid | diffusion agg | AR-guided agg | GSM8K N=20 |
|---|---:|---:|---:|---:|---:|---:|
| **C0 — no fresh conversion** (init+RL-v2 = current promoted) | 47/63 | 44/63 | 83/184 | 130/247 | 127/247 | 13/20 |
| **C1 — convert-before-RL** (Run-1 pre-RL → then RL-v2) | Run-1: 34 → RL: 47 | Run-1: 34 → RL: 44 | — | — | — | Run-1: 0.70–0.75 → RL: 0.65 |
| **A_new — convert-AFTER-RL** (this experiment) | measure vs 47 | measure vs 44 | measure vs 83 | measure vs 130 | measure vs 127 | measure vs 0.65 |
| (optional) A_new-strict (B@1000 mix) | measure | measure | measure | measure | measure | measure |

C0 is the level A_new must hold (no-reconvert baseline). C1 is the symmetric mirror: in the historical order conversion produced
34 and RL added +13 → conversion never had to preserve RL; it also fixes the **erosion floor (34)** the FAIL band is anchored to.

---

## 10. Kill criteria (stop, do not interpret preservation)

- **KILL-1** — merge sanity fails (not bit-exact `init+2.0·B@A`, or `mask_token_id≠248077` / `bd_size≠32`). Base is wrong.
- **KILL-2** — A_new conversion GSM8K < 11/20 on the pinned legacy sampler. The conversion itself is broken (below the B@1000
  floor); a preservation reading would be meaningless.
- **KILL-3** — any value-projection audit counter nonzero, or `zero_forward_rows>0`, or sampler path ≠ the pinned function. The
  measurement is contaminated (this class has produced every phantom win in this project).
- **KILL-4** — recipe confound: (b) AR-agg < 118 AND (c) GSM8K ≤ 11 together ⇒ the recipe damaged the base broadly. Retune
  (steps/rank), not a loop verdict.
- **INCONCLUSIVE handling** — a1 in 39–43 after **two seeds** ⇒ report inconclusive; do not force a promote/erode call
  (retrain-freely, but do not extend steps past 600 to manufacture a pass).

---

## 11. GPU-hours + wall-clock estimate (RTX 5090, single GPU, RAM-caged, one process at a time)

| step | GPU-h | note |
|---|---:|---|
| Merge M_{t+1} (Step 1) | ~0.1 | load/save + sanity, mostly IO |
| A_new conversion, 400 steps, 5055 mix (Step 2) | ~0.6 | = Run-1's 2068 s train_runtime |
| Export A_new clean stream (Step 3) | ~0.0 | CPU/shard IO |
| Eval (a) hybrid matched-20 + careful matched-20 + hybrid never-train | ~0.5 | 246 s + 421 s + 390 s compute + loads |
| Eval (b) AR-guided matched-20 + never-train (vLLM) | ~0.3 | ~0.7 s/turn × 247 + vLLM boot |
| Eval (c) GSM8K legacy diffusion full-context N=20 | ~0.4 | full-context diffusion is the slow eval |
| Audits (d) | ~0.0 | CPU |
| **Primary single-seed total** | **~2** | wall ≈ half a day, load-dominated |
| + confirmatory 2nd seed (train + evals) | +~2 | ~4 GPU-h cumulative |
| + optional A_new-strict arm (B@1000 recipe ≤500 steps + evals) | +~4 | ~8 GPU-h cumulative |

**Plan for ~2 GPU-h primary; ~4 GPU-h with the confirmatory second seed (recommended before any verdict); ~8 GPU-h if the
optional strict-mix arm is greenlit.** Wall-clock ~1–1.5 days sequential (model loads dominate).

---

## 12. Provenance checklist (attach to the result report)

Per row: git commit; script sha256 (re-verified vs §6 table); sampler function name; base-model + adapter paths; dataset path +
manifest hash; decode flags; value-projection audit JSON path. Record the merged-base sanity gate (maxabs 0.0), the export
manifest (`replacement_count`, `lora_merge_count`), and both statistical outputs (episode-bootstrap CI + paired-McNemar `(b,c)`
and p) for a1/a3/b. Commit + push each artifact to origin/main per the commit-workflow rule.
