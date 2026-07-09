# CONVERSION_READY — K-carrying re-conversion inputs (do NOT execute here)

**Status:** SWE-SFT arms **both PASS KILL-T1** (2026-07-09 night). D1 recorded in
`swe_tuning_campaign_design.md` STATUS(2026-07-09, night). This file freezes the inputs the
**next, monitor-dispatched** step (the two-stream FLARE re-conversion of M_swe → twin@K1, then
the K-track) needs. Authority: `k_raise_campaign_design.md` §1/§3.9/§4/§6/§7 + the
convert-after-RL #29 protocol. **Nothing here is launched by this turn** (conversion is a
separate GPU-tenant step).

---

## 0. Advancing objects (D1 default)

| role | object | how to build (diffusion-loadable base for re-conversion) |
|---|---|---|
| **PRIMARY** | **M_swe_S** = init+RL-v2 + SWE-SFT | merge `runs/swe_sft_arm1/Aswe_S_step400_seed71101/checkpoint-400` into `models/qwen3.5-9b-fastdllm-mtplus1-merged` |
| **CONTROL TWIN** | **M_swe_T** = stock init + SWE-SFT | merge `runs/swe_sft_arm2/Aswe_T_step400_seed71101/checkpoint-400` into `models/qwen3.5-9b-fastdllm-init` |

Both re-convert (twin@K1 par-power gate decides per-arm). D2 tiebreak = **N=5 AR SWE resolve@1**
(design step 2b), **not yet measured** — the one open input before committing K-track spend to a
single arm.

## 1. Merged SFT weights path / adapter+base recipe (the re-conversion base)

The K-track re-conversion needs a **diffusion-loadable** merged base (mask token 248077 / bd_size
32 / bridge preserved) — **NOT** the AR vllm-bf16 exports (those strip mask/bridge). Use the
HF-stack CPU-exact merge (bit-identical, gate reads maxabs 0.0):

- Script: `scripts/merge_adapter_into_fastdllm_candidate.py --device cpu`
- **M_swe_S (primary):** `--init models/qwen3.5-9b-fastdllm-mtplus1-merged` `--adapter runs/swe_sft_arm1/Aswe_S_step400_seed71101/checkpoint-400` `--out models/qwen3.5-9b-fastdllm-mswe-S-merged`
- **M_swe_T (twin):** `--init models/qwen3.5-9b-fastdllm-init` `--adapter runs/swe_sft_arm2/Aswe_T_step400_seed71101/checkpoint-400` `--out models/qwen3.5-9b-fastdllm-mswe-T-merged`
- Adapter recipe (both, frozen): r16/α32 dropout0.05, W += 2.0·(B@A); 11 targets = q,k,v,o + GDN in_proj_{qkv,z,a,b} + out_proj + **MLP gate_up_proj/down_proj** (`lora_merge_count=184` verified on the AR export of both arms). Merge sanity gate must PASS (mask 248077, bd_size 32, has_weights). ANY failure ⇒ KILL-1.
- AR-serving exports already built (for the anchor/AR-SWE evals, not for re-conversion): `models/qwen3.5-9b-fastdllm-mswe-S-vllm-bf16`, `-mswe-T-vllm-bf16`, `-stockinit-vllm-bf16`.

## 2. Conversion data / curriculum (leakage firewall — the sharp-test premise)

- **Re-conversion mix:** `data/flare_redesign_run1_copy_retention_mix` (Run-1 copy+retention; **excludes the RL-v2 pool AND the SWE-SFT pool** — the re-conversion is *not* trained on the capabilities it must preserve; SWE capability lives in the merged base weights, per `swe_tuning_campaign_design §3.1`).
- **L1-SWE census (do FIRST, ~0.5 GPU-h, may pre-KILL the K-track; §6.1):** run the CAD sampler counters over `runs/swe_datagen_s1/keepers` + a decode-only pass on Tier1-C46 → measure code content mix (grammar/value/structural %) and top-1 conditional entropy on reasoning-vs-value spans; calibrates `f_value` (est. 0.25–0.40 for code) and tells a-priori whether the entropy wall transfers.
- Firewall re-assert at train-launch AND eval-launch: `build_frontier.py::firewall_assert` (KILL-D1). Current `data/swe_sft_pool/pool_manifest.json::kill_d1_check` = **PASS** (`intersect_eval_holdout=0`, enforced_holdout = inner5 ∪ tier0_20 ∪ tier1_100).

## 3. K schedule (§6 staged targets; decode-only first, then twin@Kc only if it stalls)

| rung | k_max | pass bar (blended avg tok/fwd) | γ sweep | train arm | on PASS / SPEED-FAIL / PAR-KILL |
|---|---|---|---|---|---|
| **K1.5** | 2 | ≥1.5 (reasoning-span) | {0.7,0.8,0.9} | decode-only; twin@Kc if stalls | ship K1.5→K2 / try Kc→else stop-ship K=1 / revert twin@K1 |
| **K2** | 2 | ≥2.0 | {0.6,0.7} | twin@Kc (O1+O2, curriculum→k2) | ship K2→K4 / ship K1.5 / revert |
| **K4** | 4 | ≥ ~2.1 blended (ambitious) | push low-γ | twin@Kc (curriculum→k4) | ship K4 / ship last-pass / revert |

- twin@Kc objective (§4): two-stream FLARE (`FASTDLLM_FLARE_TWO_STREAM=1`; clean stream `L_AR` byte-identical to AR), **plain `L_diff`** = twin@K1; **twin@Kc adds** O1 frontier-adjacency mask schedule (train==serve) + O2 span-class weighting (**VALUE spans standard K=1, `VALUE_SPAN_LOSS_WEIGHT=2.0`**, reasoning spans K-consistency), ramped by O3 K-curriculum. 400 steps, ~0.57 GPU-h/seed × 2 seeds. **Promotion discipline:** credit training only if twin@Kc beats decode-only-on-twin@K1 at par.
- Decode instrument: entropy-gated adaptive-K (`k_max` 2→4), **values FSM-forced K=1 / sub-γ position blocks the run**, native EOS-stop, never remask.

## 4. ENTRY GATE — twin@K1 par-power precondition (§1.1) — do not skip

- Build **Tier1-C46** = `Tier1-100 ∖ (w2_n50_ids ∪ gate_ladder_5)` (≈46–48 instances, pristine — never used in any prior tuning/eval decision). Emit `k_raise_pool_manifest.json` (id list + per-id source ring + `pool_sha256`). **KILL-D1 hash asserts (build+run time):** `Tier1_C46 ∩ train_ids = ∅` · `∩ w2_n50_ids = ∅` · `∩ gate_ladder_5 = ∅` · `Tier1_C46 ⊂ tier1_100`. Any nonzero intersection ⇒ do not eval.
- **Entry bar:** twin@K1 (M_swe re-converted, `L_diff` plain, decoded K=1) must **resolve ≥12/46 (≈26 %)** on Tier1-C46 (the floor at which McNemar has power to detect a 3–4 resolve loss). If below floor ⇒ **INCONCLUSIVE**, do not spend K rungs (the SWE-SFT base is too weak; escalate per USER_LEVER_BELT).
- Par reference is **always twin@K1** (its own K=1 self, paired, same seed) — never AR. Primary stat: paired McNemar on shared instances; temp-0.6 twin@K1 same-seed is the par reference for the trained arm.

## 5. Serving cert plan (the diffusion twin must serve at held exactness)

- **Anchor (§2.3, the certified capability must not move):** tool-call matched-20 exact_args, **diffusion** twin served via `hf_route_i_flare` per-call-waves / P2 engine, **adaptive-K OFF on the tool-call path (values FSM-forced K=1)**. Anchor = twin@K1's own matched-20 exact_args; PASS = McNemar net-loss vs anchor not significant. (AR-mode anchor already banked this turn: S 49/63, T 51/63, both b=c=0.)
- **Retention (§2.4):** GSM8K legacy full-context N=20, temp 0.0, seed 20260701; anchor = twin@K1 GSM8K (≈13/20 conversion floor); PASS ≥ anchor; KILL-retention ≤ anchor−2 OR in-training rolling KL-to-base > 0.05 unrecovered.
- **Audits (§2.5, KILL-3):** value-projection audits **all-0**, zero tolerance.
- **Byte/parity cert:** the twin@K1 clean stream is K=1 byte-exact at temp 0 (the 6/6 snapshot-restore discipline); reuse the P2-engine online-vs-offline + APC multi-turn certs (`runs/stage_a_cert/`) for the served path.

## 6. Budget + discipline

- K-track total through K4 ≈ **20–28 GPU-h** (dominated by SWE par-eval occupancy at ~21 eps/GPU-h, not train); each trained rung's re-conversion is the ~2–4 GPU-h class. Host cage 22G, **one GPU tenant**. Do not extend past 600 steps. Infra track B-P1 (twin eps/GPU-h) is what makes later rungs affordable.

## 7. Open item before single-arm commit

- **N=5 AR SWE resolve@1 (both arms), design step 2b** — the D2 tiebreak (max SWE resolve × anchor-held). Not yet measured. Until then, D1 default = **S primary / T control twin**; both re-convert.

---

## STATUS(2026-07-09, eve) — PRIMARY arm M_swe_S re-conversion steps 1-3 EXECUTED (merge → convert → export)

Monitor-dispatched. Ran the certified #29 protocol (merge → two-stream conversion → clean-stream export)
VERBATIM on the SWE-SFT primary. Only the base, output paths, and seed differ from the #29 A_new run.
Gate steps 4-5 (engine byte/quality cert · anchor matched-20 · Tier1-C46 twin@K1 entry ≥12/46) are the
NEXT step, not run here.

- **STEP 1 — merge (PASS, KILL-1 not triggered):** `scripts/merge_adapter_into_fastdllm_candidate.py --device cpu`
  folded SWE-SFT arm-1 (`runs/swe_sft_arm1/Aswe_S_step400_seed71101/checkpoint-400`, 11-target QLoRA, W += 2.0·B@A)
  into `models/qwen3.5-9b-fastdllm-mtplus1-merged` → **`models/qwen3.5-9b-fastdllm-mswe-S-merged`** (diffusion-loadable;
  mask 248077 / bd_size 32 / has_weights preserved). Merge sanity gate: `merged == init+2.0·B@A` maxabs **0.0**
  (layer0 GDN in_proj_qkv, layer3 attn o_proj), scaling 2.0. 23s CPU wall, caged 22G. Gate JSON:
  `runs/kraise_reconvert/step1_merge_mswe_S/merge_sanity_gate.json`. (Merged-base manifest lineage label corrected
  to M_swe_S/SWE-SFT for honest provenance.)
- **STEP 2 — convert (COMPLETE, twin@K1):** `scripts/kraise_reconvert_mswe_S_driver.sh` (detached, caged, single
  continuous 400-step cosine — the reference the #29 4-chunk resume reproduced). Recipe = Run-1 two-stream FLARE,
  9-target conversion LoRA (q/k/v/o + in_proj_{qkv,z,a,b} + out_proj), BLOCK_SIZE 512 / TRAIN_BD_SIZE 32 (its own
  feasible block, **NOT** the SFT 12288), value_span_weight 2.0, LR 1e-5 cosine, seed **81101**. Mix =
  `data/flare_redesign_run1_copy_retention_mix` (excludes RL-v2 AND SWE-SFT pools; leakage firewall — SWE capability
  lives in the merged base). **Wall 2016 s (33:36) @5.04 s/it**, ≈ #29's 2034 s. Loss curve sane: 80 pts, min/max/mean
  **1.602 / 5.686 / 3.602**, no NaN/Inf, seg-means 3.86→3.68→3.68→3.19; LR peak 1.0e-5 → final **1.639e-10** (bit-matches
  #29's cosine-400 horizon). Adapter 304 tensors = 152 pairs, all lora_B nonzero, 0 nonfinite.
  Output: **`runs/kraise_reconvert/mswe_S_twinK1_run1recipe_step400_seed81101/`** (`adapter_model.safetensors` +
  `trainer_state.json` + checkpoint-{100..400}).
- **STEP 3 — export (DONE) + smoke (PASS):** export script sha `6d507ec9…` (matches design §6 pin) → **`models/qwen3.5-9b-fastdllm-mswe-S-twinK1-vllm-bf16`**
  (official Qwen3.5 conditional layout, mask stripped; dual-loadable stock-AR-vLLM + FLARE engine). Profile:
  mapped_text_tensors **427** / replacement_count **427** / lora_merge_count **152** / lora_scale 2.0 — matches the RL-v2
  export profile. Loads in the FLARE vLLM pin (`.venv-vllm-p2-main`, `VLLM_USE_FLASHINFER_SAMPLER=0`); 3-prompt smoke
  (frozen envelope 0.6/0.95/20) coherent + grammar-valid: reasoning + Python `fib` correct; tool prompt emits exact
  native `<tool_call><function=get_weather><parameter=location>Paris…` grammar.

**Ready for the gate:** twin@K1 = base `models/qwen3.5-9b-fastdllm-mswe-S-merged` + adapter
`runs/kraise_reconvert/mswe_S_twinK1_run1recipe_step400_seed81101` (diffusion, `--no-merge-adapter`, block 32, K=1);
served/AR-guided via `models/qwen3.5-9b-fastdllm-mswe-S-twinK1-vllm-bf16`. NO K>1 work done (separate decision on the gate).

---

## STATUS(2026-07-09, night) — STEP 4 CERT EXECUTED (serving byte/quality + anchor preservation) — BOTH PASS

Monitor-dispatched. Ran step-4 of the #29 convert-after-RL protocol on the M_swe_S twin@K1: the A6-style
online==offline engine cert + the anchor-preservation check. **Both PASS; no kill fired.** Step 5 (Tier1-C46
twin@K1 entry gate ≥12/46) is the NEXT step, NOT run here. NO K>1 work.

### Part A — A6-style ENGINE byte/quality spot-cert (online == offline), twin export

Served the twin export `models/qwen3.5-9b-fastdllm-mswe-S-twinK1-vllm-bf16` on the FLARE vLLM pin
(`.venv-vllm-p2-main` + `/home/mark/shared/lumoFlyWheel_codex_fork/scripts/qwen35_9b_flare_hybrid_serve.sh`,
`policy=hybrid_clean decode=hybrid_clean flare=1 canvas=32 bidir=1 mask=248077 temp=0.0`). **Launcher-gap note:** the
twin export's `conversion_manifest.json` carries `mask_token_id` at top-level (not under `base_model`), so the
launcher's parser resolved mask=None — passed `MASK_TOKEN_ID=248077` explicitly (offline mirror reads it from
`matched20_ref.json`). Offline in-process engine + online AsyncLLM server captured on the same A6 (10 matched-20
turns, fresh APC) + A7 (10 turns, warm APC) turnset; `compare` joins them.

| set | n | token-ident online==offline | byte-ident online==offline | quality-ident (exact+valid) | zero-value-projection | divergent turns |
|---|---:|---:|---:|---:|---|---|
| A6 (single-turn, fresh APC) | 10 | **10/10** | **10/10** | **10/10** | True | none |
| A7 (multi-turn, warm APC) | 10 | **10/10** | **10/10** | **10/10** | True | none |

**PASS** — the served twin is byte-identical to the offline engine on every certified turn (offline booted
mask=248077, boot 14.5 s). The `off_reproduces_battery_*` cross-check (twin-offline vs the RL-v2 gates2 battery — a
*different* model, so NOT the twin cert) reads A6 8/10 byte · 9/10 exact, A7 9/10 byte · 10/10 exact, i.e. the twin
tracks RL-v2 on most tool-call turns and legitimately diverges on the rest. Cert JSON:
`runs/kraise_reconvert/stage_a_cert_mswe_S/cert.json` (+ scripts `capture_offline_twin.py`/`online_client_twin.py`/
`compare_twin.py`, adapted from the certified `runs/stage_a_cert/` A6 harness).

### Part B — ANCHOR preservation: twin diffusion hybrid-clean matched-20 vs the AR arm 49/63 (#29 bar)

`eval_flare_northstar_hybrid_clean.py` (`.venv-fastdllm`, base `models/qwen3.5-9b-fastdllm-mswe-S-merged` + adapter
`runs/kraise_reconvert/mswe_S_twinK1_run1recipe_step400_seed81101`, `--no-merge-adapter`, block 32, temp 0.0,
top-p 0.95, grammar-topk 256, K=1 FSM values). Paired McNemar vs the banked AR-arm anchor (`runs/swe_sft_arm1/
anchor_gate/mswe_S_matched20/ar-vllm-guided/turns.jsonl`, 49/63), `gold_sha256` mismatch **0/63**.

| metric | twin@K1 diffusion | AR arm anchor | bar | verdict |
|---|---:|---:|---|---|
| exact_args | **50/63** | 49/63 | raw ≥ anchor−3 = 46 | **PASS** (+1) |
| valid_tool_call | 63/63 | 63/63 | — | held |
| episode_exact | 14/20 | 13/20 | — | +1 |
| McNemar b (AR-right, twin-wrong) | **0** | — | — | zero erosion |
| McNemar c (twin-right, AR-wrong) | **1** (`heldout_seed_run1clean_0031#t1`) | — | — | a gain |
| net-loss b−c / p (two-sided exact) | **−1** / **1.0** | — | not significant (p≥0.05) | **PASS** |

**VERDICT PASS** — the SWE-SFT+conversion twin, decoded in the served diffusion lane, is a strict **superset** of the
AR arm on the certified tool-call anchor: `b=0` (lost nothing the AR arm had) and gained one turn. Also ≥ the #29
diffusion anchor (47/63, +3). **KILL-3 value-projection audit CLEAN** on the twin turns.jsonl:
`verification_mode=no_projection_events`; every hard counter 0 (`projected_token_record_count`,
`parallel_commit_forced_tokens_counter`, `wave1/wave2_*`, `zero_forward_rows`,
`exact_rows_dependent_on_projected_values` all 0) — the 50/63 is uncontaminated (no phantom-win).

### Compute discipline
Every heavy step caged `systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=4G`; one GPU tenant (offline
capture → boot server → online client → **server killed before** the CPU compare). GPU wall ≈ hybrid m20 ~7 min +
offline capture ~3 min + server boot ~1.5 min + online client ~40 s. Artifacts under gitignored
`runs/kraise_reconvert/{step4_cert_mswe_S,stage_a_cert_mswe_S}/` (`step4_cert_summary.json`,
`anchor_mcnemar_twin_vs_ar49.json`, `cert.json`, `.../projection_value_audit.json`, twin turns.jsonl).

**Gate readiness:** twin@K1 anchor+serving certified. Step 5 = 46-episode Tier1-C46 twin@K1 through qwen-code on the
FLARE engine (official swebench images + docker scoring, frozen envelope, entry bar ≥12/46) is the next, separate
step. NO K>1 work (K-curriculum is a separate decision on the gate result).
