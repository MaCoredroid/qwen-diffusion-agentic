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
