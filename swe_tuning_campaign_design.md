# SWE-Tuning Campaign — Design (give the diffusion twin SWE-capable base weights)

**Author:** design synthesis, 2026-07-05. **Task:** #54 follow-on (the Stage-C `stage_c_n5v2` pause decision).
**Mode:** CPU-only design; no CUDA run. All GPU-hour / yield numbers are honest estimates flagged "measure, never
assume." **Owner discipline:** [[qwen-diffusion-commit-workflow]] (commit+push each step, narrate reasoning),
[[diffusion-promotion-discipline]] (promote only on raw/constrained model-only gains), [[native-function-format-rule]]
(qwen3_xml native tool format consistently across gen/train/eval), [[retrain-freely-rule]], [[gpu-utilization-standard]].

---

## STATUS (2026-07-06) — PREMISE DISSOLVED at n=5; campaign PARKED; D1 repriced GO

**Read this before the rest of the doc.** This campaign was designed to **recover the −2 RL-v2 SWE
"wrong-payload" tax** measured in the v2 *greedy* ladder. The sampling-corrected re-run
(`runs/stage_c_n5v3/report.md`, reference envelope temp 0.6 / top_p 0.95 / top_k 20, official docker)
**erases that tax:**

> **v2 greedy:** stock-AR 4/5 > merged-AR(RL-v2) 2/5 > diffusion(RL-v2) 1/5 > diffstock 0/5
> **v3 envelope:** stock-AR **3/5 == merged-AR 3/5 == diffusion 3/5** > diffstock 1/5

The three model arms resolve the **identical** instance set {django-11119, django-13741, pytest-8399};
paired McNemar discordance **b = c = 0** for every pair. **There is no −2 to recover** — the greedy
−2 weights tax and −1 paradigm tax were sampling artifacts (retired). So:

- **PREMISE DISSOLVED.** The doc's §0 FRAME (below) rests on "RL-v2 is the wrong SWE payload (−2 even as
  AR)" and "the diffusion paradigm compounds it (−1)". **Both findings are withdrawn** at this
  resolution. The FRAME and §2.2 base-choice argument (S-recovers-the-tax vs T) are **preserved below as
  the historical greedy-era rationale**, but they no longer motivate spend.
- **CAMPAIGN PARKED (not killed).** The machinery is sound and the data path is GO-priced — the Stage-0
  probe v2 (`runs/stage0_swegym_probe_v2/report.json`) measured **corrected yield 5/20 = 0.25**, which
  **clears the 0.20 GO bar**, with patch_produced 19/20 (empty-patch loss fixed) and env-images that
  *pull* prebuilt. But there is currently **no measured deficit for it to close.** Do not spend the
  ~80–150 GPU-h until a properly powered run shows a gap.
- **D1 → GO-IF-EVER-NEEDED.** The phase-3 D1 ledger below (yield 0.15 → ADJUST) is superseded by the
  envelope reprice: **yield 0.25 ≥ 0.20 ⇒ GO_single_attempt** on the generator economics. D1 is no
  longer the blocker; the blocker is the dissolved premise. If a future N=25–50 result *reopens* a
  diffusion-vs-AR gap, this campaign is the pre-priced, GO-gated response — unpark it then.
- **THE JUSTIFIED NEXT STEP is the N=25–50 horse race** (new section at the end of this doc), not this
  training campaign. The v2 "N=25–50 is premature, it measures a known gap" objection is void — there is
  no known gap; the honest prior is the tie, and only a powered run can detect or bound a real paradigm
  difference.

*(D1 reprice detail: the probe-v2 GO-bar and campaign price are updated in §1.4 and §5.1 below; the
original greedy-yield ledger is retained there as the superseded record.)*

---

## STATUS (2026-07-08) — datagen LIVE: false-kill reversed, L1 near-miss-first re-strat applied

The `runs/swe_datagen_s1` data-gen orchestrator is **running** (dual-source SWE-Gym + Verified-adjacent,
official/fork docker scoring, best-of-k). Trajectory to the **400-keeper floor**, corrected:

- **Keepers 236 / 1000** (floor 400, not yet met); attempts_real 828; lifetime_yield 0.295;
  rolling_yield(200) 0.35 — **CONTINUE** (kill bar 0.10/200). The 2026-07-07 false kill (a
  score-side fork-harness crash miscounted as a yield collapse) is reversed; 209 poisoned gym rows
  → `infra_invalid`, 50 re-scored → +14 keepers (218→232→**236**). See `runs/swe_datagen_s1/KILL_AUTOPSY.md`.
- **L1 lever applied live (2026-07-08):** the officially-scorable VA head is exhausted; the recoverable
  value is **best-of-k re-draws of the 479 near-misses** (non-empty failing patches) at their measured
  family yields. Frontier re-stratified to a **near-miss exploit head** (family-yield-ranked, sphinx-doc
  0.0 last) + unattempted coverage tail (near-zero getmoto/facebookresearch last, not dropped), with
  `best_of_k.exploit_priority="frontier"` and a gated ledger change so the frontier order draws
  authoritatively. KILL-D1 hash-asserted (sha==pin `c56f473…`, frontier ∩ 113-id eval-holdout == 0).
  Details: `runs/swe_datagen_s1/L1_RESTRAT_NOTE.md`, `restratify_frontier_l1.py`.
- **Corrected trajectory:** 479 near-misses × best-of-3 at measured yields ⇒ **E[+100–200] keepers**
  ⇒ 236 + ~100–150 **clears the 400 floor** off L1 alone; mid-yield unattempted coverage
  (python/iterative/modin/dask ~0.18) is the cushion; the 0.10/200 kill bar governs the low-yield tail
  honestly. The 1000 target still needs new imaged sources (L4) or an accept-and-SFT decision (L5).

---

## STATUS (2026-07-09) — ARM-1 SWE-SFT **ARMED, blocked on GPU handover**; design block_size 32768 measured **INFEASIBLE**

Dataset built + validated + committed (`6a8a0c7`): **323 clean keepers**, firewall/quarantine asserted.
Arm-1 = merged-RL-v2 `models/qwen3.5-9b-fastdllm-mtplus1-merged` (HF-stack realization of arm S).
Full launch machinery is built + preflight-verified but **NOT launched** — two hard blockers
(detail: `runs/swe_sft_arm1/ARM1_LAUNCH_STATUS.md`):

1. **GPU held by live datagen.** `datagen_orch.sh` is generating `batch_0005` (auto-advanced past
   the batch_0004 the brief expected to be last); the 27B teacher holds ~30 GB @ 98 % util. Handover
   (27B GPU teacher stops; Opus API track continues on CPU) has not happened. The launcher's preflight
   **refuses to fight the datagen** (aborts exit 9 while GPU busy) — verified.
2. **block_size 32768 (§2.3) is infeasible on the 32 GB 5090.** The certified two-stream FLARE trainer
   concatenates clean+noisy to length **2L**, so LM-head logits are `[2L, vocab 248320]` ≈ 16 GB at
   L=16384 plus an O((2L)²) mask ⇒ **measured OOM at block 16384**. Two-stream tops out ≈ **8192**
   (48.9 % assistant-label retention; 12288 → 69.6 % to be probed at handover; 24–32 k far-OOM). The
   §2.3/§5 "block up to 32768 / 6–12 GPU-h" line did not price the vocab-248k × 2L logits term. Resolved
   by measurement: **train at the largest block that fits** (launcher auto-selects via a 2-step probe
   ladder). If retention proves inadequate, retention-recovery is chunked/fused-CE or a single-stream
   path (retrain-freely follow-ups).

**To launch at handover:** `bash scripts/swe_sft_arm1_driver.sh` (preflight → feasibility ladder → full
323-row dataset build → 2-step verify-smoke → detached caged resumable run, pidfile + metrics + 100-step
erosion-sweep checkpoints). Config frozen per §2.3/§2.4 (r16/α32/drop0.05; targets q,k,v,o + GDN
in_proj_{qkv,z,b,a}+out_proj + **MLP gate_up/down**; LR 1e-5 cosine warmup 0.03; HORIZON 400; seed 71101),
objective = two-stream FLARE via the S2 pretok passthrough (serve-exact ids, native qwen3_xml; L_AR is the
"AR-side SFT" signal). Anchor gate (§2.5 / KILL-T1) runs at each 100-step checkpoint.

---

## STATUS (2026-07-09, later) — HANDOVER EXECUTED: trainer AMENDED to AR single-stream QLoRA; block **12288** measured; LAUNCHED

The GPU→training handover fired (monitor decision; datagen orch stopped, `batch_0005`
sacrificed, GPU settled 387 MiB/0%). Resolutions (2)+(3) executed. **Two amendments to
the §2 plan, both by measurement, recorded here per the "amend-in-the-status-block-
with-evidence" discipline:**

**AMENDMENT A — training mode: two-stream FLARE → AR-style SINGLE-STREAM CAUSAL QLoRA.**
The §2.3/STATUS(2026-07-09) two-stream FLARE path materialises `[2L, vocab 248320]`
logits and OOM'd above block 8192. It is REPLACED by a plain autoregressive
single-stream causal QLoRA on the same merged-RL-v2 base
(`models/qwen3.5-9b-fastdllm-mtplus1-merged`), `scripts/swe_sft_arm1_qlora_train.py`.
- **Objective:** AR next-token CE, shift-by-one (identical to the FLARE clean-stream
  `L_AR`, `modeling.py:2149`), loss on assistant-label spans only, mean over valid
  targets. This IS the "AR-side SFT" the design calls for — now without the diffusion
  co-stream (which §3.1's fresh conversion re-supplies anyway, on a SWE-excluded mix).
- **EVIDENCE the amendment is sound (the #29 chain):** convert-after-RL preservation
  audit (**#29**, `convert_after_rl_result.md`, commit **b019b86**) certified that
  **plain training + fresh re-conversion PRESERVES fresh gains** (McNemar **zero net
  loss**, two seeds, 126 paired turns × 2). train==serve parity is enforced at the
  **CONVERSION stage** (`k_raise_campaign_design.md`), **not** required bit-for-bit
  during SFT. So the SFT stage is free to use a memory-lean AR objective + a numerically-
  (not bitwise-) equivalent attention/GDN kernel; the served twin is re-certified at
  conversion by the #29 protocol (§3.1) + the matched-20 anchor gate (§2.5).
- **How it fits long sequences (the two-stream OOM root-cause was NOT the logits alone):**
  (1) SDPA causal attention monkeypatched onto the 8 `full_attention` layers **in the
  trainer process only** (the shipped forward is EAGER — `attn_weights [1,16,L,L]`
  ~34 GB/layer@32k — the true wall; served `modeling.py` is untouched); (2)
  `FASTDLLM_GDN_KERNEL=fla` fused GDN (O(L)) replacing the memory-heavy torch fallback;
  (3) chunked/gradient-checkpointed `lm_head`+CE so `[L, vocab]` (16 GB@32k) is never
  materialised; (4) 4-bit NF4 QLoRA base + per-layer gradient checkpointing.

**AMENDMENT B — block_size: measured, `12288` (not 32768/24576/16384).** Dry 2-step
caged fwd/bwd probes on the worst-case (longest-8 truncated to the block), RTX 5090
31.3 GiB usable, GDN=fla + expandable_segments:

| block | peak (GiB) | margin | verdict |
|---:|---:|---:|:--|
| **32768** (design) | OOM | — | infeasible (fla-kernel `chunk_fwd_o`) |
| **24576** | OOM | — | infeasible |
| **16384** | 29.4 | ~1.9 | fits worst-case probe but **thin margin** — rejected for an unattended run on a live-desktop GPU |
| **12288** ✅ | **24.8** | **~6.5** | **CHOSEN — robustly stable** |
| 8192 | 21.4 | ~9.9 | (floor; the two-stream ceiling) |

The monitor's candidate set {16384, 24576, 32768} is measured INFEASIBLE-or-thin;
the ladder was extended downward (as the driver anticipated) to the largest robustly-
stable block. Single-stream+fla lifts the feasible block **1.5×** over the two-stream
~8192 ceiling. Reducing the CE chunk did not move 16384's peak (the binding term is
the transformer backward, not CE).

**Truncation stats @ 12288 (front-truncation keeps the final edit-and-verify turns —
the taught capability):** 334 rows, **328 left-truncated**, **0 zero-label after trunc**
(every episode retains assistant targets), assistant-label retention **69.88 %**
(911,531 / 1,304,354 target tokens). Dataset rebuilt from the **334-keeper** pool
(323→334, +11 all 27B-teacher/SWE-Gym; firewall re-asserted: holdout 113 sha==pin,
keeper∩holdout=0, quarantine excluded).

**Config (frozen, per §2.3/2.4 except the measured block):** r16/α32/drop0.05; targets
q,k,v,o + GDN in_proj_{qkv,z,b,a}+out_proj + MLP gate_up/down; LR 1e-5 cosine warmup
0.03; HORIZON 400 (≤600); seed 71101; per-device bsz1 grad-accum1; SAVE_STEPS 100 →
{100,200,300,400}; caged MemoryMax=22G; faithful chunked-resume (adapter+optim+sched+
rng+step + resume manifest). **LAUNCHED detached** via `bash scripts/swe_sft_arm1_driver.sh`
(`runs/swe_sft_arm1/Aswe_S_step400_seed71101/`, pidfile `runs/swe_sft_arm1/train.pid`,
metrics `runs/swe_sft_arm1/metrics.jsonl`). Anchor gate (§2.5 / KILL-T1) runs at each
100-step checkpoint. **QLoRA-merge caveat** (adapter trained on the 4-bit base merges
into the bf16/dequantised base at §3.1 re-conversion — the standard QLoRA merge; the
preservation battery + matched-20 anchor certify no erosion; retrain-freely applies).

**LIVE CONFIRMATION (2026-07-09T20:06Z — launch verified healthy).** The detached caged
run is stable and will outlive the launching session: **pid 185042, PPID 1**, systemd
scope `run-p185042-i4368305.scope` **MemoryMax=22 GiB verified**, env `FASTDLLM_GDN_KERNEL=fla`
+ `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. **GPU 100 % sm-util sustained**
(`nvidia-smi dmon` 18 × 10 s = 3 min, **min = mean = 100 %**), 27.3 / 32 GB resident,
per-step **peak 26.67 GiB** (~4.7 GiB live margin, matches the 12288 probe). Steady
state **7.40 s/step** (step 5→45, 5-step cadence). Loss **finite, no NaN/inf**; the
minimum-loss steps trend down (0.270 → 0.204 → 0.182 → 0.171) with per-step variance
tracking `n_valid` (assistant-target token count per row: 1196–4302), **not** divergence;
grad_norm 0.23–0.56; LR warming toward the 1e-5 cosine peak (warmup 12). **ETA ≈ 49 min**
wall for 400 steps; the 100-step erosion-sweep checkpoints {100,200,300,400} land at
≈ **12 / 25 / 37 / 49 min** from launch (checkpoint-100 not yet written at this note).
Resume verified `--resume auto` (latest-checkpoint restore of adapter + optimizer +
scheduler + torch/cuda/python/numpy RNG + step) — a kill/resume reproduces the fixed
cosine horizon bit-faithfully. **Datagen handover clean:** orch pid 2888673 gone,
`datagen_ar.scope` inactive (not failed), 0 `swe_ep` containers, trainer is the sole GPU
tenant (`batch_0005` sacrificed, ids re-drawable — `runs/swe_datagen_s1/DATAGEN_STATUS.txt`).
**Monitor's next gate = the ANCHOR GATE at each 100-step checkpoint** (§2.5 / KILL-T1;
matched-20 exact_args McNemar vs arm-S pre-SFT anchor ≈ 47/63 hybrid · 44/63 careful,
PASS = net-loss not significant AND raw ≥ anchor − 3; secondary GSM8K N=20 ≥ anchor − 1,
value-projection audit all-0).

---

## STATUS (2026-07-09, evening) — ARM-1 COMPLETE + ANCHOR GATE **PASS** (zero erosion); ARM-2 (stock T) LAUNCHED

**Arm-1 (M_swe_S) finished clean and PASSED the KILL-T1 anchor gate with the sharpest
possible evidence — turn-for-turn tool-call preservation (McNemar b=c=0).**

**Arm-1 final training summary** (`runs/swe_sft_arm1/Aswe_S_step400_seed71101`, `metrics.jsonl`,
80 logged points, steps 5→400): **DONE step=400, wall 2963.1 s (7.41 s/step)**, no crash.
Loss curve **healthy, 0 NaN/Inf**: first 0.270 → last 0.151 (min 0.051, mean 0.265); window
means decline monotonically **0.262 [5–50] → 0.207 [100–150] → 0.195 [200–250]**. The only
grad_norm>10 event is step 365 (grad_norm 280, loss 4.20) — an isolated **n_valid=1**
single-target-token row at the cosine tail (LR 2e-7 ⇒ negligible weight impact); steps
370–400 are all 0.06–0.25 / grad_norm 0.28–0.64. Per-step peak 26.67 GiB (12288 block, matches
the probe). **Final adapter: `runs/swe_sft_arm1/Aswe_S_step400_seed71101/checkpoint-400/adapter_model.safetensors`**
(r16/α32, 11 targets incl. MLP gate_up/down; 26.5 M trainable params, 0.295 %).

**Anchor gate (§2.5 / KILL-T1), AR-mode, tool-call matched-20 exact_args — PASS.** Served both
arms through the **identical** export→vLLM→`eval_flare_northstar_matched.py --backend
ar-vllm-guided` path (greedy, byte-comparable; the standard AR-guided path of #29 row b / #28
merged-AR), same 20-episode/63-turn native set (`flare_scaleup_native_58.jsonl`, gold_sha256
mismatch **0/63**):

| metric | pre-SFT base (rlv2, init+RL-v2) | post-SFT (M_swe_S) | banked anchor | verdict |
|---|---:|---:|---:|:--|
| tool-call matched-20 **exact_args** | **49/63** | **49/63** | 50/63 (#29 b; #28 127/247 C0) | raw 49 ≥ 47 (anchor−3) **PASS** |
| valid_tool_call | 63/63 | **63/63** | 62/63 (A_new) | no format erosion **PASS** |
| episode_exact | 13/20 | 13/20 | — | equal |
| **paired McNemar** (pre-right/post-wrong `b`, post-right/pre-wrong `c`) | — | **b=0, c=0, net-loss 0, p=1.0** | net-loss not sig | **PASS** |

**Not a no-op:** the SFT model is behaviorally active — **5/63 turns diverge** in greedy
generated text/parsed calls (e.g. argument values "SKU001" vs "product_001", amount 10000 vs
50000) — yet **none of those flips changed a correctness outcome**, so the certified tool-call
exactness is preserved turn-for-turn (mirrors #29 a1's b=0). The pre-SFT base re-measures at
**49/63** through this pipeline (vs the banked 50/63 — a 1-turn serving drift; the banked point
was the A_new/C0 export), so the paired comparison (both 49/63, identical correct-set) is the
authoritative byte-comparable result. **Value-projection audit N/A** (AR mode has no diffusion
projection events; KILL-3 is a diffusion-only probe). Export merged the full SWE target set
correctly: `lora_merge_count=184` (RL-v2 was 152; +32 = MLP gate_up/down × 16 layers), no
dropped targets. Artifacts: `runs/swe_sft_arm1/anchor_gate/{mswe_S_matched20,rlv2_base_matched20}/ar-vllm-guided/turns.jsonl`,
`anchor_mcnemar_result.json`, `export_mswe_S.log`; served model
`models/qwen3.5-9b-fastdllm-mswe-S-vllm-bf16`; scorer `scripts/swe_sft_arm1_anchor_mcnemar.py`.
**Secondary GSM8K N=20 retention (§2.5 row 2) was NOT run in this gate pass** — deferred, not
blocking (KILL-T1 is the load-bearing blocking gate; #28 already bounds merged-AR GSM8K
within-noise). Recommend the monitor run it before D2.

**⇒ KILL-T1 does NOT fire for arm S. Arm-2 launched** per the design (arm T control, §2.2):
IDENTICAL config/pool/steps (block 12288, 334-keeper pool, horizon 400, seed 71101, r16/α32,
same 11 targets, LR 1e-5 cosine warmup 0.03) on the **stock base**
`models/qwen3.5-9b-fastdllm-init` (Fast_dLLM conversion of stock Qwen3.5-9B, **no RL-v2 merge**;
conversion_manifest lineage = stock). Detached+caged (`scripts/swe_sft_arm2_driver.sh`, pid
188960 PPID 1, scope `run-p188960` MemoryMax=22G, `runs/swe_sft_arm2/`), **verified healthy:
GPU 100 % util sustained, step-5 loss 0.263 / grad_norm 0.31 (sane), peak 26.67 GiB, trainable
26.5 M (0.295 %), warmup 12 → stop 400**. ETA ≈ 49 min; checkpoints {100,200,300,400}. Next
gate = the same AR-guided matched-20 anchor on arm-T's checkpoint-400 vs the stock anchor
(~50/63 careful stock), then **D2 base-choice** (max SWE resolve × anchor-held).

---

## STATUS (2026-07-09, night) — ARM-2 COMPLETE + ANCHOR GATE **PASS** (zero erosion); **D1 recorded** — both arms conversion-ready

**Arm-2 (M_swe_T, stock base) finished clean and PASSED the KILL-T1 anchor gate with the same
turn-for-turn signature as arm-1 (McNemar b=c=0).** Both SWE-SFT arms now hold the certified
tool-call anchor; the campaign advances to K-carrying re-conversion (`k_raise_campaign_design.md`).

**Arm-2 final training summary** (`runs/swe_sft_arm2/Aswe_T_step400_seed71101`, `metrics.jsonl`,
80 pts): **DONE step=400, wall 2966.4 s (7.42 s/step)**, no crash, **0 NaN/Inf**; loss first
0.263 → last 0.150 (min 0.049). The only grad_norm>10 event is step 365 (grad 450.6, **n_valid=1**
single-target-token tail row at LR 1.99e-7 ⇒ negligible), identical failure-mode to arm-1's step
365. Per-step peak 26.67 GiB. **Final adapter:
`runs/swe_sft_arm2/Aswe_T_step400_seed71101/checkpoint-400/adapter_model.safetensors`** (r16/α32,
same 11 targets incl. MLP gate_up/down; `lora_merge_count=184` on export, no dropped targets).
Training dynamics are **near-identical to arm-1** (0.270→0.151/min 0.051), as designed (only the
base differs).

**Anchor gate (§2.5 / KILL-T1), AR-mode, matched-20 exact_args — PASS.** Same protocol as arm-1;
**comparability-first the stock base was re-measured through the identical pipeline** (no
same-pipeline stock reference existed — arm-1 measured the *rlv2* base). Both arm-2 legs used the
identical export→vLLM→`eval_flare_northstar_matched.py --backend ar-vllm-guided` path (greedy,
vLLM 0.23, gmu 0.66 / max-len 4096 / enforce-eager / mamba-align-1024 / gdn-triton), same
20-episode/63-turn native set, gold_sha256 mismatch **0/63**. **Pre-SFT reference = the stock
`fastdllm-init` base exported vllm-bf16 with NO adapter** (`scripts/export_fastdllm_vllm_noadapter.py`;
byte-identical export path minus the LoRA merge), so pre vs post differ **only by the SWE adapter**
— exactly mirroring arm-1 (pre = init+RLv2-adapter, post = init+SWE-adapter):

| metric | pre-SFT stock base (init, no adapter) | post-SFT (M_swe_T) | verdict |
|---|---:|---:|:--|
| tool-call matched-20 **exact_args** | **51/63** | **51/63** | raw 51 ≥ 48 (anchor−3) **PASS** |
| valid_tool_call | 63/63 | **63/63** | no format erosion **PASS** |
| episode_exact | 14/20 | 14/20 | equal |
| **paired McNemar** (`b` pre-right/post-wrong, `c` post-right/pre-wrong) | — | **b=0, c=0, net-loss 0, p=1.0** | net-loss not sig **PASS** |

**Not a no-op:** the SFT model is active — **7/63 turns diverge** in greedy generated text — yet
**zero correctness flips** (same "active but zero-erosion" signature as arm-1's 5/63). The
measured same-pipeline stock anchor **51/63** matches the design's "~50/63 careful stock" (§2.4).
Value-projection audit N/A (AR mode). Artifacts:
`runs/swe_sft_arm2/anchor_gate/{stock_base_matched20,mswe_T_matched20}/ar-vllm-guided/turns.jsonl`,
`anchor_mcnemar_result.json`, `export_{mswe_T,stockinit}.log`; served exports
`models/qwen3.5-9b-fastdllm-{mswe-T,stockinit}-vllm-bf16`; scorer `scripts/swe_sft_arm1_anchor_mcnemar.py`;
gate driver `scripts/swe_sft_arm2_gate_driver.sh`. Secondary GSM8K N=20 deferred (not blocking, as
for arm-1).

### D1 DECISION — recorded (the pivotal go/no-go for the K-carrying re-conversion spend)

**Decision inputs (both arms):**

| input | arm-1 **M_swe_S** (init+RL-v2 + SWE-SFT) | arm-2 **M_swe_T** (stock init + SWE-SFT) |
|---|---|---|
| final / min loss (400 steps) | 0.151 / **0.051** | 0.150 / **0.049** |
| wall (s/step) | 2963.1 s (7.41) | 2966.4 s (7.42) |
| NaN/Inf; grad spikes | 0; one n_valid=1 tail row | 0; one n_valid=1 tail row |
| pre-SFT anchor (same-pipeline) | 49/63 (rlv2 base) | 51/63 (stock base) |
| post-SFT M_swe anchor | **49/63** | **51/63** |
| McNemar b / c / p | 0 / 0 / 1.0 | 0 / 0 / 1.0 |
| KILL-T1 verdict | **PASS** (zero erosion) | **PASS** (zero erosion) |
| SFT active (text-divergent turns) | 5/63 | 7/63 |

**Decision:** **KILL-T1 fires for NEITHER arm.** Both hold the certified tool-call anchor turn-for-turn
(b=c=0). Per design §2.4 default + the campaign directive, **arm-1 / M_swe_S = PRIMARY** advances to
K-carrying re-conversion (it *preserves the certified tool-call gain* — the whole #29 apparatus —
and is single-stage ⇒ fastest to the re-conversion decision); **arm-2 / M_swe_T = CONTROL TWIN**
(stock is the strictly-better SWE substrate but discards the tool-call RL gain ⇒ two-stage, needs a
later tool-call RL redo). **Both are conversion-ready** and both re-convert (the twin@K1 entry gate
≥12/46 on Tier1-C46 is evaluated per-arm at conversion). **Caveat:** arm-2's raw anchor is +2 (51 vs
49) but is measured against a *different* (stock) pre-SFT base that never carried the certified
tool-call gain — it is **not** a capability delta over arm-1. The true D2 tiebreak (design §2.4:
**max SWE resolve@1 AR subject to anchor-held**) requires the **N=5 AR SWE resolve** measurement
(design step 2b) which is **NOT yet run** — that is the one remaining D1/D2 input; if **T ≫ S on SWE
at an acceptable anchor**, flip primary to T and schedule the tool-call RL redo. Readiness inputs for
the K-carrying conversion are captured in **`CONVERSION_READY.md`** (do-not-execute; monitor-dispatched).

---

## STATUS (2026-07-13) — ITERATION-2 DATA SHAPE AMENDMENT: **EPISODE WINDOWING replaces front-truncation** (§2.3 amended C; built + audited, retrain not launched)

The C46 paired AR-arm read (**#117**) flagged **trajectory shape** as a candidate deficit:
iteration-1 trained on **front-truncated** episodes. The single training window per episode
was the *last* `12288` tokens (`build_swe_sft_lmflow_pretok.py`, `truncation_side=left`),
keeping the final edit-and-verify turns but **dropping every early/mid turn** — measured
**assistant-label retention 69.88 %** (911,531 / 1,304,354 target tokens). The model **never
trained on mid-episode context management** (of the early-third label tokens it saw **453**,
i.e. **0.2 %**). Iteration-2 fixes the shape.

**AMENDMENT C — §2.3 data materialisation: single front-truncated window → a set of
SERVE-EXACT SLIDING WINDOWS that TILE each episode.** Builder
`runs/swe_datagen_s1/build_windowed_dataset.py` (CPU-only, deterministic seed 71101),
ALONGSIDE (not replacing) `build_swe_sft_dataset.py`. It emits the **same tokenized schema**
(`conversation_id / input_ids / assistant_spans / n_tokens / n_label_tokens`, + window
metadata), so `build_swe_sft_lmflow_pretok.py` → `swe_sft_arm1_qlora_train.py` consume it
**unchanged** (the downstream left-truncation becomes a **no-op**, 0 rows truncated, since
every window ≤ block). Windowing scheme, spelled out:

1. **Render once, serve-exact.** Each episode is rendered ONCE with the 9B **serving**
   `chat_template.jinja` via the reused `keeper_to_instance` / `conv_for_template` (byte-identical
   to `train_swe_sft.tokenized.jsonl`; verified 5/5 re-render == stored ids+spans). A window is
   ALWAYS a **contiguous slice** `full_ids[w_start:w_end]` with `w_start,w_end` on
   **`<|im_start|>` turn boundaries** — so no message / `<tool_call>` / `<tool_response>`
   envelope is ever split (consecutive `tool` messages share one `<|im_start|>user` wrapper =
   ONE turn-block), and the slice is byte-identical to what the served model sees at that read
   position under the campaign's `truncation_side=left`. We do **NOT** re-render sub-conversations
   (the template's `last_query_index`/`<think>` gating and `raise_exception('No user query
   found')` make standalone re-rendering of a mid-episode slice non-serve-exact — proven, so the
   slice-of-the-full-render construction is the correct serve-exact primitive).

2. **PREFIX-anchored forward tiling with bounded read-back.** Window 1 = the **prefix**
   `[0 : ≤block]` — carries **system + task + early turns** (the model trains on the *fresh task
   → first actions*, mirroring serve where early turns see the full prefix). Each later window
   starts `ctx_overlap=3072` tokens before the previous window's end (snapped to a turn boundary):
   a bounded slab of **loss-masked read-back context**, then it owns the next run of assistant
   turns that fit its ≤block budget. Stride ≈ `block − ctx_overlap` ≈ 9216 target tokens ⇒
   window count is **bounded** (`ceil(episode / stride)`), not the O(#turns) blow-up of maximal
   read-back. Later windows reproduce serve-time left-truncated read windows, so **mid-episode
   context management is now a training target**.

3. **Block-fit — by construction, not after the fact.** Every window ≤ `block=12288`
   (`block % 32 == 0` for the pretok pad path). Guaranteed because the right edge is the largest
   turn boundary ≤ `w_start+block`. **Measured: 0 / 889 windows exceed block; max window 12286.**

4. **Label policy = iteration-1 (assistant spans only).** A window's `assistant_spans` are ONLY
   the assistant turns it **owns** (remapped by `−w_start`); read-back context turns (incl.
   assistant turns owned by an earlier window) are **loss-masked**. Targets are **DISJOINT**
   across windows (each assistant turn owned exactly once) ⇒ every label trained ⇒ retention
   ~100 %.

5. **Coverage target (early/mid/late), quantified.** Windows tile the whole episode, so retained
   labels are spread across the trajectory instead of concentrated LATE. **Measured on the 334
   keepers:** windows by position **early 166 · mid 273 · late 424 · full (single-window) 26**;
   retained label tokens per episode-third recover from iteration-1's **{early 453, mid 263k,
   late 648k}** to iteration-2's **full {early 189k, mid 467k, late 648k}** (100 % of each third).
   System+task present in the earliest window for **312 / 334** episodes (93.4 %) vs iteration-1
   which dropped it on all 328 truncated episodes.

6. **Dedup / anti-domination rule.** A long episode yields more windows (rows) than a short one.
   **Primary rule (works with the trainer's UNCHANGED uniform row sampler): CAP windows/episode
   at `--max-windows=6`**, and if an episode needs more, keep the FIRST (early) + LAST (late)
   window and evenly-spaced interior windows (**stratified**, so early/mid/late survive the cap).
   At the current pool the cap is a **no-op safety ceiling** (measured max = **4** windows/episode,
   mean **2.66**; 0 episodes capped ⇒ retention stays 100 %); it protects the FINAL rebuild if
   longer episodes appear. **Secondary/optional:** every row carries `sample_weight = 1/n_windows`
   so a future *weighted* trainer can exactly equalise per-episode without a rebuild (not required;
   effective row inflation is bounded ≤ cap× and proportionate to trajectory length). *Note for the
   retrain-task owner:* rows grow **334 → 889**, so the fixed-horizon step budget sees each window
   fewer times — revisit HORIZON/epochs for iteration-2 (retrain-freely).

**VALIDATION (334 keepers, all gates green).** retention **100.00 %** (1,304,354 / 1,304,354;
**+30.12 pp** vs 69.88 %, beats the bar) · **889 windows** (hist {1:26, 2:70, 3:229, 4:9}) ·
block-fit **0 over 12288** · serve-exact spot-audit **10/10 PASS** (exact-slice + turn-boundary +
decoded-byte-identical + spans-wrap-assistant + within-block, across 1–4-window episodes) ·
leakage **no-op** (holdout 113 sha==pin, keeper∩holdout **0**, quarantine-in-train **0**, windows
from keeper episodes only, external text NONE) · downstream `build_swe_sft_lmflow_pretok.py`
consumes it **unchanged** (889 instances, **0 rows left-truncated**, retention 1.0, max_seq 12286).
Artifacts: builder `runs/swe_datagen_s1/build_windowed_dataset.py`; dataset
`data/swe_sft_pool/train_swe_sft_windowed.tokenized.jsonl`; audit
`data/swe_sft_pool/windowed_dataset_audit.json`; report
`data/swe_sft_pool/windowed_dataset_report.md`.

**PRE-REGISTERED (final rebuild).** The FINAL post-promotion rebuild is a **MECHANICAL re-run of
this exact script** (same seed 71101, block 12288, ctx_overlap 3072, cap 6) on the updated
`keepers.jsonl` after tranche-2 promotion (**#125**) — **no design changes**. Iteration-2 retrain
(**#127**) points `build_swe_sft_lmflow_pretok.py --tokenized` at the windowed file. **No training
launched by this step.**

---

## 0. FRAME — what the ladder established and the decision this campaign resolves

> **SUPERSEDED (greedy-era rationale — kept for the record; see STATUS block above).** The ladder and the
> two "load-bearing findings" in this section were measured under greedy `temp=0` and are **retired as
> sampling artifacts** by `runs/stage_c_n5v3/report.md`. Read as history, not as live motivation.

The first real SWE-bench-Verified resolve table (`runs/stage_c_n5v2/report.md`, official docker, aligned
episode-in-container runtime, n=5) is a **4-arm ladder**:

> **stock-AR 4/5  >  merged-AR (RL-v2 as AR) 2/5  >  diffusion (same RL-v2 weights) 1/5  >  diffusion (B@1000 stock-conversion) 0/5**

Two load-bearing findings, both mechanistically explained (not just n=5 noise):

1. **RL-v2 is the WRONG PAYLOAD for SWE.** RL-v2 was diffu-GRPO-trained on *short structured tool-call turns*
   (`data/rl_multiturn_v2_public_pool`, audited `exact_args` reward, 300 steps) — not SWE-style long-horizon repo edits.
   As an **AR** payload it *loses* -2 SWE resolves vs stock (4/5→2/5). A loop-halt appears even in merged-**AR** ⇒ the
   looping is substantially **weights-driven** (a tool-call-specialization tax), not a pure diffusion artifact.
2. **The diffusion paradigm compounds it (-1 more), and the twin tracks the twin's general agentic capability.** The
   0/5 B@1000 arm is the pre-RL foundation (34/63-class tool calls) — a weaker agent, so a weaker twin.

**Consequence (adjudicated):** the AR-vs-diffusion horse race at N=25–50 is **premature** — it would measure this known
gap at higher significance. The path to a competitive diffusion twin on SWE is the methodology's own certified loop:
**train the AR base toward SWE-style episodes, then re-convert** — the `convert→RL→re-convert` machinery certified by #29
(`convert_after_rl_result.md`: McNemar `b=0` both seeds, re-diffusionization lost **zero** tool-call turns across 126
paired turns × 2 seeds). This campaign builds the missing piece: **SWE-capable base weights that still carry the certified
tool-call exactness**, so the re-converted twin has something worth serving.

**The one decision this campaign resolves (measured, not assumed):**
> Can we inject SWE capability into an AR base *while preserving the certified tool-call exactness*, and does that base,
> re-converted, move the diffusion arm off the floor? — decided by measured SWE resolve@1 × the tool-call matched-20 anchor.

**What this campaign is NOT:** it is **not** on-policy SWE-RL, and it does **not** use the diffusion twin as a rollout
engine. The twin is measured-**not** a rollout-throughput multiplier (0.73–0.94× guided-AR, ratio worsens with batch,
`runs/p2_batched_rollout_bench`; best-of-N GRPO signal refuted, `runs/p2_bestofn_grpo`). So **rollouts/trajectories are
generated by stock guided-AR** exactly as the methodology's step-2 prescribes. On-policy SWE-GRPO with a terminal
resolve reward is a **future cycle**, gated on this campaign first standing up a SWE-capable base (see §4 note).

---

## 1. DATA — self-generated, verifiable-reward-filtered SWE trajectories

### 1.1 Generator (the 4/5-class agent, aligned runtime, verifiable reward)

- **Agent = stock Qwen3.5-9B served AR** (`runs/stage_c_driver/runcage_ar.sh`, stock vLLM 0.23, cudagraph, native
  `qwen3_xml` tools, temp-0 greedy), driven by **Qwen Code @0.19.2** headless through
  `scripts/run_swe_bench_qwen_code.py` (the flywheel Codex-orchestrator ported to Qwen Code; per the 2026-07-05 user
  directive, LumoFlyWheel is the reference). This is the exact 4/5-class agent from the ladder — the best available SWE
  generator we have; the diffusion twin does not generate (it is not a rollout multiplier).
- **Aligned runtime (mandatory):** every generation episode runs **inside the official per-instance swebench docker
  image** (`--runtime container`, image `swebench/sweb.eval.x86_64.<inst>`), so the agent can import the package and run
  the instance test command in-episode (acceptance gate 5/5, task #64). A trajectory generated in a broken bare checkout
  is a confound; do not generate outside the container.
- **Verifiable reward = official docker score.** Each trajectory's patch (`git diff --binary base_commit → patch.diff`)
  is scored by the **official `swebench.harness.run_evaluation` docker harness** (the flywheel verifiable-reward
  convention; the same harness that produced the `stage_c_n5v2` verdicts). **Keep only `resolved=true`** trajectories
  (patch applied ∧ all FAIL_TO_PASS + PASS_TO_PASS green). This is rejection sampling on a *ground-truth* reward, not a
  proxy — the class of phantom-win contamination that KILL-3 exists for cannot occur here.
- **Format = native `qwen3_xml`** end-to-end (chat template `qwen3-openai-codex.jinja`), so the SFT distribution == the
  generation distribution == the eval distribution ([[native-function-format-rule]]). The trajectory we train on is the
  qwen-code conversation as emitted; assistant turns (reasoning + `qwen3_xml` tool calls + terminating free text) are the
  SFT targets, user/tool-result turns are context-only (loss-masked).

### 1.2 Leakage firewall (spell it out — this is the correctness spine of the whole campaign)

> **⚠ BELT-LEVER ENACTED 2026-07-07 (USER greenlit) — the firewall was RELAXED to the
> "Verified-train-adjacent" fallback documented later in this section.** The
> belt-and-suspenders `verified_500_tier2` ring (500) is **DROPPED from enforcement**;
> the ENFORCED holdout is now **`inner5 ∪ tier0_20 ∪ tier1_100` = 113 DISTINCT ids**
> (the rings nest/overlap, so the nominal 125 dedupes to 113), and the **387
> Verified-adjacent** ids (500 Verified − the 113 held) were added to the data-gen
> frontier's exploit head. KILL-D1 now **hash-asserts** the trainable pool is disjoint
> from those 113 (sha256 pinned, `runs/swe_datagen_s1/.eval_holdout_sha256`). What this
> buys/costs and the exact enactment mechanics are in
> `runs/swe_datagen_s1/USER_LEVER_BELT.md` (§ ENACTMENT 2026-07-07). **Standard-practice
> invariant still held: NO evaluated instance ever trains.** The three rings below are
> now the ENFORCED holdout; the `verified_500` row is retained for provenance but is no
> longer enforced.
>
> **Dual-source scoring caveat (fixed 2026-07-07).** The first genuinely mixed batch
> (batch_0007: 43 Verified + 6 SWE-Gym) generated 49 real patches but recorded 50/50
> `no_prediction` — `datagen_score.sh` fed the merged predictions file to each
> single-source harness and swebench's `get_dataset_from_preds` aborts the whole run
> when any prediction id is absent from its dataset (checked before `--instance_ids`).
> Fixed with per-source filtered prediction files; batch_0007 + orphaned batch_0008
> re-marked `infra_invalid` (excluded from yield/kill; re-drawable). Any future
> dual-source change must pass a both-sources LIVE gate. See `USER_LEVER_BELT.md`.

The eval sets are **frozen and held out by `instance_id`**. A single leaked instance would invalidate every downstream
resolve number, so the firewall is explicit, over-inclusive, and asserted in code (mirroring the RL-v2 pool's
`selected_overlap_counts==0` leak-check).

**HELD-OUT (never in training), three nested rings by `instance_id`:**

| ring | set | n | why held out |
|---|---|---:|---|
| inner | the 5 Tier0 used in `stage_c_n5v2` (`django-11119/12754/13741`, `pytest-8399`, `sympy-13757`) | 5 | the current N=5 eval |
| Tier0 | full Tier0-20 (`runs/stage_c_driver/data/swe-bench-tier0-verified-instances-20260520.json`) | 20 | N=5 ⊂ Tier0; the N=25 eval option |
| Tier1 | full Tier1-100 (`…/auto_research/swe-bench-tier1-verified-instances-20260520.json`) | 100 | the N=25–50 pool is a stratified slice **of Tier1** |
| **belt-and-suspenders** | **the ENTIRE SWE-bench_Verified 500** (= Tier2, `…tier2-verified-instances-20260520.json`) | **500** | exclude *every* Verified `instance_id` so no eval tier can ever leak, at any future N |

**Primary training source = SWE-Gym** (`SWE-Gym/SWE-Gym`, 2,438 executable Python task instances across 11 repos;
`SWE-Gym-Lite` = 230). SWE-Gym was **constructed disjoint from the SWE-bench test/Verified instance set** — the cleanest
possible firewall — and gives a large, difficulty-comparable pool. Repo names overlap with Verified (django, sympy,
sklearn, …) but **instance_ids are disjoint**; additionally screen for **near-duplicate PRs** (same repo, `base_commit`
within a small commit-distance of any held-out instance) and drop them. Infra cost line item: SWE-Gym instances do **not**
all have prebuilt `swebench/sweb.eval.x86_64.*` images, so environments must be **built** via the swebench harness env
builder (real, one-time cost — budget it, §5).

**Fallback / supplement = "Verified-train-adjacent" = the ~380 Verified instances NOT in Tier0∪Tier1** (500 − 120). These
already have official prebuilt images (the aligned-runtime works as-is, zero env-build cost) but the pool is smaller and
foreclosing eval-tier expansion. Use only if SWE-Gym env-building proves too costly, and **only after** the belt ring is
relaxed to `Tier0∪Tier1` (never touching held-out ids). Recommend SWE-Gym primary; document the fallback.

**Firewall artifact (required):** `data/swe_sft_pool/pool_manifest.json` listing every training `instance_id` + source +
env-build status, with a hard assertion at build time and at train-launch.
**Pre-2026-07-07:** `train_ids ∩ verified_500 == ∅` **and** `train_ids ∩ (Tier0 ∪ Tier1) == ∅`.
**Post-belt-lever (2026-07-07, ENFORCED):** `train_ids ∩ (inner5 ∪ tier0_20 ∪ tier1_100) == ∅`, **hash-asserted**
against the pinned sha256 of the 113-id eval holdout (`expand_frontier.py`, `runs/swe_datagen_s1/.eval_holdout_sha256`);
`verified_500` is retired to `manifest.relaxed_rings` and `intersect_verified_500` is now INTENTIONALLY nonzero (the 387
adjacent ids). Any overlap with the enforced 113 ⇒ **KILL-D1** (do not train). This is the `rl_v2_leak_check` convention
extended to SWE.

### 1.3 Data scale target + yield + generation cost (honest)

SWE trajectories are **long and information-dense** (the `stage_c_n5v2` episodes ran 200K–990K cumulative tokens over
16–52 turns, effective final context capped near `max_model_len=32768`). Hundreds of *verified-correct* long trajectories
already shift a 9B's behavior meaningfully (Run-1 shifted on a 5,055-sample mix of far shorter examples).

| quantity | value | basis |
|---|---|---|
| **target verified-correct trajectories** | **600–1,000** (min viable ~400; stretch ~1,500) | RFT set for a LoRA on 9B; SWE trajectories long ⇒ hundreds move behavior |
| **generator yield (resolve@1)** | **35–55%** on SWE-Gym-class (honest; Verified-adjacent easier ~40–70%, SWE-Gym broader/harder ~25–45%) | stock-AR was 4/5 on the *easy validated* Tier0; SWE-Gym is broader ⇒ discount |
| **attempts needed** | **~1,300–2,900** for 600–1,000 keepers @ ~40% (use best-of-k per instance, keep any resolve) | N/yield |
| **serving GPU per attempt** | **~1.5–3 min** (stock-AR wall was 56–188 s/episode; turn-limited ~ higher) | `stage_c_n5v2` stock-AR wall column |
| **generation GPU-h (5090, serving)** | **~30–60 GPU-h** — run data-gen at **concurrency 4–8** (stock AR co-batches near-100% util, unlike the diffusion engine ⇒ GPU-h *drops* with concurrency) | attempts × min ÷ batch |
| **docker eval (correct-filter)** | **~40–90 wall-h, OFF the serving GPU** (alienware x86 or local), ~1–5 min/attempt, fully parallelizable | env build + test run per attempt |

**Leverage note:** unlike the paired *eval* runs (one server, one episode, in the RAM cage), **data generation is
throughput-work** — run the stock-AR server at `max_num_seqs` 4–8 with concurrent qwen-code episodes. Stock AR is the
regime where batching *helps* (the measured 100% util in the rollout bench), so wall-clock compresses ~4–8× at roughly
constant-or-lower GPU-h. This is the [[gpu-utilization-standard]] applied: do not generate at batch-1.

**Yield-rescue levers (documented fallbacks, not primary):** if the self-gen yield or trajectory *quality* is inadequate
(e.g. <25% and thin coverage), (a) **best-of-k** attempts per instance with temp>0 sampling to harvest more resolves;
(b) oversample easier repos/instances; (c) **teacher-distill** from the GB10 Qwen3.5-27B flywheel model or a frontier
model as a documented purity caveat (changes "self-generated" → "distilled"; report it, do not silently mix). Primary
remains self-generated per the task; levers are pulled only on a measured pilot shortfall (§5 decision point D1).

### 1.4b ENVELOPE-CORRECTED PROBE (v2) — SUPERSEDES 1.4; yield 0.25 clears the GO bar (2026-07-06)

`runs/stage0_swegym_probe_v2/report.json` re-ran the SAME 20 SWE-Gym instances under the **reference
envelope** (temp 0.6 / top_p 0.95 / top_k 20 / seed 1234, proxy-forced) + **empty-patch re-drive
(retries=1)**. Verified against the official scorer JSON (`score/probe-stockAR-env.probe20env.json`,
schema_version 2) and the primary `all_predictions.jsonl`:

| metric | greedy (v1, §1.4) | **envelope (v2)** |
|---|---|---|
| **resolve@1 yield** | 0.15 (3/20) | **0.25 (5/20)** — Wilson95 [0.112, 0.469] |
| patch_produced | 0.75 (15/20) | **0.95 (19/20)** — empty-patch loss fixed by the re-drive |
| GPU-min/attempt (conc. 4) | 0.66 | 0.77 (97.8% util, 100% median) |
| docker-min/eval | 0.38 | 0.61 |

resolved (v2): `bokeh-12841`, `conan-10213`, `conan-10408`, `dvc-10218`, `pydantic-4911`.
**D1 verdict flips: 0.25 ≥ 0.20 GO bar ⇒ GO_single_attempt.** Repriced full campaign @0.25: 2,400–4,000
attempts, 30.7–51.2 serving GPU-h, 24.4–40.7 docker-eval wall-h (vs the greedy 44–73 GPU-h). The
generator economics are no longer the blocker — the **dissolved premise** is (STATUS block). Everything
in §1.4 below is the superseded greedy record.

### 1.4 PHASE-2 PROBE — MEASURED (2026-07-06; 20 SWE-Gym instances, stock-AR @concurrency 4) [SUPERSEDED by 1.4b]

`runs/stage0_swegym_probe/` (README + `report.json`). Stratified 20 (2× each of the 10 repos the SWE-Gym harness-fork
spec map covers; **MONAI excluded** — its repo is absent from `SWE-Bench-Fork`'s `MAP_REPO_VERSION_TO_SPECS`). All 20 are
`source=SWE-Gym` rows in `pool_manifest.json` (KILL-D1 clean: disjoint from verified_500 ∪ Tier0 ∪ Tier1 by
construction). Toolchain, all three legs measured:
- **Env acquisition = docker PULL of the official prebuilt `xingyaoww/sweb.eval.x86_64.<id: __→_s_>` images (NOT a
  from-scratch env build), re-tagged** to the driver key (`swebench/…_1776_…`) + the fork scorer key. **This retires the
  design's dominant priced risk:** SWE-Gym images are *pullable*, not buildable-only — **0.6 min/instance, 39 GB/20, 0
  failures**, vs the assumed one-time BUILD cost.
- **Generation = one stock-AR vLLM server at `max_num_seqs=4`** + 4 concurrent qwen-code shards (episode-in-official-
  container, native `qwen3_xml`, temp-0). **GPU util mean 97.5% / median 100%** (158 samples, 400 W) — the batching keeps
  the GPU saturated ([[gpu-utilization-standard]] met). Episode wall median 144 s (turns median 45; 5/20 hit the 50-turn
  cap). **13.2 min wall for all 20 ⇒ 0.66 GPU-min/attempt** (design assumed 1.5–3).
- **Official filter = SWE-Gym/`SWE-Bench-Fork`@242429c** (one-line patch `artifacts/fork_reuse_prebuilt.patch` so the
  harness *reuses the pulled instance image* instead of rebuilding env images). Validated on the gold patch
  (`hydra-1006`→resolved). **0.38 docker-min/eval, 0 harness errors.**

| measured | value | vs design assumption |
|---|---|---|
| **generator yield resolve@1** | **3/20 = 0.15** (Wilson95 **[0.05, 0.36]**) | assumed 25–45% (SWE-Gym); **BELOW** |
| patch-produced rate | 15/20 = 0.75 (5 empty patches) | — (empty-patch = fixable gen shortfall) |
| env acquisition / instance | **0.6 min (PULL)** | assumed one-time BUILD; **far cheaper** |
| GPU-min / attempt (conc. 4) | **0.66** | assumed 1.5–3 |
| GPU util during gen | **97.5% mean / 100% median** | util-standard: PASS |
| docker-min / eval | **0.38** | assumed 1–5 |

resolved: `dvc-10218`, `pydantic-4911`, `mypy-10036`. empty-patch: both pandas, `dask-10027`, `dvc-10213`, `modin-5507`.

---

## 2. TRAINING — what gets trained, and onto which base

### 2.1 SFT (rejection-sampling / RFT), NOT on-policy SWE-GRPO — argue the choice

**Train by rejection-sampling SFT (RFT/STaR) on the verified-correct trajectories.** Reasons, evidence-anchored:
- **Reward structure.** SWE resolve@1 is a *single terminal, sparse* 0/1 over a *long, expensive* episode. GRPO needs
  many rollouts per prompt for a usable group advantage; on these episodes that is enormously more expensive than the
  short structured turns RL-v2 used, and the twin gives **no throughput help** (measured). RFT converts the same verified
  signal into a stable supervised target at a fraction of the cost.
- **Fit to existing machinery.** The self-gen + official-docker-filter pipeline (§1) *is* the RFT data step. No new RL
  infrastructure.
- **Erosion is well-characterized for SFT** (the ≤400–600-step law), and the promotion discipline is cleanest on SFT.

On-policy **SWE-GRPO with a terminal resolve reward** is the *right* next cycle **after** this establishes a SWE-capable
base — it is out of scope here and explicitly deferred (a base that can resolve is a prerequisite for a resolve-reward RL
group to have any non-degenerate signal).

### 2.2 The base-choice question — argue it, then RUN BOTH ARMS and let the measurement decide

The crux the ladder forces: RL-v2 carries the **certified +12 tool-call gain** we must not lose, but it is a **-2 SWE
base** vs stock. Two candidate bases:

| arm | base | rationale FOR | risk |
|---|---|---|---|
| **S (primary)** | **merged RL-v2** (`models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16`, init+RL-v2, AR) | preserves the certified tool-call exactness (the whole point of the #29 preservation apparatus); SWE-SFT directly supplies the missing long-horizon capability; the -2 tax is a *narrowing* and broad SWE-SFT is exactly the corrective for narrowing; **single training stage → fastest to a re-conversion decision** | SWE-SFT could erode the narrow tool-call exactness (⇒ protect with the matched-20 anchor gate) or the RL-v2 narrowing could be sticky |
| **T (control)** | **stock Qwen3.5-9B** (the 4/5 SWE base) | starts from the strictly-better SWE substrate; no RL-v2 tax to fight | throws away the certified tool-call gain ⇒ must **re-do tool-call RL in a later cycle**; two-stage path |

**Recommendation: run BOTH as AR-SFT arms** (they are cheap — LoRA SFT + AR eval, §5) and **decide by measurement**, in
exactly the 2×2-discipline spirit that produced the ladder. Do **not** assume S recovers or that T dominates. Pick the
base for the (expensive) re-conversion by the joint criterion:
> **max SWE resolve@1 (AR, N=5 aligned/official) subject to tool-call matched-20 anchor ≥ its arm's pre-SFT anchor − (paired-CI margin).**

Prior belief (state it, then test it): **S is favored** — a good SWE agent *is* a tool-call agent, so tool-call exactness
and SWE capability are complementary, and preserving the certified capability is the methodology's stated goal. But the
ladder's dominant finding ("RL-v2 is the wrong payload") is precisely the hypothesis that T could win; that is why T is
run, not argued away. If **T ≫ S on SWE at an acceptable anchor**, flip to T and schedule the tool-call RL re-do.

### 2.3 LoRA config (per the recipes)

AR-side LoRA SFT (the base is an AR model; SFT precedes conversion). Anchored on the RL-v2 / Run-1 target philosophy,
widened for broad SWE capability (SWE needs code reasoning + multi-file edits, not just tool structure):

- **Targets:** attention `q_proj,k_proj,v_proj,o_proj` + GDN `in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj` +
  **MLP `gate_up_proj,down_proj`** (the widen vs RL-v2's tool-call-only set — SWE capability lives partly in MLP). Keep
  `in_proj_ba` δ/α gates and `conv1d`/norms out (the NVFP4-exclusion rationale: recurrent-state-sensitive, near-free to
  protect).
- **Rank / α:** `r=16–32 / α=32–64`, dropout 0.05 (RL-v2 was r16/α32; widen a notch for the broader target set).
- **LR / schedule:** `1e-5` cosine (Run-1's conversion LR; conservative for retention), `grad_accum` to a healthy
  effective batch given the long sequences (`block_size` up to 32768 with left-truncation, native template).
- **Seeds:** two (per [[retrain-freely-rule]]) so a promote/erode call is never on a single seed.

### 2.4 Step cap (per the erosion law) + retrain-freely

The documented erosion law is **~400–600 steps** on this stack; beyond it capability erodes. **Sweep `{100, 200, 300,
400}` steps** with retention-gated early selection; **do not extend past ~600** to "rescue" a weak SWE number. If a
checkpoint looks over/under-trained, **retrain at a different step count** rather than reasoning around it (S and T
adapters are disposable). Save every 100 steps for the sweep.

### 2.5 KL safety kit + retention probes — INCLUDING the tool-call matched-20 anchor spot-gate

The certified capability must not silently erode. The safety kit, per the RL-v2 recipe + #29 conventions:

- **In-training:** retention probe every 50 steps (GSM8K N=5 quick, the existing `run_retention_probe` cadence);
  early-stop on retention collapse below the flex-accuracy floor. For the SFT setting add a **light KL-to-base
  regularizer** (RL-v2 used `KL_TO_BASE_COEFF=0.05` on value/free tokens, structural tokens masked) or, equivalently,
  rely on low-LR + LoRA + step-cap; report `max_KL_to_base` either way.
- **Post-SFT spot-gate battery (the load-bearing gate — a fast subset of the #28/#29 batteries), on each candidate
  adapter, AR mode:**

  | probe | anchor | PASS | source |
  |---|---|---|---|
  | **tool-call matched-20 exact_args** (the CERTIFIED capability) | S: 47/63 hybrid · 44/63 careful; T: ~50/63 careful stock | **McNemar net-loss vs anchor not significant (p≥0.05)** AND raw ≥ anchor − 3 | #29 a1/a2; `eval_flare_northstar_matched.py` |
  | GSM8K legacy full-context N=20 | 0.65 (S) / stock ~0.75 (T) | ≥ anchor − 1 row (rerun-once band) | REPRODUCE_V2 §4/5 |
  | MBPP / instruction (no-collapse) | #28 battery (88% / 84%) | within Wilson noise, no class collapse | `conversion_tax_result.md` |
  | value-projection audits (diffusion probes only) | 0 | all counters 0 | KILL-3 |

  The matched-20 tool-call anchor is the **standing capability that #29 certified and the diffusion twin serves through the
  grammar path** — if SWE-SFT quietly erodes it, the whole exact-args spine (and every diffusion resolve that rides it) is
  compromised. This gate is per-arm and blocking (see KILL-T1).

### 2.6 Output of Stage-1

Two AR checkpoints — `M_swe_S` (merged-RL-v2 + SWE-SFT) and `M_swe_T` (stock + SWE-SFT) — each with a passing spot-gate
and a measured AR SWE resolve@1 (N=5, aligned/official). The **winner** (§2.2 criterion) proceeds to re-conversion; the
loser is banked as a control row. **Decision point D2 (§5).**

---

## 3. RE-CONVERSION + EVAL — the #29 protocol on the new weights, then the 4-arm re-eval

### 3.1 Fresh two-stream conversion + preservation gates (now preserving TWO fresh capabilities)

Run the certified #29 protocol on the winning base `M_swe*`:
1. **Merge** the SWE-SFT adapter into the Fast-dLLM `init` candidate **in the HF stack** (mask token 248077, `bd_size=32`,
   bridge intact — the vLLM export strips these), producing `models/qwen3.5-9b-fastdllm-mswe-merged`. **Merge sanity gate
   = KILL-1** (bit-exact `init + scale·B@A`, manifest mask/bd_size).
2. **Fresh two-stream conversion** (Run-1 recipe, ~400 steps, r16/α32, targets q/k/v/o + in_proj_{qkv,z,b,a} + out_proj,
   `data/flare_redesign_run1_copy_retention_mix` — the mix that **excludes both** the RL-v2 pool **and** the SWE-SFT pool,
   so the conversion is *not trained on* the capabilities it must preserve — the whole sharp-test premise), **two seeds**.
3. **Export the clean stream** → `models/qwen3.5-9b-fastdllm-mswe-Anew-vllm-bf16` (AR-mode eval) and load the diffusion
   twin via the FLARE hybrid_clean engine.
4. **Preservation battery (a/b/c/d), sampler-pinned + audited**, exactly as `convert_after_rl_design.md §6`, but the
   "fresh capability" now has **two** components that must both survive:
   - **tool-call exactness** (matched-20 hybrid/careful + never-train breadth) — McNemar `b−c ≤ 0` vs the pre-conversion
     `M_swe*` anchor, no significant net-loss (the #29 bar);
   - **SWE capability** — a **diffusion-mode SWE spot-check**: the same 5 (or a small held-out-SWE-Gym) instances through
     the diffusion twin, resolve@1 ≥ the `M_swe*` AR resolve − (paired margin). This is the *new* preservation axis this
     campaign adds; it is the reason re-conversion is gated, not assumed (though #29 gives strong prior that it preserves).
   - GSM8K retention ≥ anchor; value-projection audits all-0 (KILL-3).

### 3.2 The 4-arm N=5 re-eval (the ladder, re-run on the new weights)

Same aligned runtime + official docker scoring as `stage_c_n5v2`, the same 5 Tier0 instances, the new 2×2:

| arm | weights | paradigm | role |
|---|---|---|---|
| stock-AR | stock | AR | the 4/5 control (unchanged) |
| **M_swe-AR** | winning SWE-SFT base | AR | did SWE-SFT lift the AR base? (Stage-1 already answers; re-confirm paired) |
| **M_swe-diffusion** | re-converted twin | diffusion | **the deliverable** — did the ladder move? |
| diffusion-on-stock (B@1000) | stock | diffusion | the 0/5 pivot retained for the {weights}×{paradigm} 2×2 |

**Gate C-G2':** `M_swe-diffusion resolve@1 ≥ M_swe-AR resolve@1 − (paired-CI margin)` (paradigm tax bounded) **AND**
`M_swe-diffusion > diffusion(RL-v2) 1/5` (ladder moved). Losslessness assertion in-loop (C7: diffusion episodes ran under
a byte-lossless cache-on cert). Report paired stats (McNemar on shared instances), not marginal rates only.

### 3.3 Only-then N=25–50 (the significance tier)

Only if the N=5 re-eval clears C-G2' (D4, §5): scale to **Tier0-20 or a 25–50 stratified slice of Tier1** (never the
held-out inner-5 for training; all held-out for eval), **both/all arms paired**, temp-0 greedy, native `qwen3_xml`,
**turn cap raised 50→75** (per `stage_c_n5v2` §4: 6/15 episodes hit the 50-turn `FatalTurnLimitedError` and one *resolved
at the cap* — the cap truncates real work; book the ~1.5× wall/token cost on affected episodes; do **not** go unbounded).
Report resolve@1 + Wilson CIs + paired McNemar + per-turn economics both ways. n=25–50 **ranks and surfaces large
effects**; a small residual paradigm tax will stay inside the CIs (~80–90/arm needed for a 0.2–0.3 gap at power) — state
that honestly, as the report already does.

---

## 4. KILL CRITERIA (stop; do not interpret)

- **KILL-D1 — leakage.** `train_ids ∩ verified_500 ≠ ∅` or `∩ (Tier0∪Tier1) ≠ ∅`. Firewall breach ⇒ every downstream
  number is void. Rebuild the pool.
- **KILL-D2 — yield collapse.** Pilot (first ~50–100 instances) resolve@1 < ~15% AND coverage thin ⇒ the self-gen RFT set
  will be too small/narrow; pull a yield-rescue lever (§1.3) or descope, do not grind a weak generator.
- **KILL-T1 — certified-capability erosion.** Any SWE-SFT candidate fails the **tool-call matched-20 anchor spot-gate**
  (significant McNemar net-loss OR raw < anchor − 3) with no step-count that both lifts SWE *and* holds the anchor ⇒ that
  base is unusable; fall back to the other base arm. If **both** arms fail the anchor, SWE-SFT-on-this-recipe erodes the
  spine ⇒ escalate to the user (retune targets/LR/steps, or reconsider joint convert-and-SFT).
- **KILL-T2 — no SWE lift.** Neither `M_swe_S` nor `M_swe_T` beats the RL-v2 **2/5** AR SWE floor at an acceptable anchor
  (N=5 informational; confirm at the pilot/Stage-1 boundary) ⇒ the RFT set did not inject SWE capability; do not spend the
  re-conversion. Diagnose (data scale/quality) before re-attempting.
- **KILL-1 / KILL-3 (inherited from #29).** Merge sanity not bit-exact / manifest wrong ⇒ base wrong, do not train.
  Any value-projection audit counter nonzero ⇒ the diffusion measurement is contaminated (this class has produced every
  phantom win in this project), run invalid.
- **INCONCLUSIVE handling.** SWE resolve deltas inside n=5 CIs after two seeds ⇒ report a *ranking*, not a verdict, and
  decide N=25–50 spend on the AR-mode + anchor evidence (the tight rows), per the `stage_c_n5v2` binomial-honesty
  discipline. Do not manufacture a pass by extending steps past 600.

---

## 5. BUDGET (GPU-h + wall) + DECISION POINTS

**GPU-hour / wall rollup (RTX 5090 serving; docker eval off-GPU on alienware/local; single-researcher, measure-not-assume).**

| phase | 5090 GPU-h | off-GPU eval wall | eng/wall | note |
|---|---:|---:|---|---|
| **1. Data-gen** (self-gen + docker-filter, 600–1k keepers @ ~40%, concurrency 4–8) | **30–60** | 40–90 h (parallelizable) | ~1 wk | + one-time SWE-Gym env-build cost if SWE-Gym primary |
| **1. Data-gen — MEASURED-REPRICE @ probe yield 0.15 (best-of-1)** | **44–73** | 25–42 h | ~1–1.5 wk | env-build cost **retired** (images PULL @0.6 min/inst); GPU-min/attempt 0.66, docker-min/eval 0.38 (all measured) |
| **2a. SWE-SFT** (2 arms × step-sweep, long seqs, 2 seeds) | **6–12** | — | ~2–3 d | LoRA SFT; ≤600 steps/arm |
| **2b. AR spot-gate + N=5 AR SWE** (both arms) | **3–6** | 5–15 h docker | ~2 d | tool-call anchor + GSM8K + SWE resolve |
| **3a. Re-conversion (#29, winner, 2 seeds)** | **3–5** | — | ~1–1.5 d | ~0.6 train + preservation battery |
| **3b. 4-arm N=5 re-eval** (aligned/official) | **3–5** | 5–15 h docker | ~1–2 d | the ladder re-run |
| **3c. N=25–50** (2–4 arms, turn cap 75, ~1.5×) | **35–60** | 40–80 h docker | ~1–2 wk | only after D4 |
| **TOTAL (through the deliverable)** | **~80–150 GPU-h** | ~90–200 wall-h eval | **~4–6 wk** | dominated by data-gen + N=25–50 serving occupancy |

**Decision points for the user (each is a commit + a short report; the user steers, not dictated):**
- **D1 — after the data-gen pilot (~50–100 instances):** measured yield + coverage. GO full-scale generation / pull a
  rescue lever / descope. (Guards against burning 30–60 GPU-h on a weak generator.)
  > **D1 UPDATE (2026-07-06 ENVELOPE probe v2, §1.4b) → GO_single_attempt.** The envelope re-run measured
  > **yield 0.25 (5/20) ≥ the 0.20 GO bar** and fixed the empty-patch loss (patch_produced 0.95). The
  > greedy ADJUST verdict below is SUPERSEDED. D1 is now a GO on economics — but the campaign is PARKED on
  > the dissolved premise (STATUS block), so this GO is "unpark-if-a-gap-reopens," not "launch now."
  >
  > **D1 RESOLVED [SUPERSEDED — greedy record] (2026-07-06 phase-2 probe, n=20) → ADJUST (fall back per design; NOT a clean GO, NOT KILL).** §1.4.
  > **Yield resolve@1 = 3/20 = 0.15 (Wilson95 [0.05, 0.36])**, below the 20% GO bar → per the task rule the *single-attempt*
  > SWE-Gym self-gen is **mispriced** (repriced ~44–73 GPU-h for 600–1k keepers, ≈2× the estimate; still tractable). It is
  > **not** KILL-D2: coverage is broad (resolves span dvc/pydantic/mypy), harness errors = 0, and the toolchain is proven.
  > Two measured wins de-risk the path regardless: **(i) the env-build risk is retired** — SWE-Gym images *pull* prebuilt
  > (0.6 min/inst, no build); **(ii) generation is cheap + 97.5%-util** (0.66 GPU-min/attempt). The shortfall is *patch
  > correctness*, and 25% of attempts (5/20) produced **no patch at all** (turn/wall-cap or non-committing episodes) — a
  > fixable generation loss, not a hard ceiling. **Recommended levers before full-scale spend (user steers):** (a)
  > **best-of-k** (k=3–5, temp>0, keep-any-resolve — the design's primary lever; lifts effective yield the most); (b)
  > **diagnose/close the 25% empty-patch rate** (raise turn cap 50→75, enforce a final edit); (c) the **Verified-train-
  > adjacent fallback** is now *also* cheap (prebuilt images, easier ~40–70%) and can supplement/replace SWE-Gym after the
  > belt is relaxed to Tier0∪Tier1. Do **not** launch the full 600–1k self-gen at best-of-1 0.15 without pulling (a).
- **D2 — after Stage-1 (both SWE-SFT arms, AR spot-gate + N=5 AR SWE):** **pick the base** (S vs T) by
  `max SWE resolve × anchor-held`; or KILL-T1/T2. This is the pivotal go/no-go — the re-conversion is only spent on a base
  that measurably has SWE *and* holds the certified tool-call anchor.
- **D3 — after re-conversion preservation battery (#29):** confirm the twin preserves **both** capabilities (tool-call
  McNemar `b−c≤0`, SWE diffusion spot-check, audits clean) before any eval spend.
- **D4 — after the 4-arm N=5 re-eval:** confirm C-G2' (twin ≥ its own AR, ladder moved off 1/5). GO N=25–50 / iterate the
  base / stop. (Guards the ~35–60 GPU-h N=25–50 tier.)
- **D5 — after N=25–50:** the deliverable AR-vs-diffusion verdict at ranking-tier significance, with the honest CI caveat.

### 5.1 PHASE-3 PRICED DECISION LEDGER (measured vs thresholds → verdict)

The D1 gate reduced to its governing thresholds, each measured on the n=20 probe (§1.4). Verdict is per-threshold, then the
overall call is the join.

| governing threshold | bar | measured (n=20) | per-threshold |
|---|---|---|---|
| **GO — generator yield resolve@1** | ≥ 0.20 (task GO bar) | **0.15** (Wilson95 [0.05, 0.36]) | **MISS** — CI straddles the bar; best-of-1 self-gen is mispriced |
| **KILL-D2 — yield collapse** | < 0.15 **AND** coverage thin | 0.15 (at bar, not below) **AND** coverage **broad** (resolves span dvc/pydantic/mypy) | **NOT triggered** (both conjuncts must hold) |
| **KILL-D1 — leakage** | any `train_id ∩ (verified_500 ∪ Tier0 ∪ Tier1)` | 0 (all 20 `source=SWE-Gym`, disjoint by construction) | **CLEAN** |
| **util-standard — GPU util in gen** | not LOW | **97.5% mean / 100% median** (158 samples) | **PASS** |
| harness integrity | 0 errors | **0** harness errors, 0.75 patch-produced | **PASS** |

**Recomputed campaign price (measured inputs, best-of-1 @ yield 0.15):** for 600–1,000 keepers → **4,000–6,667 attempts ·
44–73 serving GPU-h · 25–42 docker-eval wall-h**. This is ≈2× the design's 30–60 GPU-h data-gen line, but two measured wins
lower total risk: **env-build cost is retired** (SWE-Gym images *pull* prebuilt at 0.6 min/inst, not build) and **generation
is cheap + GPU-saturated** (0.66 GPU-min/attempt at 97.5% util). The shortfall is patch *correctness*, and 25% of attempts
(5/20) produced **no patch at all** — a fixable generation loss, not a capability ceiling.

**Overall verdict: ADJUST** (not a clean GO — yield below the 0.20 bar; not KILL — KILL-D2 not triggered, KILL-D1 clean,
toolchain proven). **Do not launch the full 600–1k self-gen at best-of-1 0.15.** Pull the design's primary lever first —
**best-of-k** (k=3–5, temp>0, keep-any-resolve) — and/or close the 25% empty-patch rate (turn cap 50→75 + forced final
edit); the now-also-cheap **Verified-train-adjacent** pool (prebuilt images, easier ~40–70%) can supplement after the belt
ring is relaxed to Tier0∪Tier1. User steers the lever choice at D1.

**Composability with the rest of the plan:** this campaign is the "SWE-style training data" investment the `stage_c_n5v2`
GO/NO-GO called for; it feeds the winning `M_swe*` into Stage-C's existing serve path (Stage A certified, NVFP4 optional
per Stage B). If it succeeds, the next cycle is on-policy **SWE-GRPO** with a terminal resolve reward on the now-SWE-capable
base (the deferred RL step, §2.1) — closing the methodology's flywheel on the SWE distribution.

---

## THE N=25–50 PROPOSAL — the properly-powered SWE horse race (2026-07-06; the justified next step)

**Motivation (why now, not before).** The v2 report deferred N=25–50 as "premature — it would measure a
known diffusion-vs-AR gap at higher significance." The envelope-corrected run
(`runs/stage_c_n5v3/report.md`) **erased that gap**: stock-AR 3/5 == merged-AR 3/5 == diffusion 3/5,
identical resolve sets, paired McNemar b=c=0. The honest prior is now **the tie**. A powered run is the
only instrument that can either (a) detect a real paradigm difference the n=5 tie is too coarse to see,
or (b) bound it tightly enough to certify the diffusion twin as an AR-equivalent SWE server. Either
outcome is decision-grade. This proposal is the horse race, priced from the v3 measured per-episode
costs.

### Arms

| arm | weights | paradigm | role |
|---|---|---|---|
| **stock-AR** | stock Qwen3.5-9B | AR (vLLM) | the shipping AR reference |
| **diffusion** | merged RL-v2 twin | block-diffusion (FLARE hybrid_clean) | **the deliverable under test** |
| merged-AR *(optional 3rd)* | merged RL-v2 | AR | run ONLY to decompose weights-vs-paradigm **if** the primary pair diverges; skip if budget-tight |

diffstock is **dropped** — its v3 1/5 is a known general-agentic-capability floor (pre-RL B@1000
foundation, loops out of 4/5 episodes), not a diffusion-vs-AR paradigm question. The 2×2 it completed is
no longer the live question now that the RL-v2 twin ties the AR arms.

### Protocol (frozen to the v3 contract)

- **Sampling:** reference envelope **temp 0.6 / top_p 0.95 / top_k 20**, forced proxy-side via
  `LUMO_PROXY_FORCE_*`, **seeded per-request** (reproducible); empty-patch re-drive retries=1.
- **Runtime:** episode-in-official-container (`--runtime container`, `swebench/sweb.eval.x86_64.<inst>`),
  native `qwen3_xml` tools, one server per arm.
- **Execution = BATCHED, concurrency 4+ (USER FROZEN CONFIG, `runs/loop_halt_polish/USER_DIRECTIVE_BATCHED_NRUN.md`, 2026-07-06).**
  NOT serial. Each arm runs its N episodes **concurrently against its server via continuous batching**:
  baseline **concurrency 4**, probe **6/8 if HBM headroom holds** (engine correctness certified to **bs=8 @
  gpu_mem 0.82**; **b16 needs gmu ≤ 0.62** — pick the safest high setting, measure, don't assume). This
  overrides the v3 `--max-num-seqs 1` cage; no week-long serial run.
- **Scoring:** OFFICIAL `swebench.harness.run_evaluation` docker harness; official `scoring/*.json`
  verdicts, no mock.
- **Design:** **paired** (every arm runs the same instances), **resolve@1** = one seeded attempt per
  (arm, instance). **resolve@1 + paired McNemar is the PRIMARY output and is UNAFFECTED by batching** —
  per-request seeds stay deterministic per episode. Report resolve@1 + Wilson CIs + paired McNemar on the
  shared instances. Losslessness assertion in-loop for the diffusion arm (cache-on byte cert).

### Turn cap = 75 (raised from the v3 cap of 50) — justified from the v3 turn distributions

Primary evidence (`runs/stage_c_n5v3/report.json`, `run_v3_arm.sh` `MAX_TURNS=50`):

- **All clean AR resolves finished by turn 47** (stock-AR resolves at 38/47/38; merged-AR at 35/43/37).
  A cap of 75 contains every observed clean AR resolve with ~1.6× margin.
- **The diffusion resolves land at turns 49–50 — pressed against the 50 cap** (django-13741 at 50,
  pytest-8399 at 49, django-11119 at the 50 turn-limit). The diffusion paradigm demonstrably needs *more*
  turns to resolve than the AR arms, and at cap 50 its resolves are at the ceiling. **75 gives the
  diffusion arm ~50% headroom above where its resolves currently occur** — the arm most likely to be
  truncated by a tight cap.
- **Raising the cap is not unbounded-spend.** The v3 run had **7 turn-limit exits** (proxy-req counts
  80–103 against the 50-session-turn cap) and **only one** of them resolved (diffstock django-11119 at
  102 reqs — and diffstock is dropped). Episodes that spin past ~50 turns almost never convert; 75 is the
  economical ceiling that fits all real work while still bounding the dead spins. Corroborates the v2
  finding (6/15 greedy episodes hit the 50 cap, one *resolved at* it).

Booked cost of 75 vs 50: ~1.2× blended wall on the ~40% of episodes that would otherwise turn-limit
(already included in the pricing headroom below).

### Instance pool = stratified N=25–50 slice of Tier1-100 (leakage-firewalled)

- Draw from **Tier1-100** (`…/auto_research/swe-bench-tier1-verified-instances-20260520.json`), stratified
  across repos. **N=25 minimum** (ranks arms, surfaces ≳0.3 effects), **N=50 preferred** (tighter CIs).
- **Leakage rules (the §1.2 firewall, enforced):** the eval pool is **held out from ALL training** —
  `eval_ids ∩ train_ids = ∅` and, if this campaign ever unparks, the SWE-SFT `train_ids` must satisfy
  `train_ids ∩ (Tier0 ∪ Tier1 ∪ verified_500) = ∅`. Never include the inner-5
  (`stage_c_n5v3`) so the v3 baseline stays an independent point. Screen for near-duplicate PRs.
- Images: Tier1 Verified instances have **prebuilt official `swebench` images** (pullable, per the probe
  finding), so runtime alignment is a one-time pull (~0.6 min/instance), not a build.

### PRICING — throughput, not latency (batched c=4+; USER FROZEN CONFIG)

Per the directive, **speed is reported as THROUGHPUT (episodes/GPU-h) at the chosen concurrency; the v3
b=1 per-episode walls are cited for latency CONTEXT only — concurrent wall is queue-inflated and must NOT
be presented as latency.**

*Latency context (v3, b=1, `logs/*_driver.log` means, do NOT read as the batched cost):* stock-AR
~107s/episode, diffusion ~141s, merged-AR ~119s.

**GPU-compute (occupancy) is ~concurrency-invariant to first order** — total episode-compute is the same;
batching compresses WALL, not GPU-h. So the compute envelope is unchanged: **N=50 ≈ 3.5–4.3 GPU-h (2-arm)
/ 5.1–6.2 GPU-h (3-arm)** with a 1.2× turn-cap-75 headroom booked. What batching changes is the **wall**:

| item | serial (v3 cage, superseded) | **BATCHED c=4 (frozen)** |
|---|---|---|
| serving wall / arm, N=50 | ~1.5–2 h | **~2–4 h/arm** at the queue-inflated envelope in the directive |
| image pulls | ~0.6 min/inst off-GPU | **~50 Tier1 pulls (~200 GB class), ~2–4 h**, pull stage in the orchestrator, **disk-checked** |
| official scoring | ~1–2 h off-GPU | hours (off-GPU, parallel) |
| **total wall** | ~half a day | **~1–2 days** (dominated by pulls + serving + scoring) |
| **throughput (report this)** | — | **episodes/GPU-h at c=4** (probe c=6/8), e.g. ~50 episodes / (3.5–4.3 GPU-h) ≈ **12–14 ep/GPU-h** for the 2-arm compute |

**Compute reprice still stands: ~4–6 GPU-h (N=50) vs the campaign §5 "3c" line's 35–60 GPU-h — ~10× DOWN**
(that estimate conflated the eval tier with data-gen attempts; a paired resolve@1 eval is only N × arms
episodes, ≤150 at N=50/3-arm). **What the frozen config corrects vs my first draft: the WALL is ~1–2 days
(batched, ~200 GB pulls), not ~half a day, and speed is throughput not latency.** Recommendation: **N=50,
2-arm (stock-AR vs diffusion), batched c=4 (probe 6/8 HBM-gated), ~4–6 GPU-h compute / ~1–2 days wall**;
add merged-AR only if the pair diverges.

### Decision this run produces (D5)

- **If diffusion ≈ stock-AR** (CIs overlap, McNemar b−c not significant) at N=50 → the diffusion twin is
  certified an **AR-equivalent SWE server** at ranking-tier significance; the remaining diffusion problem
  is purely the **behavioral texture** (loop-halts, no clean terminals, ~1.3× wall) — an engine/decode
  fix, not a capability gap. The SWE-tuning campaign stays parked.
- **If diffusion < stock-AR** by a detectable margin → a real paradigm tax exists after all; **unpark
  this campaign** (D1 already GO at yield 0.25) to inject SWE-capable base weights and re-convert, and/or
  prioritize the decode-loop loop-halt fix. Only *this* outcome re-justifies the training spend.
- Either way, N=50 is the cheapest experiment (~4 GPU-h) that converts the n=5 tie into a decision.
