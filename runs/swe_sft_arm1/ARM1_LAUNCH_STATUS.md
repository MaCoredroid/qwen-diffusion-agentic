# SWE-SFT arm-1 (M_swe_S) — LAUNCH STATUS: **LAUNCHED** (handover executed)

> **UPDATE 2026-07-09 (later) — BOTH BLOCKERS RESOLVED; run is LIVE.**
> - **Blocker 1 (GPU handover):** DONE. datagen orch stopped, `batch_0005` sacrificed
>   (ids re-drawable), GPU settled 387 MiB/0%. Preflight passes.
> - **Blocker 2 (block_size):** RESOLVED by amendment + measurement. The two-stream FLARE
>   path is **replaced** by an **AR single-stream causal QLoRA** trainer
>   (`scripts/swe_sft_arm1_qlora_train.py`), evidence = #29 (`convert_after_rl_result.md`,
>   b019b86: plain-train+reconvert preserves gains, McNemar 0, 2 seeds; parity enforced at
>   CONVERSION not SFT). SDPA-attn (in-process) + `FASTDLLM_GDN_KERNEL=fla` + chunked-CE +
>   4-bit QLoRA lift the feasible block to **12288** (measured; 16384 thin-margin 1.9 GiB
>   rejected; 24576/32768 OOM). Dataset rebuilt 323→**334** keepers. Truncation @12288:
>   328/334 truncated, 0 zero-label, **69.88 %** assistant-label retention.
> - **Config unchanged** (r16/α32; targets q,k,v,o+GDN+MLP gate_up/down; LR1e-5 cosine
>   warmup0.03; HORIZON400; seed71101; SAVE_STEPS100). **Objective changed**: single-stream
>   AR CE (shift-by-one, = FLARE `L_AR`), NOT two-stream.
> - **LAUNCHED detached+caged** (`bash scripts/swe_sft_arm1_driver.sh`):
>   out `runs/swe_sft_arm1/Aswe_S_step400_seed71101/`, pidfile `runs/swe_sft_arm1/train.pid`,
>   metrics `runs/swe_sft_arm1/metrics.jsonl`. To stop for a faithful resume:
>   `kill -TERM -$(cat runs/swe_sft_arm1/train.pid)` (trainer checkpoints then exits; also
>   resumable from the latest 100-step checkpoint). The full amendment + probe table +
>   #29 evidence chain live in `swe_tuning_campaign_design.md` STATUS (2026-07-09, later).
>
> _Everything below is the pre-handover ARMED record (retained for provenance; the
> two-stream feasibility table and the "objective = two-stream FLARE" line are SUPERSEDED
> by the amendment above)._

---

# (ARCHIVE) SWE-SFT arm-1 (M_swe_S) — LAUNCH STATUS: ARMED, BLOCKED ON GPU HANDOVER

**As of 2026-07-09T19:12Z.** Owner: task #110 / campaign `swe_tuning_campaign_design.md` §2.
Base = merged-RL-v2 `models/qwen3.5-9b-fastdllm-mtplus1-merged` (the HF-stack realization of
design arm S; same base convert-after-RL step-2 and the S2 pilot trained on). Pool =
`data/swe_sft_pool/train_swe_sft.tokenized.jsonl`, **323 clean keepers** (serve-exact
input_ids + assistant_spans; firewall/quarantine already asserted by the builder).

## NOT launched this turn — two independent hard blockers

### BLOCKER 1 — GPU is held by LIVE datagen (external; not mine to stop)
`datagen_orch.sh` (pid 2888673) is running and is **actively generating `batch_0005`**
(started 18:50Z, i.e. *after* this task was written — the orchestrator auto-advanced past
batch_0004). The 27B NVFP4 vLLM teacher (pid 176283) holds **~30 GB at 98% util**. The
"GPU handover" the task brief anticipated after batch_0004 has **not** occurred. Launching
training now is impossible (no free VRAM) and would corrupt the in-flight datagen batch, so
per the standing "do NOT interfere with the datagen machinery" rule the launcher **refuses**
to run while the GPU is busy (`swe_sft_arm1_driver.sh` preflight, verified: aborts exit 9).
Handover = the 27B GPU teacher stops; the **Opus-4.8 API track (CPU/API) keeps collecting
keepers under training by design**, and chunked-resume folds them in later.

### BLOCKER 2 — the design's block_size 32768 is INFEASIBLE (MEASURED, not assumed)
The certified SFT path on this stack is the two-stream FLARE trainer
(`run_flare_redesign_run1.sh`; used by multiturn-SFT-warmstart, convert-after-RL step-2, S2
pilot). It **concatenates the clean + noisy streams to length 2L**, so:
- LM-head logits are `[2L, vocab=248320]` ≈ **16 GB** at L=16384 (×bf16), and
- it materialises an **O((2L)^2) boolean attention mask** (`flare_two_stream_bool_mask`).

**Measured on the 5090 (32 GB):** block_size **16384 → CUDA OOM** at 30.7 GB in use, inside
the mask build (`modeling.py:223`). So two-stream tops out at **block_size ≈ 8192**
(2L=16384 → logits ~8 GB). All 323 SWE trajectories are long (p50 24 047, max 32 768 tokens;
~84 % is loss-masked tool/context, ~16 % assistant targets). Left-truncation to a feasible
block keeps the **final edit turns** (highest-value targets) and drops earliest context.
Assistant-**target** retention vs block (builder-measured):

| block_size | label retention | rows truncated | two-stream fit (5090) |
|---:|---:|---:|:--|
| 8192  | 48.9 % | 323/323 | fits (logits 2L ~8 GB) — **safe floor** |
| 12288 | 69.6 % | 320/323 | probe at handover (logits 2L ~12 GB; likely fits) |
| 16384 | 83.0 % | 305/323 | **OOM (measured)** |
| 24576 | 99.8 % | 131/323 | far OOM |
| 32768 | 100 % | 0/323 | far OOM (design value) |

The design's "6–12 GPU-h / block up to 32768" (§2.3/§5) did not account for the vocab-248k ×
2L logits term. This is a genuine hardware-vs-spec conflict, resolved by measurement:
**train at the largest block that fits.** The launcher auto-selects it via a 2-step probe
ladder `CANDIDATES="12288 8192 6144"`.

## Config (frozen, per design §2.3/§2.4 except the measured block cap)
- objective: two-stream FLARE (the only proven chunked-resume SFT path). Its **L_AR** stream
  is the autoregressive SWE-SFT objective the design's "AR-side SFT" calls for; L_diff also
  exercises the diffusion stream (harmless — §3.1's fresh conversion still excludes the SWE
  pool, so the preservation sharp-test is intact). *Owner note:* pure-AR (mdm 0) is not a
  proven chunked-resume path here; if 48.9–69.6 % retention proves inadequate, the
  retention-recovery levers are (a) a chunked/fused cross-entropy to kill the logits term and
  unlock block 24–32 k, or (b) a single-stream path — both are retrain-freely follow-ups.
- ingestion: S2 pre-tokenized passthrough (`train_s2_finetune.py` + `FASTDLLM_S2_PRETOK=1`)
  → **zero re-tokenization**, native qwen3_xml guaranteed (bypasses the `fast_dllm_v2_native`
  whitespace divergence the dataset manifest flagged).
- LoRA: r=16 α=32 dropout 0.05; targets = q,k,v,o + GDN in_proj_{qkv,z,b,a} + out_proj +
  **MLP gate_up_proj, down_proj** (the design §2.3 widen for SWE capability).
- LR 1e-5 **cosine**, warmup 0.03; per_device_bsz 1, grad_accum 1; bd_size 32;
  HORIZON=400 steps (fixed cosine horizon), SAVE_STEPS=100 → {100,200,300,400} erosion-sweep
  checkpoints; seed 71101. Caged `MemoryMax=22G`.
- chunked resume: `FASTDLLM_RESUMABLE_CKPT=1` (adapter@root + optimizer + scheduler + rng +
  trainer_state); a killed run resumes bit-faithfully from the latest checkpoint (proven in
  convert-after-RL step-2).

## Artifacts (all absolute)
- builder: `/home/mark/qwen_diffusion/runs/swe_datagen_s1/build_swe_sft_lmflow_pretok.py`
  (tokenized keepers → LMFlow text_only `input_ids`+`labels`, left-trunc + span remap)
- segment: `/home/mark/qwen_diffusion/scripts/swe_sft_arm1_segment.sh` (one training segment;
  verified to load data + model + reach the two-stream forward)
- launcher: `/home/mark/qwen_diffusion/scripts/swe_sft_arm1_driver.sh` (preflight → feasibility
  ladder → full-dataset build → detached caged launch; preflight-guard verified)

## TO LAUNCH once the GPU is free (handover done)
```
cd /home/mark/qwen_diffusion && bash scripts/swe_sft_arm1_driver.sh
```
It preflights (aborts if GPU busy), picks the largest feasible block, builds the 323-row
dataset, runs a 2-step verify-smoke, then detaches the full run
(`runs/swe_sft_arm1/Aswe_S_step400_seed71101/`, pidfile `runs/swe_sft_arm1/train.pid`,
metrics `runs/swe_sft_arm1/metrics.jsonl`). ETA is unknown until the first real steps are
timed at the chosen block (record s/step from the probe log to project 400 steps).

## ANCHOR GATE (blocking, per design §2.5 / KILL-T1) — run at each 100-step checkpoint
For checkpoint-{100,200,300,400}, AR-mode: **tool-call matched-20 exact_args** via
`eval_flare_northstar_matched.py`. PASS = McNemar net-loss vs the arm's pre-SFT anchor not
significant (p≥0.05) AND raw ≥ anchor − 3. Anchor for arm S ≈ 47/63 hybrid · 44/63 careful.
Fail with no step-count that both lifts SWE and holds the anchor ⇒ KILL-T1 (fall back to
arm-2 stock). Secondary: GSM8K legacy N=20 ≥ anchor−1; value-projection audits all-0.
