# ITERATION-2 TWO-ARM SWE-SFT RETRAIN — LAUNCH NOTE (#127)

**Launched 2026-07-13 (detached + caged). Runner LIVE; arm-1 training.**

Retrain-freely re-run of the iteration-1 two-arm SWE-SFT, **mirroring iteration-1
EXACTLY except the dataset**. Provenance for the mirrored config: iteration-1
arm-1 = `scripts/swe_sft_arm1_driver.sh`, arm-2 = `scripts/swe_sft_arm2_driver.sh`,
both driving `scripts/swe_sft_arm1_qlora_train.py` (AR single-stream causal QLoRA,
SDPA attn monkeypatch, chunked-CE, 4-bit NF4). Those produced
`Aswe_S_step400_seed71101` / `Aswe_T_step400_seed71101`.

## What changed vs iteration-1 (and ONLY these)
1. **Dataset.** iter-1 trained on a single front-truncated window per episode
   (last 12288 tokens, 334 keepers, 69.88% label retention). iter-2 trains on the
   **#126 shape-corrected EPISODE-WINDOWED pool**:
   `data/swe_sft_pool/train_swe_sft_windowed.tokenized.jsonl`
   → **987 windows / 383 episodes / 100% assistant-label retention / 0 truncated**.
   Consumed via the audit-prescribed downstream path
   (`build_swe_sft_lmflow_pretok.py --tokenized <windowed> --max-len 12288`;
   left-trunc is a no-op since every window ≤ 12286 ≤ block). Built LMFlow json:
   `data/swe_sft_pool/lmflow_pretok_iter2/swe_sft_train.json`
   (987 instances, 1,454,824 label tokens, sha256 dc551ac2…).
2. **Output dirs.** `runs/swe_sft_arm1_iter2/` (arm S, merged-RL-v2 base) +
   `runs/swe_sft_arm2_iter2/` (arm T, stock-init base).
3. **Step count = 400 (UNCHANGED).** iteration-1's `HORIZON=400` is a **fixed
   cosine horizon / design constant** — it defines the {100,200,300,400}
   erosion-sweep checkpoints and the anchor-gate cadence, and is **not derived
   from dataset row count** anywhere in the iter-1 config (literal default in the
   driver, the segment script, and the trainer). Per the mirror-the-LOGIC rule
   (the cap-600 branch applies ONLY if steps were dataset-size-derived, which they
   are not), we keep 400.
   - *Observation (not a change):* 400 steps over 987 windows ≈ **0.41 pass**;
     iter-1 was ≈ 1.2 passes over 334 episode-rows. Retrain-freely: if 400 steps
     undertrains the larger pool, escalate the horizon later. Kept 400 here to
     preserve an exact same-hyperparameter mirror for the arm-vs-arm comparison.

## Identical to iteration-1 (frozen)
arm-1 base = merged-RL-v2 `models/qwen3.5-9b-fastdllm-mtplus1-merged`; arm-2 base =
stock `models/qwen3.5-9b-fastdllm-init`. QLoRA r16 / α32 / dropout 0.05, **same 11
targets** (`q,k,v,o + in_proj_qkv/z/b/a + out_proj + gate_up_proj,down_proj`),
AR single-stream, **block 12288**, chunked-CE (logits-chunk 2048), **seed 71101**,
LR 1e-5 **cosine** warmup 0.03, save-steps 100, save-total-limit 6, logging-steps 5,
resume auto. Block 12288 is **not re-probed** (windows are block-fit by construction;
arm-2 iter-1 also hard-set 12288).

## Runner (`runs/swe_sft_iter2_runner.sh`)
Sequential: **arm-1 → completion → arm-2**. Emits `[state]` lines per arm
(step / loss / s_per_step / ETA, parsed from each arm's `metrics.jsonl`). Pidfile
`runs/swe_sft_iter2.pid` (runner pid; removed on EXIT). STOP-file
`runs/swe_sft_iter2.STOP` checked **between arms** → graceful stop after arm-1.
On arm crash (nonzero rc or missing `checkpoint-400`) → logs `[state] ARM_FAILED`
and **stops; NO auto-retry**. Cage (mirrors #110-era): setsid + per-arm
`systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=4G` + `reset-failed`
pre-boot + EXIT cleanup trap.

## Preflight asserts (all PASS at launch)
- **dataset rows = 987** (windowed jsonl; hard assert). episodes = 383.
  *Hash note:* the file sha256 = `909c92ea…`, which **matches the #126 audit's own
  recorded output sha exactly** (`windowed_dataset_audit.json:27`) → this is the
  audited artifact. The task brief's cited "30ff604…" matches neither the file nor
  the audit and appears to be a mis-citation; proceeded on the binding row-count
  assert (987 ✓).
- GPU free: 386 MiB used / 0% util (< 1 GB). host RAM free 27 G. disk 2.6 T free.

## First-step evidence (verify PASS)
pid alive; first optimizer steps logged; loss finite & decreasing:
`step 5 loss 0.3789 (warmup lr 4.17e-6)`, `step 10 loss 0.2609`, ~5.97 s/step,
peak 26.6 GiB (safe on 32 GB; iter-1 peaked ~24.8–26). ETA ≈ **40 min/arm**,
≈ 80 min both arms sequential.

## Monitor / control
- `tail -f runs/swe_sft_iter2_runner.log` (the `[state]` stream)
- `tail -f runs/swe_sft_arm1_iter2/Aswe_S_step400_seed71101/train.log`
- Graceful stop after arm-1: `touch runs/swe_sft_iter2.STOP`
- Checkpoints: `ls runs/swe_sft_arm{1,2}_iter2/Aswe_*/checkpoint-*`

## Next (post-run, NOT this task)
KILL-T1 anchor gates per arm at {100,200,300,400}, then #128/#129 K-conversion +
C46 re-gate.
