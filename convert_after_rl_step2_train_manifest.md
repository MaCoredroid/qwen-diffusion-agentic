# Convert-After-RL STEP 2 — Fresh two-stream conversion A_new on M_{t+1} (work-item #29)

Status: **COMPLETE — 400/400 steps, training-health PASS, adapter saved.** Produced 2026-07-04, seed 80101.
Follows `convert_after_rl_design.md` §4 (design commit 6f5d20f) exactly on all pinned hyperparameters.

## Result summary

- **A_new adapter (final, step 400):** `runs/convert_after_rl/Anew_run1recipe_step400_seed80101/adapter_model.safetensors`
  - sha256 `d77dad68ecaf7f2e7800647df9a38a3edd4db4860276ee60073d55778d6944ec` (byte-identical to `checkpoint-400/adapter_model.safetensors`)
  - `adapter_config.json` sha256 `dde9c746ecc48b0886430138dbd4007dddf2535a4221884fb7ef9f156255c492`
  - r=16, α=32, dropout=0.05, targets = q/k/v/o_proj + in_proj_{qkv,z,a,b} + out_proj (9 modules, 304 tensors = 152 LoRA pairs), all finite, lora_B nonzero.
- **Training health (KILL check): PASS** — 80 logged points (steps 5–400), **zero NaN/Inf**, loss all 80 values distinct (not flat), converging. Kill NOT triggered.
- **Base = M_{t+1} merged:** `models/qwen3.5-9b-fastdllm-mtplus1-merged` (bd_size 32, mask_token_id 248077, has_weights true — STEP-1 gate PASS).

## Loss curve summary, per segment

Full run = one 400-step cosine (LR 1e-5, warmup 0.03, r16/α32, bd_size 32), executed as 4 resumable chunks.

| seg | steps | n | loss mean | min | max | first (step/loss/lr) | last (step/loss/lr) | train_runtime | tool wall | resume from |
|----:|:-----:|--:|----------:|----:|----:|:--------------------:|:-------------------:|-------------:|---------:|:-----------:|
| 1 | 1–100   | 20 | 3.818 | 2.788 | 5.395 | 5 / 3.076 / 3.33e-6 | 100 / 2.851 / 8.810e-6 | 509.2 s | 543 s | — (fresh) |
| 2 | 101–200 | 20 | 3.586 | 2.504 | 4.618 | 105 / 2.926 / 8.676e-6 | 200 / 3.880 / 5.283e-6 | 509.4 s | 538 s | checkpoint-100 |
| 3 | 201–300 | 20 | 3.185 | 2.208 | 4.355 | 205 / 2.338 / 5.081e-6 | 300 / 2.842 / 1.581e-6 | 501.7 s | 529 s | checkpoint-200 |
| 4 | 301–400 | 20 | 3.342 | 2.625 | 4.572 | 305 / 3.203 / 1.436e-6 | 400 / 3.805 / 1.639e-10 | 514.0 s | 542 s | checkpoint-300 |

Overall: loss min 2.208 / max 5.395 / mean 3.483; gentle downward trend (seg means 3.82 → 3.59 → 3.19 → 3.34) with the high per-step variance intrinsic to the MDM/two-stream objective. A_new's loss starts **lower** than the pre-RL Run-1 reference (which began ~5–7) because the merged M_{t+1} base already carries the RL-v2 capability. Total pure `train_runtime` 2034 s ≈ Run-1's 2068 s; total GPU tool-wall (4 real segments) 2152 s ≈ 0.6 GPU-h (matches design §11 estimate).

## Faithful-chunking validation (decisive)

- **LR schedule vs reference Run-1: max relative diff = 0.00e+00 at every one of the 80 logged steps.** The 4-chunk resumed run reproduces the single-run 400-horizon cosine bit-for-bit.
- Resume was proven real, not a restart: the first logged step of each resumed segment lands mid-cosine (105→8.676e-6, 205→5.081e-6, 305→1.436e-6), never at the warmup value — impossible unless optimizer/scheduler/global_step were restored from the checkpoint.
- HF Trainer restores optimizer, LR scheduler, RNG, and data-order (sampler skip) on resume, so the segmented run is equivalent to a single continuous 400-step run (the stochastic mask realizations are drawn from the restored RNG stream).

**Why not cumulative `--max_steps`:** chunking by raising max_steps per segment (100→200→300→400) would rebuild the cosine on the *segment's* horizon, decaying LR to ~0 at step 100 and corrupting the recipe. Instead every segment ran `--max_steps 400` (fixed cosine horizon) and stopped early at an absolute global_step, then resumed. Verified above.

## Provenance (design §12)

- **Dataset (design §4 ORIGINAL 5055 mix, verified):** `data/flare_redesign_run1_copy_retention_mix`, count 5055 (2048 copy-synth + 2560 gsm8k/mbpp retention + 447 public toolcall pool — does NOT include the RL-v2 pool, per design). Manifest sha256 `43ba99c1c6290ec09b277efbe34e5a5dc71c6c105bd40bd763dd418a0fe6aac2`; train json sha256 `5bc8c6feff550522d7d765c17123678fc77f7116315519c6a9672e55b4971d85`.
- **Exact command:** `scripts/convert_after_rl_step2_segment.sh` wraps design §4's verbatim env block and drives `scripts/run_flare_redesign_run1.sh`. Design-pinned hyperparameters (MAX_STEPS=400, LR 1e-5 cosine, SAVE_STEPS=100, SAVE_TOTAL_LIMIT=4, LORA_R/ALPHA/DROPOUT, targets, TRAIN_BD_SIZE=32, BLOCK_SIZE=512, GRAD_ACCUM=1, VALUE_SPAN_* , CONVERSATION_TEMPLATE, SEED/DATA_SEED=80101) are unchanged. Added only: `SKIP_DATASET_BUILD=1` (dataset prebuilt+hash-verified), `OVERWRITE_CACHE=1` fresh / `0` on resume (cache reuse; deterministic), and the chunk controls below.
- **Runner exports (unchanged two-stream/copy schedule):** FASTDLLM_FLARE_TWO_STREAM=1, GDN route_i, mask-rate 0.3–0.8, adaptive-copy schedule on, GDN kernel fla, cosine LR. Confirmed live in logs (`L_AR` + `L_diff`, `bd_size=32`, value_span weight 2.0).
- **Host-safety:** every segment ran inside `systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=4G`; GPU pre-flight (<2 GB) + `free -g` (≥8 G) gate before each; one process at a time; each segment finished inside one ≤595 s tool call.

## Infra edits (required for chunked/resumable training; env-gated, backward-compatible)

Captured as a reproducible patch in this repo: **`convert_after_rl_step2_infra.patch`** (apply to the fast-dllm working tree). The fast-dllm dir is the upstream NVlabs repo, so these are not pushable there; the patch is the version-controlled record.

- `fast-dllm/third_party/lmflow/pipeline/finetuner.py` — `StopAtStepCallback`, arms only when `FASTDLLM_STOP_AT_STEP` is set; stops training at an absolute global_step and forces a checkpoint. No-op otherwise.
- `fast-dllm/third_party/lmflow/pipeline/utils/peft_trainer.py` — when `FASTDLLM_RESUMABLE_CKPT=1`: `_save_checkpoint` falls back to stock HF checkpointing (adapter@root + optimizer + scheduler + rng + trainer_state; still adapter-only for a PeftModel, ~72 MB), and `on_save`/`on_epoch_end`/`on_train_end` skip the `adapter_model/` subdir save (which would otherwise make HF's `_load_from_checkpoint` load an untrained adapter from the subdir). No-op otherwise. This is why the original Run-1 was never resumable (adapter-only checkpoints).
- `scripts/run_fastdllm_qwen35_9b_agentic_qlora_pilot.sh` — passes `--resume_from_checkpoint` when `RESUME_FROM_CHECKPOINT` is set (no-op otherwise).

## Artifacts (all absolute)

- Adapter + checkpoints: `/home/mark/qwen_diffusion/runs/convert_after_rl/Anew_run1recipe_step400_seed80101/` (`adapter_model.safetensors`, `checkpoint-{100,200,300,400}/`, `trainer_state.json`, per-segment logs `seg0{1..4}_stop*.log`)
- Machine-readable manifest: `/home/mark/qwen_diffusion/runs/convert_after_rl/Anew_run1recipe_step400_seed80101/step2_training_manifest.json`
- Segment wrapper: `/home/mark/qwen_diffusion/scripts/convert_after_rl_step2_segment.sh`
- Infra patch: `/home/mark/qwen_diffusion/convert_after_rl_step2_infra.patch`

## Next (per design; NOT part of this step)

STEP 3 export A_new clean stream (§5) → STEP 4 eval battery a/b/c/d (§6) with KILL-2/3/4 gates and the confirmatory second seed (SEED=80102). A_new is disposable (retrain-freely): if §6 lands INCONCLUSIVE (a1 in 39–43) after two seeds, do not extend past 600 steps.
