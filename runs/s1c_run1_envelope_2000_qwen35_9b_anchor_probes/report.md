# S1c Run-1 Envelope Probe

Superseded anchor note: the anchor probes in this report used the mutable-remask
fixed-K sampler from `scripts/measure_block_quality_curve.py`. That path is now
classified as an invalid third scale for GDN-state/serving continuity. Use
`runs/eval_drift_rebaseline/report.md` for the corrected legacy full-context
rebaseline.

## Verdict

The S1c training run completed, but the decisive sanity gate failed.

Checkpoint-400 did not reproduce the Run-1 anchor. It scored `5/20 = 0.25` instead of the expected historical `~0.55`. A same-harness recheck of the original Run-1 adapter scored only `3/20 = 0.15`, so the S1c results do not isolate the effect of training steps. Under the requested rule, the budget comparison is void until the anchor/eval environment drift is resolved.

No r128 escalation and no S2 should be launched from these numbers.

## S1-Final Careful Check

Before S1c, S1-final was evaluated with the validated full-context careful GSM8K path:

| Adapter | Decode | GSM8K strict | GSM8K flex | Interpretation |
| --- | --- | ---: | ---: | --- |
| `runs/s1_budget_retrain_r64_qwen35_9b` | fullctx careful, B32/small32, threshold 0.9 | 6/20 = 0.300 | 6/20 = 0.300 | general careful-mode damage, not block-only |

Artifact: `runs/s1_budget_retrain_r64_qwen35_9b_careful_gsm8k_final/report.md`.

## S1c Training

Launcher: `scripts/run_s1c_run1_envelope_2000.sh`

Run directory: `runs/s1c_run1_envelope_2000_qwen35_9b`

The launcher pins the Run-1 envelope and changes only the intended probe variables:

- `MAX_STEPS=2000`
- `SAVE_STEPS=200`
- `SAVE_TOTAL_LIMIT=20`
- `SKIP_DATASET_BUILD=1` to keep the existing Run-1 mix bytes fixed

Run-1 envelope settings pinned by the launcher:

- model: `/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-init`
- data: `data/flare_redesign_run1_copy_retention_mix/train_agentic_mix.json`
- LoRA: `r=16`, `alpha=32`, `dropout=0.05`
- target modules: `q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj`
- scheduler: `cosine`, `learning_rate=1e-5`, `warmup_ratio=0.03`
- batch/masking: `BLOCK_SIZE=512`, `TRAIN_BD_SIZE=32`, `GRAD_ACCUM=1`
- GDN kernel: `fla`
- seed/data seed: `71101`
- no `LORA_MODEL_PATH`

Training result:

| Metric | Value |
| --- | ---: |
| Steps | 2000 |
| Runtime | 10293.8012 s |
| Train loss | 2.928263385176659 |
| Steps/sec | 0.194 |
| Epoch | 0.5589714924538849 |
| Peak CUDA allocated | 18836.2007 MiB |

Hashes:

| Artifact | SHA256 |
| --- | --- |
| `data/flare_redesign_run1_copy_retention_mix/train_agentic_mix.json` | `5bc8c6feff550522d7d765c17123678fc77f7116315519c6a9672e55b4971d85` |
| `scripts/run_s1c_run1_envelope_2000.sh` | `0acbd50909edc1e46e835518dc07363cfe982f3e7d1ead1da92db194abd467db` |
| `scripts/run_flare_redesign_run1.sh` | `bd9ce05f90fbbcca9703dcdcdc6bdc38b6219f639c7898f8bcbb5807876e1579` |
| `scripts/measure_block_quality_curve.py` | `34b9957bb255cd59ac02cab313a5b4453d728ad50761dc40c8440580b231bf65` |

## Anchor Probes

Probe harness:

- `scripts/measure_block_quality_curve.py`
- anchor-only mutable-remask fixed-K full-context sampler
- GSM8K first20 with Phase-A 5-shot prompt
- `B=32`, `K=32`
- `max_new_tokens=256`
- `temperature=0.0`, `top_p=0.95`
- `FASTDLLM_GDN_KERNEL=fla`
- `--anchor-min-strict-accuracy 0.0` so every checkpoint is reported even if bad

| Adapter | Historical expectation | Current strict | Result |
| --- | ---: | ---: | --- |
| Original Run-1 `runs/flare_redesign_run1_copy_grounded_qwen35_9b` | ~0.55 | 3/20 = 0.150 | current harness does not reproduce historical Run-1 |
| S1c checkpoint-400 | should replicate ~0.55 | 5/20 = 0.250 | sanity gate FAIL |
| S1c checkpoint-1000 | step-effect point | 6/20 = 0.300 | invalid as budget comparison |
| S1c checkpoint-2000 | step-effect point | 6/20 = 0.300 | invalid as budget comparison |

## Interpretation

The S1c 400/1000/2000 curve is `0.25 -> 0.30 -> 0.30`, but it cannot answer whether more steps help or hurt because the required checkpoint-400 replication failed.

The original Run-1 adapter also fails to reproduce the historical anchor under the same current command (`0.15`), which points to eval/environment drift or a stale-anchor mismatch in the block-quality path. This must be fixed before using these numbers for any scaling decision.

## Artifacts

- S1c train results: `runs/s1c_run1_envelope_2000_qwen35_9b/train_results.json`
- S1c adapter config: `runs/s1c_run1_envelope_2000_qwen35_9b/adapter_config.json`
- S1c train log: `runs/s1c_run1_envelope_2000_qwen35_9b/train.log`
- Checkpoint-400 summary: `runs/s1c_run1_envelope_2000_qwen35_9b_anchor_probes/s1c_ckpt400_anchor_b32_k32_gsm20.summary.json`
- Checkpoint-1000 summary: `runs/s1c_run1_envelope_2000_qwen35_9b_anchor_probes/s1c_ckpt1000_anchor_b32_k32_gsm20.summary.json`
- Checkpoint-2000 summary: `runs/s1c_run1_envelope_2000_qwen35_9b_anchor_probes/s1c_ckpt2000_anchor_b32_k32_gsm20.summary.json`
- Original Run-1 current-harness summary: `runs/s1c_run1_envelope_2000_qwen35_9b_anchor_probes/run1_original_current_anchor_b32_k32_gsm20.summary.json`
