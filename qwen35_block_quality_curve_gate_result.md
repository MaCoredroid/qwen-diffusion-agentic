# Qwen3.5 B@1000 Block Quality Curve Gate

Date: 2026-07-01

This run stopped at STEP 0. No speed curve or tool-call speed table was run,
because the full-denoise GSM8K quality anchor did not reproduce.

## Setup

- Base: `models/qwen3.5-9b-fastdllm-init`
- Adapter: `runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000`
- Harness: `scripts/measure_block_quality_curve.py`
- Sampler under test: fixed-K mutable-remask block denoising
  - fresh full masked block
  - full-context forward
  - mask token banned
  - mutable block estimate reconditioned each step
  - low-confidence positions re-masked and re-sampled
  - block finalized after exactly K denoise forwards
- Current model sentinel IDs:
  - `mask_id=248077`
  - `stop_token_ids=[248044, 248045, 248046]`
- GSM8K scoring: strict `####` extraction on
  `data/phaseA_retention/gsm8k_main_test_first20.jsonl`.

Commands:

```text
.venv-fastdllm/bin/python -m py_compile scripts/measure_block_quality_curve.py

.venv-fastdllm/bin/python scripts/measure_block_quality_curve.py \
  --run-name smoke_b8_k8_4_t16_fixedids \
  --skip-anchor-gate \
  --gsm8k-limit 1 \
  --toolcall-limit 1 \
  --block-sizes 8 \
  --k-values 8,4 \
  --max-new-tokens 16 \
  --ar-timing-rows 1 \
  --ar-timing-steps 2 \
  --out-dir runs/block_quality_curve_b1000

.venv-fastdllm/bin/python scripts/measure_block_quality_curve.py \
  --run-name anchor_b32_k32_gsm20 \
  --anchor-only \
  --gsm8k-limit 20 \
  --toolcall-limit 0 \
  --max-new-tokens 256 \
  --anchor-min-strict-accuracy 0.60 \
  --out-dir runs/block_quality_curve_b1000
```

## STEP 0 Gate

Mutable-remask full-denoise anchor, `B=32,K=32`:

- Strict GSM8K: `5/20 = 0.25`
- Gate floor used: `>=0.60`, targeting the known `~0.65` anchor
- Denoise forwards: `4000`
- Generated tokens: `3868`
- Actual generated tokens/forward: `0.967`
- Mean denoise forward seconds: `0.1441`
- Unresolved masks: `0`

Artifacts:

- `runs/block_quality_curve_b1000/anchor_b32_k32_gsm20.jsonl`
- `runs/block_quality_curve_b1000/anchor_b32_k32_gsm20.summary.json`
- `runs/block_quality_curve_b1000/anchor_b32_k32_gsm20.report.md`

Result: **FAIL**. Per the steer, the speed sweep is invalid and was not run.

## Red-Team Diagnostic

I also reran the older validated full-context fresh-block commit sampler as a
non-speed diagnostic:

```text
.venv-fastdllm/bin/python scripts/eval_flare_stage1_ab_diffusion.py \
  --base-model models/qwen3.5-9b-fastdllm-init \
  --adapter-b runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000 \
  --model-names B_two_stream \
  --skip-nll \
  --generation-tasks gsm8k \
  --generation-limit 20 \
  --block-size 32 \
  --small-block-size 32 \
  --max-new-tokens 256 \
  --threshold 0.9 \
  --temperature 0.0 \
  --top-p 0.95 \
  --out-dir runs/block_quality_curve_b1000/legacy_fullctx_commit_anchor_current
```

Current-checkout result:

- Strict GSM8K: `11/20 = 0.55`
- Flex GSM8K: `11/20 = 0.55`
- Unresolved masks: `0`
- Elapsed: `370.8s`

Artifacts:

- `runs/block_quality_curve_b1000/legacy_fullctx_commit_anchor_current/summary.json`
- `runs/block_quality_curve_b1000/legacy_fullctx_commit_anchor_current/B_two_stream_generation.json`
- `runs/block_quality_curve_b1000/legacy_fullctx_commit_anchor_current/B_two_stream_generation.jsonl`

This means the current checkout's legacy sampler is still far stronger than
the mutable-remask sampler (`0.55` vs `0.25`), but it also does not exactly
reproduce the older saved `13/20 = 0.65` artifact.

## Interpretation

The definitive big-block-at-quality speed test is blocked at the quality
anchor. The mutable fixed-K block sampler is degraded at full denoise, so any
tokens/forward or wall-clock number from lower K would be meaningless.

Do not use this run to claim a block-diffusion speedup. First fix or specify
the canonical sampler until `B=32,K=32` reproduces the known GSM8K anchor on
the current model artifacts.
