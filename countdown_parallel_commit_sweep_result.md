# Countdown constrained parallel-commit sweep result

Date: 2026-07-01

Task: measure the parallel-decode ceiling under the constrained Countdown decoder. This is the direct test of whether
the constrained lane can beat AR by committing more than one valid grammar token per diffusion forward.

## Setup

- Model: `models/qwen3.5-9b-fastdllm-init` + `runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000`.
- Data: public `reasoning-gym` Countdown, `eval_seed=2000`, 16 prompts per dataset.
- Datasets: easy 3-number (`1..10`, target `3..30`, max 24 generated tokens) and standard 4-number (`1..20`,
  target `10..100`, max 32 generated tokens).
- Decoder: Countdown arithmetic grammar stays on for every committed token.
- Serving: `RequestDiffusionState` HF route-I cache, `block_size=32`.
- Decode: greedy (`temperature=0`). The old one-token/forward path is measured as baseline. In multi-commit mode, the
  first token in a forward preserves baseline progress, then additional same-forward tokens are committed only while
  their top grammar-allowed confidence is greater than `tau`.
- Sweep: `tau in {0.99, 0.95, 0.90, 0.80, 0.70, 0.50}`.
- Metric: `tokens/forward = committed_tokens / denoise_forwards`.
- No promotion.

Command:

```bash
.venv-fastdllm/bin/python scripts/eval_countdown_parallel_commit_sweep.py \
  --datasets easy3,standard4 \
  --eval-size 16 \
  --commit-thresholds 0.99,0.95,0.9,0.8,0.7,0.5 \
  --out-dir runs/countdown_parallel_commit_sweep_final
```

Verification before the full run:

```bash
.venv-fastdllm/bin/python -m py_compile \
  scripts/rl_pilot_countdown.py \
  scripts/eval_countdown_parallel_commit_sweep.py \
  scripts/eval_countdown_sample_decode_bestofn.py

.venv-fastdllm/bin/python scripts/eval_countdown_parallel_commit_sweep.py \
  --datasets easy3 \
  --eval-size 1 \
  --commit-thresholds 0.99,0.5 \
  --out-dir runs/countdown_parallel_commit_sweep_smoke
```

## Easy 3-number Countdown

Baseline one-token/forward: `1.000 tokens/forward`, strict `7/16`, mean RG `0.4656`.

| tau | tokens/forward | strict pass | mean RG | committed tokens | forwards |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.99 | 1.603 | 7/16 | 0.4656 | 117 | 73 |
| 0.95 | 1.625 | 7/16 | 0.4656 | 117 | 72 |
| 0.90 | 1.625 | 7/16 | 0.4656 | 117 | 72 |
| 0.80 | 1.721 | 7/16 | 0.4656 | 117 | 68 |
| 0.70 | 1.746 | 7/16 | 0.4656 | 117 | 67 |
| 0.50 | 2.600 | 3/16 | 0.2281 | 117 | 45 |

Held-quality headline: `tau=0.70` gives `1.746 tokens/forward` while matching the one-token greedy baseline
(`7/16`, mean RG `0.4656`).

## Standard 4-number Countdown

Baseline one-token/forward: `1.000 tokens/forward`, strict `2/16`, mean RG `0.1688`.

| tau | tokens/forward | strict pass | mean RG | committed tokens | forwards |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.99 | 1.456 | 2/16 | 0.1688 | 182 | 125 |
| 0.95 | 1.468 | 2/16 | 0.1688 | 182 | 124 |
| 0.90 | 1.480 | 2/16 | 0.1688 | 182 | 123 |
| 0.80 | 1.583 | 2/16 | 0.1688 | 182 | 115 |
| 0.70 | 1.701 | 2/16 | 0.1688 | 182 | 107 |
| 0.50 | 2.563 | 1/16 | 0.1094 | 182 | 71 |

Held-quality headline: `tau=0.70` gives `1.701 tokens/forward` while matching the one-token greedy baseline
(`2/16`, mean RG `0.1688`).

## Interpretation

The constrained decoder can safely exploit some diffusion parallelism, but the measured ceiling is small:
about `1.7 tokens/forward` at held quality on both easy and standard Countdown. Dropping to `tau=0.50` reaches
about `2.6 tokens/forward`, but quality breaks immediately on both datasets.

This is not the `4-8 tokens/forward` result that would keep a 10x path alive from constrained parallel commit alone,
and it is nowhere near a 100x path. It also is not exactly dead at `~1.0`: the decoder can take a real but modest
diffusion-specific speedup. Honest conclusion: constrained parallel decode gives roughly a `1.7x` held-quality
commit multiplier in this test. The 10x/100x goal still needs other levers; this result aligns with the prior
raw-corruption finding that aggressive multi-position joint prediction is unreliable.

GPU/process summary: 180.48s total, GPU peak memory 22,079 MiB by `nvidia-smi`, CUDA peak allocated 11.19 GiB,
CUDA peak reserved 14.82 GiB, sampled mean GPU utilization 53%, max 56%.
