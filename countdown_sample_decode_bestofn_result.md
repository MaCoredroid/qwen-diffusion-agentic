# Countdown sample-and-decode best-of-N result

Date: 2026-07-01

Task: test the sample-and-decode best-of-N thesis for Countdown. This is inference-only and uses public
`reasoning-gym` Countdown rows. No checkpoint is promoted.

## Setup

- Model: `models/qwen3.5-9b-fastdllm-init` + `runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000`.
- Sampler: constrained Countdown arithmetic grammar from `scripts/rl_pilot_countdown.py`.
- Serving path: `RequestDiffusionState` HF route-I fast cache, `block_size=32`.
- Verifier/select: `reasoning_gym` Countdown target check; keep the first expression in the prefix that hits the target.
  The gold expression is not used for selection.
- Score split: 16 prompts per dataset, `eval_seed=2000`, no dataset seed stride.
- Stochastic best-of-N: `temperature=1.0`, nested prefixes of a single N=16 rollout per prompt.
- Greedy anchor: separate `temperature=0.0`, N=1 run to compare with the prior single-sample constrained baseline.
- AR reference for useful token throughput: `89 tok/s`.

Primary command:

```bash
.venv-fastdllm/bin/python scripts/eval_countdown_sample_decode_bestofn.py \
  --datasets easy3,standard4 \
  --score-size 16 \
  --throughput-size 16 \
  --n-values 1,2,4,8,16 \
  --out-dir runs/countdown_sample_decode_bestofn_final_t1
```

Greedy anchor command:

```bash
.venv-fastdllm/bin/python scripts/eval_countdown_sample_decode_bestofn.py \
  --datasets easy3,standard4 \
  --score-size 16 \
  --throughput-size 16 \
  --n-values 1 \
  --temperature 0.0 \
  --out-dir runs/countdown_sample_decode_bestofn_greedy_anchor_aligned
```

## Best-of-N Score Curve

Stochastic sample-and-decode (`temperature=1.0`):

| dataset | N=1 | N=2 | N=4 | N=8 | N=16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| easy 3-number | 2/16 (0.125) | 3/16 (0.188) | 7/16 (0.438) | 9/16 (0.563) | 13/16 (0.813) |
| standard 4-number | 1/16 (0.063) | 2/16 (0.125) | 3/16 (0.188) | 5/16 (0.313) | 9/16 (0.563) |

Greedy constrained N=1 anchor (`temperature=0.0`):

| dataset | greedy N=1 |
| --- | ---: |
| easy 3-number | 7/16 (0.438) |
| standard 4-number | 2/16 (0.125) |

Read: best-of-N pays off. On easy Countdown, N=16 reaches 13/16, well above both the stochastic N=1
and the known greedy 7/16 single-sample anchor. On standard Countdown, N=16 reaches 9/16 versus 1/16
stochastic N=1 and 2/16 greedy N=1.

## Throughput

Throughput is measured by separate batched N sweeps on the same 16 prompts. `expr tok/sec` counts emitted
Countdown expression tokens across all N samples and is the AR-normalized useful-compute number.

| dataset | N | samples/sec | expr tok/sec | vs 89 tok/s AR | wall x N=1 | batched speedup vs sequential N=1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy 3-number | 1 | 1.172 | 8.4 | 0.09x | 1.00x | 1.00x |
| easy 3-number | 2 | 2.337 | 15.9 | 0.18x | 1.00x | 1.99x |
| easy 3-number | 4 | 4.223 | 29.7 | 0.33x | 1.11x | 3.60x |
| easy 3-number | 8 | 8.004 | 56.0 | 0.63x | 1.17x | 6.83x |
| easy 3-number | 16 | 13.198 | 90.7 | 1.02x | 1.42x | 11.26x |
| standard 4-number | 1 | 0.866 | 9.1 | 0.10x | 1.00x | 1.00x |
| standard 4-number | 2 | 1.638 | 18.0 | 0.20x | 1.06x | 1.89x |
| standard 4-number | 4 | 3.170 | 34.0 | 0.38x | 1.09x | 3.66x |
| standard 4-number | 8 | 6.004 | 64.9 | 0.73x | 1.15x | 6.93x |
| standard 4-number | 16 | 9.747 | 104.5 | 1.17x | 1.42x | 11.25x |

GPU/process summary for the stochastic run: 229.46s total measured experiment time, GPU peak memory 22,107 MiB
by `nvidia-smi`, CUDA peak allocated 13.53 GiB, CUDA peak reserved 14.85 GiB, mean sampled GPU utilization 76.9%,
max 99%.

## Verdict

The test-time-compute payoff is real: decoder-constrained stochastic samples contain independent successes that
verify-select can harvest, especially by N=16.

The current 9B torch/HF implementation does not yet prove the 100x useful-throughput claim. At N=16 it gives
about 11.25x more useful samples than sequential N=1 for only 1.42x the per-prompt wall-clock, but the AR-normalized
useful expression-token throughput is only 1.02x easy / 1.17x standard versus the 89 tok/s AR reference. This validates
the sample-and-decode direction, but not the final 100x magnitude.
