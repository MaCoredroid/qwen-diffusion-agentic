# HF FLARE Cache Build Result (2026-07-01)

Scope: Stage-1 HF serving foundation after Stage-0a resolved the correct head as
route_i FLARE noisy logits plus the train-matched `+1`/right shift.

Implementation commit:

- `7718336` `Add HF FLARE serving cache foundation`

## Built

- `scripts/flare_hf_cache.py`
  - `RequestDiffusionState` with batched lockstep `[B,...]` state.
  - 24 GDN clean-boundary states stored fp32.
  - Raw pre-conv GDN tails stored from `in_proj_qkv`.
  - 8 full-attention clean-prefix KV caches.
  - `read`: route_i noisy active-block forward seeded from clean carriers.
  - `advance`: separate clean-causal incremental forward over the committed block.
  - Shift splice uses the previous FLARE noisy-stream block logit, not a clean-causal
    substitute.
- `scripts/eval_fastdllm_toolcall_cases.py`
  - `--use-block-cache --full-context-sampling --denoise-logit-mode flare_shift`
    dispatches all three denoise helper call sites through the cache.
  - Cache mode is fail-closed for non-`flare_shift`.
  - Active window is doc-block anchored from `state.block_start`.
  - Residual full-context model-call assertion is wired and reported in cache stats.
- `scripts/validate_flare_hf_cache.py`
  - T1/T2/T3 validator with drift distribution metrics.
- `scripts/measure_flare_hf_cache_throughput.py`
  - B=1/B>1 cached read and commit-one throughput harness.

## T1/T2/T3

Adapter: `runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000` on
`models/qwen3.5-9b-fastdllm-init`.

Artifacts:

- Tiny smoke: `runs/flare_hf_cache/tiny_validation_drift_metrics.json`
- Real 8-block gate: `runs/flare_hf_cache/real_validation_b1_blocks8_drift_metrics.json`

Real-weight result:

| Gate | Result |
| --- | --- |
| T1 per-token argmax parity | PASS, `0` argmax flips over 8 blocks / 256 active positions |
| T2 shifted serving-vs-training drift | ACCEPTABLE for exact-re-score spine; not bit-exact |
| T3 byte canary | PASS, cache-ON token stream and decoded bytes identical |
| Residual full-context calls | `0` in T1/T2 and T3 cache stats |

T2 magnitude over 8 real-weight blocks:

- max abs logit diff: `0.2890625`
- max abs logprob diff: `0.2992525`
- mean abs logit diff: `0.01933`
- mean abs logprob diff: `0.01903`
- mean KL(reference || cached): `3.39e-4`
- max abs logprob delta at the reference top-1 token: `0.08880`
- argmax flips: `0`

Per-block abs logit quantiles were stable rather than growing with block index:
median around `0.015625`, p90 around `0.03125-0.05078`, p99 around
`0.0625-0.08594`, max around `0.15625-0.28906`.

Noise-vs-systematic read: **this looks like bf16 shape/order drift, not a semantic
cache mismatch.** Reasons:

- Tiny fp32 cache-vs-reference is near exact (`~2e-6` max), so the mask/shift/state
  logic is correct in exact-ish arithmetic.
- Real drift is already present on block 0, where the cache has zero GDN state and no
  prefix KV, which rules out accumulated clean-boundary corruption as the primary cause.
- The real reference computes two complementary noisy views while serving computes only
  the served noisy view; bf16 matmul/GDN reductions are shape/order dependent here.
- Drift does not grow monotonically with prefix/block index.
- Top-token decisions stayed identical in T1 and T3, and the exact-re-score spine uses
  the training forward for RL log-probs.

Therefore this is not promoted as bit-exact, but it is acceptable for the intended
sample-with-serving / score-with-training spine unless a downstream stochastic sampler
shows actual token-distribution flips near ties.

## Throughput

Artifacts:

- `runs/flare_hf_cache/throughput_b1_b16_prefix256.json`
- `runs/flare_hf_cache/throughput_b1_b16_prefix1024.json`
- `runs/flare_hf_cache/throughput_b8_b12_prefix1024.json`

Commit-one cached generation throughput:

| Prefix | Batch | Result |
| ---: | ---: | ---: |
| 256 | 1 | `19.68` policy tok/s |
| 256 | 16 | `224.17` policy tok/s |
| 1024 | 1 | `19.87` policy tok/s |
| 1024 | 8 | `153.46` policy tok/s |
| 1024 | 12 | `199.20` policy tok/s |
| 1024 | 16 | OOM on 32 GB RTX 5090 |

Short-prefix B=16 clears the Stage-0b projection (`85-128 tok/s`) with room to spare.
At 1024-token prefixes, B=16 does not fit on this card, but B=12 still exceeds the
projection and the overnight RL throughput target. All measured cache stats report
`residual_full_context_model_calls=0`.

## Status

Foundation is built, validated for argmax/byte identity, and pushed. It is **not promoted
as bit-exact**; T2 is accepted as small bf16 serving drift under the exact-re-score RL
spine. Monitor red-team is still required before any production promotion.
