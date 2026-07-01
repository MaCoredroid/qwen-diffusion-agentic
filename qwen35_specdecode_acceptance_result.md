# Qwen3.5-9B B@1000 Self-Spec Decode Acceptance

Date: 2026-07-01

This is a measurement only. No promotion decision is made.

## Setup

- Base: `models/qwen3.5-9b-fastdllm-init`
- Adapter: `runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000`
- Harness: `scripts/measure_specdecode_acceptance.py`
- Tool-call slice:
  `runs/planner_decomposition/heldout_seed_multicall_policy_targets.jsonl`
  (`12` rows)
- GSM8K slice:
  `data/phaseA_retention/gsm8k_main_test_first20.jsonl` first `12` rows,
  with `phasea_fewshot` prompts.
- K sweep: `4, 8, 16, 32`
- Max generated tokens per row/K: `16`
- Draft: one route_i FLARE diffusion masked-block forward, argmax per
  position (`D=1`).
- Verify: one clean causal AR forward over `prefix + draft`.
- Cost model: `2` full 9B forwards per round (`1` draft + `1` verify).
- Losslessness: by construction; on first draft mismatch, emit the AR verifier
  token.

Commands:

```text
.venv-fastdllm/bin/python -m py_compile scripts/measure_specdecode_acceptance.py

.venv-fastdllm/bin/python scripts/measure_specdecode_acceptance.py \
  --toolcall-limit 1 \
  --gsm8k-limit 1 \
  --k-values 8,16 \
  --max-new-tokens 8 \
  --verify-ar-baseline-rows 1 \
  --run-name smoke_2rows \
  --out-dir runs/specdecode_acceptance_b1000

.venv-fastdllm/bin/python scripts/measure_specdecode_acceptance.py \
  --toolcall-limit 12 \
  --gsm8k-limit 12 \
  --k-values 4,8,16,32 \
  --max-new-tokens 16 \
  --verify-ar-baseline-rows 1 \
  --run-name full_tool12_gsm12_k4_8_16_32_t16 \
  --out-dir runs/specdecode_acceptance_b1000
```

Artifacts:

- `runs/specdecode_acceptance_b1000/smoke_2rows.summary.json`
- `runs/specdecode_acceptance_b1000/full_tool12_gsm12_k4_8_16_32_t16.jsonl`
- `runs/specdecode_acceptance_b1000/full_tool12_gsm12_k4_8_16_32_t16.summary.json`
- `runs/specdecode_acceptance_b1000/full_tool12_gsm12_k4_8_16_32_t16.report.md`

AR-greedy equality spot checks passed: `8/8` full-run checked records. The
remaining rows are lossless by construction from the AR verifier commit rule.

## Results

Overall full sweep:

- Records: `96`
- Rounds: `670`
- Generated tokens: `1536`
- Mean accepted draft tokens per round: `1.388`
- Median accepted draft tokens per round: `1.0`
- Mean emitted tokens per round: `2.293`
- Median emitted tokens per round: `2.0`
- Draft-token acceptance fraction: `19.54%`
- Full-draft accept round rate: `9.55%`
- Accepted-length histogram: `0:26, 1:409, 2:188, 3:43, 4:4`
- Forward-count net speedup, emitted tokens / 2 full forwards: `1.146x`
- Accepted-only speedup / 2 full forwards: `0.694x`
- Wall time: `1891.1s`
- Draft seconds: `1805.9s`
- Verify seconds: `85.1s`

By slice and K:

| Slice | K | Rounds | Mean accepted draft toks/round | Mean emitted toks/round | Net speedup emitted/2fwds | Accepted-only/2fwds | Full-draft accept % | Histogram accepted length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| GSM8K fewshot | 4 | 80 | 1.525 | 2.400 | 1.200x | 0.762x | 12.5% | `0:5, 1:39, 2:26, 3:9, 4:1` |
| GSM8K fewshot | 8 | 77 | 1.584 | 2.494 | 1.247x | 0.792x | 9.1% | `0:5, 1:34, 2:27, 3:10, 4:1` |
| GSM8K fewshot | 16 | 76 | 1.618 | 2.526 | 1.263x | 0.809x | 9.2% | `0:4, 1:33, 2:28, 3:10, 4:1` |
| GSM8K fewshot | 32 | 76 | 1.618 | 2.526 | 1.263x | 0.809x | 9.2% | `0:4, 1:33, 2:28, 3:10, 4:1` |
| toolcall heldout-12 | 4 | 91 | 1.209 | 2.110 | 1.055x | 0.604x | 9.9% | `0:2, 1:69, 2:19, 3:1` |
| toolcall heldout-12 | 8 | 90 | 1.222 | 2.133 | 1.067x | 0.611x | 8.9% | `0:2, 1:67, 2:20, 3:1` |
| toolcall heldout-12 | 16 | 90 | 1.222 | 2.133 | 1.067x | 0.611x | 8.9% | `0:2, 1:67, 2:20, 3:1` |
| toolcall heldout-12 | 32 | 90 | 1.222 | 2.133 | 1.067x | 0.611x | 8.9% | `0:2, 1:67, 2:20, 3:1` |

## Interpretation

The lossless self-speculative decode route does not clear the build gate.

- The core agentic/tool-call slice tops out at `1.067x` emitted-token speedup
  in the idealized forward-count model, and only `0.611x` accepted-only.
- The GSM8K slice is better but still tops out at only `1.263x` emitted-token
  speedup, below the `~2x` gate.
- Increasing K from `8` to `32` does not improve the tool-call slice because
  the draft diverges early; extra draft capacity is unused.
- The full 9B diffusion draft is much more expensive in wall time than the AR
  verifier on this HF path (`1805.9s` draft vs `85.1s` verify), so the real
  wall-clock economics are worse than the already-weak forward-count result.

Conclusion: do not start a multi-week vLLM integration for this exact
lossless self-spec path. The decode-side thesis needs a cheaper drafter
or a different route, such as a smaller/early-exit draft head or an explicit
lossy parallel-decode path with recovery.
