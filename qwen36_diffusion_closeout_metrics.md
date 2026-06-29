# Qwen3.6 Diffusion Closeout Metrics

Date: 2026-06-26

## Purpose

Define what it means to finish the Qwen3.6 diffusion experiment. The target is
not just reproducing Fast-dLLM. The target is a converted block-diffusion
Qwen-family model that can run agentic coding and tool workflows with measurable
quality.

The main closeout metric is SWE-bench Verified, but it is too expensive and too
coarse to use as the first signal. We should only run full SWE-bench Verified
after strict tool-call, code-edit, and harness-stability metrics are already
passing.

Goal framing, 2026-06-28: closeout is behavior preservation, not just
diffusion training. `DIFF_TRAIN_Q36` must be measured as a converted version of
the AR Qwen policy. Protected tool-calling scores are useful operational
metrics, but the closeout evidence must include raw and constrained-decoder
movement toward `AR_Q36`, not only deterministic repair/projection wins.

## Where This Fits

Primary plan:

- `agentic_diffusion_qwen_plan.md`

Lower-level conversion runbook:

- `diffusion_qwen_distillation_runbook.md`

This document adds the missing closeout definition: baseline measurements,
training-improvement targets, and readiness gates.

## Benchmark Grounding

SWE-bench Verified is the right final benchmark because it measures real GitHub
issue resolution by patch generation and unit-test verification.

Current source notes:

- Official SWE-bench Verified page: 500 human-filtered instances from SWE-bench.
- Hugging Face dataset card: 500 test examples; evaluation is unit-test
  verification against post-PR behavior.
- Official leaderboard note: `% Resolved` is solved instances out of 500
  Verified examples.
- Official Verified page recommends mini-SWE-agent-style bash-only results when
  comparing LMs directly, instead of mixing arbitrary full agent systems.

Links:

- https://www.swebench.com/verified.html
- https://www.swebench.com/
- https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified
- https://github.com/SWE-agent/mini-swe-agent

As of 2026-06-26, the official leaderboard data lists `Qwen3-Coder
480B/A35B Instruct` at 55.4% resolved in the bash-only mini-SWE-agent setting.
That is not our baseline because it is a different model and agent version, but
it gives scale: a 27B local diffusion model scoring 30-45% on full Verified
would already be a serious result.

## Baselines To Record

Every SWE-style run must record the exact model, quantization, harness config,
agent version, context length, max steps, sampling params, and hardware.

Use three baselines:

| Name | Meaning | Required before closeout? |
| --- | --- | --- |
| `AR_Q36` | SGLang-served Qwen3.6-27B AR teacher/reference | yes |
| `DIFF_INIT_Q36` | Qwen3.6 converted to diffusion mode before diffusion training | yes |
| `DIFF_TRAIN_Q36` | trained Qwen3.6 diffusion checkpoint | yes |

The key comparison is:

```text
AR retention = DIFF_TRAIN_Q36 resolved % / AR_Q36 resolved %
training lift = DIFF_TRAIN_Q36 resolved % - DIFF_INIT_Q36 resolved %
```

Do not use public leaderboard numbers as the project baseline. Use our local
mini-SWE-agent configuration and run all three models through the same harness.

## SWE-Bench Closeout Metrics

Use a fixed instance order and publish the IDs in the run output.

### Slice Runs

Run these before the full 500:

- `SWEV-20`: 20 fixed instances for smoke tests.
- `SWEV-50`: 50 fixed instances for checkpoint selection.
- `SWEV-100`: 100 fixed instances before any full run.

Slice pass gates:

| Gate | Minimum result |
| --- | ---: |
| `DIFF_INIT_Q36` measured | required, no minimum |
| `AR_Q36` measured | required, should be at least 20% on `SWEV-50` or the harness/model setup is suspect |
| `DIFF_TRAIN_Q36` on `SWEV-50` | at least `DIFF_INIT_Q36 + 10` solved instances |
| `DIFF_TRAIN_Q36` retention on `SWEV-50` | at least 60% of `AR_Q36` solved instances |
| patch apply rate | at least 80% |
| harness crash/no-generation rate | below 5% |

If the trained diffusion model cannot clear `SWEV-50`, do not spend the compute
on full Verified. Fix tool boundaries, patch generation, or block decoding first.

### Full Verified

Full closeout on 500 instances:

```text
DIFF_TRAIN_Q36 resolved % >= max(
    DIFF_INIT_Q36 resolved % + 15 percentage points,
    0.70 * AR_Q36 resolved %,
    30%
)
```

Project-success target:

```text
DIFF_TRAIN_Q36 resolved % >= max(
    DIFF_INIT_Q36 resolved % + 25 percentage points,
    0.80 * AR_Q36 resolved %,
    45%
)
```

Stretch target:

```text
DIFF_TRAIN_Q36 resolved % >= max(
    0.90 * AR_Q36 resolved %,
    55%
)
```

These are deliberately tiered. The minimum closeout proves the diffusion model
is not just broken AR weights in a diffusion sampler. The success target says it
is plausibly useful. The stretch target says it is competitive with serious
agentic coding models.

## Required Non-SWE Gates

SWE-bench should not be the first sign that the model can act. The following
must pass first.

### Tool Calls

Run synthetic and public tool-call evals with no repair, then with constrained
repair.

| Metric | 9B gate | Qwen3.6 closeout gate |
| --- | ---: | ---: |
| synthetic one-call raw exact tool name | 90% | 95% |
| synthetic one-call valid strict JSON | 80% | 90% |
| public one-call exact tool name | 80% | 90% |
| public one-call argument schema valid | 75% | 85% |
| two-step trace exact tool sequence | 60% | 80% |
| repeated-call loop rate | below 10% | below 5% |
| unresolved mask examples | 0% | 0% |

Repair metrics are secondary. A model that only works after name-only repair is
not ready for SWE-bench.

### Code Edit Harness

Before SWE Verified, run a local repo-edit harness:

- 20 tiny patch tasks with unit tests.
- 20 medium patch tasks with real dependency setup.
- same prompt/harness for AR and diffusion.

Gate:

- tiny tasks: at least 70% tests pass.
- medium tasks: at least 40% tests pass.
- patch apply rate: at least 90%.
- no stop-boundary or infinite-loop failures.

### Generic Retention

The diffusion conversion must not erase base competence:

- HumanEval or MBPP slice: at least 70% of `AR_Q36`.
- IFEval or instruction-following slice: at least 75% of `AR_Q36`.
- GSM8K/math slice is optional, but track if cheap.

## Speed And Serving Metrics

Quality alone is not enough; diffusion needs a speed reason to exist.

Record:

- generated tokens/sec
- resolved SWE instances/hour
- denoising steps
- block size
- confidence threshold or entropy bound
- context length
- VRAM/memory use
- cache mode and GDN state-cache mode

Speed gate:

```text
DIFF_TRAIN_Q36 resolved instances/hour >= 1.3 * AR_Q36 resolved instances/hour
```

If the diffusion model is slower than AR at the same quality tier, it is still
useful research, but not a deployment closeout.

## Training Checkpoint Selection

Do not choose checkpoints by training loss alone.

Checkpoint sweep order:

1. strict synthetic one-call
2. public one-call
3. two-step tool traces
4. code-edit tiny harness
5. `SWEV-20`
6. `SWEV-50`
7. `SWEV-100`
8. full `SWEV-500`

Promote a checkpoint only if it improves at least two downstream metrics without
regressing strict JSON/tool-call validity by more than 2 percentage points.

## Concrete Closeout Statement

The project can be called closed for Qwen3.6 diffusion when we can write a result
table like this:

| Model | SWEV-500 resolved | SWEV-50 resolved | Tool JSON valid | Patch apply | Tokens/s | Resolved/hour |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `AR_Q36` | measured | measured | measured | measured | measured | measured |
| `DIFF_INIT_Q36` | measured | measured | measured | measured | measured | measured |
| `DIFF_TRAIN_Q36` | meets closeout formula | meets slice gate | meets gate | meets gate | measured | at least 1.3x AR |

Until that table exists, the experiment is still in progress.
