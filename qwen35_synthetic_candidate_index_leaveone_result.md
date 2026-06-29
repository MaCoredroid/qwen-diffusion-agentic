# Qwen3.5 Synthetic Candidate-Index Leave-One-Out Result

Date: 2026-06-28

## Purpose

Test whether explicit candidate-index selector training can generalize to the
remaining synthetic voice-command camera miss without training on that exact
row.

This is a diagnostic selector objective, not a promoted generator run.

## Split

Builder:

```text
scripts/build_synthetic_candidate_index_leaveone_curriculum.py
```

Source examples:

```text
data/candidate_ranking/synthetic_multicall_failure_analogue_index_ranking.jsonl
```

Curriculum:

```text
data/qwen35_9b_synthetic_candidate_index_leaveone_voice003_curriculum/train_agentic_mix.json
```

Heldout eval row:

```text
synthetic_voice_command_camera_003
```

Manifest:

- train examples: `7`
- heldout examples: `1`
- repeated training rows: `84`
- rejected rows: `0`
- holdout in training: `false`
- train source counts:
  - voice-command camera tool-name: `36`
  - security-code argument-value: `48`
- kept-label min/p50/p90/max: `3/3/3/3`
- promotion allowed: `false`

## Training

Source adapter:

```text
runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model
```

Output:

```text
runs/fastdllm_qwen35_9b_synthetic_candidate_index_leaveone_voice003_from_ckpt275_step30
```

Training ran under a systemd user scope with `MemoryMax=28G` and
`MemorySwapMax=4G`.

Training settings:

- max steps: `30`
- save steps: `10`
- block size: `1024`
- learning rate: `1e-5`
- train samples: `84`
- train runtime: `74.9995s`
- train loss: `1.013732647895813`

## Masked Candidate-Index Eval

Eval examples:

```text
data/candidate_ranking/synthetic_multicall_failure_analogue_index_ranking.jsonl
```

| adapter | overall | voice-command camera tool-name | security-code argument-value | min margin |
| --- | ---: | ---: | ---: | ---: |
| diffusion init | `6/8` | `2/4` | `4/4` | `-0.875` |
| checkpoint-275 | `7/8` | `3/4` | `4/4` | `-0.25` |
| leave-one checkpoint-10 | `8/8` | `4/4` | `4/4` | `0.0` |
| leave-one checkpoint-20 | `8/8` | `4/4` | `4/4` | `0.125` |
| leave-one checkpoint-30 | `8/8` | `4/4` | `4/4` | `0.125` |

Heldout row `synthetic_voice_command_camera_003`:

| adapter | predicted value | correct | target margin |
| --- | --- | ---: | ---: |
| diffusion init | `set_thermostat` | no | `-0.875` |
| checkpoint-275 | `set_thermostat` | no | `-0.25` |
| leave-one checkpoint-10 | `activate_voice_command` | yes | `0.0` |
| leave-one checkpoint-20 | `activate_voice_command` | yes | `0.125` |
| leave-one checkpoint-30 | `activate_voice_command` | yes | `0.125` |

Interpretation: the selector-style objective can move the missing tool-name
choice in a heldout synthetic analogue. This is a real model-side masked-ranking
signal, not a deterministic projector result.

## Generation Eval

Generation was evaluated with the same settings used for the prior synthetic
planner result:

- full-context sampling
- forced `<tool_call>\n` prefix
- max new tokens: `256`
- block size: `32`
- small block size: `8`
- constrained tool decoding
- sequence-preserving constrained projection
- constrained max calls: `3`

| adapter | raw sequence | raw arguments | raw valid JSON | constrained sequence | constrained arguments | constrained valid JSON |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| checkpoint-275 | `1/8` | `0/8` | `4/8` | `2/8` | `0/8` | `8/8` |
| leave-one checkpoint-10 | `0/8` | `0/8` | `2/8` | `2/8` | `0/8` | `8/8` |
| leave-one checkpoint-20 | `1/8` | `0/8` | `1/8` | `1/8` | `0/8` | `8/8` |

Interpretation: the selector objective improved masked candidate ranking but did
not improve full tool-call generation. Checkpoint-10 ties constrained sequence
but regresses raw sequence and raw JSON validity. Checkpoint-20 regresses
constrained sequence and raw JSON validity.

## Decision

Do not promote this adapter as a generator checkpoint.

Positive signal:

- A tiny train-only selector objective generalized to the heldout synthetic
  voice-command camera selector miss.

Negative signal:

- The same objective did not transfer to full block-diffusion generation.
- Directly training the main generator on short index answers can perturb
  tool-call generation quality.

Next implication:

- Treat candidate-index/value selection as a side objective or sidecar head, not
  as a standalone main-generator continuation.
- If folded into the generator, use a much lower weight and strong replay over
  full tool-call outputs, then promote only if generation gates move.
- The more promising path remains generation-time candidate-constrained
  decoding plus a learned selector/value head for ambiguous choices.

Follow-up:

```text
qwen35_synthetic_selector_replay_mix_result.md
```

The first replay-heavy mix also fails promotion: it loses the heldout masked
selector lift and regresses constrained generation. This strengthens the case
for keeping selector learning separate from the main generator adapter.
