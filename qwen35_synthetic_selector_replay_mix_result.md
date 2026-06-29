# Qwen3.5 Synthetic Selector-Replay Mix Result

Date: 2026-06-28

## Purpose

Follow up the leave-one-out selector result by mixing a small amount of
candidate-index supervision with full tool-call replay.

Hypothesis:

```text
Low-ratio selector rows plus full-call replay might keep the heldout selector
lift without damaging full tool-call generation.
```

Result: negative.

## Curriculum

Builder:

```text
scripts/build_synthetic_selector_replay_mix.py
```

Output:

```text
data/qwen35_9b_synthetic_selector_replay_mix_leaveone_voice003_curriculum/train_agentic_mix.json
```

Heldout:

```text
synthetic_voice_command_camera_003
```

Manifest:

- rows: `98`
- rejected rows: `0`
- planner replay rows: `84`
- selector-index rows: `14`
- planner train base rows: `21`
- planner holdout base rows: `3`
- selector train examples: `7`
- selector holdout examples: `1`
- holdout in training: `false`
- kept-label min/p50/p90/max: `3/108/120/120`

## Training

Source adapter:

```text
runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model
```

Output:

```text
runs/fastdllm_qwen35_9b_synthetic_selector_replay_mix_leaveone_voice003_from_ckpt275_step20
```

Training ran under a systemd user scope with `MemoryMax=28G` and
`MemorySwapMax=4G`.

Training settings:

- max steps: `20`
- save steps: `10`
- block size: `1024`
- learning rate: `1e-5`
- train samples: `98`
- train runtime: `50.1937s`
- train loss: `2.079222249984741`

## Masked Candidate-Index Eval

| adapter | overall | voice-command camera tool-name | security-code argument-value | min margin |
| --- | ---: | ---: | ---: | ---: |
| checkpoint-275 | `7/8` | `3/4` | `4/4` | `-0.25` |
| selector-only checkpoint-20 | `8/8` | `4/4` | `4/4` | `0.125` |
| replay-mix checkpoint-10 | `7/8` | `3/4` | `4/4` | `-0.375` |
| replay-mix checkpoint-20 | `7/8` | `3/4` | `4/4` | `-0.5` |

The replay-heavy mix does not retain the selector-only heldout lift. It falls
back to the checkpoint-275 top line and worsens the minimum margin.

## Generation Eval

Generation settings match the prior synthetic analogue gates:

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
| selector-only checkpoint-10 | `0/8` | `0/8` | `2/8` | `2/8` | `0/8` | `8/8` |
| selector-only checkpoint-20 | `1/8` | `0/8` | `1/8` | `1/8` | `0/8` | `8/8` |
| replay-mix checkpoint-10 | `1/8` | `0/8` | `1/8` | `0/8` | `0/8` | `8/8` |

The replay mix does not promote. It ties raw exact sequence but regresses raw
JSON validity and constrained sequence.

## Decision

Do not promote this mixed adapter.

Interpretation:

- Selector-only SFT can move a masked selector preference, but does not transfer
  safely into full block-diffusion generation.
- A simple replay-heavy blend dilutes the selector signal and still perturbs
  generation.
- The next useful implementation should separate the selector from the main
  generator path: use it as a sidecar scorer or value/head objective, then
  integrate it into generation-time candidate-constrained decoding instead of
  asking ordinary SFT to make short index answers and full tool-call chains
  coexist in one small adapter.

Follow-up:

```text
qwen35_selector_sidecar_projection_result.md
```

Using the selector-only checkpoint as a separate sidecar, rather than merging it
into the generator, fixes the heldout synthetic projection path: bad drafts move
from `4/8` exact sequence and `0/8` exact arguments to `8/8` and `8/8`. This is
a protected sidecar result, not raw generator promotion.
