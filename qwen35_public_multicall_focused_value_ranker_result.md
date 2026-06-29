# Qwen3.5 Public Multi-Call Focused Value-Ranker Result

Date: 2026-06-28

## Purpose

Test whether a tiny diagnostic value-span continuation can move the five stable
public multi-call candidate-ranking failures identified by the v5 sampler audit.

This trains on the public eval misses, so it is diagnostic only. It is not
promotion evidence.

## Focused Target

Examples:

```text
data/candidate_ranking/public_multicall_targetcandidate_v5_miss_targets.jsonl
data/candidate_ranking/public_multicall_targetcandidate_v5_miss_targets.train.json
```

The target contains `5/5` usable argument-value examples:

- thermostat `schedule_time`: target `19:00`, model-ranked miss `11:00`
- fridge `start_time`: target `23:00`, model-ranked miss `22:00`
- finance `invoice_data[1].client_id`: target `CLI-102`, miss `CLI-103`
- finance `invoice_data[1].invoice_id`: target `INV-302`, miss `INV-301`
- finance `invoice_data[2].client_id`: target `CLI-103`, miss `CLI-101`

Baseline checkpoint-275 masked candidate ranking:

| context mode | correct |
| --- | ---: |
| prefix-only | `0/5` |
| full-gold | `0/5` |

## Curriculum

Builder:

```text
scripts/build_candidate_ranking_curriculum.py
```

Output:

```text
data/qwen35_9b_public_multicall_v5_focused_miss_value_span_diag_curriculum
```

Settings:

- answer mode: `target_text`
- repeats: `24` per example
- accepted rows: `120`
- rejected rows: `0`
- block size: `1024`
- contains eval slice: `true`
- diagnostic only: `true`
- promotion allowed: `false`

The first attempt with `VALUE_SPAN_LABEL_ONLY=1` failed before training because
the helper that derives `FASTDLLM_VALUE_SPAN_TOKEN_IDS` expects tool-call JSON
fragments, while this curriculum's assistant answer is only the value span.

## Training

Plain value-span conversation SFT:

```text
runs/fastdllm_qwen35_9b_public_multicall_v5_focused_miss_valuespan_diag_from_ckpt275_step10_plain
```

Start adapter:

```text
runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model
```

Training:

- max steps: `10`
- save steps: `5`
- max train samples: `64`
- learning rate: `1e-6`
- train loss: `2.4058822631835937`
- checkpoint-5 and checkpoint-10 saved

## Focused Gate

Prefix-only masked candidate ranking:

| adapter | correct | p50 margin |
| --- | ---: | ---: |
| checkpoint-275 | `0/5` | `-0.875` |
| focused step-5 | `0/5` | `-1.0625` |
| focused step-10 | `0/5` | `-0.875` |

Row-level behavior stays essentially unchanged:

- `19:00` still ranks behind `11:00`
- `23:00` still ranks behind `22:00`
- `INV-302` still ranks behind `INV-301`
- `CLI-102` / `CLI-103` row alignment still favors nearby wrong row values

## Interpretation

The focused examples are good diagnostic pressure, but plain conversation
value-span SFT does not move the masked candidate scorer, even when trained on
the exact five failed public spans.

The next route should be one of:

- a separate candidate-index/value sidecar evaluated as a direct classifier;
- an explicit pairwise/ranking loss over candidate scores;
- sampler-side row-local candidate grouping for tables, instead of independent
  scalar scoring;
- public-train analogue generation for the same five failure families, followed
  by heldout evaluation on this `0/5` focused target.

Do not scale this exact plain value-span SFT recipe.
