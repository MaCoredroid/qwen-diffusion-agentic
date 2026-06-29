# Qwen3.5 Public Multi-Call Focused Index-Sidecar Result

Date: 2026-06-28

## Purpose

Test whether the five remaining public multi-call target-candidate misses are
easier as an explicit candidate-index sidecar than as masked value-span ranking.

This trains on the public eval misses, so it is diagnostic only. It is not
promotion evidence.

## Focused Target

Input:

```text
data/candidate_ranking/public_multicall_targetcandidate_v5_miss_targets.jsonl
```

The target contains `5/5` usable argument-value examples:

- thermostat `schedule_time`: target `19:00`, model-ranked miss `11:00`
- fridge `start_time`: target `23:00`, model-ranked miss `22:00`
- finance `invoice_data[1].client_id`: target `CLI-102`, miss `CLI-103`
- finance `invoice_data[1].invoice_id`: target `INV-302`, miss `INV-301`
- finance `invoice_data[2].client_id`: target `CLI-103`, miss `CLI-101`

## Curriculum

Builder:

```text
scripts/build_candidate_ranking_curriculum.py
```

Output:

```text
data/qwen35_9b_public_multicall_v5_focused_miss_index_diag_curriculum
```

Settings:

- answer mode: `index`
- repeats: `24` per example
- accepted rows: `120`
- rejected rows: `0`
- block size audit: `1024`
- chosen length: min `998`, p50 `1708`, max `1708`
- kept labels: min/p50/max `3`
- contains eval slice: `true`
- diagnostic only: `true`
- promotion allowed: `false`

## Direct Index Rank Gate

Evaluator:

```text
scripts/eval_fastdllm_candidate_index_ranking.py
```

This appends mask slots after the prompt and scores the candidate index tokens.

| adapter | train setting | correct | p50 margin |
| --- | --- | ---: | ---: |
| checkpoint-275 | baseline | `2/5` | `-3.625` |
| step-5 plain | 1024 block, grouped, `1e-6` | `2/5` | `-3.5` |
| step-10 plain | 1024 block, grouped, `1e-6` | `2/5` | `-3.625` |
| step-10 b1536 | 1536 block, per-example, `1e-5` | `2/5` | `-3.625` |
| step-20 b1536 | 1536 block, per-example, `1e-5` | `2/5` | `-3.625` |

The two time fields are correct at baseline. The three finance table fields
remain wrong and continue to prefer earlier nearby candidate IDs:

- `CLI-102` target index `4` predicts `CLI-101` index `2`
- `INV-302` target index `3` predicts `INV-301` index `1`
- `CLI-103` target index `6` predicts `CLI-101` index `2`

## Training Runs

Conservative run:

```text
runs/fastdllm_qwen35_9b_public_multicall_v5_focused_miss_index_diag_from_ckpt275_step10_plain
```

- start adapter: checkpoint-275 protection adapter
- max steps: `10`
- max train samples: `64`
- block size: `1024`
- learning rate: `1e-6`
- train runtime: `99.4488` seconds
- train loss: `2.37245512008667`

Fuller 5090-fit run:

```text
runs/fastdllm_qwen35_9b_public_multicall_v5_focused_miss_index_diag_from_ckpt275_step20_b1536_lr1e5
```

- start adapter: checkpoint-275 protection adapter
- max steps: `20`
- max train samples: `120`
- `DISABLE_GROUP_TEXTS=1`
- block size: `1536`
- learning rate: `1e-5`
- train runtime: `312.2927` seconds
- train loss: `2.3265947818756105`

The attempted `2048` full-context run OOMed on the RTX 5090 during the first
backward pass. It had about `3.76 GiB` free and needed about `3.79 GiB`, so
`1536` is the current practical 9B QLoRA block-size ceiling for this setup.

## Generation Gate

New evaluator:

```text
scripts/eval_fastdllm_candidate_index_generation.py
```

This uses the same prompt but asks the diffusion sampler to generate a short
index answer.

| adapter | correct | in range |
| --- | ---: | ---: |
| checkpoint-275 | `0/5` | `0/5` |
| b1536 step-20 | `0/5` | `0/5` |

The generated text is not index-like (`<think>`, `You`, or repeated `#`
fragments), so generation is not currently a useful selector path for this
prompt. Direct masked ranking remains the stable sidecar diagnostic.

## Interpretation

The explicit numeric index sidecar is better than direct masked value-span
ranking at baseline (`2/5` versus `0/5`), but focused LoRA SFT on numeric index
answers does not yet move the hard finance-table decisions. The failure is not
tool syntax, JSON validity, or candidate coverage. It is row-local value
selection under plausible repeated table IDs.

Do not scale this exact numeric-index conversation SFT recipe. Next selector
experiments should remove or calibrate index-position priors:

- use pairwise candidate comparison instead of a single numeric index;
- randomize candidate order during training and evaluate order robustness;
- score table-row-local tuples such as `(invoice_id, client_id, amount)` rather
  than independent scalar values;
- add an explicit ranking loss over all candidate scores instead of ordinary
  chat SFT;
- use a small external sidecar classifier as an oracle/protected path, then
  distill only after the sidecar proves heldout movement.

