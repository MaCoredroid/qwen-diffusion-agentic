# Qwen3.5 Sequence/Value/Retention Mix Result

Date: 2026-06-28

## Purpose

Build the next train-only branch after the balanced and planner-heavy mixes
failed promotion on the heldout policy gate.

The previous planner-heavy branch showed one useful signal: constrained exact
arguments can move above `0/12`, but sequence retention regressed. This branch
therefore removes low-label pairwise selector rows and combines:

- sequence-planner rows for missing-call/tool-order pressure;
- explicit candidate value-span rows for argument grounding;
- route-delta retention rows for one-call and tool-result behavior.

This is currently a corpus/provenance/trainability result, not a model-quality
result.

## Builder

Script:

```text
scripts/build_qwen35_sequence_value_retention_mix.py
```

The builder consumes three existing train-only dataset directories:

- retention:
  `data/qwen35_9b_route_delta_trainonly_mix_curriculum`
- value:
  `data/qwen35_9b_candidate_value_span_public_train_curriculum`
- planner:
  `data/qwen35_9b_toolcall_sequence_planner_distill_no_public_multicall_smoke_curriculum`

For retention, it excludes the embedded `public_train_value_span` rows so the
value objective is represented explicitly by the value-span source rather than
silently duplicated inside route-delta retention.

Retention sources kept:

- `fastdllm_toolcall_train`
- `synthetic_onecall_train`
- `synthetic_toolresult_text_train`
- `synthetic_toolresult_openai_train`

## Corpus

Output:

```text
data/qwen35_9b_sequence_value_retention_mix_nooverlap_curriculum
```

Build settings:

- retention cap: all eligible rows
- value cap: all rows
- planner repeat: `4`
- block size: `1536`
- truncation side: `left`
- require full labels: `true`

Manifest:

```text
data/qwen35_9b_sequence_value_retention_mix_nooverlap_curriculum/train_agentic_mix.manifest
```

Composition:

- total accepted rows: `387`
- route-delta retention: `154`
- candidate value-span: `173`
- sequence planner: `60`
- removed eval overlaps: `8`, all from retention
- rejected rows after token-label audit: `0`
- zero-label / partial-label rows after truncation: `0` / `0`

Label audit:

- sequence length: min `405`, p50 `840`, p90 `1315`, max `1726`
- full labels: min `3`, p50 `30`, p90 `122`, max `440`
- kept labels match full labels for all accepted rows

## Overlap Audit

Independent overlap audit:

```text
runs/sequence_value_retention_mix_nooverlap_audit/train_vs_public_synthetic_heldout.json
```

Eval files checked:

- `data/toolcall_eval/public_multicall_hermes_smoke.jsonl`
- `data/toolcall_eval/synthetic_multicall_failure_analogues.jsonl`
- `data/toolcall_eval/heldout_seed_multicall_2to3_clean.jsonl`
- `runs/planner_decomposition/heldout_seed_multicall_policy_targets.jsonl`

Result:

- train records: `387`
- eval records: `45`
- exact overlap count: `0`
- user-prompt overlap count: `0`

## One-Step Gate

Run:

```text
runs/fastdllm_qwen35_9b_sequence_value_retention_mix_nooverlap_from_ckpt275_step1_gate
```

Start adapter:

```text
runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model
```

Settings:

- max steps: `1`
- max train samples: `32`
- block size: `1536`
- truncation side: `left`
- learning rate: `1e-6`
- gradient accumulation: `4`
- LoRA: `r=8`, `alpha=16`
- target modules:
  `q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,out_proj,in_proj_a,in_proj_b`
- `VALUE_SPAN_LABEL_ONLY=1`
- systemd user scope:
  `MemoryMax=28G`, `MemorySwapMax=4G`

Training result:

- checkpoint saved
- train loss: `1.2485730648040771`
- train runtime: `16.2236s`
- train samples/sec: `0.247`

## Short Heldout Sweep

Training run:

```text
runs/fastdllm_qwen35_9b_sequence_value_retention_mix_nooverlap_from_ckpt275_step10
```

Training settings:

- max steps: `10`
- saved checkpoints: `5`, `10`
- max train samples: `387`
- block size: `1536`
- truncation side: `left`
- learning rate: `1e-6`
- gradient accumulation: `4`
- LoRA: `r=8`, `alpha=16`
- `VALUE_SPAN_LABEL_ONLY=1`

Training result:

- train loss: `0.5634021759033203`
- train runtime: `156.1189s`
- train samples/sec: `0.256`
- observed training VRAM during run: about `30.5 GiB`

Heldout eval target:

```text
runs/planner_decomposition/heldout_seed_multicall_policy_targets.jsonl
```

Eval settings:

- full-context sampling
- forced `<tool_call>\n` prefix
- max new tokens: `900`
- block size: `32`
- small block size: `8`
- constrained tool decoding
- sequence-preserving constrained projection
- constrained max calls: `3`
- no protected sampler schedule
- no selector/sidecar injection

Outputs:

```text
runs/fastdllm_qwen35_9b_sequence_value_retention_mix_nooverlap_from_ckpt275_step10/checkpoint5_policy_targets_forcedprefix.summary.json
runs/fastdllm_qwen35_9b_sequence_value_retention_mix_nooverlap_from_ckpt275_step10/checkpoint10_policy_targets_forcedprefix.summary.json
```

Comparison:

| run | raw valid | raw exact seq | raw exact args | constrained name set | constrained seq | constrained args | extra calls | missing calls | repeated calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| checkpoint-275 baseline | `0/12` | `0/12` | `0/12` | `6/12` | `5/12` | `0/12` | `7` | `22` | `7` |
| sequence/value checkpoint-5 | `1/12` | `0/12` | `0/12` | `7/12` | `5/12` | `0/12` | `3` | `23` | `2` |
| sequence/value checkpoint-10 | `1/12` | `0/12` | `0/12` | `6/12` | `4/12` | `0/12` | `1` | `24` | `1` |

Row-level notes:

- checkpoint-5 keeps the aggregate constrained exact sequence at checkpoint-275's
  `5/12`, but it swaps which rows pass.
- checkpoint-5 is the first clean branch in this line to move raw valid JSON to
  `1/12` without lowering the aggregate constrained sequence score.
- checkpoint-10 keeps raw valid JSON at `1/12` but regresses constrained exact
  sequence to `4/12`.
- neither checkpoint moves constrained exact arguments above `0/12`.

Eval runtime:

- checkpoint-5: `1065.99s`, `6189` generated tokens, `5.81` tokens/sec
- checkpoint-10: `1014.39s`, `5776` generated tokens, `5.69` tokens/sec

## Interpretation

This branch is cleaner than the prior balanced/planner-heavy mixes for the
specific failure we saw:

- it keeps explicit sequence-planner pressure;
- it keeps the value-span objective that previously moved public heldout
  argument ranking at checkpoint-5;
- it removes pairwise selector rows from the main generator mix;
- it keeps retention rows for one-call and tool-result behavior;
- it has a clean eval-overlap audit.

It is not a promoted model. Checkpoint-5 is the best local signal from this
branch because it ties checkpoint-275 constrained sequence while adding one raw
valid JSON row. It still fails the promotion bar because exact arguments remain
`0/12`, and checkpoint-10 regresses sequence.

Next useful experiment: avoid another broad data mix. Either compose/check
adapter deltas from the isolated positive signals, or build a surgical value
objective that targets argument grounding while explicitly preserving the
checkpoint-275 constrained-sequence hit rows.
