# Qwen3.5 Public-Train Pairwise Phrase SFT Result

Date: 2026-06-28

## Purpose

Test whether the promotion-eligible train-only path-aware phrase pairwise
curriculum can be distilled into the current Qwen3.5-9B diffusion adapter
without regressing the heldout selector gate.

This is a selector-mode SFT test. It is not yet raw generation/model promotion.

## Training Data

Curriculum:

```text
data/qwen35_9b_public_train_pairwise_pathaware_phrase_curriculum
```

Manifest:

- accepted rows: `376`
- rejected labels: `0`
- examples source:
  `data/candidate_ranking/public_train_multicall_toolname_argument_ranking_pathaware_phrase.jsonl`
- contains eval slice: `false`
- diagnostic only: `false`
- promotion allowed by provenance: `true`
- block size: `1536`
- truncation side: `left`
- no zero-label or partial-label truncation

## Training Run

Output:

```text
runs/fastdllm_qwen35_9b_public_train_pairwise_pathaware_phrase_from_ckpt275_step10_lr1e6
```

Start adapter:

```text
runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model
```

Settings:

- max steps: `10`
- save steps: `5`
- max train samples: `80`
- block size: `1536`
- truncation side: `left`
- `DISABLE_GROUP_TEXTS=1`
- learning rate: `1e-6`
- gradient accumulation: `4`
- LoRA: `r=8`, `alpha=16`
- target modules:
  `q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,out_proj,in_proj_a,in_proj_b`

Training result:

- checkpoint-5 and checkpoint-10 saved
- train runtime: `156.2s`
- train loss: `4.9155`
- logged step-5 loss: `4.8952`
- logged step-10 loss: `4.9358`

## Heldout Selector Gate

Heldout examples:

```text
data/candidate_ranking/public_multicall_toolname_argument_ranking_pathaware_phrase_12.jsonl
```

Baseline:

```text
runs/candidate_ranking/public_multicall_pathaware_phrase12_ckpt275_pairwise_tournament.summary.json
```

Trained checkpoints:

```text
runs/candidate_ranking/public_multicall_pathaware_phrase12_public_train_pairwise_ckpt5_tournament.summary.json
runs/candidate_ranking/public_multicall_pathaware_phrase12_public_train_pairwise_ckpt10_tournament.summary.json
```

Results:

| run | overall | argument values | tool names |
| --- | ---: | ---: | ---: |
| checkpoint-275 | `98/99` | `68/68` | `30/31` |
| pairwise phrase checkpoint-5 | `98/99` | `68/68` | `30/31` |
| pairwise phrase checkpoint-10 | `98/99` | `68/68` | `30/31` |

Row-level comparison:

- checkpoint-5 changes `0/99` heldout predictions versus checkpoint-275
- checkpoint-10 changes `0/99` heldout predictions versus checkpoint-275
- the single miss remains the tool-name row:
  `set_thermostat` predicted as `activate_security_cameras`

Because the argument-value predictions are identical and already correct, the
existing phrase-aware injected schedule remains the appropriate protected
sampler evidence:

```text
runs/tool_sensitive_block_plans/public_multicall_pathaware_phrase12_argselector_structguard_ckpt275_generation.summary.json
```

That gate is still `12/12` exact sequence, `12/12` exact arguments, and `12/12`
valid JSON.

## Interpretation

The train-only pairwise phrase SFT is a no-regression result, not a promotion:

- It proves the 1536-token train-only pairwise curriculum can be trained from
  checkpoint-275 on the RTX 5090 without OOM.
- It preserves the heldout selector gate exactly.
- It does not improve margins or fix the heldout tool-name selector miss.
- It does not move raw or constrained decoder behavior.

Do not replace checkpoint-275 with this checkpoint. The next model-side step
should target either:

- a tool-name selector curriculum, because the only heldout selector miss is a
  tool-name row;
- a direct selected-value span / copy objective if we want raw argument
  generation movement instead of sidecar ranking;
- a larger synthetic/teacher heldout selector gate where the argument selector
  is not already saturated.

## Tool-Name Follow-Up

Follow-up result:

```text
qwen35_public_train_toolname_pairwise_sft_result.md
```

A separate train-only tool-name pairwise curriculum was built and trained for
10 steps from checkpoint-275. It also preserves the heldout gate exactly but
does not fix the remaining tool-name miss:

- checkpoint-275: `98/99` overall, `68/68` argument values, `30/31` tool names
- tool-name checkpoint-5: `98/99` overall, `68/68` argument values, `30/31`
  tool names
- tool-name checkpoint-10: `98/99` overall, `68/68` argument values, `30/31`
  tool names

Row-level predictions changed on `0/99` heldout rows. This points to prompt
under-specification rather than needing more of the same SFT.
