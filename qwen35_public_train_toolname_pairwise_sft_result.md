# Qwen3.5 Public-Train Tool-Name Pairwise SFT Result

Date: 2026-06-28

## Purpose

Test whether a train-only tool-name pairwise selector curriculum can fix the
single heldout tool-name selector miss left after the path-aware phrase
argument selector gate.

This is a selector-mode SFT test. It is not raw generation/model promotion.

## Known Heldout Miss

Baseline checkpoint-275 heldout phrase tournament:

```text
runs/candidate_ranking/public_multicall_pathaware_phrase12_ckpt275_pairwise_tournament.jsonl
```

Single miss:

- case: `c483f963-8a29-4ff0-a684-89be0d0f2843`
- kind: `tool_name`
- tool call index: `1`
- target: `set_thermostat`
- predicted: `activate_security_cameras`
- candidates: `set_thermostat`, `activate_security_cameras`

The prompt contains the user request and call index, but not a same-call
argument sketch. That makes this row a good test for whether more tool-name SFT
is enough, or whether the selector prompt needs richer local call context.

## Training Data

Curriculum:

```text
data/qwen35_9b_public_train_pairwise_toolname_pathaware_phrase_curriculum
```

Manifest:

- accepted rows: `240`
- rejected labels: `0`
- examples source:
  `data/candidate_ranking/public_train_multicall_toolname_argument_ranking_pathaware_phrase.jsonl`
- include kinds: `tool_name`
- contains eval slice: `false`
- diagnostic only: `false`
- promotion allowed by provenance: `true`
- block size: `1536`
- truncation side: `left`
- no zero-label or partial-label truncation

## Training Run

Output:

```text
runs/fastdllm_qwen35_9b_public_train_pairwise_toolname_pathaware_phrase_from_ckpt275_step10_lr1e6
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
- train runtime: `156.3s`
- train loss: `5.0694`
- logged step-5 loss: `5.0573`
- logged step-10 loss: `5.0815`

## Heldout Selector Gate

Heldout examples:

```text
data/candidate_ranking/public_multicall_toolname_argument_ranking_pathaware_phrase_12.jsonl
```

Trained checkpoint summaries:

```text
runs/candidate_ranking/public_multicall_pathaware_phrase12_public_train_toolname_pairwise_ckpt5_tournament.summary.json
runs/candidate_ranking/public_multicall_pathaware_phrase12_public_train_toolname_pairwise_ckpt10_tournament.summary.json
```

Results:

| run | overall | argument values | tool names |
| --- | ---: | ---: | ---: |
| checkpoint-275 | `98/99` | `68/68` | `30/31` |
| tool-name checkpoint-5 | `98/99` | `68/68` | `30/31` |
| tool-name checkpoint-10 | `98/99` | `68/68` | `30/31` |

Row-level comparison:

- checkpoint-5 changes `0/99` heldout predictions versus checkpoint-275
- checkpoint-10 changes `0/99` heldout predictions versus checkpoint-275
- the heldout tool-name miss remains unchanged

## Interpretation

This is a no-regression training smoke, not a promotion. More SFT on the current
tool-name pairwise prompt did not fix the heldout miss.

The likely bottleneck is prompt/context design: tool-name selection needs
same-call evidence, such as planned argument keys/values, a sequence-plan sketch,
or a boundary policy that delays tool-name commitment until enough argument
context is available. The existing protected sampler can still reach `12/12`
tool sequence because it uses schedule-level tool-name sequence constraints,
but a standalone pairwise selector prompt with only call index is under-specified
for this row.

Next candidate direction:

- build a path-aware tool-name selector prompt that includes same-call argument
  keys and available evidence snippets;
- evaluate that prompt zero-shot on heldout before training;
- only then train a new train-only tool-name curriculum if the prompt gives a
  measurable heldout lift.
