# Qwen3.5 Heldout Policy Planner-Distill Diagnostic Result

Date: 2026-06-28

## Purpose

Test a different model-side pressure from the failed derived-pairwise SFT:
directly teacher-force the heldout planner-policy tool-call outputs, then
evaluate whether raw or constrained generation moves on the same 12 policy
targets.

This is diagnostic only. The training rows come from the heldout policy-target
slice, so the resulting checkpoint is non-promotable even if it improves.

## Training Data

Corpus:

```text
data/qwen35_9b_heldout_policy_planner_distill_diagnostic_curriculum
```

Source:

```text
runs/planner_decomposition/heldout_seed_multicall_policy_targets.jsonl
```

Target field:

```text
policy_planner_assistant
```

Manifest highlights:

- raw policy cases: `12`
- exact policy targets: `12/12`
- raw candidates: `24` from full + compact schemas
- accepted rows: `15`
- label rejected: `9`
- block size: `1024`
- accepted rows with zero/partial labels: `0/0`
- diagnostic only: `true`
- contains eval slice: `true`
- promotion allowed: `false`

## Training Run

Start adapter:

```text
runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model
```

Output:

```text
runs/fastdllm_qwen35_9b_heldout_policy_planner_from_ckpt275_step25_diag
```

Settings:

- max steps: `25`
- max train samples: `15`
- block size: `1024`
- truncation side: `right`
- `DISABLE_GROUP_TEXTS=1`
- learning rate: `5e-6`
- gradient accumulation: `4`
- argument-span loss weight: `1.5`
- LoRA: `r=8`, `alpha=16`
- target modules:
  `q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,out_proj,in_proj_a,in_proj_b`
- systemd user scope:
  `MemoryMax=28G`, `MemorySwapMax=4G`

Training result:

- checkpoint-25 saved
- train loss: `2.863811378479004`
- train runtime: `233.9705s`
- train samples/sec: `0.427`
- max observed GPU memory during training: about `26.3 GiB`

## Generation Eval

Eval target:

```text
runs/planner_decomposition/heldout_seed_multicall_policy_targets.jsonl
```

Both baseline and checkpoint-25 were evaluated with:

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
runs/fastdllm_qwen35_9b_heldout_policy_planner_from_ckpt275_step25_diag/ckpt275_baseline_policy_targets_forcedprefix.summary.json
runs/fastdllm_qwen35_9b_heldout_policy_planner_from_ckpt275_step25_diag/checkpoint25_policy_targets_forcedprefix.summary.json
```

Results:

| run | raw valid | raw exact seq | raw exact args | constrained seq | constrained args |
| --- | ---: | ---: | ---: | ---: | ---: |
| checkpoint-275 baseline | `0/12` | `0/12` | `0/12` | `5/12` | `0/12` |
| policy-planner checkpoint-25 | `1/12` | `0/12` | `0/12` | `6/12` | `0/12` |

Other deltas:

- raw schema-valid: `1/12 -> 3/12`
- raw required args present: `2/12 -> 4/12`
- total extra calls: `7 -> 2`
- total repeated calls: `7 -> 1`
- total missing calls: `22 -> 23`
- generated tokens/sec: `5.71 -> 5.87`

Row-level changed records: `6/12`.

Notable changes:

- `heldout_seed_multicall_0003` becomes raw valid JSON and removes the
  repeated/extra IoT add-device calls, but still misses target calls.
- `heldout_seed_multicall_0007` gains constrained exact sequence.
- `heldout_seed_multicall_0012` gains constrained exact tool-name set and
  constrained exact sequence.
- `heldout_seed_multicall_0011` regresses constrained exact sequence.

## Interpretation

This is weak but real model-side movement:

- direct planner-policy SFT changes raw generation behavior, unlike the focused
  derived-pairwise SFT where `0/152` selector predictions changed;
- it improves raw JSON validity from `0/12` to `1/12`;
- it improves constrained exact sequence from `5/12` to `6/12`;
- it reduces extra/repeated raw calls.

It is not a successful planner model:

- raw exact sequence remains `0/12`;
- exact arguments remain `0/12` in raw and constrained modes;
- one constrained sequence row regresses;
- the training/eval slice is heldout-derived and non-promotable.

Next implication: full-output planner-policy distillation has a stronger
effect than pairwise derived-rule SFT, but it is still too weak and unstable
alone. The next model-side recipe should mix planner-policy rows with retention
and selector/value supervision, then gate on separate heldout slices. For this
heldout policy route, keep using the protected sampler/selector sidecar as the
oracle ceiling while treating planner-policy SFT as a weak direction to combine
with anti-regression data, not a standalone fix.
