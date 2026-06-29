# Qwen3.5 9B Skeleton Value-Infill Checkpoint Eval

Date: 2026-06-28

## Question

Did the skeleton-conditioned value-infill continuation from checkpoint-275
produce a promotion-worthy adapter at checkpoints `25`, `50`, or `75`?

Short answer: no. The adapter line is trainable and shows a small one-call raw
signal at checkpoint-25, but all three checkpoints tie the active checkpoint-275
on the public and heldout multi-call guard gates. They do not fix the public
voice-command value miss or the heldout nested JSON skeleton failure.

## Inputs

Training result:

```text
qwen35_skeleton_value_infill_training_gate_result.md
runs/fastdllm_qwen35_9b_skeleton_value_infill_from_ckpt275_step75
```

Evaluated checkpoint adapters:

```text
runs/fastdllm_qwen35_9b_skeleton_value_infill_from_ckpt275_step75/checkpoint-25/adapter_model
runs/fastdllm_qwen35_9b_skeleton_value_infill_from_ckpt275_step75/checkpoint-50/adapter_model
runs/fastdllm_qwen35_9b_skeleton_value_infill_from_ckpt275_step75/checkpoint-75/adapter_model
```

Active baseline adapter:

```text
runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model
```

## Public Multi-Call Closeguard

Output directory:

```text
runs/fastdllm_qwen35_9b_skeleton_value_infill_from_ckpt275_step75_public_closeguard_eval
```

Comparable baseline:

```text
runs/tool_sensitive_block_plans/public_multicall_mode_prefix_name_value_closeguard_12.summary.json
```

Settings:

- cases: `data/toolcall_eval/public_multicall_hermes_smoke.jsonl`
- schedule:
  `runs/tool_sensitive_block_plans/public_multicall_gold_sampler_schedule_with_pairwise_argsketch_choices_12.jsonl`
- guards: mode, JSON-prefix, tool-name candidates, value candidates, close-tag completeness
- stop: `--stop-after-schedule-tool-calls`
- no model-repair pass

| Adapter | Valid JSON | Exact Sequence | Exact Args | Constrained Args | Extra / Missing / Repeated | Close Deferrals | Prefix Rejects / Unsafe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| active checkpoint-275 | 12/12 | 12/12 | 11/12 | 8/12 | 0 / 0 / 1 | 1 | 0 / 0 |
| value-infill checkpoint-25 | 12/12 | 12/12 | 11/12 | 8/12 | 0 / 0 / 1 | 1 | 0 / 0 |
| value-infill checkpoint-50 | 12/12 | 12/12 | 11/12 | 8/12 | 0 / 0 / 1 | 1 | 0 / 0 |
| value-infill checkpoint-75 | 12/12 | 12/12 | 11/12 | 8/12 | 0 / 0 / 1 | 1 | 0 / 0 |

All three checkpoints miss the same raw public row:

```text
c483f963-8a29-4ff0-a684-89be0d0f2843
```

The tool sequence is correct; the remaining failure is still the third
`activate_voice_command` argument value (`location: ""` vs the model's
plausible `location: "home"`).

Completability diagnostic:

```text
runs/fastdllm_qwen35_9b_skeleton_value_infill_from_ckpt275_step75_public_closeguard_eval/completability.json
```

Result for every checkpoint:

- raw assistant: `31/31` complete JSON segments, `0` invalid segments
- constrained assistant: `31/31` complete JSON segments, `0` invalid segments
- raw exact sequence: `12/12`
- raw exact arguments: `11/12`

## Heldout Policy Lean Closeguard

Output directory:

```text
runs/fastdllm_qwen35_9b_skeleton_value_infill_from_ckpt275_step75_heldout_lean_closeguard_eval
```

Comparable baseline:

```text
runs/tool_sensitive_block_plans/heldout_seed_policy_evidence_selector/derived_pairwise_mode_prefix_name_value_closeguard_ckpt275_generation.summary.json
```

Settings:

- cases: `runs/planner_decomposition/heldout_seed_multicall_policy_targets.jsonl`
- schedule:
  `runs/tool_sensitive_block_plans/heldout_seed_policy_evidence_selector/sampler_schedule_with_derived_pairwise_choices.jsonl`
- guards: mode, JSON-prefix, tool-name candidates, value candidates, close-tag completeness
- no oracle `json_key,json_structure` forcing
- no model-repair pass

| Adapter | Valid JSON | Exact Sequence | Exact Args | Constrained Args | Extra / Missing / Repeated | Close Deferrals | Prefix Rejects / Unsafe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| active checkpoint-275 | 11/12 | 11/12 | 11/12 | 2/12 | 0 / 1 / 0 | 6 | 83 / 83 |
| value-infill checkpoint-25 | 11/12 | 11/12 | 11/12 | 2/12 | 0 / 1 / 0 | 6 | 83 / 83 |
| value-infill checkpoint-50 | 11/12 | 11/12 | 11/12 | 2/12 | 0 / 1 / 0 | 6 | 83 / 83 |
| value-infill checkpoint-75 | 11/12 | 11/12 | 11/12 | 2/12 | 0 / 1 / 0 | 6 | 83 / 83 |

Completability diagnostic:

```text
runs/fastdllm_qwen35_9b_skeleton_value_infill_from_ckpt275_step75_heldout_lean_closeguard_eval/completability.json
```

Result for every checkpoint:

- raw assistant: `27/29` complete JSON segments, `2` invalid segments
- invalid row: `heldout_seed_multicall_0004`
- failure class: unrecoverable nested JSON skeleton/key corruption
- constrained assistant: `29/29` complete JSON segments, but only `2/12`
  exact arguments

This exactly matches the prior lean heldout failure class. The value-infill
continuation did not teach the missing nested skeleton behavior.

## One-Call Sweep

Output directory:

```text
runs/fastdllm_qwen35_9b_skeleton_value_infill_from_ckpt275_step75_onecall_checkpoint_sweep_eval96_modelrepair_max1
```

Summary:

```text
runs/fastdllm_qwen35_9b_skeleton_value_infill_from_ckpt275_step75_onecall_checkpoint_sweep_eval96_modelrepair_max1/checkpoint_sweep_summary.tsv
```

| Checkpoint | Slice | Raw Valid | Raw Seq | Raw Args | Constrained Args | Model-Repair Seq | Model-Repair Args |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | public one-call 8 | 3/8 | 4/8 | 3/8 | 8/8 | 5/8 | 3/8 |
| 50 | public one-call 8 | 1/8 | 2/8 | 1/8 | 8/8 | 3/8 | 1/8 |
| 75 | public one-call 8 | 3/8 | 3/8 | 2/8 | 8/8 | 4/8 | 2/8 |
| 25 | teacher train 12 | 3/12 | 2/12 | 2/12 | 5/12 | 5/12 | 4/12 |
| 50 | teacher train 12 | 1/12 | 1/12 | 1/12 | 5/12 | 4/12 | 3/12 |
| 75 | teacher train 12 | 3/12 | 2/12 | 2/12 | 6/12 | 5/12 | 2/12 |
| 25 | teacher heldout 8 | 2/8 | 2/8 | 1/8 | 6/8 | 4/8 | 2/8 |
| 50 | teacher heldout 8 | 2/8 | 1/8 | 0/8 | 5/8 | 3/8 | 1/8 |
| 75 | teacher heldout 8 | 2/8 | 1/8 | 0/8 | 6/8 | 3/8 | 1/8 |

Checkpoint-25 has the best one-call raw/model-repair signal, including public
one-call raw arguments `3/8` versus active checkpoint-275's `2/8` raw public
one-call argument score. This is not enough to promote because multi-call
agentic gates do not move.

## Decision

Do not promote checkpoints `25`, `50`, or `75` from this value-infill line.

What the experiment proved:

- the clean skeleton-conditioned value-infill corpus is trainable;
- the public closeguard structure remains stable after the continuation;
- checkpoint-25 shows a small one-call raw/model-repair improvement.

What it did not prove:

- no checkpoint improves public multi-call exact arguments beyond `11/12`;
- no checkpoint fixes the public empty-location value miss;
- no checkpoint improves the heldout lean closeguard ceiling beyond `11/12`;
- no checkpoint fixes the heldout nested skeleton/key corruption.

## Next Direction

Do not scale this standalone fixed-skeleton value-answer objective by itself.
The next useful adapter should train a joint objective over:

1. skeleton/key/structure prediction or acceptance;
2. value candidates under the same schedule state used at inference;
3. boundary/close behavior;
4. retention against the active checkpoint-275 public multi-call behavior.

The fastest next diagnostic is a selector-style objective that scores candidate
value and skeleton choices in the actual scheduled sampler state, not a
standalone assistant answer that returns only the JSON value.

Follow-up materialized:

```text
qwen35_schedule_state_selector_curriculum_gate_result.md
```

The new builder converts clean skeleton value slots into a schedule-state
selector/policy curriculum and a one-step checkpoint-275 QLoRA gate trains and
saves successfully. This is the next line to sweep and evaluate.
