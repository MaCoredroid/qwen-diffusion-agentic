# Qwen3.5 Selector Sidecar Projection Result

Date: 2026-06-28

## Purpose

Test the architecture implied by the previous negative SFT results:

```text
Keep the main diffusion generator at checkpoint-275, and use a separately
trained selector adapter only as a sidecar scorer for ambiguous tool/value
choices.
```

This is a protected-path result. It is not raw generator promotion.

## Implementation

Script:

```text
scripts/apply_synthetic_selector_sidecar_projection.py
```

The script consumes:

- synthetic cases,
- a selector-ranking JSONL produced by `eval_fastdllm_candidate_index_ranking.py`,
- a source draft field such as `bad_draft_assistant`.

It applies only the selected sidecar decision:

- `tool_name`: replace the selected call's tool name and, for
  `activate_voice_command`, fill the command/device/location from local request
  evidence.
- `argument_value`: replace the selected JSON key with the selector's chosen
  value.

It does not run the full deterministic sequence planner. That keeps the test
focused on whether a learned selector sidecar can supply the missing ambiguous
choice.

## Inputs

Cases:

```text
data/toolcall_eval/synthetic_multicall_failure_analogues.jsonl
```

Source draft:

```text
bad_draft_assistant
```

Selector JSONL files:

```text
runs/candidate_ranking/synthetic_multicall_failure_analogue_ckpt275_index_rank.jsonl
runs/candidate_ranking/synthetic_multicall_failure_analogue_leaveone_voice003_ckpt20_index_rank.jsonl
```

Outputs:

```text
runs/synthetic_multicall_failure_analogues/selector_sidecar_ckpt275_projection.jsonl
runs/synthetic_multicall_failure_analogues/selector_sidecar_leaveone_ckpt20_projection.jsonl
```

## Results

| path | exact sequence | exact arguments | valid tool JSON | extra/missing rows |
| --- | ---: | ---: | ---: | ---: |
| bad draft input | `4/8` | `0/8` | `8/8` | `4` extra, `4` missing |
| checkpoint-275 selector sidecar | `7/8` | `7/8` | `8/8` | `1` extra, `1` missing |
| selector-only leave-one ckpt20 sidecar | `8/8` | `8/8` | `8/8` | `0` extra, `0` missing |

The checkpoint-275 sidecar misses the same heldout selector row as the masked
ranking diagnostic. The selector-only leave-one checkpoint fixes that row:

```text
synthetic_voice_command_camera_003:
  checkpoint-275 selector: set_thermostat
  selector-only sidecar: activate_voice_command
```

## Interpretation

This is the first positive result for the separated architecture:

- selector-only SFT should not be merged directly into the generator adapter;
- as a separate sidecar scorer, the selector checkpoint can fix the ambiguous
  heldout decision;
- the main generator/protected draft path can remain checkpoint-275 while the
  selector sidecar supplies only the fragile candidate choice.

Caveat:

- The projection still uses deterministic local evidence extraction for
  `activate_voice_command` arguments after the sidecar picks the tool name.
- Therefore this is a protected system result, not evidence that raw block
  diffusion generation learned the full tool-call behavior.

Next implication:

- Build a real generation-time candidate scorer interface: the diffusion
  sampler should ask a selector/value sidecar to choose among candidate tool
  names or scalar values, then constrain that block accordingly.
- Keep reporting raw generator, constrained decoder, and protected sidecar
  metrics separately.

Follow-up:

- The first scheduled-sampler handoff is documented in
  `qwen35_selector_sidecar_scheduled_sampler_result.md`.
- The initial evidence-selected schedule improved tool sequence but left exact
  arguments at `0/8`, exposing a metadata bug where extractor-selected evidence
  overwrote correct sidecar-plan spans.
- The corrected target-selected schedule reaches `8/8` raw exact sequence,
  `8/8` raw exact arguments, and `8/8` valid JSON.
- A stronger follow-up leaves selected-candidate forcing off and lets the model
  choose whole candidate sequences; it still reaches `8/8` raw exact sequence
  and `8/8` raw exact arguments, even without selector injection on this
  synthetic slice.
- That makes sidecar-guided scheduling a live lane; the next blocker is
  proposing target-containing candidate sets and validating model-ranked choices
  on public/harder multi-call schedules.
