# Qwen3.5-9B Contextual Projection Suite Result

Date: 2026-06-27

## Status

This is a CPU-only rescoring check over the active Qwen3.5-9B diffusion
checkpoint-275 constrained outputs. It tests whether deterministic
request-evidence scalar projection generalizes beyond the public multi-call
slice.

This is not model-only learning. It is a decoding/postprocessing diagnostic for
the next generation-time constrained scalar/per-field decoder.

## Result

| Slice | Input sequence | Input args | Projected sequence | Projected args | Replacements |
| --- | ---: | ---: | ---: | ---: | --- |
| public one-call, max-1 | 8/8 | 5/8 | 8/8 | 5/8 | none |
| Qwen3.6 teacher train one-call, max-1 | 10/12 | 5/12 | 10/12 | 5/12 | none |
| Qwen3.6 teacher heldout one-call, max-1 | 8/8 | 3/8 | 8/8 | 3/8 | none |
| public multi-call | 7/12 | 4/12 | 7/12 | 7/12 | 2 datetime, 2 quoted-ID, 1 explicit property |
| synthetic tool-result, max-1 | 10/10 | 8/10 | 10/10 | 8/10 | none |
| OpenAI-style tool-result, max-1 | 10/10 | 9/10 | 10/10 | 9/10 | none |

## Interpretation

- Contextual projection is specifically useful for public multi-call scalar
  grounding. It does not move one-call or tool-result gates because those
  constrained outputs already lack the explicit request-evidence pattern this
  rule targets.
- Public multi-call improves from `7/12` sequence and `4/12` arguments to
  `7/12` sequence and `7/12` arguments.
- The exact-sequence-but-wrong-argument row count on public multi-call drops to
  `0/12`.
- Remaining public multi-call failures are now missing calls and complex
  payloads, not simple scalar-copy errors inside otherwise correct tool
  sequences.
- Future checkpoint sweeps should include this projection for public multi-call
  promotion, but raw and model-only constrained metrics must still be reported
  separately.

## Complex Context Projection Follow-Up

Date: 2026-06-27.

The constrained decoder now also has a conservative complex-context extraction
path for array/object arguments in `scripts/eval_fastdllm_toolcall_cases.py`.
It handles request-evidence shapes such as markdown tables, bullet lists, and
inline lists before accepting malformed generated JSON fragments.

This is still a deterministic decoding/postprocessing diagnostic, not model-only
learning. The useful change is that complex public multi-call payloads are no
longer blocked on another short LoRA continuation.

| Slice | v1/v2 constrained args | Complex v3 constrained args | Direction |
| --- | ---: | ---: | --- |
| public one-call, max-1 | 5/8 | 5/8 | neutral |
| Qwen3.6 teacher train one-call, max-1 | 5/12 | 5/12 | neutral |
| Qwen3.6 teacher heldout one-call, max-1 | 3/8 | 4/8 | positive |
| public multi-call, sequence + scalar projection | 7/12 | 7/12 | neutral top line; schema/required improves to 12/12 |
| synthetic tool-result, max-1 | 8/10 | 8/10 | neutral |
| OpenAI-style tool-result, max-1 | 9/10 | 9/10 | neutral |

Complex held-out gap eval:

- active checkpoint-275 improves from `2/7` to `7/7` constrained exact
  arguments.
- the 25-step complex-only adapter improves from `3/7` to `7/7` constrained
  exact arguments, but remains a negative model-promotion result because raw
  exact arguments are worse than the active checkpoint.

Decision: promote the complex context projection as part of the constrained
decoder baseline; do not promote the complex-only adapter.

## Guarded Sequence-Planner Projection

Date: 2026-06-27.

Added `scripts/rescore_toolcall_sequence_planner_projection.py`, a CPU-only
diagnostic that uses request list/table structure plus tool schema text to
propose a multi-call order. It is guarded by default so it only replaces outputs
that already contain at least two tool calls; one-call and tool-result slices
therefore keep their existing constrained outputs.

This is not model-only learning. It is a prototype for generation-time
tool-plan constraints or a lightweight planner stage in front of argument
decoding.

| Slice | Input sequence | Input args | Planned sequence | Planned args | Direction |
| --- | ---: | ---: | ---: | ---: | --- |
| public one-call, max-1 | 8/8 | 5/8 | 8/8 | 5/8 | neutral |
| Qwen3.6 teacher train one-call, max-1 | 10/12 | 5/12 | 10/12 | 5/12 | neutral |
| Qwen3.6 teacher heldout one-call, max-1 | 8/8 | 4/8 | 8/8 | 4/8 | neutral |
| public multi-call | 7/12 | 7/12 | 11/12 | 10/12 | positive |
| synthetic tool-result, max-1 | 10/10 | 8/10 | 10/10 | 8/10 | neutral |
| OpenAI-style tool-result, max-1 | 10/10 | 9/10 | 10/10 | 9/10 | neutral |

Remaining public multi-call failures:

- one semantic tool-choice case where the request asks to use voice commands
  for security cameras, but a direct `activate_security_cameras` tool is also
  available
- one exact-sequence row with a scalar code assignment mismatch in the smart
  home security case

Decision: keep the sequence planner as a promoted deterministic diagnostic. Do
not count it as a model-only metric. The next model/data step should teach the
student to emit the planner's order directly, while the next decoder step should
continue improving segment-local scalar evidence after a planned reorder.

## Grounded One-call Projection Follow-Up

Date: 2026-06-27.

After several short checkpoint-275 continuations regressed the public one-call
gate, I kept the trained adapter fixed and improved the constrained projector
instead. The new grounded projection keeps the same request-evidence principle,
but adds conservative schema-pattern extraction for:

- weekly schedule arrays with `day`, `temperature`, and `time` item fields
- ID-like string fields where the generated scalar is a truncated copy
- request-local string fields such as voice command, music playlist, lighting
  scene, and periodic function type

This is still deterministic decoding/postprocessing, not model-only learning.
The value is that it gives a concrete generation-time constrained-decoding
target without spending another 5090 training run on a replay mix.

| Slice | Previous constrained seq | Previous constrained args | Grounded seq | Grounded args | Direction |
| --- | ---: | ---: | ---: | ---: | --- |
| public one-call, max-1 | 8/8 | 5/8 | 8/8 | 8/8 | positive |
| Qwen3.6 teacher train one-call, max-1 | 10/12 | 5/12 | 10/12 | 6/12 | positive |
| Qwen3.6 teacher heldout one-call, max-1 | 8/8 | 4/8 | 8/8 | 6/8 | positive |
| public multi-call, constrained-draft sequence-preserve | 7/12 | 4/12 | 7/12 | 4/12 | neutral |
| synthetic tool-result, max-1 | 10/10 | 8/10 | 10/10 | 8/10 | neutral |
| OpenAI-style tool-result, max-1 | 10/10 | 9/10 | 10/10 | 9/10 | neutral |

The public one-call argument-diff audit over `constrained_assistant` has
`rows_with_diff=0` and `rows_exact_tool_sequence_but_not_arguments=0`.

Decision: promote grounded projection as the current one-call constrained
decoder baseline. Keep raw strict metrics and the active checkpoint-275 adapter
unchanged. The next implementation step should move these request-evidence
rules closer to generation-time constrained span filling rather than another
broad repair/full-span training replay.

## Artifacts

- public one-call:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_eval96_modelrepair_max1/public_onecall_8_constrained_contextual_projection_v1.jsonl`
- Qwen3.6 teacher train one-call:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_eval96_modelrepair_max1/teacher_train_labelaware_12_constrained_contextual_projection_v1.jsonl`
- Qwen3.6 teacher heldout one-call:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_eval96_modelrepair_max1/teacher_heldout_labelaware_8_constrained_contextual_projection_v1.jsonl`
- public multi-call:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_multicall_eval384_modelrepair/public_multicall_12_constrained_contextual_projection_v2.jsonl`
- synthetic tool-result:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_toolresult_eval160_modelrepair_max1/synthetic_toolresult_10_constrained_contextual_projection_v1.jsonl`
- OpenAI-style tool-result:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_openai_toolresult_eval160_modelrepair_max1/synthetic_openai_toolresult_10_constrained_contextual_projection_v1.jsonl`
- public one-call, complex projection v3:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_eval96_modelrepair_max1/public_onecall_8_complex_projection_v3.jsonl`
- Qwen3.6 teacher train one-call, complex projection v3:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_eval96_modelrepair_max1/teacher_train_labelaware_12_complex_projection_v3.jsonl`
- Qwen3.6 teacher heldout one-call, complex projection v3:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_eval96_modelrepair_max1/teacher_heldout_labelaware_8_complex_projection_v3.jsonl`
- public multi-call, sequence-preserving complex + contextual projection v4:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_multicall_eval384_modelrepair/public_multicall_12_sequence_preserve_complex_contextual_v4.jsonl`
- synthetic tool-result, complex projection v3:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_toolresult_eval160_modelrepair_max1/synthetic_toolresult_10_complex_projection_v3.jsonl`
- OpenAI-style tool-result, complex projection v3:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_openai_toolresult_eval160_modelrepair_max1/synthetic_openai_toolresult_10_complex_projection_v3.jsonl`
- public multi-call, guarded sequence-planner projection:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_multicall_eval384_modelrepair/public_multicall_12_sequence_planner_segmentargs_v3.jsonl`
- public one-call, guarded sequence-planner neutral check:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_eval96_modelrepair_max1/public_onecall_8_sequence_planner_segmentargs_v2.jsonl`
- Qwen3.6 teacher train one-call, guarded sequence-planner neutral check:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_eval96_modelrepair_max1/teacher_train_labelaware_12_sequence_planner_segmentargs_v2.jsonl`
- Qwen3.6 teacher heldout one-call, guarded sequence-planner neutral check:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_eval96_modelrepair_max1/teacher_heldout_labelaware_8_sequence_planner_segmentargs_v2.jsonl`
- synthetic tool-result, guarded sequence-planner neutral check:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_toolresult_eval160_modelrepair_max1/synthetic_toolresult_10_sequence_planner_segmentargs_v2.jsonl`
- OpenAI-style tool-result, guarded sequence-planner neutral check:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_openai_toolresult_eval160_modelrepair_max1/synthetic_openai_toolresult_10_sequence_planner_segmentargs_v2.jsonl`
- public one-call, grounded projection v2:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_eval96_modelrepair_max1/public_onecall_8_grounded_projection_v2.jsonl`
- public one-call, grounded projection v2 argdiff:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_eval96_modelrepair_max1/public_onecall_8_grounded_projection_v2_argdiff.jsonl`
- Qwen3.6 teacher train one-call, grounded projection v2:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_eval96_modelrepair_max1/teacher_train_labelaware_12_grounded_projection_v2.jsonl`
- Qwen3.6 teacher heldout one-call, grounded projection v2:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_eval96_modelrepair_max1/teacher_heldout_labelaware_8_grounded_projection_v2.jsonl`
- public multi-call, constrained-draft grounded sequence-preserve v1:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_multicall_eval384_modelrepair/public_multicall_12_constrained_grounded_sequence_preserve_v1.jsonl`
- synthetic tool-result, grounded projection v2:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_toolresult_eval160_modelrepair_max1/synthetic_toolresult_10_grounded_projection_v2.jsonl`
- OpenAI-style tool-result, grounded projection v2:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_openai_toolresult_eval160_modelrepair_max1/synthetic_openai_toolresult_10_grounded_projection_v2.jsonl`
