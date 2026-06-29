# Qwen3.5-9B Diffusion Checkpoint-275 Agentic Scorecard

Date: 2026-06-27

## Status

This is the current promoted Qwen3.5-9B Fast-DLLM diffusion/QLoRA checkpoint scorecard.
It consolidates the one-call, multi-call, and tool-result gates used by the roadmap.

This checkpoint is not an agentic closeout model yet. It is the active 9B diffusion
comparison point for the next data/training iteration.

## Checkpoint

```text
adapter: runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model
tokenizer: runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300
base model: models/qwen3.5-9b-fastdllm-init
sampler: Fast-DLLM full-context sampling
projection: deterministic grounded scalar/complex tool-call projection
```

## Results

| Slice | Valid JSON | Raw seq | Raw args | Constrained seq | Constrained args | Model-repair seq | Model-repair args | Extra / missing / repeated | Tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| public one-call, max-1 | 2/8 | 3/8 | 2/8 | 8/8 | 8/8 | n/a | n/a | 0 / 5 / 0 | n/a |
| Qwen3.6 teacher train one-call, max-1 | 1/12 | 2/12 | 2/12 | 10/12 | 6/12 | n/a | n/a | 0 / 10 / 0 | n/a |
| Qwen3.6 teacher heldout one-call, max-1 | 2/8 | 1/8 | 0/8 | 8/8 | 6/8 | n/a | n/a | 1 / 7 / 0 | n/a |
| public multi-call, sequence-preserving complex projection | 11/12 | 7/12 | 1/12 | 7/12 | 4/12 | n/a | n/a | 1 / 3 / 0 | n/a |
| public multi-call, complex + contextual projection | 12/12 | 7/12 | 7/12 | n/a | n/a | n/a | n/a | 1 / 3 / 0 | n/a |
| public multi-call, sequence-planner projection | 12/12 | 11/12 | 10/12 | n/a | n/a | n/a | n/a | 1 / 1 / 0 | n/a |
| public multi-call, scalar repair + contextual projection | 12/12 | 7/12 | 7/12 | n/a | n/a | n/a | n/a | 1 / 3 / 0 | n/a |
| synthetic tool-result, max-1 | 1/10 | 5/10 | 3/10 | 10/10 | 8/10 | n/a | n/a | 1 / 4 / 1 | n/a |
| OpenAI-style tool-result, max-1 | 3/10 | 6/10 | 6/10 | 10/10 | 9/10 | n/a | n/a | 1 / 3 / 0 | n/a |

## Gate Readout

- Public one-call has nonzero strict signal: raw `3/8` sequence and `2/8` arguments; grounded constrained max-1 reaches `8/8` / `8/8`.
- Grounded one-call projection also improves Qwen3.6 teacher-train and heldout
  one-call constrained exact arguments to `6/12` and `6/8`, without changing
  tool-result top lines or the public multi-call constrained-draft baseline.
- Public multi-call remains the main gap: sequence-preserving complex constrained projection reaches `7/12` sequence and `4/12` arguments.
- The best postprocessed public multi-call path now reaches `7/12` sequence and `7/12` arguments with direct constrained complex/contextual projection; scalar repair plus contextual projection ties that score but is slower. This is a deterministic projection prototype, not a model-only metric.
- Cross-slice grounded/contextual projection is neutral-to-positive on one-call and tool-result slices; see `qwen35_9b_contextual_projection_suite_result.md`.
- A guarded request-evidence sequence planner raises public multi-call to `11/12` sequence and `10/12` arguments. This is also a deterministic projection prototype, not a model-only metric.
- synthetic tool-result, max-1 is strong under constrained max-1 projection: `10/10` sequence and `8/10` arguments.
- OpenAI-style tool-result, max-1 is strong under constrained max-1 projection: `10/10` sequence and `9/10` arguments.

## Interpretation

- This checkpoint beats the 1.5B diffusion lab baseline on strict public tool-call metrics.
- It is still far below the Qwen3.5 AR and Qwen3.6 teacher multi-call baselines.
- The next training step should preserve the tool-result behavior while targeting missing-call recovery, raw complex-payload emission, and repeated-call-safe sequence control.
- Continue reporting raw strict metrics beside constrained metrics; grounded constrained projection is useful but not a substitute for the model learning valid tool calls.

## Source Artifacts

- public one-call, max-1: `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_eval96_modelrepair_max1/public_onecall_8_grounded_projection_v2.jsonl`
- Qwen3.6 teacher train one-call, max-1: `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_eval96_modelrepair_max1/teacher_train_labelaware_12_grounded_projection_v2.jsonl`
- Qwen3.6 teacher heldout one-call, max-1: `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_eval96_modelrepair_max1/teacher_heldout_labelaware_8_grounded_projection_v2.jsonl`
- public multi-call, sequence-preserving complex projection: `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_multicall_eval384_modelrepair/public_multicall_12_sequence_preserve_complex_v4.jsonl`
- public multi-call, complex + contextual projection: `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_multicall_eval384_modelrepair/public_multicall_12_sequence_preserve_complex_contextual_v4.jsonl`
- public multi-call, sequence-planner projection: `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_multicall_eval384_modelrepair/public_multicall_12_sequence_planner_segmentargs_v3.jsonl`
- public multi-call, scalar repair + contextual projection: `runs/fastdllm_qwen35_9b_toolcall_multicall_scalar_argspanw1p5_b896_step300_eval/public_multicall_12_ckpt300_contextual_projection_v4.jsonl`
- synthetic tool-result, max-1: `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_toolresult_eval160_modelrepair_max1/synthetic_toolresult_10_grounded_projection_v2.jsonl`
- OpenAI-style tool-result, max-1: `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_openai_toolresult_eval160_modelrepair_max1/synthetic_openai_toolresult_10_grounded_projection_v2.jsonl`
