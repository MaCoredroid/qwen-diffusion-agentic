# Qwen3.5-9B Split-Route Sidecar Scorecard

Date: 2026-06-27

## Status

This is a routing/protection scorecard, not a newly promoted single adapter.
It tests the immediate implication of the checkpoint-24 experiments: keep the
better one-call generator behavior from staged checkpoint-24, but route known
multi-call/tool-result protection lanes through the active checkpoint-275
projection path where checkpoint-24 regresses.

The scorecard is an upper-bound target for a future sidecar or router. It is
not a claim that a deployed router has been implemented.

Machine-readable output:

- JSON: `runs/qwen35_9b_split_route_sidecar_eval/route_scorecard.json`
- TSV: `runs/qwen35_9b_split_route_sidecar_eval/route_scorecard.tsv`
- route manifest: `runs/qwen35_9b_split_route_sidecar_eval/route_manifest.json`
- gate verdict: `PASS`

Replay runner:

- script: `scripts/run_qwen35_split_route_sidecar_manifest.py`
- default plan JSON: `runs/qwen35_9b_split_route_sidecar_eval/replay_plan/route_runner_plan.json`
- default plan shell: `runs/qwen35_9b_split_route_sidecar_eval/replay_plan/route_runner_plan.sh`
- partial execution: `scripts/run_qwen35_split_route_sidecar_manifest.py --slice public_one_call --execute`
- output verifier: `scripts/run_qwen35_split_route_sidecar_manifest.py --verify-outputs --plan-json <plan.json>`
- historical verification: `runs/qwen35_9b_split_route_sidecar_eval/historical_verify_plan/route_runner_plan_verification.json`
- live public one-call smoke: `runs/qwen35_9b_split_route_sidecar_eval/live_smoke_public_onecall/route_runner_plan_verification.json`
- live OpenAI-style tool-result smoke: `runs/qwen35_9b_split_route_sidecar_eval/live_smoke_openai_toolresult/route_runner_plan_verification.json`
- live public multi-call planner smoke: `runs/qwen35_9b_split_route_sidecar_eval/live_smoke_public_multicall_planner/route_runner_plan_verification.json`
- live synthetic text tool-result smoke: `runs/qwen35_9b_split_route_sidecar_eval/live_smoke_synthetic_text_toolresult/route_runner_plan_verification.json`
- live teacher one-call smoke: `runs/qwen35_9b_split_route_sidecar_eval/live_smoke_teacher_onecall/route_runner_plan_verification.json`
- live coverage: all `6` split-route lanes have verified live replay artifacts

## Route Table

| Slice | Route | Active protected seq/args | Ckpt-24 protected seq/args | Routed protected seq/args | Routed raw seq/args | Rationale |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| public one-call | staged24_generator | 8/8 / 8/8 | 8/8 / 8/8 | 8/8 / 8/8 | 4/8 / 3/8 | keeps public constrained perfect and improves raw exact sequence/arguments |
| teacher-train one-call | staged24_generator | 10/12 / 6/12 | 11/12 / 6/12 | 11/12 / 6/12 | 2/12 / 2/12 | improves constrained sequence while preserving constrained arguments |
| teacher-heldout one-call | staged24_generator | 8/8 / 6/8 | 8/8 / 6/8 | 8/8 / 6/8 | 2/8 / 1/8 | improves raw heldout while preserving constrained heldout |
| public multi-call planner | active_protection_path | 11/12 / 10/12 | 11/12 / 9/12 | 11/12 / 10/12 | 7/12 / 7/12 | active planner keeps one more exact-argument case |
| synthetic text tool-result | staged24_generator | 10/10 / 8/10 | 10/10 / 9/10 | 10/10 / 9/10 | 6/10 / 4/10 | staged checkpoint-24 improves constrained exact arguments |
| OpenAI-style tool-result | active_protection_path | 10/10 / 9/10 | 10/10 / 8/10 | 10/10 / 9/10 | 6/10 / 6/10 | active checkpoint-275 keeps one more exact-argument case |

## Readout

- The split route preserves the staged checkpoint-24 public one-call raw gain
  (`4/8` sequence, `3/8` arguments) while keeping public constrained
  recovery at `8/8` / `8/8`.
- It keeps the active multi-call protected top line at `11/12` sequence and
  `10/12` arguments by routing that lane through checkpoint-275's guarded
  sequence planner.
- It keeps the active OpenAI-style tool-result protected top line at `10/10`
  sequence and `9/10` arguments by routing that lane through checkpoint-275.
- It uses checkpoint-24 for text-compatible synthetic tool-result, where
  checkpoint-24 reaches `10/10` sequence and `9/10` arguments versus active
  checkpoint-275's `10/10` / `8/10`.

## Gate Results

| Slice | Route | Routed source | Gate | Failed checks |
| --- | --- | --- | ---: | --- |
| public one-call | staged24_generator | `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_staged_retention_ckpt24_broad_eval96_modelrepair_agentic/checkpoint-24/public_onecall_8.summary.json` | PASS | none |
| teacher-train one-call | staged24_generator | `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_staged_retention_ckpt24_broad_eval96_modelrepair_agentic/checkpoint-24/teacher_train_labelaware_12.summary.json` | PASS | none |
| teacher-heldout one-call | staged24_generator | `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_staged_retention_ckpt24_broad_eval96_modelrepair_agentic/checkpoint-24/teacher_heldout_labelaware_8.summary.json` | PASS | none |
| public multi-call planner | active_protection_path | `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_multicall_eval384_modelrepair/public_multicall_12_sequence_planner_segmentargs_v3.summary.json` | PASS | none |
| synthetic text tool-result | staged24_generator | `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_staged_retention_ckpt24_broad_eval96_modelrepair_agentic/checkpoint-24/synthetic_toolresult_10.summary.json` | PASS | none |
| OpenAI-style tool-result | active_protection_path | `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_openai_toolresult_eval160_modelrepair_max1/synthetic_openai_toolresult_10_grounded_projection_v2.summary.json` | PASS | none |

## Next Experiment

Do not train broad anti-regression rows into the same generator adapter again.
The next practical experiment should implement a runtime router or sidecar
repair path with these gates:

- one-call prompts route to staged checkpoint-24 and must keep public raw
  `>=4/8` sequence and `>=3/8` arguments
- multi-call prompts route through active checkpoint-275 planner/projection
  until a sidecar matches `11/12` sequence and `10/12` arguments
- OpenAI-style tool-result prompts route through active checkpoint-275 until a
  sidecar matches `10/10` sequence and `9/10` arguments
- text tool-result prompts may route to checkpoint-24 if the route preserves
  `10/10` sequence and `9/10` arguments

## Source Artifacts

- public one-call, active: `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_eval96_modelrepair_max1/public_onecall_8_grounded_projection_v2.summary.json`
- public one-call, checkpoint-24: `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_staged_retention_ckpt24_broad_eval96_modelrepair_agentic/checkpoint-24/public_onecall_8.summary.json`
- teacher-train one-call, active: `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_eval96_modelrepair_max1/teacher_train_labelaware_12_grounded_projection_v2.summary.json`
- teacher-train one-call, checkpoint-24: `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_staged_retention_ckpt24_broad_eval96_modelrepair_agentic/checkpoint-24/teacher_train_labelaware_12.summary.json`
- teacher-heldout one-call, active: `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_eval96_modelrepair_max1/teacher_heldout_labelaware_8_grounded_projection_v2.summary.json`
- teacher-heldout one-call, checkpoint-24: `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_staged_retention_ckpt24_broad_eval96_modelrepair_agentic/checkpoint-24/teacher_heldout_labelaware_8.summary.json`
- public multi-call planner, active: `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_multicall_eval384_modelrepair/public_multicall_12_sequence_planner_segmentargs_v3.summary.json`
- public multi-call planner, checkpoint-24: `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_staged_retention_ckpt24_broad_eval96_modelrepair_agentic/checkpoint-24/public_multicall_12_sequence_planner_projection.summary.json`
- synthetic text tool-result, active: `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_toolresult_eval160_modelrepair_max1/synthetic_toolresult_10_grounded_projection_v2.summary.json`
- synthetic text tool-result, checkpoint-24: `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_staged_retention_ckpt24_broad_eval96_modelrepair_agentic/checkpoint-24/synthetic_toolresult_10.summary.json`
- OpenAI-style tool-result, active: `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_openai_toolresult_eval160_modelrepair_max1/synthetic_openai_toolresult_10_grounded_projection_v2.summary.json`
- OpenAI-style tool-result, checkpoint-24: `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_staged_retention_ckpt24_broad_eval96_modelrepair_agentic/checkpoint-24/synthetic_openai_toolresult_10.summary.json`
