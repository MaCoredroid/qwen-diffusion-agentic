# Qwen3.5-9B Multi-Call Scalar Adapter Result

Date: 2026-06-27

## Status

Trained and evaluated a short second-stage scalar repair adapter on the
multi-call scalar curriculum.

This is not a promoted first-pass generator. It is a per-call repair stage over
the active checkpoint-275 constrained multi-call drafts.

## Training

Dataset:

```text
data/qwen35_9b_toolcall_multicall_scalar_curriculum
```

Adapter:

```text
runs/fastdllm_qwen35_9b_toolcall_multicall_scalar_argspanw1p5_b896_step100
```

Settings:

```text
MAX_STEPS=100
MAX_TRAIN_SAMPLES=512
BLOCK_SIZE=896
GRAD_ACCUM=1
LEARNING_RATE=3e-5
ARGUMENT_SPAN_LOSS_WEIGHT=1.5
LORA_R=8
LORA_ALPHA=16
LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj
```

Training result:

```text
global_step: 100
train_loss: 5.111620244979858
runtime: 215.4411s
throughput: 0.464 steps/s
readiness: ready=true
adapter saved: yes
```

## Eval

New evaluator:

```text
scripts/eval_fastdllm_toolcall_scalar_repair_outputs.py
```

Eval input:

```text
runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_multicall_eval384_modelrepair/public_multicall_12.jsonl
```

Eval output:

```text
runs/fastdllm_qwen35_9b_toolcall_multicall_scalar_argspanw1p5_b896_step100_eval/public_multicall_12_constrained_draft_v4.jsonl
```

The evaluator preserves the draft tool-call sequence, runs one scalar repair
generation per parsed call, conservatively accepts only missing/noisy/repeated
argument replacements, then scores both the repaired chain and a
sequence-preserving constrained projection.

## Results

Public Hermes multi-call, 12 rows:

| Path | Exact sequence | Exact args | Valid JSON | Schema valid | Missing-call records | Repeated-call records |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| input constrained draft | 7/12 | 1/12 | 11/12 | 10/12 | 3/12 | 0/12 |
| scalar repair composed chain | 7/12 | 3/12 | 12/12 | 11/12 | 3/12 | 0/12 |
| scalar repair + sequence-preserving constrained projection | 7/12 | 5/12 | 12/12 | 11/12 | 3/12 | 0/12 |

Comparison to the prior best deterministic projection:

```text
prior best sequence-preserving constrained projection: 7/12 sequence, 4/12 args
new scalar-repair constrained projection:          7/12 sequence, 5/12 args
```

Generation/eval runtime:

```text
records: 12
scalar generation calls: 29
scalar repaired calls accepted by conservative merge: 4
elapsed: 304.7584s
generated tokens/s: 8.5937
unresolved mask examples: 0
CUDA max allocated/reserved: 17.49 / 23.17 GiB
```

Arg-diff on `scalar_repair_constrained`:

```text
rows with exact tool sequence but wrong arguments: 2/12
value mismatches: 18
missing tool calls: 2
missing required fields: 1
array length mismatches: 1
```

## Interpretation

- Positive result for the multi-call lane: a staged scalar adapter improves
  public multi-call exact arguments from the current active best `4/12` to
  `5/12` without reducing exact tool-sequence recovery.
- The conservative accept policy matters. An unconstrained scalar merge can
  overwrite good draft values; the retained evaluator only accepts repairs for
  missing, noisy, truncated, or repeated-value arguments.
- This is still too slow and too procedural for deployment. It is useful as a
  training/eval signal and as a blueprint for a future generation-time
  constrained scalar-repair loop.
- Remaining failures are mostly wrong values in hard multi-call context,
  missing calls from the first-pass draft, and complex invoice payload fields.

## 300-Step Extension

I extended the same scalar-repair curriculum to 300 steps to test whether the
separate repair lane was still undertrained.

Adapter:

```text
runs/fastdllm_qwen35_9b_toolcall_multicall_scalar_argspanw1p5_b896_step300
```

Training result:

```text
global_step: 300
train_loss: 3.1057442967096964
runtime: 646.1586s
throughput: 0.464 steps/s
retained checkpoints: checkpoint-275, checkpoint-300
readiness: ready=true
adapter saved: yes
```

Eval outputs:

```text
runs/fastdllm_qwen35_9b_toolcall_multicall_scalar_argspanw1p5_b896_step300_eval/public_multicall_12_constrained_draft_ckpt275.jsonl
runs/fastdllm_qwen35_9b_toolcall_multicall_scalar_argspanw1p5_b896_step300_eval/public_multicall_12_constrained_draft_ckpt300.jsonl
```

Both retained checkpoints match the 100-step top-line score instead of
improving it:

| Scalar adapter | Exact sequence | Exact args | Valid JSON | Schema valid | Missing-call records | Repeated-call records |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 100-step adapter, constrained | 7/12 | 5/12 | 12/12 | 11/12 | 3/12 | 0/12 |
| 300-step checkpoint-275, constrained | 7/12 | 5/12 | 12/12 | 11/12 | 3/12 | 0/12 |
| 300-step checkpoint-300, constrained | 7/12 | 5/12 | 12/12 | 11/12 | 3/12 | 0/12 |

Runtime:

```text
checkpoint-275 eval: 271.4064s, 29 scalar calls, 3 accepted repairs, 9.40 generated tokens/s
checkpoint-300 eval: 278.4683s, 29 scalar calls, 4 accepted repairs, 9.79 generated tokens/s
CUDA max allocated/reserved: 17.46 / 21.79 GiB
```

Arg-diff on checkpoint-300 `scalar_repair_constrained`:

```text
rows with exact tool sequence but wrong arguments: 2/12
value mismatches: 15
missing tool calls: 2
missing required fields: 1
array length mismatches: 1
scalar value mismatch diffs: 15
complex missing-required diffs: 1
```

Conclusion:

- Longer scalar-adapter training is not the next useful lever; the separate
  scalar repair lane appears to plateau at `7/12` sequence and `5/12`
  arguments on this public multi-call slice.
- Keep the 100-step adapter as the cheaper active scalar-repair result unless a
  future run changes the data or decoding/accept policy.
- The next useful work is not more steps on this same curriculum. It is
  generation-time constrained scalar decoding, better per-field acceptance, or
  new hard-failure rows built from the remaining exact-sequence/wrong-argument
  cases.

## Contextual Scalar Projection Prototype

After the 300-step plateau, I inspected the remaining exact-tool-sequence but
wrong-argument rows. The scalar adapter was not producing correct replacement
values for clean but wrong draft scalars; it repeated the draft values. The next
useful lever was therefore request-evidence projection, not more training steps.

New script:

```text
scripts/rescore_scalar_repair_contextual_projection.py
```

The prototype keeps the active generator and scalar adapter fixed. It rescored
existing scalar-repair outputs by applying conservative scalar replacements from
the request context:

- exact numbered/function-specific request line first
- datetime fields from call-local date/time evidence
- ID fields from a single quoted ID in the specific request line
- extra ID plausibility guard to avoid replacing IDs with enum-like labels
- missing required scalar fields from an explicit property line under the same
  function section

Checkpoint-300 output:

```text
runs/fastdllm_qwen35_9b_toolcall_multicall_scalar_argspanw1p5_b896_step300_eval/public_multicall_12_ckpt300_contextual_projection_v3.jsonl
```

After adding explicit missing-required scalar fills, the same projection can run
directly on the active constrained drafts without the scalar adapter:

```text
runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_multicall_eval384_modelrepair/public_multicall_12_constrained_contextual_projection_v2.jsonl
```

Result on public multi-call:

| Path | Exact sequence | Exact args | Valid JSON | Schema valid | Missing-call records | Repeated-call records |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| active deterministic projection | 7/12 | 4/12 | 12/12 | n/a | 3/12 | 0/12 |
| scalar repair constrained | 7/12 | 5/12 | 12/12 | 11/12 | 3/12 | 0/12 |
| scalar repair + contextual projection v3 | 7/12 | 7/12 | 12/12 | 11/12 | 3/12 | 0/12 |
| constrained draft + contextual projection v2 | 7/12 | 7/12 | 12/12 | 11/12 | 3/12 | 0/12 |

Direct constrained-draft replacements made by v2:

```text
specific quoted ID replacements: 2
datetime-from-context replacements: 2
explicit property value replacements: 1
```

The two fixed exact-sequence rows were:

```text
get_recorded_feed:
  camera_id front_door -> front_garden
  start_time 2023-04-22T15:00:00 -> 2023-04-22T15:00:00Z
  end_time 2023-04-22T15:30:00 -> 2023-04-22T17:00:00Z

set_thermostat_temperature:
  device_id living-room-light-001 -> hallway-thermostat-002

activate_irrigation:
  duration missing -> 15
```

Arg-diff after direct constrained contextual projection v2:

```text
rows with exact tool sequence but wrong arguments: 0/12
rows with any remaining diff: 5/12
value mismatches: 10
missing tool calls: 2
missing required fields: 2
```

Interpretation:

- The cheaper direct contextual projection now ties the scalar-repair path at
  `7/12` sequence and `7/12` arguments. The scalar adapter is no longer needed
  to reach the current public multi-call postprocessed ceiling on this slice.
- It is not model learning and should not be reported as a model-only metric.
  It is evidence that generation-time constrained scalar decoding/per-field
  extraction is the right next implementation direction.
- The remaining ceiling is now mostly missing calls from the fixed generator
  draft and complex payload fields; the exact-sequence/wrong-scalar gap is
  closed on this 12-row slice by contextual projection.
