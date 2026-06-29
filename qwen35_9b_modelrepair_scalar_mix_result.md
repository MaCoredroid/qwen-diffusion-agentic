# Qwen3.5-9B Model-Repair Scalar-Mix Result

Date: 2026-06-27

## Status

Negative main-generator result.

This tested the next planned corpus change after the positive scalar repair
adapter: mix a lower-weight sample of the public multi-call scalar extraction
curriculum into the existing model-repair generator curriculum.

The run trained and saved cleanly, but checkpoint-275 regressed the public
multi-call gate and should not be promoted.

## Dataset

Builder:

```text
scripts/build_toolcall_modelrepair_scalar_mix.py
```

Output:

```text
data/qwen35_9b_toolcall_modelrepair_scalar_mix_curriculum
```

Mix:

```text
base model-repair rows: 227
multi-call scalar rows: 128
total rows: 355
```

Scalar row balance:

```text
empty_args: 32
missing_field: 32
null_field: 32
wrong_scalar: 32
```

Label/window audit:

```text
block size: 896
full-label rows: 355/355
zero-label rows: 0
partial-label rows: 0
length min/p50/p90/max: 239/577/828/890
kept labels min/p50/p90/max: 24/42/84/315
```

## Training

One-step gate:

```text
runs/fastdllm_qwen35_9b_toolcall_modelrepair_scalar_mix_argspanw1p5_b896_step1_gate
global_step: 1
train_loss: 6.1835551261901855
adapter saved: yes
readiness: ready=true
```

300-step run:

```text
runs/fastdllm_qwen35_9b_toolcall_modelrepair_scalar_mix_argspanw1p5_b896_step300
global_step: 300
train_loss: 3.4268787542978925
runtime: 646.165s
throughput: 0.464 steps/s
retained checkpoints: checkpoint-275, checkpoint-300
adapter saved: yes
```

Main settings:

```text
MAX_STEPS=300
MAX_TRAIN_SAMPLES=600
BLOCK_SIZE=896
GRAD_ACCUM=1
LEARNING_RATE=3e-5
ARGUMENT_SPAN_LOSS_WEIGHT=1.5
LORA_R=8
LORA_ALPHA=16
LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj
```

## Eval

Partial checkpoint sweep:

```text
runs/fastdllm_qwen35_9b_toolcall_modelrepair_scalar_mix_argspanw1p5_b896_step300_checkpoint_sweep_eval96_modelrepair_max1/checkpoint-275
```

The sweep was stopped after checkpoint-275 public multi-call regressed clearly.
`checkpoint-300` and completed tool-result summaries were not run for this
result.

Checkpoint-275 one-call and multi-call results:

| Slice | Raw seq | Raw args | Constrained seq | Constrained args | Model-repair seq | Model-repair args |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| public one-call | 1/8 | 1/8 | 7/8 | 3/8 | 5/8 | 3/8 |
| Qwen3.6 teacher train one-call | 1/12 | 1/12 | 11/12 | 3/12 | 5/12 | 4/12 |
| Qwen3.6 teacher heldout one-call | 0/8 | 0/8 | 7/8 | 2/8 | 1/8 | 0/8 |
| public multi-call | 0/12 | 0/12 | 4/12 | 1/12 | 1/12 | 1/12 |

Public multi-call sequence-preserving projection:

```text
valid JSON: 12/12
exact sequence: 4/12
exact arguments: 1/12
constrained exact sequence: 4/12
constrained exact arguments: 2/12
missing-call records: 5/12
repeated-call records: 0/12
```

Comparison to active paths:

```text
active checkpoint-275 sequence-preserving public multi-call: 7/12 seq, 4/12 args
scalar repair two-stage public multi-call:                    7/12 seq, 5/12 args
this scalar-mix checkpoint-275 sequence-preserving path:      4/12 seq, 2/12 args
```

## Interpretation

- The dataset and training path are healthy: labels are retained, loss is
  nonzero, adapters save, and the run fits the local 5090.
- Directly mixing 128 scalar extraction rows into the main generator hurts tool
  sequence behavior. The public multi-call targeted metric falls from the
  active `7/12` sequence / `4/12` argument path to `4/12` / `2/12`.
- One-call also regresses versus the active checkpoint-275 comparison point:
  public constrained arguments fall from `5/8` to `3/8`, and heldout
  constrained arguments fall from `3/8` to `2/8`.
- Do not promote this as a first-pass generator recipe.
- Keep the scalar curriculum as a separate repair/decoding signal for now. The
  positive result remains the two-stage scalar repair adapter, not direct
  generator mixing.

## Next

The next multi-call work should preserve the active generator and move scalar
repair into generation-time constrained decoding, per-field extraction, or a
separate lightweight repair stage. If scalar rows are retried in the generator,
use a much lower ratio with stronger one-call replay and require an early
public multi-call sequence gate before running the full scorecard.
