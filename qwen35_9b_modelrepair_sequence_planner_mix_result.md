# Qwen3.5-9B Model-Repair + Sequence-Planner Mix Result

Date: 2026-06-27.

## Summary

Built and trained a low-ratio sequence-planner replay mix from the active
model-repair curriculum and the new train-only sequence-planner distillation
rows.

Result: negative for checkpoint promotion. The mix trains cleanly but regresses
the active checkpoint-275 public multi-call and tool-result gates. Keep the
deterministic sequence planner as a decoding/projection lane for now, not as a
main-generator training mix in this form.

## Dataset

Builder:

```text
scripts/build_toolcall_modelrepair_sequence_planner_mix.py
```

Output:

```text
data/qwen35_9b_toolcall_modelrepair_sequence_planner_mix_curriculum
```

Manifest:

```text
total rows:          240
base model-repair:  227
sequence-planner:    13
zero-label rows:      0
partial-label rows:   0
full-label rows:    240
```

Token audit:

```text
length:      min 239, p50 596, p90 841, max 896
full labels: min 24,  p50 41,  p90 93,  max 315
kept labels: min 24,  p50 41,  p90 93,  max 315
```

The mix is intentionally small: sequence-planner rows are about 5.4% of the
training set.

## Training

Run:

```text
runs/fastdllm_qwen35_9b_toolcall_modelrepair_seqplanner_mix_from_ckpt275_argspanw1p5_b896_step100
```

Starting adapter:

```text
runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model
```

Profile:

```text
systemd-run --user --scope
MemoryMax=28G
MemorySwapMax=4G
MAX_STEPS=100
MAX_TRAIN_SAMPLES=240
BLOCK_SIZE=896
GRAD_ACCUM=1
LR=3e-5
ARGUMENT_SPAN_LOSS_WEIGHT=1.5
LoRA targets: q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj
```

Training metrics:

```text
train loss:     1.738793797492981
runtime:        215.3991 s
samples/sec:    0.464
steps/sec:      0.464
epoch:          0.4167
```

Saved checkpoints:

```text
checkpoint-75
checkpoint-100
```

## Checkpoint-100 Eval

Sweep output:

```text
runs/fastdllm_qwen35_9b_toolcall_modelrepair_seqplanner_mix_from_ckpt275_argspanw1p5_b896_step100_checkpoint_sweep_eval96_modelrepair_max1
```

Summary:

| Eval | Raw seq/args | Constrained seq/args | Model repair seq/args |
| --- | ---: | ---: | ---: |
| public one-call 8 | 2/8, 2/8 | 7/8, 2/8 | 4/8, 3/8 |
| teacher train one-call 12 | 2/12, 2/12 | 11/12, 2/12 | 3/12, 2/12 |
| teacher heldout one-call 8 | 1/8, 1/8 | 7/8, 2/8 | 3/8, 1/8 |
| public multi-call 12 | 0/12, 0/12 | 5/12, 3/12 | 0/12, 0/12 |
| synthetic tool-result 10 | 5/10, 3/10 | 10/10, 7/10 | 6/10, 3/10 |
| OpenAI tool-result 10 | 2/10, 1/10 | 10/10, 5/10 | 3/10, 2/10 |

Public multi-call projection summary:

| Projection | Input seq/args | Projected seq/args |
| --- | ---: | ---: |
| sequence-preserving | 5/12, 3/12 | 5/12, 3/12 |
| contextual scalar | 5/12, 3/12 | 5/12, 4/12 |
| sequence planner | 5/12, 4/12 | 7/12, 5/12 |

## Comparison To Active Checkpoint-275

Active checkpoint-275 remains better:

```text
public one-call constrained:       active 8/8 seq, 5/8 args;  mix 7/8 seq, 2/8 args
teacher-train constrained:         active 10/12 seq, 5/12 args; mix 11/12 seq, 2/12 args
teacher-heldout constrained:       active 8/8 seq, 3/8 args;  mix 7/8 seq, 2/8 args
public multi-call contextual:      active 7/12 seq, 7/12 args; mix 5/12 seq, 4/12 args
public multi-call sequence-plan:   active 11/12 seq, 10/12 args; mix 7/12 seq, 5/12 args
synthetic tool-result constrained: active 10/10 seq, 8/10 args; mix 10/10 seq, 7/10 args
OpenAI tool-result constrained:    active 10/10 seq, 9/10 args; mix 10/10 seq, 5/10 args
```

## Interpretation

The 13 planner rows are label-clean, but adding them directly to the main
generator replay mix does not transfer the deterministic sequence-planner
benefit into model-only behavior. It also harms argument exactness on several
established gates.

Do not promote:

```text
runs/fastdllm_qwen35_9b_toolcall_modelrepair_seqplanner_mix_from_ckpt275_argspanw1p5_b896_step100/checkpoint-100
```

Next useful directions:

- keep `scripts/rescore_toolcall_sequence_planner_projection.py` as the
  promoted constrained-decoding/planner diagnostic
- use sequence-planner rows only in a separate lightweight repair/planner
  adapter, or require a much tighter acceptance policy before mixing them into
  the first-pass generator
- if retrying generator training, shorten the planner prompt and evaluate an
  early checkpoint before the full scorecard
- do not spend more 5090 time scaling this exact 240-row mix
