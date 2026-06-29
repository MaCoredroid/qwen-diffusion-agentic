# Qwen3.5-9B Grounded Span-Fill Curriculum Result

Date: 2026-06-27

## Status

This is the first attempt to turn the promoted grounded one-call projection into
a trainable Qwen3.5-9B diffusion objective.

It is not a promoted checkpoint. The result is a fit and safety gate: block
size `1024` grounded span-fill training fits on the local RTX 5090, preserves
the grounded constrained top line after one step, but does not improve raw
model-only tool-call behavior.

## Why This Exists

The active checkpoint-275 can be rescued by deterministic grounded projection,
but that is not model learning. The next useful step is to convert the
projection into supervised examples where the model sees:

- original user request and tool schema
- previous raw assistant draft
- previous constrained projection
- exact grounded `<tool_call>` target

This tests whether the model can start internalizing request-evidence copying
instead of relying on postprocessing.

## Curriculum Builder

Added:

- `scripts/build_toolcall_grounded_spanfill_curriculum.py`

The builder consumes the Qwen3.6 teacher-train one-call slice plus active
checkpoint-275 raw and grounded-projection outputs. By default it only accepts
rows where grounded projection is exact.

Block size `896` build:

- output: `data/qwen35_9b_toolcall_grounded_spanfill_teacher_train_curriculum`
- accepted rows: `12`
- rejected rows: `12`
- skipped rows: `6` not grounded exact
- accepted label retention: full labels kept for chosen rows

Block size `1024` build:

- output:
  `data/qwen35_9b_toolcall_grounded_spanfill_teacher_train_b1024_curriculum`
- accepted rows: `16`
- rejected rows: `8`
- skipped rows: `6` not grounded exact
- chosen length min / p50 / p90 / max: `710 / 835 / 979 / 979`
- chosen kept labels min / p50 / p90 / max: `24 / 55 / 315 / 315`
- chosen rows with zero or partial labels after truncation: `0`

The `1024` build recovers the longer schedule-style rows that were partially
truncated at block size `896`.

## One-Step Gate

Training command was run under the user cgroup cap:

```text
systemd-run --user --scope --quiet -p MemoryMax=28G -p MemorySwapMax=4G
```

Run:

- starting adapter:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model`
- dataset:
  `data/qwen35_9b_toolcall_grounded_spanfill_teacher_train_b1024_curriculum`
- output:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_spanfill_from_ckpt275_b1024_step1`
- block size: `1024`
- max steps: `1`
- max train samples: `16`
- LR: `5e-6`
- argument-span weight: `1.5`
- GDN mode: `option_a_causal_gdn_v0`
- train loss: `3.068970203399658`
- runtime: `3.165` seconds

Peak eval memory stayed within the current local 5090 path:

- CUDA allocated: about `17.85 GiB`
- CUDA reserved: up to `27.35 GiB`

## Eval Result

Eval used full-context Fast-DLLM sampling with grounded constrained decoding,
`max_new_tokens=96`, and `constrained_max_calls=1`.

| Slice | Raw seq | Raw args | Constrained seq | Constrained args | Extra / missing / repeated |
| --- | ---: | ---: | ---: | ---: | ---: |
| public one-call | 3/8 | 2/8 | 8/8 | 8/8 | 0 / 5 / 0 |
| Qwen3.6 teacher train one-call | 2/12 | 2/12 | 10/12 | 6/12 | 0 / 10 / 0 |
| Qwen3.6 teacher heldout one-call | 1/8 | 0/8 | 8/8 | 6/8 | 1 / 7 / 0 |

Compared with active checkpoint-275 plus grounded projection, this is a tie on
the constrained top line and no improvement on raw model-only metrics.

## Decision

Do not promote this one-step adapter over active checkpoint-275.

What we learned:

- The grounded projection can be represented as a label-aware training corpus.
- Block size `1024` is practical for these one-call grounded examples on the
  RTX 5090 under the current memory cap.
- A one-step CE continuation from checkpoint-275 preserves the projected score
  but does not make the model emit better raw calls.
- More broad replay is unlikely to be the right main path unless the objective
  changes.

## Next Innovation Path

There is no mature recipe for Qwen3.5/Qwen3.6 Gated DeltaNet diffusion
conversion, so the next experiments should be controlled ablations rather than
larger blind data mixes:

- span-local denoising target: force masks only on grounded argument-value spans
  and train the model to fill those values from request evidence
- teacher-KL over argument spans: store a small span-local teacher signal from
  Qwen3.6 where feasible, instead of relying only on token CE
- two-stage planner/filler: train tool-sequence planning separately from
  argument filling, then compose at decoding time
- GDN-specific state ablation: re-test clean-state injection or GDN-only LoRA
  only when paired with a sharper span objective
- generation-time constrained span filling: move the grounded projection rules
  closer to decoding instead of treating them only as postprocessing

Promotion should require a raw/model-only gain, not just a projected-score tie.

Evidence policy:

- The `1/5/25`-step runs in this note are fit and regression gates. They are not
  sufficient evidence to declare a training mechanism dead.
- The next useful evidence level is a dose curve: lower LR, saved checkpoints,
  and identical eval slices at each checkpoint. A mechanism is only a promotion
  candidate if it improves at least one raw/model-only metric while preserving
  the active constrained/projected top line.
- The next concrete dose probe is value-span label-only denoising from active
  checkpoint-275 at LR `1e-6`, block `1024`, `100` steps, saved/evaluated every
  `25` steps.

## Value-Span Mask Follow-Up

Date: 2026-06-27.

Added a narrower value-span mask-forcing hook:

- model env: `FASTDLLM_VALUE_SPAN_MASK_PROB`
- launcher env: `VALUE_SPAN_MASK_PROB`
- model file:
  `models/qwen3.5-9b-fastdllm-init/modeling.py`
- launcher:
  `scripts/run_fastdllm_qwen35_9b_agentic_qlora_pilot.sh`

This differs from earlier `ARGUMENT_SPAN_MASK_PROB`: it only forces labels that
are both inside the derived `arguments ... </tool_call>` span and in the
dataset-derived scalar argument-value token set. It is a sharper denoising
objective for grounded copying, not another broad full-span replay.

Grounded b1024 value-token audit:

- dataset:
  `data/qwen35_9b_toolcall_grounded_spanfill_teacher_train_b1024_curriculum`
- tool calls: `16`
- scalar values: `200`
- unique scalar values: `18`
- value token IDs: `27`
- argument span boundary IDs:
  - start `arguments`: `15889`
  - end `</tool_call>`: `248059`

One-step gate:

- output:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespanmask_from_ckpt275_b1024_step1`
- starting adapter:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model`
- block size: `1024`
- LR: `5e-6`
- argument-span loss weight: `1.5`
- value-span loss weight: `2.0`
- value-span mask probability: `1.0`
- GDN mode: `option_a_causal_gdn_v0`
- first debug batch:
  - valid labels: `55`
  - argument-span labels: `41`
  - value-span labels: `17`
  - forced argument-span labels: `0`
  - forced value-span labels: `17`
- train loss: `3.5500881671905518`

Eval result:

| Slice | Raw seq | Raw args | Constrained seq | Constrained args |
| --- | ---: | ---: | ---: | ---: |
| public one-call | 3/8 | 2/8 | 8/8 | 8/8 |
| Qwen3.6 teacher train one-call | 2/12 | 2/12 | 10/12 | 6/12 |
| Qwen3.6 teacher heldout one-call | 1/8 | 0/8 | 8/8 | 6/8 |

Decision: do not promote the value-span-mask one-step adapter. It ties the
active grounded constrained top line and raw scores, so it is a clean
non-regression infrastructure result, not a better model. The hook is still a
better next lever than whole argument-span mask forcing because it applies
pressure exactly to copied grounded values.

The guarded 25-step test below checks whether this hook scales beyond a
one-step fit gate. If it does not improve raw metrics, the next objective should
move to lower mask pressure, a true span-only label target, or teacher-KL over
grounded argument values.

### Value-Span-Only 25-Step Continuation

Date: 2026-06-27.

Ran the guarded short continuation to check whether the cleaner hook scales
beyond the one-step smoke.

Run:

- output:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespanmask_only_from_ckpt275_b1024_step25`
- starting adapter:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model`
- dataset:
  `data/qwen35_9b_toolcall_grounded_spanfill_teacher_train_b1024_curriculum`
- block size: `1024`
- max steps: `25`
- LR: `5e-6`
- argument-span loss weight: `1.0`
- value-span loss weight: `2.0`
- value-span mask probability: `1.0`
- GDN mode: `option_a_causal_gdn_v0`
- first debug batch:
  - valid labels: `55`
  - argument-span labels: `41`
  - value-span labels: `17`
  - forced argument-span labels: `0`
  - forced value-span labels: `17`
  - weighted labels: `17`
- train loss: `2.068201866149902`
- runtime: `62.6146` seconds

Eval result:

| Slice | Raw seq | Raw args | Constrained seq | Constrained args | Direction vs active |
| --- | ---: | ---: | ---: | ---: | --- |
| public one-call | 2/8 | 1/8 | 8/8 | 8/8 | raw regression, constrained tie |
| Qwen3.6 teacher train one-call | 1/12 | 1/12 | 10/12 | 7/12 | raw regression, constrained args +1 |
| Qwen3.6 teacher heldout one-call | 1/8 | 0/8 | 7/8 | 5/8 | constrained regression |

Decision: do not promote or scale this setting. Value-span-only masking is a
cleaner probe than whole argument-span forcing, but `VALUE_SPAN_MASK_PROB=1.0`
for 25 steps overfits/damages first-pass emission and even loses heldout
constrained recovery. Keep the hook, but the next objective needs either lower
mask pressure, a true span-only label target, or teacher-KL over grounded
argument values.

### Value-Span Label-Only One-Step Gate

Date: 2026-06-27.

Added a true span-only label objective:

- model env: `FASTDLLM_VALUE_SPAN_LABEL_ONLY`
- launcher env: `VALUE_SPAN_LABEL_ONLY`
- model file:
  `models/qwen3.5-9b-fastdllm-init/modeling.py`
- launcher:
  `scripts/run_fastdllm_qwen35_9b_agentic_qlora_pilot.sh`

When enabled, the model keeps the full rendered tool-call sequence as context
but drops non-value assistant labels before MDM masking. The target is narrower
than value-span weighting: only copied grounded scalar value tokens contribute
loss. This is meant to avoid damaging wrapper/tool-name formatting while testing
whether the model can learn request-evidence value copying.

One-step gate:

- output:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_from_ckpt275_b1024_step1`
- starting adapter:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model`
- dataset:
  `data/qwen35_9b_toolcall_grounded_spanfill_teacher_train_b1024_curriculum`
- block size: `1024`
- max steps: `1`
- LR: `5e-6`
- argument-span loss weight: `1.0`
- value-span loss weight: `1.0`
- value-span label-only: enabled
- value-span mask probability: `1.0`
- first debug batch:
  - pre-MDM assistant labels: `55`
  - value-only labels: `17`
  - forced value labels: `17`
  - post-MDM labels: `[17, 0]`
- train loss: `0.7235846519470215`

Eval result:

| Slice | Raw seq | Raw args | Constrained seq | Constrained args | Direction vs active |
| --- | ---: | ---: | ---: | ---: | --- |
| public one-call | 3/8 | 2/8 | 8/8 | 8/8 | tie |
| Qwen3.6 teacher train one-call | 2/12 | 2/12 | 10/12 | 6/12 | tie |
| Qwen3.6 teacher heldout one-call | 1/8 | 0/8 | 8/8 | 6/8 | tie |

Decision: do not promote the one-step adapter because raw/model-only metrics do
not improve. Keep `VALUE_SPAN_LABEL_ONLY` as the safer next objective hook. It
preserves all active one-call gates after one step, unlike the 25-step
`VALUE_SPAN_MASK_PROB=1.0` weighting run, which regressed public raw and heldout
constrained metrics. The next controlled run should test a short continuation
with span-only labels and strict promotion guards.

### Value-Span Label-Only 25-Step Continuation

Date: 2026-06-27.

Ran the short guarded continuation to see whether the span-only objective scales
beyond a one-step fit gate.

Run:

- output:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_from_ckpt275_b1024_step25`
- starting adapter:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model`
- dataset:
  `data/qwen35_9b_toolcall_grounded_spanfill_teacher_train_b1024_curriculum`
- block size: `1024`
- max steps: `25`
- LR: `5e-6`
- argument-span loss weight: `1.0`
- value-span loss weight: `1.0`
- value-span label-only: enabled
- value-span mask probability: `1.0`
- first debug batch:
  - pre-MDM assistant labels: `55`
  - value-only labels: `17`
  - forced value labels: `17`
  - post-MDM labels: `[17, 0]`
- train loss: `0.3458780336380005`
- runtime: `62.568` seconds

Eval result:

| Slice | Raw seq | Raw args | Constrained seq | Constrained args | Direction vs active |
| --- | ---: | ---: | ---: | ---: | --- |
| public one-call | 3/8 | 2/8 | 8/8 | 7/8 | constrained args -1 |
| Qwen3.6 teacher train one-call | 1/12 | 1/12 | 11/12 | 6/12 | raw regression, constrained seq +1 |
| Qwen3.6 teacher heldout one-call | 2/8 | 1/8 | 8/8 | 6/8 | raw improvement, constrained tie |

Decision: do not promote the 25-step label-only adapter. It improves heldout
raw exact sequence and arguments, and improves teacher-train constrained
sequence, but it regresses public constrained exact arguments and teacher-train
raw exact sequence/arguments. The hook remains the cleanest value-copy
objective so far, but the next run should expose earlier checkpoints or reduce
update pressure rather than accepting the step-25 adapter.

### Value-Span Label-Only 5-Step Continuation

Date: 2026-06-27.

Ran a shorter continuation to test whether the useful step-25 heldout raw signal
appears before the public/train regressions.

Run:

- output:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_from_ckpt275_b1024_step5`
- starting adapter:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model`
- dataset:
  `data/qwen35_9b_toolcall_grounded_spanfill_teacher_train_b1024_curriculum`
- block size: `1024`
- max steps: `5`
- LR: `5e-6`
- argument-span loss weight: `1.0`
- value-span loss weight: `1.0`
- value-span label-only: enabled
- value-span mask probability: `1.0`
- first debug batch:
  - pre-MDM assistant labels: `55`
  - value-only labels: `17`
  - forced value labels: `17`
  - post-MDM labels: `[17, 0]`
- train loss: `0.4865542411804199`
- runtime: `13.0861` seconds

Eval result:

| Slice | Raw seq | Raw args | Constrained seq | Constrained args | Direction vs active |
| --- | ---: | ---: | ---: | ---: | --- |
| public one-call | 2/8 | 1/8 | 8/8 | 8/8 | raw regression, constrained tie |
| Qwen3.6 teacher train one-call | 1/12 | 1/12 | 10/12 | 6/12 | raw regression, constrained tie |
| Qwen3.6 teacher heldout one-call | 1/8 | 0/8 | 7/8 | 5/8 | constrained regression |

Decision: do not promote or continue this pressure setting. Step 5 regresses
raw behavior without producing the step-25 heldout raw improvement, and heldout
constrained recovery regresses. The `VALUE_SPAN_LABEL_ONLY` hook remains useful
as infrastructure, but repeated full-strength `VALUE_SPAN_MASK_PROB=1.0`
updates are too brittle. The next controlled test should lower update pressure
or move to teacher-KL/span distillation rather than more same-setting step
sweeps.

### Value-Span Label-Only LR 1e-6 Checkpoint Sweep

Date: 2026-06-27.

After reviewing the evidence strength, reran the value-span label-only objective
as a dose curve rather than another isolated short gate. The purpose was to test
whether the earlier regressions were caused by too much update pressure at
`5e-6`, not to declare the objective good or bad from a tiny sample.

Run:

- output:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_from_ckpt275_b1024_lr1e6_step100`
- archived planned checkpoints:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_from_ckpt275_b1024_lr1e6_step100_checkpoint_archive`
- starting adapter:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model`
- dataset:
  `data/qwen35_9b_toolcall_grounded_spanfill_teacher_train_b1024_curriculum`
- block size: `1024`
- max steps: `100`
- LR: `1e-6`
- saved/evaluated checkpoints: `25`, `50`, `75`, `100`
- value-span label-only: enabled
- value-span mask probability: `1.0`
- first debug batch:
  - pre-MDM assistant labels: `55`
  - value-only labels: `17`
  - forced value labels: `17`
  - post-MDM labels: `[17, 0]`
- train loss: `0.36141203343868256`
- train runtime: `249.1233` seconds

One-call checkpoint sweep:

| Checkpoint | Slice | Raw valid | Raw seq | Raw args | Constrained seq | Constrained args | Model-repair seq | Model-repair args |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | public one-call | 2/8 | 3/8 | 2/8 | 8/8 | 8/8 | 4/8 | 2/8 |
| 25 | teacher train | 1/12 | 2/12 | 2/12 | 10/12 | 6/12 | 5/12 | 3/12 |
| 25 | teacher heldout | 2/8 | 1/8 | 0/8 | 7/8 | 5/8 | 3/8 | 1/8 |
| 50 | public one-call | 2/8 | 3/8 | 2/8 | 8/8 | 8/8 | 5/8 | 3/8 |
| 50 | teacher train | 1/12 | 2/12 | 2/12 | 10/12 | 6/12 | 5/12 | 4/12 |
| 50 | teacher heldout | 2/8 | 1/8 | 0/8 | 8/8 | 6/8 | 4/8 | 2/8 |
| 75 | public one-call | 2/8 | 3/8 | 2/8 | 8/8 | 8/8 | 3/8 | 2/8 |
| 75 | teacher train | 1/12 | 2/12 | 2/12 | 11/12 | 7/12 | 4/12 | 4/12 |
| 75 | teacher heldout | 2/8 | 1/8 | 0/8 | 8/8 | 6/8 | 2/8 | 1/8 |
| 100 | public one-call | 1/8 | 3/8 | 2/8 | 8/8 | 8/8 | 4/8 | 2/8 |
| 100 | teacher train | 0/12 | 2/12 | 2/12 | 10/12 | 6/12 | 5/12 | 4/12 |
| 100 | teacher heldout | 2/8 | 1/8 | 0/8 | 8/8 | 6/8 | 3/8 | 1/8 |

Decision: do not promote this adapter family yet. LR `1e-6` gives a real
scaling signal: checkpoints `50` and `75` preserve the active one-call
constrained top line, and checkpoint `75` improves teacher-train constrained
sequence/arguments to `11/12` / `7/12`. But no checkpoint improves raw
model-only sequence or arguments, and checkpoint `100` shows raw-valid formatting
drift (`1/8` public, `0/12` teacher-train). Treat checkpoint `75` as the current
best scaling candidate for a broader eval/data expansion, not as a promoted
model.

### Synthetic-48 Value-Span Label-Only Scaling Probe

Date: 2026-06-27.

The next probe scaled the best non-regressing 16-row recipe to a larger clean
grounded data slice instead of treating the 16-row result as final evidence.

Active checkpoint-275 was first evaluated on a 48-row synthetic one-call slice:

- output:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_synthetic_onecall48_eval96_modelrepair_max1`
- input:
  `data/toolcall_eval/synthetic_onecall_smoke.jsonl`
- raw exact sequence / arguments: `16/48` / `11/48`
- constrained exact sequence / arguments: `48/48` / `44/48`
- model-repair exact sequence / arguments: `25/48` / `25/48`
- full-context eval runtime: `778.684` seconds

The grounded span-fill builder accepted the exact constrained rows into:

- dataset:
  `data/qwen35_9b_toolcall_grounded_spanfill_synthetic_onecall48_b1024_curriculum`
- manifest:
  `data/qwen35_9b_toolcall_grounded_spanfill_synthetic_onecall48_b1024_curriculum/train_agentic_mix.manifest`
- accepted rows: `44`
- rejected rows: `0`
- skipped rows: `4` not grounded exact
- chosen length min / p50 / p90 / max: `536 / 689 / 779 / 851`
- chosen kept labels min / p50 / p90 / max: `27 / 32 / 46 / 53`
- zero or partial labels after truncation: `0`

Training:

- output:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_synth48_from_ckpt275_b1024_lr1e6_step75`
- starting adapter:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model`
- dataset:
  `data/qwen35_9b_toolcall_grounded_spanfill_synthetic_onecall48_b1024_curriculum`
- block size: `1024`
- max steps: `75`
- max train samples: `44`
- LR: `1e-6`
- saved checkpoints: `25`, `44`, `50`, `75`
- value-span label-only: enabled
- value-span mask probability: `1.0`
- first debug batch:
  - pre-MDM assistant labels: `46`
  - value-only labels: `17`
  - forced value labels: `17`
  - post-MDM labels: `[17, 0]`
- train loss: `0.3349521501859029`
- runtime: `186.9101` seconds

Checkpoint-75 one-call eval:

| Slice | Raw valid | Raw seq | Raw args | Constrained seq | Constrained args | Model-repair seq | Model-repair args |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| public one-call | 3/8 | 4/8 | 3/8 | 8/8 | 8/8 | 5/8 | 3/8 |
| teacher train | 2/12 | 2/12 | 2/12 | 10/12 | 7/12 | 5/12 | 4/12 |
| teacher heldout | 2/8 | 2/8 | 1/8 | 7/8 | 5/8 | 3/8 | 1/8 |

Decision: do not promote checkpoint `75`. This is the first training branch that
improves raw/model-only behavior versus active checkpoint-275 on public one-call
(`4/8` sequence and `3/8` arguments versus `3/8` and `2/8`) and teacher-heldout
(`2/8` and `1/8` versus `1/8` and `0/8`). It also improves teacher-train
constrained arguments to `7/12`. But it regresses heldout constrained recovery
from active `8/8` / `6/8` to `7/8` / `5/8`, so the active checkpoint remains
checkpoint-275.

Checkpoint-50 follow-up:

| Slice | Raw valid | Raw seq | Raw args | Constrained seq | Constrained args | Model-repair seq | Model-repair args |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| public one-call | 3/8 | 3/8 | 2/8 | 8/8 | 8/8 | 5/8 | 3/8 |
| teacher train | 2/12 | 2/12 | 2/12 | 10/12 | 6/12 | 4/12 | 4/12 |
| teacher heldout | 2/8 | 1/8 | 0/8 | 7/8 | 5/8 | 3/8 | 1/8 |

Checkpoint `50` is also not promotable. It preserves public constrained
recovery and ties active raw on public/train, but it loses the checkpoint-75 raw
gain and still regresses teacher-heldout constrained recovery to `7/8` / `5/8`.

Interpretation: scaling the cleaner value-span label-only objective from 16 rows
to 44 cleaner rows produced the first raw gain. That is a positive scaling
signal, not a finished recipe. The raw-gain checkpoint (`75`) and the more
conservative checkpoint (`50`) both fail the same heldout constrained guard. The
next training branch should mix the synthetic-48 rows with a small
replay/preservation set from the original grounded teacher-train curriculum
rather than trying to promote either single checkpoint.

### Synthetic-48 Plus Teacher-Train Replay Mix

Date: 2026-06-27.

Built an explicit replay/preservation mix to test whether the raw gains from the
synthetic-48 data can be retained while restoring protected constrained metrics.

Builder:

- script:
  `scripts/build_toolcall_grounded_replay_mix.py`
- output dataset:
  `data/qwen35_9b_toolcall_grounded_spanfill_synth48_replay_teacher2_b1024_curriculum`
- synthetic source:
  `data/qwen35_9b_toolcall_grounded_spanfill_synthetic_onecall48_b1024_curriculum`
- replay source:
  `data/qwen35_9b_toolcall_grounded_spanfill_teacher_train_b1024_curriculum`
- synthetic repeat: `1`
- replay repeat: `2`
- total rows: `76`
- mix counts: `44` synthetic grounded rows, `32` teacher-train replay rows
- zero or partial labels after truncation: `0`
- length min / p50 / p90 / max: `536 / 714 / 979 / 979`
- kept labels min / p50 / p90 / max: `24 / 33 / 315 / 315`

Training:

- output:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_synth48_replay_teacher2_from_ckpt275_b1024_lr1e6_step100`
- starting adapter:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model`
- block size: `1024`
- max steps: `100`
- max train samples: `76`
- LR: `1e-6`
- saved/evaluated checkpoints: `50`, `75`, `100`
- value-span label-only: enabled
- value-span mask probability: `1.0`
- train loss: `0.3505465775728226`
- train runtime: `248.9709` seconds

One-call checkpoint sweep:

| Checkpoint | Slice | Raw valid | Raw seq | Raw args | Constrained seq | Constrained args | Model-repair seq | Model-repair args |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | public one-call | 3/8 | 4/8 | 3/8 | 8/8 | 7/8 | 5/8 | 3/8 |
| 50 | teacher train | 2/12 | 2/12 | 2/12 | 11/12 | 5/12 | 6/12 | 5/12 |
| 50 | teacher heldout | 2/8 | 2/8 | 1/8 | 7/8 | 5/8 | 3/8 | 1/8 |
| 75 | public one-call | 1/8 | 3/8 | 2/8 | 8/8 | 8/8 | 4/8 | 2/8 |
| 75 | teacher train | 0/12 | 2/12 | 2/12 | 10/12 | 6/12 | 5/12 | 4/12 |
| 75 | teacher heldout | 2/8 | 1/8 | 0/8 | 7/8 | 5/8 | 3/8 | 1/8 |
| 100 | public one-call | 2/8 | 2/8 | 1/8 | 8/8 | 8/8 | 5/8 | 3/8 |
| 100 | teacher train | 1/12 | 1/12 | 1/12 | 10/12 | 7/12 | 5/12 | 5/12 |
| 100 | teacher heldout | 2/8 | 1/8 | 0/8 | 8/8 | 6/8 | 4/8 | 2/8 |

Decision: do not promote any replay-mix checkpoint. The replay mix did what it
was designed to test: it separates raw-gain and preservation behavior. Checkpoint
`50` keeps the synthetic raw gain on public (`4/8` / `3/8`) and heldout
(`2/8` / `1/8`), but still regresses constrained public arguments and heldout
constrained recovery. Checkpoint `100` restores heldout constrained recovery to
the active top line (`8/8` / `6/8`) and improves teacher-train constrained
arguments to `7/12`, but raw public/train/heldout all regress. Checkpoint `75`
is mostly a formatting-drift preservation point and is not useful.

Interpretation: simple replay is not enough. The objective can move raw
model-only copying, and replay can restore some constrained recovery, but one CE
adapter is trading these off. Next trainer work should either add an explicit
retention/anti-regression term or split the problem: keep the generator at the
raw-gain checkpoint while training a constrained repair/span-fill head, or move
the retention pressure into decoding rather than the same value-span CE update.

### Staged Retention From Replay Checkpoint-50

Date: 2026-06-27.

Tested a staged schedule instead of one mixed run: start from the raw-gain
replay-mix checkpoint `50`, then continue only on the original grounded
teacher-train retention rows at lower LR.

Training:

- output:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_staged_retention_from_synth48replay_ckpt50_b1024_lr5e7_step50`
- starting adapter:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_synth48_replay_teacher2_from_ckpt275_b1024_lr1e6_step100/checkpoint-50/adapter_model`
- dataset:
  `data/qwen35_9b_toolcall_grounded_spanfill_teacher_train_b1024_curriculum`
- block size: `1024`
- max steps: `50`
- max train samples: `16`
- LR: `5e-7`
- saved/evaluated checkpoints: `24`, `40`, `50`
- value-span label-only: enabled
- value-span mask probability: `1.0`
- train loss: `0.46239189386367796`
- train runtime: `124.9199` seconds

One-call checkpoint sweep:

| Checkpoint | Slice | Raw valid | Raw seq | Raw args | Constrained seq | Constrained args | Model-repair seq | Model-repair args |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 24 | public one-call | 1/8 | 4/8 | 3/8 | 8/8 | 8/8 | 5/8 | 3/8 |
| 24 | teacher train | 0/12 | 2/12 | 2/12 | 11/12 | 6/12 | 5/12 | 3/12 |
| 24 | teacher heldout | 2/8 | 2/8 | 1/8 | 8/8 | 6/8 | 4/8 | 2/8 |
| 40 | public one-call | 2/8 | 3/8 | 2/8 | 8/8 | 8/8 | 5/8 | 3/8 |
| 40 | teacher train | 1/12 | 2/12 | 2/12 | 10/12 | 6/12 | 6/12 | 4/12 |
| 40 | teacher heldout | 2/8 | 1/8 | 0/8 | 8/8 | 6/8 | 4/8 | 2/8 |
| 50 | public one-call | 1/8 | 1/8 | 0/8 | 8/8 | 8/8 | 5/8 | 3/8 |
| 50 | teacher train | 0/12 | 0/12 | 0/12 | 12/12 | 7/12 | 5/12 | 5/12 |
| 50 | teacher heldout | 2/8 | 1/8 | 0/8 | 8/8 | 6/8 | 4/8 | 2/8 |

Decision: checkpoint `24` is the first staged candidate worth broader eval, but
it is not yet the active promoted model. It improves public raw from active
checkpoint-275 `3/8` sequence and `2/8` arguments to `4/8` and `3/8`, preserves
public constrained `8/8` / `8/8`, preserves teacher-heldout constrained
`8/8` / `6/8`, and improves teacher-train constrained sequence from active
`10/12` to `11/12` while tying teacher-train constrained arguments at `6/12`.

Risks:

- raw valid JSON remains weak (`1/8` public and `0/12` teacher-train), so this is
  a model-behavior gain, not a formatting solution
- checkpoint `40` loses the public raw gain and checkpoint `50` collapses raw
  sequence exactness, so the retention schedule overshoots

Interpretation: staged retention is better than the single replay mix because it
finds a point where raw public gains and protected constrained heldout recovery
coexist. The effect is transient, so the next scaling experiment should not be
"more of the same" at checkpoint `50`. Use checkpoint `24` for broader
multi-call/tool-result eval, then run a gentler dose curve around this region:
lower LR, more frequent checkpoints, and either interleaved synthetic/retention
minibatches or an explicit anti-regression term.

### Staged Checkpoint-24 Broad Agentic Eval

Date: 2026-06-27.

Ran the broader public multi-call plus tool-result gate on staged checkpoint
`24`:

- output:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_staged_retention_ckpt24_broad_eval96_modelrepair_agentic`
- adapter:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_staged_retention_from_synth48replay_ckpt50_b1024_lr5e7_step50/checkpoint-24/adapter_model`

Summary:

| Slice | Raw valid | Raw seq | Raw args | Projected/constrained seq | Projected/constrained args | Model-repair seq | Model-repair args |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| public one-call | 1/8 | 4/8 | 3/8 | 8/8 | 8/8 | 5/8 | 3/8 |
| teacher train one-call | 0/12 | 2/12 | 2/12 | 11/12 | 6/12 | 5/12 | 3/12 |
| teacher heldout one-call | 2/8 | 2/8 | 1/8 | 8/8 | 6/8 | 4/8 | 2/8 |
| public multi-call, constrained draft | 3/12 | 1/12 | 0/12 | 7/12 | 5/12 | 1/12 | 1/12 |
| public multi-call, contextual projection | 12/12 | 7/12 | 5/12 | 7/12 | 6/12 | n/a | n/a |
| public multi-call, guarded sequence planner | 12/12 | 7/12 | 6/12 | 11/12 | 9/12 | n/a | n/a |
| synthetic tool-result | 1/10 | 6/10 | 4/10 | 10/10 | 9/10 | 2/10 | 2/10 |
| OpenAI-style tool-result | 4/10 | 5/10 | 5/10 | 10/10 | 8/10 | 5/10 | 5/10 |

Decision: checkpoint `24` remains a useful candidate, but it does not replace
active checkpoint-275 globally. It keeps the one-call raw gain and passes the
heldout constrained guard. It also keeps public multi-call guarded sequence
planner sequence at active `11/12`, but exact arguments are `9/12` versus the
active `10/12`. Text-compatible synthetic tool-result improves constrained
arguments to `9/10` versus active `8/10`, but OpenAI-style tool-result regresses
from active `9/10` constrained arguments to `8/10`.

Interpretation: staged checkpoint `24` is the best generator-side signal so far,
not a finished agentic checkpoint. It should be used as the seed for the next
gentle scaling run, with OpenAI-style tool-result replay/anti-regression kept in
the mix. A promotion over checkpoint-275 requires preserving both active
multi-call planner arguments (`10/12`) and OpenAI-style tool-result constrained
arguments (`9/10`) while keeping the one-call raw improvement.

### Checkpoint-24 Anti-Regression Mix

Date: 2026-06-27.

Built a broader anti-regression curriculum to test whether checkpoint `24` could
keep its one-call raw gain while protecting OpenAI-style tool-result and
multi-call planner behavior.

Builder:

- script:
  `scripts/build_toolcall_checkpoint24_antiregression_mix.py`
- output dataset:
  `data/qwen35_9b_toolcall_checkpoint24_antiregression_b1024_curriculum`
- accepted rows: `127`
- rejected rows: `0`
- mix counts:
  - synthetic grounded span-fill: `44`
  - teacher-train grounded retention: `32`
  - sequence-planner compact retention: `21`
  - synthetic text tool-result retention: `10`
  - native OpenAI-style tool-result retention: `20`
- length min / p50 / p90 / max: `530 / 701 / 846 / 980`
- kept labels min / p50 / p90 / max: `24 / 55 / 136 / 315`

Implementation note:

- `scripts/fastdllm_value_copy_token_ids.py` now extracts scalar argument values
  from native `assistant.tool_calls` as well as text `<tool_call>` blocks. This
  keeps value-span label-only training meaningful for OpenAI-style tool-call
  rows.

Training:

- output:
  `runs/fastdllm_qwen35_9b_toolcall_checkpoint24_antiregression_mix_from_staged24_b1024_lr2e7_step80`
- starting adapter:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_staged_retention_from_synth48replay_ckpt50_b1024_lr5e7_step50/checkpoint-24/adapter_model`
- block size: `1024`
- max steps: `80`
- LR: `2e-7`
- save steps: `10`
- value-span label-only: enabled
- value-span mask probability: `1.0`
- train loss: `0.4683162711560726`
- train runtime: `199.4963` seconds

One-call dose sweep:

The planned sweep covered checkpoints `10`, `20`, `40`, and `80`, but it was
stopped after checkpoint `40`. Checkpoints `10`, `20`, and `40` had already
shown that the continuation erased the checkpoint-24 public raw gain; checkpoint
`40` also regressed heldout constrained recovery.

| Checkpoint | Slice | Raw valid | Raw seq | Raw args | Constrained seq | Constrained args | Model-repair seq | Model-repair args |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | public one-call | 1/8 | 3/8 | 2/8 | 8/8 | 8/8 | 3/8 | 1/8 |
| 10 | teacher train | 0/12 | 2/12 | 2/12 | 10/12 | 6/12 | 4/12 | 3/12 |
| 10 | teacher heldout | 2/8 | 1/8 | 0/8 | 8/8 | 6/8 | 3/8 | 1/8 |
| 20 | public one-call | 1/8 | 3/8 | 2/8 | 8/8 | 8/8 | 4/8 | 2/8 |
| 20 | teacher train | 0/12 | 2/12 | 2/12 | 10/12 | 6/12 | 3/12 | 2/12 |
| 20 | teacher heldout | 2/8 | 1/8 | 0/8 | 8/8 | 6/8 | 4/8 | 2/8 |
| 40 | public one-call | 2/8 | 3/8 | 2/8 | 8/8 | 8/8 | 4/8 | 2/8 |
| 40 | teacher train | 1/12 | 2/12 | 2/12 | 11/12 | 7/12 | 5/12 | 5/12 |
| 40 | teacher heldout | 2/8 | 1/8 | 0/8 | 7/8 | 5/8 | 3/8 | 1/8 |

Decision: do not promote this anti-regression continuation. The direct
same-adapter mix improves the teacher-train constrained argument line by
checkpoint `40`, but it removes the checkpoint-24 raw public gain immediately
and eventually damages the heldout constrained guard. This is useful evidence
that broad anti-regression rows should not be pushed through the same
value-span-label-only generator update.

Next implication:

- keep staged checkpoint `24` as the best generator-side seed
- do not continue broad anti-regression CE in the same adapter
- move protection into a separate repair/projection path, two-adapter routing,
  or a much smaller sidecar objective evaluated first on OpenAI tool-result and
  multi-call projection, not on the main generator

### Split-Route Sidecar Scorecard

Date: 2026-06-27.

Generated `qwen35_9b_split_route_sidecar_scorecard.md` from the existing
checkpoint-24 and checkpoint-275 eval artifacts. This is a routing/protection
target, not a promoted single adapter.

The scorecard writer also emits machine-readable gate artifacts:

- `runs/qwen35_9b_split_route_sidecar_eval/route_scorecard.json`
- `runs/qwen35_9b_split_route_sidecar_eval/route_scorecard.tsv`
- `runs/qwen35_9b_split_route_sidecar_eval/route_manifest.json`

The current executable route verdict is `PASS` across all six routed slices.
Run `scripts/write_qwen35_split_route_sidecar_scorecard.py --check` to fail
nonzero on any route-gate regression.

The manifest is the concrete handoff for the first router/sidecar
implementation: it records the shared base model, staged checkpoint-24 generator
adapter, active checkpoint-275 protection adapter, per-slice input case files,
routed summaries, and post-processing chains.

Replay runner:

- script: `scripts/run_qwen35_split_route_sidecar_manifest.py`
- check command:
  `.venv-fastdllm/bin/python scripts/run_qwen35_split_route_sidecar_manifest.py --check-only --strict-replayable`
- plan command:
  `.venv-fastdllm/bin/python scripts/run_qwen35_split_route_sidecar_manifest.py --strict-replayable`
- plan JSON:
  `runs/qwen35_9b_split_route_sidecar_eval/replay_plan/route_runner_plan.json`
- plan shell:
  `runs/qwen35_9b_split_route_sidecar_eval/replay_plan/route_runner_plan.sh`
- current validation: `6` routes, `10` replayable steps, `0` unknown steps
- output verification command:
  `.venv-fastdllm/bin/python scripts/run_qwen35_split_route_sidecar_manifest.py --verify-outputs --plan-json runs/qwen35_9b_split_route_sidecar_eval/replay_plan/route_runner_plan.json`
- historical-output verification:
  `runs/qwen35_9b_split_route_sidecar_eval/historical_verify_plan/route_runner_plan_verification.json`
- historical verification result: `6` records, `0` missing summaries,
  `0` failed records
- partial execution command:
  `.venv-fastdllm/bin/python scripts/run_qwen35_split_route_sidecar_manifest.py --strict-replayable --slice public_one_call --out-root runs/qwen35_9b_split_route_sidecar_eval/live_smoke_public_onecall --execute`
- live public-onecall execution:
  `runs/qwen35_9b_split_route_sidecar_eval/live_smoke_public_onecall/route_runner_execution.json`
- live public-onecall verification:
  `runs/qwen35_9b_split_route_sidecar_eval/live_smoke_public_onecall/route_runner_plan_verification.json`
- live public-onecall result: raw `4/8` sequence, raw `3/8` arguments, and
  protected `8/8` / `8/8`
- active-protection execution command:
  `.venv-fastdllm/bin/python scripts/run_qwen35_split_route_sidecar_manifest.py --strict-replayable --slice openai_style_tool_result --out-root runs/qwen35_9b_split_route_sidecar_eval/live_smoke_openai_toolresult --execute`
- live OpenAI-style tool-result execution:
  `runs/qwen35_9b_split_route_sidecar_eval/live_smoke_openai_toolresult/route_runner_execution.json`
- live OpenAI-style tool-result verification:
  `runs/qwen35_9b_split_route_sidecar_eval/live_smoke_openai_toolresult/route_runner_plan_verification.json`
- live OpenAI-style tool-result result: raw `6/10` sequence, raw `6/10`
  arguments, and protected `10/10` sequence / `9/10` arguments
- live public multi-call planner command:
  `.venv-fastdllm/bin/python scripts/run_qwen35_split_route_sidecar_manifest.py --strict-replayable --slice public_multi_call_planner --out-root runs/qwen35_9b_split_route_sidecar_eval/live_smoke_public_multicall_planner --execute`
- live public multi-call planner execution:
  `runs/qwen35_9b_split_route_sidecar_eval/live_smoke_public_multicall_planner/route_runner_execution.json`
- live public multi-call planner verification:
  `runs/qwen35_9b_split_route_sidecar_eval/live_smoke_public_multicall_planner/route_runner_plan_verification.json`
- live public multi-call planner result: raw `7/12` sequence, raw `7/12`
  arguments, and protected `11/12` sequence / `10/12` arguments after
  sequence-preserving rescore, contextual projection, and sequence-planner
  projection
- live synthetic text tool-result command:
  `.venv-fastdllm/bin/python scripts/run_qwen35_split_route_sidecar_manifest.py --strict-replayable --slice synthetic_text_tool_result --out-root runs/qwen35_9b_split_route_sidecar_eval/live_smoke_synthetic_text_toolresult --execute`
- live synthetic text tool-result execution:
  `runs/qwen35_9b_split_route_sidecar_eval/live_smoke_synthetic_text_toolresult/route_runner_execution.json`
- live synthetic text tool-result verification:
  `runs/qwen35_9b_split_route_sidecar_eval/live_smoke_synthetic_text_toolresult/route_runner_plan_verification.json`
- live synthetic text tool-result result: raw `6/10` sequence, raw `4/10`
  arguments, and protected `10/10` sequence / `9/10` arguments
- live teacher one-call command:
  `.venv-fastdllm/bin/python scripts/run_qwen35_split_route_sidecar_manifest.py --strict-replayable --slice teacher_train_one_call --slice teacher_heldout_one_call --out-root runs/qwen35_9b_split_route_sidecar_eval/live_smoke_teacher_onecall --execute`
- live teacher one-call execution:
  `runs/qwen35_9b_split_route_sidecar_eval/live_smoke_teacher_onecall/route_runner_execution.json`
- live teacher one-call verification:
  `runs/qwen35_9b_split_route_sidecar_eval/live_smoke_teacher_onecall/route_runner_plan_verification.json`
- live teacher-train result: raw `2/12` sequence, raw `2/12` arguments, and
  protected `11/12` sequence / `6/12` arguments
- live teacher-heldout result: raw `2/8` sequence, raw `1/8` arguments, and
  protected `8/8` sequence / `6/8` arguments
- live route coverage: all `6` split-route scorecard lanes now have verified
  live replay artifacts

The routed target is:

- route public/teacher one-call prompts to staged checkpoint `24`
- route text-compatible synthetic tool-result prompts to staged checkpoint `24`
- route public multi-call planner prompts through active checkpoint-275's
  guarded planner/projection path
- route OpenAI-style tool-result prompts through active checkpoint-275's
  projection path

The resulting routed scorecard keeps the checkpoint-24 public one-call raw gain
at `4/8` exact sequence and `3/8` exact arguments while keeping public
constrained recovery at `8/8` / `8/8`. It also keeps active checkpoint-275's
multi-call protected `11/12` / `10/12`, active OpenAI-style tool-result
protected `10/10` / `9/10`, and checkpoint-24 text tool-result protected
`10/10` / `9/10`.

Decision: the next implementation should be a runtime router, sidecar
repair/projection path, or separately trained sidecar objective. The gate for
that implementation is to match this routed scorecard before attempting another
broad same-adapter generator continuation.

### Evidence Policy And Scaling Context

Small `1/5/25`-step runs in this project are fit, wiring, and regression gates.
They can reject settings that immediately damage protected metrics, but they are
not enough to declare a training mechanism dead.

The stronger evidence ladder is:

1. One-step fit gate: labels survive tokenization/windowing, loss is nonzero,
   memory fits, and no obvious protected eval regression appears.
2. Dose curve: lower LR, saved checkpoints, identical eval slices, and a check
   for formatting drift.
3. Data-scale probe: same objective on a larger clean slice, with train/heldout
   separated.
4. Promotion gate: improve at least one raw/model-only metric while preserving
   active constrained/projected top lines.

This policy is consistent with LLM scaling work: Kaplan et al. show smooth
power-law behavior with model size, data, and compute; Hoffmann et al. show that
compute-optimal training requires scaling model and data together. Those results
do not prove this diffusion recipe will work, but they do argue against reading a
tiny undertrained probe as a final negative result.

Primary references:

- `https://arxiv.org/abs/2001.08361`
- `https://arxiv.org/abs/2203.15556`

## Artifacts

- builder:
  `scripts/build_toolcall_grounded_spanfill_curriculum.py`
- block-896 dataset:
  `data/qwen35_9b_toolcall_grounded_spanfill_teacher_train_curriculum`
- block-1024 dataset:
  `data/qwen35_9b_toolcall_grounded_spanfill_teacher_train_b1024_curriculum`
- one-step adapter:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_spanfill_from_ckpt275_b1024_step1`
- eval outputs:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_spanfill_from_ckpt275_b1024_step1_eval96_modelrepair_max1`
- value-span-mask one-step adapter:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespanmask_from_ckpt275_b1024_step1`
- value-span-mask eval outputs:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespanmask_from_ckpt275_b1024_step1_eval96_modelrepair_max1`
- value-span-only 25-step adapter:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespanmask_only_from_ckpt275_b1024_step25`
- value-span-only 25-step eval outputs:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespanmask_only_from_ckpt275_b1024_step25_eval96_modelrepair_max1`
- value-span label-only one-step adapter:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_from_ckpt275_b1024_step1`
- value-span label-only one-step eval outputs:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_from_ckpt275_b1024_step1_eval96_modelrepair_max1`
- value-span label-only 25-step adapter:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_from_ckpt275_b1024_step25`
- value-span label-only 25-step eval outputs:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_from_ckpt275_b1024_step25_eval96_modelrepair_max1`
- value-span label-only 5-step adapter:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_from_ckpt275_b1024_step5`
- value-span label-only 5-step eval outputs:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_from_ckpt275_b1024_step5_eval96_modelrepair_max1`
- value-span label-only LR-1e-6 sweep adapter:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_from_ckpt275_b1024_lr1e6_step100`
- value-span label-only LR-1e-6 checkpoint archive:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_from_ckpt275_b1024_lr1e6_step100_checkpoint_archive`
- value-span label-only LR-1e-6 checkpoint sweep outputs:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_from_ckpt275_b1024_lr1e6_step100_checkpoint_sweep_eval96_modelrepair_max1_onecall`
- synthetic-48 active-checkpoint eval outputs:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_synthetic_onecall48_eval96_modelrepair_max1`
- synthetic-48 grounded span-fill dataset:
  `data/qwen35_9b_toolcall_grounded_spanfill_synthetic_onecall48_b1024_curriculum`
- synthetic-48 value-span label-only adapter:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_synth48_from_ckpt275_b1024_lr1e6_step75`
- synthetic-48 checkpoint-75 one-call eval outputs:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_synth48_from_ckpt275_b1024_lr1e6_step75_checkpoint75_eval96_modelrepair_max1_onecall`
- synthetic-48 checkpoint-50 one-call eval outputs:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_synth48_from_ckpt275_b1024_lr1e6_step75_checkpoint50_eval96_modelrepair_max1_onecall`
- replay-mix builder:
  `scripts/build_toolcall_grounded_replay_mix.py`
- replay-mix dataset:
  `data/qwen35_9b_toolcall_grounded_spanfill_synth48_replay_teacher2_b1024_curriculum`
- replay-mix adapter:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_synth48_replay_teacher2_from_ckpt275_b1024_lr1e6_step100`
- replay-mix checkpoint sweep outputs:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_synth48_replay_teacher2_from_ckpt275_b1024_lr1e6_step100_checkpoint_sweep_eval96_modelrepair_max1_onecall`
- staged-retention adapter:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_staged_retention_from_synth48replay_ckpt50_b1024_lr5e7_step50`
- staged-retention checkpoint sweep outputs:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_staged_retention_from_synth48replay_ckpt50_b1024_lr5e7_step50_checkpoint_sweep_eval96_modelrepair_max1_onecall`
- staged checkpoint-24 broad eval outputs:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_staged_retention_ckpt24_broad_eval96_modelrepair_agentic`
- checkpoint-24 anti-regression dataset:
  `data/qwen35_9b_toolcall_checkpoint24_antiregression_b1024_curriculum`
- checkpoint-24 anti-regression adapter:
  `runs/fastdllm_qwen35_9b_toolcall_checkpoint24_antiregression_mix_from_staged24_b1024_lr2e7_step80`
- checkpoint-24 anti-regression one-call sweep outputs:
  `runs/fastdllm_qwen35_9b_toolcall_checkpoint24_antiregression_mix_from_staged24_b1024_lr2e7_step80_onecall_sweep_eval96_modelrepair_max1`
- train log:
  `logs/qwen35_grounded_spanfill_ckpt275_b1024_step1_train.log`
- eval log:
  `logs/qwen35_grounded_spanfill_ckpt275_b1024_step1_onecall_eval.log`
- value-span-mask train log:
  `logs/qwen35_grounded_valuespanmask_ckpt275_b1024_step1_train.log`
- value-span-mask eval log:
  `logs/qwen35_grounded_valuespanmask_ckpt275_b1024_step1_onecall_eval.log`
- value-span-only 25-step train log:
  `logs/qwen35_grounded_valuespanmask_only_ckpt275_b1024_step25_train.log`
- value-span-only 25-step eval log:
  `logs/qwen35_grounded_valuespanmask_only_ckpt275_b1024_step25_onecall_eval.log`
- value-span label-only one-step train log:
  `logs/qwen35_grounded_valuespan_labelonly_ckpt275_b1024_step1_train.log`
- value-span label-only one-step eval log:
  `logs/qwen35_grounded_valuespan_labelonly_ckpt275_b1024_step1_onecall_eval.log`
- value-span label-only 25-step train log:
  `logs/qwen35_grounded_valuespan_labelonly_ckpt275_b1024_step25_train.log`
- value-span label-only 25-step eval log:
  `logs/qwen35_grounded_valuespan_labelonly_ckpt275_b1024_step25_onecall_eval.log`
- value-span label-only 5-step train log:
  `logs/qwen35_grounded_valuespan_labelonly_ckpt275_b1024_step5_train.log`
- value-span label-only 5-step eval log:
  `logs/qwen35_grounded_valuespan_labelonly_ckpt275_b1024_step5_onecall_eval.log`
- value-span label-only LR-1e-6 sweep train log:
  `logs/qwen35_grounded_valuespan_labelonly_ckpt275_b1024_lr1e6_step100_train.log`
- value-span label-only LR-1e-6 checkpoint sweep log:
  `logs/qwen35_grounded_valuespan_labelonly_ckpt275_b1024_lr1e6_step100_checkpoint_sweep_onecall.log`
- synthetic-48 build log:
  `logs/qwen35_grounded_spanfill_synthetic_onecall48_b1024_build.log`
- synthetic-48 train log:
  `logs/qwen35_grounded_valuespan_labelonly_synth48_ckpt275_b1024_lr1e6_step75_train.log`
- synthetic-48 checkpoint-75 eval log:
  `logs/qwen35_grounded_valuespan_labelonly_synth48_ckpt275_b1024_lr1e6_step75_checkpoint75_onecall_eval.log`
- synthetic-48 checkpoint-50 eval log:
  `logs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_synth48_from_ckpt275_b1024_lr1e6_step75_checkpoint50_eval96_modelrepair_max1_onecall_checkpoint-50_onecall.log`
- replay-mix build log:
  `logs/qwen35_grounded_spanfill_synth48_replay_teacher2_b1024_build.log`
- replay-mix train log:
  `logs/qwen35_grounded_valuespan_labelonly_synth48_replay_teacher2_ckpt275_b1024_lr1e6_step100_train.log`
- replay-mix checkpoint sweep logs:
  `logs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_synth48_replay_teacher2_from_ckpt275_b1024_lr1e6_step100_checkpoint_sweep_eval96_modelrepair_max1_onecall_checkpoint-*.log`
- staged-retention train log:
  `logs/qwen35_grounded_valuespan_labelonly_staged_retention_from_synth48replay_ckpt50_b1024_lr5e7_step50_train.log`
- staged-retention checkpoint sweep logs:
  `logs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_staged_retention_from_synth48replay_ckpt50_b1024_lr5e7_step50_checkpoint_sweep_eval96_modelrepair_max1_onecall_checkpoint-*.log`
- staged checkpoint-24 broad eval log:
  `logs/qwen35_grounded_valuespan_labelonly_staged_retention_ckpt24_broad_eval96_modelrepair_agentic.log`
- checkpoint-24 anti-regression build log:
  `logs/qwen35_checkpoint24_antiregression_b1024_build.log`
- checkpoint-24 anti-regression train log:
  `logs/qwen35_checkpoint24_antiregression_mix_from_staged24_b1024_lr2e7_step80_train.log`
- checkpoint-24 anti-regression one-call sweep logs:
  `logs/fastdllm_qwen35_9b_toolcall_checkpoint24_antiregression_mix_from_staged24_b1024_lr2e7_step80_onecall_sweep_eval96_modelrepair_max1_checkpoint-*.log`
