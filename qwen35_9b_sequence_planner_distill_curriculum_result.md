# Qwen3.5-9B Sequence-Planner Distill Curriculum Result

Date: 2026-06-27.

## Summary

Built a train-only multi-call sequence-planner curriculum and ran a one-step
Qwen3.5-9B Fast-DLLM QLoRA gate from the active checkpoint-275 adapter.

This is a data-path and training-load gate, not a promoted checkpoint.

## GDN Architecture Check

The online config check confirms the current target should stay on the
Qwen3.5/Qwen3.6 GDN-family path:

- Qwen3.5-9B raw config:
  `Qwen3_5ForConditionalGeneration`, `model_type: qwen3_5`, text model
  `model_type: qwen3_5_text`, 32 layers, repeating
  `linear_attention, linear_attention, linear_attention, full_attention`.
- Qwen3.6-27B raw config:
  same Qwen3.5-family architecture, 64 layers, same 3:1
  linear-attention/full-attention pattern.
- Qwen2.5-7B raw config:
  `Qwen2ForCausalLM`, `model_type: qwen2`, 28 layers, no `layer_types`.

Conclusion: Qwen2.5 remains useful only as a cheap objective/sampler lab.
Qwen3.5-9B is the right first real student because it exercises the same
Gated DeltaNet/full-attention hybrid family as the Qwen3.6-27B target.

Sources:

- `https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/config.json`
- `https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/config.json`
- `https://huggingface.co/Qwen/Qwen2.5-7B/raw/main/config.json`
- `qwen35_gdn_vs_qwen25_research.md`

## Curriculum Build

New script:

```text
scripts/build_toolcall_sequence_planner_distill_curriculum.py
```

Output:

```text
data/qwen35_9b_toolcall_sequence_planner_distill_curriculum/
```

The builder uses only public training multi-call rows:

```text
data/fastdllm_toolcall_train/train_toolcall.json
```

It does not use eval rows. The deterministic planner is used as a selector:
if the request/schema planner predicts the gold tool-call sequence, the
training target is the train gold assistant call payload. Strict
exact-argument planner mode remains available through `--accept-mode
exact_arguments`, but the default is `exact_sequence` because that matches the
teacher-forcing objective better.

Manifest result:

```text
public multi-call train records: 56
planner exact sequence:         27
planner exact arguments:        2
selected before label gate:     27
accepted after 896-token gate:  13
label rejected:                 14
no eval leakage:                true
```

Accepted-row token audit:

```text
chosen length:      min 643, p50 825, p90 871, max 896
kept label tokens:  min 66,  p50 105, p90 120, max 122
zero/partial labels among accepted rows: 0 / 0
```

This confirms a real bottleneck: long tool schemas plus long multi-call
requests still discard many useful rows at block size `896`.

## One-Step QLoRA Gate

Command profile:

```text
systemd-run --user --scope
MemoryMax=28G
MemorySwapMax=4G
base model: models/qwen3.5-9b-fastdllm-init
starting adapter: runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model
dataset: data/qwen35_9b_toolcall_sequence_planner_distill_curriculum
block size: 896
max steps: 1
max train samples: 13
LoRA target modules: q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj
argument-span loss weight: 1.5
```

Output:

```text
runs/fastdllm_qwen35_9b_toolcall_sequence_planner_distill_from_ckpt275_step1_gate
logs/fastdllm_qwen35_9b_toolcall_sequence_planner_distill_from_ckpt275_step1_gate.log
```

Training result:

```text
train loss:     1.42819082736969
runtime:        2.809 s
samples/sec:    0.356
steps/sec:      0.356
train samples:  13
```

The run produced a root-level adapter artifact, so the load/train/save path is
valid.

## Interpretation

The strict exact-argument planner target is too sparse for model training:
only `2/56` train multi-call rows pass, and only `1` survives the 896-token
label gate.

The sequence-selected gold-target path is viable as a small curriculum
component: `13` fully labeled rows survive at block size `896`, and the
one-step QLoRA gate trains cleanly from the active adapter.

Do not promote the one-step adapter. The next useful experiment is a mixed
run that adds these sequence-planner rows at low ratio to the active
model-repair replay set, then evaluates with the checkpoint sweep and the
public multi-call contextual/sequence-planner diagnostics.

Follow-up: that low-ratio mix has now been tested and should not be promoted.
See `qwen35_9b_modelrepair_sequence_planner_mix_result.md`.

Possible ways to recover more train signal:

- shorten planner prompts now that the system message already states the task
- add a compact-tools variant for very long schemas
- test a slightly larger block size only if the 5090 memory profile stays safe
- keep deterministic sequence planning as a decoding/projection lane while the
  model learns gold multi-call payload formatting by teacher forcing

## Compact-Schema Recovery Probe

Date: 2026-06-27.

After the first low-ratio replay mix regressed, I did not launch another long
training run. Instead I tested whether the sequence-planner rows could be made
denser without partial-label leakage.

Script update:

```text
scripts/build_toolcall_sequence_planner_distill_curriculum.py
```

New options:

```text
--tool-schema-mode {full,compact,both}
--prompt-mode {instruction,request_only}
--block-size
```

The builder now also writes `label_rejected.jsonl`, so the label-retention
failures can be inspected directly.

Dataset variants:

| Variant | Block | Accepted rows | Label rejected | One-step gate loss |
| --- | ---: | ---: | ---: | ---: |
| full schema + instruction | 896 | 13 | 14 | 1.42819082736969 |
| compact schema + request only | 896 | 18 | 9 | not run |
| compact schema + request only | 1024 | 21 | 6 | 4.3694586753845215 |
| compact schema + instruction | 896 | 17 | 10 | not run |
| compact schema + instruction | 1024 | 20 | 7 | 2.6394574642181396 |

The 1024-token smoke gates both fit on the local RTX 5090 under:

```text
systemd-run --user --scope -p MemoryMax=28G -p MemorySwapMax=4G
```

Outputs:

```text
runs/fastdllm_qwen35_9b_toolcall_sequence_planner_compact_b1024_from_ckpt275_step1_gate
runs/fastdllm_qwen35_9b_toolcall_sequence_planner_compact_instruction_b1024_from_ckpt275_step1_gate
logs/fastdllm_qwen35_9b_toolcall_sequence_planner_compact_b1024_from_ckpt275_step1_gate.log
logs/fastdllm_qwen35_9b_toolcall_sequence_planner_compact_instruction_b1024_from_ckpt275_step1_gate.log
```

Interpretation:

- Compact schemas recover train rows by reducing tool-schema pressure: the best
  count is `21` fully labeled rows at block size `1024`, versus `13` for the
  original block-size-`896` full-schema path.
- Removing the planner instruction recovers one more row than keeping it, but
  the one-step loss is much worse (`4.3695` versus `2.6395`), so
  `compact + instruction + 1024` is the safer compact variant.
- The original full-schema planner gate still has the lowest one-step loss.
  Because real eval/runtime schemas are full schemas, compact rows should be
  treated as an auxiliary recovery trick, not a promoted training recipe.
- No long compact planner run is promoted from this probe. The next useful
  model-learning experiment needs a small full-schema eval gate before any
  longer sequence-planner continuation.

## Compact-Instruction Full-Schema Sweep

Date: 2026-06-27.

I ran the required full-schema checkpoint sweep on the compact-schema +
instruction one-step adapter before considering any longer compact planner
training.

Run:

```text
runs/fastdllm_qwen35_9b_toolcall_sequence_planner_compact_instruction_b1024_from_ckpt275_step1_gate/checkpoint-1
```

Sweep output:

```text
runs/fastdllm_qwen35_9b_toolcall_sequence_planner_compact_instruction_b1024_from_ckpt275_step1_gate_fullschema_sweep_eval96_modelrepair_max1/checkpoint_sweep_summary.tsv
logs/fastdllm_qwen35_9b_toolcall_sequence_planner_compact_instruction_b1024_from_ckpt275_step1_fullschema_sweep.log
```

Summary:

| Eval | Constrained/projected sequence | Constrained/projected args |
| --- | ---: | ---: |
| public one-call 8 | 8/8 | 5/8 |
| teacher train one-call 12 | 10/12 | 5/12 |
| teacher heldout one-call 8 | 8/8 | 4/8 |
| public multi-call contextual projection 12 | 7/12 | 7/12 |
| public multi-call guarded planner projection 12 | 11/12 | 10/12 |
| synthetic tool-result 10 | 10/10 | 8/10 |
| OpenAI-style tool-result 10 | 10/10 | 9/10 |

Raw public multi-call remains weak:

```text
raw valid/sequence/args: 3/12, 1/12, 0/12
constrained sequence/args: 7/12, 4/12
records with repeated calls: 2
```

Interpretation:

- The compact/instruction one-step adapter does not regress the active
  full-schema projected top lines.
- It improves teacher-heldout constrained exact arguments from the active
  documented `3/8` to `4/8`, but this is a small one-slice gain from a
  one-step adapter and is not enough to promote the checkpoint.
- Public multi-call still depends on deterministic contextual and
  sequence-planner projection for the top line; model-only raw generation is
  not improved.
- The gate clears compact/instruction as a safe row-recovery path for a future
  short continuation, but the active promoted checkpoint remains
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model`.

## Compact-Instruction 25-Step Continuation

Date: 2026-06-27.

After the one-step compact/instruction full-schema gate tied the active
projected top lines, I tried a deliberately small continuation instead of a
long run.

Training run:

```text
runs/fastdllm_qwen35_9b_toolcall_sequence_planner_compact_instruction_b1024_from_ckpt275_step25
```

Profile:

```text
start adapter: active checkpoint-275
dataset: data/qwen35_9b_toolcall_sequence_planner_distill_compact_instruction_b1024_curriculum
rows: 20
block size: 1024
max steps: 25
train loss: 2.5472710800170897
runtime: 62.5488s
```

I started the standard full-schema sweep, then stopped it after the one-call
slices because the continuation had already failed the promotion gate.

Partial sweep output:

```text
runs/fastdllm_qwen35_9b_toolcall_sequence_planner_compact_instruction_b1024_from_ckpt275_step25_fullschema_sweep_eval96_modelrepair_max1/checkpoint-25/
logs/fastdllm_qwen35_9b_toolcall_sequence_planner_compact_instruction_b1024_from_ckpt275_step25_fullschema_sweep.log
```

One-call results:

| Eval | Raw sequence | Raw args | Constrained sequence | Constrained args |
| --- | ---: | ---: | ---: | ---: |
| public one-call 8 | 2/8 | 1/8 | 8/8 | 4/8 |
| teacher train one-call 12 | 1/12 | 1/12 | 12/12 | 4/12 |
| teacher heldout one-call 8 | 1/8 | 0/8 | 8/8 | 2/8 |

Interpretation:

- Do not promote checkpoint-25.
- It regresses versus active checkpoint-275 on public one-call constrained
  arguments (`5/8` -> `4/8`) and teacher-heldout constrained arguments
  (`3/8` active, `4/8` one-step compact gate, `2/8` after 25 steps).
- It improves teacher-train constrained sequence to `12/12`, but that is not
  enough to offset the argument regressions.
- The full sweep was interrupted after the one-call failure was clear to avoid
  spending more 5090 time on a non-promotable branch.
- Conclusion: compact/instruction rows are useful as a label-retention and
  one-step safety gate, but scaling this planner-only continuation is not the
  right next move. The next experiments should be GDN-specific architecture or
  sampler ablations rather than more planner-row replay.
