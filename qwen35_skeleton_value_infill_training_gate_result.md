# Qwen3.5 9B Skeleton Value-Infill Training Gate

Date: 2026-06-28

## Question

Can the clean skeleton-conditioned value-infill corpus train as a QLoRA
continuation from the active checkpoint-275 adapter on the local RTX 5090?

This is a plumbing and first-dose gate, not promotion evidence. The target is
to verify that fixed-skeleton argument-value infill can be trained without
loader leakage, OOM, or checkpoint-save failure.

## Source Data

Training staging directory:

```text
data/qwen35_9b_skeleton_value_infill_no_public_smoke_curriculum/
```

The staging directory exposes only:

```text
train_agentic_mix.json -> ../skeleton_value_infill/public_train_no_public_smoke/value_infill_train.json
train_agentic_mix.manifest
```

This avoids passing the raw artifact directory directly to the LMFlow loader,
because that directory also contains non-training `summary.json` and audit
JSON files.

Corpus summary:

- source artifact note: `qwen35_skeleton_value_infill_artifacts_result.md`
- source artifact dir: `data/skeleton_value_infill/public_train_no_public_smoke/`
- source records: `45`
- value-infill instances: `331`
- usable value slots: `331`
- candidate rows: `711`
- boundary rows: `4667`
- public/heldout overlap audit: `0` exact overlaps, `0` user overlaps
- promotion allowed: yes, subject to separate heldout/public gates

## One-Step Fit Gate

Output:

```text
runs/fastdllm_qwen35_9b_skeleton_value_infill_from_ckpt275_step1_gate
logs/fastdllm_qwen35_9b_skeleton_value_infill_from_ckpt275_step1_gate.log
```

Settings:

- start adapter:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model`
- `MAX_STEPS=1`
- `MAX_TRAIN_SAMPLES=64`
- `BLOCK_SIZE=1024`
- `LEARNING_RATE=1e-6`
- `GRAD_ACCUM=4`
- `DISABLE_GROUP_TEXTS=1`
- `TRUNCATION_SIDE=left`
- `VALUE_SPAN_LABEL_ONLY=0`
- CPU cgroup: `MemoryHigh=27G`, `MemoryMax=28G`, `MemorySwapMax=4G`

Result:

- global step: `1`
- train loss: `3.3804564476013184`
- runtime: `10.59s`
- adapter saved: yes
- checkpoint adapter saved: yes
- OOM: no

## Step-75 Sweep

Output:

```text
runs/fastdllm_qwen35_9b_skeleton_value_infill_from_ckpt275_step75
logs/fastdllm_qwen35_9b_skeleton_value_infill_from_ckpt275_step75.log
```

Settings:

- start adapter:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model`
- `MAX_STEPS=75`
- `MAX_TRAIN_SAMPLES=331`
- `SAVE_STEPS=25`
- `SAVE_TOTAL_LIMIT=4`
- `BLOCK_SIZE=1024`
- `LEARNING_RATE=1e-6`
- `GRAD_ACCUM=4`
- `DISABLE_GROUP_TEXTS=1`
- `TRUNCATION_SIDE=left`
- `VALUE_SPAN_LABEL_ONLY=0`
- CPU cgroup: `MemoryHigh=27G`, `MemoryMax=28G`, `MemorySwapMax=4G`

Result:

- global step: `75`
- train runtime: `745.26s`
- train samples/sec: `0.403`
- train steps/sec: `0.101`
- final train loss: `2.647933057149251`
- OOM: no
- GPU returned idle after completion

Saved checkpoint adapters:

```text
runs/fastdllm_qwen35_9b_skeleton_value_infill_from_ckpt275_step75/checkpoint-25/adapter_model
runs/fastdllm_qwen35_9b_skeleton_value_infill_from_ckpt275_step75/checkpoint-50/adapter_model
runs/fastdllm_qwen35_9b_skeleton_value_infill_from_ckpt275_step75/checkpoint-75/adapter_model
```

Loss log:

| Step | Loss |
| ---: | ---: |
| 5 | 2.9884 |
| 10 | 3.1266 |
| 15 | 3.1636 |
| 20 | 2.8708 |
| 25 | 2.5634 |
| 30 | 2.4567 |
| 35 | 2.5603 |
| 40 | 2.3525 |
| 45 | 2.8260 |
| 50 | 2.4672 |
| 55 | 2.5223 |
| 60 | 2.3231 |
| 65 | 1.8908 |
| 70 | 2.9447 |
| 75 | 2.6625 |

## Interpretation

The clean skeleton-conditioned value-infill corpus is trainable on the 5090 as
a continuation from checkpoint-275. The staged one-file dataset directory was
necessary because the raw artifact directory contains non-training JSON files.

This does not yet show downstream agentic improvement. It only establishes the
first trainable adapter line for the behavior-preserving recipe:

1. keep tool-call skeleton, names, keys, and JSON structure protected;
2. train argument-value choice/infill under fixed skeleton context;
3. evaluate checkpoints 25/50/75 on public and heldout tool-call gates;
4. only promote if value accuracy improves without structural regressions.

## Next Gate

Evaluate checkpoints `25`, `50`, and `75` against:

- public one-call exact arguments;
- public multi-call raw/constrained exact sequence and arguments;
- heldout policy-target raw/constrained sequence and arguments;
- close-tag completeness and JSON segment validity.

Promotion should require movement on heldout or public value accuracy without
giving back the protected structural ceilings from the mode/name/value/close
guard stack.

Follow-up result:

```text
qwen35_skeleton_value_infill_checkpoint_eval_result.md
```

The checkpoint eval is complete. Checkpoints `25`, `50`, and `75` tie the
active checkpoint-275 adapter on public multi-call closeguard (`12/12` valid,
`12/12` exact sequence, `11/12` exact arguments) and heldout lean closeguard
(`11/12` valid, `11/12` exact sequence, `11/12` exact arguments). Checkpoint
`25` has a small one-call raw/model-repair improvement, but no checkpoint is
promotion-worthy.
