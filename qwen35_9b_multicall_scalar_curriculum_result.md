# Qwen3.5-9B Multi-Call Scalar Curriculum Result

Date: 2026-06-27

## Status

Built and audited a targeted public multi-call scalar extraction curriculum for
the next Qwen3.5-9B diffusion training iteration.

This is a data/training-gate result, not a promoted model. The purpose is to
avoid repeating the failed full-chain multi-call curriculum by training smaller
one-call argument repair windows first.

## Builder

Script:

```text
scripts/build_toolcall_multicall_scalar_curriculum.py
```

Output:

```text
data/qwen35_9b_toolcall_multicall_scalar_curriculum/train_agentic_mix.json
data/qwen35_9b_toolcall_multicall_scalar_curriculum/train_agentic_mix.audit.jsonl
data/qwen35_9b_toolcall_multicall_scalar_curriculum/train_agentic_mix.manifest
```

The builder reads public multi-call training conversations, extracts each
individual tool call, keeps only the relevant single tool schema, creates a
short request excerpt around argument values, corrupts one scalar argument, and
targets exactly one corrected Qwen `<tool_call>` block.

Variants:

```text
empty_args
missing_field
wrong_scalar
null_field
```

## Dataset Audit

```text
public multi-call source records: 56
candidates: 1184
accepted: 1184
rejected: 0
skipped calls with no scalar props: 4
```

Variant balance:

```text
empty_args: 296
missing_field: 296
wrong_scalar: 296
null_field: 296
```

896-token label-retention audit:

```text
length min/p50/p90/max: 340 / 551 / 667 / 883
kept labels min/p50/p90/max: 25 / 45 / 95 / 194
zero-label rows after truncation: 0
partial-label rows after truncation: 0
```

## One-Step Gate

Ran the Qwen3.5-9B Fast-DLLM QLoRA launcher against the new dataset under a user
cgroup on the local RTX 5090:

```text
DATASET_DIR=/home/mark/qwen_diffusion/data/qwen35_9b_toolcall_multicall_scalar_curriculum
OUTPUT_DIR=/home/mark/qwen_diffusion/runs/fastdllm_qwen35_9b_toolcall_multicall_scalar_argspanw1p5_b896_step1_gate
MAX_STEPS=1
MAX_TRAIN_SAMPLES=8
BLOCK_SIZE=896
ARGUMENT_SPAN_LOSS_WEIGHT=1.5
LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj
```

Result:

```text
readiness: ready=true
global_step: 1
train_loss: 7.224119186401367
train_runtime: 2.8063s
train_samples_per_second: 0.356
adapter saved: yes
```

Output:

```text
runs/fastdllm_qwen35_9b_toolcall_multicall_scalar_argspanw1p5_b896_step1_gate
```

## Interpretation

- Positive data-path result: the new scalar curriculum is compact, balanced
  across corruption variants, and fully label-retained at block size 896.
- Positive trainer-gate result: the Qwen3.5-9B bridge can train on these rows
  with GDN-aware LoRA targets and argument-span weighting.
- Do not promote the one-step adapter. The next useful test is a short scalar
  adapter run or mixed-generator run, followed by the public multi-call
  sequence-preserving scorecard.
- Keep this curriculum staged or lower-weighted until it proves it improves
  public multi-call arguments without damaging one-call/tool-result behavior.
