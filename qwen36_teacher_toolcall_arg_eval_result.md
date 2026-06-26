# Qwen3.6 Teacher Tool-Call Argument Eval Result

Date: 2026-06-26

## Purpose

Move the teacher/eval loop beyond tool-name detection. This result adds
argument-level and schema-level scoring for Qwen-native tool calls, then runs
the live Qwen3.6 NVFP4 teacher on synthetic and public one-call eval slices.

## Code Changes

- `scripts/eval_toolcall_jsonl.py`
  - parses JSON `<tool_call>` objects
  - parses Qwen-native `<tool_call><function=...><parameter=...>` calls
  - validates required arguments and a practical JSON-schema subset
  - compares teacher calls against gold tool sequence and arguments
- `scripts/teacher_distill_toolcall_cases.py`
  - uses top-level `chat_template_kwargs.enable_thinking=false`
  - records teacher calls plus name, argument, and schema metrics
  - supports larger `--max-tokens` for public examples
- `scripts/build_public_toolcall_eval_cases.py`
  - builds public eval slices from normalized seed data
  - can filter to one-call gold cases for the first public gate

## Teacher Profile

Same live teacher as `qwen36_teacher_serving_result.md`:

- Qwen3.6-27B NVFP4
- SGLang `0.5.14`
- RTX 5090
- 2k context
- Triton attention
- CUTLASS FP4 GEMM
- CUDA graph disabled
- MTP disabled

## Synthetic One-Call Held-Out

Input:

```text
data/toolcall_eval/synthetic_onecall_smoke.jsonl
```

Output:

```text
data/toolcall_eval/synthetic_onecall_teacher_q36_nvfp4_arg_heldout48.jsonl
```

Summary:

| Metric | Result |
| --- | ---: |
| records | 48 |
| endpoint ok | 48/48 |
| recognized valid tool-call emission | 48/48 |
| exact tool-name set | 48/48 |
| exact tool sequence | 48/48 |
| exact arguments | 48/48 |
| all schema-valid | 48/48 |
| all required args present | 48/48 |
| elapsed | 74.97s |

This means the teacher is clean on the controlled synthetic one-call curriculum
and can be used as a verifier/labeler for the 1.5B diffusion lab loop.

## Public Hermes One-Call Slice

Eval slice build:

```bash
.venv-lmeval/bin/python scripts/build_public_toolcall_eval_cases.py \
  --input data/toolcall_seed/qwen_toolcall_seed.jsonl \
  --out data/toolcall_eval/public_onecall_hermes_smoke.jsonl \
  --limit 24 \
  --max-gold-calls 1 \
  --sources hermes
```

Teacher run:

```bash
.venv-lmeval/bin/python scripts/teacher_distill_toolcall_cases.py \
  --input-jsonl data/toolcall_eval/public_onecall_hermes_smoke.jsonl \
  --out-jsonl data/toolcall_eval/public_onecall_hermes_teacher_q36_nvfp4_arg24.jsonl \
  --limit 24 \
  --timeout 240 \
  --temperature 0 \
  --max-tokens 512
```

Summary:

| Metric | Result |
| --- | ---: |
| records | 24 |
| endpoint ok | 24/24 |
| recognized valid tool-call emission | 24/24 |
| exact tool-name set | 21/24 |
| exact tool sequence | 21/24 |
| exact arguments | 18/24 |
| all schema-valid | 23/24 |
| all required args present | 24/24 |
| elapsed | 89.13s |

## Failure Modes

The public one-call failures are aligned with the agentic risks in the main
plan:

- **Over-calling:** 3 examples emitted an extra plausible tool despite the gold
  being one call.
- **Argument drift:** 3 examples picked the right tool but differed from the gold
  argument object.
- **Schema drift:** 1 over-call also produced one argument that failed the
  declared schema.

These are exactly the cases to keep as hard negatives for constrained decoding,
teacher repair, and the later Qwen3.5-9B diffusion loop.

## Next Step

The next useful gate is public multi-call/tool-result traces:

1. Build a public multi-call eval slice.
2. Add repeated-call and extra-call metrics.
3. Add argument repair/regenerate output that preserves schema.
4. Use synthetic + public failures as the next 1.5B diffusion curriculum.
5. Measure Qwen3.5-9B AR on the same eval before training its diffusion target.
