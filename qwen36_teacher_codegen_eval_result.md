# Qwen3.6 Teacher Codegen Eval Result

Date: 2026-06-26

## Status

The roadmap's 10 small code-generation tasks now exist and have a Qwen3.6
teacher baseline. This is a deterministic Python function-generation smoke
gate, separate from the higher-level Qwen Code repo-edit harness.

## Eval Slice

Built with:

```bash
python3 scripts/build_synthetic_codegen_tasks.py
```

Output:

```text
data/codegen_eval/synthetic_codegen_10.jsonl
```

The 10 tasks are:

```text
slugify_text
merge_intervals
top_k_frequent_words
parse_env_lines
balanced_brackets
word_wrap
multiset_added_lines
redact_secrets
stable_dedupe
apply_patch_ops
```

## Harness

Runner:

```text
scripts/eval_openai_codegen_cases.py
```

The harness asks an OpenAI-compatible model for a single Python function, strips
markdown fences if present, performs a conservative AST check, then runs the
task's unit tests in an isolated Python subprocess with CPU/address-space limits.

This first slice intentionally says "no imports" to measure instruction
following and standalone-function behavior. Import-using solutions are counted
as static-check failures.

## Command

```bash
.venv-lmeval/bin/python scripts/eval_openai_codegen_cases.py \
  --input-jsonl data/codegen_eval/synthetic_codegen_10.jsonl \
  --out-jsonl data/codegen_eval/synthetic_codegen_q36_mtp4k_10.jsonl \
  --endpoint http://127.0.0.1:30000/v1 \
  --model qwen3.6-27b-teacher \
  --timeout 120 \
  --temperature 0 \
  --max-tokens 1024 \
  --test-timeout 5
```

## Result

```text
records: 10
endpoint ok: 10
code extracted: 10
static check passed: 8
unit tests passed: 7
errors: 0
elapsed: 53.41s
```

## Failure Modes

1. `slugify_text`: used `import re`, violating the no-import instruction.
2. `top_k_frequent_words`: used `from collections import Counter`, violating
   the no-import instruction.
3. `word_wrap`: passed static checks but failed unit tests because line-length
   accounting double-counted word lengths.

## Interpretation

The teacher is functional but not perfect on constrained standalone codegen. The
failures are useful training signals:

- constraint following around "no imports"
- small algorithmic state accounting
- unit-test execution as a promotion gate before SWE-style tasks

Next step is to run the same codegen slice against Qwen3.5-9B AR and local
diffusion baselines, then create tiny repo-edit tasks for Qwen Code.
