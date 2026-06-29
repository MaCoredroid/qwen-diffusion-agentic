# Qwen3.5 Tool-Call JSON Completability Diagnostic

Date: 2026-06-28

## Purpose

Add a lightweight structural diagnostic between raw generation and final
repair/projection:

```text
Did each generated <tool_call> body remain valid JSON or at least a prefix that
could be completed into valid JSON, or did the sampler commit an unrecoverable
JSON state?
```

This is a proxy for the next generation-time grammar guard. It analyzes final
eval output strings, not every intermediate denoising trace, so it should be
used to prioritize sampler work rather than as a full constrained-decoding
implementation.

## Code

Added:

```text
scripts/diagnose_toolcall_json_completability.py
```

The script:

- joins eval output rows back to their source case JSONL via the eval summary;
- scans each requested assistant text field for `<tool_call>...</tool_call>`
  blocks;
- classifies each JSON body as `complete`, `incomplete` but still
  prefix-completable, `invalid`, `empty`, or `no_json`;
- reports missing closing tags, extra/fewer tool-call segments versus gold,
  schema scores when source tools are available, and row examples.

Compile gate:

```bash
.venv-fastdllm/bin/python -m py_compile \
  scripts/diagnose_toolcall_json_completability.py
```

Result: passed.

## Runs

Raw/constrained diagnostic:

```bash
.venv-fastdllm/bin/python scripts/diagnose_toolcall_json_completability.py \
  runs/target_geometry_eval/bd16_checkpoint5_policy_targets_forcedprefix.jsonl \
  runs/target_geometry_eval/bdchoices8_16_32_checkpoint5_policy_targets_forcedprefix.jsonl \
  runs/target_geometry_eval/bd16_argvalue_lowpressure_checkpoint10_policy_targets_forcedprefix.jsonl \
  runs/target_geometry_eval/bd16_checkpoint5_scalar_repair_sidecar_policy_targets.jsonl \
  --out-json runs/target_geometry_eval/toolcall_json_completability_diagnostic.json \
  --out-jsonl runs/target_geometry_eval/toolcall_json_completability_diagnostic.examples.jsonl \
  --max-examples 12
```

Scalar sidecar fields:

```bash
.venv-fastdllm/bin/python scripts/diagnose_toolcall_json_completability.py \
  runs/target_geometry_eval/bd16_checkpoint5_scalar_repair_sidecar_policy_targets.jsonl \
  --field scalar_repair_assistant \
  --field scalar_repair_constrained_assistant \
  --out-json runs/target_geometry_eval/scalar_repair_toolcall_json_completability_diagnostic.json \
  --out-jsonl runs/target_geometry_eval/scalar_repair_toolcall_json_completability_diagnostic.examples.jsonl \
  --max-examples 12
```

## Summary

Heldout policy-target slice, 12 rows:

| Lane | Field | Segments | Complete JSON | Incomplete but completable | Invalid JSON | Rows with unrecoverable JSON | Valid tool JSON | Exact seq | Exact args | Schema valid | Required args |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed `bd_size=16` ckpt5 | raw `assistant` | 30 | 5 | 0 | 25 | 12 | 0/12 | 0/12 | 0/12 | 2/12 | 2/12 |
| fixed `bd_size=16` ckpt5 | `constrained_assistant` | 28 | 28 | 0 | 0 | 0 | 12/12 | 6/12 | 0/12 | 7/12 | 8/12 |
| dynamic `8,16,32` ckpt5 | raw `assistant` | 38 | 5 | 2 | 31 | 12 | 0/12 | 0/12 | 0/12 | 2/12 | 2/12 |
| dynamic `8,16,32` ckpt5 | `constrained_assistant` | 26 | 26 | 0 | 0 | 0 | 12/12 | 5/12 | 0/12 | 7/12 | 8/12 |
| low-pressure arg/value ckpt10 | raw `assistant` | 45 | 10 | 4 | 31 | 12 | 0/12 | 1/12 | 0/12 | 4/12 | 4/12 |
| low-pressure arg/value ckpt10 | `constrained_assistant` | 27 | 27 | 0 | 0 | 0 | 12/12 | 3/12 | 0/12 | 8/12 | 8/12 |
| scalar repair sidecar | `scalar_repair_assistant` | 28 | 28 | 0 | 0 | 0 | 12/12 | 6/12 | 0/12 | 9/12 | 10/12 |
| scalar repair sidecar | `scalar_repair_constrained_assistant` | 28 | 28 | 0 | 0 | 0 | 12/12 | 6/12 | 0/12 | 9/12 | 10/12 |

Dominant unrecoverable raw JSON reasons:

- expected object key string,
- expected comma or object close,
- expected colon after object key,
- control character in string,
- extra content after JSON value.

Example failure pattern from the fixed `bd_size=16` raw lane:

```text
{"namename": "synchronizeRoomAvailability",argumentsarguments": ...
```

and:

```text
"date_range": {"start_date "2023-04-01", ...
```

These are not simple missing-suffix cases. They are illegal committed token
states that a final projection can rewrite, but a behavior-preserving diffusion
sampler should reject or re-denoise before commit.

## Interpretation

1. The raw generator is failing structure before it reaches argument grounding.
   All three raw generation lanes have unrecoverable JSON-prefix errors on
   `12/12` heldout rows.

2. The constrained/projection path solves grammar after the fact. It gets
   complete JSON on every row, but exact arguments remain `0/12`.

3. The scalar repair sidecar is a syntax/schema aid, not the grounding answer.
   It improves schema-valid and required-argument counts, but preserves the
   same `6/12` sequence and `0/12` exact-argument ceiling.

4. Dynamic block size by itself is not enough. The dynamic `8,16,32` lane has
   more tool-call segments and still has unrecoverable JSON in every row.

## Next Experiment Implication

The next sampler should be tool-call grammar aware at commit time:

- freeze or force `<tool_call>` / `</tool_call>` sentinels and JSON structural
  tokens in tiny blocks;
- before committing a token/span inside a tool-call region, check that the
  partially filled JSON with holes can still complete under the target grammar;
- if a candidate creates an unrecoverable JSON state, reject it, shrink the
  block to one token, or re-denoise the span;
- after route/order is grammar-safe, use a separate value-infill objective or
  sidecar ranker for exact IDs, dates, numbers, enums, and nested objects.

Do not spend the next 5090 run on broader uniform arg/value masking alone.
The immediate missing mechanism is constrained, completable generation inside
tool-call spans plus a skeleton-then-value-infill split.
