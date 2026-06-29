# Qwen3.5 Public Eval Overlap Audit

Date: 2026-06-28

## Purpose

Audit whether the local `fastdllm_toolcall_train` public-train source is
actually disjoint from the public multi-call smoke eval slice.

This matters because the next planner/value-selection target is based on public
multi-call failures. We should not train on the exact public-12 eval rows and
then claim heldout improvement.

## Scripts

Added:

- `scripts/audit_toolcall_eval_overlap.py`
- `scripts/filter_toolcall_eval_overlap.py`

The audit compares normalized `user text + assistant tool calls` fingerprints
between conversation-style train JSON and eval JSONL files.

## Audit Result

Command:

```bash
.venv-fastdllm/bin/python scripts/audit_toolcall_eval_overlap.py \
  --train-conversation-json data/fastdllm_toolcall_train/train_toolcall.json \
  --eval-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --out-json runs/public_eval_overlap_audit/fastdllm_train_vs_public_multicall_hermes_smoke.json
```

Result:

- train records: `96`
- eval records: `12`
- exact overlaps: `11`
- user-only overlaps: `11`

Artifact:

`runs/public_eval_overlap_audit/fastdllm_train_vs_public_multicall_hermes_smoke.json`

This means the local public-train source contains most of the public multi-call
smoke slice verbatim. In particular, the two current public multi-call planner
failure families are exact overlaps:

- voice-command security-camera row
- motion-detector installation-code row

There are no non-eval analogues for those two exact families in the current
`fastdllm_toolcall_train` file.

## Filtered Source

Command:

```bash
.venv-fastdllm/bin/python scripts/filter_toolcall_eval_overlap.py \
  --train-conversation-json data/fastdllm_toolcall_train/train_toolcall.json \
  --eval-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --out-json data/fastdllm_toolcall_train/train_toolcall_no_public_multicall_smoke.json
```

Result:

- input records: `96`
- removed overlaps: `11`
- kept records: `85`

Verification:

```bash
.venv-fastdllm/bin/python scripts/audit_toolcall_eval_overlap.py \
  --train-conversation-json data/fastdllm_toolcall_train/train_toolcall_no_public_multicall_smoke.json \
  --eval-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --out-json runs/public_eval_overlap_audit/filtered_train_vs_public_multicall_hermes_smoke.json
```

Filtered result:

- exact overlaps: `0`
- user-only overlaps: `0`

## Clean Planner Curriculum

`scripts/build_toolcall_sequence_planner_distill_curriculum.py` now supports:

```bash
--exclude-eval-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl
```

Clean planner curriculum command:

```bash
.venv-fastdllm/bin/python scripts/build_toolcall_sequence_planner_distill_curriculum.py \
  --public-train data/fastdllm_toolcall_train/train_toolcall.json \
  --exclude-eval-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --out-dir data/qwen35_9b_toolcall_sequence_planner_distill_excluding_public_multicall_smoke_curriculum \
  --tool-schema-mode compact \
  --prompt-mode instruction \
  --block-size 1024 \
  --truncation-side right \
  --accept-mode exact_sequence
```

Manifest:

`data/qwen35_9b_toolcall_sequence_planner_distill_excluding_public_multicall_smoke_curriculum/train_agentic_mix.manifest`

Key numbers:

- raw public multi-call records: `56`
- eval overlaps removed: `11`
- remaining public multi-call records: `45`
- planner exact sequence: `22/45`
- planner exact arguments: `1/45`
- accepted after label/token checks and dedupe: `15`
- `no_eval_leakage=true`

## Decision

- Treat previous public-train-derived planner/value-span runs as diagnostic,
  not clean promotion evidence, unless their data source is rechecked with the
  overlap auditor.
- Do not train directly on the exact public-12 failure rows.
- Use the filtered source for future public-train planner/value experiments.
- For the voice-command camera and motion-detector installation-code failures,
  create synthetic or teacher-generated analogues instead of mining the current
  public-train source, because the only local analogues are the eval rows
  themselves.
- Any future manifest that claims `contains_eval_slice=false` should point to an
  overlap-audit artifact.
