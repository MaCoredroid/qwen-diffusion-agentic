# Qwen3.6 Teacher Multi-Call Eval Result

Date: 2026-06-26

## Status

The Qwen3.6 NVFP4 MTP teacher now has a public Hermes multi-call baseline. This
extends the one-call tool-selection/argument eval to 2-3 sequential tool calls
and records extra, missing, and repeated calls.

## Eval Slice

Built from the existing public seed file:

```bash
python3 scripts/build_public_toolcall_eval_cases.py \
  --out data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --limit 12 \
  --min-gold-calls 2 \
  --max-gold-calls 3 \
  --sources hermes
```

Manifest:

```text
records_seen: 26
records_written: 12
skipped_call_count: 14
source: hermes
gold calls per example: 2-3
```

## Teacher Profile

Live SGLang server:

```text
model: qwen3.6-27b-teacher
checkpoint: sakamakismile/Qwen3.6-27B-NVFP4
context: 4096
MTP: NEXTN, steps=3, topk=1, draft_tokens=4
CUDA graph: disabled
radix cache: disabled
max running requests: 1
```

The 1k MTP profile was too small for public multi-call cases. The first selected
Hermes case was 1353 input tokens and returned HTTP 400 against a 1024-token
server. The 2k profile worked but truncated the longest case. The 4k profile
fits and is the better local teacher setting for public eval/data generation.

Observed memory after a 4k smoke:

```text
RTX 5090: about 23.7 GiB used, about 8.2 GiB free
```

## Command

```bash
.venv-lmeval/bin/python scripts/teacher_distill_toolcall_cases.py \
  --input-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --out-jsonl data/toolcall_eval/public_multicall_hermes_teacher_q36_mtp4k_12.jsonl \
  --endpoint http://127.0.0.1:30000/v1 \
  --model qwen3.6-27b-teacher \
  --timeout 120 \
  --max-tokens 1024
```

## Result

```text
records: 12
ok: 12
valid tool-call emissions: 12
exact tool-name set: 11
exact tool sequence: 11
exact tool-name multiset: 11
same tool-call count: 12
exact arguments: 10
all schema valid: 12
all required args present: 12
records with extra calls: 1
records with missing calls: 1
records with repeated calls: 0
total extra calls: 1
total missing calls: 1
total repeated calls: 0
elapsed: 72.88s
```

## Failure Modes

1. Timestamp normalization mismatch:
   - tool sequence was correct
   - arguments were schema-valid
   - teacher emitted timestamps without the trailing `Z`

2. Wrong third tool in a repeated-call-style prompt:
   - gold: `activate_voice_command`, `set_thermostat`, `activate_voice_command`
   - teacher: `activate_voice_command`, `set_thermostat`, `activate_security_cameras`
   - counted as one extra call and one missing call

No repeated-call loop appeared in this 12-case slice.

## Code Changes

- `scripts/build_public_toolcall_eval_cases.py`
  - added `--min-gold-calls`
- `scripts/eval_toolcall_jsonl.py`
  - added tool-name multiset, same-count, extra/missing/repeated metrics
- `scripts/teacher_distill_toolcall_cases.py`
  - reports the new metrics
  - now records HTTP 400 response bodies, which exposed context-length failures

## Next Gate

Completed follow-ups:

1. Qwen3.5-9B AR baseline on the same one-call, multi-call, and tool-result
   slices: `qwen35_9b_ar_baseline_result.md`.
2. Stricter OpenAI `tool_calls` plus `role=tool` tool-result variant:
   `qwen36_teacher_openai_toolresult_eval_result.md`.
3. Local 1.5B Fast-dLLM diffusion baseline comparison:
   `qwen25_1p5b_diffusion_baseline_result.md`.

Next gate: first Qwen3.5-9B diffusion/QLoRA run against offline Qwen3.6 labels.
