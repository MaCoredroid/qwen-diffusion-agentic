# Qwen3.6 Teacher OpenAI Tool-Result Eval Result

Date: 2026-06-26

## Status

The synthetic two-step tool-result harness now has a stricter OpenAI-native
variant:

```text
assistant.tool_calls -> role=tool -> next assistant.tool_calls
```

This complements the earlier text-compatible tool-result slice, which encoded
the tool result as a user message. The stricter variant is closer to the protocol
used by coding agents and OpenAI-compatible serving stacks.

## Added

Builder update:

```text
scripts/build_synthetic_toolresult_traces.py
```

It now writes both:

```text
data/toolcall_eval/synthetic_toolresult_smoke.jsonl
data/toolcall_eval/synthetic_toolresult_openai_smoke.jsonl
```

New evaluator:

```text
scripts/eval_openai_toolcall_cases.py
```

The evaluator sends OpenAI Chat Completions requests, scores native
`message.tool_calls` when present, and optionally scores textual fallback output
with the same shared tool-call scorer.

Shared scorer update:

```text
scripts/eval_toolcall_jsonl.py
```

It now supports normalized scoring of native OpenAI tool-call objects, including
exact sequence, exact arguments, schema validity, extra/missing/repeated calls,
and required-argument presence.

## Gold Validation

The generated native gold tool calls were validated against the existing
Qwen-text gold calls:

```text
records: 10
native gold exact sequence: 10/10
native gold exact arguments: 10/10
native gold schema valid: 10/10
```

## Qwen3.6 Teacher Runs

Runtime:

```text
model: qwen3.6-27b-teacher
checkpoint: sakamakismile/Qwen3.6-27B-NVFP4
server: SGLang 0.5.14
profile: 4k context, NVFP4, MTP/NEXTN, CUDA graph disabled
hardware: local RTX 5090
RTX 5080 use: none
```

### Native Tool Calls, `tool_choice=auto`

Command shape:

```bash
.venv-lmeval/bin/python scripts/eval_openai_toolcall_cases.py \
  --input-jsonl data/toolcall_eval/synthetic_toolresult_openai_smoke.jsonl \
  --out-jsonl data/toolcall_eval/synthetic_toolresult_openai_teacher_q36_mtp4k_10.jsonl \
  --endpoint http://127.0.0.1:30000/v1 \
  --model qwen3.6-27b-teacher \
  --temperature 0 \
  --max-tokens 256 \
  --tool-choice auto
```

Result:

```text
native tool-call responses: 0/10
strict native exact sequence: 0/10
strict native exact arguments: 0/10
```

The model produced correct Qwen text calls in `message.content`, but SGLang did
not expose them as native `message.tool_calls` in this mode.

### Text Fallback, `tool_choice=auto`

Same request, with `--allow-text-fallback`:

```text
text fallback responses: 10/10
valid tool JSON / Qwen function calls: 10/10
exact sequence: 10/10
exact arguments: 10/10
schema valid: 10/10
extra / missing / repeated: 0 / 0 / 0
```

This means the model understands the role=tool continuation, even though native
tool-call surfacing is not automatic under `tool_choice=auto`.

### Native Tool Calls, `tool_choice=required`

Command shape:

```bash
.venv-lmeval/bin/python scripts/eval_openai_toolcall_cases.py \
  --input-jsonl data/toolcall_eval/synthetic_toolresult_openai_smoke.jsonl \
  --out-jsonl data/toolcall_eval/synthetic_toolresult_openai_teacher_q36_mtp4k_required_10.jsonl \
  --endpoint http://127.0.0.1:30000/v1 \
  --model qwen3.6-27b-teacher \
  --temperature 0 \
  --max-tokens 256 \
  --tool-choice required
```

Result:

| Metric | Result |
| --- | ---: |
| native tool-call responses | 10/10 |
| exact tool sequence | 10/10 |
| exact tool-name multiset | 10/10 |
| exact arguments | 8/10 |
| schema valid | 8/10 |
| extra / missing / repeated calls | 0 / 0 / 0 |

Two failures emitted the right native function name with empty `{}` arguments:

```text
synthetic-toolresult-00001: create_support_ticket {}
synthetic-toolresult-00008: route_incident {}
```

## Interpretation

For the async data/eval loop:

- Use saved JSONL labels/results; the 27B teacher and 9B student do not need to
  be live at the same time.
- For behavior scoring, Qwen text fallback is currently strongest: 10/10 exact
  on this slice.
- For strict OpenAI-native agent protocol scoring, use `tool_choice=required`
  and track argument-drop failures separately.
- Do not treat `tool_choice=auto` native misses as reasoning failures; in this
  run they were mostly serving/parser surfacing failures.

Next training implication:

- Add native OpenAI tool-call examples to the 9B diffusion curriculum.
- Weight required argument spans heavily.
- Track empty-argument native calls as a first-class failure mode.
