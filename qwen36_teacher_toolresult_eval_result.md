# Qwen3.6 Teacher Tool-Result Eval Result

Date: 2026-06-26

## Status

The Qwen3.6 NVFP4 MTP teacher now has a small synthetic two-step tool-result
baseline. These cases test whether the model uses a prior tool observation to
choose the next action and avoids repeating the completed tool call.

This file covers the text-compatible version of the slice: assistant
`<tool_call>` followed by a user-visible `Tool result for ...` message. The
stricter OpenAI `assistant.tool_calls` plus `role=tool` variant now exists and is
recorded separately in:

```text
qwen36_teacher_openai_toolresult_eval_result.md
```

## Eval Slice

Built with:

```bash
python3 scripts/build_synthetic_toolresult_traces.py
```

Output:

```text
data/toolcall_eval/synthetic_toolresult_smoke.jsonl
data/toolcall_eval/synthetic_toolresult_openai_smoke.jsonl
```

The builder also writes a tiny training file for the diffusion curriculum:

```text
data/synthetic_toolresult_train/train_synthetic_toolresult.json
```

The 10 traces cover order escalation, delivered-order email, refund policy,
coupon application, inventory reservation, inventory shortage escalation,
critical incident paging, noncritical incident routing, subscription upgrade,
and callback scheduling.

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

## Command

```bash
.venv-lmeval/bin/python scripts/teacher_distill_toolcall_cases.py \
  --input-jsonl data/toolcall_eval/synthetic_toolresult_smoke.jsonl \
  --out-jsonl data/toolcall_eval/synthetic_toolresult_teacher_q36_mtp4k_10.jsonl \
  --endpoint http://127.0.0.1:30000/v1 \
  --model qwen3.6-27b-teacher \
  --timeout 120 \
  --temperature 0 \
  --max-tokens 384
```

## Result

```text
records: 10
ok: 10
valid tool-call emissions: 10
exact tool-name set: 10
exact tool sequence: 10
exact tool-name multiset: 10
same tool-call count: 10
exact arguments: 10
all schema valid: 10
all required args present: 10
records with extra calls: 0
records with missing calls: 0
records with repeated calls: 0
elapsed: 17.65s
```

## Interpretation

The teacher handles this controlled two-step gate cleanly once exact argument
values are grounded in the observation. The first draft of this slice had
under-specified natural-language fields such as callback date/time and email
subject/body; that made exact-argument scoring unfair. The current slice exposes
those target values explicitly in the tool result, which better measures whether
the model copies grounded state into the next call.

## Next Gate

1. Mix these examples into the next Qwen3.5-9B diffusion curriculum.
2. Track text-compatible and native OpenAI tool-call metrics separately.
3. Track empty-argument native calls as a first-class failure mode.
4. Track repeated-call
   loops before and after training.
