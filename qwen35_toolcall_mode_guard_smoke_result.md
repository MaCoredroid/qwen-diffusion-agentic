# Qwen3.5 Tool-Call Mode Guard Smoke

Date: 2026-06-28

## Purpose

Close the boundary gap found by the first JSON-prefix guard smoke:

```text
JSON-prefix checking only works after generation has entered a <tool_call> body.
If the model emits prose/thinking first, there is no active JSON body to guard.
```

This smoke adds a named tool-call mode/sentinel primitive, separate from generic
oracle structure forcing.

## Code

Updated:

```text
scripts/eval_fastdllm_toolcall_cases.py
```

New opt-in flag:

```text
--guard-tool-call-mode
```

Behavior:

- applies only to scheduled `tool_tag` intervals;
- hard-fills the schedule's target sentinel tokens such as `<tool_call>`,
  newline, and `</tool_call>`;
- records separate counters:
  `tool_call_mode_force_interval_visits` and
  `tool_call_mode_force_token_visits`;
- leaves `json_key`, `json_structure`, `tool_name`, and `argument_value`
  intervals to the sampler unless other flags are explicitly enabled.

This makes tool-call mode protection visible separately from
`--force-schedule-token-kinds`.

Compile gate:

```bash
.venv-fastdllm/bin/python -m py_compile \
  scripts/eval_fastdllm_toolcall_cases.py \
  scripts/diagnose_toolcall_json_completability.py
```

Result: passed.

## Smoke

Base settings:

```text
base: models/qwen3.5-9b-fastdllm-init
adapter: runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model
input: data/toolcall_eval/public_multicall_hermes_smoke.jsonl
schedule: runs/tool_sensitive_block_plans/public_multicall_gold_sampler_schedule_with_pairwise_argsketch_choices_12.jsonl
limit: 1
max_new_tokens: 260
full-context sampling
```

Mode+prefix guard output:

```text
runs/tool_sensitive_block_plans/public_multicall_mode_jsonprefix_guard_smoke1.jsonl
```

Comparison target:

```text
runs/tool_sensitive_block_plans/public_multicall_jsonprefix_guard_unforced_smoke1.jsonl
```

Completability diagnostic:

```text
runs/tool_sensitive_block_plans/mode_jsonprefix_guard_comparison_completability.json
```

## Result

| metric | JSON-prefix only | mode + JSON-prefix |
|---|---:|---:|
| raw tool-call segments | `0` | `3` |
| raw complete JSON segments | `0` | `3` |
| raw valid JSON | `0/1` | `1/1` |
| raw exact tool sequence | `0/1` | `1/1` |
| raw exact arguments | `0/1` | `0/1` |
| raw schema valid | `0/1` | `1/1` |
| raw required args present | `0/1` | `1/1` |
| constrained exact sequence | `1/1` | `1/1` |
| constrained exact arguments | `0/1` | `0/1` |
| tool-call mode forced tokens | `0` | `12` |
| JSON-prefix guard intervals | `142` | `130` |
| left-to-right dropped commits | `14` | `12` |

The unforced JSON-prefix-only output started with prose/thinking and never
entered raw tool-call mode:

```text
<think>
The user is asking me to perform three different functions:
...
```

The mode+prefix output entered tool-call mode and produced complete raw JSON
for all three calls:

```text
<tool_call>
{"name": "get_camera_live_feed", "arguments": {"camera_id": "front_door", "stream_quality": "1080p"}}
</tool_call>
...
```

Exact arguments still fail because the final call emits timestamps without the
required `Z` suffix:

```text
"start_time": "2023-04-22T15:00:00"
"end_time": "2023-04-22T17:00:00"
```

## Interpretation

This is a positive sampler primitive, still not a promoted model result.

What it proves:

- tool-call mode/sentinel forcing prevents prose from bypassing the active
  JSON-prefix guard;
- mode + prefix protection can recover raw valid JSON and exact sequence on the
  one-row smoke without forcing keys, structure, tool names, or values;
- the protected mechanism is now decomposed into visible pieces:
  mode/sentinel protection, JSON-prefix commit checking, and later value
  grounding.

What it does not solve:

- exact scalar value grounding;
- timestamp normalization;
- ID/path/numeric copy precision;
- broader 12-case robustness.

Next step:

Add schema/value infill on top of mode+prefix protection:

1. schema-aware masks for `"name"`, `"arguments"`, and property keys;
2. value-candidate constraints for timestamps, IDs, paths, numbers, enums;
3. candidate/ranker selection for paired values such as start/end times;
4. then run the 12-case public/heldout scheduled scorecards.
