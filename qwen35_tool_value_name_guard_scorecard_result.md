# Qwen3.5 Tool Value/Name Guard Scorecard

Date: 2026-06-28

## Purpose

Continue decomposing protected tool-call generation into named sampler
primitives instead of generic oracle forcing:

1. `--guard-tool-call-mode`: force only scheduled `<tool_call>` / `</tool_call>`
   sentinel tokens.
2. `--guard-tool-json-prefix`: keep active tool-call JSON prefix completable.
3. `--guard-tool-value-candidates`: force one whole argument-value candidate
   sequence, with separate counters from `--force-best-candidate-sequence`.
4. `--guard-tool-name-candidates`: force one compatible tool-name candidate
   sequence, with separate counters from `--force-best-tool-name-sequence`.

## Code

Updated:

```text
scripts/eval_fastdllm_toolcall_cases.py
```

New opt-in flags added in this step:

```text
--guard-tool-value-candidates
--guard-tool-name-candidates
```

Close-tag completeness was then added under `--guard-tool-call-mode`: scheduled
tool-call sentinel forcing is deferred when an active `<tool_call>` body has
started JSON but the JSON body is not yet complete.

Compile gate:

```bash
.venv-fastdllm/bin/python -m py_compile \
  scripts/eval_fastdllm_toolcall_cases.py \
  scripts/diagnose_toolcall_json_completability.py
```

Result: passed.

## One-Row Value Smoke

Compared:

```text
runs/tool_sensitive_block_plans/public_multicall_mode_jsonprefix_guard_smoke1.jsonl
runs/tool_sensitive_block_plans/public_multicall_mode_prefix_valueguard_smoke1.jsonl
```

| metric | mode + prefix | mode + prefix + value |
|---|---:|---:|
| raw valid JSON | `1/1` | `1/1` |
| raw exact sequence | `1/1` | `1/1` |
| raw exact arguments | `0/1` | `1/1` |
| value candidate forced tokens | `0` | `58` |

The value guard fixed the previous timestamp miss by forcing the candidate
sequence with the required `Z` suffix:

```text
"start_time": "2023-04-22T15:00:00Z"
"end_time": "2023-04-22T17:00:00Z"
```

## 12-Case Public Scorecard: Value Guard

Run:

```text
runs/tool_sensitive_block_plans/public_multicall_mode_prefix_valueguard_12.jsonl
```

Settings:

```text
--guard-tool-call-mode
--guard-tool-json-prefix
--guard-tool-value-candidates
--stop-after-schedule-tool-calls
```

Result:

| metric | score |
|---|---:|
| raw valid JSON | `12/12` |
| raw exact tool sequence | `11/12` |
| raw exact arguments | `11/12` |
| raw schema valid | `12/12` |
| raw required args present | `12/12` |
| constrained exact sequence | `11/12` |
| constrained exact arguments | `8/12` |
| mode forced tokens | `124` |
| value candidate forced tokens | `343` |
| JSON-prefix guard intervals | `1166` |

The only raw miss is a route/name choice in the voice-command case:

```text
gold third call: activate_voice_command
raw third call:  activate_security_cameras
```

This means value grounding is no longer the blocker on this public-12 route;
the remaining raw miss is route/tool-name selection.

## 12-Case Public Scorecard: Name + Value Guard

Run:

```text
runs/tool_sensitive_block_plans/public_multicall_mode_prefix_name_value_guard_12.jsonl
```

Settings:

```text
--guard-tool-call-mode
--guard-tool-json-prefix
--guard-tool-name-candidates
--guard-tool-value-candidates
--stop-after-schedule-tool-calls
```

Result:

| metric | score |
|---|---:|
| raw valid JSON | `11/12` |
| raw exact tool-name set | `12/12` |
| raw exact tool sequence | `11/12` |
| raw exact arguments | `11/12` |
| raw schema valid | `12/12` |
| raw required args present | `12/12` |
| constrained exact sequence | `12/12` |
| constrained exact arguments | `8/12` |
| mode forced tokens | `124` |
| tool-name guard forced tokens | `152` |
| value guard forced tokens | `343` |
| JSON-prefix guard intervals | `1014` |

The same voice-command case remains the miss, but the failure mode changes.
Tool-name guarding chooses the right third tool name, then the closing sentinel
is forced while one JSON string is still incomplete:

```text
{"name": "activate_voice_command", "arguments": {
  "command": "Activate security cameras in away mode",
  "device_type": "camera",
  "location": "home
</tool_call>
```

The completability diagnostic classifies this as incomplete-but-completable,
not unrecoverable JSON. Final projection can recover sequence, but raw valid
JSON drops to `11/12`.

## 12-Case Public Scorecard: Name + Value + Close Guard

Run:

```text
runs/tool_sensitive_block_plans/public_multicall_mode_prefix_name_value_closeguard_12.jsonl
```

Settings:

```text
--guard-tool-call-mode
--guard-tool-json-prefix
--guard-tool-name-candidates
--guard-tool-value-candidates
--stop-after-schedule-tool-calls
```

Result:

| metric | score |
|---|---:|
| raw valid JSON | `12/12` |
| raw exact tool-name set | `12/12` |
| raw exact tool sequence | `12/12` |
| raw exact arguments | `11/12` |
| raw schema valid | `12/12` |
| raw required args present | `12/12` |
| constrained exact sequence | `12/12` |
| constrained exact arguments | `8/12` |
| mode forced tokens | `123` |
| close-tag deferrals | `1` |
| tool-name guard forced tokens | `152` |
| value guard forced tokens | `343` |
| JSON-prefix guard intervals | `1015` |
| max CUDA allocated / reserved | `18.45 GiB / 27.71 GiB` |

Completability diagnostic:

| field | complete JSON segments | invalid segments | exact sequence | exact arguments |
|---|---:|---:|---:|---:|
| raw assistant | `31/31` | `0` | `12/12` | `11/12` |
| constrained assistant | `31/31` | `0` | `12/12` | `8/12` |

The close guard fires exactly once on the earlier voice-command close-tag
failure and converts that row from incomplete JSON to complete JSON. The only
remaining raw miss is now a value-grounding mismatch:

```text
gold third call:
{"command": "Activate security cameras in away mode",
 "device_type": "camera",
 "location": ""}

raw third call:
{"command": "Activate security cameras in away mode",
 "device_type": "camera",
 "location": "home"}
```

That means the current sampler line has solved this public-12 structural gate.
The remaining error is a learned/evidence target problem: the prompt's explicit
argument list contains an empty location for that voice command, while the
model fills a plausible home location.

## Interpretation

Positive:

- Mode + prefix + value guards raise the public multi-call protected/raw route
  to `12/12` valid JSON and `11/12` exact arguments without generic
  `--force-schedule-token-kinds`.
- Value candidates directly fix the timestamp `Z` miss from the previous
  smoke.
- The remaining non-exact row is no longer a scalar value problem in the
  value-only run.
- Close-tag completeness removes the raw JSON regression from adding
  tool-name guarding and gives `12/12` raw exact sequence on public multi-call
  12.

Remaining limitation:

- Exact arguments are still `11/12`. The remaining miss is benchmark-specific
  value grounding (`location: ""` vs `location: "home"`), so the next learned
  step should use evidence-targeted value supervision or a skeleton-then-value
  infill target rather than more structural forcing.

Next gate:

1. Run the heldout policy-target scheduled route with mode + prefix + name +
   value + close guard.
2. Run the six-lane split-route scorecard with raw, constrained, and protected
   columns separated.
3. Build the next training target around evidence-grounded value infill and
   on-policy AR-teacher correction for benchmark-exact arguments.
