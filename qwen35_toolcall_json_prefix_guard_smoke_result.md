# Qwen3.5 Tool-Call JSON Prefix Guard Smoke

Date: 2026-06-28

## Purpose

Move one step from post-hoc projection toward generation-time constrained
decoding for agentic tool calls.

The new sampler option checks scheduled JSON/tool-call intervals before
committing sampled tokens. It keeps commits left-to-right inside guarded
intervals and verifies that the active `<tool_call>` body remains a valid JSON
prefix. If the top token would make the JSON prefix unrecoverable, the sampler
can replace it with the first safe token from a top-k scan.

## Code

Updated:

```text
scripts/eval_fastdllm_toolcall_cases.py
```

New opt-in flags:

```text
--guard-tool-json-prefix
--json-prefix-guard-kinds
--json-prefix-guard-topk
--json-prefix-guard-target-fallback
--no-json-prefix-guard-left-to-right
```

Default behavior is unchanged unless `--guard-tool-json-prefix` is set.

Compile gate:

```bash
.venv-fastdllm/bin/python -m py_compile \
  scripts/eval_fastdllm_toolcall_cases.py \
  scripts/diagnose_toolcall_json_completability.py
```

Result: passed.

## Helper Check

The prefix checker accepts incomplete but completable JSON and rejects
unrecoverable grammar states:

| fragment | result |
|---|---:|
| empty active tool body | `True` |
| partial JSON object/string | `True` |
| complete JSON plus partial close tag | `True` |
| unquoted/malformed key | `False` |
| missing colon after key | `False` |
| complete closed tool call | `True` |

## GPU Smokes

All GPU smokes used the local RTX 5090 with:

```text
base: models/qwen3.5-9b-fastdllm-init
adapter: runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model
input: data/toolcall_eval/public_multicall_hermes_smoke.jsonl
schedule: runs/tool_sensitive_block_plans/public_multicall_gold_sampler_schedule_with_pairwise_argsketch_choices_12.jsonl
limit: 1
full-context sampling
```

### Protected Path Compatibility

Output:

```text
runs/tool_sensitive_block_plans/public_multicall_jsonprefix_guard_smoke1.jsonl
```

This run kept the existing protected settings:

```text
--force-schedule-token-kinds json_key,json_structure,tool_tag
--force-argument-boundary-target-tokens
--force-best-candidate-sequence
--force-best-tool-name-sequence
--ban-argument-boundary-tokens
--ban-argument-newline-tokens
--guard-tool-json-prefix
```

Result:

| metric | value |
|---|---:|
| raw valid JSON | `1/1` |
| raw exact sequence | `1/1` |
| raw exact arguments | `1/1` |
| constrained exact arguments | `1/1` |
| guard interval visits | `0` |

Interpretation: the new flag does not break the existing protected path. Guard
visits are zero because structural/candidate forcing fills the relevant
scheduled intervals before normal sampling reaches the guard hook.

### Unforced Guard Limitation

Output:

```text
runs/tool_sensitive_block_plans/public_multicall_jsonprefix_guard_unforced_smoke1.jsonl
```

Settings: schedule plus `--guard-tool-json-prefix`, without structural forcing.

Result:

| metric | value |
|---|---:|
| raw valid JSON | `0/1` |
| raw exact sequence | `0/1` |
| raw exact arguments | `0/1` |
| constrained exact sequence | `1/1` |
| constrained exact arguments | `0/1` |
| guard interval visits | `142` |
| guard accepted tokens | `142` |
| left-to-right dropped commits | `14` |
| rejected/replaced/unsafe tokens | `0` |

The raw output started with prose/thinking rather than a tool-call envelope, so
the active JSON-prefix guard had no `<tool_call>` body to constrain. This shows
the first boundary condition: grammar-prefix checking must be paired with a
tool-call mode/envelope detector or literal sentinel forcing.

### Tool-Tag-Only Comparison

Guarded output:

```text
runs/tool_sensitive_block_plans/public_multicall_jsonprefix_guard_tooltag_smoke1.jsonl
```

Unguarded comparison:

```text
runs/tool_sensitive_block_plans/public_multicall_tooltag_noguard_smoke1.jsonl
```

Both forced only `tool_tag`; JSON keys, structure, tool names, and values were
left to normal sampling.

| metric | no guard | JSON-prefix guard |
|---|---:|---:|
| raw valid JSON | `0/1` | `1/1` |
| raw exact tool sequence | `0/1` | `1/1` |
| raw exact arguments | `0/1` | `0/1` |
| raw schema valid | `1/1` | `1/1` |
| raw required args present | `1/1` | `1/1` |
| constrained exact sequence | `1/1` | `1/1` |
| constrained exact arguments | `0/1` | `0/1` |
| guard interval visits | n/a | `130` |
| guard accepted tokens | n/a | `130` |
| left-to-right dropped commits | n/a | `12` |
| rejected/replaced/unsafe tokens | n/a | `0` |

Completability diagnostic:

```text
runs/tool_sensitive_block_plans/jsonprefix_guard_tooltag_comparison_completability.json
```

The unguarded raw output had `1` unrecoverable JSON segment:

```text
"stream_quality": "1080
p
</tool_call>
```

The guarded raw output had `3/3` complete JSON segments and exact tool
sequence, but still missed exact arguments because it produced timestamps
without `Z`:

```text
"start_time": "2023-04-22T15:00:00"
"end_time": "2023-04-22T17:00:00"
```

## Interpretation

This is a positive sampler primitive, not a promoted checkpoint.

What it proves:

- generation-time left-to-right JSON-prefix guarding can prevent at least one
  raw malformed tool-call failure without final projection;
- once tool-call envelopes are present, the guard can move raw generation from
  invalid/no exact sequence to valid/exact sequence on the one-case smoke;
- the guard is compatible with the existing protected route.

What it does not solve:

- entering tool-call mode from ordinary prose/thinking;
- exact value grounding, timestamp normalization, ID/path copying, or paired
  argument consistency;
- full 12-case heldout robustness.

Next step:

1. Pair the prefix guard with a tool-call mode detector or literal sentinel
   protection so prose cannot bypass the active JSON checker.
2. Keep the guard in the sampler for `tool_tag`, `json_structure`, `json_key`,
   `tool_name`, and `argument_value` intervals.
3. Add schema-aware key/value masks and value-candidate/ranker infill for exact
   arguments.
4. Then run the 12-case public and heldout scheduled/protected scorecards.
