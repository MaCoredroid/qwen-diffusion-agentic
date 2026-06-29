# Qwen3.5 Live-Planner Arg-Sketch Selector Sampler Result

Date: 2026-06-28

## Purpose

Move the arg-sketch selector sampler from a gold-tokenized schedule to a
non-gold live planner schedule.

The previous sampler gate used gold assistant text to define sensitive token
spans. This gate uses the live `sequence_planner_assistant` text produced by
the public multi-call planner route, then runs the same selector-owned
tool-name and argument-value path over that planned text.

This is still a protected runtime result. It is not raw diffusion model
promotion because structural JSON/tool tags and stop boundaries remain guarded.

## Live Planner Source

Planner artifact:

```text
runs/qwen35_9b_split_route_sidecar_eval/live_smoke_public_multicall_planner/public_multi_call_planner/public_multicall_12_sequence_planner_segmentargs_v5_voice_safe.jsonl
```

The row-level `sequence_planner_*` metrics show:

- sequence-planner valid tool JSON: `12/12`
- sequence-planner exact tool sequence: `12/12`
- sequence-planner exact arguments: `12/12`
- sequence-planner schema valid: `12/12`
- sequence-planner required args present: `12/12`

This is a planner/post-processing artifact derived from live model outputs and
request/schema evidence, not the gold assistant text.

## Live Schedule Build

Block plan:

```bash
.venv-fastdllm/bin/python scripts/plan_tool_sensitive_blocks.py \
  --input-jsonl runs/qwen35_9b_split_route_sidecar_eval/live_smoke_public_multicall_planner/public_multi_call_planner/public_multicall_12_sequence_planner_segmentargs_v5_voice_safe.jsonl \
  --text-field sequence_planner_assistant \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_live_v5_sequence_planned_blocks_tokenized_with_ids.jsonl \
  --tokenizer-path models/qwen3.5-9b-fastdllm-init \
  --include-token-ids
```

Coverage:

- records: `12`
- tool calls: `31`
- token blocks: `628`
- tool-name blocks: `31`
- argument-value blocks: `100`
- all tool-name and argument-value blocks have `tool_call_index`, path
  metadata, and target token IDs

Sampler schedule:

```bash
.venv-fastdllm/bin/python scripts/emit_tool_sensitive_sampler_schedule.py \
  --input-jsonl runs/tool_sensitive_block_plans/public_multicall_live_v5_sequence_planned_blocks_tokenized_with_ids.jsonl \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_live_v5_sequence_planned_sampler_schedule_with_ids.jsonl \
  --include-token-ids
```

Coverage:

- records: `12`
- schedule blocks: `1079`
- tool-name schedule blocks: `152`
- argument-value schedule blocks: `115`

## Candidate Coverage

Evidence-only candidates:

```text
runs/tool_sensitive_block_plans/public_multicall_live_v5_sequence_planned_sampler_schedule_with_candidates_evidence.summary.json
data/candidate_ranking/public_multicall_live_v5_sequence_planned_toolname_argument_ranking_evidence.summary.json
```

Coverage:

- ranking examples: `100`
- usable examples: `99`
- tool-name examples: `31/31` usable
- argument-value examples: `68/69` usable among evidence-candidate spans
- evidence extraction has sequence candidates for only `69/100` planned
  argument-value blocks

Interpretation: pure request/schema evidence extraction is close but still not
complete. It cannot yet replace planner-target candidates for full replay.

Evidence-only missing logical argument blocks: `31`.

Most common missing keys:

| key | missing blocks |
| --- | ---: |
| `date` | `4` |
| `category` | `4` |
| `description` | `4` |
| `due_date` | `3` |
| `location` | `2` |
| `room` | `2` |
| `installation_code` | `2` |
| `date_received` | `2` |

Representative misses:

- copied command phrase: `"Activate security cameras in away mode"`
- empty string argument: `""`
- room/location phrases: `"living room"`, `"front door"`
- enum/boolean values: `"sunny"`, `"energy_saving"`, `true`
- device/security IDs: `"73829SL"`, `"91MHZPIR"`, `"ALRM328SEC"`
- model/product phrase: `"Honeywell SiXPIR"`
- table/list row fields: dates, categories, descriptions, due dates, and
  payment received dates in `expense_data`, `invoice_data`, and `payment_data`

Next candidate extraction work should focus on table-row/list extraction,
quoted/capitalized phrase spans without digit requirements, ID-like values with
mixed letters/digits, enum/boolean schema values, and explicit support for
empty strings when the planner supplies them.

Planner-target-included candidates:

```text
runs/tool_sensitive_block_plans/public_multicall_live_v5_sequence_planned_sampler_schedule_with_candidates_targetincluded.summary.json
data/candidate_ranking/public_multicall_live_v5_sequence_planned_toolname_argument_ranking_targetincluded.summary.json
```

Coverage:

- ranking examples: `131`
- usable examples: `131`
- tool-name examples: `31/31`
- argument-value examples: `100/100`
- target missing from candidates: `0`

The target is the live planner text, not gold assistant text. This makes the
schedule replayable while still separating planner choice from sampler
commitment.

## Selector Gate

Tournament:

```text
runs/candidate_ranking/public_multicall_live_v5_sequence_planned_targetincluded_ckpt275_pairwise_tournament.summary.json
```

Result:

- overall: `131/131`
- argument values: `100/100`
- tool names: `31/31`
- pair comparisons: `1025`
- elapsed: `429.0s`
- max allocated VRAM: `18.46 GiB`
- max reserved VRAM: `27.46 GiB`

Code fix made during this gate:

- `scripts/inject_pairwise_tournament_schedule_choices.py` now skips only
  `None` predictions, not empty strings. Empty-string argument values are valid
  tool-call values; the live planner includes one `location: ""` row.

## Injected Live Schedule

Command:

```bash
.venv-fastdllm/bin/python scripts/inject_pairwise_tournament_schedule_choices.py \
  --schedule-jsonl runs/tool_sensitive_block_plans/public_multicall_live_v5_sequence_planned_sampler_schedule_with_candidates_targetincluded.jsonl \
  --selector-jsonl runs/candidate_ranking/public_multicall_live_v5_sequence_planned_targetincluded_ckpt275_pairwise_tournament.jsonl \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_live_v5_sequence_planned_sampler_schedule_with_pairwise_choices.jsonl \
  --include-kinds tool_name argument_value
```

Injection result:

- selectors consumed: `131`
- selectors correct: `131`
- restricted schedule items: `267`
- candidate missing items: `0`
- records: `12`

Breakdown:

- tool-name selectors: `31`, restricting `152` token-level schedule items
- argument-value selectors: `100`, restricting `115` token-level schedule
  items

## Generation Gate

Command:

```bash
systemd-run --user --scope --quiet -p MemoryMax=28G -p MemorySwapMax=4G bash -lc 'cd /home/mark/qwen_diffusion && CUDA_VISIBLE_DEVICES=0 .venv-fastdllm/bin/python scripts/eval_fastdllm_toolcall_cases.py --base-model models/qwen3.5-9b-fastdllm-init --adapter runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model --tokenizer-path models/qwen3.5-9b-fastdllm-init --input-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl --out-jsonl runs/tool_sensitive_block_plans/public_multicall_live_v5_argsketch_toolargselector_structguard_ckpt275_generation.jsonl --max-new-tokens 560 --conversation-template fast_dllm_v2 --full-context-sampling --sampler-schedule-jsonl runs/tool_sensitive_block_plans/public_multicall_live_v5_sequence_planned_sampler_schedule_with_pairwise_choices.jsonl --force-schedule-token-kinds json_key,json_structure,tool_tag --force-argument-boundary-target-tokens --force-best-candidate-sequence --force-best-tool-name-sequence --ban-argument-boundary-tokens --ban-argument-newline-tokens --stop-after-schedule-tool-calls --constrained-tool-decoding --constrained-sequence-preserving --constrained-max-calls 3 --no-merge-adapter'
```

Summary:

```text
runs/tool_sensitive_block_plans/public_multicall_live_v5_argsketch_toolargselector_structguard_ckpt275_generation.summary.json
```

Result:

- records: `12`
- valid tool JSON: `12/12`
- exact tool sequence: `12/12`
- exact tool-name multiset: `12/12`
- exact arguments: `12/12`
- schema valid: `12/12`
- required args present: `12/12`
- extra calls: `0`
- missing calls: `0`
- stop-boundary trims: `8`
- elapsed: `379.7s`
- generated tokens/sec: `10.83`
- max allocated VRAM: `18.39 GiB`
- max reserved VRAM: `27.90 GiB`

Sampler counters:

- scheduled interval visits: `814`
- scheduled token visits: `863`
- forced structural/tool-tag/key token visits: `844`
- argument candidate sequence force visits: `126` intervals / `459` tokens
- tool-name sequence force visits: `152` intervals / `152` tokens
- semantic candidate model-choice events: `0`, because selector injection
  reduced tool-name and argument-value spans to singleton candidate sequences

## Audit

Command:

```bash
.venv-fastdllm/bin/python scripts/analyze_toolcall_candidate_misses.py \
  --eval-jsonl runs/tool_sensitive_block_plans/public_multicall_live_v5_argsketch_toolargselector_structguard_ckpt275_generation.jsonl \
  --cases-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --schedule-jsonl runs/tool_sensitive_block_plans/public_multicall_live_v5_sequence_planned_sampler_schedule_with_pairwise_choices.jsonl \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_live_v5_argsketch_toolargselector_structguard_ckpt275_candidate_miss_audit.jsonl
```

Audit result:

- failed records: `0`
- mismatches: `0`
- missing calls: `0`
- extra calls: `0`
- invalid tool blocks: `0`

## Interpretation

This is the first clean 12-case protected sampler gate over a non-gold live
planner schedule:

- the planner proposes the tool-call text and sensitive spans;
- the selector chooses tool names and argument values over candidate sequences;
- the sampler commits those semantic spans while deterministic guards handle
  JSON structure, tool tags, and stop boundaries.

Compared with the gold-schedule sampler gate, this removes the biggest
mechanical caveat: sensitive spans no longer come from gold assistant text.

Remaining caveats:

- The planner route is still a deterministic/post-processing sidecar, not raw
  diffusion generation.
- Planner-target-included candidates are needed for full argument coverage;
  evidence-only extraction has sequence candidates for `69/100` planned
  argument-value blocks.
- Structural JSON/tool tags and stop boundaries are still hard guarded.

Next target: close the evidence-only argument candidate gap, then rerun the
live planner sampler without planner-target inclusion. That would make the
selector choose only from request/schema-derived candidates plus available tool
names.

## Evidence-Only Follow-Up

Follow-up result:

```text
qwen35_live_planner_evidence_selector_sampler_result.md
```

The remaining planner-target-inclusion caveat is closed for this public
multi-call slice:

- evidence-only candidate extraction now covers `100/100` planned
  argument-value blocks and `31/31` tool-name blocks.
- row-local markdown table extraction fixes the financial table selector
  misses from the broad evidence candidate set.
- evidence-only selector tournament reaches `131/131`: `100/100` argument
  values and `31/31` tool names.
- injected live schedule consumes `131` correct selectors, restricts `267`
  schedule items, and has `0` candidate misses.
- generation reaches `12/12` valid JSON, `12/12` exact tool sequence, and
  `12/12` exact arguments.
- final audit has `0` failed records and `0` mismatches.

This is still protected runtime evidence, but the semantic candidate path is
now request/schema-derived rather than planner-target-included.
