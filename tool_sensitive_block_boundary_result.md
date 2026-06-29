# Tool-Sensitive Block Boundary Prototype Result

Date: 2026-06-27

## Purpose

Prototype the dynamic block-boundary idea for agentic diffusion decoding:
ordinary prose can use larger diffusion blocks, while tool-call regions should
be split into smaller constrained spans.

Implementation:

- script: `scripts/plan_tool_sensitive_blocks.py`
- output root: `runs/tool_sensitive_block_plans/`
- current unit: character spans over known assistant text
- next unit: Qwen tokenizer-offset spans for sampler integration

## Span Policy

The prototype emits these region kinds:

- `prose`: large blocks, light/no constraint
- `tool_tag`: literal tiny block for `<tool_call>` and `</tool_call>`
- `tool_name`: tiny block constrained to available tool-name enum
- `json_key`: tiny block constrained to schema keys
- `argument_value`: small block with schema and request-evidence constraints
- `json_structure`: small block with JSON grammar constraints

## Research Readout

Current public research supports the pieces of this design, but I do not see a
mature public recipe that already solves agentic tool-call-aware diffusion
decoding end to end.

Relevant pieces:

- AR-to-block-diffusion conversion: Fast-dLLM v2 and NBDiff-like work support
  preserving an AR model while growing from next-token to block generation.
- Adaptive block sizing: Swordsman, AdaBlock-dLLM, DSB, and SemBlock support
  changing block boundaries based on entropy, confidence dynamics, sliding
  active blocks, or learned semantic boundary predictors.
- Constrained diffusion decoding: DINGO, CFG-constrained diffusion decoding,
  and Lookahead-then-Verify support keeping diffusion samples inside grammar or
  regex-like constraints.

The missing piece is the combination: tool-call-aware dynamic block boundaries
plus JSON/schema/tool-name/stop constraints plus behavior preservation against
an agentic AR teacher. That is the project-specific innovation target.

Implication for this prototype:

- `plan_tool_sensitive_blocks.py` is currently rule-based, not learned.
- `--sampler-schedule-jsonl` is constrained scheduling, not weight training.
- `--force-tool-call-prefix` is prefix control, not evidence that the adapter
  learned the behavior.
- A later SemBlock-like boundary predictor would be learned, but it should be
  trained/evaluated against agentic spans rather than generic sentence chunks.
- Tool-sensitive boundaries should be behavior-preserving: use large blocks
  where prose can vary, small blocks where wrong tokens change actions, and
  grammar/schema/candidate constraints where partial outputs must stay
  completable.

## Smoke Results

Gold public one-call, first 8 records:

```bash
.venv-fastdllm/bin/python scripts/plan_tool_sensitive_blocks.py \
  --input-jsonl data/toolcall_eval/public_onecall_hermes_smoke.jsonl \
  --out-jsonl runs/tool_sensitive_block_plans/public_onecall_hermes_gold_blocks.jsonl \
  --limit 8
```

- records: `8`
- tool calls: `8`
- segments: `346`
- tool-name spans: `8`
- JSON-key spans: `86`
- argument-value spans: `67`
- records without segments: `0`

Gold public multi-call, 12 records:

```bash
.venv-fastdllm/bin/python scripts/plan_tool_sensitive_blocks.py \
  --input-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_hermes_gold_blocks.jsonl \
  --limit 12
```

- records: `12`
- tool calls: `31`
- segments: `712`
- tool-name spans: `31`
- JSON-key spans: `169`
- argument-value spans: `100`
- records without segments: `0`

Live raw public one-call, 8 generated records:

```bash
.venv-fastdllm/bin/python scripts/plan_tool_sensitive_blocks.py \
  --input-jsonl runs/qwen35_9b_split_route_sidecar_eval/live_smoke_public_onecall/public_one_call/public_onecall_8.jsonl \
  --text-field assistant \
  --out-jsonl runs/tool_sensitive_block_plans/public_onecall_live_raw_blocks.jsonl
```

- records: `8`
- parsed tool calls: `8`
- records with parsed tool calls: `6`
- segments: `154`
- tool-name spans: `5`
- JSON-key spans: `35`
- argument-value spans: `20`

Live raw public multi-call, 12 generated records:

```bash
.venv-fastdllm/bin/python scripts/plan_tool_sensitive_blocks.py \
  --input-jsonl runs/qwen35_9b_split_route_sidecar_eval/live_smoke_public_multicall_planner/public_multi_call_planner/public_multicall_12.jsonl \
  --text-field assistant \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_live_raw_blocks.jsonl
```

- records: `12`
- parsed tool calls: `25`
- records with parsed tool calls: `11`
- segments: `515`
- tool-name spans: `28`
- JSON-key spans: `117`
- argument-value spans: `59`

Live sequence-planner protected public multi-call, 12 records:

```bash
.venv-fastdllm/bin/python scripts/plan_tool_sensitive_blocks.py \
  --input-jsonl runs/qwen35_9b_split_route_sidecar_eval/live_smoke_public_multicall_planner/public_multi_call_planner/public_multicall_12_sequence_planner_segmentargs_v3.jsonl \
  --text-field sequence_planner_assistant \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_blocks.jsonl
```

- records: `12`
- parsed tool calls: `31`
- records with parsed tool calls: `12`
- segments: `708`
- tool-name spans: `31`
- JSON-key spans: `168`
- argument-value spans: `99`

## Tokenizer-Offset Handoff

The planner now supports Qwen tokenizer offsets:

```bash
.venv-fastdllm/bin/python scripts/plan_tool_sensitive_blocks.py \
  --input-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_hermes_gold_blocks_tokenized.jsonl \
  --limit 12 \
  --tokenizer-path models/qwen3.5-9b-fastdllm-init
```

Each record now includes:

- `segments`: exact character-level semantic spans
- `token_count`: Qwen tokenizer length for the assistant text
- `token_blocks`: non-overlapping sampler-ready token ranges, assigned to the
  character span they overlap most

Tokenized gold public one-call, first 8 records:

- tokens: `710`
- token blocks: `343`
- tool-tag blocks/tokens: `16` / `32`
- tool-name blocks/tokens: `8` / `46`
- JSON-key blocks/tokens: `86` / `211`
- argument-value blocks/tokens: `67` / `219`

Tokenized gold public multi-call, 12 records:

- tokens: `1652`
- token blocks: `710`
- tool-tag blocks/tokens: `62` / `124`
- tool-name blocks/tokens: `31` / `152`
- JSON-key blocks/tokens: `169` / `408`
- argument-value blocks/tokens: `100` / `525`

Tokenized live raw public multi-call, 12 generated records:

- tokens: `2355`
- token blocks: `475`
- parsed tool calls: `25`
- records with parsed tool calls: `11/12`
- tool-name blocks/tokens: `28` / `149`
- argument-value blocks/tokens: `59` / `280`

Tokenized live sequence-planner protected public multi-call, 12 records:

- tokens: `1520`
- token blocks: `626`
- parsed tool calls: `31`
- records with parsed tool calls: `12/12`
- tool-name blocks/tokens: `31` / `153`
- argument-value blocks/tokens: `99` / `507`

## Sampler Schedule Dry Run

The next handoff script consumes tokenized plans and emits per-token decoding
chunks:

```bash
.venv-fastdllm/bin/python scripts/emit_tool_sensitive_sampler_schedule.py \
  --input-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_blocks_tokenized.jsonl \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_sampler_schedule.jsonl
```

Default chunk limits:

- prose: `128` tokens
- argument values: `8` tokens
- JSON structure: `4` tokens
- tiny regions (`tool_tag`, `json_key`, `tool_name`): `1` token

Gold public multi-call schedule:

- source token blocks: `710`
- scheduled decoding blocks: `1163`
- one-step/literal-or-key tokens: `532`
- two-step tool-name tokens: `152`
- three-step JSON-structure tokens: `424`
- eight-step argument-value tokens: `525`

Live raw public multi-call schedule:

- source token blocks: `475`
- scheduled decoding blocks: `926`
- one-step/literal-or-key tokens: `392`
- two-step tool-name tokens: `149`
- three-step JSON-structure tokens: `606`
- eight-step argument-value tokens: `280`
- prose tokens: `928`

Live sequence-planner protected public multi-call schedule:

- source token blocks: `626`
- scheduled decoding blocks: `1076`
- one-step/literal-or-key tokens: `529`
- two-step tool-name tokens: `153`
- three-step JSON-structure tokens: `312`
- eight-step argument-value tokens: `507`
- prose tokens: `19`

Readout:

- The tokenized representation preserves the earlier raw/protected boundary
  signal: raw public multi-call exposes `25` parsed tool calls, while protected
  sequence-planner output recovers the gold `31`.
- BPE tokens often absorb nearby punctuation, so token blocks are sampler-safe
  ownership ranges rather than perfect semantic spans. Character `segments`
  remain the exact semantic audit source.
- The raw output is longer (`2355` tokens) but has fewer structured tool spans
  than the protected output (`1520` tokens). That is a direct measure of
  diffusion-agent failure: extra prose/malformed structure consumes tokens
  instead of producing valid action spans.
- The dry-run schedule converts that into test-time compute: raw output spends
  `928` tokens in prose regions, while the protected action plan spends only
  `19` prose tokens and shifts compute into argument-value and schema/tool-name
  regions.

## Fast-DLLM Sampler Trace

The non-generating Fast-DLLM trace maps generated-token-relative schedule chunks
onto absolute sequence positions:

```bash
.venv-fastdllm/bin/python scripts/trace_tool_sensitive_fastdllm_schedule.py \
  --schedule-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_sampler_schedule.jsonl \
  --cases-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --tokenizer-path models/qwen3.5-9b-fastdllm-init \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_fastdllm_trace.jsonl \
  --conversation-template fast_dllm_v2 \
  --block-size 32 \
  --small-block-size 8
```

It does not load the model or generate. It tokenizes the same prompt template,
adds the prompt length to each generated-token schedule span, and splits spans
at Fast-DLLM `block_size=32` and `small_block_size=8` boundaries.

Public multi-call Fast-DLLM trace:

| Output | Generated tokens | Schedule blocks | Fast-DLLM pieces | Crosses block | Crosses small block | Prose tokens | Argument tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gold | 1652 | 1163 | 1217 | 9 | 54 | 19 | 525 |
| raw live | 2355 | 926 | 1105 | 28 | 80 | 928 | 280 |
| protected live | 1520 | 1076 | 1137 | 15 | 61 | 19 | 507 |

Readout:

- Raw live output is structurally worse and mechanically harder to schedule:
  it crosses `28` Fast-DLLM block boundaries and `80` small-block boundaries,
  versus gold `9` / `54`.
- Protected output recovers the action structure and is closer to gold:
  `15` block crossings and `61` small-block crossings.
- This is the first sampler-integration risk metric. A generation-time
  tool-sensitive sampler should reduce raw block-crossing fragmentation while
  increasing structured argument/tool span coverage.

## Scheduled Sampler Override Smoke

The first actual generation-path override is implemented in
`scripts/eval_fastdllm_toolcall_cases.py` behind the opt-in flag
`--sampler-schedule-jsonl`. It only applies to `--full-context-sampling`.

Mechanics:

- existing Fast-DLLM block commits remain aligned to `block_size`
- the inner denoising windows are no longer uniformly `small_block_size`
- schedule chunks replace uniform windows inside the current block
- tool tags, function names, and schema keys can become one-token intervals
- argument values can use their schedule chunk size
- current implementation uses scheduled interval sizes; it records but does not
  yet enforce repeated re-masking from `denoise_steps`

Smoke command:

```bash
systemd-run --user --scope --quiet -p MemoryMax=28G -p MemorySwapMax=4G \
  .venv-fastdllm/bin/python scripts/eval_fastdllm_toolcall_cases.py \
  --base-model models/qwen3.5-9b-fastdllm-init \
  --adapter runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model \
  --tokenizer-path models/qwen3.5-9b-fastdllm-init \
  --input-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_scheduled_sampler_smoke1.jsonl \
  --limit 1 \
  --block-size 32 \
  --small-block-size 8 \
  --max-new-tokens 160 \
  --threshold 0.9 \
  --temperature 0.0 \
  --top-p 0.95 \
  --conversation-template fast_dllm_v2 \
  --full-context-sampling \
  --sampler-schedule-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_sampler_schedule.jsonl \
  --repair-mode none \
  --no-merge-adapter
```

Smoke result:

- output: `runs/tool_sensitive_block_plans/public_multicall_scheduled_sampler_smoke1.jsonl`
- records: `1`
- status: `ok`
- generated tokens: `160`
- elapsed: `27.5989s`
- generated tokens/s: `5.7973`
- max allocated / reserved VRAM: `17.95` / `24.14` GiB
- schedule used: `true`
- scheduled interval visits: `132`
- default interval visits: `23`
- scheduled token visits: `256`
- default token visits: `90`
- strict tool score: `0/1` valid / sequence / arguments

Readout:

- The scheduled sampler override is mechanically live: generation completed and
  visited scheduled intervals.
- This smoke is not a quality win. The output started with thinking/prose and
  missed all three tool calls under strict scoring.
- The next sampler change should constrain mode/format before or during the
  first scheduled tool-tag interval, otherwise schedule-aware windows still
  spend compute on a prose trajectory.

## Forced Tool-Call Prefix Smoke

Generation-time format control is now available through:

- `--force-tool-call-prefix`: fixes `<tool_call>\n` as the first assistant
  tokens before sampling.
- `--forced-assistant-prefix <text>`: appends an arbitrary fixed assistant
  prefix before sampling.

Implementation detail: the forced prefix is appended to the model input as fixed
assistant context, but output decoding still starts from the original prompt
length, so strict scoring sees the prefix as part of the assistant response.

Smoke command:

```bash
systemd-run --user --scope --quiet -p MemoryMax=28G -p MemorySwapMax=4G \
  .venv-fastdllm/bin/python scripts/eval_fastdllm_toolcall_cases.py \
  --base-model models/qwen3.5-9b-fastdllm-init \
  --adapter runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model \
  --tokenizer-path models/qwen3.5-9b-fastdllm-init \
  --input-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_forced_prefix_scheduled_sampler_smoke1.jsonl \
  --limit 1 \
  --block-size 32 \
  --small-block-size 8 \
  --max-new-tokens 160 \
  --threshold 0.9 \
  --temperature 0.0 \
  --top-p 0.95 \
  --conversation-template fast_dllm_v2 \
  --full-context-sampling \
  --sampler-schedule-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_sampler_schedule.jsonl \
  --force-tool-call-prefix \
  --repair-mode none \
  --no-merge-adapter
```

Result:

- output:
  `runs/tool_sensitive_block_plans/public_multicall_forced_prefix_scheduled_sampler_smoke1.jsonl`
- records: `1`
- status: `ok`
- generated tokens: `160`
- elapsed: `25.4835s`
- generated tokens/s: `6.2786`
- max allocated / reserved VRAM: `17.95` / `24.14` GiB
- schedule used: `true`
- scheduled interval visits: `126`
- default interval visits: `17`
- scheduled token visits: `251`
- default token visits: `68`
- strict tool score: `0/1` valid / sequence / arguments
- parsed called names: `get_camera_live_feed`, `get_recorded_feed`

Readout:

- The forced prefix fixed the previous `<think>`/prose trajectory. The output
  begins directly with `<tool_call>` and emits plausible tool-call blocks.
- It still fails strict scoring because the second call has malformed JSON
  (`"camera_id "front_door"` missing a colon) and generation continues into an
  extra `<tool_call>\nuser...` fragment after the third call.
- This is a better failure mode than prose drift: the next generation-time
  constraints are now precise.

Next constraints:

- JSON-key/value grammar inside `"arguments"` so keys cannot absorb separators.
- Stop-boundary control after the planned number of `</tool_call>` blocks.
- Optional literal forcing for `</tool_call>` once the tool-count target is
  reached.

## Gold-Schedule Candidate-Deferred 12-Case Ablation

The strongest current protected scheduled sampler run uses the gold public
multi-call schedule, forced structural/tool-tag tokens, whole-candidate argument
selection, deferred tool-name candidate commitment, stop guarding, and
sequence-preserving constrained projection:

```bash
systemd-run --user --scope --quiet -p MemoryMax=28G -p MemorySwapMax=4G \
  .venv-fastdllm/bin/python scripts/eval_fastdllm_toolcall_cases.py \
  --base-model models/qwen3.5-9b-fastdllm-init \
  --adapter runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model \
  --tokenizer-path models/qwen3.5-9b-fastdllm-init \
  --input-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_gold_schedule_toolname_candidate_deferred_12.jsonl \
  --limit 12 \
  --block-size 32 \
  --small-block-size 8 \
  --max-new-tokens 560 \
  --threshold 0.9 \
  --temperature 0.0 \
  --top-p 0.95 \
  --conversation-template fast_dllm_v2 \
  --full-context-sampling \
  --sampler-schedule-jsonl runs/tool_sensitive_block_plans/public_multicall_gold_sampler_schedule_with_toolname_candidates_v3_12.jsonl \
  --force-schedule-token-kinds tool_tag,json_key,json_structure \
  --force-best-tool-name-sequence \
  --force-argument-boundary-target-tokens \
  --force-best-candidate-sequence \
  --force-selected-candidate-tokens \
  --ban-argument-boundary-tokens \
  --ban-argument-newline-tokens \
  --stop-after-schedule-tool-calls \
  --constrained-tool-decoding \
  --constrained-sequence-preserving \
  --constrained-assume-utc-z \
  --repair-mode none \
  --no-merge-adapter
```

Result summary:

- output:
  `runs/tool_sensitive_block_plans/public_multicall_gold_schedule_toolname_candidate_deferred_12.jsonl`
- records: `12`
- raw valid tool JSON: `11/12`
- raw exact tool sequence: `11/12`
- raw exact arguments: `3/12`
- constrained valid tool JSON: `12/12`
- constrained exact tool sequence: `12/12`
- constrained exact arguments: `4/12`
- all constrained schemas valid and required arguments present: `12/12`
- stop-boundary trims: `8`
- tool-name deferred shared-prefix choices: `2`
- max allocated / reserved VRAM: `18.49` / `28.70` GiB

Readout:

- Deferred candidate commitment fixes the shared-prefix tool-name failure mode;
  the sampler no longer commits to the wrong tool just because two tool names
  share an initial Qwen token.
- Structural/tool-name constraints are now strong enough to make the 12-case
  protected sequence score perfect under this gold-schedule ablation.
- Exact arguments remain weak. The remaining failures are value grounding and
  row alignment: repeated substrings, paired device IDs, multi-row tables,
  start/end times, invoice/client IDs, and complex array/object payloads.
- This is not a promoted model-only result. It is the clearest current scaffold
  blueprint for what the model or a learned side objective must internalize.

## Stop-Guard Plus Sequence Projection Smoke

The evaluator now has opt-in controls for the two forced-prefix failure modes:

- `--stop-after-tool-calls N`: trim decoded assistant text after `N` complete
  `</tool_call>` blocks.
- `--stop-after-schedule-tool-calls`: use the schedule row's planned
  `tool_call_count` as the stop target.
- `--stop-after-gold-tool-calls`: eval-only oracle stop count.
- `--constrained-sequence-preserving`: repair each complete `<tool_call>` body
  in generated order before constrained projection.
- `--constrained-assume-utc-z`: append `Z` to naive ISO-8601 datetime strings
  in time-like fields during constrained projection.

Implementation detail: this is still a protected runtime path, not evidence
that the raw diffusion model learned JSON separators or stopping. It is useful
because it isolates the next model/sampler failure after removing extra
continuation, malformed middle-call parsing, and simple timestamp normalization.

Smoke command:

```bash
systemd-run --user --scope --quiet -p MemoryMax=28G -p MemorySwapMax=4G \
  .venv-fastdllm/bin/python scripts/eval_fastdllm_toolcall_cases.py \
  --base-model models/qwen3.5-9b-fastdllm-init \
  --adapter runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model \
  --tokenizer-path models/qwen3.5-9b-fastdllm-init \
  --input-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_forced_prefix_stopguard_seqproject_smoke1.jsonl \
  --limit 1 \
  --block-size 32 \
  --small-block-size 8 \
  --max-new-tokens 160 \
  --threshold 0.9 \
  --temperature 0.0 \
  --top-p 0.95 \
  --conversation-template fast_dllm_v2 \
  --full-context-sampling \
  --sampler-schedule-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_sampler_schedule.jsonl \
  --force-tool-call-prefix \
  --stop-after-schedule-tool-calls \
  --constrained-tool-decoding \
  --constrained-sequence-preserving \
  --constrained-max-calls 3 \
  --repair-mode none \
  --no-merge-adapter
```

Result:

- output:
  `runs/tool_sensitive_block_plans/public_multicall_forced_prefix_stopguard_seqproject_smoke1.jsonl`
- records: `1`
- status: `ok`
- generated tokens: `160`
- elapsed: `25.4612s`
- generated tokens/s: `6.2841`
- max allocated / reserved VRAM: `17.95` / `24.14` GiB
- schedule used: `true`
- scheduled interval visits: `126`
- default interval visits: `17`
- stop-boundary guard trimmed: `1/1`
- raw strict score: `0/1` valid / sequence / arguments
- protected constrained score:
  - valid JSON/tool calls: `1/1`
  - exact tool sequence: `1/1`
  - exact argument match: `0/1`
  - schema valid and required args present: `1/1`

Readout:

- The stop guard removed the extra `<tool_call>\nuser...` continuation after
  the third intended call.
- The sequence-preserving projection repaired the malformed middle call
  (`"camera_id "front_door"` missing a colon) and recovered the generated tool
  order: `get_camera_live_feed`, `record_camera_feed`, `get_recorded_feed`.
- Exact arguments still fail because the generated/protected timestamps omit
  the `Z` suffix expected by the gold answer. That is now the next isolated
  value-normalization/value-copy failure.

UTC-normalized protected smoke:

```bash
systemd-run --user --scope --quiet -p MemoryMax=28G -p MemorySwapMax=4G \
  .venv-fastdllm/bin/python scripts/eval_fastdllm_toolcall_cases.py \
  --base-model models/qwen3.5-9b-fastdllm-init \
  --adapter runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model \
  --tokenizer-path models/qwen3.5-9b-fastdllm-init \
  --input-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_forced_prefix_stopguard_seqproject_utc_smoke1.jsonl \
  --limit 1 \
  --block-size 32 \
  --small-block-size 8 \
  --max-new-tokens 160 \
  --threshold 0.9 \
  --temperature 0.0 \
  --top-p 0.95 \
  --conversation-template fast_dllm_v2 \
  --full-context-sampling \
  --sampler-schedule-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_sampler_schedule.jsonl \
  --force-tool-call-prefix \
  --stop-after-schedule-tool-calls \
  --constrained-tool-decoding \
  --constrained-sequence-preserving \
  --constrained-assume-utc-z \
  --constrained-max-calls 3 \
  --repair-mode none \
  --no-merge-adapter
```

Result:

- output:
  `runs/tool_sensitive_block_plans/public_multicall_forced_prefix_stopguard_seqproject_utc_smoke1.jsonl`
- records: `1`
- status: `ok`
- generated tokens: `160`
- elapsed: `25.4605s`
- generated tokens/s: `6.2842`
- max allocated / reserved VRAM: `17.95` / `24.14` GiB
- schedule used: `true`
- scheduled interval visits: `126`
- default interval visits: `17`
- stop-boundary guard trimmed: `1/1`
- raw strict score: `0/1` valid / sequence / arguments
- protected constrained score:
  - valid JSON/tool calls: `1/1`
  - exact tool sequence: `1/1`
  - exact argument match: `1/1`
  - schema valid and required args present: `1/1`

Readout:

- The protected path now fully rescues this one public multi-call example after
  forced prefix, schedule replay, stop guarding, sequence-preserving projection,
  and opt-in UTC timestamp normalization.
- This is not a raw model improvement. The raw output still has malformed JSON
  in the second call and misses strict sequence/arguments.
- The next useful step is to move the repaired pieces into generation-time
  constraints: JSON key/value separators, tool-count stop control, and
  timestamp/value-copy constraints before token commit.

## Schedule Target-Token Forcing Smokes

The scheduler can now carry target token IDs from tokenized block plans:

```bash
.venv-fastdllm/bin/python scripts/plan_tool_sensitive_blocks.py \
  --input-jsonl runs/qwen35_9b_split_route_sidecar_eval/live_smoke_public_multicall_planner/public_multi_call_planner/public_multicall_12_sequence_planner_segmentargs_v3.jsonl \
  --text-field sequence_planner_assistant \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_blocks_tokenized_with_ids.jsonl \
  --limit 1 \
  --tokenizer-path models/qwen3.5-9b-fastdllm-init \
  --include-token-ids

.venv-fastdllm/bin/python scripts/emit_tool_sensitive_sampler_schedule.py \
  --input-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_blocks_tokenized_with_ids.jsonl \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_sampler_schedule_with_ids_1.jsonl \
  --include-token-ids
```

The evaluator can consume those targets through:

- `--force-schedule-token-kinds <kinds>`: comma/space separated schedule kinds
  whose `target_token_ids` are hard-filled during full-context sampling.

This is a sampler-time forcing mechanism, not a learned model improvement. If
the schedule comes from a protected sequence planner or gold answer, it is an
oracle/protected path. Its value is that it tests exact token alignment and
separates "can the sampler enforce this span?" from "can the diffusion model
choose this span itself?"

### Structural-Only Force

Command:

```bash
systemd-run --user --scope --quiet -p MemoryMax=28G -p MemorySwapMax=4G \
  .venv-fastdllm/bin/python scripts/eval_fastdllm_toolcall_cases.py \
  --base-model models/qwen3.5-9b-fastdllm-init \
  --adapter runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model \
  --tokenizer-path models/qwen3.5-9b-fastdllm-init \
  --input-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_schedule_forced_structure_smoke1.jsonl \
  --limit 1 \
  --block-size 32 \
  --small-block-size 8 \
  --max-new-tokens 160 \
  --threshold 0.9 \
  --temperature 0.0 \
  --top-p 0.95 \
  --conversation-template fast_dllm_v2 \
  --full-context-sampling \
  --sampler-schedule-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_sampler_schedule_with_ids_1.jsonl \
  --force-schedule-token-kinds tool_tag,json_key,json_structure,tool_name \
  --stop-after-schedule-tool-calls \
  --constrained-tool-decoding \
  --constrained-sequence-preserving \
  --constrained-assume-utc-z \
  --constrained-max-calls 3 \
  --repair-mode none \
  --no-merge-adapter
```

Result:

- output:
  `runs/tool_sensitive_block_plans/public_multicall_schedule_forced_structure_smoke1.jsonl`
- forced schedule tokens: `77`
- forced schedule intervals: `77`
- generated tokens/s: `11.1291`
- max allocated / reserved VRAM: `17.95` / `24.14` GiB
- raw strict score: `0/1` valid / sequence / arguments
- protected constrained score:
  - valid JSON/tool calls: `1/1`
  - exact tool sequence: `0/1`
  - exact arguments: `0/1`

Readout:

- Structural target forcing prevents pure prose drift and forces many tool
  anchors, but it is not enough.
- The model generated an early `</tool_call>` inside an unforced argument value
  (`stream_quality`), which made later forced structure semantically misaligned.
- Implication: generation-time JSON constraints need value-span constraints too.
  It is unsafe to only force punctuation/keys/tool names while leaving values
  free to emit structural delimiters.

### Oracle All-Schedule Force

Command:

```bash
systemd-run --user --scope --quiet -p MemoryMax=28G -p MemorySwapMax=4G \
  .venv-fastdllm/bin/python scripts/eval_fastdllm_toolcall_cases.py \
  --base-model models/qwen3.5-9b-fastdllm-init \
  --adapter runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model \
  --tokenizer-path models/qwen3.5-9b-fastdllm-init \
  --input-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_schedule_forced_all_oracle_smoke1.jsonl \
  --limit 1 \
  --block-size 32 \
  --small-block-size 8 \
  --max-new-tokens 160 \
  --threshold 0.9 \
  --temperature 0.0 \
  --top-p 0.95 \
  --conversation-template fast_dllm_v2 \
  --full-context-sampling \
  --sampler-schedule-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_sampler_schedule_with_ids_1.jsonl \
  --force-schedule-token-kinds tool_tag,json_key,json_structure,tool_name,argument_value,prose \
  --stop-after-schedule-tool-calls \
  --constrained-tool-decoding \
  --constrained-sequence-preserving \
  --constrained-assume-utc-z \
  --constrained-max-calls 3 \
  --repair-mode none \
  --no-merge-adapter
```

Result:

- output:
  `runs/tool_sensitive_block_plans/public_multicall_schedule_forced_all_oracle_smoke1.jsonl`
- forced schedule tokens: `137`
- forced schedule intervals: `93`
- generated tokens/s: `74.6019`
- max allocated / reserved VRAM: `17.94` / `18.55` GiB
- raw strict score:
  - valid JSON/tool calls: `1/1`
  - exact tool sequence: `1/1`
  - exact arguments: `1/1`
- protected constrained score:
  - valid JSON/tool calls: `1/1`
  - exact tool sequence: `1/1`
  - exact arguments: `1/1`

Readout:

- The target-token schedule is correctly aligned: when all scheduled spans are
  forced, the sampler emits the planned tool-call output exactly.
- This is an oracle upper bound, not a deployable policy. It proves the sampler
  can enforce scheduled tokens; it does not prove the model can choose them.
- The next useful non-oracle step is constrained value decoding: for argument
  spans, ban structural delimiters unless the value grammar permits them, and
  restrict values to schema/evidence candidate sets when possible.

### Argument-Value Boundary Bans

The evaluator now has two additional argument-value controls:

- `--force-argument-boundary-target-tokens`: inside `argument_value` intervals,
  hard-fill target tokens that decode as boundary-only JSON pieces, such as
  `","`. This handles BPE tokens that are assigned to the value span even
  though they carry closing JSON structure.
- `--ban-argument-boundary-tokens`: inside `argument_value` intervals, ban
  tool/role boundary token IDs from logits.
- `--ban-argument-json-boundary-tokens`: with the previous flag, also ban the
  schedule-derived boundary-only JSON target IDs at non-forced value positions.
- `--ban-argument-newline-tokens`: with the previous flag, also ban newline
  tokens inside scalar value intervals.

Smoke command:

```bash
systemd-run --user --scope --quiet -p MemoryMax=28G -p MemorySwapMax=4G \
  .venv-fastdllm/bin/python scripts/eval_fastdllm_toolcall_cases.py \
  --base-model models/qwen3.5-9b-fastdllm-init \
  --adapter runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model \
  --tokenizer-path models/qwen3.5-9b-fastdllm-init \
  --input-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_schedule_forced_structure_argban_jsonnewline_smoke1.jsonl \
  --limit 1 \
  --block-size 32 \
  --small-block-size 8 \
  --max-new-tokens 160 \
  --threshold 0.9 \
  --temperature 0.0 \
  --top-p 0.95 \
  --conversation-template fast_dllm_v2 \
  --full-context-sampling \
  --sampler-schedule-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_sampler_schedule_with_ids_1.jsonl \
  --force-schedule-token-kinds tool_tag,json_key,json_structure,tool_name \
  --force-argument-boundary-target-tokens \
  --ban-argument-boundary-tokens \
  --ban-argument-json-boundary-tokens \
  --ban-argument-newline-tokens \
  --stop-after-schedule-tool-calls \
  --constrained-tool-decoding \
  --constrained-sequence-preserving \
  --constrained-assume-utc-z \
  --constrained-max-calls 3 \
  --repair-mode none \
  --no-merge-adapter
```

Result:

- output:
  `runs/tool_sensitive_block_plans/public_multicall_schedule_forced_structure_argban_jsonnewline_smoke1.jsonl`
- forced structural/tool/key tokens: `77`
- forced argument boundary tokens: `4`
- argument-boundary ban visits: `50` intervals / `167` masked tokens
- generated tokens/s: `12.0211`
- max allocated / reserved VRAM: `17.95` / `24.14` GiB
- raw strict score: `0/1` valid / sequence / arguments
- protected constrained score:
  - valid JSON/tool calls: `1/1`
  - exact tool sequence: `1/1`
  - exact arguments: `0/1`

Readout:

- This improves over structural-only forcing: the protected path recovers the
  full three-call sequence instead of only two calls.
- Raw strict scoring still fails. The first two raw tool calls are structurally
  good, but the third call still has value-copy corruption in the datetime
  fields and extra JSON text around `end_time`.
- The constrained projection also still misses exact arguments because the
  malformed third body causes the parser to fall back to context extraction for
  `camera_id`, choosing `front_door` instead of the generated `front_garden`.
- The next non-oracle sampler step is not more delimiter banning. It is
  candidate-constrained value decoding for scalar spans, especially dates,
  camera IDs, paths, numbers, and enums extracted from prompt evidence.

### Argument-Value Candidate Diagnostic

The planner and scheduler now preserve semantic metadata needed for candidate
value constraints:

- `tool_call_index`
- `json_key`
- `target_text`
- `segment_texts`

Regenerate the one-case metadata-rich schedule:

```bash
.venv-fastdllm/bin/python scripts/plan_tool_sensitive_blocks.py \
  --input-jsonl runs/qwen35_9b_split_route_sidecar_eval/live_smoke_public_multicall_planner/public_multi_call_planner/public_multicall_12_sequence_planner_segmentargs_v3.jsonl \
  --text-field sequence_planner_assistant \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_blocks_tokenized_with_ids_meta.jsonl \
  --limit 1 \
  --tokenizer-path models/qwen3.5-9b-fastdllm-init \
  --include-token-ids

.venv-fastdllm/bin/python scripts/emit_tool_sensitive_sampler_schedule.py \
  --input-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_blocks_tokenized_with_ids_meta.jsonl \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_sampler_schedule_with_ids_meta_1.jsonl \
  --include-token-ids
```

Run the diagnostic:

```bash
.venv-fastdllm/bin/python scripts/diagnose_schedule_value_candidates.py \
  --schedule-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_sampler_schedule_with_ids_meta_1.jsonl \
  --cases-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_value_candidate_diagnostic_1.jsonl
```

Result:

- argument values: `7`
- deterministic exact candidate choice: `2/7`
- target value present in evidence candidate set: `7/7`
- output:
  `runs/tool_sensitive_block_plans/public_multicall_value_candidate_diagnostic_1.jsonl`

Refined paired-datetime diagnostic:

- output:
  `runs/tool_sensitive_block_plans/public_multicall_value_candidate_diagnostic_2.jsonl`
- deterministic exact candidate choice: `4/7`
- newly exact spans: `start_time=2023-04-22T15:00:00Z` and
  `end_time=2023-04-22T17:00:00Z`

Readout:

- The current context extractor is weak as a single-value chooser. It chooses
  exactly only `stream_quality=1080p` and `duration=30`.
- It is strong enough as a candidate generator for this case: the evidence set
  contains both camera IDs (`front_door`, `front_garden`) and both datetime
  candidates (`2023-04-22T15:00:00Z`, `2023-04-22T17:00:00Z`).
- This supports the next sampler design: candidate-constrained value decoding
  should restrict each value span to prompt-evidence candidates, while leaving
  the model to choose the correct candidate under the current tool-call context.

### Candidate-Constrained Value Decoding Smoke

The candidate diagnostic now feeds a sampler-consumable augmented schedule:

```bash
.venv-fastdllm/bin/python scripts/augment_schedule_value_candidates.py \
  --schedule-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_sampler_schedule_with_ids_meta_1.jsonl \
  --cases-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --tokenizer-path models/qwen3.5-9b-fastdllm-init \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_sampler_schedule_with_candidates_1.jsonl
```

Augmentation result:

- argument blocks augmented: `7`
- evidence candidate values: `26`
- candidate-constrained positions: `58`
- multi-token-choice positions: `12`
- blocks with a deterministic selected candidate: `2`

The paired-datetime refinement writes
`runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_sampler_schedule_with_candidates_2.jsonl`
and keeps the same `7` argument blocks / `13` whole-candidate sequences, while
raising deterministic selected argument blocks from `2` to `4`.

The evaluator has two new value-span controls:

- `--constrain-argument-candidate-tokens`: mask each argument-value position to
  candidate token IDs from the augmented schedule.
- `--force-selected-candidate-tokens`: when the extractor has a concrete
  selected candidate, hard-fill that selected candidate's token IDs. This is
  not oracle gold forcing; it uses the extractor's chosen candidate. It is still
  a protected/runtime control, not learned raw behavior.

Candidate-set-only smoke:

```bash
systemd-run --user --scope --quiet -p MemoryMax=28G -p MemorySwapMax=4G \
  .venv-fastdllm/bin/python scripts/eval_fastdllm_toolcall_cases.py \
  --base-model models/qwen3.5-9b-fastdllm-init \
  --adapter runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model \
  --tokenizer-path models/qwen3.5-9b-fastdllm-init \
  --input-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_schedule_candidate_constrained_smoke1.jsonl \
  --limit 1 \
  --block-size 32 \
  --small-block-size 8 \
  --max-new-tokens 160 \
  --threshold 0.9 \
  --temperature 0.0 \
  --top-p 0.95 \
  --conversation-template fast_dllm_v2 \
  --full-context-sampling \
  --sampler-schedule-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_sampler_schedule_with_candidates_1.jsonl \
  --force-schedule-token-kinds tool_tag,json_key,json_structure,tool_name \
  --force-argument-boundary-target-tokens \
  --constrain-argument-candidate-tokens \
  --ban-argument-boundary-tokens \
  --ban-argument-newline-tokens \
  --stop-after-schedule-tool-calls \
  --constrained-tool-decoding \
  --constrained-sequence-preserving \
  --constrained-assume-utc-z \
  --constrained-max-calls 3 \
  --repair-mode none \
  --no-merge-adapter
```

Result:

- output:
  `runs/tool_sensitive_block_plans/public_multicall_schedule_candidate_constrained_smoke1.jsonl`
- raw valid JSON/tool calls: `1/1`
- raw exact tool sequence: `1/1`
- raw exact arguments: `0/1`
- candidate-constrained token visits: `57`
- generated tokens/s: `27.1320`
- max allocated / reserved VRAM: `17.94` / `22.94` GiB

Readout:

- Candidate-set masking alone fixes the structural problem: raw output is valid
  Qwen tool-call JSON with the exact three-call sequence.
- It still allows invalid recombinations across candidate tokens. In this smoke,
  `stream_quality` became `100pp`, assembled from allowed candidate-position
  tokens but not itself a valid candidate.

Selected-candidate plus candidate-set smoke:

```bash
systemd-run --user --scope --quiet -p MemoryMax=28G -p MemorySwapMax=4G \
  .venv-fastdllm/bin/python scripts/eval_fastdllm_toolcall_cases.py \
  --base-model models/qwen3.5-9b-fastdllm-init \
  --adapter runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model \
  --tokenizer-path models/qwen3.5-9b-fastdllm-init \
  --input-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_schedule_candidate_selected_smoke1.jsonl \
  --limit 1 \
  --block-size 32 \
  --small-block-size 8 \
  --max-new-tokens 160 \
  --threshold 0.9 \
  --temperature 0.0 \
  --top-p 0.95 \
  --conversation-template fast_dllm_v2 \
  --full-context-sampling \
  --sampler-schedule-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_sampler_schedule_with_candidates_1.jsonl \
  --force-schedule-token-kinds tool_tag,json_key,json_structure,tool_name \
  --force-argument-boundary-target-tokens \
  --constrain-argument-candidate-tokens \
  --force-selected-candidate-tokens \
  --ban-argument-boundary-tokens \
  --ban-argument-newline-tokens \
  --stop-after-schedule-tool-calls \
  --constrained-tool-decoding \
  --constrained-sequence-preserving \
  --constrained-assume-utc-z \
  --constrained-max-calls 3 \
  --repair-mode none \
  --no-merge-adapter
```

Result:

- output:
  `runs/tool_sensitive_block_plans/public_multicall_schedule_candidate_selected_smoke1.jsonl`
- raw valid JSON/tool calls: `1/1`
- raw exact tool sequence: `1/1`
- raw exact arguments: `1/1`
- raw schema valid / required args: `1/1`
- candidate-constrained token visits: `48`
- selected-candidate forced token visits: `7`
- generated tokens/s: `31.2553`
- max allocated / reserved VRAM: `17.94` / `22.94` GiB

Readout:

- This is the first non-oracle schedule-based run in this line that reaches raw
  exact sequence and exact arguments on the public multi-call case.
- It is still protected/runtime behavior: structure/tool/key tokens are forced,
  boundary pieces are forced, argument candidates are built from prompt
  evidence, and two selected candidate spans are hard-filled.
- The remaining research question is how to replace selected-candidate forcing
  with sequence-consistent candidate decoding, where the model chooses one
  whole candidate string rather than mixing per-position candidate tokens.

Whole-candidate sequence smoke:

```bash
systemd-run --user --scope --quiet -p MemoryMax=28G -p MemorySwapMax=4G \
  .venv-fastdllm/bin/python scripts/eval_fastdllm_toolcall_cases.py \
  --base-model models/qwen3.5-9b-fastdllm-init \
  --adapter runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model \
  --tokenizer-path models/qwen3.5-9b-fastdllm-init \
  --input-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_schedule_candidate_sequence_smoke1.jsonl \
  --limit 1 \
  --block-size 32 \
  --small-block-size 8 \
  --max-new-tokens 160 \
  --threshold 0.9 \
  --temperature 0.0 \
  --top-p 0.95 \
  --conversation-template fast_dllm_v2 \
  --full-context-sampling \
  --sampler-schedule-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_sampler_schedule_with_candidates_1.jsonl \
  --force-schedule-token-kinds tool_tag,json_key,json_structure,tool_name \
  --force-argument-boundary-target-tokens \
  --force-best-candidate-sequence \
  --ban-argument-boundary-tokens \
  --ban-argument-newline-tokens \
  --stop-after-schedule-tool-calls \
  --constrained-tool-decoding \
  --constrained-sequence-preserving \
  --constrained-assume-utc-z \
  --constrained-max-calls 3 \
  --repair-mode none \
  --no-merge-adapter
```

Result:

- output:
  `runs/tool_sensitive_block_plans/public_multicall_schedule_candidate_sequence_smoke1.jsonl`
- raw valid JSON/tool calls: `1/1`
- raw exact tool sequence: `1/1`
- raw exact arguments: `0/1`
- raw schema valid / required args: `1/1`
- whole-candidate sequence forced token visits: `54`
- whole-candidate model choices: `4`
- generated tokens/s: `33.4286`
- max allocated / reserved VRAM: `17.96` / `23.08` GiB

Readout:

- Whole-candidate sequence decoding fixes the `100pp` style recombination
  class: the sampler chooses complete candidate strings rather than independent
  per-position candidate tokens.
- It is still not enough for semantic argument choice. In this smoke the model
  chooses `end_time=2023-04-22T15:00:00Z` instead of the gold
  `2023-04-22T17:00:00Z`.
- This isolates the next learned-model target: improve value-candidate ranking
  under tool-call context, especially paired arguments such as start/end times.

Whole-candidate sequence plus evidence-selected protection:

```bash
systemd-run --user --scope --quiet -p MemoryMax=28G -p MemorySwapMax=4G \
  .venv-fastdllm/bin/python scripts/eval_fastdllm_toolcall_cases.py \
  --base-model models/qwen3.5-9b-fastdllm-init \
  --adapter runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model \
  --tokenizer-path models/qwen3.5-9b-fastdllm-init \
  --input-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_schedule_candidate_sequence_selected_smoke1.jsonl \
  --limit 1 \
  --block-size 32 \
  --small-block-size 8 \
  --max-new-tokens 160 \
  --threshold 0.9 \
  --temperature 0.0 \
  --top-p 0.95 \
  --conversation-template fast_dllm_v2 \
  --full-context-sampling \
  --sampler-schedule-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_sampler_schedule_with_candidates_2.jsonl \
  --force-schedule-token-kinds tool_tag,json_key,json_structure,tool_name \
  --force-argument-boundary-target-tokens \
  --force-best-candidate-sequence \
  --force-selected-candidate-tokens \
  --ban-argument-boundary-tokens \
  --ban-argument-newline-tokens \
  --stop-after-schedule-tool-calls \
  --constrained-tool-decoding \
  --constrained-sequence-preserving \
  --constrained-assume-utc-z \
  --constrained-max-calls 3 \
  --repair-mode none \
  --no-merge-adapter
```

Result:

- output:
  `runs/tool_sensitive_block_plans/public_multicall_schedule_candidate_sequence_selected_smoke1.jsonl`
- raw valid JSON/tool calls: `1/1`
- raw exact tool sequence: `1/1`
- raw exact arguments: `1/1`
- raw schema valid / required args: `1/1`
- selected-candidate forced token visits: `47`
- whole-candidate sequence forced token visits: `7`
- whole-candidate model choices: `1`
- generated tokens/s: `58.6075`
- max allocated / reserved VRAM: `17.94` / `20.72` GiB

Readout:

- This is a protected selector result, not raw learned behavior. The selected
  candidates come from prompt/schema evidence, including the paired start/end
  datetime extractor.
- The result is still valuable because it defines the runtime scaffold we want
  the model to internalize: literal structure protection, whole-candidate
  value spans, and key-aware argument selection.

### Tool-Name Candidate Sequence Smoke

The previous exact smoke still hard-filled `tool_name` tokens from the planned
schedule. The next sampler step removes that oracle-style force and constrains
tool names to length-compatible available tools from the case schema.

Augment the schedule:

```bash
.venv-fastdllm/bin/python scripts/augment_schedule_value_candidates.py \
  --schedule-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_sampler_schedule_with_ids_meta_1.jsonl \
  --cases-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --tokenizer-path models/qwen3.5-9b-fastdllm-init \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_sampler_schedule_with_toolname_candidates_2.jsonl
```

Augmentation result:

- argument blocks augmented: `7`
- tool-name blocks augmented: `3`
- available tool-name values considered: `15`
- length-compatible tool-name sequences: `9`
- tool-name blocks with target in compatible set: `3/3`

Fixed-length caveat:

- This Fast-DLLM sampler cannot change the number of generated tokens inside an
  already planned tool-name span.
- Candidate tool names are therefore constrained to names whose tokenized
  shape fits the planned span. For this Qwen tokenizer, the tool-name token
  block also absorbs the closing quote, comma, and next-key opening quote
  (`","`), so compatible candidates must preserve that suffix.
- Variable-length tool-name choice needs a later flexible-block or resampling
  design.

Smoke command:

```bash
systemd-run --user --scope --quiet -p MemoryMax=28G -p MemorySwapMax=4G \
  .venv-fastdllm/bin/python scripts/eval_fastdllm_toolcall_cases.py \
  --base-model models/qwen3.5-9b-fastdllm-init \
  --adapter runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model \
  --tokenizer-path models/qwen3.5-9b-fastdllm-init \
  --input-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_schedule_toolname_candidate_sequence_selected_smoke2.jsonl \
  --limit 1 \
  --block-size 32 \
  --small-block-size 8 \
  --max-new-tokens 160 \
  --threshold 0.9 \
  --temperature 0.0 \
  --top-p 0.95 \
  --conversation-template fast_dllm_v2 \
  --full-context-sampling \
  --sampler-schedule-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_sampler_schedule_with_toolname_candidates_2.jsonl \
  --force-schedule-token-kinds tool_tag,json_key,json_structure \
  --force-best-tool-name-sequence \
  --force-argument-boundary-target-tokens \
  --force-best-candidate-sequence \
  --force-selected-candidate-tokens \
  --ban-argument-boundary-tokens \
  --ban-argument-newline-tokens \
  --stop-after-schedule-tool-calls \
  --constrained-tool-decoding \
  --constrained-sequence-preserving \
  --constrained-assume-utc-z \
  --constrained-max-calls 3 \
  --repair-mode none \
  --no-merge-adapter
```

Result:

- output:
  `runs/tool_sensitive_block_plans/public_multicall_schedule_toolname_candidate_sequence_selected_smoke2.jsonl`
- raw valid JSON/tool calls: `1/1`
- raw schema valid / required args: `1/1`
- raw exact tool sequence: `0/1`
- raw exact arguments: `0/1`
- generated tool sequence:
  `get_camera_live_feed`, `record_camera_feed`, `get_camera_live_feed`
- missing gold tool: `get_recorded_feed`
- repeated/extra tool: `get_camera_live_feed`
- tool-name sequence forced token visits: `14`
- tool-name model choices: `2`
- generated tokens/s: `32.3305`
- max allocated / reserved VRAM: `17.95` / `22.98` GiB

Readout:

- The suffix fix converts the earlier invalid-JSON failure into valid tool-call
  JSON. Tool names are now constrained to available-tool candidates rather than
  arbitrary text.
- This first implementation still committed too early when multiple candidates
  shared the current one-token prefix. On the third call, both
  `get_camera_live_feed` and `get_recorded_feed` start with the same first Qwen
  token, so the sampler locked onto `get_camera_live_feed` before the later
  tokens could disambiguate.

Deferred-choice fix:

- The sampler now keeps an allowed candidate subset per scheduled group and
  defers committing a whole-candidate index when the current chunk is shared by
  multiple candidates.
- This is still sequence-consistent decoding: it forces the selected current
  token chunk, but only finalizes the candidate once the remaining candidate
  set is unambiguous.

Deferred-choice smoke result:

- output:
  `runs/tool_sensitive_block_plans/public_multicall_schedule_toolname_candidate_sequence_selected_deferred_smoke1.jsonl`
- raw valid JSON/tool calls: `1/1`
- raw exact tool sequence: `1/1`
- raw exact arguments: `1/1`
- raw schema valid / required args: `1/1`
- generated tool sequence:
  `get_camera_live_feed`, `record_camera_feed`, `get_recorded_feed`
- tool-name sequence forced token visits: `14`
- tool-name model choices: `2`
- tool-name deferred choices: `2`
- candidate-sequence model choices: `1`
- generated tokens/s: `46.1506`
- max allocated / reserved VRAM: `17.94` / `21.78` GiB

Readout:

- The previous repeated-tool failure was a sampler commitment bug, not evidence
  that the model could not rank the correct tool under the generated prefix.
- The remaining learned candidate-ranking target is now narrower: row/time
  argument values, not tool names on this smoke.

### Candidate-Ranking Training Artifact

The function/value choice failures are now materialized as supervised ranker
examples. The builder consumes an augmented sampler schedule and emits both
JSONL audit rows and conversation-format training instances:

```bash
.venv-fastdllm/bin/python scripts/build_candidate_ranking_examples.py \
  --schedule-jsonl runs/tool_sensitive_block_plans/public_multicall_gold_sampler_schedule_with_toolname_candidates_v3_12.jsonl \
  --cases-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --out-jsonl data/candidate_ranking/public_multicall_toolname_argument_ranking_v3_12.jsonl
```

Preceding full-slice gold schedule steps:

```bash
.venv-fastdllm/bin/python scripts/plan_tool_sensitive_blocks.py \
  --input-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --text-field gold_assistant \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_gold_blocks_tokenized_with_ids_meta_12.jsonl \
  --limit 12 \
  --tokenizer-path models/qwen3.5-9b-fastdllm-init \
  --include-token-ids

.venv-fastdllm/bin/python scripts/emit_tool_sensitive_sampler_schedule.py \
  --input-jsonl runs/tool_sensitive_block_plans/public_multicall_gold_blocks_tokenized_with_ids_meta_12.jsonl \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_gold_sampler_schedule_with_ids_meta_12.jsonl \
  --include-token-ids

.venv-fastdllm/bin/python scripts/augment_schedule_value_candidates.py \
  --schedule-jsonl runs/tool_sensitive_block_plans/public_multicall_gold_sampler_schedule_with_ids_meta_12.jsonl \
  --cases-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --tokenizer-path models/qwen3.5-9b-fastdllm-init \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_gold_sampler_schedule_with_toolname_candidates_v3_12.jsonl
```

Coverage:

- full public multi-call gold records: `12`
- gold tool calls: `31`
- ranker examples: `86`
- usable training examples: `86`
- tool-name examples: `31/31` usable
- argument-value examples: `55/55` usable
- target missing from candidate set: `0`
- candidate values across examples: `276`
- train file:
  `data/candidate_ranking/public_multicall_toolname_argument_ranking_v3_12.train.json`

Extractor fixes needed to reach full coverage:

- ID-like unquoted tokens from tables, such as `INV-301`, `CLI-101`, and
  `PAY-401`.
- location-like quoted strings such as `living room`.
- plain clock strings such as `23:00`, not only full ISO datetimes.
- integer-valued float normalization, so `72.0` tokenizes as `72` when the gold
  span is integer text.

Readout:

- This is the first concrete artifact for the learned candidate-ranking target.
- It is still small and should not be treated as a final dataset. Its purpose is
  to make the next objective executable: train or evaluate a lightweight ranker
  or adapter target that chooses the behavior-preserving candidate instead of
  relying on protected post-processing.

### Masked-Span Candidate-Ranking Baseline

The current Qwen3.5-9B diffusion adapter can now be scored directly on the
ranker artifact without running full generation. The evaluator masks the gold
tool/value span inside the gold assistant trace, scores each candidate sequence
under the model, and reports whether the gold candidate is ranked first.

```bash
systemd-run --user --scope --quiet -p MemoryMax=28G -p MemorySwapMax=4G \
  .venv-fastdllm/bin/python scripts/eval_fastdllm_candidate_ranking.py \
  --base-model models/qwen3.5-9b-fastdllm-init \
  --adapter runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model \
  --tokenizer-path models/qwen3.5-9b-fastdllm-init \
  --examples-jsonl data/candidate_ranking/public_multicall_toolname_argument_ranking_v3_12.jsonl \
  --cases-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --out-jsonl runs/candidate_ranking/public_multicall_qwen35_ckpt275_masked_span_rank_v3_12.jsonl \
  --conversation-template fast_dllm_v2 \
  --no-merge-adapter
```

Result:

- output:
  `runs/candidate_ranking/public_multicall_qwen35_ckpt275_masked_span_rank_v3_12.jsonl`
- examples: `86`
- overall accuracy: `80/86` (`93.0%`)
- multi-candidate accuracy: `52/58` (`89.7%`)
- tool-name accuracy: `31/31`
- multi-candidate tool-name accuracy: `15/15`
- argument-value accuracy: `49/55`
- multi-candidate argument-value accuracy: `37/43`
- examples/s: `6.16`
- max allocated / reserved VRAM: `18.41` / `20.22` GiB

Context-mode ablation:

| Context mode | Meaning | Overall | Multi-candidate | Tool names | Arguments |
| --- | --- | ---: | ---: | ---: | ---: |
| `full_gold` | full gold assistant, only target span masked | `80/86` | `52/58` | `31/31` | `49/55` |
| `prefix_only` | gold assistant prefix before span, no future tokens | `80/86` | `52/58` | `31/31` | `49/55` |
| `future_masked` | full gold length, target and future assistant tokens masked | `80/86` | `52/58` | `31/31` | `49/55` |

Additional artifacts:

- `runs/candidate_ranking/public_multicall_qwen35_ckpt275_masked_span_rank_v3_12_prefix_only.jsonl`
- `runs/candidate_ranking/public_multicall_qwen35_ckpt275_masked_span_rank_v3_12_future_masked.jsonl`

Failure classes:

- ambiguous schedule times: `19:00` ranked below `11:00`; `23:00` ranked below
  `22:00`
- row-aligned table IDs: `CLI-102` / `CLI-103` and `INV-301` / `INV-302`
  occasionally rank below a neighboring row ID

Readout:

- Under gold-context masked-span scoring, the current adapter already ranks all
  tool-name candidates correctly. The full-generation `get_camera_live_feed`
  repeat therefore comes from iterative generation context drift, not an
  inability to identify the tool when the surrounding gold trace is fixed.
- Removing future gold tokens does not change the result: `prefix_only` and
  `future_masked` fail on the same six argument spans. The masked-span ranking
  gap is stable and localized to value alignment, not future-token leakage.
- Remaining learned-ranker pressure should focus on row/temporal alignment for
  argument values, then be tested again in full generation where context can
  drift.

## Readout

The planner distinguishes the current raw/protected gap in a way that matches
the scorecard:

- gold public multi-call has `31` intended tool calls
- raw generated public multi-call exposes only `25` parseable tool-call spans
- protected sequence-planner output recovers `31` parseable tool-call spans

This is a useful bridge metric for generation-time dynamic block decoding. If a
future sampler uses tool-sensitive blocks, it should move the raw generated span
distribution toward the gold/protected distribution before post-hoc projection.

## Next Implementation Step

Prototype sampler-side block-size overrides from the tokenized plan:

- normal prose block size: current Fast-dLLM block size
- tool tags, function names, and JSON keys: tiny deterministic blocks
- scalar argument values: small blocks with extra denoising passes
- stop boundary after `</tool_call>`: literal/stop-constrained tiny block

Tokenizer-offset support, the sampler dry-run, the non-generating Fast-DLLM
sampler trace, and the first scheduled full-context sampler override are now
implemented. First-prefix format control is also implemented. The next
implementation step is schema/stop control inside the tool-call body: constrain
JSON key/value separators and stop after the planned number of tool calls.

Promotion should be based on raw generation movement, especially parsed
tool-call count, exact tool sequence, repeated/extra/missing-call rate, and
exact argument rate.
