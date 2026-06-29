# Qwen3.5 9B Schedule-State Selector Curriculum Gate

Date: 2026-06-28

## Question

After the standalone skeleton value-infill adapter failed to move multi-call
gates, can we build a trainable objective that targets the actual scheduled
sampler decision instead of asking the model to free-generate only the JSON
value span?

Short answer: yes. The new curriculum builds cleanly from the no-overlap
skeleton value slots, audits cleanly at block size `1024`, and a one-step
checkpoint-275 QLoRA continuation trains and saves on the local RTX 5090.

## Builder

New script:

```text
scripts/build_schedule_state_selector_curriculum.py
```

It consumes:

```text
data/skeleton_value_infill/public_train_no_public_smoke/skeleton_value_slots.jsonl
data/skeleton_value_infill/public_train_no_public_smoke/boundary_labels.jsonl
data/skeleton_value_infill/public_train_no_public_smoke/summary.json
```

It writes:

```text
data/qwen35_9b_schedule_state_selector_no_public_smoke_curriculum/
```

The assistant target is a compact JSON selector/policy decision:

```json
{"candidate_index":0,"span_kind":"argument_value","protection":"value_candidate_json_prefix_close_guard","block_size":8,"denoise_steps":8,"force_candidate_sequence":true,"require_json_prefix_safe":true,"close_tool_call_only_when_json_complete":true}
```

This deliberately does not ask the model to emit the argument value itself. It
asks for the same kind of decision the sampler needs: candidate index plus the
local protection policy for the active scheduled value span.

## Curriculum Summary

Command shape:

```bash
.venv-fastdllm/bin/python scripts/build_schedule_state_selector_curriculum.py \
  --out-dir data/qwen35_9b_schedule_state_selector_no_public_smoke_curriculum \
  --block-size 1024 \
  --truncation-side left \
  --singleton-repeat 1 \
  --ambiguous-repeat 2 \
  --nonzero-target-repeat 3
```

Manifest:

```text
data/qwen35_9b_schedule_state_selector_no_public_smoke_curriculum/train_agentic_mix.manifest
```

Key counts:

- raw slots: `331`
- singleton slots: `190`
- ambiguous slots: `141`
- accepted train instances: `539`
- rejected train instances: `0`
- accepted ambiguous instances: `349`
- accepted nonzero-target instances: `201`
- accepted singleton instances: `190`
- p50 tokenized length: `938`
- p90 tokenized length: `1733`
- kept labels: p50 `58`, p90 `58`
- labels lost to truncation: `0`
- promotion allowed: yes, because the source is the previously audited
  no-public-smoke clean corpus

## One-Step Fit Gate

Output:

```text
runs/fastdllm_qwen35_9b_schedule_state_selector_no_public_smoke_from_ckpt275_step1_gate
logs/fastdllm_qwen35_9b_schedule_state_selector_no_public_smoke_from_ckpt275_step1_gate.log
```

Settings:

- start adapter:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model`
- dataset:
  `data/qwen35_9b_schedule_state_selector_no_public_smoke_curriculum`
- `MAX_STEPS=1`
- `MAX_TRAIN_SAMPLES=64`
- `BLOCK_SIZE=1024`
- `LEARNING_RATE=1e-6`
- `GRAD_ACCUM=4`
- `DISABLE_GROUP_TEXTS=1`
- `TRUNCATION_SIDE=left`
- CPU cgroup: `MemoryHigh=27G`, `MemoryMax=28G`, `MemorySwapMax=4G`

Result:

- global step: `1`
- train loss: `4.879097938537598`
- runtime: `10.55s`
- adapter saved: yes
- checkpoint adapter saved: yes
- OOM: no
- GPU returned idle after completion

The higher one-step loss versus the standalone value-infill gate is expected:
the assistant label is a structured selector/policy JSON object of about
`58-59` kept label tokens, not a short scalar value span.

## Interpretation

This is the first concrete replacement objective after the non-promoted
standalone value-infill line:

- it trains the model toward candidate selection in the same schedule-state
  framing used by the protected sampler;
- it includes block-size, denoise-step, JSON-prefix, value-candidate, and
  close-completeness policy labels;
- it oversamples ambiguous and nonzero-target candidate choices;
- it remains no-overlap with public/heldout eval slices through the source
  artifact audit.

This still is not promotion evidence. It only proves the next objective is
materialized and trainable.

## Next Gate

Train a short sweep from active checkpoint-275, likely `25/50/75` steps, then
evaluate with a selector parser that checks whether generated JSON decisions
choose the right `candidate_index` and policy. Only after selector accuracy
moves should this be wired back into the sampler or promoted to public/heldout
multi-call generation gates.

## 75-Step Free-Generation Sweep

Added evaluator:

```text
scripts/eval_fastdllm_schedule_state_selector.py
```

It prompts each schedule-state row without the assistant label, samples a
decision, parses JSON or a regex fallback, and reports:

- valid selector JSON
- exact `candidate_index`
- exact protection-policy fields
- exact full decision

Training output:

```text
runs/fastdllm_qwen35_9b_schedule_state_selector_no_public_smoke_from_ckpt275_step75
logs/fastdllm_qwen35_9b_schedule_state_selector_no_public_smoke_from_ckpt275_step75.log
```

Settings:

- start adapter:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model`
- `MAX_STEPS=75`
- `MAX_TRAIN_SAMPLES=539`
- `BLOCK_SIZE=1024`
- `LEARNING_RATE=1e-6`
- `GRAD_ACCUM=4`
- checkpoint saves: `25`, `50`, `75`
- CPU cgroup: `MemoryHigh=27G`, `MemoryMax=28G`, `MemorySwapMax=4G`

Training result:

- runtime: `745.59s`
- final train loss: `4.5926`
- train samples/sec: `0.402`
- train steps/sec: `0.101`
- OOM: no
- GPU returned idle after completion

Free-generation selector sweep:

```text
runs/schedule_state_selector/no_public_smoke_step75_selector_sweep_eval16/
logs/fastdllm_qwen35_9b_schedule_state_selector_sweep_eval16.log
```

Slice: first `16` ambiguous schedule-state examples. Generation: `max_new_tokens=96`,
`block_size=32`, `small_block_size=8`, no adapter merge.

| Checkpoint | Valid JSON | Candidate Index | Policy Exact | Decision Exact |
| --- | ---: | ---: | ---: | ---: |
| active ckpt-275 | `0/16` | `3/16` | `0/16` | `0/16` |
| selector step 25 | `0/16` | `1/16` | `0/16` | `0/16` |
| selector step 50 | `0/16` | `3/16` | `0/16` | `0/16` |
| selector step 75 | `0/16` | `2/16` | `0/16` | `0/16` |

Observed raw generations remain dominated by `<think>` chat fragments, numeric
fragments, and unrelated text. Regex fallback sometimes extracts a number, but
no checkpoint emits executable selector JSON.

## Decision After Sweep

Do not promote this free-generation selector objective.

What worked:

- the schedule-state selector corpus is buildable and auditable;
- checkpoint-275 QLoRA continuation trains under the local 5090 memory budget;
- the evaluator now gives an explicit executable-decision gate.

What failed:

- free-form diffusion generation did not learn minified selector JSON in this
  short sweep;
- candidate-index accuracy did not improve over active checkpoint-275;
- the Qwen chat template/thinking prefix is an active confound for using this
  as ordinary assistant generation.

Next direction: keep the schedule-state representation, but stop asking the
model to freely emit selector JSON. Use one of:

1. constrained JSON prefix-forced decoding for the fixed policy keys, with only
   `candidate_index` scored/generated;
2. masked likelihood or pairwise ranking over candidate indices and policy
   templates;
3. a learned sidecar selector whose output is injected into the protected
   sampler schedule.

Promotion should require nonzero valid/exact selector decisions first, then
public/heldout tool-call generation gates.

## Constrained Candidate-Index Ranking

Added evaluator:

```text
scripts/eval_fastdllm_schedule_state_selector_ranking.py
```

This keeps the selector as control state instead of generated assistant text.
The useful mode is:

```text
score_mode=index_only
```

It builds the normal schedule-state prompt, then force-prefixes:

```text
{"candidate_index":
```

and scores only candidate index values `0..N-1` by masked likelihood. The rest
of the selector JSON policy is deterministic and can be injected by the sampler:

```json
{"span_kind":"argument_value","protection":"value_candidate_json_prefix_close_guard","block_size":8,"denoise_steps":8,"force_candidate_sequence":true,"require_json_prefix_safe":true,"close_tool_call_only_when_json_complete":true}
```

This exactly matches the recommended constrained-control path after the
free-generation failure.

### 64-Row Checkpoint Sweep

Output:

```text
runs/schedule_state_selector/no_public_smoke_step75_selector_indexonly_rank64/
logs/fastdllm_qwen35_9b_schedule_state_selector_indexonly_rank64.log
```

Slice: first `64` ambiguous schedule-state examples.

| Checkpoint | Index Accuracy | Target Top-2 | Notes |
| --- | ---: | ---: | --- |
| active ckpt-275 | `59/64` | `63/64` | baseline constrained scorer |
| selector step 25 | `59/64` | `63/64` | ties baseline |
| selector step 50 | `59/64` | `63/64` | ties baseline |
| selector step 75 | `59/64` | `63/64` | ties baseline |

Interpretation: the 75-step selector SFT does not improve constrained
candidate-index scoring. The useful capability is already present in active
checkpoint-275 when the interface is constrained.

### Full Ambiguous Active-Checkpoint Sweep

Output:

```text
runs/schedule_state_selector/no_public_smoke_ckpt275_indexonly_rank_all_ambiguous.jsonl
runs/schedule_state_selector/no_public_smoke_ckpt275_indexonly_rank_all_ambiguous.summary.json
logs/fastdllm_qwen35_9b_schedule_state_selector_ckpt275_indexonly_rank_all_ambiguous.log
```

Slice: all `349` ambiguous schedule-state rows.

Result:

- exact candidate-index accuracy: `312/349` (`89.40%`)
- target in top 2: `334/349` (`95.70%`)
- errors: `0`
- examples/sec: `1.48`
- max allocated GPU memory: `18.43 GiB`

Miss analysis:

- misses: `37`
- target-rank distribution among misses: rank 2 = `22`, rank 3 = `12`,
  rank 4 = `3`
- most common miss key: `device_type` (`9` misses)
- repeated semantic miss examples:
  - `smart thermostat` loses to `smart_light` or `thermostat`
  - `regional` loses to location-like candidates
  - `Photo Evidence` loses to a person-name candidate in a large document-title
    candidate set

### Full-Template Scorer Sanity Check

Output:

```text
runs/schedule_state_selector/no_public_smoke_ckpt275_fulldecision_rank16.jsonl
logs/fastdllm_qwen35_9b_schedule_state_selector_ckpt275_fulldecision_rank16.log
```

Scoring the entire minified decision JSON from an empty assistant continuation
is weak: `5/16` exact index accuracy on the first ambiguous slice. This is not
the desired constrained path because the model must score the fixed schema from
scratch rather than only the variable control choice.

## Decision After Constrained Ranking

Promote the design pattern, not the selector-SFT checkpoint:

- use active checkpoint-275 as the current constrained index scorer baseline;
- force the selector JSON prefix and inject the fixed protection-policy suffix;
- treat `candidate_index` as a scored control variable, not free generated
  assistant text;
- do not scale the 75-step selector free-generation SFT unchanged.

Next implementation gate: wire this scorer into the protected schedule injector
and evaluate public/heldout tool-call generation with `target_top1` and a
top-2 fallback/repair mode reported separately.

## Schedule Injection Smoke

Added bridge:

```text
scripts/inject_schedule_state_selector_ranking_choices.py
```

It consumes a sampler schedule and the constrained selector ranking JSONL, then
restricts each matching `argument_value` schedule item to the ranked top-k
candidate sequences.

Preflight compatibility:

- selector prompt candidate order exactly matches schedule
  `candidate_sequence_values` for all `539` schedule-state curriculum
  instances;
- all `349` ranking rows collapse to `141` unique schedule-state keys with
  `0` prediction conflicts;
- rank-1 and rank-2 schedule injection both restrict `163` argument schedule
  items across `30` records.

Injected schedules:

```text
runs/tool_sensitive_block_plans/public_train_no_public_smoke_sampler_schedule_with_schedule_state_selector_rank1.jsonl
runs/tool_sensitive_block_plans/public_train_no_public_smoke_sampler_schedule_with_schedule_state_selector_rank2.jsonl
```

Four-case smoke inputs:

```text
data/toolcall_eval/public_train_multicall_no_public_smoke_cases_selector_rank_smoke4.jsonl
runs/tool_sensitive_block_plans/public_train_no_public_smoke_sampler_schedule_with_schedule_state_selector_rank1_smoke4.jsonl
runs/tool_sensitive_block_plans/public_train_no_public_smoke_sampler_schedule_with_schedule_state_selector_rank2_smoke4.jsonl
```

Generation outputs:

```text
runs/tool_sensitive_block_plans/public_train_no_public_smoke_schedule_state_selector_rank1_smoke4_generation.jsonl
runs/tool_sensitive_block_plans/public_train_no_public_smoke_schedule_state_selector_rank2_smoke4_generation.jsonl
```

Guard stack:

- active checkpoint-275 adapter
- full-context scheduled sampler
- `--guard-tool-value-candidates`
- `--guard-tool-call-mode`
- `--guard-tool-json-prefix`
- `--stop-after-schedule-tool-calls`
- sequence-preserving constrained projection enabled for reporting

Smoke result:

| Schedule | Raw Valid JSON | Raw Sequence | Raw Args | Constrained Args | Value-Guard Choices |
| --- | ---: | ---: | ---: | ---: | ---: |
| rank-1 | `4/4` | `4/4` | `3/4` | `2/4` | `0` |
| rank-2 | `4/4` | `4/4` | `2/4` | `2/4` | `19` |

Rank-1 miss:

- `public_train_no_public_smoke_0001`: `search_deals.limit` predicted `1`
  instead of gold `5`.

Rank-2 additional miss:

- `public_train_no_public_smoke_0002`: `campaign_end_date` chosen as
  `2023-06-01` instead of gold `2023-08-31`.

Interpretation:

- the bridge/injection path works and activates the protected sampler;
- rank-1 hard restriction is stronger than allowing top-2 in-sampler choice on
  this smoke;
- top-2 should be treated as a separate repair/rerank path, not the default
  sampler mode.

Next gate: run a broader no-public-smoke train slice with rank-1, then port the
same bridge to heldout/public evidence-selector schedules where the ranking
examples are already promotion-audited.
