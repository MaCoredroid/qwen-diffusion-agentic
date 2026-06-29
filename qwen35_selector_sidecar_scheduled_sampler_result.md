# Qwen3.5 Selector Sidecar Scheduled Sampler Result

Date: 2026-06-28

## Purpose

Move the synthetic selector sidecar from post-hoc projection into the diffusion
sampling loop.

This tests whether the sampler can consume sidecar-selected tool/value
candidates during generation, while the main generator remains the active
Qwen3.5-9B Fast-dLLM checkpoint-275 adapter.

This is still protected-sampler evidence. It is not raw model promotion.

## Inputs

Cases:

```text
data/toolcall_eval/synthetic_multicall_failure_analogues.jsonl
```

Main generator:

```text
models/qwen3.5-9b-fastdllm-init
runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model
```

Selector sidecar ranking:

```text
runs/candidate_ranking/synthetic_multicall_failure_analogue_leaveone_voice003_ckpt20_index_rank.jsonl
```

Schedule artifacts:

```text
runs/tool_sensitive_block_plans/synthetic_selector_sidecar_blocks_tokenized_with_ids.jsonl
runs/tool_sensitive_block_plans/synthetic_selector_sidecar_sampler_schedule_with_ids.jsonl
runs/tool_sensitive_block_plans/synthetic_selector_sidecar_sampler_schedule_augmented.jsonl
runs/tool_sensitive_block_plans/synthetic_selector_sidecar_sampler_schedule_with_selector_choices.jsonl
runs/tool_sensitive_block_plans/synthetic_selector_sidecar_sampler_schedule_augmented_targetselected.jsonl
runs/tool_sensitive_block_plans/synthetic_selector_sidecar_sampler_schedule_with_selector_choices_targetselected.jsonl
```

Generation output:

```text
runs/tool_sensitive_block_plans/synthetic_selector_sidecar_scheduled_ckpt275_generation.jsonl
runs/tool_sensitive_block_plans/synthetic_selector_sidecar_scheduled_ckpt275_generation.summary.json
runs/tool_sensitive_block_plans/synthetic_selector_sidecar_scheduled_targetselected_ckpt275_generation.jsonl
runs/tool_sensitive_block_plans/synthetic_selector_sidecar_scheduled_targetselected_ckpt275_generation.summary.json
runs/tool_sensitive_block_plans/synthetic_selector_sidecar_scheduled_modelranked_values_ckpt275_generation.jsonl
runs/tool_sensitive_block_plans/synthetic_selector_sidecar_scheduled_modelranked_values_ckpt275_generation.summary.json
runs/tool_sensitive_block_plans/synthetic_targetcandidate_modelranked_ckpt275_generation.jsonl
runs/tool_sensitive_block_plans/synthetic_targetcandidate_modelranked_ckpt275_generation.summary.json
```

## Schedule Stats

Token-sensitive block planning over the sidecar-projected assistant produced:

- `8` records
- `24` tool calls
- `871` tokens
- `416` token-sensitive blocks

Schedule augmentation added:

- `60` argument blocks with selected candidates
- `54` argument blocks with sequence candidates
- `24` tool-name blocks with sequence candidates
- `24` tool-name blocks with target candidates

Selector-choice injection restricted `16` schedule items across `4` records.
The restricted records are the ambiguous voice-command/tool-name rows.

The first run exposed a schedule metadata bug: `selected_candidate` on argument
blocks came from the deterministic evidence extractor, not from the protected
sidecar plan. That caused exact target spans such as the driveway camera command
to be overwritten by earlier pantry-light evidence.

A corrected target-selected schedule was built with:

```text
--include-target-candidate
--selected-candidate-mode target
```

The corrected augmentation has sequence candidates for `60/60` argument blocks,
up from `54/60`. Selector-choice injection then restricted `20` schedule items,
with `0` candidate-missing items.

## Result

| path | raw exact sequence | raw exact args | raw valid JSON | constrained exact sequence | constrained exact args | constrained valid JSON |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| checkpoint-275 baseline | `1/8` | `0/8` | `4/8` | `2/8` | `0/8` | `8/8` |
| selector sidecar post-hoc projection | `8/8` | `8/8` | `8/8` | n/a | n/a | n/a |
| selector sidecar scheduled sampler, evidence-selected metadata | `4/8` | `0/8` | `4/8` | `8/8` | `0/8` | `8/8` |
| selector sidecar scheduled sampler, target-selected metadata | `8/8` | `8/8` | `8/8` | `8/8` | `4/8` | `8/8` |
| selector-injected schedule, model-ranked argument values | `8/8` | `8/8` | `8/8` | `8/8` | `4/8` | `8/8` |
| no-selector target-candidate schedule, model-ranked tool/value choices | `8/8` | `8/8` | `8/8` | `8/8` | `4/8` | `8/8` |

Sampler counters for the corrected target-selected run:

- schedule used: `8/8`
- scheduled token visits: `508`
- forced schedule-token visits: `492`
- selected-candidate force-token visits: `227`
- tool-name sequence choices: `12`
- stop-boundary guard trims: `3`
- max reserved VRAM: `17.95 GiB`
- generated tokens/s: `21.88`

Additional model-ranked ablations:

- Turning off `--force-selected-candidate-tokens` while keeping selector-injected
  restrictions still reaches raw `8/8` exact sequence and `8/8` exact
  arguments. This run makes `22` argument candidate-sequence choices and uses
  `0` selected-candidate force tokens.
- Removing selector injection as well, while keeping target-inclusive candidate
  sets, also reaches raw `8/8` exact sequence and `8/8` exact arguments. This
  run makes `26` argument candidate-sequence choices and `12` tool-name
  sequence choices with `0` selected-candidate force tokens.

## Interpretation

This is the first positive generation-time handoff from the learned selector
sidecar into the diffusion sampler:

- raw tool sequence improves from `1/8` to `8/8`;
- raw exact arguments improve from `0/8` to `8/8`;
- the sampler consumes sidecar choices instead of relying only on post-hoc
  projection.

The remaining caveat is important: these are protected schedules built from the
sidecar-projected or target-candidate assistant. They prove the sampler can
preserve fragile spans and that checkpoint-275 can rank the provided
whole-candidate choices on this synthetic slice. They do not prove the main
diffusion generator can propose the right candidate set or discover the
tool-call plan from an unconstrained raw sample.

The immediate implication is that sidecar-guided scheduling and whole-candidate
sequence scoring are worth keeping. On this synthetic slice, the bottleneck
moves from "can the sampler force argument spans?" to:

- tighter scoping of argument-value candidate blocks;
- proposing target-containing candidate sets without gold/target assistance;
- validating model-ranked tool/value choices on the public multi-call slice;
- learning or distilling dynamic block/candidate proposal from AR traces;
- separating tool-name sequence choice success from argument-value success in
  all reports.

Do not promote a checkpoint from this result. Promote the sampler path as an
active experimental lane.

Public follow-up:

- The 12-case public multi-call target-candidate follow-up is documented in
  `qwen35_public_multicall_targetcandidate_sampler_result.md`.
- With selected-candidate forcing off, the public run reaches raw `11/12` exact
  sequence, `9/12` exact arguments, and `11/12` valid JSON.
- With target-selected forcing on, it reaches raw `12/12` exact sequence,
  `12/12` exact arguments, and `12/12` valid JSON.
