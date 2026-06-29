# Qwen3.5 Public Multi-Call Target-Candidate Sampler Result

Date: 2026-06-28

## Purpose

Test whether the corrected target-inclusive candidate schedule generalizes from
the synthetic analogue slice to the 12-case public Hermes multi-call slice.

This is still protected-sampler evidence. The schedule is built from gold public
multi-call targets. The point is to separate:

- sampler span preservation;
- model-ranked whole-candidate choice;
- target-containing candidate proposal.

## Inputs

Cases:

```text
data/toolcall_eval/public_multicall_hermes_smoke.jsonl
```

Main generator:

```text
models/qwen3.5-9b-fastdllm-init
runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model
```

Corrected public schedule:

```text
runs/tool_sensitive_block_plans/public_multicall_gold_sampler_schedule_with_candidates_targetselected_v4_12.jsonl
runs/tool_sensitive_block_plans/public_multicall_gold_sampler_schedule_with_candidates_targetselected_v5_12.jsonl
```

Generation outputs:

```text
runs/tool_sensitive_block_plans/public_multicall_targetcandidate_modelranked_ckpt275_generation_v4_12.jsonl
runs/tool_sensitive_block_plans/public_multicall_targetcandidate_modelranked_ckpt275_generation_v4_12.summary.json
runs/tool_sensitive_block_plans/public_multicall_targetcandidate_forcedselected_ckpt275_generation_v4_12.jsonl
runs/tool_sensitive_block_plans/public_multicall_targetcandidate_forcedselected_ckpt275_generation_v4_12.summary.json
runs/tool_sensitive_block_plans/public_multicall_targetcandidate_modelranked_ckpt275_candidate_miss_audit_v4_12.jsonl
runs/tool_sensitive_block_plans/public_multicall_targetcandidate_modelranked_ckpt275_candidate_miss_audit_v4_12.summary.json
runs/tool_sensitive_block_plans/public_multicall_targetcandidate_modelranked_ckpt275_generation_v5_12.jsonl
runs/tool_sensitive_block_plans/public_multicall_targetcandidate_modelranked_ckpt275_generation_v5_12.summary.json
runs/tool_sensitive_block_plans/public_multicall_targetcandidate_modelranked_ckpt275_candidate_miss_audit_v5_12.jsonl
runs/tool_sensitive_block_plans/public_multicall_targetcandidate_modelranked_ckpt275_candidate_miss_audit_v5_12.summary.json
runs/tool_sensitive_block_plans/public_multicall_pairwise_path_singleton_structguard6_ckpt275_generation_v5_12.jsonl
runs/tool_sensitive_block_plans/public_multicall_pairwise_path_singleton_structguard6_ckpt275_generation_v5_12.summary.json
runs/tool_sensitive_block_plans/public_multicall_pairwise_path_singleton_structguard6_ckpt275_candidate_miss_audit_v5_12.summary.json
```

## Schedule Coverage

Rebuilt with:

```text
--include-target-candidate
--selected-candidate-mode target
```

Coverage:

- records: `12`
- argument blocks augmented: `100`
- argument blocks with sequence candidates: `90`
- argument candidate values: `587`
- tool-name blocks augmented: `31`
- tool-name blocks with target candidate: `31`

This is much stronger than the older public v3 schedule, which had only `69`
argument blocks augmented and `55` argument blocks with sequence candidates.

Follow-up v5 schedule fix:

- `scripts/augment_schedule_value_candidates.py` now uses the target token IDs
  directly when a candidate value equals the target value.
- This preserves decimal numeric target forms such as `1500.0`; previously
  they collapsed to `1500` and could not form length-compatible candidate
  sequences.
- v5 coverage: `100/100` argument blocks with sequence candidates, up from
  v4's `90/100`.

## Results

| path | selected-candidate forcing | raw exact sequence | raw exact args | raw valid JSON | constrained exact sequence | constrained exact args |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| older v3 public schedule | on, evidence-selected | `11/12` | `3/12` | `11/12` | `12/12` | `4/12` |
| v4 target-candidate, model-ranked values | off | `11/12` | `9/12` | `11/12` | `12/12` | `8/12` |
| v4 target-candidate, target-selected forcing | on | `12/12` | `12/12` | `12/12` | `12/12` | `8/12` |
| v5 target-candidate, decimal-safe model-ranked values | off | `12/12` | `9/12` | `12/12` | `12/12` | `8/12` |
| v5 path-aware pairwise singleton values | off | `12/12` | `12/12` | `12/12` | `12/12` | `8/12` |

Model-ranked value run counters:

- selected-candidate force tokens: `0`
- argument candidate-sequence choices: `43`
- tool-name sequence choices: `15`
- max reserved VRAM: `19.41 GiB`
- elapsed: `416.4s`

Target-selected upper-bound counters:

- selected-candidate force tokens: `452`
- argument candidate-sequence choices: `0`
- tool-name sequence choices: `15`
- max reserved VRAM: `19.26 GiB`
- elapsed: `396.6s`

V5 model-ranked counters:

- selected-candidate force tokens: `0`
- argument candidate-sequence choices: `43`
- argument candidate-sequence forced tokens: `470`
- tool-name sequence choices: `15`
- max reserved VRAM: `19.43 GiB`

Path-aware pairwise singleton counters:

- selected-candidate force tokens: `0`
- argument candidate-sequence choices: `37`
- argument candidate-sequence forced tokens: `470`
- tool-name sequence choices: `15`
- forced structural schedule tokens: `956`
- forced argument-boundary target tokens: `55`
- max reserved VRAM: `27.95 GiB`

## Misses

Audit script:

```text
scripts/analyze_toolcall_candidate_misses.py
```

Audit summary:

- failed records: `3/12`
- mismatches: `3`
- scalar argument mismatches: `2`
- scalar misses where the gold value was already in sequence candidates: `2/2`
- invalid tool-call blocks: `1`

The model-ranked value run missed three rows:

- `3f440c20-b332-48e2-aaa5-a7bfb0781ae9`: correct tool sequence, but
  thermostat `schedule_time` is `11:00` instead of `19:00`.
  Candidate set: `["19:00", "07:00", "11:00", "06:00", "23:00"]`.
- `e279e98f-095a-4d44-9c2d-170b3cfdc4bb`: correct tool sequence, but fridge
  `start_time` is `22:00` instead of `23:00`.
  Candidate set: `["23:00", "07:00", "22:00"]`.
- `adc48a37-6341-4ea6-972a-8ec2b5421321`: long finance/table row; raw JSON for
  `process_invoices` is malformed and repeats invoice fields, so the middle
  call is not parsed. The invalid block is still named `process_invoices`,
  confirming this is an array/table body consistency failure, not a tool-name
  planning failure.

All three are repaired by target-selected forcing.

V5 changes the third miss:

- The long finance row is now valid JSON and exact tool sequence.
- Remaining finance misses are row-local values inside `process_invoices`:
  - `invoice_data[1].invoice_id`: generated `INV-301`, gold `INV-302`
  - `invoice_data[1].client_id`: generated `CLI-103`, gold `CLI-102`
  - `invoice_data[2].client_id`: generated `CLI-101`, gold `CLI-103`
- The two time-ranking misses are unchanged:
  - thermostat `schedule_time`: generated `11:00`, gold `19:00`
  - fridge `start_time`: generated `22:00`, gold `23:00`

Focused miss target artifacts:

```text
data/candidate_ranking/public_multicall_targetcandidate_ranking_v5_12.jsonl
data/candidate_ranking/public_multicall_targetcandidate_v5_miss_targets.jsonl
data/candidate_ranking/public_multicall_targetcandidate_v5_miss_targets.train.json
runs/candidate_ranking/public_multicall_targetcandidate_v5_miss_targets_ckpt275_prefix_only.jsonl
runs/candidate_ranking/public_multicall_targetcandidate_v5_miss_targets_ckpt275_full_gold.jsonl
```

The focused miss target has `5/5` usable examples:

- `schedule_time` target `19:00`, generated `11:00`
- `start_time` target `23:00`, generated `22:00`
- `client_id` target `CLI-102`, generated `CLI-103`
- `invoice_id` target `INV-302`, generated `INV-301`
- `client_id` target `CLI-103`, generated `CLI-101`

Checkpoint-275 candidate-ranker baseline on this focused target is `0/5` in
both prefix-only and full-gold context modes. This makes the next training or
sidecar experiment sharply scoped.

Follow-up path-aware pairwise sidecar:

```text
qwen35_public_multicall_pairwise_path_sidecar_result.md
```

The missing ingredient was the full structural path. Pairwise prompts with only
`JSON key: client_id` could not reliably distinguish
`invoice_data[1].client_id` from `invoice_data[2].client_id`. Adding
`JSON path` from the miss audit gives checkpoint-275 `60/60` focused A/B
comparisons and `5/5` focused tournament choices. After injecting those five
choices, one more finance span surfaced:
`payment_data[0].invoice_id`; the same path-aware tournament selected that
correctly as well.

Injecting all six path-aware tournament choices as singleton candidate
sequences, while keeping the known-good structural guard flags, reaches raw
`12/12` exact sequence, `12/12` exact arguments, and `12/12` valid JSON. The
final miss audit has `0` failed records and `0` mismatches.

## Interpretation

The corrected candidate schedule is a real sampler improvement:

- model-ranked public exact arguments improve from `3/12` to `9/12`;
- target-selected forcing reaches `12/12` raw exact sequence and arguments;
- max VRAM stays under `20 GiB`, leaving 5090 headroom.

The remaining public gap is no longer local span preservation. It is:

- path-aware value ranking among plausible time/device candidates;
- table-row and array-argument candidate consistency through JSON paths;
- candidate proposal without gold target inclusion;
- eventually learning/dynamically predicting these protected spans from AR
  traces instead of relying on gold schedules.

Do not promote a model checkpoint from this result. Promote the sampler lane and
use the path-aware pairwise selector as the next value-scorer/candidate-proposal
target. This still needs train-only/synthetic selector data and heldout gates
before it becomes model-promotion evidence.
