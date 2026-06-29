# Candidate Ranking Delta

Before: `runs/candidate_ranking/public_train_multicall_qwen35_diffusion_init_masked_span_rank.jsonl`

After: `runs/candidate_ranking/public_train_multicall_qwen35_ckpt275_masked_span_rank.jsonl`

## Accuracy

| Run | Overall | Tool names | Argument values |
| --- | ---: | ---: | ---: |
| diffusion-init | 294/299 (98.3%) | 154/155 (99.4%) | 140/144 (97.2%) |
| checkpoint-275 | 295/299 (98.7%) | 155/155 (100.0%) | 140/144 (97.2%) |

## Delta Counts

- shared examples: `299`
- improved examples: `2`
- regressed examples: `1`
- remaining after-run failures: `4`
- by kind: `{'improved:argument_value': 1, 'improved:tool_name': 1, 'regressed:argument_value': 1, 'remaining:argument_value': 4}`

## Improved

- id: `public_train_toolcall_0000`
  - kind: `tool_name`, call: `1`, key: `name`
  - target: `plan_drilling_operations`
  - before predicted: `analyze_geological_data`; after predicted: `plan_drilling_operations`
  - margins before/after: `-6.75` / `1.5`
  - candidates: `['analyze_geological_data', 'plan_drilling_operations']`
- id: `public_train_toolcall_0054`
  - kind: `argument_value`, call: `2`, key: `start_time`
  - target: `2023-04-22T15:00:00Z`
  - before predicted: `2023-04-22T17:00:00Z`; after predicted: `2023-04-22T15:00:00Z`
  - margins before/after: `-1.0` / `1.0`
  - candidates: `['2023-04-22T15:00:00Z', '2023-04-22T17:00:00Z']`

## Regressed

- id: `public_train_toolcall_0055`
  - kind: `argument_value`, call: `1`, key: `material_id`
  - target: `mat_001`
  - before predicted: `mat_001`; after predicted: `mat_002`
  - margins before/after: `0.0` / `-0.6875`
  - candidates: `['mat_001', 'mat_002', 'trans_001']`

## Remaining Failures

- id: `public_train_toolcall_0023`
  - kind: `argument_value`, call: `0`, key: `schedule_time`
  - target: `19:00`
  - before predicted: `11:00`; after predicted: `11:00`
  - margins before/after: `-8.4375` / `-4.25`
  - candidates: `['19:00', '07:00', '11:00', '06:00', '23:00']`
- id: `public_train_toolcall_0034`
  - kind: `argument_value`, call: `0`, key: `start_time`
  - target: `23:00`
  - before predicted: `22:00`; after predicted: `22:00`
  - margins before/after: `-5.875` / `-0.875`
  - candidates: `['23:00', '07:00', '22:00']`
- id: `public_train_toolcall_0041`
  - kind: `argument_value`, call: `2`, key: `bandwidth_requirement`
  - target: `0`
  - before predicted: `1`; after predicted: `1`
  - margins before/after: `-4.78125` / `-3.71875`
  - candidates: `[0, 1, 2, 3]`
- id: `public_train_toolcall_0055`
  - kind: `argument_value`, call: `1`, key: `material_id`
  - target: `mat_001`
  - before predicted: `mat_001`; after predicted: `mat_002`
  - margins before/after: `0.0` / `-0.6875`
  - candidates: `['mat_001', 'mat_002', 'trans_001']`
