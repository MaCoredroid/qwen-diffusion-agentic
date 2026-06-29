# Candidate Ranking Delta

Before: `runs/candidate_ranking/public_multicall_qwen35_diffusion_init_masked_span_rank_v3_12.jsonl`

After: `runs/candidate_ranking/public_multicall_qwen35_ckpt275_masked_span_rank_v3_12_prefix_only.jsonl`

## Accuracy

| Run | Overall | Tool names | Argument values |
| --- | ---: | ---: | ---: |
| diffusion-init | 78/86 (90.7%) | 31/31 (100.0%) | 47/55 (85.5%) |
| checkpoint-275 | 80/86 (93.0%) | 31/31 (100.0%) | 49/55 (89.1%) |

## Delta Counts

- shared examples: `86`
- improved examples: `3`
- regressed examples: `1`
- remaining after-run failures: `6`
- by kind: `{'improved:argument_value': 3, 'regressed:argument_value': 1, 'remaining:argument_value': 6}`

## Improved

- id: `85f6c398-69c7-4df2-aed1-29d614a93a26`
  - kind: `argument_value`, call: `2`, key: `start_time`
  - target: `2023-04-22T15:00:00Z`
  - before predicted: `2023-04-22T17:00:00Z`; after predicted: `2023-04-22T15:00:00Z`
  - margins before/after: `-1.0` / `1.0`
  - candidates: `['2023-04-22T15:00:00Z', '2023-04-22T17:00:00Z']`
- id: `adc48a37-6341-4ea6-972a-8ec2b5421321`
  - kind: `argument_value`, call: `1`, key: `invoice_id`
  - target: `INV-301`
  - before predicted: `INV-303`; after predicted: `INV-301`
  - margins before/after: `-1.75` / `1.875`
  - candidates: `['XYZ-123', 'INV-301', 'CLI-101', 'INV-302', 'CLI-102', 'INV-303', 'CLI-103', 'PAY-401', 'PAY-402']`
- id: `adc48a37-6341-4ea6-972a-8ec2b5421321`
  - kind: `argument_value`, call: `2`, key: `payment_id`
  - target: `PAY-401`
  - before predicted: `PAY-402`; after predicted: `PAY-401`
  - margins before/after: `-0.125` / `0.875`
  - candidates: `['XYZ-123', 'INV-301', 'CLI-101', 'INV-302', 'CLI-102', 'INV-303', 'CLI-103', 'PAY-401', 'PAY-402']`

## Regressed

- id: `adc48a37-6341-4ea6-972a-8ec2b5421321`
  - kind: `argument_value`, call: `1`, key: `client_id`
  - target: `CLI-103`
  - before predicted: `CLI-103`; after predicted: `CLI-101`
  - margins before/after: `1.125` / `-1.25`
  - candidates: `['XYZ-123', 'INV-301', 'CLI-101', 'INV-302', 'CLI-102', 'INV-303', 'CLI-103', 'PAY-401', 'PAY-402']`

## Remaining Failures

- id: `3f440c20-b332-48e2-aaa5-a7bfb0781ae9`
  - kind: `argument_value`, call: `0`, key: `schedule_time`
  - target: `19:00`
  - before predicted: `11:00`; after predicted: `11:00`
  - margins before/after: `-8.4375` / `-4.25`
  - candidates: `['19:00', '07:00', '11:00', '06:00', '23:00']`
- id: `adc48a37-6341-4ea6-972a-8ec2b5421321`
  - kind: `argument_value`, call: `1`, key: `client_id`
  - target: `CLI-102`
  - before predicted: `CLI-101`; after predicted: `CLI-103`
  - margins before/after: `-0.75` / `-0.5`
  - candidates: `['XYZ-123', 'INV-301', 'CLI-101', 'INV-302', 'CLI-102', 'INV-303', 'CLI-103', 'PAY-401', 'PAY-402']`
- id: `adc48a37-6341-4ea6-972a-8ec2b5421321`
  - kind: `argument_value`, call: `1`, key: `client_id`
  - target: `CLI-103`
  - before predicted: `CLI-103`; after predicted: `CLI-101`
  - margins before/after: `1.125` / `-1.25`
  - candidates: `['XYZ-123', 'INV-301', 'CLI-101', 'INV-302', 'CLI-102', 'INV-303', 'CLI-103', 'PAY-401', 'PAY-402']`
- id: `adc48a37-6341-4ea6-972a-8ec2b5421321`
  - kind: `argument_value`, call: `1`, key: `invoice_id`
  - target: `INV-302`
  - before predicted: `INV-303`; after predicted: `INV-301`
  - margins before/after: `-1.375` / `-0.25`
  - candidates: `['XYZ-123', 'INV-301', 'CLI-101', 'INV-302', 'CLI-102', 'INV-303', 'CLI-103', 'PAY-401', 'PAY-402']`
- id: `adc48a37-6341-4ea6-972a-8ec2b5421321`
  - kind: `argument_value`, call: `2`, key: `invoice_id`
  - target: `INV-301`
  - before predicted: `INV-303`; after predicted: `INV-303`
  - margins before/after: `-1.875` / `-0.125`
  - candidates: `['XYZ-123', 'INV-301', 'CLI-101', 'INV-302', 'CLI-102', 'INV-303', 'CLI-103', 'PAY-401', 'PAY-402']`
- id: `e279e98f-095a-4d44-9c2d-170b3cfdc4bb`
  - kind: `argument_value`, call: `0`, key: `start_time`
  - target: `23:00`
  - before predicted: `22:00`; after predicted: `22:00`
  - margins before/after: `-5.875` / `-0.875`
  - candidates: `['23:00', '07:00', '22:00']`
