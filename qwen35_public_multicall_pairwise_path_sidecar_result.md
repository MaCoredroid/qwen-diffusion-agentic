# Qwen3.5 Public Multi-Call Path-Aware Pairwise Sidecar Result

Date: 2026-06-28

## Purpose

Test a selector sidecar that compares candidate values pairwise using the full
tool-call path, then injects selected values into the tool-sensitive sampler as
singleton whole-candidate sequences.

This is protected-sampler evidence, not model promotion. The selector and
candidate schedule are built from public eval/gold artifacts.

## Key Finding

The focused value-ranking failures were partly under-specified. The old
selector prompts included `JSON key: client_id`, but not the array path such as
`invoice_data[1].client_id` versus `invoice_data[2].client_id`.

Adding `JSON path` from the miss audit changes the result:

| selector gate | correct |
| --- | ---: |
| masked value-span ranking | `0/5` |
| numeric index ranking | `2/5` |
| pathless pairwise tournament | `3/5` |
| path-aware pairwise A/B comparisons | `60/60` |
| path-aware pairwise tournament | `5/5` |

After those five spans were fixed, one new finance miss surfaced:
`payment_data[0].invoice_id`, generated `INV-303` instead of `INV-301`.
The same path-aware tournament selected the correct value for that span too.

## Scripts

New scripts:

```text
scripts/build_candidate_pairwise_curriculum.py
scripts/eval_fastdllm_candidate_pairwise_ranking.py
scripts/eval_fastdllm_candidate_pairwise_tournament.py
scripts/inject_pairwise_tournament_schedule_choices.py
```

The pairwise prompt now includes `JSON path` when `miss_path` is present.

## Selector Artifacts

Focused five-miss path-aware pairwise curriculum:

```text
data/qwen35_9b_public_multicall_v5_focused_miss_pairwise_path_diag_curriculum
```

Build result:

- rows: `120`
- rejected: `0`
- balanced A/B orders: yes
- p50 chosen length: `1645`
- labels kept: `3`
- promotion allowed: `false`

Checkpoint-275 path-aware pairwise A/B gate:

```text
runs/candidate_ranking/public_multicall_targetcandidate_v5_miss_targets_ckpt275_pairwise_path_rank.jsonl
```

Result: `60/60`, all five groups all-correct, min target margin `0.375`.

Checkpoint-275 path-aware tournament:

```text
runs/candidate_ranking/public_multicall_targetcandidate_v5_miss_targets_ckpt275_pairwise_path_tournament.jsonl
```

Result: `5/5`, `121` pair comparisons.

Newly exposed sixth span:

```text
data/candidate_ranking/public_multicall_pairwise_path_remaining_miss_targets_v5.jsonl
runs/candidate_ranking/public_multicall_pairwise_path_remaining_miss_ckpt275_tournament_v5.jsonl
```

Result: `1/1`, `36` pair comparisons.

## Sampler Injection

Injected schedule:

```text
runs/tool_sensitive_block_plans/public_multicall_gold_sampler_schedule_with_pairwise_path_choices_v5_12.jsonl
```

The injector clears stale oracle `selected_candidate` fields, then restricts
only selector-approved spans to singleton `candidate_sequence_values`.

Final injected spans:

- `schedule_time`, span `39:44`, selected `19:00`
- `start_time`, span `41:47`, selected `23:00`
- `invoice_data[1].invoice_id`, span `275:281`, selected `INV-302`
- `invoice_data[1].client_id`, span `286:292`, selected `CLI-102`
- `invoice_data[2].client_id`, span `334:340`, selected `CLI-103`
- `payment_data[0].invoice_id`, span `415:421`, selected `INV-301`

Important ablation: forcing `selected_candidate_token_ids_by_offset` directly
is unstable on this v5 schedule. The working route is singleton candidate
sequence restriction plus `--force-best-candidate-sequence`.

## Public 12-Case Result

Final generation:

```text
runs/tool_sensitive_block_plans/public_multicall_pairwise_path_singleton_structguard6_ckpt275_generation_v5_12.jsonl
runs/tool_sensitive_block_plans/public_multicall_pairwise_path_singleton_structguard6_ckpt275_generation_v5_12.summary.json
```

Final audit:

```text
runs/tool_sensitive_block_plans/public_multicall_pairwise_path_singleton_structguard6_ckpt275_candidate_miss_audit_v5_12.summary.json
```

Result:

| path | exact sequence | exact args | valid JSON | failed records |
| --- | ---: | ---: | ---: | ---: |
| v5 model-ranked candidate baseline | `12/12` | `9/12` | `12/12` | `3` |
| path-aware pairwise, first five spans | `12/12` | `11/12` | `12/12` | `1` |
| path-aware pairwise, six spans | `12/12` | `12/12` | `12/12` | `0` |

Final sampler counters:

- candidate sequence choices: `37`
- selected-candidate force tokens: `0`
- forced structural schedule tokens: `956`
- forced argument-boundary target tokens: `55`
- tool-name sequence choices: `15`
- elapsed: `384.3s`
- max reserved VRAM: `27.95 GiB`

## Interpretation

This is the strongest current protected-sampler result on the public multi-call
slice. It says the next boundary/value selector should be path-aware and should
operate on whole candidate sequences, not raw numeric indices or standalone
value spans.

For the larger Qwen diffusion objective, this is a useful recipe component:

- dynamic block metadata must include the structural JSON path;
- array/table values should be selected at path/row granularity;
- sidecar choices should narrow candidate sets before denoising, not force
  arbitrary selected tokens as a separate mechanism;
- promotion still requires train-only or synthetic analogue selectors and a
  heldout public/teacher gate, because this result uses public eval/gold
  candidate metadata.

## Follow-Up Gate

The cleaner train/heldout variant is documented in:

```text
qwen35_pathaware_phrase_selector_gate_result.md
```

It propagates `json_path` through the train pipeline, adds conservative
free-form phrase evidence candidates, builds a promotion-eligible train-only
pairwise curriculum, and reaches `12/12` exact sequence, `12/12` exact
arguments, and `12/12` valid JSON on the heldout public 12-case protected
sampler gate by injecting only argument-value selector choices.
