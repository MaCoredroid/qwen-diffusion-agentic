# Qwen3.5 Skeleton-Conditioned Value Infill Artifacts

Date: 2026-06-28

## Purpose

Turn the heldout/public sampler findings into the first concrete data path for
model-side training. The current protected ceiling says:

- tool-call mode, names, JSON keys/structure, and stop boundaries form the
  skeleton;
- argument values should be filled under that skeleton from prompt/tool/schema
  evidence;
- public/heldout diagnostic artifacts must not be mixed into trainable
  promotion data.

Builder:

```text
scripts/build_skeleton_value_infill_artifacts.py
```

It consumes an existing candidate-augmented sampler schedule and matching cases,
then emits:

```text
skeleton_value_slots.jsonl
value_candidate_bank.jsonl
boundary_labels.jsonl
value_infill_train.json
summary.json
```

## Diagnostic Heldout Artifacts

Source:

```text
runs/tool_sensitive_block_plans/heldout_seed_policy_evidence_selector/
  sampler_schedule_with_derived_pairwise_choices.jsonl
runs/planner_decomposition/heldout_seed_multicall_policy_targets.jsonl
```

Output:

```text
data/skeleton_value_infill/heldout_policy_diagnostic/
```

Summary:

| metric | count |
|---|---:|
| records | `12` |
| value slots | `123` |
| usable slots | `123` |
| candidate rows | `123` |
| boundary rows | `1441` |
| value-infill instances | `123` |
| promotion allowed | `false` |

This is for diagnostics and heldout analysis only.

## Clean Trainable Artifacts

The earlier `public_train_multicall_gold_cases.jsonl` source is not clean: a
direct provenance check found `11/12` public multi-call user/exact overlaps.
So the trainable artifact path starts from the filtered source:

```text
data/fastdllm_toolcall_train/train_toolcall_no_public_multicall_smoke.json
```

Materialized clean multi-call cases:

```text
data/toolcall_eval/public_train_multicall_no_public_smoke_cases.jsonl
```

Source overlap audit:

```text
data/skeleton_value_infill/public_train_no_public_smoke/source_overlap_audit.json
```

Result:

| audit metric | count |
|---|---:|
| train source records | `85` |
| eval records checked | `37` |
| exact overlaps | `0` |
| user overlaps | `0` |

Trainable output:

```text
data/skeleton_value_infill/public_train_no_public_smoke/
```

Summary:

| metric | count |
|---|---:|
| records | `45` |
| value slots | `331` |
| usable slots | `331` |
| candidate rows | `711` |
| boundary rows | `4667` |
| value-infill train instances | `331` |
| target candidate rows | `331` |
| selected candidate rows | `327` |
| promotion allowed | `true` |

Boundary labels:

| kind | count |
|---|---:|
| `tool_tag` | `508` |
| `json_structure` | `1274` |
| `json_key` | `1632` |
| `tool_name` | `587` |
| `argument_value` | `584` |
| `prose` | `82` |

## Schema Notes

Each value slot records:

- `slot_id`, case id/source/provenance;
- `tool_call_index`, `json_key`, `json_path`, schema type;
- target value/text/tokens;
- whole-candidate values and candidate token sequences;
- selected candidate, target index, and trainability flag;
- focused skeleton and all-slot skeleton;
- nearby same-call peer arguments;
- lightweight evidence matches in the user/tool context.

Each boundary row records the existing schedule span plus:

- recommended block size;
- recommended denoise steps;
- `must_shrink`;
- `must_constrain`;
- `must_be_json_completable`;
- `structure_or_value`.

## Next Gate

Use `data/skeleton_value_infill/public_train_no_public_smoke/value_infill_train.json`
as the first trainable value-slot corpus. The first run should be a small
adapter/side objective, not a broad generator replacement:

1. 25/50/75-step sweep from the active Qwen3.5-9B checkpoint.
2. Low LR, value-slot loss emphasized.
3. Evaluate raw/constrained/protected lanes separately.
4. Require no regression on split-route protected lanes.
5. Require movement in at least one raw or constrained heldout/public value
   metric before promotion.

The heldout diagnostic artifacts remain a measurement target and should not be
used for promotion training.
