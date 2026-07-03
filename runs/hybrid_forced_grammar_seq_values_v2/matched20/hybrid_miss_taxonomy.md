# Hybrid-Clean Matched-20 Miss Taxonomy

Status: complete.

Scope: the 16 `diffusion_hybrid_forced_grammar_seq_values` misses on the matched-20 battery (`47/63` exact_args).

Audit context: hybrid-clean remains label-free and structurally constrained only. The promotion audit reports `projected_value_tokens_exact=0`, `parallel_commit_forced_tokens_counter=0`, and `zero_forward_rows=0`.

## Summary

| primary class | turns | notes |
|---|---:|---|
| wrong_value | 9 | Direct value content, scale, type-shape, identifier, or date errors. |
| history_compounding | 5 | Later turns copy or react to a prior generated wrong value/result id. |
| close_timing | 2 | Value is a strict premature close/prefix of the gold string. |
| missing_or_wrong_call | 0 | No hybrid miss has wrong tool sequence, missing call, or extra call. |
| invalid_xml | 0 | All 16 misses are valid Qwen-native tool calls. |

This is a value-learning problem, not a structure problem. All 16 misses have the correct tool sequence. Hybrid’s grammar lane already solved format/call-order on this slice.

## Per-Row Taxonomy

| ep | turn | primary | secondary | careful exact | AR-guided exact | failure |
|---:|---:|---|---|---:|---:|---|
| 0 | 1 | wrong_value |  | 0 | 0 | `apply_gates`: `h/cx` instead of `H/CX`. |
| 6 | 0 | wrong_value | schema_value_shape | 0 | 0 | `inventory_data`: object emitted where string phrase is required. |
| 6 | 1 | wrong_value | schema_value_shape | 0 | 0 | `accounting_data`: object emitted where string phrase is required. |
| 6 | 2 | wrong_value | schema_value_shape | 0 | 0 | `crm_data`: object emitted where string phrase is required. |
| 7 | 1 | wrong_value | numeric_scale, case_normalization | 0 | 0 | Scenario labels lowercased; percent rates converted to fractions. |
| 7 | 2 | wrong_value | numeric_rounding | 1 | 0 | Portfolio weights rounded to `0.33/0.33/0.34`. |
| 10 | 1 | wrong_value | wrong_entity | 0 | 0 | Selected `device_001/smart_lock` instead of `device_002/smart_light`. |
| 14 | 0 | wrong_value | identifier_abstraction, close_timing | 0 | 0 | `event_id=evt-1`; timestamp loses `Z`; duration normalized. |
| 14 | 1 | history_compounding | wrong_value, close_timing | 0 | 0 | Reuses prior `evt-1`; timestamp loses `Z`; one amenity pluralized. |
| 14 | 2 | history_compounding | wrong_value, close_timing | 0 | 0 | Reuses prior `evt-1`; event timestamp loses `Z`. |
| 18 | 0 | close_timing | wrong_value | 0 | 0 | `CRM` closes before `CRM System`. |
| 18 | 1 | history_compounding | wrong_value | 0 | 0 | Uses generated prior result id instead of `Analysis Results Placeholder`; criteria simplified. |
| 18 | 2 | wrong_value | date_year_shift | 0 | 1 | Campaign dates shift from 2023 to 2024. |
| 19 | 0 | close_timing | wrong_value | 0 | 1 | `Sound` closes before `Soundwave`; artist list is otherwise exact. |
| 19 | 1 | history_compounding |  | 0 | 0 | Propagates prior `festival_id=Sound`; date is exact. |
| 19 | 2 | history_compounding |  | 0 | 1 | Propagates prior `festival_id=Sound`; logistics object is exact. |

## Reward Implications

RL-v6 should train under the hybrid-clean serve policy. The reward should focus on value spans and episode-level compounding, not grammar structure:

- Wrong-value direct reward: exact scalar/string/list/object value matching, with strong penalties for numeric scale shifts, case changes where strings are exact, date/year shifts, and schema type-shape substitutions.
- Close-timing reward: explicit penalty when a string value is a strict prefix of gold and the next gold character is emitted by closing the parameter too early.
- History-compounding reward: episode-level credit so early wrong identifiers like `evt-1` or `Sound` are penalized again when downstream turns copy them.
- Structure reward should be low weight because hybrid already gives `63/63` valid and `63/63` exact sequence on matched-20.

The distance to the `50/63` bar is only three turns. The three AR-guided-only misses are ep18 turn2, ep19 turn0, and ep19 turn2; two of those are close/history issues, not missing-call issues.
