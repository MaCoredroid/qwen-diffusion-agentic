# Never-Train Decode Forensics

## Verdict

- The published diffusion `181/184` never-train number is invalid for model-quality claims.
- Old diffusion forwards histogram: `{'0': 171, '3': 10, '5': 1, '31': 1, '44': 1}`.
- Old diffusion zero-forward turns: 171/184; zero-forward turns with XML values: 171/184.
- Exact turns dependent on projected values, lower bound: 181/181.
- Strict old-artifact no-value-projection exact, lower-bound filter: 0/184.
- Corrected no-value-projection rerun: 88/184 exact_args, 10.293 forwards/turn.
- Corrected forwards histogram: `{'2': 23, '3': 23, '4': 29, '5': 28, '6': 7, '7': 13, '8': 7, '9': 4, '10': 6, '11': 1, '14': 1, '15': 3, '16': 4, '17': 4, '18': 1, '19': 2, '21': 1, '23': 2, '24': 2, '25': 2, '26': 2, '27': 1, '32': 1, '34': 3, '35': 1, '36': 1, '37': 3, '40': 1, '41': 1, '42': 2, '45': 3, '46': 1, '74': 1}`.
- Corrected wave-1 value tokens: 0; corrected zero-forward turns: 0.

## Corrected Three-Way Table

| backend | exact_args | episode_exact | exact_seq | valid_xml | schema_ok | sec_per_turn | total_wall_seconds | model_forwards_per_turn |
|---|---|---|---|---|---|---|---|---|
| AR vLLM FR13 | 74/184 | 19/60 | 118/184 | 169/184 | 166/184 | 0.68 | 125.087 | n/a |
| AR vLLM FR13 guided | 77/184 | 19/60 | 126/184 | 184/184 | 184/184 | 0.591 | 108.68 | n/a |
| Diffusion per-call waves (corrected) | 88/184 | 19/60 | 141/184 | 182/184 | 147/184 | 1.604 | 295.134 | 10.293 |

## Value Projection Split

- Old wave-1 projected tokens: 7287.
- True XML value tokens in old outputs: 1433.
- Projected true-value tokens lower bound: 1377.
- Projected scaffold tokens upper bound: 5910.

The lower bound is exact for zero-forward turns. For nonzero turns the old logs do not retain per-token source, so the split credits model-sampled value tokens as generously as the counters allow.

## AR-Guided Failure Taxonomy

- AR-guided failures: 107.
- First-20 sample counts: {'missing/extra arg': 13, 'wrong value': 7}.
- All-failure counts: {'missing/extra arg': 66, 'wrong value': 41}.
- Order-insensitive/type-coerced recoveries: 0.

Formatting/type/canonicalization artifacts do not dominate, so no scorer fix was applied.

## Source Breakdown

| source | backend | exact_args | episode_exact | sec_per_turn |
|---|---|---|---|---|
| API-Bank-Lv1 | AR vLLM FR13 | 7/13 | 7/13 | 0.803 |
| API-Bank-Lv2 | AR vLLM FR13 | 4/12 | 4/12 | 1.034 |
| BFCL-AST | AR vLLM FR13 | 12/12 | 8/8 | 0.645 |
| BFCL-multi_turn | AR vLLM FR13 | 51/147 | 0/27 | 0.643 |
| API-Bank-Lv1 | AR vLLM FR13 guided | 7/13 | 7/13 | 0.903 |
| API-Bank-Lv2 | AR vLLM FR13 guided | 4/12 | 4/12 | 1.043 |
| BFCL-AST | AR vLLM FR13 guided | 12/12 | 8/8 | 0.65 |
| BFCL-multi_turn | AR vLLM FR13 guided | 54/147 | 0/27 | 0.521 |
| API-Bank-Lv1 | Diffusion per-call waves (corrected) | 7/13 | 7/13 | 2.89 |
| API-Bank-Lv2 | Diffusion per-call waves (corrected) | 5/12 | 5/12 | 2.985 |
| BFCL-AST | Diffusion per-call waves (corrected) | 10/12 | 7/8 | 1.357 |
| BFCL-multi_turn | Diffusion per-call waves (corrected) | 66/147 | 0/27 | 1.398 |

## Artifacts

- Forensic summary JSON: `runs/agentic_eval/northstar_nevertrain_bfcl_apibank/forensics/forensics_summary.json`
- Per-turn projection audit JSONL: `runs/agentic_eval/northstar_nevertrain_bfcl_apibank/forensics/projection_value_audit.jsonl`
- AR-guided taxonomy JSONL: `runs/agentic_eval/northstar_nevertrain_bfcl_apibank/forensics/ar_guided_failure_taxonomy.jsonl`
- Corrected diffusion turns: `runs/agentic_eval/northstar_nevertrain_bfcl_apibank_forensics/diffusion/turns.jsonl`
