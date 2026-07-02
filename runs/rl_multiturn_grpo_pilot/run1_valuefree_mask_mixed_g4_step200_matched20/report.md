# RL Direct Pilot Matched-20 Eval

Result: `38/63` exact_args with clean structural-only projection audit. This is `+4/63` versus the pre-RL careful baseline `34/63`, but below the promotion bar `50/63`.

| row | exact_args | episode_exact | sec/turn | forwards/turn | value projected |
| --- | ---: | ---: | ---: | ---: | ---: |
| diffusion careful, Run-1 base before RL | 34/63 | 8/20 | 6.049 | 95.254 | 0 |
| diffusion careful, RL adapter | 38/63 | 7/20 | 5.793 | 86.048 | 0 |

Audit: `zero_projected_value_tokens_verified=1`, `offset_source:generated_token_ids=63`, `zero_forward_rows=0`.

Gate context: this adapter failed the validated GSM8K retention gate (`strict=2/20`, `flex=0/20`; required `>=0.70`), so the matched-20 gain is not promotable.
