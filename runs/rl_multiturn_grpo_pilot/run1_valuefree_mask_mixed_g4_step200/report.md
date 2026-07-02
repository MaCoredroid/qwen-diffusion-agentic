# Multi-Turn diffu-GRPO Direct Pilot

Verdict: **FAIL / not promoted**. The pilot used the approved Run-1 copy-grounded checkpoint as the base (`runs/flare_redesign_run1_copy_grounded_qwen35_9b`), but the saved RL adapter failed the validated GSM8K retention gate: strict `2/20 = 0.10`, flex `0/20 = 0.00`, below the required `>=0.70`.

## Setup

- Base adapter: Run-1 copy-grounded checkpoint, previously validated at GSM8K `0.75` and matched-20 careful `34/63`.
- SFT warm-start: abandoned per stop-rule; likely residual cause noted separately as tiny `98`-row set trained for `400` steps, about `40` effective epochs.
- RL pilot: `200` steps, group size `4`, mixed adjacent public episodes. Same-prompt groups were deterministic and produced zero advantages, so mixed groups were used transparently for this plumbing pilot.
- Policy loss: raw replay logprob over parameter-value/free tokens only. Grammar-forced structural tokens were masked out of GRPO policy loss.

## Training Accounting

| metric | value |
| --- | ---: |
| steps | 200 |
| nonzero-advantage steps | 166/200 |
| value/free policy tokens | 18516 |
| grammar-forced structural tokens masked | 87075 |
| mean step reward | 0.8895 |
| max grad norm | 14.034 |
| elapsed hours | 3.87 |

## Gates

| eval | result | bar | pass |
| --- | ---: | ---: | --- |
| GSM8K strict | 2/20 = 0.10 | >=0.70 | no |
| GSM8K flex | 0/20 = 0.00 | >=0.70 | no |
| matched-20 exact_args | 38/63 | 50/63 | no |
| matched-20 vs start | +4/63 over 34/63 | improve and retain | no, retention failed |

## Matched-20 Details

| row | exact_args | episode_exact | sec/turn | forwards/turn | value projected |
| --- | ---: | ---: | ---: | ---: | ---: |
| diffusion careful, Run-1 base before RL | 34/63 | 8/20 | 6.049 | 95.254 | 0 |
| diffusion careful, RL adapter | 38/63 | 7/20 | 5.793 | 86.048 | 0 |

Projection audit: `zero_projected_value_tokens_verified=1`, `rows=63`, `offset_source:generated_token_ids=63`, `zero_forward_rows=0`.

Interpretation: masking grammar-forced tokens out of the replay loss worked mechanically, and the matched-20 score rose from `34/63` to `38/63`, but the adapter catastrophically damaged general retention. This pilot is a negative result; do not promote this adapter.
