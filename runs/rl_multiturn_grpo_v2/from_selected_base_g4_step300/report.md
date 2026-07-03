# Multi-Turn diffu-GRPO RL-v2 Pilot

Verdict: **RETENTION PASS / NOT PROMOTED**. The pilot started from the approved Run-1 copy-grounded checkpoint (`runs/flare_redesign_run1_copy_grounded_qwen35_9b`) and passed the corrected-careful GSM8K retention gate exactly: strict `13/20 = 0.65`, flex `13/20 = 0.65`, required `>=0.65`. Matched-20 improved to `44/63` exact_args, but did not clear the `50/63` promotion bar. No S2 launch was performed; S2 remains a user stop-point.

## Setup

- Base adapter: Run-1 copy-grounded checkpoint, corrected careful GSM8K `0.75`, matched-20 careful `34/63`.
- RL-v2: `300` steps, group size `4`, mixed adjacent public episodes, KL-to-base coefficient `0.05`.
- Policy loss: raw replay logprob over parameter-value/free assistant tokens only. Grammar-forced structural tokens were masked out of GRPO policy loss.
- Leak filter: `240/240` public-pool rows kept; rejected `0` against the eval batteries.

## Training Accounting

| metric | value |
| --- | ---: |
| steps | 300 |
| nonzero-advantage steps | 214/300 |
| zero-advantage steps | 86/300 |
| value/free policy tokens | 13858 |
| grammar-forced structural tokens masked | 58886 |
| mean step reward | 0.9556 |
| mean reward last 20 | 0.9796 |
| max KL-to-base loss | 0.07482 |
| max grad norm | 43.746 |
| elapsed hours | 3.98 |

## Gates

| eval | result | bar | pass |
| --- | ---: | ---: | --- |
| GSM8K strict | 13/20 = 0.65 | >=0.65 | yes |
| GSM8K flex | 13/20 = 0.65 | >=0.65 | yes |
| matched-20 exact_args | 44/63 | 50/63 | no |
| matched-20 vs Run-1 start | +10/63 over 34/63 | improve and retain | partial |
| matched-20 vs RL direct pilot | +6/63 over 38/63 | improve | yes |

## Matched-20 Details

| row | exact_args | episode_exact | sec/turn | forwards/turn | value projected |
| --- | ---: | ---: | ---: | ---: | ---: |
| diffusion careful, Run-1 base before RL | 34/63 | 8/20 | 6.049 | 95.254 | 0 |
| diffusion careful, RL direct pilot step200 | 38/63 | 7/20 | 5.793 | 86.048 | 0 |
| diffusion careful, RL-v2 KL0.05 step300 | 44/63 | 11/20 | 6.686 | 95.508 | 0 |

Projection audit: `zero_projected_value_tokens_verified=1`, `rows=63`, `offset_source:generated_token_ids=63`, `zero_forward_rows=0`. Prefix cache hits: `42/43` eligible follow-up turns.

## Harness Pin

- Matched-20 sampler: `scripts/eval_flare_northstar_matched.py::run_diffusion -> scripts/eval_fastdllm_toolcall_cases.py::full_context_sample`.
- Matched-20 git hash at run: `bdc8001730c5c64443f8047e53e1bc20200a233a`.
- Matched-20 script sha256: `eval_flare_northstar_matched.py=4cda3acf752c093a0ee3d3e1208c2cdc5deb064b027d984cdf54e8fa93b6203f`, `audit_value_projection_tokens.py=7b203e3e8e2a7a7bbfa6f831be295543c728b08d9228bd241f0f07e35a620b40`.
- GSM8K gate sampler: `scripts/eval_flare_stage1_ab_diffusion.py` full-context legacy careful path, script sha256 `eaa78d7a9abfb32b7ab73c7753cf87026741e372ef13a1c0f8e44ead79b5e503`.

Interpretation: KL-to-base plus the value/free-token-only replay loss avoided the catastrophic retention collapse from the first RL pilot, and the matched-20 score moved from `34/63` to `44/63`. The effect is real but insufficient for promotion; the `50/63` bar remains unmet.
