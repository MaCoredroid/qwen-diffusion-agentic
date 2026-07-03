# RL-v4 From V2 Mixed-Pool Pilot

Verdict: **RETENTION PASS / MATCHED-20 REGRESSION / NOT PROMOTED**.

RL-v4 started from the v2 adapter, used KL coefficient `0.15`, N=10 retention probes every 50 steps, and a mixed pool with 27.6% solved/easy episodes restored. It stopped at step `222` by the KL guard: last-50 mean KL `0.05026` > `0.05`.

## Gates

| eval | result | bar | pass |
| --- | ---: | ---: | --- |
| GSM8K strict | 14/20 = 0.70 | >=0.65 | yes |
| GSM8K flex | 14/20 = 0.70 | >=0.65 | yes |
| matched-20 exact_args | 37/63 | >=44 to replace v2 teacher; 50 promotion | no |
| matched-20 vs v2 | -7/63 | >=0 | no |

## Training Accounting

| metric | value |
| --- | ---: |
| steps | 222 |
| nonzero-advantage steps | 214/222 |
| zero-advantage steps | 8/222 |
| value/free policy tokens | 12781 |
| grammar-forced structural tokens masked | 56919 |
| mean step reward | 0.9211 |
| max KL-to-base loss | 0.19233 |
| final KL-window mean | 0.05026 |
| elapsed hours | 3.09 |

## Matched-20 Details

| row | exact_args | episode_exact | sec/turn | forwards/turn | value projected |
| --- | ---: | ---: | ---: | ---: | ---: |
| diffusion careful, RL-v2 KL0.05 step300 | 44/63 | 11/20 | 6.686 | 95.508 | 0 |
| diffusion careful, RL-v4 KL0.15 step222 | 37/63 | 8/20 | 6.469 | 91.302 | 0 |

Projection audit: `zero_projected_value_tokens_verified=1`, `offset_source:generated_token_ids=63`, `zero_forward_rows=0`. Prefix cache hits: `41/43` eligible follow-up turns.

## S2 Teacher Selection

Selected teacher for S2: `runs/rl_multiturn_grpo_v2/from_selected_base_g4_step300/adapter_model`.

Reason: v4 held retention, but matched-20 fell to `37/63`, below v2's `44/63`. Per the user rule, v4 is not the teacher.

## Harness Pin

- Matched-20 sampler: `scripts/eval_flare_northstar_matched.py::run_diffusion -> scripts/eval_fastdllm_toolcall_cases.py::full_context_sample`.
- Matched-20 git hash at run: `ee0e1ec08a059c5d67f222e3a75e22938d5c8ebc`.
- Matched-20 script sha256: `eval_flare_northstar_matched.py=4cda3acf752c093a0ee3d3e1208c2cdc5deb064b027d984cdf54e8fa93b6203f`, `audit_value_projection_tokens.py=7b203e3e8e2a7a7bbfa6f831be295543c728b08d9228bd241f0f07e35a620b40`.
- GSM8K gate sampler: `scripts/eval_flare_stage1_ab_diffusion.py` full-context legacy careful path, script sha256 `eaa78d7a9abfb32b7ab73c7753cf87026741e372ef13a1c0f8e44ead79b5e503`.
