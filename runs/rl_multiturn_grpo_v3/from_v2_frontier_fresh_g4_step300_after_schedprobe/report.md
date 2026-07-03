# Multi-Turn diffu-GRPO RL-v3 Pilot

Verdict: **RETENTION FAIL / NOT PROMOTED**. RL-v3 continued from the RL-v2 adapter on the difficulty-filtered frontier/fresh pool and completed 300 steps, but the validated full GSM8K retention gate landed at strict/flex `11/20 = 0.55`, below the required `>=0.65`. The matched-20 battery was not run because the retention gate failed.

## Setup

- Warm start: `runs/rl_multiturn_grpo_v2/from_selected_base_g4_step300/adapter_model` (RL-v2, matched-20 `44/63`, GSM8K gate `13/20 = 0.65`).
- Training data: `data/rl_multiturn_v3_frontier_pool/episodes.jsonl`, `199` leak-checked frontier/fresh public episodes.
- RL-v3: `300` steps, group size `4`, KL-to-base coefficient `0.05`, seed `20260702`.
- Policy loss: raw replay logprob over parameter-value/free assistant tokens only. Grammar-forced structural tokens were masked out of GRPO policy loss.

## Training Accounting

| metric | value |
| --- | ---: |
| steps | 300 |
| nonzero-advantage steps | 289/300 |
| zero-advantage steps | 11/300 |
| value/free policy tokens | 18122 |
| grammar-forced structural tokens masked | 83447 |
| mean step reward | 0.8894 |
| mean reward last 20 | 0.9340 |
| max KL-to-base loss | 0.5729 |
| mean KL last 50 | 0.2282 |
| steps with KL >= 0.1 | 64 |
| steps with KL >= 0.2 | 34 |
| max grad norm | 43.208 |
| elapsed hours | 4.17 |

## Gates

| eval | result | bar | pass |
| --- | ---: | ---: | --- |
| quick GSM8K probes | `4/5 = 0.80` at steps 50,100,150,200,250,300 | collapse stop `<0.40` | yes |
| full GSM8K strict | `11/20 = 0.55` | `>=0.65` | no |
| full GSM8K flex | `11/20 = 0.55` | `>=0.65` | no |
| matched-20 exact_args | skipped | `50/63` | no |

## Interpretation

The cheap 5-example probes were optimistic and did not catch the late KL drift. The full validated retention gate shows real retention damage versus RL-v2 (`13/20 -> 11/20`), so RL-v3 cannot be promoted and should not be used for matched-20 headline scoring without a new user decision.

## Harness Pin

- Training script: `scripts/run_rl_multiturn_grpo_v2_pilot.sh`, sha256 `cf58932d16caf8092fc3788fe66bb8e862e00b1bfcdbaa5acf32acc11b2b8066`.
- GSM8K gate: `scripts/eval_flare_stage1_ab_diffusion.py` full-context legacy careful path, sha256 `eaa78d7a9abfb32b7ab73c7753cf87026741e372ef13a1c0f8e44ead79b5e503`.
- Git hash at run: `24f0645ac05c9faa048fcfea5103a7453ea7fc6c`.
