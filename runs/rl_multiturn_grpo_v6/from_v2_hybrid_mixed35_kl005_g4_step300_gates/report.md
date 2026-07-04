# RL-v6 Gates

Status: FAIL / not promoted.

| eval | result | bar | pass |
| --- | ---: | ---: | --- |
| GSM8K strict | 14/20 = 0.700 | >=0.65 | True |
| GSM8K flex | 14/20 = 0.700 | >=0.65 | True |
| matched-20 hybrid exact_args | 47/63 | >=50/63 | False |
| matched-20 vs v2 hybrid | +0/63 | >0 and bar 50 | False |
| never-train hybrid exact_args | 50/124 | >=80 exact args (v2 floor) | False |

Audit:

- matched-20 projected value tokens: `0`; verified=1
- never-train projected value tokens: `0`; verified=1

Harness:

- Git hash: `a5dda1b7e0635e00d081fb099af34ef63f5ce7d5`
- Hybrid sampler: `scripts/eval_flare_northstar_hybrid_clean.py::sample_hybrid_clean`
- GSM8K sampler: `scripts/eval_flare_stage1_ab_diffusion.py::full_context_sample_one`
