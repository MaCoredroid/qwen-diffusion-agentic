# Cached-SDTT Round-1 Gates

## Verdict

**FAIL: cached-SDTT is dead for S2.** The probe fails retention, shows no positive block-quality movement, and does not move the fixed-K cliff at `value_tpf>=1.5`.

Kill rule is triggered: both movement gates are null.

## Corpus and Training

| item | value |
| --- | ---: |
| cached records | 160 |
| cached target tokens | 1,146 |
| exact/audit-clean teacher turns | 223 |
| rejected inexact/contaminated turns | 46 |
| rejected targetless exact turns | 63 |
| SDTT micro-steps | 4,000 |
| optimizer steps | 4,000 |
| mean reverse-KL | 0.773 |
| target tokens seen | 28,650 |
| peak CUDA allocated | 20.67 GiB |

Training caveats: sparse top-k reverse-KL over cached teacher support, and a target-preserving 240-token crop was required to fit the two-stream training pass on this desktop 5090.

## Legacy GSM8K Curve

Corrected legacy sampler: `scripts/eval_flare_stage1_ab_diffusion.py::full_context_sample_one`.

| K (`small_block_size`) | Run-1 flex | SDTT flex | SDTT strict | result |
| ---: | ---: | ---: | ---: | --- |
| 32 | 14/20 | 10/20 | 10/20 | retention fail |
| 16 | 15/20 | 10/20 | 10/20 | negative |
| 8 | 15/20 | 10/20 | 10/20 | negative |
| 4 | 11/20 | 11/20 | 11/20 | flat |

Retention gate is `>=0.65`; SDTT K=32 is `0.50`, so retention fails.

## Fixed-K Cliff

K=16 is sufficient for the requested cliff check because observed `tokens_per_forward=1.899` is above `1.5`.

| fixed-K row | exact_args | valid tool call | schema ok | sec/turn | forwards/turn | tokens/forward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SDTT K=16 | 0/63 | 23/63 | 1/63 | 25.652 | 108.952 | 1.899 |

The initial K=16 run OOMed mid-run in the diagnostic sampler while materializing full-sequence logits. I patched `measure_block_quality_curve.py` to apply `lm_head` only to active shifted positions; the sampler schedule and top-confidence visible-set decisions are unchanged. The rerun completed with the same matched-20 episode hash used by the v2 fixed-K probe: `72463dc9d7ac6afa5835fbc8ee0c2b9dcf31c2781104c7d9dd650b6507c8c362`.

K=8 was not run because K=16 already satisfies the `value_tpf>=1.5` cliff condition and scored `0/63` exact_args, so the cliff movement gate is null.

## Conclusion

Cached-SDTT did fit the cached teacher support, but it damaged the corrected legacy retention scale and did not produce a usable schedule-gated tool-call operating point. Per the fallback recipe, S2 is dead on this branch; the remaining decision is quality-only RL.
