# RL-v2 Schedule-Gated Fixed-K Matched-20 Probe

Verdict: **no agentic speed transfer in the unconstrained schedule-gated sampler**. The probe achieved genuine fixed-K throughput, but exact tool-call quality collapsed to `0/63` at both K=16 and K=8.

| row | exact_args | valid XML | episode_exact | sec/turn | forwards/turn | tokens/forward | gen tok/turn | value projected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| confidence-gated careful reference | 44/63 | 62/63 | 11/20 | 6.686 | 95.508 | 1.016 | 97.000 | 0 |
| schedule-gated fixed-K K=16 | 0/63 | 8/63 | 0/20 | 39.789 | 155.175 | 1.966 | 305.127 | 0 |
| schedule-gated fixed-K K=8 | 0/63 | 0/63 | 0/20 | 25.103 | 95.365 | 3.997 | 381.127 | 0 |

## Interpretation

- The requested schedule-gated mechanism did run: K=16 reached `1.966` tokens/forward and K=8 reached `3.997` tokens/forward.
- Agentic quality did not survive: K=16 scored `0/63` exact_args with only `8/63` valid XML; K=8 scored `0/63` exact_args with `0/63` valid XML.
- Wall time did not improve end-to-end because invalid generations usually ran to the `384` token cap. The K=32 careful reference generated `97.0` tokens/turn; K=16 generated `305.1`; K=8 generated `381.1`.
- Projection audit stayed clean: no projected value tokens and no zero-forward rows. This failure is model/sampler format quality, not value projection contamination.

## Harness Pin

- Sampler: fresh masked 32-token blocks, exactly K denoise forwards per block, mutable top-confidence visible set, mask token banned, no confidence-run parallel commit.
- This is a schedule-gated diagnostic scale, not the legacy `small_block_size` continuity scale.
- Git hash at run: `67a45718e86e496674eb40cfffa55e0693d126bd` with new uncommitted diagnostic script sha256 `6bcd45205c406b65082dcfa1ac3f2ff39741cdaae71a51c49cccbd849c9f3f97`.
- Input slice and adapter match the RL-v2 matched-20 reference.
