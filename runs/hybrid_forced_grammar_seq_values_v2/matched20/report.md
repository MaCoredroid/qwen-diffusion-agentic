# Hybrid Clean Serving Probe

Status: complete.

Adapter: `runs/rl_multiturn_grpo_v2/from_selected_base_g4_step300/adapter_model`

Input: `data/toolcall_eval_native/flare_scaleup_native_58.jsonl`, matched-20 episode battery, 63 turns.

Sampler: hybrid clean serving. Truly-forced Qwen-native grammar tokens are FSM-committed in bulk without model forwards. Every value token and every non-forced token is decoded sequentially with one denoise forward, preserving the chain rule for values. No value projection.

## Scoreboard

| condition | exact_args | valid_tool_call | exact_tool_sequence | episode_exact | forwards/turn | tokens/forward | sec/turn |
|---|---:|---:|---:|---:|---:|---:|---:|
| careful v2 reference | 44/63 | 62/63 | 57/63 | 11/20 | 95.51 | ~1.02 | 6.69 |
| constrained fixed-K K=16 | 0/63 | 52/63 | 49/63 | 0/20 | 83.56 | 1.84 | 19.39 |
| constrained fixed-K K=8 | 0/63 | 46/63 | 42/63 | 0/20 | 46.10 | 3.72 | 11.03 |
| hybrid grammar-bulk + sequential values | 47/63 | 63/63 | 63/63 | 13/20 | 56.83 | 1.36 | 3.90 |

Paired against careful on the same 63 turns: `+4/-1/58` exact-args.

## Audit

| metric | value |
|---|---:|
| projected value tokens | 0 |
| force counters | 0 |
| zero projected value tokens verified | 1 |
| exact rows dependent on projected values | 0 |
| zero-forward rows | 0 |
| reported model value tokens | 2846 |
| true XML value tokens | 2061 |
| forced grammar tokens | 1293 |
| model structural tokens | 734 |
| value close-timing tokens | 337 |

Audit file: `diffusion_hybrid_forced_grammar_seq_values/projection_value_audit.json`.

## Taxonomy

| condition | exact | invalid XML/tool call | valid but wrong sequence | right sequence, missing required | right sequence, schema invalid | right sequence, schema-ok value mismatch |
|---|---:|---:|---:|---:|---:|---:|
| hybrid | 47 | 0 | 0 | 0 | 3 | 13 |
| careful v2 reference | 44 | 1 | 5 | 1 | 0 | 12 |
| constrained fixed-K K=16 | 0 | 11 | 3 | 17 | 17 | 15 |
| constrained fixed-K K=8 | 0 | 17 | 4 | 2 | 22 | 18 |

## Timing Shape

Hybrid removes the structural-forward cost without touching value conditionals:

- Forward reduction vs careful: `95.51 -> 56.83` forwards/turn, a `1.68x` forward reduction.
- Wall reduction vs careful: `6.69 -> 3.90` sec/turn, a `1.71x` wall speedup.
- Generated-token throughput: `1.36` tokens/forward, because only truly-forced grammar tokens are bulk-committed.
- Close-timing cost is nonzero but bounded on this slice: `337` total close-timing tokens, median `4` per turn, max `25`.
- Forward tail remains value-length dominated: max row was `242` forwards with `237` model value tokens and `25` close-timing tokens.

## S4 Design Review

This is the clean constrained-serving win available today: quality is at or above careful (`47/63` vs `44/63`) while forwards and wall time drop by about `1.7x`, with audited zero value projection.

S4-for-parallel-values has a very low prior. The DSCD precheck showed the single-forward teacher signal was weak; the cached-SDTT probe did not establish useful movement; and the constrained fixed-K probe shows the decomposition directly: grammar masking rescues structure (`52/63` valid at K=16) but values remain `0/63` exact when generated in parallel.

The only S4-style direction still worth a narrow design pass is close-timing, not value content. Hybrid still pays sequential forwards for value-close timing, but that cost is much smaller than value-token content on this slice. Recommendation: ship or harden the hybrid serving path first; consider S4-for-close-timing only if broader timing profiles show close timing, rather than value length, is the dominant remaining latency tail.
