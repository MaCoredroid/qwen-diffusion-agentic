# Constrained-Parallel Fixed-K Probe

Status: complete.

Adapter: `runs/rl_multiturn_grpo_v2/from_selected_base_g4_step300/adapter_model`

Input: `data/toolcall_eval_native/flare_scaleup_native_58.jsonl`, matched-20 episode battery, 63 turns.

Sampler: fixed-K schedule-gated block diffusion with Qwen-native grammar filtering at every committed position. Structural tokens are grammar-constrained; parameter values remain model-chosen from logits. No gold value projection.

## Scoreboard

| condition | exact_args | valid_tool_call | exact_tool_sequence | episode_exact | forwards/turn | tokens/forward | sec/turn |
|---|---:|---:|---:|---:|---:|---:|---:|
| careful v2 reference | 44/63 | 62/63 | 57/63 | 11/20 | 95.51 | ~1.02 | 6.69 |
| raw fixed-K K=16 reference | 0/63 | 8/63 | 1/63 | 0/20 | 155.17 | 1.97 | 39.79 |
| raw fixed-K K=8 reference | 0/63 | 0/63 | 0/63 | 0/20 | 95.37 | 4.00 | 25.10 |
| constrained fixed-K K=16 | 0/63 | 52/63 | 49/63 | 0/20 | 83.56 | 1.84 | 19.39 |
| constrained fixed-K K=8 | 0/63 | 46/63 | 42/63 | 0/20 | 46.10 | 3.72 | 11.03 |

## Audit

| condition | projected value tokens | force counters | model value tokens reported | true XML value tokens |
|---|---:|---:|---:|---:|
| constrained K=16 | 0 | 0 | 7009 | 3187 |
| constrained K=8 | 0 | 0 | 7884 | 1932 |

Both audits report `zero_projected_value_tokens_verified=1` with `verification_mode=no_projection_events`.

## Taxonomy

| condition | invalid XML/tool call | valid but wrong sequence | right sequence, missing required | right sequence, schema invalid | right sequence, schema-ok value mismatch |
|---|---:|---:|---:|---:|---:|
| constrained K=16 | 11 | 3 | 17 | 17 | 15 |
| constrained K=8 | 17 | 4 | 2 | 22 | 18 |
| raw K=16 | 55 | 7 | 1 | 0 | 0 |
| raw K=8 | 63 | 0 | 0 | 0 | 0 |
| careful v2 | 1 | 5 | 1 | 0 | 12 |

## Verdict

Pinned structure rescues syntax and call order substantially versus raw fixed-K, but it does not rescue exact values. The clean decomposition is:

- Structure failure was real: raw K=8 had `0/63` valid calls; constrained K=8 has `46/63`.
- Value/semantic conditionals remain dead under real parallel value generation: both constrained K=16 and K=8 are `0/63` exact_args.
- The speed surface is real but not useful as-is: K=8 reaches `3.72` tokens/forward and `46.10` forwards/turn, but still `0/63` exact_args.
- S4-style training for parallel values has a low prior unless a train==serve objective can create value conditionals that DSCD/SDTT did not.

Next probe queued by user: hybrid clean serving, with grammar-bulk structural tokens and strictly sequential value spans.
