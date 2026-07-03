# RL-v2 K-Transfer Matched-20 Probe

Verdict: **quality transfers, speed does not materially transfer on this careful sampler path**. K=16 and K=8 both preserve the K=32 reference exact-args result (`44/63`) and episode exactness (`11/20`) with clean value-projection audits, but denoise forwards stay flat at `95.508` forwards/turn and tokens/forward stays about `1.016`.

| K | exact_args | episode_exact | sec/turn | wall speedup vs K=32 | forwards/turn | tokens/forward | value projected |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 44/63 | 11/20 | 6.686 | 1.000x | 95.508 | 1.016 | 0 |
| 16 | 44/63 | 11/20 | 6.678 | 1.001x | 95.508 | 1.016 | 0 |
| 8 | 44/63 | 11/20 | 6.494 | 1.030x | 95.508 | 1.016 | 0 |

## Interpretation

- Agentic exact-args survival: yes. The 63-turn decision pattern is unchanged at K=16 and K=8 versus the K=32 reference.
- Speed transfer: no material transfer in this implementation. K=8 gives only a small wall improvement (`1.030x`) while forwards/turn and tokens/forward are unchanged; K=16 is effectively identical to K=32.
- Audit: all rows have `zero_projected_value_tokens_verified=1`, `offset_source:generated_token_ids=63`, and `zero_forward_rows=0`.

## Harness Pin

- Sampler: `scripts/eval_flare_northstar_matched.py::run_diffusion -> scripts/eval_fastdllm_toolcall_cases.py::full_context_sample`.
- Git hash at run: `43c1b7f73065f151f7089ee3e908b8843bced885`.
- Decode condition: `baseline_careful`, `block_size=32`, `small_block_size in {32,16,8}`.
- Script sha256: `eval_flare_northstar_matched.py=4cda3acf752c093a0ee3d3e1208c2cdc5deb064b027d984cdf54e8fa93b6203f`, `audit_value_projection_tokens.py=7b203e3e8e2a7a7bbfa6f831be295543c728b08d9228bd241f0f07e35a620b40`.
