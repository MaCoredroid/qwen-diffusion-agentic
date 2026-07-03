# RL-v2 Matched-20 Eval

Result: `44/63` exact_args with clean tokenizer-offset projection audit. This is `+10/63` versus the Run-1 careful baseline `34/63`, and `+6/63` versus the first RL direct pilot `38/63`, but below the promotion bar `50/63`.

| row | exact_args | episode_exact | sec/turn | forwards/turn | value projected |
| --- | ---: | ---: | ---: | ---: | ---: |
| diffusion careful, Run-1 base before RL | 34/63 | 8/20 | 6.049 | 95.254 | 0 |
| diffusion careful, RL direct pilot step200 | 38/63 | 7/20 | 5.793 | 86.048 | 0 |
| diffusion careful, RL-v2 KL0.05 step300 | 44/63 | 11/20 | 6.686 | 95.508 | 0 |

Audit: `zero_projected_value_tokens_verified=1`, `offset_source:generated_token_ids=63`, `zero_forward_rows=0`.

Harness pin: sampler `scripts/eval_flare_northstar_matched.py::run_diffusion -> scripts/eval_fastdllm_toolcall_cases.py::full_context_sample`, git `bdc8001730c5c64443f8047e53e1bc20200a233a`, condition `baseline_careful`, script sha256 `eval_flare_northstar_matched.py=4cda3acf752c093a0ee3d3e1208c2cdc5deb064b027d984cdf54e8fa93b6203f`.
