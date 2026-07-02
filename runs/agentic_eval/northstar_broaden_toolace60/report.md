# FLARE North-Star Matched Eval

Slice: 60 episodes, 191 turns.
Episode set hash: `4547eb57e258de4eb69ca7812b2c63533641616bdb6958694ad9299cbac3b4ef`.
Generated-history loop: prefix-stable completion prompts; each backend appends its sampled assistant text, then the same synthetic tool-result schema and next generation prompt.

| Backend | exact_args | episode exact | exact_seq | valid_xml | schema_ok | sec/turn | total wall | gen tok/turn | model forwards/turn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AR vLLM FR13 | 133/191 | 30/60 | 167/191 | 176/191 | 174/191 | 0.922 | 176.073s | 63.953 | n/a |
| AR vLLM FR13 guided | 142/191 | 32/60 | 189/191 | 191/191 | 187/191 | 0.758 | 144.803s | 50.827 | n/a |
| Diffusion per-call waves | 183/191 | 55/60 | 191/191 | 191/191 | 186/191 | 0.969 | 185.145s | 50.717 | 2.937 |


## Headline

- Diffusion >= AR-guided on exact-args and episode exactness: YES (turns 183/191 vs 142/191; episodes 55/60 vs 32/60).
## Matched Deltas

- Turn exact-args delta, diffusion - AR: 50 / 191
- Episode exact-args delta, diffusion - AR: 25 / 60
- Wall latency ratio AR / diffusion: 0.951x
- Wall latency ratio diffusion / AR: 1.052x
- Turn exact-args delta, diffusion - AR guided: 41 / 191
- Episode exact-args delta, diffusion - AR guided: 23 / 60
- Turn exact-args flips: diffusion-only 43; AR-guided-only 2; both 140; neither 6
- Episode exact-args flips: diffusion-only 24; AR-guided-only 1; both 31; neither 4
- Valid XML delta, diffusion - AR guided: 0 / 191
- Schema-valid delta, diffusion - AR guided: -1 / 191
- Wall latency ratio AR guided / diffusion: 0.782x
- Wall latency ratio diffusion / AR guided: 1.279x

## Source Breakdown

| Source | Backend | exact_args | episode exact | exact_seq | valid_xml | schema_ok | sec/turn |
|---|---|---:|---:|---:|---:|---:|---:|
| ToolACE-derived | AR vLLM FR13 | 87/128 | 17/40 | 113/128 | 118/128 | 117/128 | 0.675 |
| ToolACE-derived | AR vLLM FR13 guided | 92/128 | 19/40 | 126/128 | 128/128 | 128/128 | 0.576 |
| ToolACE-derived | Diffusion per-call waves | 128/128 | 40/40 | 128/128 | 128/128 | 128/128 | 0.748 |
| our-synthetic | AR vLLM FR13 | 46/63 | 13/20 | 54/63 | 58/63 | 57/63 | 1.424 |
| our-synthetic | AR vLLM FR13 guided | 50/63 | 13/20 | 63/63 | 63/63 | 59/63 | 1.127 |
| our-synthetic | Diffusion per-call waves | 55/63 | 15/20 | 63/63 | 63/63 | 58/63 | 1.420 |

### Diffusion vs AR-Guided by Source

- ToolACE-derived: exact_args delta 36 / 128; episode delta 21 / 40; turn flips diffusion-only 36, AR-guided-only 0; episode flips diffusion-only 21, AR-guided-only 0.
- our-synthetic: exact_args delta 5 / 63; episode delta 2 / 20; turn flips diffusion-only 7, AR-guided-only 2; episode flips diffusion-only 3, AR-guided-only 1.

## Fairness Manifest

- AR: `/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-b1000-vllm-bf16`, bf16, no quant, FR13 APC on.
- AR guided: regex structured outputs from Qwen XML tool schemas, FR13 APC on.
- Diffusion: `/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-init` + `/home/mark/qwen_diffusion/runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000`, bf16, no quant.
- Prompt tokenizer: `/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-b1000-vllm-bf16`.
- Chat template: `/home/mark/shared/lumoFlyWheel_codex_fork/docker/chat_templates/qwen3-openai-codex.jinja` (`c166a05aaf5ad4b807a7c46497f92180e3df24e64d4b54d27fd26ec61bec38da`).
- Prompt loop: `{"assistant_generation_prompt": "<|im_start|>assistant\n<think>\n\n</think>\n\n", "followup_prompt": "previous_prompt_plus_sampled_assistant_plus_tool_response_plus_optional_next_user_plus_generation_prompt", "initial_prompt": "chat_template_with_tools_and_generation_prompt", "mode": "prefix_stable_incremental_completion_prompt", "next_user_role": "user", "tool_response_role": "user"}`.
- Stop policy: `{"ar_include_stop_str_in_output": true, "diffusion_stop_token_included": true, "max_new_tokens": 384, "stop_string": "</tool_call>", "temperature": 0.0, "turn_budget": "same_absolute_max_new_tokens"}`.
- Full manifest: `runs/agentic_eval/northstar_broaden_toolace60/fairness_manifest.json`.
- Per-turn rows: `runs/agentic_eval/northstar_broaden_toolace60/ar-vllm/turns.jsonl`, `runs/agentic_eval/northstar_broaden_toolace60/ar-vllm-guided/turns.jsonl`, `runs/agentic_eval/northstar_broaden_toolace60/diffusion/turns.jsonl`.
