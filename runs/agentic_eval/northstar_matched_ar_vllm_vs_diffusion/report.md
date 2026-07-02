# FLARE North-Star Matched Eval

Slice: 20 episodes, 63 turns.
Episode set hash: `baf90863e2fe080a03c32c9bd8473d029ead19f470fe31919ad7001ac3b07871`.
Generated-history loop: prefix-stable completion prompts; each backend appends its sampled assistant text, then the same synthetic tool-result schema and next generation prompt.

| Backend | exact_args | episode exact | exact_seq | valid_xml | schema_ok | sec/turn | total wall | gen tok/turn | model forwards/turn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AR vLLM FR13 | 46/63 | 13/20 | 54/63 | 58/63 | 57/63 | 1.428 | 89.938s | 101.190 | n/a |
| AR vLLM FR13 guided | 50/63 | 13/20 | 63/63 | 63/63 | 59/63 | 1.120 | 70.573s | 77.619 | n/a |
| Diffusion per-call waves | 55/63 | 15/20 | 63/63 | 63/63 | 58/63 | 1.442 | 90.866s | 75.841 | 8.905 |

## Matched Deltas

- Turn exact-args delta, diffusion - AR: 9 / 63
- Episode exact-args delta, diffusion - AR: 2 / 20
- Wall latency ratio AR / diffusion: 0.990x
- Wall latency ratio diffusion / AR: 1.010x
- Turn exact-args delta, diffusion - AR guided: 5 / 63
- Episode exact-args delta, diffusion - AR guided: 2 / 20
- Valid XML delta, diffusion - AR guided: 0 / 63
- Schema-valid delta, diffusion - AR guided: -1 / 63
- Wall latency ratio AR guided / diffusion: 0.777x
- Wall latency ratio diffusion / AR guided: 1.288x

## Fairness Manifest

- AR: `/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-b1000-vllm-bf16`, bf16, no quant, FR13 APC on.
- AR guided: regex structured outputs from Qwen XML tool schemas, FR13 APC on.
- Diffusion: `/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-init` + `/home/mark/qwen_diffusion/runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000`, bf16, no quant.
- Prompt tokenizer: `/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-b1000-vllm-bf16`.
- Chat template: `/home/mark/shared/lumoFlyWheel_codex_fork/docker/chat_templates/qwen3-openai-codex.jinja` (`c166a05aaf5ad4b807a7c46497f92180e3df24e64d4b54d27fd26ec61bec38da`).
- Prompt loop: `{"assistant_generation_prompt": "<|im_start|>assistant\n<think>\n\n</think>\n\n", "followup_prompt": "previous_prompt_plus_sampled_assistant_plus_tool_response_plus_generation_prompt", "initial_prompt": "chat_template_with_tools_and_generation_prompt", "mode": "prefix_stable_incremental_completion_prompt", "tool_response_role": "user"}`.
- Stop policy: `{"ar_include_stop_str_in_output": true, "diffusion_stop_token_included": true, "max_new_tokens": 384, "stop_string": "</tool_call>", "temperature": 0.0, "turn_budget": "same_absolute_max_new_tokens"}`.
- Server launch notes: `{"ar_guided_gpu_memory_utilization": 0.745, "ar_guided_kv_cache_tokens": 64950, "ar_guided_reason": "0.75 startup failed because GNOME-held GPU memory left 23.39 GiB free vs vLLM requested 23.49 GiB", "ar_unguided_gpu_memory_utilization": 0.75, "unchanged": ["same model/export", "same max_model_len=4096", "same FR13 APC enabled", "same mamba_block_size=1024", "same dtype=bf16", "same gdn_prefill_backend=triton"]}`.
- Full manifest: `runs/agentic_eval/northstar_matched_ar_vllm_vs_diffusion/fairness_manifest.json`.
- Per-turn rows: `runs/agentic_eval/northstar_matched_ar_vllm_vs_diffusion/ar-vllm/turns.jsonl`, `runs/agentic_eval/northstar_matched_ar_vllm_vs_diffusion/ar-vllm-guided/turns.jsonl`, `runs/agentic_eval/northstar_matched_ar_vllm_vs_diffusion/diffusion/turns.jsonl`.
