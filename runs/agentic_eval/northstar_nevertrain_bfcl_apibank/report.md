# FLARE North-Star Matched Eval

Slice: 60 episodes, 184 turns.
Episode set hash: `1987ee5e9e440475bf69ed307e3b5129a210e1ea66748168cd838615f7c8bd8c`.
Generated-history loop: prefix-stable completion prompts; each backend appends its sampled assistant text, then the same synthetic tool-result schema and next generation prompt.

| Backend | exact_args | episode exact | exact_seq | valid_xml | schema_ok | sec/turn | total wall | gen tok/turn | model forwards/turn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AR vLLM FR13 | 74/184 | 19/60 | 118/184 | 169/184 | 166/184 | 0.680 | 125.088s | 46.234 | n/a |
| AR vLLM FR13 guided | 77/184 | 19/60 | 126/184 | 184/184 | 184/184 | 0.591 | 108.684s | 39.049 | n/a |
| Diffusion per-call waves | 181/184 | 57/60 | 184/184 | 184/184 | 184/184 | 0.875 | 160.983s | 40.152 | 0.598 |


## Headline

- Diffusion >= AR-guided on exact-args and episode exactness: YES (turns 181/184 vs 77/184; episodes 57/60 vs 19/60).
## Matched Deltas

- Turn exact-args delta, diffusion - AR: 107 / 184
- Episode exact-args delta, diffusion - AR: 38 / 60
- Wall latency ratio AR / diffusion: 0.777x
- Wall latency ratio diffusion / AR: 1.287x
- Turn exact-args delta, diffusion - AR guided: 104 / 184
- Episode exact-args delta, diffusion - AR guided: 38 / 60
- Turn exact-args flips: diffusion-only 104; AR-guided-only 0; both 77; neither 3
- Episode exact-args flips: diffusion-only 38; AR-guided-only 0; both 19; neither 3
- Valid XML delta, diffusion - AR guided: 0 / 184
- Schema-valid delta, diffusion - AR guided: 0 / 184
- Wall latency ratio AR guided / diffusion: 0.675x
- Wall latency ratio diffusion / AR guided: 1.481x

## Source Breakdown

| Source | Backend | exact_args | episode exact | exact_seq | valid_xml | schema_ok | sec/turn |
|---|---|---:|---:|---:|---:|---:|---:|
| API-Bank-Lv1 | AR vLLM FR13 | 7/13 | 7/13 | 12/13 | 12/13 | 12/13 | 0.803 |
| API-Bank-Lv1 | AR vLLM FR13 guided | 7/13 | 7/13 | 13/13 | 13/13 | 13/13 | 0.903 |
| API-Bank-Lv1 | Diffusion per-call waves | 12/13 | 12/13 | 13/13 | 13/13 | 13/13 | 1.753 |
| API-Bank-Lv2 | AR vLLM FR13 | 4/12 | 4/12 | 9/12 | 12/12 | 12/12 | 1.034 |
| API-Bank-Lv2 | AR vLLM FR13 guided | 4/12 | 4/12 | 9/12 | 12/12 | 12/12 | 1.043 |
| API-Bank-Lv2 | Diffusion per-call waves | 10/12 | 10/12 | 12/12 | 12/12 | 12/12 | 1.888 |
| BFCL-AST | AR vLLM FR13 | 12/12 | 8/8 | 12/12 | 12/12 | 12/12 | 0.645 |
| BFCL-AST | AR vLLM FR13 guided | 12/12 | 8/8 | 12/12 | 12/12 | 12/12 | 0.650 |
| BFCL-AST | Diffusion per-call waves | 12/12 | 8/8 | 12/12 | 12/12 | 12/12 | 0.859 |
| BFCL-multi_turn | AR vLLM FR13 | 51/147 | 0/27 | 85/147 | 133/147 | 130/147 | 0.643 |
| BFCL-multi_turn | AR vLLM FR13 guided | 54/147 | 0/27 | 92/147 | 147/147 | 147/147 | 0.521 |
| BFCL-multi_turn | Diffusion per-call waves | 147/147 | 27/27 | 147/147 | 147/147 | 147/147 | 0.716 |

### Diffusion vs AR-Guided by Source

- API-Bank-Lv1: exact_args delta 5 / 13; episode delta 5 / 13; turn flips diffusion-only 5, AR-guided-only 0; episode flips diffusion-only 5, AR-guided-only 0.
- API-Bank-Lv2: exact_args delta 6 / 12; episode delta 6 / 12; turn flips diffusion-only 6, AR-guided-only 0; episode flips diffusion-only 6, AR-guided-only 0.
- BFCL-AST: exact_args delta 0 / 12; episode delta 0 / 8; turn flips diffusion-only 0, AR-guided-only 0; episode flips diffusion-only 0, AR-guided-only 0.
- BFCL-multi_turn: exact_args delta 93 / 147; episode delta 27 / 27; turn flips diffusion-only 93, AR-guided-only 0; episode flips diffusion-only 27, AR-guided-only 0.

## Fairness Manifest

- AR: `/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-b1000-vllm-bf16`, bf16, no quant, FR13 APC on.
- AR guided: regex structured outputs from Qwen XML tool schemas, FR13 APC on.
- Diffusion: `/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-init` + `/home/mark/qwen_diffusion/runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000`, bf16, no quant.
- Prompt tokenizer: `/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-b1000-vllm-bf16`.
- Chat template: `/home/mark/shared/lumoFlyWheel_codex_fork/docker/chat_templates/qwen3-openai-codex.jinja` (`c166a05aaf5ad4b807a7c46497f92180e3df24e64d4b54d27fd26ec61bec38da`).
- Prompt loop: `{"assistant_generation_prompt": "<|im_start|>assistant\n<think>\n\n</think>\n\n", "followup_prompt": "previous_prompt_plus_sampled_assistant_plus_tool_response_plus_optional_next_user_plus_generation_prompt", "initial_prompt": "chat_template_with_tools_and_generation_prompt", "mode": "prefix_stable_incremental_completion_prompt", "next_user_role": "user", "tool_response_role": "user"}`.
- Stop policy: `{"ar_include_stop_str_in_output": true, "diffusion_stop_token_included": true, "max_new_tokens": 384, "stop_string": "</tool_call>", "temperature": 0.0, "turn_budget": "same_absolute_max_new_tokens"}`.
- Full manifest: `runs/agentic_eval/northstar_nevertrain_bfcl_apibank/fairness_manifest.json`.
- Per-turn rows: `runs/agentic_eval/northstar_nevertrain_bfcl_apibank/ar-vllm/turns.jsonl`, `runs/agentic_eval/northstar_nevertrain_bfcl_apibank/ar-vllm-guided/turns.jsonl`, `runs/agentic_eval/northstar_nevertrain_bfcl_apibank/diffusion/turns.jsonl`.
