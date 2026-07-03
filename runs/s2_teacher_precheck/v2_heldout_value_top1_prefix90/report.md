# S2.0 Teacher Value-Span Precheck

Verdict: **FAIL**

| Metric | Value |
|---|---:|
| Top-1 value-token accuracy | 0.5556 |
| Threshold | 0.6000 |
| Correct / scored tokens | 100 / 180 |
| Exact masked value spans | 65 / 84 |
| Scored rows | 12 / 12 |
| Cropped rows | 0 |

## Teacher Lineage

- Base: `/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-init`
- Teacher adapter: `/home/mark/qwen_diffusion/runs/rl_multiturn_grpo_v2/from_selected_base_g4_step300/adapter_model`
- Selected because v2 held matched-20 at 44/63 while v4 regressed to 37/63.
- Active adapters: `['s2_teacher']`
- `disable_adapter` used: `False`
- All parameters frozen: `True`
- Quantization: `nf4_4bit`

## Pins

- Git HEAD: `d5a6b86b03a15dc5c14ea61fff47249f1293aaa7`
- Script SHA256: `3df71bdbfb33aeed69f85e304455366ef06cd57cf764306fa4230f01766145f7`
- Chat template: `/home/mark/shared/lumoFlyWheel_codex_fork/docker/chat_templates/qwen3-openai-codex.jinja` (`c166a05aaf5ad4b807a7c46497f92180e3df24e64d4b54d27fd26ec61bec38da`)
- Tokenizer: `/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-b1000-vllm-bf16`
- Input: `/home/mark/qwen_diffusion/data/toolcall_eval_native/heldout_seed_multicall_policy_targets_qwen_native.jsonl`
