# S1-Final Careful GSM8K Check

## Result

S1-final careful GSM8K fails the retention sanity check.

- Adapter: `/home/mark/qwen_diffusion/runs/s1_budget_retrain_r64_qwen35_9b`
- Decode path: `scripts/eval_flare_stage1_ab_diffusion.py` full-context recompute careful generation
- Slice: `data/phaseA_retention/gsm8k_main_test_first20.jsonl`
- Fewshot: `data/phaseA_retention/gsm8k_main_train_first5.jsonl`, 5 examples
- Config: `block_size=32`, `small_block_size=32`, `threshold=0.9`, `temperature=0.0`, `top_p=0.95`, `max_new_tokens=256`
- Kernel env: `FASTDLLM_GDN_KERNEL=torch`, matching the S1 gate wrapper

| Metric | Value |
| --- | ---: |
| Strict GSM8K | 6/20 = 0.300 |
| Flex GSM8K | 6/20 = 0.300 |
| Unresolved-mask examples | 0 |
| Generated tokens | 2178 |
| Elapsed seconds | 266.942 |
| Generated tokens/sec | 8.159 |

## Interpretation

The S1 damage is general, not only block-anchor-specific. S1-final is far below the expected careful retention band near 0.70, so the 0.10 -> 0.25 -> 0.25 block-anchor trend is consistent with broader erosion from extended training under the S1 recipe.

This strengthens the contraindication for r128 escalation before the S1c single-variable step-budget probe.

## Artifacts

- Summary: `runs/s1_budget_retrain_r64_qwen35_9b_careful_gsm8k_final/summary.json`
- Per-example rows: `runs/s1_budget_retrain_r64_qwen35_9b_careful_gsm8k_final/A_diffusion_only_generation.jsonl`
- Raw log: `runs/s1_budget_retrain_r64_qwen35_9b_careful_gsm8k_final/eval.log`
