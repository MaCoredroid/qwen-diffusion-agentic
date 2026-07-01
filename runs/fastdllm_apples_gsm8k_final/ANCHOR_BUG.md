# GSM8K Commit-Anywhere Anchor Bug

Date: 2026-07-01

This partial sweep is intentionally stopped and should not be used as a held-quality anchor.

The literal `greedy_one` run in this directory scored only `5/20` strict (`0.25`) and `4/20` flex (`0.20`) on the
phaseA GSM8K first-20 slice:

- `runs/fastdllm_apples_gsm8k_final/baseline_greedy_one.summary.json`
- `decode_mode=greedy_one`
- `block_size=32`
- `small_block_size=32`
- `max_new_tokens=256`
- `prompt_mode=phasea_fewshot`

That does not reproduce the validated B@1000 diffusion-generation reference of roughly `0.65-0.70` GSM8K strict.
Therefore the tau sweep would compare against a broken baseline and cannot answer the intended held-quality
Fast-dLLM apples-to-apples question.

The stopped partial tau artifacts are left in place for audit only. Redo this experiment only after the evaluator first
reproduces the validated B@1000 GSM8K configuration in-harness, then sweep commit-anywhere tau against that baseline.
