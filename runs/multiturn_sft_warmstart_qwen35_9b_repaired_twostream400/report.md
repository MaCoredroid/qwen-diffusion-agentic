# Repaired Multi-Turn SFT Warm-Start

## Diagnosis

The trainer path was not the primary defect. The concrete data defect was serialization of
self-generated multi-turn traces: earlier assistant tool-call turns were preserved as
assistant messages, so LMFlow's assistant-mask supervision labeled prior generated calls
as training targets alongside the current audited target. Several rows were also too long
for the 512-token block budget, causing partial label retention under left truncation.

Known-good Run-1 rows were single user-to-assistant targets. The repaired warm-start rows
now match that shape: one compact user prompt containing prior calls as context text and
one assistant target containing only the next audited Qwen-native tool-call block.

## Repair Audit

- Rows: `98`
- Self-generated rows repaired: `49`
- Retention rows copied unchanged: `49`
- Source counts: `google-research-datasets/mbpp:full:train=24`, `openai/gsm8k:main:train=25`, `selfgen_multiturn_exact_audited=49`
- Original token lengths: min `97`, p50 `672`, p90 `971`, max `2641`
- Repaired token lengths: min `97`, p50 `193`, p90 `294`, max `404`
- Original rows over 512 tokens: `49`
- Repaired rows over 512 tokens: `0`
- Original partial labels at 512-left: `4`
- Repaired partial labels at 512-left: `0`
- Repaired rows with full labels kept at 512-left: `98/98`

## Training

- Command wrapper: `scripts/run_flare_redesign_run1.sh`
- Dataset: `data/multiturn_sft_warmstart_repaired/lmflow_dataset`
- Output: `runs/multiturn_sft_warmstart_qwen35_9b_repaired_twostream400`
- Two-stream path: `FASTDLLM_FLARE_TWO_STREAM=1`
- Global steps: `400`
- Epoch: `9.3023`
- Final train loss: `4.5116`
- First logged loss: `8.3731`
- Last logged loss: `4.2028`
- Runtime: `2112.9714` seconds

## Gate

Validated GSM8K retention gate:

```text
.venv/bin/python scripts/eval_flare_stage1_ab_diffusion.py \
  --base-model models/qwen3.5-9b-fastdllm-init \
  --adapter-a runs/multiturn_sft_warmstart_qwen35_9b_repaired_twostream400 \
  --adapter-b runs/multiturn_sft_warmstart_qwen35_9b_repaired_twostream400 \
  --model-names A_diffusion_only \
  --train-path data/multiturn_sft_warmstart_repaired/train_agentic_mix.json \
  --heldout-nll data/flare_stage1_ab_pilot/heldout_nll.jsonl \
  --gsm8k-path data/phaseA_retention/gsm8k_main_test_first20.jsonl \
  --gsm8k-fewshot-path data/phaseA_retention/gsm8k_main_train_first5.jsonl \
  --out-dir runs/multiturn_sft_warmstart_qwen35_9b_repaired_twostream400_gsm8k_gate \
  --skip-nll \
  --generation-tasks gsm8k \
  --generation-limit 20 \
  --generation-batch-size 1 \
  --block-size 32 \
  --small-block-size 32 \
  --max-new-tokens 256 \
  --threshold 0.9 \
  --temperature 0.0 \
  --top-p 0.95
```

Results:

- Disjointness: `full_hash_overlap_count=0`, `content_hash_overlap_count=0`
- GSM8K strict: `2/20 = 0.10`
- GSM8K flex: `3/20 = 0.15`
- Elapsed: `732.5256` seconds
- Generated tokens/sec: `6.6264`

## Stop-Rule Outcome

The third SFT attempt failed the `GSM8K >= 0.70` retention gate. Per stop-rule, the SFT
warm-start path is abandoned. Next work proceeds directly to the diffu-GRPO pilot from
`runs/flare_redesign_run1_copy_grounded_qwen35_9b`.
