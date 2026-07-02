# Multi-Turn SFT Warm-Start

This corpus is for the SFT warm-start stage only. Tool-call rows are self-generated on public training episodes, then kept only when the audited scorer verifies exact arguments and zero projected value tokens.

## Counts

- Final instances: `98`
- Tool-call instances: `49`
- Retention instances: `49`
- Raw accepted self-generated turns: `36`
- Rollout exact turns: `36/50`

## Gates

- Eval-battery filter rejected rows: `199`
- Value projection audit verified: `1`
- Projected value tokens: `0`

## SFT Settings

- Train with `CONVERSATION_TEMPLATE=fast_dllm_v2_native` so the tool schema prompt and assistant targets use the native function/parameter contract.
- Point `DATASET_DIR` at `data/multiturn_sft_warmstart/lmflow_dataset` so LMFlow only sees the conversation JSON.
- The next gate is GSM8K retention accuracy `>=0.70`; stop before RL if it fails.
