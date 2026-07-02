# Qwen3.5 B@1000 Block Quality Curve

Run name: `s1_ckpt1000_gsm8k_anchor`
Anchor gate passed: `False`
Anchor strict accuracy: `0.1`

## Configuration

- Base model: `models/qwen3.5-9b-fastdllm-init`
- Adapter: `runs/s1_budget_retrain_r64_qwen35_9b/checkpoint-1000/adapter_model`
- Max new tokens: `256`
- Temperature: `0.0`
- Top-p: `0.95`
- Sampler: mutable-remask fixed-K full-context fresh blocks, mask-token banned

## AR Timing

- Not measured.

## Quality Curve

| Slice | B | K | Nominal toks/fwd | Actual toks/fwd | Accuracy | Correct | Mean diff fwd s | Wall vs cached AR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gsm8k_first20_strict | 32 | 32 | 1.000 | 0.891 | 0.100 | 2/20 | 0.1459 | - |

## Held-Quality Headline

