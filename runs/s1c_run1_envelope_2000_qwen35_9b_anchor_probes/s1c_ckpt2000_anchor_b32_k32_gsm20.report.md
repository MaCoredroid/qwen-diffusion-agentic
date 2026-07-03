# Qwen3.5 B@1000 Block Quality Curve

Run name: `s1c_ckpt2000_anchor_b32_k32_gsm20`
Anchor gate passed: `True`
Anchor strict accuracy: `0.3`

## Configuration

- Base model: `/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-init`
- Adapter: `/home/mark/qwen_diffusion/runs/s1c_run1_envelope_2000_qwen35_9b/checkpoint-2000/adapter_model`
- Max new tokens: `256`
- Temperature: `0.0`
- Top-p: `0.95`
- Sampler: mutable-remask fixed-K full-context fresh blocks, mask-token banned

## AR Timing

- Not measured.

## Quality Curve

| Slice | B | K | Nominal toks/fwd | Actual toks/fwd | Accuracy | Correct | Mean diff fwd s | Wall vs cached AR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gsm8k_first20_strict | 32 | 32 | 1.000 | 0.902 | 0.300 | 6/20 | 0.1072 | - |

## Held-Quality Headline

