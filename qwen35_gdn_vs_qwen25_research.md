# Qwen3.5 GDN vs Qwen2.5 Architecture Check

Date: 2026-06-27.

## Conclusion

Qwen3.5 is materially different from Qwen2.5 for this project. In this memo,
GDN means Gated DeltaNet. Qwen3.5-9B and Qwen3.6-27B are Gated DeltaNet /
full-attention hybrid models. Qwen2.5 is a
Qwen2 dense transformer family model with standard attention layers and no
`layer_types` linear-attention layout.

That means Qwen2.5 diffusion work is still useful for the diffusion objective,
sampler diagnostics, and cheap tool-call curriculum tests, but it is not a
drop-in implementation proxy for Qwen3.5 or Qwen3.6.

## Primary Source Check

| Model | HF architecture/config signal | Layer pattern |
| --- | --- | --- |
| Qwen3.5-9B | `Qwen3_5ForConditionalGeneration`, `model_type: qwen3_5_text`, 32 layers | 24 `linear_attention` GDN layers and 8 `full_attention` layers |
| Qwen3.6-27B | `Qwen3_5ForConditionalGeneration`, `model_type: qwen3_5_text`, 64 layers | 48 `linear_attention` GDN layers and 16 `full_attention` layers |
| Qwen3.6-27B-FP8 | same Qwen3.5-family text architecture as 27B bf16, plus FP8 `e4m3` quantization config | same 48 / 16 split |
| Qwen2.5-7B | `Qwen2ForCausalLM`, `model_type: qwen2`, 28 layers | no `layer_types`; standard Qwen2 attention stack |
| Qwen2.5-1.5B-Instruct | `Qwen2ForCausalLM`, `model_type: qwen2`, 28 layers | no `layer_types`; standard Qwen2 attention stack |

The Qwen3.5-9B model card describes the hidden layout as:

```text
8 x (3 x (Gated DeltaNet -> FFN) -> 1 x (Gated Attention -> FFN))
```

The Qwen3.6-27B model card uses the same pattern scaled to:

```text
16 x (3 x (Gated DeltaNet -> FFN) -> 1 x (Gated Attention -> FFN))
```

The raw configs confirm `full_attention_interval: 4` and a repeating
`linear_attention, linear_attention, linear_attention, full_attention`
`layer_types` list.

NVIDIA's Megatron Bridge Qwen3.5-VL model documentation also describes Qwen3.5
as a hybrid architecture that alternates Gated DeltaNet and full-attention
layers. That independently matches the Hugging Face config signal and supports
treating Qwen3.5/3.6 as a different implementation target from Qwen2.5.

Qwen2.5-7B's model card instead lists a transformer architecture with RoPE,
SwiGLU, RMSNorm, and Attention QKV bias. Its raw config is `model_type: qwen2`
and has no `layer_types` list.

## Consequences For The Diffusion Plan

- Keep Qwen3.5-9B as the first real student target. It exercises the same 3:1
  GDN/full-attention family as Qwen3.6-27B.
- Keep Qwen2.5-1.5B as a cheap lab model only. It can test objectives and eval
  plumbing, but not the Qwen3.5/3.6 GDN implementation.
- The Fast-DLLM conversion cannot be only "Qwen2.5 plus a different config".
  It needs a Qwen3.5/GDN bridge.
- Current v0 bridge assumption remains reasonable for early correctness:
  apply block-diffusion masking to full-attention layers and leave GDN layers
  causal as cross-block state carriers.
- A faster sampler/serving path must cache or recompute GDN recurrent state at
  block boundaries. A normal KV-cache-only sampler is not enough.
- LoRA/QLoRA target modules must include GDN projections as well as attention
  projections. The local bridge target list already includes `in_proj_qkv`,
  `in_proj_z`, `in_proj_b`, `in_proj_a`, and `out_proj`, plus `q_proj`,
  `k_proj`, `v_proj`, and `o_proj`.
- For teacher serving, prefer engines with explicit Qwen3.5/3.6 support.
  SGLang remains the local default; vLLM's Qwen3.6 recipe also describes FP8
  single-GPU serving and MTP speculative decoding.

## Sources

- Qwen3.5-9B model card:
  `https://huggingface.co/Qwen/Qwen3.5-9B`
- Qwen3.5-9B raw config:
  `https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/config.json`
- Qwen3.6-27B model card:
  `https://huggingface.co/Qwen/Qwen3.6-27B`
- Qwen3.6-27B raw config:
  `https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/config.json`
- Qwen3.6-27B-FP8 raw config:
  `https://huggingface.co/Qwen/Qwen3.6-27B-FP8/raw/main/config.json`
- Qwen2.5-7B model card:
  `https://huggingface.co/Qwen/Qwen2.5-7B`
- Qwen2.5-7B raw config:
  `https://huggingface.co/Qwen/Qwen2.5-7B/raw/main/config.json`
- Qwen2.5-1.5B-Instruct raw config:
  `https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/raw/main/config.json`
- vLLM Qwen3.6 recipe:
  `https://recipes.vllm.ai/Qwen/Qwen3.6-27B`
- NVIDIA Megatron Bridge Qwen3.5-VL documentation:
  `https://docs.nvidia.com/nemo/megatron-bridge/latest/models/vlm/qwen35-vl.html`
