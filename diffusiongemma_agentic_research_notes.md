# DiffusionGemma and Agentic dLLM Research Notes

Date: 2026-06-25

## Bottom Line

DiffusionGemma is the clearest current public example of a serious diffusion LLM
deployment path: block/canvas diffusion, vLLM serving, official fine-tuning recipes,
and published function-calling/coding claims. It is still not proof that diffusion
LMs are good coding-agent backbones. The strongest agentic paper I found argues the
opposite for current dLLMs: they can be fast and useful in non-causal helper roles,
but tool-calling and long-horizon agent loops remain weak because symbolic precision
and causal state tracking degrade under denoising.

For our Qwen diffusion experiment, the practical lesson is:

1. Keep the block-diffusion reproduction/eval harness strict.
2. Track format/tool-call metrics early, not only math/reasoning.
3. Treat diffusion as a candidate fast sampler or non-causal helper until it proves
   itself on BFCL/tau-bench/SWE-style loops.

## DiffusionGemma Recipe Signals

### Model and Serving

Google describes DiffusionGemma as an experimental open diffusion text-generation
model based on Gemma 4 26B MoE with about 3.8B active parameters during inference.
The developer guide describes generation as a 256-token canvas refined in parallel,
with block-autoregressive operation for longer outputs: finalized 256-token blocks
are committed to KV cache, then the next canvas is denoised.

Reported serving/deployment signals:

- Up to 700+ tokens/s on RTX 5090 and 1000+ tokens/s on H100 in Google’s developer
  guide.
- Quantized deployment target within about 18 GB VRAM in the same guide.
- vLLM command path exists using `google/diffusiongemma-26B-A4B-it`,
  `--diffusion-config '{"canvas_length": 256}'`, entropy-bounded sampling, and
  chunked prefill.

Sources:

- Google developer guide: https://developers.googleblog.com/diffusiongemma-the-developer-guide/
- Google model overview: https://ai.google.dev/gemma/docs/diffusiongemma
- Google model card: https://ai.google.dev/gemma/docs/diffusiongemma/model_card

### Sampling

Google’s model card recommends diffusion sampling with entropy-bounded denoising
and adaptive stopping:

- maximum denoising steps: 48
- temperature schedule: linear decay from 0.8 to 0.4
- token selection: choose low-entropy tokens under entropy bound 0.1
- re-noise non-selected tokens

This differs from the current Fast-dLLM confidence-threshold sampler, but both
share the same high-level structure: iterative block/canvas refinement with a
quality/speed dial.

### Fine-Tuning Recipes

Google released a Hackable Diffusion adapter route for DiffusionGemma fine-tuning.
The developer guide describes a Sudoku example where the base model is around 0%
success, while an SFT adapter reaches about 80% success and stops earlier. That is
useful evidence that dLLM adapters can teach a constrained infill/fill-the-canvas
task and reduce denoising steps.

Hackable Diffusion is a JAX toolbox with architecture/corruption/inference/loss/
sampling components and a Gemma fine-tuning integration for text diffusion models.

Sources:

- Hackable Diffusion repo: https://github.com/google/hackable_diffusion
- Gemma diffusion adapter path referenced by Hackable Diffusion:
  https://github.com/google-deepmind/gemma/tree/main/gemma/diffusion/hackable_diffusion_adapter
- Google developer guide section on Sudoku/fine-tuning:
  https://developers.googleblog.com/diffusiongemma-the-developer-guide/

NVIDIA NeMo AutoModel also has a DiffusionGemma SFT/LoRA recipe. The guide says:

- Model: `google/diffusiongemma-26B-A4B-it`
- SFT target: DiffusionGemma 26B-A4B, 26B total / about 4B active
- Supports full fine-tune and LoRA
- Runs on a single 8-GPU node with expert parallelism EP=8
- Uses FSDP2 + expert parallelism
- Mixed precision: fp32 master weights, bf16 compute
- Canvas length: 256
- Training mechanics:
  - causal encoder reads clean prompt + response
  - bidirectional decoder denoises the response canvas
  - uniform random corruption, no mask token
  - self-conditioning is optional
  - MoE router frozen during SFT
  - final response turn is supervised; multi-turn histories are masked

Source:

- NVIDIA NeMo AutoModel DiffusionGemma guide:
  https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/diffusiongemma

## Agentic Coding / Tool Calling Evidence

### Positive Claims

Google’s model card lists:

- native function calling
- thinking mode
- coding and reasoning
- long context up to 256K
- multimodal inputs

The card presents DiffusionGemma as agent-capable, but those are vendor/model-card
claims. They are not enough to conclude that diffusion decoding is already robust
for coding-agent loops.

Source:

- https://ai.google.dev/gemma/docs/diffusiongemma/model_card

### Negative Evidence / Reality Check

The strongest directly relevant paper found is:

`The Bitter Lesson of Diffusion Language Models for Agentic Workflows: A Comprehensive Reality Check`

It evaluates dLLMs including LLaDA/Dream-style models on embodied-agent and
tool-calling settings such as AgentBoard and BFCL. Its core conclusion is that
current dLLMs are not reliable agentic backbones:

- embodied settings: repeated attempts and failure to branch under feedback
- tool-calling settings: poor symbolic precision, malformed strict JSON schemas,
  hallucinated API parameters
- dLLMs may work better in non-causal helper roles such as memory summarization,
  redundant trajectory detection, and tool selection

Sources:

- arXiv HTML: https://arxiv.org/html/2601.12979v1
- OpenReview: https://openreview.net/forum?id=Jm9Syqh4N7
- Code/project page: https://github.com/Coldmist-Lu/DiffuAgent

## Implications for Our Qwen Plan

### What We Should Borrow

- Evaluate block/canvas diffusion with real generation, not just training loss.
- Add denoising-step and confidence/entropy sweeps as first-class metrics.
- Use checkpoint sweeps to confirm training direction.
- Include strict structured-output tests early: IFEval, JSON/schema, BFCL-like
  tool calls.
- Consider a future explicit distillation/KL objective from AR teacher logits,
  but keep the current Fast-dLLM reproduction separate from that research change.

### What We Should Not Assume

- A lower denoising loss means agentic usefulness.
- Faster generation means better multi-turn behavior.
- Function-calling support in a model card means robust coding-agent behavior.
- A Sudoku-style success transfers to SWE-bench or tool calling.

### Recommended Next Benchmarks

Near term:

- checkpoint sweep on GSM8K and IFEval
- unresolved mask-token count
- repetition/truncation rate
- speed: tokens/s and examples/s

Next after local sweep:

- BFCL subset or local JSON-schema tool-call eval
- HumanEval/MBPP or LiveCodeBench slice
- tiny SWE-bench Verified or repo-edit harness only after structured output looks
  stable

## Sources

- Google DiffusionGemma model overview:
  https://ai.google.dev/gemma/docs/diffusiongemma
- Google DiffusionGemma model card:
  https://ai.google.dev/gemma/docs/diffusiongemma/model_card
- Google developer guide:
  https://developers.googleblog.com/diffusiongemma-the-developer-guide/
- Google Hackable Diffusion:
  https://github.com/google/hackable_diffusion
- Google Gemma diffusion adapter:
  https://github.com/google-deepmind/gemma/tree/main/gemma/diffusion/hackable_diffusion_adapter
- NVIDIA NeMo AutoModel DiffusionGemma guide:
  https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/diffusiongemma
- DiffuAgent paper, arXiv:
  https://arxiv.org/html/2601.12979v1
- DiffuAgent paper, OpenReview:
  https://openreview.net/forum?id=Jm9Syqh4N7
- DiffuAgent code:
  https://github.com/Coldmist-Lu/DiffuAgent
