# Qwen Diffusion Experiment Plan

Date: 2026-06-25

## Bottom line

Keep **Qwen3.6-27B** as the end target, but do not start by trying to fully train a
27B block-diffusion model on the available local machines. The realistic path is:

1. Prove the Fast-dLLM conversion mechanics on a small non-GDN baseline.
2. Prove the Gated DeltaNet adaptation on **Qwen3.5-4B**, then **Qwen3.5-9B**.
3. Attempt a **27B LoRA / QLoRA / selective-layer diffusion adapter**.
4. Export/serve the result as **NVFP4** for local 5090 inference.

Revised recommendation after environment setup: make the first real milestone a
**Fast-dLLM v2 reproduction on its native Qwen2.5 target**, then move immediately
to **Qwen3.5-4B Option A**. Treat NVFP4 as an early serving/deployment target, not
the first training format.

For local serving, **FP8 should be treated as a quality/reference format, not the
practical deployment target on the 32 GB RTX 5090**. The official FP8 checkpoint is
about **30.9 GB** of safetensors, while the local 5090 has **32 GB** total VRAM; that
leaves too little room for runtime buffers, block caches, KV/GDN state, MTP, vision,
and long context. The existing NVFP4 checkpoint is about **19.7 GB**, which is the
right direction for local experimentation.

## Facts checked

### Model

Qwen3.6-27B is a 27B dense multimodal model with:

- 64 language layers
- hidden size 5120
- 3:1 hybrid stack: three Gated DeltaNet layers for every one full attention layer
- native 262,144 token context, extensible to ~1,010,000 tokens
- MTP trained with multiple steps
- Hugging Face config class: `Qwen3_5ForConditionalGeneration`
- model type: `qwen3_5`

The config naming matters. Implementation work should target the existing Qwen3.5
code path in Transformers/vLLM/SGLang/NVIDIA bridge code, not assume a distinct
Qwen3.6 architecture class.

Qwen3.5-4B and Qwen3.5-9B use the same 3:1 `linear_attention,
linear_attention, linear_attention, full_attention` pattern. They are better
engineering stepping stones than Qwen2.5 because they exercise the GDN-specific
part of the runbook.

### Hardware

Verified locally:

- RTX 5090
- 32,607 MiB VRAM
- driver 595.71.05
- CUDA reported by `nvidia-smi`: 13.2
- compute capability: 12.0
- core env: `.venv`, PyTorch 2.12.1+cu130, CUDA verified
- vLLM env: `.venv-vllm`, vLLM 0.23.0, CUDA verified
- SGLang env: `.venv-sglang`, SGLang 0.5.9, CUDA verified

Verified over Tailscale SSH:

- Alienware: RTX 5080, 16 GB VRAM, core PyTorch env installed
- GX10: GB10, ~117 GiB unified memory visible to Linux,
  CUDA PyTorch/vLLM verified through `lumo-flywheel-vllm:26.01-py3-v0.19.0`

The 5090 and 5080 are consumer Blackwell cards with no useful NVLink path for a
single distributed 27B training job. Treat them as separate workers unless proven
otherwise. The GB10 box is valuable for memory capacity, but its unified LPDDR5x
bandwidth is much lower than 5090 GDDR7 bandwidth, so it is better for memory-heavy
experiments than for high-throughput full training.

## Precision recommendation

### FP8

Use FP8 for:

- AR teacher/reference runs
- quality comparison
- reduced-context smoke serving on the 5090
- possible adapter training if the training stack supports quantized frozen bases

Do not rely on FP8 for:

- full 262K context on the 5090
- final local serving
- full fine-tuning with optimizer state

Measured from Hugging Face file headers:

- `Qwen/Qwen3.6-27B`: BF16 safetensors index total size ~= 55.6 GB
- `Qwen/Qwen3.6-27B-FP8`: safetensors total ~= 30.9 GB
- `sakamakismile/Qwen3.6-27B-NVFP4`: single safetensors size ~= 19.7 GB

The vLLM Qwen3.6 recipe lists FP8 serving as a **single 40 GB GPU** target and int4
as a **single 24 GB GPU** target. That matches the observed 5090 memory pressure.

### NVFP4 / Q4

Use NVFP4 for:

- local 5090 serving
- throughput experiments
- deployment after diffusion adaptation
- possible QAT/export stage after the adapter works

Do not use NVFP4 as the first training format unless we deliberately adopt
Transformer Engine / ModelOpt QAT. NVFP4 training exists on Blackwell, but wiring it
through Qwen3.5/3.6 GDN, Fast-dLLM, custom masks, recurrent state caches, and
diffusion loss is a separate systems project. For the first pass, keep training
simple and quantize after.

## Feasibility by machine

### Local RTX 5090, 32 GB

Good for:

- NVFP4 Qwen3.6-27B serving at practical context sizes
- FP8 Qwen3.6-27B reduced-context serving tests
- Qwen3.5-4B full or LoRA diffusion pilots
- Qwen3.5-9B LoRA/QLoRA diffusion pilots
- block sampler and harness development
- evaluation workers

Risky or impractical:

- 27B full fine-tuning
- 27B FP8 full-context serving
- 27B BF16 anything beyond sharded/offloaded experiments

### Alienware RTX 5080, assumed 16 GB

Good for:

- Qwen3.5-4B experiments
- smaller eval workers
- data preprocessing
- maybe Qwen3.5-9B in 4-bit

Not enough for:

- 27B FP8
- 27B NVFP4 without offload
- serious 27B training

### GX10 / GB10, assumed 128 GB unified memory

Good for:

- BF16 27B loading
- memory-heavy adapter training
- long-context correctness checks
- exporting/quantizing checkpoints

Risky or slow:

- high-throughput 1B-token training
- multi-node training with the 5090/5080 unless network and software stack are
  explicitly validated

## Experimental sequence

### Phase 0: Environment and baseline serving

Goal: make sure the hardware/software stack is boring before touching diffusion.

1. Create a clean Python environment with PyTorch built for Blackwell, vLLM, SGLang,
   Transformers, Accelerate, PEFT, datasets, and lm-eval or local benchmark tools.
2. Verify CUDA kernels on the 5090 with a small tensor op and a small model.
3. Run Qwen3.6-27B-FP8 on the 5090 with:
   - `--language-model-only`
   - reduced `--max-model-len`, e.g. 8K, 16K, 32K
   - MTP on/off
4. Run the NVFP4 checkpoint on the 5090 with the same prompts.
5. Record:
   - load success/failure
   - VRAM at idle and during generation
   - tokens/sec
   - tool-call formatting
   - short coding prompts

Exit gate: we can serve AR Qwen3.6-27B locally in at least one quantized format and
have a baseline speed/quality table.

### Phase 1: Reproduce Fast-dLLM behavior before GDN work

Goal: avoid debugging Fast-dLLM and Qwen GDN at the same time.

1. Clone/pin Fast-dLLM v2.
2. Reproduce its existing Qwen2.5 1.5B or 7B workflow.
3. Verify:
   - mask token insertion
   - shifted prediction
   - complementary mask objective
   - block-causal attention mask
   - confidence-threshold decode
   - hierarchical cache behavior

Exit gate: a known non-GDN model trains/evals/serves with the expected block sampler.

### Phase 2: GDN proxy on Qwen3.5-4B, then Qwen3.5-9B

Goal: solve the novel architectural part cheaply.

Start with Qwen3.5-4B because it has the same hybrid attention pattern but is small
enough to iterate. Move to 9B only after the 4B path works.

Implementation:

1. Add mask-token handling using a reserved vocab id, initialized from mean embedding.
2. Apply block-causal bidirectional masking only to full attention layers.
3. Implement GDN Option A first:
   - keep GDN causal
   - treat GDN recurrent state as the cross-block state carrier
   - snapshot GDN state at block boundaries
   - rescan only the current block during denoising
4. Include all Qwen GDN state in the boundary cache. The config has
   `linear_conv_kernel_dim: 4`; cache planning should include both recurrent scan
   state and any local convolution state, not just a single abstract matrix.
5. Use LoRA first, targeting GDN projections, full-attention projections, and FFN
   projections. If LoRA underfits badly on 4B, run a short full fine-tune on GB10.
6. Train small:
   - 5M to 10M tokens for smoke
   - 50M to 100M tokens for real signal
   - block warmup: 1 -> 4 -> 16 -> 64, then 128/256 if stable
   - low LR, starting around 1e-5 for full/partial, lower if LoRA destabilizes

Metrics:

- masked denoising CE on held-out code and text
- exact-match infill snippets
- JSON/tool-call validity
- short code edit tasks
- block size vs quality degradation
- comparison to the AR base at equal context

Exit gate: Qwen3.5-4B/9B can generate coherent block-diffusion outputs without
format collapse, and Option A either passes or produces clear evidence for
bidirectional GDN.

### Phase 3: 27B adapter attempt

Goal: get a credible 27B diffusion variant without pretending we can do a full
1B-token dense adaptation locally.

Preferred first attempt:

- base: Qwen3.6-27B BF16 on GB10 or FP8/NF4 frozen base on 5090
- trainable: LoRA/selective adapters
- input: text-only/code-only first; freeze/ignore vision tower
- data: code/repo-edit/infill-heavy mixture plus general text
- scale: 50M tokens first, then 100M-300M if metrics improve
- block warmup: 1 -> 4 -> 16 -> 64 -> 128
- objective: masked diffusion CE, optionally add teacher KL from AR Qwen3.6-FP8 on
  selected positions to reduce drift

Do not start with:

- full 27B Adam fine-tune
- 1B tokens
- 262K context
- bidirectional GDN kernel
- multimodal input

Those are final-stage goals, not first-stage experiments.

Exit gate: the 27B adapter beats the 9B proxy on code/infill and keeps acceptable
tool-call formatting at useful block sizes.

### Phase 4: Quantize and serve

Goal: move from research checkpoint to local interactive inference.

1. Merge adapter if quality is acceptable, or keep base+adapter if serving stack can
   handle it cleanly.
2. Quantize/export to NVFP4 with either LLM Compressor or NVIDIA ModelOpt.
3. Preserve sensitive modules in higher precision if NVFP4 hurts coding/tool-call
   quality. Likely candidates:
   - LM head
   - embeddings
   - first/last layers
   - GDN output gates or recurrent state math
4. Patch vLLM or SGLang diffusion path:
   - block scheduler
   - mask-token canvas
   - confidence/entropy unmasking
   - GDN boundary-state cache
   - block-level stop/tool-call detection
5. Benchmark FP8 vs NVFP4:
   - same prompts
   - same block size and denoising budget
   - same context length
   - tokens/sec and task quality

Exit gate: NVFP4 27B diffusion serving on 5090 is faster than AR FP8/NVFP4 at an
acceptable quality loss.

## Decision gates

### Keep 27B or reduce size?

Keep 27B as the target, but run the first GDN experiments on Qwen3.5-4B/9B. If 9B
does not show a clean speed/quality curve after the GDN state-cache implementation,
27B will probably waste time.

### FP8 or NVFP4?

Use both, but for different purposes:

- FP8: reference quality, teacher, reduced-context serving tests
- NVFP4: local deployment target on 5090

Given the measured 30.9 GB FP8 checkpoint size and the 32 GB 5090, **yes, we should
reconsider FP8 as the local serving target and plan around NVFP4/Q4**.

### Option A or bidirectional GDN?

Start with Option A. It is the cheapest and least invasive hypothesis:

- GDN remains causal and stable.
- Full attention layers provide within-block bidirectional mixing every fourth layer.
- Boundary state caching stays simple.

Only implement bidirectional GDN if 4B/9B tests show obvious within-block
under-mixing, large-block incoherence, or infill failures that do not improve with
more denoising steps.

### Local only or cloud?

Local is enough for:

- prototype
- GDN state-cache research
- 4B/9B training
- 27B adapter attempts
- NVFP4 serving

Cloud or larger multi-GPU hardware is needed for:

- full 27B dense diffusion adaptation
- 1B-token run at practical wall-clock
- serious SWE-bench-scale sweeps

## Immediate next actions

1. Fix SSH access or provide hostnames/IPs for `alienware` and `gx10`.
2. Build the local Blackwell Python environment.
3. Run AR baseline serving:
   - Qwen3.6-27B-FP8 at reduced context
   - Qwen3.6-27B-NVFP4 at the largest stable context
4. Clone Fast-dLLM and reproduce a non-GDN baseline.
5. Implement Qwen3.5-4B GDN Option A and train a 5M-token smoke run.

## Sources

- Qwen3.6-27B model card: https://huggingface.co/Qwen/Qwen3.6-27B
- Qwen3.6-27B-FP8 model card: https://huggingface.co/Qwen/Qwen3.6-27B-FP8
- vLLM Qwen3.6-27B recipe: https://recipes.vllm.ai/Qwen/Qwen3.6-27B
- Qwen3.6-27B-NVFP4 checkpoint: https://huggingface.co/sakamakismile/Qwen3.6-27B-NVFP4
- Fast-dLLM v2 paper: https://arxiv.org/abs/2509.26328
- Fast-dLLM v2 project page: https://nvlabs.github.io/Fast-dLLM/v2/
- DiffusionGemma developer guide: https://developers.googleblog.com/diffusiongemma-the-developer-guide/
- Dream 7B blog: https://hkunlp.github.io/blog/2025/dream/
- Block Diffusion paper: https://arxiv.org/abs/2503.09573
- Agentic dLLM caution paper: https://arxiv.org/abs/2601.12979
- NVIDIA NVFP4 explainer: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
- TensorRT-LLM quantization docs: https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html
- LLM Compressor NVFP4 docs: https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w4a4_fp4/
- NVIDIA DGX Spark hardware docs: https://docs.nvidia.com/dgx/dgx-spark/hardware.html
- NVIDIA RTX 5090 specs: https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/
- NVIDIA RTX 5080 specs: https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5080/
