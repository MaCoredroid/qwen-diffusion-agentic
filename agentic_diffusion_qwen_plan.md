# Agentic Diffusion-Qwen Plan

Date: 2026-06-25

## Ultimate Goal

Build a converted block-diffusion Qwen-family model that can perform agentic coding
and tool workflows, not merely reproduce Fast-dLLM.

Fast-dLLM v2 is the scaffold. The actual target is a dLLM that can:

- generate valid tool/function calls
- preserve JSON/schema constraints
- perform multi-step reasoning without action-loop collapse
- edit code in repositories
- run through coding-agent harnesses with stable stop/tool boundaries
- eventually serve fast enough locally to matter

## Available Hardware

### Local Workstation

- GPU: RTX 5090
- VRAM: about 32 GB
- Best role:
  - primary fast iteration machine
  - SGLang serving/eval for quantized Qwen3.6-27B teacher/reference at reduced context
  - Qwen3.5-9B LoRA/QLoRA pilots
  - sampler/harness development
  - NVFP4/FP8 serving experiments
- Weak role:
  - full 27B training
  - long-context 27B FP8 serving with large caches

### Alienware over Tailscale

- Tailscale SSH configured; exact private address omitted in the public notes.
- GPU: RTX 5080
- VRAM: about 16 GB
- Best role:
  - small eval worker
  - 1.5B/4B smoke jobs
  - data preprocessing
  - lightweight LoRA/QLoRA if memory fits
- Weak role:
  - 27B serving/training
  - larger block/canvas sweeps

### GX10 / GB10 over Tailscale

- Tailscale SSH configured; exact private address omitted in the public notes.
- Hardware: GB10-class system
- RAM/unified memory seen by Linux: about 117 GB
- Best role:
  - memory-heavy model loading
  - 9B/27B correctness experiments
  - quant/export experiments
  - long-context smoke tests
  - jobs where capacity matters more than raw GDDR bandwidth
- Weak role:
  - high-throughput training compared with 5090 GDDR7
  - multi-node training unless networking/software is explicitly validated

## Current State

We have:

- Fast-dLLM v2 repo cloned and patched locally.
- Local hybrid Fast-dLLM/Qwen2.5-1.5B base:
  `/home/mark/qwen_diffusion/models/qwen2.5-1.5b-fastdllm-init`
- Completed Alpaca LoRA run:
  `/home/mark/qwen_diffusion/runs/fastdllm_qwen25_1p5b_alpaca_lora_full`
- Adapter-aware `lm-eval` path in `fast-dllm/v2/eval.py`.
- Isolated eval environment:
  `/home/mark/qwen_diffusion/.venv-lmeval`
- Checkpoint sweep runner:
  `/home/mark/qwen_diffusion/scripts/run_fastdllm_checkpoint_sweep.py`
- DiffusionGemma / agentic dLLM research notes:
  `/home/mark/qwen_diffusion/diffusiongemma_agentic_research_notes.md`

The Alpaca LoRA is functional but far behind the released Fast-dLLM v2 1.5B
checkpoint. That is expected; it is a plumbing proof, not the final training
recipe.

## Principle

Do not optimize for benchmark reproduction alone. Optimize for agentic failure
modes:

- malformed tool calls
- invalid JSON
- wrong function choice
- repeated action loops
- failure to stop
- lossy reasoning under large blocks
- code edits that do not apply
- output that looks fluent but violates schema

## Phase 1: SGLang Teacher / Reference Serving

Goal: put a strong Qwen3.6-family AR model behind an OpenAI-compatible local
endpoint so it can act as:

- label generator
- repair/verifier for tool-call data
- AR quality baseline
- logit/behavior teacher for later distillation
- reference implementation for Qwen tool-call formatting

Preferred server stack:

- SGLang first. Local notes and upstream support suggest Qwen3.6 support is
  better there than in our current vLLM path.
- Current local `.venv-sglang` is `sglang==0.5.9`; Qwen3.6 serving should use
  `sglang>=0.5.10` before serious 27B work.

Teacher selection policy:

Use **Qwen3.6-27B** as the teacher/reference for the eval/data loop. The teacher
should be served in whichever Qwen3.6-27B precision/profile gives the best
quality-throughput-memory tradeoff on the RTX 5090:

- FP8 first if it fits with reduced context and acceptable cache headroom.
- NVFP4/Q4 fallback is acceptable and likely practical for local 5090 serving.
- MTP/speculative decoding should be enabled once validated for this model/server
  path.
- Fast attention/GEMM backends should be used when stable on Blackwell.
- GX10/GB10 is the backup for capacity-heavy checks, but the preferred teacher
  loop should run on the 5090 if quality and speed are acceptable.

Teacher profile priority:

1. `Qwen/Qwen3.6-27B-FP8` on the RTX 5090 with reduced context.
2. Qwen3.6-27B NVFP4 / Q4 variant if FP8 is too tight.
3. Same Qwen3.6-27B profile with MTP/speculative and fast attention enabled.
4. GX10/GB10 for capacity-heavy 27B checks if local 5090 serving is unstable.

Speed knobs to expose:

- MTP/speculative decoding when supported by the model/server path.
- `--attention-backend fa3` or another proven fast backend on Blackwell.
- `--fp8-gemm-backend auto` or a backend validated on 5090.
- `--fp4-gemm-backend auto` / NVFP4 backend for Q4 fallback.
- reduced `--context-length` first, then increase only after stable load.
- constrained JSON/tool-call parser options for agentic evals.
- small `max-running-requests` and conservative memory fraction until stable.

Exit gate:

- SGLang serves Qwen3.6-27B-class teacher locally or on GX10.
- A simple OpenAI-compatible chat request succeeds.
- Tool-call formatting works with the Qwen parser/template.
- Throughput and VRAM/memory use are recorded.

## Phase 2: Agentic Eval and Data Loop on Qwen3.5/3.6

Goal: build eval/data plumbing around the actual target family, not only the
Qwen2.5/Fast-dLLM lab model.

Primary model set:

- SGLang-served Qwen3.6-27B AR teacher/reference.
- Qwen3.5-9B as first real GDN diffusion target.
- Qwen3.5-4B only as an architecture/debug smoke target.
- Local Fast-dLLM/Qwen2.5-1.5B remains a cheap sampler/objective lab, not the
  main target.

Eval set:

1. **Strict JSON/tool-call formatting**
   - model must emit exactly one JSON object
   - schema validation
   - no prose before/after
   - nested arguments and string escaping

2. **Function-choice tests**
   - choose correct tool from 3-8 tools
   - include only required args
   - avoid hallucinated args

3. **Multi-step tool traces**
   - two or three sequential calls
   - previous observation must affect next call
   - detect loops/repeated calls

4. **Code generation**
   - HumanEval/MBPP or small local tests
   - exact runnable code, not just explanation

5. **Patch generation**
   - small repo edit tasks
   - diff applies cleanly
   - tests pass

6. **Existing generic checks**
   - GSM8K limited
   - IFEval limited
   - unresolved mask count
   - repetition/truncation rate
   - tokens/s

Exit gate:

- Qwen3.6 teacher/reference passes the local tool-call eval and can label/repair
  public examples.
- Qwen3.5-9B AR baseline is measured on the same eval.
- Failure modes are categorized enough to train against.

## Phase 3: Agentic/Code Data Instead of Alpaca

Goal: train on the behavior we actually need.

Public candidate data:

- Hermes function-calling v1: open, schema/tool-call examples.
- Glaive function-calling v2: open, mixed tool/no-tool examples.
- ToolACE: open, multi-turn tool-use traces.
- ToolBench / ToolLLM: larger real-API tool-use trajectories.
- xLAM function-calling 60K: useful, but gated in the current HF environment;
  use only when authenticated access is available.
- BFCL: evaluation gate, not a training set.

Generated/teacher data:

- Qwen3.6 teacher rewrites public examples into Qwen tool-call chat format.
- Qwen3.6 teacher repairs invalid JSON/tool calls.
- Qwen3.6 teacher generates “think / tool / observation / final” traces where
  allowed by the target format.
- Hard negatives from failed local eval cases.

Data requirements:

- JSON-schema constrained outputs
- function-call / tool-call conversations
- coding instruction data tied to tools
- repo-edit traces
- patch generation examples
- some general instruction data to prevent narrow collapse

Do not rely on Alpaca as the main corpus. Alpaca is useful only as a plumbing
smoke test.

## Phase 4: Better Objective

Fast-dLLM’s core recipe is AR initialization plus masked-token CE on ground-truth
tokens. For agentic behavior, we should test adding explicit AR-teacher
distillation:

- masked-token cross-entropy against gold tokens
- KL/logit distillation from frozen AR Qwen teacher on masked positions
- extra weighting for structural tokens:
  - `{`, `}`, `[`, `]`, `:`, `,`
  - quote tokens
  - tool/function names
  - stop and boundary tokens
- optional sequence-level checks for JSON validity and call format

Hypothesis:

Agentic tasks need symbolic precision and causal ordering. Ground-truth CE alone
may not preserve enough AR behavior after block diffusion conversion.

## Phase 5: Block-Size Curriculum

Goal: avoid jumping straight into large-block denoising that smears action order.

Initial curriculum:

- block size 1 or 4: near-AR behavior
- 8 / 16: early parallelism
- 32: current Fast-dLLM default
- 64 / 128 only after structured-output metrics are stable

Track metrics per block size:

- task score
- invalid JSON rate
- repeated action rate
- unresolved mask rate
- denoising steps
- tokens/s

## Phase 6: GDN Qwen3.5 / Qwen3.6 Target Sequence

The main target loop should move to Qwen3.5/3.6 as soon as the eval/data plumbing
exists. The 1.5B path remains useful for cheap sampler debugging, but it should
not dominate the roadmap.

Target sequence:

1. Qwen3.6-27B AR teacher/reference via SGLang.
2. Qwen3.5-9B AR baseline and first real GDN LoRA/QLoRA target.
3. Qwen3.5-4B only when a cheap GDN architecture smoke test is needed.
4. Qwen3.6-27B diffusion LoRA/selective adapter after the 9B loop proves out.

Why 9B before 4B:

- 4B is too weak to be a serious teacher or quality target.
- 9B is still plausibly trainable with LoRA/QLoRA on the available hardware.
- 9B exercises the same GDN/full-attention hybrid family as the 27B target.
- 9B quality is more likely to make agentic eval trends meaningful.

GDN starting strategy:

- Option A first:
  - keep GDN causal
  - use it as cross-block state carrier
  - bidirectionality comes from full-attention layers inside block
  - snapshot GDN state at block boundaries
- Option B only if needed:
  - add backward within-block GDN scan
  - more expensive and more implementation risk

## Phase 7: Serving and Quantization

Training format:

- keep first training runs simple: bf16 base plus LoRA/QLoRA where needed
- do not make NVFP4 training the first systems problem

Serving/export:

- use SGLang FP8 as reference/quality format when memory allows
- use NVFP4/Q4 as the practical 5090 deployment fallback
- expose generation knobs:
  - block size
  - denoising steps
  - threshold or entropy bound
  - temperature
  - top-p
  - cache on/off

## Immediate Next Goal

Build the Qwen3.6 teacher serving path and public-data agentic eval/data harness.

Minimum first version:

- SGLang launch script for Qwen3.6-27B FP8 teacher.
- NVFP4/Q4 fallback launch path.
- speed knobs exposed: MTP/speculative options, attention backend, GEMM backend,
  reduced context, memory fraction, tool parser.
- data prep script for Hermes/Glaive/ToolACE, with xLAM optional when HF access
  is available.
- 20 strict JSON/tool-call prompts
- 20 function-choice prompts
- 10 two-step tool traces
- 10 small code-generation tasks
- model set: Qwen3.6 teacher, Qwen3.5-9B AR baseline, local diffusion baselines
- summary with:
  - schema pass rate
  - correct tool rate
  - repeated-call rate
  - stop-boundary failures
  - tokens/s

Then use the failure cases to define the next training corpus and objective.

## Open Questions

- Should the first real agentic training use only LoRA, or LoRA plus selected
  structural-token embedding/head updates?
- Is AR-teacher KL enough, or do we need sequence-level constrained decoding loss?
- Should block size stay small for tool-call spans and grow only for natural text?
- Can the sampler force structural tokens more safely without ruining speed?
- Is the released Fast-dLLM 1.5B good enough on tool calls to serve as a local
  diffusion baseline, or do we need to compare with DiffusionGemma directly?
- Does SGLang Qwen3.6 FP8 fit comfortably enough on the 5090, or should NVFP4 be
  the default local teacher path?
