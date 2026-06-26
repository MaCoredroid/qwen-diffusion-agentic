# Qwen Diffusion Agentic Lab

Personal research project for exploring whether Qwen-family autoregressive models
can be converted into block-diffusion language models that remain useful for
agentic coding and tool workflows.

This repo is intentionally a lab notebook plus reproducible scripts, not a model
release. Large assets such as virtual environments, model weights, LoRA adapters,
datasets, logs, and generated eval outputs are kept out of git.

## Current Focus

- Serve a Qwen3.6-27B teacher/reference through SGLang, using FP8 first and
  NVFP4/Q4 when that is the practical 5090 path.
- Build a public tool-call data/eval loop around Hermes, Glaive, ToolACE, and
  optional gated xLAM.
- Build agentic/tool-call evals that catch JSON/schema, tool-choice, loop, stop,
  and code-edit failures.
- Keep the small Fast-dLLM/Qwen2.5 1.5B model as a cheap sampler/objective lab,
  while moving the real target loop toward Qwen3.5/3.6-family models.

## Hardware Context

- Local RTX 5090, about 32 GB VRAM: primary iteration/eval/training machine.
- Remote RTX 5080, about 16 GB VRAM: smaller eval and preprocessing worker.
- Remote GB10-class machine, about 117 GB unified memory visible to Linux:
  memory-heavy loading, correctness, export, and long-context experiments.

## Important Docs

- [agentic_diffusion_qwen_plan.md](agentic_diffusion_qwen_plan.md): end-to-end
  strategy for agentic diffusion Qwen.
- [diffusion_qwen_distillation_runbook.md](diffusion_qwen_distillation_runbook.md):
  original Qwen3.6 block-diffusion distillation runbook.
- [experiment_plan.md](experiment_plan.md): hardware and model feasibility plan.
- [diffusiongemma_agentic_research_notes.md](diffusiongemma_agentic_research_notes.md):
  DiffusionGemma and agentic dLLM research notes.
- [machine_notes.md](machine_notes.md): sanitized hardware notes.

## Scripts

- `scripts/serve_sglang_qwen36_teacher.sh`: launches a Qwen3.6-27B
  teacher/reference through SGLang with FP8 and NVFP4/Q4 profiles plus exposed
  attention, GEMM, memory, and speculative/MTP knobs.
- `scripts/prepare_toolcall_seed_data.py`: builds normalized seed JSONL from
  public tool-call datasets.
- `scripts/eval_toolcall_jsonl.py`: scores normalized JSONL for assistant
  turns, strict JSON tool calls, and known-tool matches.
- `scripts/run_fastdllm_checkpoint_sweep.py`: limited `lm-eval` checkpoint sweep
  over local base, LoRA checkpoints, and released Fast-dLLM reference.
- `scripts/eval_fastdllm_lora_gsm8k_mini.py`: small direct GSM8K smoke eval.
- `scripts/start_training_service.sh`: runs training in a contained systemd user
  service.
- `scripts/protect_interactive_processes.sh`: protects interactive tmux/Codex
  processes from training OOM pressure.

## Upstream Dependency

The main external scaffold is NVIDIA Fast-dLLM:

https://github.com/NVlabs/Fast-dLLM

This repo does not vendor the cloned upstream tree. Project-specific changes are
stored under `patches/`.

## Status

The first Alpaca LoRA run was a plumbing proof. It showed that the local
Fast-dLLM/Qwen2.5 1.5B hybrid can train and decode, but quality remains far behind
the released Fast-dLLM v2 1.5B checkpoint. The next useful milestone is an
agentic eval harness, not another generic instruction-tuning run.
