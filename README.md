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
- Bring up the first text-only Qwen3.5-9B Fast-DLLM diffusion candidate and use
  QLoRA pilots to find a viable agentic curriculum/windowing recipe.

## Hardware Context

- Local RTX 5090, about 32 GB VRAM: primary iteration/eval/training machine.
- Remote RTX 5080, about 16 GB VRAM: smaller eval and preprocessing worker.
- Remote GB10-class machine, about 117 GB unified memory visible to Linux:
  memory-heavy loading, correctness, export, and long-context experiments.

## Important Docs

- [agentic_diffusion_qwen_plan.md](agentic_diffusion_qwen_plan.md): end-to-end
  strategy for agentic diffusion Qwen.
- [qwen36_diffusion_closeout_metrics.md](qwen36_diffusion_closeout_metrics.md):
  measurable closeout targets for Qwen3.6 diffusion, including SWE-bench Verified
  baselines and success gates.
- [diffusion_qwen_distillation_runbook.md](diffusion_qwen_distillation_runbook.md):
  original Qwen3.6 block-diffusion distillation runbook.
- [experiment_plan.md](experiment_plan.md): hardware and model feasibility plan.
- [diffusiongemma_agentic_research_notes.md](diffusiongemma_agentic_research_notes.md):
  DiffusionGemma and agentic dLLM research notes.
- [early_qwen_diffusion_toolcall_result.md](early_qwen_diffusion_toolcall_result.md):
  first public-data tool-call LoRA train/eval result for the local 1.5B
  diffusion lab model.
- [synthetic_onecall_curriculum_result.md](synthetic_onecall_curriculum_result.md):
  synthetic one-call curriculum result showing tool-name learning and the
  current structural decoding gap.
- [qwen25_1p5b_diffusion_baseline_result.md](qwen25_1p5b_diffusion_baseline_result.md):
  strict local 1.5B Fast-dLLM baseline across the shared tool-call slices.
- [qwen36_teacher_serving_result.md](qwen36_teacher_serving_result.md): local
  SGLang NVFP4 Qwen3.6 teacher serving result and first 48-case synthetic
  tool-call teacher probe.
- [qwen36_teacher_toolcall_arg_eval_result.md](qwen36_teacher_toolcall_arg_eval_result.md):
  argument-level synthetic and public one-call teacher eval result.
- [qwen36_teacher_mtp_cuda_5090_result.md](qwen36_teacher_mtp_cuda_5090_result.md):
  RTX 5090 MTP serving profile, CUDA graph fit attempts, and current blocker.
- [qwen36_teacher_multicall_eval_result.md](qwen36_teacher_multicall_eval_result.md):
  public Hermes multi-call teacher baseline and extra/missing/repeated-call
  metrics.
- [qwen36_teacher_toolresult_eval_result.md](qwen36_teacher_toolresult_eval_result.md):
  synthetic two-step tool-result baseline for next-action selection.
- [qwen36_teacher_openai_toolresult_eval_result.md](qwen36_teacher_openai_toolresult_eval_result.md):
  stricter OpenAI `assistant.tool_calls` plus `role=tool` tool-result baseline.
- [qwen35_9b_ar_baseline_result.md](qwen35_9b_ar_baseline_result.md):
  Qwen3.5-9B 4-bit AR baseline on Alienware RTX 5080 across the local
  one-call, multi-call, and tool-result slices.
- [qwen35_9b_diffusion_pilot_readiness.md](qwen35_9b_diffusion_pilot_readiness.md):
  guarded readiness and smoke-training result for the Qwen3.5-9B
  diffusion/QLoRA loop, including bridge status, weight materialization, and
  current curriculum/windowing issue.
- [qwen35_gdn_vs_qwen25_research.md](qwen35_gdn_vs_qwen25_research.md):
  primary-source check showing why Qwen3.5/3.6 Gated DeltaNet models cannot be
  treated as Qwen2.5-style full-attention conversion targets.
- [qwen35_gdn_lora_ablation_gate_result.md](qwen35_gdn_lora_ablation_gate_result.md):
  first GDN-specific LoRA target-family gate for Qwen3.5-9B, covering one-step
  and 25-step GDN-only, attention-only, mixed adapters, plus the first
  noisy-block, clean-state, and clean-state plus mild structural objective
  probes, the first local dual-pass GDN probe, and the first value-copy and
  aligned value-span objective hooks, plus argument-span mask-forcing gates from
  the base candidate and active checkpoint, plus the negative checkpoint-275
  hard clean-repair/full-span continuation probe.
- [qwen35_9b_diffusion_ckpt275_agentic_scorecard.md](qwen35_9b_diffusion_ckpt275_agentic_scorecard.md):
  active Qwen3.5-9B diffusion checkpoint scorecard across one-call,
  multi-call, and tool-result gates.
- [qwen35_9b_multicall_scalar_curriculum_result.md](qwen35_9b_multicall_scalar_curriculum_result.md):
  targeted public multi-call scalar extraction curriculum and one-step
  Qwen3.5-9B QLoRA gate result.
- [qwen35_9b_multicall_scalar_adapter_result.md](qwen35_9b_multicall_scalar_adapter_result.md):
  scalar repair adapter result, including the 300-step plateau check.
- [qwen35_9b_contextual_projection_suite_result.md](qwen35_9b_contextual_projection_suite_result.md):
  cross-slice contextual projection check covering scalar grounding and the
  complex-context constrained decoder update, the guarded sequence-planner
  projection, and the grounded one-call projection that lifts active public
  one-call constrained exact arguments to `8/8`.
- [qwen35_9b_grounded_spanfill_curriculum_result.md](qwen35_9b_grounded_spanfill_curriculum_result.md):
  first attempt to turn grounded one-call projection into a trainable
  span-fill curriculum; block-1024 fits on the RTX 5090, but the one-step
  adapter is not promoted because raw model-only behavior does not improve.
- [qwen35_9b_modelrepair_scalar_mix_result.md](qwen35_9b_modelrepair_scalar_mix_result.md):
  negative lower-weight scalar-mix main-generator result; do not promote.
- [qwen35_9b_multicall_gap_curriculum_result.md](qwen35_9b_multicall_gap_curriculum_result.md):
  missing-call and complex-payload curriculum build for the next staged
  multi-call repair/extraction lane, including the complex-context decoder
  promotion and negative adapter result.
- [qwen35_9b_sequence_planner_distill_curriculum_result.md](qwen35_9b_sequence_planner_distill_curriculum_result.md):
  train-only sequence-planner distillation curriculum for Qwen3.5-9B,
  including the GDN recheck, 896-token label-retention audit, compact-schema
  row-recovery probe, and one-step QLoRA gates from checkpoint-275.
- [qwen35_public_eval_overlap_audit_result.md](qwen35_public_eval_overlap_audit_result.md):
  public-train overlap audit showing `11/12` public multi-call smoke rows were
  present verbatim in `fastdllm_toolcall_train`, plus the filtered source and
  clean planner-curriculum handoff.
- [qwen35_9b_modelrepair_sequence_planner_mix_result.md](qwen35_9b_modelrepair_sequence_planner_mix_result.md):
  negative 100-step low-ratio sequence-planner replay mix from checkpoint-275;
  do not promote.
- [qwen_code_official_harness_result.md](qwen_code_official_harness_result.md):
  Qwen Code official coding-agent harness smoke against the local Qwen3.6
  SGLang teacher.
- [qwen_code_repo_edit_eval_result.md](qwen_code_repo_edit_eval_result.md):
  5-task Qwen Code tiny repo-edit baseline against the local Qwen3.6 teacher.
- [qwen36_teacher_codegen_eval_result.md](qwen36_teacher_codegen_eval_result.md):
  10-task Python code-generation smoke baseline for Qwen3.6 teacher.
- [machine_notes.md](machine_notes.md): sanitized hardware notes.

## Scripts

- `scripts/serve_sglang_qwen36_teacher.sh`: launches a Qwen3.6-27B
  teacher/reference through SGLang with FP8 and NVFP4/Q4 profiles plus exposed
  attention, GEMM, memory, and speculative/MTP knobs.
- `scripts/prepare_toolcall_seed_data.py`: builds normalized seed JSONL from
  public tool-call datasets.
- `scripts/eval_toolcall_jsonl.py`: scores normalized JSONL for assistant
  turns, strict JSON tool calls, and known-tool matches.
- `scripts/build_fastdllm_toolcall_data.py`: converts normalized public
  tool-call JSONL into Fast-dLLM conversation train/eval files.
- `scripts/build_synthetic_onecall_curriculum.py`: creates deterministic
  synthetic single-tool-call train/eval data with distractor tools.
- `scripts/build_synthetic_toolresult_traces.py`: creates deterministic
  two-step tool-result traces for next-action eval and curriculum data,
  including text-compatible and OpenAI-native `role=tool` variants.
- `scripts/build_toolcall_sequence_planner_distill_curriculum.py`: builds a
  train-only public multi-call sequence-planner curriculum for Qwen3.5-9B,
  using deterministic request/schema planning as a selector and train gold
  calls as the teacher-forced target. Supports compact schema and prompt-mode
  variants for label-retention probes.
- `scripts/build_toolcall_modelrepair_sequence_planner_mix.py`: builds a
  low-ratio replay mix from the active model-repair curriculum and the
  sequence-planner distillation rows.
- `scripts/teacher_distill_toolcall_cases.py`: probes or records
  OpenAI-compatible Qwen3.6 teacher outputs for tool-call cases.
- `scripts/eval_openai_toolcall_cases.py`: OpenAI-compatible strict native
  `message.tool_calls` evaluator with optional text fallback scoring.
- `scripts/eval_transformers_toolcall_cases.py`: direct Transformers baseline
  runner for Qwen-family AR models, with 4-bit bitsandbytes and multi-slice
  suite support.
- `scripts/qwen_code_sglang_proxy.py`: local OpenAI-compatible bridge that
  makes Qwen Code work against the SGLang Qwen3.6 teacher profile.
- `scripts/run_qwen_code_sglang_smoke.sh`: headless Qwen Code smoke against the
  local teacher.
- `scripts/build_tiny_repo_edit_tasks.py`: creates five deterministic tiny
  Python repo-edit tasks with initially failing unit tests.
- `scripts/eval_qwen_code_repo_edit_cases.py`: runs Qwen Code on repo-edit
  cases through the SGLang proxy and independently scores final test pass plus
  changed-file metrics.
- `scripts/build_synthetic_codegen_tasks.py`: creates 10 deterministic Python
  function-generation tasks.
- `scripts/eval_openai_codegen_cases.py`: runs codegen tasks against an
  OpenAI-compatible endpoint and verifies generated functions with unit tests.
- `scripts/run_fastdllm_qwen25_1p5b_toolcall_lora_smoke.sh`: bounded LoRA
  training run for the local Qwen2.5 1.5B Fast-dLLM lab model on tool-call data.
- `scripts/eval_fastdllm_toolcall_smoke.py`: direct diffusion sampler eval for
  strict `<tool_call>` JSON, loose function-name mention metrics, and optional
  constrained tool-name repair.
- `scripts/eval_fastdllm_toolcall_cases.py`: direct Fast-dLLM sampler eval
  using the same exact-sequence, exact-argument, schema, extra/missing/repeated,
  unresolved-mask, and tokens/sec metrics as the AR tool-call baselines, with
  constrained scalar and complex payload reconstruction.
- `scripts/build_agentic_diffusion_curriculum.py`: combines public tool-call,
  synthetic one-call, synthetic tool-result, and successful Qwen Code repo-edit
  examples into the first Qwen3.5-9B agentic diffusion pilot corpus.
- `scripts/build_toolcall_modelrepair_scalar_mix.py`: combines the current
  model-repair generator curriculum with a capped, balanced sample of
  multi-call scalar extraction rows for staged main-generator tests.
- `scripts/build_toolcall_grounded_spanfill_curriculum.py`: builds train-slice
  grounded span-fill examples from active raw drafts and grounded projection
  outputs, preserving full labels at audited block sizes.
- `scripts/build_toolcall_multicall_gap_curriculum.py`: builds public
  multi-call missing-call recovery and complex-payload extraction rows with
  label-aware 896-token retention checks.
- `scripts/build_toolcall_multicall_gap_eval_cases.py`: builds held-out
  missing-call and complex-payload eval prompts from the public multi-call
  slice.
- `scripts/rescore_scalar_repair_contextual_projection.py`: deterministic
  request-evidence scalar projection over constrained drafts or scalar-repair
  outputs, including call-local ID/datetime fixes and conservative missing
  required scalar fills.
- `scripts/rescore_toolcall_sequence_planner_projection.py`: guarded
  request-evidence sequence planner for multi-call outputs, using list/table
  structure and tool schema text to diagnose missing-call/order failures.
- `scripts/init_qwen35_fastdllm_candidate.py`: creates the local Qwen3.5-9B
  Fast-DLLM text-only candidate scaffold with `bd_size`, compatible `auto_map`,
  and a real single-token `|<MASK>|`.
- `scripts/check_qwen35_diffusion_readiness.py`: preflights Qwen3.5-9B
  diffusion readiness, including training env support, mask-token conversion,
  candidate config, GDN bridge presence, cached weights, and dataset inputs.
- `scripts/materialize_qwen35_fastdllm_weights.py`: plans and writes remapped
  text-only candidate safetensor shards from the raw Qwen3.5 checkpoint once the
  four raw shards are downloaded.
- `scripts/run_fastdllm_qwen35_9b_agentic_qlora_pilot.sh`: guarded QLoRA
  launcher for the first Qwen3.5-9B diffusion pilot; exits before training until
  the converted candidate and bridge are present.
- `scripts/run_fastdllm_toolcall_checkpoint_sweep.sh`: checkpoint promotion
  suite for one-call, public multi-call, tool-result, sequence-preserving
  projection, contextual public multi-call projection, and guarded sequence
  planner projection.
- `scripts/write_qwen35_diffusion_checkpoint_scorecard.py`: regenerates the
  active Qwen3.5-9B checkpoint scorecard from regular eval and contextual
  projection summaries.
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

The first Alpaca LoRA run was a plumbing proof. The first public-data tool-call
run did not produce held-out tool calls. The synthetic one-call curriculum does
teach the 1.5B lab model to name the right tool in most held-out examples, but
the model still needs constrained decoding/repair to emit runnable tool-call
structure. The Qwen3.6 NVFP4 teacher now serves locally through SGLang with MTP
at 4k context on the RTX 5090. It gets 48/48 exact tool-name and argument match
on the synthetic one-call held-out probe. On a 24-case public Hermes one-call
slice it gets 21/24 exact tool sequence and 18/24 exact arguments. On a 12-case
public Hermes multi-call slice it gets 12/12 valid tool-call emissions, 11/12
exact tool sequence, 10/12 exact arguments, and no repeated-call loops. The next
synthetic tool-result slice gets 10/10 exact next-tool sequence and arguments
with no repeated-call loops. The stricter OpenAI `role=tool` variant shows that
Qwen3.6 gets 10/10 exact via Qwen text fallback, while strict native
`tool_choice=required` gets 10/10 exact sequence and 8/10 exact arguments.
The Qwen Code tiny repo-edit gate now runs against an 8k-context local
Qwen3.6 NVFP4/MTP teacher and gets 5/5 independent final test pass with only
expected source files changed; the current forced-tool workaround makes Qwen
Code exit on its tool budget after success, so patch/test pass is the primary
metric.
Qwen3.5-9B AR now has a 4-bit RTX 5080 baseline:
48/48 synthetic one-call, 17/24 public one-call exact sequence, 11/12 public
multi-call exact sequence, and 10/10 synthetic tool-result. The local 1.5B
Fast-dLLM diffusion lab baseline is now measured on the same tool-call slices:
base init and public-data LoRA score 0 strict hits; synthetic one-call LoRA gets
17/48 exact sequence and 10/48 exact arguments on synthetic one-call, but still
0 on public one-call, public multi-call, and tool-result continuation. The
Qwen3.5-9B diffusion/QLoRA loop is now past the first infrastructure gate: the
local text-only candidate has an implemented v0 Qwen3.5/GDN bridge, remapped
candidate weights, exact 427/427 key compatibility, and a passing readiness
gate. A local RTX 5090 QLoRA loss-smoke produced nonzero loss/gradients and
saved adapters; after adding real gradient-checkpointing in the bridge, a
512-token agentic pilot fits on the 5090. The first useful agentic learning
signal is the 5-step `DISABLE_GROUP_TEXTS=1`, `TRUNCATION_SIDE=left` pilot:
train loss 5.712 and final grad norm 19.994. The strict 2-case synthetic
one-call eval now runs for both diffusion init and the 5-step adapter, with no
sampler errors or unresolved masks, but both still score 0/2 strict tool-call
hits. A 100-step mixed agentic pilot also runs and evaluates cleanly, but still
scores 0 strict hits on small synthetic one-call, public one-call, public
multi-call, and synthetic tool-result slices. A focused synthetic-only 704-token
probe exposed and fixed the LMFlow `None`-field schema widening issue that was
removing assistant labels after truncation. The current active 9B diffusion
comparison point is the model-repair plus argument-span-1.5 checkpoint-275, with
deterministic contextual projection reaching `7/12` public multi-call exact
sequence and `7/12` exact arguments. The latest CPU-side data build adds a
181-row missing-call/complex-payload gap curriculum for the next staged repair
or generation-time constrained decoding experiment.
