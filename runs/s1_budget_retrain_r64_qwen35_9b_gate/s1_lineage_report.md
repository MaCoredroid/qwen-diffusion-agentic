# S1 Lineage Report

## Question

Did the Run-1 copy-grounded checkpoint initialize from the B@1000 two-stream adapter while S1 initialized from raw LoRA init?

## Finding

No. For the checked launch scripts and saved artifacts, Run-1 and S1 both use the same adapter-init lineage: a new LoRA/QLoRA adapter on top of `/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-init`.

The B@1000 adapter is a separate legacy foundation/eval adapter at `runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000`. It was not passed as `LORA_MODEL_PATH` by `scripts/run_flare_redesign_run1.sh` or `scripts/run_s1_budget_retrain.sh`.

## Adapter-Init Mechanism

The shared lower wrapper is `scripts/run_fastdllm_qwen35_9b_agentic_qlora_pilot.sh`.

- Line 31: `LORA_MODEL_PATH="${LORA_MODEL_PATH:-}"`.
- Lines 197-198: `--lora_model_path` is appended only when `LORA_MODEL_PATH` is non-empty.
- `fast-dllm/third_party/lmflow/models/hf_model_mixin.py` lines 371-382: if `model_args.lora_model_path is not None`, LMFlow loads `PeftModel.from_pretrained(..., is_trainable=True)`; otherwise it creates a new adapter with `get_peft_model(model, self.peft_config)`.

Therefore adapter continuation requires an explicit non-empty `LORA_MODEL_PATH`.

## Run-1 Checked Config

Script: `scripts/run_flare_redesign_run1.sh`

- Lines 7-8: `OUTPUT_DIR=/home/mark/qwen_diffusion/runs/flare_redesign_run1_copy_grounded_qwen35_9b`, `MODEL_PATH=/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-init`.
- Lines 44-72: execs the shared lower wrapper without setting `LORA_MODEL_PATH`.
- Training defaults in that wrapper invocation:
  - `MAX_STEPS=400`
  - `MAX_TRAIN_SAMPLES=5055`
  - `GRAD_ACCUM=1`
  - `LEARNING_RATE=1e-5`
  - scheduler inherited as lower-wrapper default `cosine`
  - `LORA_R=16`
  - `LORA_ALPHA=32`
  - `LORA_DROPOUT=0.05`
  - `LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj`
  - `FASTDLLM_GDN_KERNEL=fla`

Saved artifact: `runs/flare_redesign_run1_copy_grounded_qwen35_9b/adapter_config.json`

- `base_model_name_or_path=/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-init`
- `r=16`
- `lora_alpha=32`
- `lora_dropout=0.05`
- target modules include q/k/v/o plus GDN modules.

Saved train result: `runs/flare_redesign_run1_copy_grounded_qwen35_9b/train_results.json`

- `train_runtime=2068.1572`
- `train_loss=4.390680780410767`
- `train_steps_per_second=0.193`
- `epoch=0.11179429849077697`

Adapter-init verdict: new LoRA from raw base, not B@1000 continuation.

## S1 Checked Config

Script: `scripts/run_s1_budget_retrain.sh`

- Lines 7-8: `OUTPUT_DIR=/home/mark/qwen_diffusion/runs/s1_budget_retrain_r64_qwen35_9b`, `MODEL_PATH=/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-init`.
- Lines 49-80: execs the shared lower wrapper without setting `LORA_MODEL_PATH`.
- Training defaults in that wrapper invocation:
  - `MAX_STEPS=2000`
  - `MAX_TRAIN_SAMPLES=5055`
  - `GRAD_ACCUM=2`
  - `LEARNING_RATE=1e-5`
  - `LR_SCHEDULER_TYPE=warmup_stable_decay`
  - `LR_SCHEDULER_KWARGS={"num_stable_steps":1600,"num_decay_steps":300,"min_lr_ratio":0.1}`
  - `WARMUP_STEPS=100`
  - `LORA_R=64`
  - `LORA_ALPHA=128`
  - `LORA_DROPOUT=0.05`
  - `LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj`
  - `FASTDLLM_GDN_KERNEL=torch`

Saved artifact: `runs/s1_budget_retrain_r64_qwen35_9b/adapter_config.json`

- `base_model_name_or_path=/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-init`
- `r=64`
- `lora_alpha=128`
- `lora_dropout=0.05`
- target modules include q/k/v/o plus GDN modules.

Saved train result: `runs/s1_budget_retrain_r64_qwen35_9b/train_results.json`

- `train_runtime=11789.9276`
- `train_loss=4.409758522033691`
- `train_steps_per_second=0.17`
- `epoch=1.1179429849077698`

Adapter-init verdict: new LoRA from raw base, not B@1000 continuation.

## B@1000 Reference

Path: `runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000`

Saved artifact: `runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000/adapter_config.json`

- `base_model_name_or_path=/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-init`
- `r=8`
- `lora_alpha=16`
- `lora_dropout=0.05`
- `target_modules=q_proj,v_proj,k_proj,o_proj`

Saved train result: `runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000/train_results.json`

- `train_runtime=18922.0906`
- `train_loss=3.9429622707366945`
- `train_steps_per_second=0.053`
- `epoch=17.24137931034483`

This is not the parent of Run-1 or S1 under the checked scripts. The older agentic-v1 continuation path is `scripts/run_flare_agentic_v1_twostream_job.sh`, which does set `LORA_MODEL_PATH=/home/mark/qwen_diffusion/runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000`; that is a different run family from `runs/flare_redesign_run1_copy_grounded_qwen35_9b`.

## Decision

Lineages are the same with respect to adapter-init. Run-1 and S1 differ in rank, scheduler, step budget, gradient accumulation, and GDN kernel, but not in B@1000-vs-raw initialization.

Therefore S1 did not accidentally test "raw from scratch" while Run-1 benefited from B@1000. The condition for launching S1b is false. The r64/WSD/2000 S1 gate result stands as a budget-scaling failure under the raw-init native lineage, and the next decision is r128 escalation versus kill per the existing recipe.
