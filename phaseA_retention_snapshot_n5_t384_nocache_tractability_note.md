# Phase A Behavior-Retention Snapshot: N=5/T384 No-Cache Tractability Gate

Date: 2026-06-28

## Scope

This note records the tractability gate after reverting to the correctness path for
Qwen3.5 diffusion generation.

- Diffusion sampler: `use_block_cache=False`
- Token cap: `max_new_tokens=384` for diffusion, `max_gen_toks=384` for AR
- Deterministic slice: first 5 docs from each Phase A task
- No retraining
- No promotion decision

Fast-DLLM block cache was not used for the diffusion probe. The cached sampler is
invalid for this Qwen3.5 GDN bridge until a KV-cache-aware bridge is implemented
and byte-identical cache-vs-no-cache generation is proven.

## AR Denominator

Command family:

```bash
/home/mark/qwen_diffusion/.venv/bin/lm_eval \
  --model hf \
  --model_args pretrained=Qwen/Qwen3.5-9B,trust_remote_code=True,dtype=bfloat16 \
  --tasks phaseA_mbpp_first20,phaseA_gsm8k_first20,phaseA_ifeval_first20 \
  --include_path /home/mark/qwen_diffusion/tasks/phaseA_retention \
  --limit 5 \
  --batch_size 1 \
  --device cuda:0 \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --gen_kwargs do_sample=False,temperature=0.0,max_gen_toks=384 \
  --confirm_run_unsafe_code \
  --log_samples \
  --output_path /home/mark/qwen_diffusion/runs/phaseA_retention_snapshot_n5_t384/ar_qwen35_9b_bf16_thinkingoff
```

Results:

| Task | AR metric | Score |
| --- | --- | ---: |
| MBPP | `pass_at_1` | 0.20 |
| GSM8K | strict exact match | 1.00 |
| GSM8K | flexible exact match | 1.00 |
| IFEval | prompt strict | 1.00 |
| IFEval | instruction strict | 1.00 |

The checked GSM8K sample emitted a final `####` answer within the 384-token cap.

## No-Cache Diffusion Probe

Probe command:

```bash
/home/mark/qwen_diffusion/.venv-lmeval/bin/python /home/mark/qwen_diffusion/fast-dllm/v2/eval.py \
  --model fast_dllm_v2 \
  --model_args model_path=/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-init,tokenizer_path=/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-init,max_new_tokens=384,show_speed=True,threshold=0.9,bd_size=32,small_block_size=8,use_block_cache=False,temperature=0.0 \
  --tasks phaseA_mbpp_first20 \
  --include_path /home/mark/qwen_diffusion/tasks/phaseA_retention \
  --limit 1 \
  --batch_size 1 \
  --device cuda:0 \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --confirm_run_unsafe_code \
  --log_samples \
  --output_path /home/mark/qwen_diffusion/runs/phaseA_retention_snapshot_n5_t384/smoke_diff_init_mbpp1_nocache
```

Outcome:

- Model loaded successfully.
- Chat template path used `enable_thinking=False`.
- Generation entered `Generating...`.
- The first MBPP sample did not complete before the 240s wall-time cap.
- The command exited via timeout with code `124`.
- No diffusion output sample/result file was written.
- GPU was idle after cleanup; no Fast-DLLM process remained.

## Decision

The six-cell no-cache diffusion matrix was not launched. At more than 240 seconds
without one completed MBPP sample, the requested N=5/T384 no-cache matrix
(`DIFF_INIT + checkpoint-275` x `MBPP + GSM8K + IFEval`) cannot meet the target
runtime of roughly 90 minutes on the local RTX 5090 under the current sampler.

No retention ratios or catastrophic-forgetting verdict are reported from
diffusion generation in this note, because the required correctness-path
diffusion outputs were not produced.
