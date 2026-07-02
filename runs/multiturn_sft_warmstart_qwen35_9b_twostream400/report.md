# Multi-Turn SFT Warm-Start Two-Stream Retry

## Cheap Diagnosis

Objective:
- The failed `runs/multiturn_sft_warmstart_qwen35_9b_sft400` run was not plain AR-only LMFlow SFT: it used `fast-dllm/v2/train_scripts/finetune.py` with `--mdm 1`.
- It was still not the exact Run-1 FLARE path. It called `scripts/run_fastdllm_qwen35_9b_agentic_qlora_pilot.sh` directly, without the `scripts/run_flare_redesign_run1.sh` wrapper.
- Missing vs Run-1: `FASTDLLM_FLARE_TWO_STREAM=1`, FLARE GDN route/envs, r=16/full LoRA targets, lr `1e-5`, Run-1 seed, block size `512`, and value-span loss/mask settings.
- Failed SFT used r=8, targets `q_proj,k_proj,v_proj,o_proj`, lr `5e-6`, block size `1024`, and started from `runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000`.

Data:
- Actual final train file: `data/multiturn_sft_warmstart/lmflow_dataset/train_agentic_mix.json`
- Rows: `98`
- Source counts: `49` self-generated audited tool-call, `25` GSM8K train retention, `24` MBPP train retention.
- Formats: self-generated rows use Qwen-native `<tool_call><function=...><parameter=...>`; GSM8K rows use reasoning with `####`; MBPP rows use code-style retention targets.
- The 50/50 retention mix survived the loader subdirectory fix.

Loss:
- Failed SFT: train loss `2.9342`; logged losses dropped from about `3.68` to `2.67`; no divergence.
- Run-1 reference: train loss `4.3907`; noisy but stable.
- Two-stream retry: train loss `4.1232`; logged losses started around `6.45-7.18` and ended around `1.91-4.26`; no divergence.

## Corrected Retry

- Command path: `scripts/run_flare_redesign_run1.sh`
- Dataset swap only: `DATASET_DIR=/home/mark/qwen_diffusion/data/multiturn_sft_warmstart/lmflow_dataset`
- Output: `runs/multiturn_sft_warmstart_qwen35_9b_twostream400`
- `SKIP_DATASET_BUILD=1`
- Steps: `400`
- Block size: `512`
- Learning rate: `1e-5`
- LoRA: r `16`, alpha `32`, targets `q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj`
- Value-span settings: weight `2.0`, mask probability `1.0`
- Runtime: `1940.0403` seconds
- Train loss: `4.12318115234375`

## Validated GSM8K Gate

- Gate output: `runs/multiturn_sft_warmstart_qwen35_9b_twostream400_gsm8k_gate/summary.json`
- Gate path: full-context fresh-maskban diffusion sampler
- Decode config: `block_size=32`, `small_block_size=32`, `max_new_tokens=256`, `threshold=0.9`, `temperature=0.0`, `top_p=0.95`
- Disjointness: full-hash overlap `0`, content-hash overlap `0`
- Result: strict `1/20 = 0.05`, flex `1/20 = 0.05`
- Required gate: `>=0.70`

## Decision

FAIL. The exact Run-1 two-stream path does not rescue retention for this warm-start dataset. RL remains blocked.

