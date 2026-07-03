# Cached-SDTT One-Probe Training

Fallback: cached SDTT after DSCD teacher precheck failed.

| Metric | Value |
|---|---:|
| Micro-steps | 4000 |
| Optimizer steps | 4000 |
| Mean reverse KL | 0.7726649581921811 |
| Mean loss | 0.7726649581921811 |
| Trainable params | 18112512 |
| Peak CUDA allocated GiB | 20.66533088684082 |

Student init: `/home/mark/qwen_diffusion/runs/rl_multiturn_grpo_v2/from_selected_base_g4_step300/adapter_model`
Adapter out: `runs/cached_sdtt_v2_teacher_probe/round1/adapter_model`

Loss caveat: sparse top-k reverse-KL over cached teacher support, with no full-vocab teacher mass.
