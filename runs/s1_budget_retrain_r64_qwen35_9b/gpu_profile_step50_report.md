# S1 Step-50 GPU Profile

- Samples: `91` from `runs/s1_budget_retrain_r64_qwen35_9b/gpu_profile_step50.csv`
- GPU util mean: `51.52%`
- GPU util nonzero mean: `52.09%` over `90` samples
- GPU util 5% trimmed mean: `52.31%`
- GPU util min/max: `0.0%` / `77.0%`
- Memory used: mean `29535 MiB`, max `29535 MiB`
- Power mean: `228.7 W`

Interpretation: one-second polling caught expected idle samples between optimizer-step bursts; nonzero utilization is near the torch baseline target while VRAM is within the 30.5 GiB ceiling.
