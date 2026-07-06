# USER DIRECTIVE (2026-07-06): the N=50 run is BATCHED — concurrency 4+ (as high as HBM allows), not serial
No week-long serial campaign. Both arms run episodes CONCURRENTLY against their server (continuous
batching): baseline concurrency 4; probe higher (6/8) if HBM headroom holds (engine correctness certified
to bs=8 @ gpu_mem 0.82; b16 needs gmu<=0.62 — pick the safest high setting, measure, don't assume).
CONSEQUENCES FOR THE FROZEN CONFIG + DESIGN:
- resolve@1 primary (paired McNemar) is UNAFFECTED (per-request seeds still deterministic per episode).
- per-episode wall under concurrency is queue-inflated: report speed as THROUGHPUT (episodes/GPU-h) at the
  chosen concurrency + cite the v3 b=1 per-episode walls for latency context; do NOT present concurrent
  wall as latency.
- Tier1 instance images: ~50 pulls (~200GB class) — pull stage in the orchestrator, disk-checked.
- Estimated wall at c=4: image pulls ~2-4h + serving ~2-4h/arm + official scoring hours => ~1-2 days total.

## USER NOTE (2026-07-06): APC/KV sizing must leave room for the MAMBA CHECKPOINT CACHE — it lives OUTSIDE the KV pool
When setting gpu_memory_utilization / concurrency for the diffusion arm: the GDN/mamba align-checkpoint
state cache is allocated OUTSIDE the KV block pool (measured geometry: fp32 conv+ssm state slots, ~0.55GB/
layer class x 24 layers at full allocation; plus per-request read-only-denoise snapshots). This is exactly
why b16 OOM'd at gmu 0.74 and fit only at 0.62 while AR stayed flat (~22GB). RULE for the frozen N-run
config: budget = weights + KV pool (gmu) + mamba checkpoint cache + per-request snapshots + CUDA graphs;
pick concurrency AND gmu together with measured headroom (boot probe at the chosen setting, verify no
allocation failure at max concurrent long-context episodes) — never copy the AR arm's gmu to the
diffusion arm.
