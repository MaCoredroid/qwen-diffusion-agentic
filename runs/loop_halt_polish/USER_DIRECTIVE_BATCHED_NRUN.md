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
