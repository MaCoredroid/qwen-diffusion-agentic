#!/usr/bin/env bash
# Run one heavy vLLM never-train battery process inside the RAM cage. BENCH_*/EP_*
# env vars inherited from caller. Sources the v3b env (engine env + FINAL post-fix
# levers VLLM_FLARE_BIDIR_PROBE=1 + VLLM_FLARE_CUDAGRAPH=1, pin 95d8b47). Reuses the
# byte-identical v3b harness (run_battery_v3b.py); only BENCH_REF/BENCH_OUT differ.
# ONE heavy process at a time.
set -euo pipefail
cd /home/mark/qwen_diffusion
source /home/mark/qwen_diffusion/runs/p2_engine_nevertrain/env.sh
exec /home/mark/qwen_diffusion/.venv-vllm-p2-main/bin/python \
  /home/mark/qwen_diffusion/runs/p2_engine_battery_v3b/run_battery_v3b.py
