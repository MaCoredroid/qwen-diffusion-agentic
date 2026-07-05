#!/usr/bin/env bash
# Run one heavy vLLM battery process inside the RAM cage. All BENCH_*/EP_* env
# vars are inherited from the caller. Sources env.sh for the engine env + the
# FINAL-engine levers (VLLM_FLARE_BIDIR_PROBE=1 + VLLM_FLARE_CUDAGRAPH=1).
set -euo pipefail
cd /home/mark/qwen_diffusion
source /home/mark/qwen_diffusion/runs/p2_engine_battery_v3/env.sh
exec /home/mark/qwen_diffusion/.venv-vllm-p2-main/bin/python \
  /home/mark/qwen_diffusion/runs/p2_engine_battery_v3/run_battery_v3.py
