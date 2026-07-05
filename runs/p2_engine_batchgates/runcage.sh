#!/usr/bin/env bash
# One heavy batch-gates process inside the RAM cage. BENCH_CONFIG (C|P) selects
# the clean-control vs production engine config. Sources the engine env
# (BIDIR_PROBE=1). CUDAGRAPH is set by the python per config. One heavy process.
set -euo pipefail
cd /home/mark/qwen_diffusion
source /home/mark/qwen_diffusion/runs/p2_engine_batchgates/env.sh
exec /home/mark/qwen_diffusion/.venv-vllm-p2-main/bin/python \
  /home/mark/qwen_diffusion/runs/p2_engine_batchgates/batch_gates.py
