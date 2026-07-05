#!/usr/bin/env bash
# ONE heavy diffusion-engine bench process. Inherits EP_*/BENCH_* from caller.
# Invoke under the RAM cage:
#   systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=4G -- \
#     bash runs/lossless_apc/bench/runcage_engine.sh
set -euo pipefail
cd /home/mark/qwen_diffusion
source /home/mark/qwen_diffusion/runs/lossless_apc/bench/env_engine.sh
exec /home/mark/qwen_diffusion/.venv-vllm-p2-main/bin/python \
  /home/mark/qwen_diffusion/runs/lossless_apc/bench/run_engine_bench.py
