#!/usr/bin/env bash
set -euo pipefail
cd /home/mark/qwen_diffusion
source /home/mark/qwen_diffusion/runs/p2_engine_batchgates/env.sh
exec /home/mark/qwen_diffusion/.venv-vllm-p2-main/bin/python \
  /home/mark/qwen_diffusion/runs/p2_engine_batchgates/throughput_engine.py
