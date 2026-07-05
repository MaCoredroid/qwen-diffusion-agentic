#!/usr/bin/env bash
# Stock-AR guided batched-throughput sweep on stock vLLM 0.23.0 (.venv-vllm).
# ONE heavy process; RAM cage applied by the systemd-run --scope wrapper.
set -euo pipefail
cd /home/mark/qwen_diffusion
export CUDA_HOME=/home/mark/qwen_diffusion/.venv-vllm/lib/python3.12/site-packages/nvidia/cu13
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_V1=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false
exec /home/mark/qwen_diffusion/.venv-vllm/bin/python \
  /home/mark/qwen_diffusion/runs/p2_batched_rollout_bench/bench_ar_guided.py
