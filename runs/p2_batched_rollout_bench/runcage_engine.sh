#!/usr/bin/env bash
# FLARE engine batched-throughput sweep. Certified v3b/nevertrain engine env
# (pin 95d8b47, BIDIR_PROBE=1, PIECEWISE cudagraph). ONE heavy process; RAM cage
# applied by the systemd-run --scope wrapper at invocation.
set -euo pipefail
cd /home/mark/qwen_diffusion
export CUDA_HOME=/home/mark/qwen_diffusion/.venv-vllm-p2-main/lib/python3.12/site-packages/nvidia/cu13
export NVCC_APPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export VLLM_FLARE_BIDIR_PROBE=1
export VLLM_FLARE_CUDAGRAPH=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
exec /home/mark/qwen_diffusion/.venv-vllm-p2-main/bin/python \
  /home/mark/qwen_diffusion/runs/p2_batched_rollout_bench/bench_engine.py
