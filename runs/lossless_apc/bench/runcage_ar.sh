#!/usr/bin/env bash
# ONE heavy stock-AR bench process (plain AR, NO FLARE routing). Same vLLM build
# (.venv-vllm-p2-main) and attention backend as the engine arm for matched kernels.
# Invoke under the RAM cage:
#   systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=4G -- \
#     bash runs/lossless_apc/bench/runcage_ar.sh
set -euo pipefail
cd /home/mark/qwen_diffusion
export CUDA_HOME=/home/mark/qwen_diffusion/.venv-vllm-p2-main/lib/python3.12/site-packages/nvidia/cu13
export NVCC_APPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
unset VLLM_QWEN3_5_FLARE VLLM_QWEN3_5_FLARE_DECODE || true
exec /home/mark/qwen_diffusion/.venv-vllm-p2-main/bin/python \
  /home/mark/qwen_diffusion/runs/lossless_apc/bench/run_ar_bench.py
