#!/usr/bin/env bash
set -euo pipefail
cd /home/mark/qwen_diffusion
export VLLM_USE_V1=1 VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "${AR_VENV:-/home/mark/qwen_diffusion/.venv-vllm}/bin/python" \
  /home/mark/qwen_diffusion/runs/conversion_tax/run_ar_cell.py
