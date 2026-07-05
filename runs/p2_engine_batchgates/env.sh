export CUDA_HOME=/home/mark/qwen_diffusion/.venv-vllm-p2-main/lib/python3.12/site-packages/nvidia/cu13
export NVCC_APPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export VLLM_FLARE_BIDIR_PROBE=1
export VLLM_FLARE_CUDAGRAPH=1
# Reduce caching-allocator fragmentation across the many generate() calls the
# gate suite makes (numerics-neutral; avoids a spurious 122MB OOM at the ceiling).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
