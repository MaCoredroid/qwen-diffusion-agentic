# Machine Notes

Date: 2026-06-25

## Tailscale SSH

- Alienware RTX 5080: Tailscale SSH configured; exact private address omitted.
- GX10 / GB10: Tailscale SSH configured; exact private address omitted.

Short hostnames did not resolve from the local shell because Tailscale MagicDNS
reported DNS health issues. Use the private Tailscale addresses from local notes.

## Verified Local

- Host: `mark-OMEN-by-HP-45L-Gaming-Desktop-GT22-3xxx`
- GPU: NVIDIA GeForce RTX 5090
- VRAM: 32607 MiB
- Driver: 595.71.05
- CUDA reported by `nvidia-smi`: 13.2
- Compute capability: 12.0

## Verified Remote

### Alienware

- Host: `mark-Alienware-Aurora-ACT1250`
- GPU: NVIDIA GeForce RTX 5080
- VRAM: 16303 MiB
- Driver: 590.48.01
- CUDA reported by `nvidia-smi`: 13.1
- Compute capability: 12.0
- Python: 3.12.3
- Core env: `/home/mark/qwen_diffusion/.venv`
- Core env PyTorch: 2.12.1+cu130, CUDA available, GPU matmul verified
- Core env Transformers: 5.12.1, loads `Qwen/Qwen3.6-27B` config as `qwen3_5`

### GX10

- Host: `gx10-edb9`
- GPU: NVIDIA GB10
- Driver: 590.48.01
- CUDA reported by `nvidia-smi`: 13.1
- Compute capability: 12.1
- Unified memory: about 117 GiB visible to Linux
- Python: 3.12.3
- Host PyTorch: 2.4.1 CPU-only, CUDA unavailable
- Container path verified: `lumo-flywheel-vllm:26.01-py3-v0.19.0`
- Container PyTorch: 2.10.0a0 NVIDIA build, CUDA 13.1, GB10 matmul verified
- Container vLLM: 0.19.0
