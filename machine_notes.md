# Machine Notes

Date: 2026-06-25

## Tailscale SSH

- Alienware RTX 5080: `mark@100.83.202.36`
- GX10 / GB10: `mark@100.103.10.122`

Short hostnames did not resolve from the local shell because Tailscale MagicDNS
reported DNS health issues. Use the Tailscale IPs above.

## Verified Local

- Host: `mark-OMEN-by-HP-45L-Gaming-Desktop-GT22-3xxx`
- GPU: NVIDIA GeForce RTX 5090
- VRAM: 32607 MiB
- Driver: 595.71.05
- CUDA reported by `nvidia-smi`: 13.2
- Compute capability: 12.0

## Verified Remote

### Alienware

- 2026-06-26 21:56 PDT: reserved again for user work for about 3 hours. Do
  not run Qwen diffusion, teacher, student, or eval jobs here until after about
  2026-06-27 00:56 PDT unless the user releases it.
- 2026-06-26 18:22 PDT: reserved for user work for about 3 hours. Do not run
  Qwen diffusion, teacher, student, or eval jobs here until the user releases it.
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
- Added pip via `ensurepip` and installed `bitsandbytes==0.49.2` for 4-bit
  Qwen3.5-9B AR baselines.
- Verified `Qwen/Qwen3.5-9B` in 4-bit NF4 + double quant uses about 7.75 GiB
  peak CUDA allocation on the RTX 5080 through the direct Transformers path.

### Local Qwen Code Harness

- Node: 22.23.1
- npm: 10.9.8
- Project dev dependency: `@qwen-code/qwen-code@0.19.2`
- Headless smoke works against the local Qwen3.6 SGLang teacher through:
  `/home/mark/qwen_diffusion/scripts/qwen_code_sglang_proxy.py`

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
