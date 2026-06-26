# Qwen Diffusion Setup

Date: 2026-06-25

## Local RTX 5090

The local machine now has three isolated user-level environments under
`/home/mark/qwen_diffusion`:

- `.venv`: core training/dev environment
- `.venv-vllm`: vLLM serving environment
- `.venv-sglang`: SGLang serving environment

No sudo was used.

### Core environment

Path:

```bash
/home/mark/qwen_diffusion/.venv
```

Purpose:

- PyTorch development
- Transformers/Qwen config and tokenizer work
- PEFT/LoRA experiments
- dataset preparation
- Fast-dLLM/Qwen GDN adaptation work

Verified:

- Python 3.12.13
- PyTorch 2.12.1+cu130
- CUDA runtime 13.0
- RTX 5090 visible as compute capability 12.0
- GPU matmul succeeds
- `transformers` loads `Qwen/Qwen3.6-27B` config as `qwen3_5`

Activate:

```bash
cd /home/mark/qwen_diffusion
source .venv/bin/activate
```

### vLLM environment

Path:

```bash
/home/mark/qwen_diffusion/.venv-vllm
```

Purpose:

- AR serving baselines
- NVFP4 Qwen3.6 local serving
- later diffusion-serving patch work

Verified:

- vLLM 0.23.0
- PyTorch 2.11.0+cu130
- CUDA runtime 13.0
- RTX 5090 visible as compute capability 12.0
- GPU matmul succeeds
- `transformers` loads `Qwen/Qwen3.6-27B` config as `qwen3_5`

Starter command:

```bash
cd /home/mark/qwen_diffusion
HF_XET_HIGH_PERFORMANCE=1 .venv-vllm/bin/vllm serve sakamakismile/Qwen3.6-27B-NVFP4 \
  --trust-remote-code \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 \
  --port 8000
```

Start at 8K context on the 32 GB 5090. Increase to 16K/32K only after measuring
VRAM headroom.

### SGLang environment

Path:

```bash
/home/mark/qwen_diffusion/.venv-sglang
```

Purpose:

- alternative serving stack
- compare scheduler/kernel support against vLLM

Verified:

- SGLang 0.5.9
- PyTorch 2.9.1+cu130
- CUDA runtime 13.0
- RTX 5090 visible as compute capability 12.0
- GPU matmul succeeds
- `transformers` upgraded to 5.12.1 so it recognizes `qwen3_5`

Note: SGLang originally resolved `transformers==4.57.1`, which did not recognize
Qwen3.6's `qwen3_5` config. The local SGLang env was upgraded to
`transformers==5.12.1`.

## Remote Machines

Tailscale SSH endpoints:

- Alienware RTX 5080: configured locally through Tailscale SSH; exact private
  address omitted in public notes.
- GX10 / GB10: configured locally through Tailscale SSH; exact private address
  omitted in public notes.

Short hostnames failed because Tailscale MagicDNS reported DNS health issues. Use
the private Tailscale addresses from local notes.

### Alienware RTX 5080

Verified:

- host: `mark-Alienware-Aurora-ACT1250`
- arch: `x86_64`
- GPU: RTX 5080
- VRAM: 16303 MiB
- driver: 590.48.01
- CUDA reported by `nvidia-smi`: 13.1
- compute capability: 12.0
- Python: 3.12.3
- core env installed at `/home/mark/qwen_diffusion/.venv`
- core env PyTorch 2.12.1+cu130, CUDA matmul verified
- core env `transformers==5.12.1`, Qwen3.6 config load verified
- Docker and NVIDIA container runtime are installed

Recommended role:

- Qwen3.5-4B training/proxy experiments
- eval worker
- data preprocessing
- 4-bit small-model serving

Do not plan to run Qwen3.6-27B FP8 or NVFP4 fully resident on this 16 GB GPU.

### GX10 / GB10

Verified:

- host: `gx10-edb9`
- arch: `aarch64`
- GPU: NVIDIA GB10
- driver: 590.48.01
- CUDA reported by `nvidia-smi`: 13.1
- compute capability: 12.1
- about 117 GiB unified memory visible to Linux
- host Python: 3.12.3
- host PyTorch: 2.4.1 CPU-only (`torch.version.cuda is None`)
- Docker and NVIDIA container runtime are installed
- existing large local images include `lumo-flywheel-vllm:26.01-py3-v0.19.0`
- `lumo-flywheel-vllm:26.01-py3-v0.19.0` verified with:
  - PyTorch 2.10.0a0 NVIDIA build
  - CUDA 13.1 available
  - GB10 visible as compute capability 12.1
  - GPU matmul succeeds
  - vLLM 0.19.0 installed

Recommended role:

- memory-heavy 27B loading/checkpoint/export work
- adapter experiments where memory matters more than raw bandwidth
- containerized PyTorch/vLLM workloads

For GX10, prefer NVIDIA/NGC or already-built local containers over host pip
installs. GB10 is ARM64 (`aarch64`, sm_121), and ordinary host PyTorch wheels are
often CPU-only or not built for `sm_121`.

Verified GX10 container probe:

```bash
bash scripts/probe_gx10_container.sh
```

## Helper Scripts

```bash
cd /home/mark/qwen_diffusion
bash scripts/check_local_envs.sh
bash scripts/probe_machines.sh
bash scripts/probe_gx10_container.sh
bash scripts/serve_vllm_qwen36_nvfp4.sh
```

The serve script starts at 8K context. Treat that as the first smoke test, not the
final context target.

## Next Setup Steps

1. Run `scripts/check_local_envs.sh` after any package changes.
2. Smoke-test vLLM with the NVFP4 checkpoint at 8K context.
3. If vLLM starts cleanly, measure idle/load/generation VRAM with `nvidia-smi`.
4. Set up Alienware only for small/proxy work; do not duplicate the full 27B stack.
5. Use GX10 containers for GB10 GPU workloads instead of relying on host Python.
