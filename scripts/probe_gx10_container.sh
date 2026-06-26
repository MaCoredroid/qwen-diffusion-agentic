#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${GX10_TS_TARGET:-}" ]]; then
  echo "Set GX10_TS_TARGET=user@tailscale-address before running this probe." >&2
  exit 2
fi

tailscale ssh "${GX10_TS_TARGET}" 'bash -s' <<'REMOTE'
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  lumo-flywheel-vllm:26.01-py3-v0.19.0 bash -lc 'python3 - <<'"'"'PY'"'"'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0))
    x = torch.randn((512, 512), device="cuda", dtype=torch.float16)
    y = x @ x.T
    torch.cuda.synchronize()
    print("matmul", tuple(y.shape), y.dtype)
try:
    import vllm
    print("vllm", vllm.__version__)
except Exception as exc:
    print("vllm_error", repr(exc))
PY'
REMOTE
