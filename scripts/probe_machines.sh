#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== local =="
hostname
uname -m
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader || true
echo

probe_remote() {
  local label="$1"
  local target="$2"
  echo "== ${label}: ${target} =="
  tailscale ssh "${target}" 'hostname; uname -m; nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader || true; if [ -x /home/mark/qwen_diffusion/.venv/bin/python ]; then PYBIN=/home/mark/qwen_diffusion/.venv/bin/python; else PYBIN=python3; fi; echo "python_probe=${PYBIN}"; "${PYBIN}" - <<'"'"'PY'"'"'
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device", torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0))
except Exception as exc:
    print("torch_error", repr(exc))
PY'
  echo
}

if [[ -n "${ALIENWARE_TS_TARGET:-}" ]]; then
  probe_remote "Alienware RTX 5080" "${ALIENWARE_TS_TARGET}"
else
  echo "Skipping Alienware probe; set ALIENWARE_TS_TARGET=user@tailscale-address."
fi

if [[ -n "${GX10_TS_TARGET:-}" ]]; then
  probe_remote "GX10 GB10" "${GX10_TS_TARGET}"
else
  echo "Skipping GX10 probe; set GX10_TS_TARGET=user@tailscale-address."
fi
