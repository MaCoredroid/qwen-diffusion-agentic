#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

check_env() {
  local name="$1"
  local py="$2"
  echo "== ${name} =="
  "${py}" - <<'PY'
import importlib
import torch

print("python ok")
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0))
    x = torch.randn((512, 512), device="cuda", dtype=torch.float16)
    y = x @ x.T
    torch.cuda.synchronize()
    print("matmul", tuple(y.shape), y.dtype)

for mod in ("transformers", "accelerate", "datasets", "peft", "vllm", "sglang"):
    try:
        m = importlib.import_module(mod)
        print(mod, getattr(m, "__version__", "unknown"))
    except Exception as exc:
        print(f"{mod}_skip", type(exc).__name__, str(exc)[:120])

try:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3.6-27B", trust_remote_code=True)
    print("qwen", cfg.model_type, cfg.architectures)
except Exception as exc:
    print("qwen_config_error", repr(exc))
PY
  echo
}

check_env "core .venv" ".venv/bin/python"
check_env "vLLM .venv-vllm" ".venv-vllm/bin/python"
check_env "SGLang .venv-sglang" ".venv-sglang/bin/python"
