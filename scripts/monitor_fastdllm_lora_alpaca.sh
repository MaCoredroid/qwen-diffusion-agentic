#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mark/qwen_diffusion"
PY="${ROOT}/.venv-fastdllm/bin/python"
UNIT="qwen-fastdllm-lora-full.service"
LOG="${ROOT}/logs/fastdllm_qwen25_1p5b_alpaca_lora_full.log"
OUT="${ROOT}/runs/fastdllm_qwen25_1p5b_alpaca_lora_full"
EXIT_FILE="${ROOT}/runs/fastdllm_qwen25_1p5b_alpaca_lora_full.exit"

echo "== service =="
systemctl --user status "${UNIT}" --no-pager --full 2>/dev/null | sed -n '1,35p' || true

echo
echo "== gpu =="
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true

echo
echo "== exit =="
cat "${EXIT_FILE}" 2>/dev/null || true

echo
echo "== latest checkpoints =="
find "${OUT}" -maxdepth 2 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -10 || true

echo
echo "== recent results =="
for f in "${OUT}/trainer_state.json" "${OUT}/train_results.json" "${OUT}/all_results.json"; do
  [ -f "${f}" ] && ls -lh "${f}" && tail -40 "${f}"
done

echo
echo "== log tail =="
if [ -f "${LOG}" ]; then
  "${PY}" - "$LOG" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(errors="replace")
text = text.replace("\r", "\n")
lines = [line for line in text.splitlines() if line.strip()]

interesting = []
patterns = (
    "trainable params:",
    "***** train metrics *****",
    "train_loss",
    "train_runtime",
    "Saving model checkpoint",
    "checkpoint-",
    "Traceback",
    "RuntimeError",
    "OutOfMemory",
    "CUDA out of memory",
)
for line in lines:
    if any(p in line for p in patterns) or re.search(r"\|\s*\d+/\d+\s*\[", line):
        interesting.append(line)

for line in interesting[-80:]:
    if len(line) > 220:
        line = line[:217] + "..."
    print(line)
PY
fi
