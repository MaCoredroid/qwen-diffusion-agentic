#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mark/qwen_diffusion"
LOG="${ROOT}/logs/fastdllm_qwen25_1p5b_alpaca_full.log"
PID_FILE="${ROOT}/runs/fastdllm_qwen25_1p5b_alpaca_full.pid"

echo "== process =="
if [ -f "${PID_FILE}" ]; then
  pid="$(cat "${PID_FILE}")"
  if kill -0 "${pid}" 2>/dev/null; then
    ps -p "${pid}" -o pid,etime,pcpu,pmem,cmd
  else
    echo "pid ${pid} is not running"
  fi
else
  echo "no pid file"
fi

echo
echo "== gpu =="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader || true

echo
echo "== latest checkpoints =="
find "${ROOT}/runs/fastdllm_qwen25_1p5b_alpaca_full" -maxdepth 1 -name 'checkpoint-*' -type d 2>/dev/null | sort -V | tail -5 || true

echo
echo "== log tail =="
if [ -f "${LOG}" ]; then
  tail -80 "${LOG}"
else
  echo "missing log: ${LOG}"
fi
