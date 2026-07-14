#!/usr/bin/env bash
# W-1c: stop the server scope, wait GPU settle, report idle.
#   usage: teardown_server.sh <scope>
set -uo pipefail
SCOPE="${1:?scope}"; GPU_CEIL=8000
gpu_used()  { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
gpu_capps() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . ; }
echo "[teardown] stopping $SCOPE" >&2
systemctl --user stop "${SCOPE}.scope" 2>/dev/null || true
pkill -TERM -f 'qwen35_9b_flare_hybrid_serve|EngineCore|vllm serve|from multiprocessing' 2>/dev/null || true
sleep 4
settle=$((SECONDS+180))
while :; do n=$(gpu_capps); u=$(gpu_used)
  [[ "$u" -lt "$GPU_CEIL" ]] && { echo "[teardown] GPU settled capps=$n ${u}MiB" >&2; exit 0; }
  [[ $SECONDS -gt $settle ]] && { echo "[teardown] settle TIMEOUT capps=$n ${u}MiB — KILL" >&2; pkill -KILL -f 'qwen35_9b_flare_hybrid_serve|EngineCore|vllm' 2>/dev/null||true; sleep 5; exit 0; }
  sleep 5; done
