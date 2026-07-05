#!/usr/bin/env bash
# Bounded waiter: poll until the GPU frees (concurrent session's stage3 probe
# releases it), then run the 6 remaining fresh-boot certificate turns. Emits
# progress lines; exits (re-invoking the agent) on completion or timeout.
set -uo pipefail
CERT=/home/mark/qwen_diffusion/runs/p2_engine_battery_v3/parity_cert_freshboot.jsonl
TURNS="54 55 58 59 61 62"
FREE_NEED=25000   # MiB
MAX_ITERS=180     # ~15 min cap at 5s
i=0
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  free=${free:-0}
  if [ "$free" -ge "$FREE_NEED" ]; then
    echo "WAITER: GPU free=${free}MiB >= ${FREE_NEED}, launching 6 fresh boots"
    break
  fi
  i=$((i+1))
  if [ "$i" -ge "$MAX_ITERS" ]; then
    echo "WAITER: TIMEOUT after ${i} polls, GPU still busy (free=${free}MiB); 6 turns remain pending"
    exit 2
  fi
  sleep 5
done
for GT in $TURNS; do
  EP_START=0 EP_END=19 BENCH_TEMP=0.0 BENCH_ONLY="$GT" BENCH_OUT="$CERT" \
  systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=4G -- \
    bash /home/mark/qwen_diffusion/runs/p2_engine_battery_v3/runcage.sh 2>&1 | grep -E "^\[v3\] gt|ERROR"
done
n=$(grep -c . "$CERT")
echo "WAITER: DONE, cert now has ${n} turns"
