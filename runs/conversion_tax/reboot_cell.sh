#!/usr/bin/env bash
# Reboot-loop driver for the ENGINE cell: fresh engine per iteration
# (exit-on-first-hang), sweep all idxs in [CEN_START,CEN_END]. One heavy process,
# GPU-preflight each boot, RAM-caged. Foreground; bounded by LOOP_BUDGET seconds.
set -uo pipefail
cd /home/mark/qwen_diffusion
: "${CEN_REF:?}"; : "${CEN_OUT:?}"
CEN_MAXTOK="${CEN_MAXTOK:-384}"; CEN_START="${CEN_START:-0}"; CEN_END="${CEN_END:-24}"
TURN_TIMEOUT="${TURN_TIMEOUT:-40}"; BENCH_SEED="${BENCH_SEED:-20260701}"
N=$(python3 -c "import json;print(len([r for r in json.load(open('$CEN_REF')) if $CEN_START<=r['idx']<=$CEN_END]))")
DEADLINE=$(( $(date +%s) + ${LOOP_BUDGET:-540} ))
iter=0
while true; do
  done_n=$(python3 -c "import json;print(len(set(json.loads(l)['idx'] for l in open('$CEN_OUT') if l.strip())))" 2>/dev/null || echo 0)
  if [ "$done_n" -ge "$N" ]; then echo "LOOP_DONE $done_n/$N"; break; fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then echo "LOOP_DEADLINE done_n=$done_n/$N"; break; fi
  iter=$((iter+1))
  for _ in $(seq 1 30); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|head -1); [ -z "$u" ]&&u=0; [ "$u" -lt 2048 ]&&break; sleep 5; done
  echo "=== ITER $iter done_n=$done_n/$N gpu=${u}MiB $(date +%T) ==="
  timeout --signal=KILL "${ITER_TIMEOUT:-200}" systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=4G -- env \
    CEN_REF="$CEN_REF" CEN_OUT="$CEN_OUT" CEN_MAXTOK="$CEN_MAXTOK" CEN_START="$CEN_START" CEN_END="$CEN_END" \
    TURN_TIMEOUT="$TURN_TIMEOUT" BENCH_SEED="$BENCH_SEED" \
    bash /home/mark/qwen_diffusion/runs/conversion_tax/runcage_engine.sh \
    >> "$CEN_OUT.iter.log" 2>&1
  pkill -9 -f run_engine_cell.py 2>/dev/null
  sleep 3
done
echo "=== FINAL $CEN_OUT ==="
python3 -c "
import json
rows=[json.loads(l) for l in open('$CEN_OUT') if l.strip()]
print('n_rows',len(rows),'idxs',sorted(set(r['idx'] for r in rows)))
print('hang/err',sorted(r['idx'] for r in rows if r.get('hang') or r.get('error')))
print('fin_stop',sorted(r['idx'] for r in rows if r.get('finish_reason')=='stop'))
print('fin_length',sorted(r['idx'] for r in rows if r.get('finish_reason')=='length'))
"
