#!/usr/bin/env bash
# W-2b temp-0.6 recert: gate-OFF then gate-ON, one server at a time, then compare.
set -uo pipefail
HERE=/home/mark/qwen_diffusion/runs/w2b_recert
BOOT=/home/mark/qwen_diffusion/runs/w2_recert/boot.sh
PY=/home/mark/qwen_diffusion/.venv-vllm-p2-main/bin/python
W1D=/home/mark/qwen_diffusion/runs/w1d_recert
export PORT=9952 RECERT_SEED=20260714 RECERT_TEMP=0.6

log(){ echo "[run06 $(date +%H:%M:%S)] $*"; }

$PY "$HERE/mk_fa_corpus.py" || { log "FA corpus build failed"; exit 3; }

run_arm(){
  local gate=$1 tag=$2 slog=$3 scope=$4
  log "boot gate=$gate scope=$scope -> $slog"
  bash "$BOOT" "$gate" "$slog" "$scope" || { log "BOOT FAIL gate=$gate"; return 2; }
  log "boot UP; drive recert ($tag)"
  $PY "$HERE/drive06.py" "$W1D/corpus.jsonl" "$W1D/gold.json" "recert_$tag"
  log "drive FA near-dup ($tag)"
  $PY "$HERE/drive06.py" "$HERE/fa_corpus.jsonl" "$HERE/fa_gold.json" "fa_$tag"
  log "teardown scope=$scope"
  systemctl --user stop "${scope}.scope" 2>/dev/null
  for _ in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ "${used:-9999}" -lt 1500 ] && break
    sleep 3
  done
  log "GPU settled: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1)"
}

run_arm 0 off "$HERE/server_off.log" w2b_off
run_arm 1 on  "$HERE/server_on.log"  w2b_on

log "compare"
$PY "$HERE/compare06.py"
log "DONE"
