#!/usr/bin/env bash
# C46 RE-GATE ITERATION-2 — AR-mode PAIRED arm runner. ONE stock-vLLM AR server for
# the iteration-2 arm-S SWE-SFT weights (models/qwen3.5-9b-fastdllm-mswe-S-iter2-vllm-bf16
# = the SAME fold the iteration-2 KILL-T1 anchor gate served, #127/2632c13), AR-decoded,
# C parallel driver shards over the SAME frozen 48-instance shard_plan.json.
#
# Mirrors runs/k_gate_c46/run_arm_ar.sh (the iteration-1 AR paired arm) EXCEPT:
#   * weights = models/qwen3.5-9b-fastdllm-mswe-S-iter2-vllm-bf16 (iter-2 arm-S fold)
#   * docker via the docker group (plain `docker`, no sudo/askpass on this host)
# Serving = runs/stage_c_driver/runcage_ar.sh SNAP override, stock vLLM 0.23 AR engine,
# cudagraph, native qwen3_xml tools, gmu 0.85, max_num_seqs 4, ml 32768.
# Envelope = AR reference temp 0.6/top_p 0.95/top_k 20 proxy-forced, NO pp, per-request
# seeds, empty-patch re-drive 1, turn cap 75, seed base per shard {1234,101234,201234,301234}.
#   usage: run_arm_ar.sh <concurrency> <outbase> <shard_plan.json>
set -uo pipefail
cd /home/mark/qwen_diffusion

C="${1:?concurrency}"; OUTBASE="${2:?outbase}"; PLAN="${3:?shard_plan.json}"
ARM=ar
SUBSET=runs/k_gate_c46/inputs/subset_c46.json
REPO_CACHE=runs/stage_c_driver/repo_cache
DRIVER=scripts/run_swe_bench_qwen_code.py
PY=.venv/bin/python
AGENT_WALL_S=1500          # 25 min hard agent wall (mirror the twin arm)
QWEN_MAX_WALL=1440s
MAX_TURNS=75               # design §2.1 turn cap (mirror the twin arm)
GPU_CEIL=8000
export SWE_DOCKER_CMD="docker"

# ---- AR reference envelope (v3) + re-drive; presence_penalty DROPPED -----------
export LUMO_PROXY_FORCE_TEMPERATURE=0.6
export LUMO_PROXY_FORCE_TOP_P=0.95
export LUMO_PROXY_FORCE_TOP_K=20
export SWE_EMPTY_PATCH_RETRIES=1
unset LUMO_PROXY_FORCE_PRESENCE_PENALTY 2>/dev/null || true
unset LUMO_PROXY_FORCE_MIN_P 2>/dev/null || true

PORT=9951; SCOPE=c46i2_ar_server; BOOT_DL=600
MODEL=qwen3.5-9b-ar; TAG=c46i2-mswe-S-iter2-ar
PROXY_PORT_KEY=diff_proxy_port   # reuse the frozen shard_plan port assignment (proxy ports are arm-local)
# THE stage-c AR serving path, SNAP overridden to the iteration-2 arm-S SWE-SFT export
# (identical weights the KILL-T1 iter2 gate served, AR-decoded). MEMORY-BUDGET: gmu 0.85
# is the AR arm's own value (flat ~22GB, no GDN align-cache) — NOT copied to the diffusion arm.
LAUNCH='SNAP=/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-mswe-S-iter2-vllm-bf16 MAX_MODEL_LEN=32768 GPU_UTIL=0.85 MAX_NUM_SEQS=4 PORT=9951 bash runs/stage_c_driver/runcage_ar.sh'

mkdir -p "$OUTBASE/logs" "$OUTBASE/$ARM"
slog="$OUTBASE/logs/${ARM}_server.log"
monlog="$OUTBASE/logs/${ARM}_monitor.log"

gpu_used()  { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
gpu_capps() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . ; }

ACTIVE_SCOPE=""; MONPID=""
cleanup() {
  [[ -n "$MONPID" ]] && kill "$MONPID" 2>/dev/null || true
  pkill -TERM -f "run_swe_bench_qwen_code.py .*${OUTBASE}/${ARM}" 2>/dev/null || true
  if [[ -n "$ACTIVE_SCOPE" ]]; then
    echo "[cleanup] stopping $ACTIVE_SCOPE" >&2
    systemctl --user stop "${ACTIVE_SCOPE}.scope" 2>/dev/null || true
    pkill -TERM -f 'vllm|runcage_ar|EngineCore' 2>/dev/null || true
    sleep 3
    local sdl=$((SECONDS+120))
    while :; do local cn cu; cn=$(gpu_capps); cu=$(gpu_used)
      [[ "$cu" -lt "$GPU_CEIL" ]] && { echo "[cleanup] GPU settled capps=$cn ${cu}MiB" >&2; break; }
      [[ $SECONDS -gt $sdl ]] && { echo "[cleanup] GPU settle TIMEOUT capps=$cn ${cu}MiB" >&2; pkill -KILL -f 'vllm|runcage_ar|EngineCore' 2>/dev/null||true; break; }
      sleep 5; done
  fi
  docker ps -q --filter "name=swe_ep_" 2>/dev/null | xargs -r docker rm -f >/dev/null 2>&1 || true
}
trap cleanup EXIT

preflight() {
  local dl=$((SECONDS+600))
  while :; do local n u; n=$(gpu_capps); u=$(gpu_used)
    [[ "$u" -lt "$GPU_CEIL" ]] && { echo "[preflight] clear capps=$n ${u}MiB (gnome baseline tolerated)" >&2; return 0; }
    [[ $SECONDS -gt $dl ]] && { echo "[preflight] TIMEOUT capps=$n ${u}MiB" >&2; return 1; }
    echo "[preflight] busy capps=$n ${u}MiB" >&2; sleep 10; done
}

wait_ready() {
  local port=$1 scope=$2 dl=$3
  local deadline=$((SECONDS+dl)) grace=$((SECONDS+40)) seen=0
  while :; do
    curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1 && { echo "[ready] :${port} up" >&2; return 0; }
    local st; st=$(systemctl --user is-active "${scope}.scope" 2>/dev/null || true)
    if [[ "$st" == active || "$st" == activating ]]; then seen=1
    elif [[ $seen -eq 1 ]]; then echo "[ready] scope $scope died state=$st" >&2; return 2
    elif [[ $SECONDS -gt $grace ]]; then echo "[ready] scope $scope never active state=$st" >&2; return 2; fi
    [[ $SECONDS -gt $deadline ]] && { echo "[ready] :${port} BOOT TIMEOUT" >&2; return 1; }
    sleep 5; done
}

echo "==== C46-ITER2 ARM=$ARM C=$C START $(date -u +%FT%TZ) plan=$PLAN ====" >&2
preflight || { echo "[$ARM] preflight failed" >&2; exit 1; }

echo "[$ARM] launching server scope=$SCOPE port=$PORT (iter2 mswe-S SWE-SFT export, AR decode, gmu0.85)" >&2
ACTIVE_SCOPE="$SCOPE"
systemd-run --user --scope --unit="$SCOPE" \
  -p MemoryMax=24G -p MemorySwapMax=6G \
  bash -c "$LAUNCH" > "$slog" 2>&1 &
wait_ready "$PORT" "$SCOPE" "$BOOT_DL" || { echo "[$ARM] server not ready" >&2; exit 1; }

( while :; do
    echo "$(date -u +%FT%TZ) $(free -m | awk '/Mem:/{printf \"mem_used=%sM mem_avail=%sM\",$3,$7} /Swap:/{printf \" swap_used=%sM\",$3}') gpu=$(gpu_used)MiB capps=$(gpu_capps)"
    sleep 20
  done ) >> "$monlog" 2>&1 &
MONPID=$!

NSHARD=$($PY -c "import json;print(len(json.load(open('$PLAN'))['shards']))")
echo "[$ARM] fan-out over $NSHARD shards" >&2
T_START=$(date +%s)
declare -a PIDS=()
for ((k=0;k<NSHARD;k++)); do
  read IIDS PPORT BSEED < <($PY -c "
import json
p=json.load(open('$PLAN'))['shards'][$k]
print(','.join(p['instance_ids']), p['$PROXY_PORT_KEY'], p['base_seed'])")
  OUT="$OUTBASE/$ARM/shard_$k"
  DUMP="$OUTBASE/$ARM/dumps_shard_$k"
  dlog="$OUTBASE/logs/${ARM}_shard_${k}_driver.log"
  mkdir -p "$OUT" "$DUMP"
  echo "[$ARM shard=$k] n=$(echo $IIDS | tr ',' ' ' | wc -w) pport=$PPORT seed=$BSEED $(date -u +%FT%TZ)" >&2
  LUMO_PROXY_FORCE_SEED="$BSEED" \
  $PY $DRIVER \
    --subset "$SUBSET" --only "$IIDS" \
    --out-root "$OUT" \
    --runtime container \
    --endpoint "http://127.0.0.1:${PORT}/v1" \
    --model "$MODEL" --model-name "$TAG" \
    --repo-cache "$REPO_CACHE" \
    --eval-mode skip \
    --agent-wall-s $AGENT_WALL_S --qwen-max-wall $QWEN_MAX_WALL \
    --max-session-turns $MAX_TURNS \
    --proxy-port "$PPORT" \
    --proxy-dump-dir "$DUMP" \
    --proxy-tool-choice "" \
    > "$dlog" 2>&1 &
  PIDS+=($!)
  sleep 3
done

echo "[$ARM] waiting on ${#PIDS[@]} shards: ${PIDS[*]}" >&2
rc_all=0
for pid in "${PIDS[@]}"; do wait "$pid" || rc_all=$?; done
T_END=$(date +%s)
echo "[$ARM] all shards done rc_all=$rc_all $(date -u +%FT%TZ)" >&2
printf '{"arm":"%s","concurrency":%s,"wall_start_epoch":%s,"wall_end_epoch":%s,"wall_seconds":%s}\n' \
  "$ARM" "$C" "$T_START" "$T_END" "$((T_END-T_START))" > "$OUTBASE/$ARM/arm_timing.json"

kill "$MONPID" 2>/dev/null || true; MONPID=""

echo "[$ARM] stopping server scope=$SCOPE" >&2
systemctl --user stop "${SCOPE}.scope" 2>/dev/null || true
ACTIVE_SCOPE=""
sleep 5
settle=$((SECONDS+180))
while :; do n=$(gpu_capps); u=$(gpu_used)
  [[ "$u" -lt "$GPU_CEIL" ]] && { echo "[$ARM] GPU settled capps=$n ${u}MiB" >&2; break; }
  [[ $SECONDS -gt $settle ]] && { echo "[$ARM] GPU settle timeout capps=$n ${u}MiB" >&2; break; }; sleep 5; done

$PY runs/k_gate_c46/merge_predictions.py "$OUTBASE/$ARM" "$NSHARD" "$OUTBASE/$ARM/predictions.jsonl" >&2
echo "arm=$ARM C=$C rc_all=$rc_all done=$(date -u +%FT%TZ)" > "$OUTBASE/logs/STATUS_${ARM}.txt"
echo "==== C46-ITER2 ARM=$ARM END rc=$rc_all $(date -u +%FT%TZ) ====" >&2
exit 0
