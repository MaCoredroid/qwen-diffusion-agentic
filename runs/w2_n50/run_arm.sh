#!/usr/bin/env bash
# W2 N=50 arm runner: ONE server for the arm, C parallel driver shards over
# disjoint frozen-pool instances (the concurrency fan-out), one server at a time.
#
# FROZEN CONFIG (swe_endgoal_plan.md + runs/stage_c_n5v3_gate report):
#   sampling  = v3 ENVELOPE temp0.6/top_p0.95/top_k20 + SWE_EMPTY_PATCH_RETRIES=1,
#               presence_penalty DROPPED, per-request seeds (LUMO_PROXY_FORCE_SEED).
#   AR serving = gmu 0.85, max_num_seqs=4 (no mamba cache, flat ~22GB).
#   diffusion  = gmu 0.74, max_num_seqs=4, max_model_len 32768 (boot-probed).
#
#   usage: run_arm.sh <ar|diffusion> <concurrency> <outbase> <shard_plan.json>
set -uo pipefail
cd /home/mark/qwen_diffusion
export SUDO_ASKPASS="${SUDO_ASKPASS:?export SUDO_ASKPASS}"

ARM="${1:?arm}"; C="${2:?concurrency}"; OUTBASE="${3:?outbase}"; PLAN="${4:?shard_plan.json}"
SUBSET=runs/w2_n50/subset_n50.json
REPO_CACHE=runs/stage_c_driver/repo_cache
DRIVER=scripts/run_swe_bench_qwen_code.py
PY=.venv/bin/python
AGENT_WALL_S=900
QWEN_MAX_WALL=840s
MAX_TURNS=50
GPU_CEIL=8000
export SWE_DOCKER_CMD="sudo -A docker"

# ---- reference envelope (v3) + re-drive; presence_penalty DROPPED (frozen) ----
export LUMO_PROXY_FORCE_TEMPERATURE=0.6
export LUMO_PROXY_FORCE_TOP_P=0.95
export LUMO_PROXY_FORCE_TOP_K=20
export SWE_EMPTY_PATCH_RETRIES=1
unset LUMO_PROXY_FORCE_PRESENCE_PENALTY 2>/dev/null || true
unset LUMO_PROXY_FORCE_MIN_P 2>/dev/null || true

case "$ARM" in
  ar)
    PORT=9951; SCOPE=w2_ar_server; BOOT_DL=600
    MODEL=qwen3.5-9b-ar; TAG=n5v3-stock-ar
    PROXY_PORT_KEY=ar_proxy_port
    LAUNCH='MAX_MODEL_LEN=32768 GPU_UTIL=0.85 MAX_NUM_SEQS=4 PORT=9951 bash runs/stage_c_driver/runcage_ar.sh'
    ;;
  diffusion)
    PORT=9952; SCOPE=w2_diff_server; BOOT_DL=900
    MODEL=qwen3.5-9b-flare-hybrid-clean; TAG=n5v3-diffusion
    PROXY_PORT_KEY=diff_proxy_port
    LAUNCH='MAX_MODEL_LEN=32768 GPU_UTIL=0.74 MAX_NUM_SEQS=4 PORT=9952 HF_HUB_OFFLINE=1 bash /home/mark/shared/lumoFlyWheel_codex_fork/scripts/qwen35_9b_flare_hybrid_serve.sh'
    ;;
  *) echo "bad arm: $ARM" >&2; exit 2 ;;
esac

mkdir -p "$OUTBASE/logs" "$OUTBASE/$ARM"
slog="$OUTBASE/logs/${ARM}_server.log"
monlog="$OUTBASE/logs/${ARM}_monitor.log"

gpu_used()  { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
gpu_capps() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . ; }

ACTIVE_SCOPE=""; MONPID=""
cleanup() {
  [[ -n "$MONPID" ]] && kill "$MONPID" 2>/dev/null || true
  # reap any lingering driver/qwen/proxy children of this arm
  pkill -TERM -f "run_swe_bench_qwen_code.py .*${OUTBASE}/${ARM}" 2>/dev/null || true
  if [[ -n "$ACTIVE_SCOPE" ]]; then
    echo "[cleanup] stopping $ACTIVE_SCOPE" >&2
    systemctl --user stop "${ACTIVE_SCOPE}.scope" 2>/dev/null || true
    pkill -TERM -f 'vllm|qwen35_9b_flare_hybrid_serve|runcage_ar|EngineCore' 2>/dev/null || true
    sleep 3
    local sdl=$((SECONDS+120))
    while :; do local cn cu; cn=$(gpu_capps); cu=$(gpu_used)
      [[ "$cn" -eq 0 && "$cu" -lt "$GPU_CEIL" ]] && { echo "[cleanup] GPU settled capps=$cn ${cu}MiB" >&2; break; }
      [[ $SECONDS -gt $sdl ]] && { echo "[cleanup] GPU settle TIMEOUT capps=$cn ${cu}MiB" >&2; pkill -KILL -f 'vllm|qwen35_9b_flare_hybrid_serve|EngineCore' 2>/dev/null||true; break; }
      sleep 5; done
  fi
  sudo -A docker ps -q --filter "name=swe_ep_" 2>/dev/null | xargs -r sudo -A docker rm -f >/dev/null 2>&1 || true
}
trap cleanup EXIT

preflight() {
  local dl=$((SECONDS+600))
  while :; do local n u; n=$(gpu_capps); u=$(gpu_used)
    [[ "$n" -eq 0 && "$u" -lt "$GPU_CEIL" ]] && { echo "[preflight] clear capps=$n ${u}MiB" >&2; return 0; }
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

echo "==== W2 ARM=$ARM C=$C START $(date -u +%FT%TZ) plan=$PLAN ====" >&2
preflight || { echo "[$ARM] preflight failed" >&2; exit 1; }

echo "[$ARM] launching server scope=$SCOPE port=$PORT" >&2
ACTIVE_SCOPE="$SCOPE"
systemd-run --user --scope --unit="$SCOPE" \
  -p MemoryMax=24G -p MemorySwapMax=6G \
  bash -c "$LAUNCH" > "$slog" 2>&1 &
wait_ready "$PORT" "$SCOPE" "$BOOT_DL" || { echo "[$ARM] server not ready" >&2; exit 1; }

# passive host+GPU monitor (record only; C is chosen safe upfront, no reactive kill)
( while :; do
    echo "$(date -u +%FT%TZ) $(free -m | awk '/Mem:/{printf \"mem_used=%sM mem_avail=%sM\",$3,$7} /Swap:/{printf \" swap_used=%sM\",$3}') gpu=$(gpu_used)MiB capps=$(gpu_capps)"
    sleep 20
  done ) >> "$monlog" 2>&1 &
MONPID=$!

# --- launch C parallel driver shards -----------------------------------------
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
  sleep 3   # stagger container hydration a touch to avoid a docker thundering herd
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
  [[ "$n" -eq 0 && "$u" -lt "$GPU_CEIL" ]] && { echo "[$ARM] GPU settled capps=$n ${u}MiB" >&2; break; }
  [[ $SECONDS -gt $settle ]] && { echo "[$ARM] GPU settle timeout capps=$n ${u}MiB" >&2; break; }; sleep 5; done

# --- merge shard predictions into one per-arm predictions.jsonl ---------------
$PY runs/w2_n50/merge_predictions.py "$OUTBASE/$ARM" "$NSHARD" "$OUTBASE/$ARM/predictions.jsonl" >&2
echo "arm=$ARM C=$C rc_all=$rc_all done=$(date -u +%FT%TZ)" > "$OUTBASE/logs/STATUS_${ARM}.txt"
echo "==== W2 ARM=$ARM END rc=$rc_all $(date -u +%FT%TZ) ====" >&2
exit 0
