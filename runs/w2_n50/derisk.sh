#!/usr/bin/env bash
# HOST-RAM DE-RISK: boot the arm server (capped scope), measure idle host RAM,
# then fire a C-way CONCURRENT smoke over the first C pool instances (short agent
# wall) while sampling min-MemAvailable / peak-mem-used, tear everything down.
# Decides whether C concurrent episode-containers + the model server fit the cage.
#   usage: derisk.sh <ar|diffusion> <C> <smoke_wall_s>
set -uo pipefail
cd /home/mark/qwen_diffusion
export SUDO_ASKPASS="${SUDO_ASKPASS:?export SUDO_ASKPASS}"
ARM="${1:?arm}"; C="${2:-4}"; SW="${3:-200}"
SUBSET=runs/w2_n50/subset_n50.json
DRIVER=scripts/run_swe_bench_qwen_code.py
PY=.venv/bin/python
OUTBASE=runs/w2_n50/_derisk
GPU_CEIL=8000
export SWE_DOCKER_CMD="sudo -A docker"
export LUMO_PROXY_FORCE_TEMPERATURE=0.6 LUMO_PROXY_FORCE_TOP_P=0.95 LUMO_PROXY_FORCE_TOP_K=20 SWE_EMPTY_PATCH_RETRIES=0
mkdir -p "$OUTBASE/logs"
case "$ARM" in
  ar) PORT=9951; SCOPE=w2_derisk_ar; BOOT_DL=600; MODEL=qwen3.5-9b-ar; TAG=derisk-ar
      LAUNCH='MAX_MODEL_LEN=32768 GPU_UTIL=0.85 MAX_NUM_SEQS=4 PORT=9951 bash runs/stage_c_driver/runcage_ar.sh';;
  diffusion) PORT=9952; SCOPE=w2_derisk_diff; BOOT_DL=900; MODEL=qwen3.5-9b-flare-hybrid-clean; TAG=derisk-diff
      LAUNCH='MAX_MODEL_LEN=32768 GPU_UTIL=0.74 MAX_NUM_SEQS=4 PORT=9952 HF_HUB_OFFLINE=1 bash /home/mark/shared/lumoFlyWheel_codex_fork/scripts/qwen35_9b_flare_hybrid_serve.sh';;
  *) echo bad arm; exit 2;;
esac
gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1|tr -d ' '; }
gpu_capps(){ nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|grep -c .; }
memavail(){ awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
memused(){ free -m|awk '/Mem:/{print $3}'; }
ACTIVE=""
cleanup(){
  pkill -TERM -f "run_swe_bench_qwen_code.py .*_derisk" 2>/dev/null||true
  [[ -n "$ACTIVE" ]] && systemctl --user stop "${ACTIVE}.scope" 2>/dev/null||true
  pkill -TERM -f 'vllm|qwen35_9b_flare_hybrid_serve|runcage_ar|EngineCore' 2>/dev/null||true
  sudo -A docker ps -q --filter "name=swe_ep_" 2>/dev/null|xargs -r sudo -A docker rm -f >/dev/null 2>&1||true
  local dl=$((SECONDS+120)); while :; do n=$(gpu_capps);u=$(gpu_used)
    [[ "$n" -eq 0 && "$u" -lt "$GPU_CEIL" ]]&&break; [[ $SECONDS -gt $dl ]]&&{ pkill -KILL -f 'vllm|qwen35_9b_flare_hybrid_serve|EngineCore' 2>/dev/null||true;break;};sleep 4;done
}
trap cleanup EXIT
# preflight
dl=$((SECONDS+120)); while :; do n=$(gpu_capps);u=$(gpu_used); [[ "$n" -eq 0 && "$u" -lt "$GPU_CEIL" ]]&&break; [[ $SECONDS -gt $dl ]]&&{ echo "preflight TIMEOUT">&2;exit 1;};sleep 5;done
echo "==== DERISK arm=$ARM C=$C smoke_wall=${SW}s $(date -u +%FT%TZ) ===="
echo "[idle-baseline] mem_used=$(memused)M mem_avail=$(memavail)M gpu=$(gpu_used)MiB"
ACTIVE="$SCOPE"
systemd-run --user --scope --unit="$SCOPE" -p MemoryMax=24G -p MemorySwapMax=6G bash -c "$LAUNCH" > "$OUTBASE/logs/${ARM}_server.log" 2>&1 &
# wait ready
bdl=$((SECONDS+BOOT_DL));grace=$((SECONDS+40));seen=0;ready=0
while :; do curl -sf "http://127.0.0.1:${PORT}/v1/models">/dev/null 2>&1&&{ ready=1;break;}
  st=$(systemctl --user is-active "${SCOPE}.scope" 2>/dev/null||true)
  if [[ "$st" == active||"$st" == activating ]];then seen=1; elif [[ $seen -eq 1 ]];then echo "scope died $st">&2;break; elif [[ $SECONDS -gt $grace ]];then echo "never active $st">&2;break;fi
  [[ $SECONDS -gt $bdl ]]&&{ echo "BOOT TIMEOUT">&2;break;};sleep 5;done
[[ $ready -ne 1 ]]&&{ echo "[derisk] BOOT FAILED";tail -30 "$OUTBASE/logs/${ARM}_server.log";exit 2;}
sleep 8
IDLE_USED=$(memused); IDLE_AVAIL=$(memavail); IDLE_GPU=$(gpu_used)
echo "[server-idle] mem_used=${IDLE_USED}M mem_avail=${IDLE_AVAIL}M gpu=${IDLE_GPU}MiB"
# peak/min sampler
peakf=$(mktemp); minf=$(mktemp); echo 0>"$peakf"; echo 999999>"$minf"
( while :; do u=$(memused);a=$(memavail);p=$(cat "$peakf");m=$(cat "$minf")
   [[ "$u" -gt "$p" ]]&&echo "$u">"$peakf"; [[ "$a" -lt "$m" ]]&&echo "$a">"$minf"; sleep 2;done )&
SAMP=$!
# first C instances
IIDS=$($PY -c "import json;print(' '.join(json.load(open('$SUBSET'))['instance_ids'][:$C]))")
echo "[smoke] firing $C concurrent episodes: $IIDS"
declare -a PIDS=(); k=0
for iid in $IIDS; do
  OUT="$OUTBASE/$ARM/shard_$k"; mkdir -p "$OUT"
  SWE_AGENT_WALL_S=$SW LUMO_PROXY_FORCE_SEED=$((1234+k*100000)) \
  $PY $DRIVER --subset "$SUBSET" --only "$iid" --out-root "$OUT" --runtime container \
    --endpoint "http://127.0.0.1:${PORT}/v1" --model "$MODEL" --model-name "$TAG" \
    --repo-cache runs/stage_c_driver/repo_cache --eval-mode skip \
    --agent-wall-s $SW --qwen-max-wall ${SW}s --max-session-turns 50 \
    --proxy-port $((33000+k)) --proxy-dump-dir "$OUT/dump" --proxy-tool-choice "" \
    > "$OUTBASE/logs/${ARM}_smoke_${k}.log" 2>&1 &
  PIDS+=($!); k=$((k+1)); sleep 2
done
for pid in "${PIDS[@]}"; do wait "$pid" || true; done
kill "$SAMP" 2>/dev/null||true
PEAK=$(cat "$peakf"); MIN=$(cat "$minf"); rm -f "$peakf" "$minf"
echo "[RESULT arm=$ARM C=$C] idle_used=${IDLE_USED}M idle_avail=${IDLE_AVAIL}M peak_used=${PEAK}M min_avail=${MIN}M"
echo "{\"arm\":\"$ARM\",\"C\":$C,\"idle_used_m\":$IDLE_USED,\"idle_avail_m\":$IDLE_AVAIL,\"peak_used_m\":$PEAK,\"min_avail_m\":$MIN}" > "$OUTBASE/derisk_${ARM}_c${C}.json"
echo "==== DERISK DONE $(date -u +%FT%TZ) ===="
