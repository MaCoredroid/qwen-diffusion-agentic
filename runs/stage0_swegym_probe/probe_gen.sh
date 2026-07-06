#!/usr/bin/env bash
# Stage 0 phase 2 PROBE — GENERATION stage. Boots ONE stock-AR server at
# max_num_seqs=$MAX_NUM_SEQS (concurrency) in the RAM cage, samples GPU util,
# runs $NSHARDS qwen-code driver shards CONCURRENTLY against it (episode-in-
# container, eval-mode skip -> predictions only), then stops the server + settles.
# Self-bounded: GPU preflight/boot deadlines + trap that stops the scope on ANY exit.
set -uo pipefail
cd /home/mark/qwen_diffusion
RUN=runs/stage0_swegym_probe
export SUDO_ASKPASS="${SUDO_ASKPASS:?export SUDO_ASKPASS}"
export SWE_DOCKER_CMD="sudo -A docker"
PY=.venv/bin/python
DRIVER=scripts/run_swe_bench_qwen_code.py

PORT=${PORT:-9951}
SCOPE=${SCOPE:-stage0_probe_ar}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
AGENT_WALL_S=${AGENT_WALL_S:-900}
QWEN_MAX_WALL=${QWEN_MAX_WALL:-840s}
MAX_TURNS=${MAX_TURNS:-50}
BOOT_DL=${BOOT_DL:-600}
SHARDS="${SHARDS:-driver_shard0 driver_shard1 driver_shard2 driver_shard3}"
GEN=$RUN/gen
mkdir -p "$GEN" "$RUN/logs"

gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }

ACTIVE_SCOPE=""; SAMPLER_PID=""
cleanup() {
  [[ -n "$SAMPLER_PID" ]] && kill "$SAMPLER_PID" 2>/dev/null || true
  if [[ -n "$ACTIVE_SCOPE" ]]; then
    echo "[cleanup] stopping $ACTIVE_SCOPE" >&2
    systemctl --user stop "${ACTIVE_SCOPE}.scope" 2>/dev/null || true
    sleep 3
  fi
  sudo -A docker ps -q --filter "name=swe_ep_" 2>/dev/null | xargs -r sudo -A docker rm -f >/dev/null 2>&1 || true
}
trap cleanup EXIT

preflight() {
  local deadline=$((SECONDS+600))
  while :; do
    local u; u=$(gpu_used)
    [[ "$u" -lt 3000 ]] && { echo "[preflight] GPU ${u} MiB clear" >&2; return 0; }
    [[ $SECONDS -gt $deadline ]] && { echo "[preflight] TIMEOUT ${u} MiB" >&2; return 1; }
    echo "[preflight] GPU ${u} MiB busy..." >&2; sleep 10
  done
}
wait_ready() {
  local deadline=$((SECONDS+BOOT_DL))
  while :; do
    curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && { echo "[ready] :${PORT} up" >&2; return 0; }
    systemctl --user is-active --quiet "${SCOPE}.scope" || { echo "[ready] scope died" >&2; return 2; }
    [[ $SECONDS -gt $deadline ]] && { echo "[ready] BOOT TIMEOUT" >&2; return 1; }
    sleep 5
  done
}

echo "==== PROBE GEN START $(date -u +%FT%TZ) max_num_seqs=$MAX_NUM_SEQS shards=[$SHARDS] ====" >&2
preflight || exit 1
echo "[gen] booting server scope=$SCOPE port=$PORT" >&2
ACTIVE_SCOPE="$SCOPE"
systemd-run --user --scope --unit="$SCOPE" -p MemoryMax=22G -p MemorySwapMax=4G \
  bash -c "MAX_NUM_SEQS=$MAX_NUM_SEQS MAX_MODEL_LEN=32768 GPU_UTIL=0.85 PORT=$PORT bash $RUN/runcage_ar_probe.sh" \
  > "$RUN/logs/gen_server.log" 2>&1 &
wait_ready || { echo "[gen] server not ready" >&2; exit 1; }

# GPU util sampler (5s cadence) while episodes run
( while systemctl --user is-active --quiet "${SCOPE}.scope"; do
    echo "$(date -u +%FT%TZ),$(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    sleep 5
  done ) >> "$GEN/gpu_util.csv" 2>/dev/null &
SAMPLER_PID=$!

echo "[gen] launching ${SHARDS} shards concurrently $(date -u +%FT%TZ)" >&2
k=0
pids=()
for shard in $SHARDS; do
  PPORT=$((30030+k))
  $PY $DRIVER \
    --subset "$RUN/artifacts/${shard}.json" \
    --out-root "$GEN/${shard}" \
    --runtime container \
    --endpoint "http://127.0.0.1:${PORT}/v1" \
    --model qwen3.5-9b-ar --model-name "probe-stockAR" \
    --repo-cache runs/stage_c_driver/repo_cache \
    --eval-mode skip \
    --agent-wall-s "$AGENT_WALL_S" --qwen-max-wall "$QWEN_MAX_WALL" \
    --max-session-turns "$MAX_TURNS" \
    --proxy-port "$PPORT" \
    --proxy-dump-dir "$GEN/dumps_${shard}" \
    --container-name-prefix "swe_ep_s${k}" \
    --proxy-tool-choice "" \
    > "$GEN/${shard}.driver.log" 2>&1 &
  pids+=($!)
  echo "[gen] shard=$shard pid=${pids[-1]} pport=$PPORT" >&2
  k=$((k+1))
  sleep 2
done

rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[gen] all shards done rc=$rc $(date -u +%FT%TZ)" >&2

kill "$SAMPLER_PID" 2>/dev/null || true; SAMPLER_PID=""
systemctl --user stop "${SCOPE}.scope" 2>/dev/null || true; ACTIVE_SCOPE=""
sleep 5
settle=$((SECONDS+180))
while :; do u=$(gpu_used); [[ "$u" -lt 3000 ]] && { echo "[gen] GPU settled ${u} MiB" >&2; break; }
  [[ $SECONDS -gt $settle ]] && { echo "[gen] settle timeout ${u} MiB" >&2; break; }; sleep 5; done
echo "==== PROBE GEN END rc=$rc $(date -u +%FT%TZ) ====" >&2
exit $rc
