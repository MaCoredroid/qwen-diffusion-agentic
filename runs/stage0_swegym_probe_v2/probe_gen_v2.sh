#!/usr/bin/env bash
# Stage 0 phase 2 PROBE v2 — ENVELOPE-CORRECTED GENERATION stage.
#
# IDENTICAL to runs/stage0_swegym_probe/probe_gen.sh (same 20 SWE-Gym instances,
# same 4 driver shards, same stock-AR server @ max_num_seqs=4, same aligned
# episode-in-official-container runtime, same eval-mode skip) EXCEPT the
# reference envelope is FORCED per request through each shard driver's own proxy:
#
#     LUMO_PROXY_FORCE_TEMPERATURE=0.6  TOP_P=0.95  TOP_K=20  SEED=1234
#
# The driver's _start_proxy() Popen inherits this env (no env= override), so the
# proxy OVERWRITES qwen-code's greedy-ish request bodies with the certified
# anti-degenerate regime (banked in runs/stage_c_n5v2/report.md) + a per-request
# reproducible seed (base+counter, atomic itertools.count per proxy process).
# The greedy v1 run (temp 0) is the A/B baseline this corrects.
#
# Also sets SWE_EMPTY_PATCH_RETRIES=1 -> the state-conditional empty-patch
# re-drive mitigation for the known temp-0.6 tool-call-free-terminal flake.
#
# Boots ONE stock-AR server (RAM cage), samples GPU util, runs the 4 shards
# CONCURRENTLY, stops the server + settles. Self-bounded: GPU preflight/boot
# deadlines + trap that stops the scope on ANY exit.
set -uo pipefail
cd /home/mark/qwen_diffusion
RUN=runs/stage0_swegym_probe_v2
export SUDO_ASKPASS="${SUDO_ASKPASS:?export SUDO_ASKPASS}"
export SWE_DOCKER_CMD="sudo -A docker"
PY=.venv/bin/python
DRIVER=scripts/run_swe_bench_qwen_code.py

# --- reference envelope, FORCED per-request via each shard driver's proxy -----
export LUMO_PROXY_FORCE_TEMPERATURE=0.6
export LUMO_PROXY_FORCE_TOP_P=0.95
export LUMO_PROXY_FORCE_TOP_K=20
export LUMO_PROXY_FORCE_SEED=1234
# re-drive mitigation for the temp-0.6 tool-call-free-terminal flake
export SWE_EMPTY_PATCH_RETRIES="${SWE_EMPTY_PATCH_RETRIES:-1}"

PORT=${PORT:-9951}
SCOPE=${SCOPE:-stage0_probe_ar_v2}
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

echo "==== PROBE GEN v2 START $(date -u +%FT%TZ) max_num_seqs=$MAX_NUM_SEQS shards=[$SHARDS] ENVELOPE temp=0.6/top_p=0.95/top_k=20/seed=1234 retries=$SWE_EMPTY_PATCH_RETRIES ====" >&2
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
    --model qwen3.5-9b-ar --model-name "probe-stockAR-env" \
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
echo "==== PROBE GEN v2 END rc=$rc $(date -u +%FT%TZ) ====" >&2
exit $rc
