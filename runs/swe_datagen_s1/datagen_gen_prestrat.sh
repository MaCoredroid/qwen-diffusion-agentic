#!/usr/bin/env bash
# BATCH GEN — self-generation for one batch. Boots ONE stock-AR Qwen3.5-9B server
# in the RAM cage @ max_num_seqs=C, drives C concurrent qwen-code shards over the
# batch (episode-in-official-container, native qwen3_xml, REFERENCE ENVELOPE
# forced per-request via each shard's proxy, empty-patch re-drive=1), samples GPU
# util, stops the server + settles. Self-bounded: GPU preflight/boot deadlines +
# a trap that stops the scope and reaps swe_ep_ containers on ANY exit. Ported
# from runs/stage0_swegym_probe_v2/probe_gen_v2.sh, parametrized per batch, with
# --skip-existing so a resumed batch does not re-run finished instances.
#   usage: datagen_gen.sh <batchdir> <gen_root> <concurrency>
set -uo pipefail
cd /home/mark/qwen_diffusion
export SUDO_ASKPASS="${SUDO_ASKPASS:?export SUDO_ASKPASS}"
export SWE_DOCKER_CMD="sudo -A docker"
HERE=runs/swe_datagen_s1
PY=.venv/bin/python
DRIVER=scripts/run_swe_bench_qwen_code.py

BATCHDIR="${1:?batchdir}"; GEN_ROOT="${2:?gen_root}"; C="${3:-4}"
PORT="${PORT:-9951}"
SCOPE="${SCOPE:-datagen_ar}"
AGENT_WALL_S="${AGENT_WALL_S:-900}"
QWEN_MAX_WALL="${QWEN_MAX_WALL:-840s}"
MAX_TURNS="${MAX_TURNS:-50}"          # design allows 50->75; keep 50 (envelope probe cap)
BOOT_DL="${BOOT_DL:-600}"
mkdir -p "$GEN_ROOT" "$BATCHDIR/logs"

# --- reference envelope, FORCED per-request via each shard driver's proxy ------
export LUMO_PROXY_FORCE_TEMPERATURE="${LUMO_PROXY_FORCE_TEMPERATURE:-0.6}"
export LUMO_PROXY_FORCE_TOP_P="${LUMO_PROXY_FORCE_TOP_P:-0.95}"
export LUMO_PROXY_FORCE_TOP_K="${LUMO_PROXY_FORCE_TOP_K:-20}"
export LUMO_PROXY_FORCE_SEED="${LUMO_PROXY_FORCE_SEED:-1234}"
export SWE_EMPTY_PATCH_RETRIES="${SWE_EMPTY_PATCH_RETRIES:-1}"

gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
ACTIVE_SCOPE=""; SAMPLER_PID=""
cleanup() {
  [[ -n "$SAMPLER_PID" ]] && kill "$SAMPLER_PID" 2>/dev/null || true
  if [[ -n "$ACTIVE_SCOPE" ]]; then
    echo "[cleanup] stopping $ACTIVE_SCOPE" >&2
    systemctl --user stop "${ACTIVE_SCOPE}.scope" 2>/dev/null || true; sleep 3
  fi
  sudo -A docker ps -q --filter "name=swe_ep_" 2>/dev/null | xargs -r sudo -A docker rm -f >/dev/null 2>&1 || true
}
trap cleanup EXIT

preflight() {
  local deadline=$((SECONDS+600))
  while :; do
    local u; u=$(gpu_used)
    [[ "$u" -lt 3600 ]] && { echo "[preflight] GPU ${u} MiB clear" >&2; return 0; }
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

N=$(.venv/bin/python -c 'import json,sys;print(len(json.load(open(sys.argv[1]))["instance_ids"]))' "$BATCHDIR/subset.json")
echo "==== BATCH GEN START $(date -u +%FT%TZ) n=$N C=$C envelope=$LUMO_PROXY_FORCE_TEMPERATURE/$LUMO_PROXY_FORCE_TOP_P/$LUMO_PROXY_FORCE_TOP_K retries=$SWE_EMPTY_PATCH_RETRIES ====" >&2
preflight || exit 1
echo "[gen] booting server scope=$SCOPE port=$PORT" >&2
ACTIVE_SCOPE="$SCOPE"
systemd-run --user --scope --unit="$SCOPE" -p MemoryMax=22G -p MemorySwapMax=4G \
  bash -c "MAX_NUM_SEQS=$C MAX_MODEL_LEN=32768 GPU_UTIL=0.85 PORT=$PORT bash $HERE/runcage_ar_probe.sh" \
  > "$BATCHDIR/logs/gen_server.log" 2>&1 &
wait_ready || { echo "[gen] server not ready" >&2; exit 1; }

( while systemctl --user is-active --quiet "${SCOPE}.scope"; do
    echo "$(date -u +%FT%TZ),$(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    sleep 5
  done ) >> "$GEN_ROOT/gpu_util.csv" 2>/dev/null &
SAMPLER_PID=$!

echo "[gen] launching $C shards concurrently $(date -u +%FT%TZ)" >&2
pids=()
for ((k=0; k<C; k++)); do
  SUBSET="$BATCHDIR/shard_${k}.json"
  [[ -s "$SUBSET" ]] || { echo "[gen] shard $k subset missing/empty, skip" >&2; continue; }
  PPORT=$((30030+k))
  $PY $DRIVER \
    --subset "$SUBSET" \
    --out-root "$GEN_ROOT/shard_${k}" \
    --runtime container \
    --endpoint "http://127.0.0.1:${PORT}/v1" \
    --model qwen3.5-9b-ar --model-name "datagen-stockAR-env" \
    --repo-cache runs/stage_c_driver/repo_cache \
    --eval-mode skip \
    --skip-existing \
    --agent-wall-s "$AGENT_WALL_S" --qwen-max-wall "$QWEN_MAX_WALL" \
    --max-session-turns "$MAX_TURNS" \
    --proxy-port "$PPORT" \
    --proxy-dump-dir "$GEN_ROOT/dumps_shard_${k}" \
    --container-name-prefix "swe_ep_s${k}" \
    --proxy-tool-choice "" \
    > "$GEN_ROOT/shard_${k}.driver.log" 2>&1 &
  pids+=($!)
  echo "[gen] shard=$k pid=${pids[-1]} pport=$PPORT n_ids=$(.venv/bin/python -c "import json;print(len(json.load(open('$SUBSET'))['instance_ids']))")" >&2
  sleep 2
done

rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[gen] all shards done rc=$rc $(date -u +%FT%TZ)" >&2

kill "$SAMPLER_PID" 2>/dev/null || true; SAMPLER_PID=""
systemctl --user stop "${SCOPE}.scope" 2>/dev/null || true; ACTIVE_SCOPE=""
sleep 5
settle=$((SECONDS+180))
while :; do u=$(gpu_used); [[ "$u" -lt 3600 ]] && { echo "[gen] GPU settled ${u} MiB" >&2; break; }
  [[ $SECONDS -gt $settle ]] && { echo "[gen] settle timeout ${u} MiB" >&2; break; }; sleep 5; done
echo "==== BATCH GEN END rc=$rc $(date -u +%FT%TZ) ====" >&2
exit $rc
