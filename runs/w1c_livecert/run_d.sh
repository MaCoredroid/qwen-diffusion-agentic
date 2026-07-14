#!/usr/bin/env bash
# W-1c (d): drive the 6-episode ctx_overflow subset through the ALREADY-BOOTED
# gate-ON twin server (port 9952) + the CERTIFIED read-clamp proxy, using the
# exact frozen C46-iter2 envelope (temp 0.6 / top_p 0.95 / top_k 20, per-shard
# base seeds, turn cap 75, empty-patch re-drive 1, clamp=100). Mirrors
# run_arm_twin.sh's fan-out (lines 156-202) but does NOT boot/tear the server —
# the caller owns the single gate-ON server for the whole W-1c battery.
#   usage: run_d.sh <outbase> <plan.json>
set -uo pipefail
cd /home/mark/qwen_diffusion
OUTBASE="${1:?outbase}"; PLAN="${2:?plan}"; CLAMP=100
ARM=diffusion; PORT=9952
SUBSET=runs/k_gate_c46/inputs/subset_c46.json
REPO_CACHE=runs/stage_c_driver/repo_cache
DRIVER=scripts/run_swe_bench_qwen_code.py
PROXY_SCRIPT=runs/k_gate_c46/proxy_readclamp.py
PY=.venv/bin/python
AGENT_WALL_S=1500; QWEN_MAX_WALL=1440s; MAX_TURNS=75
export SWE_DOCKER_CMD="docker"
export LUMO_PROXY_FORCE_TEMPERATURE=0.6
export LUMO_PROXY_FORCE_TOP_P=0.95
export LUMO_PROXY_FORCE_TOP_K=20
export SWE_EMPTY_PATCH_RETRIES=1
unset LUMO_PROXY_FORCE_PRESENCE_PENALTY 2>/dev/null || true
unset LUMO_PROXY_FORCE_MIN_P 2>/dev/null || true
export LUMO_PROXY_READCLAMP_LIMIT="$CLAMP"
MODEL=qwen3.5-9b-flare-hybrid-clean; TAG=w1c-d-gateon-diffusion

mkdir -p "$OUTBASE/logs" "$OUTBASE/$ARM"
curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 || { echo "[d] server not up on :$PORT" >&2; exit 1; }

NSHARD=$($PY -c "import json;print(len(json.load(open('$PLAN'))['shards']))")
echo "[d] fan-out over $NSHARD shards (clamp=$CLAMP) $(date -u +%FT%TZ)" >&2
T_START=$(date +%s)
declare -a PIDS=()
for ((k=0;k<NSHARD;k++)); do
  read IIDS PPORT BSEED < <($PY -c "
import json
p=json.load(open('$PLAN'))['shards'][$k]
print(','.join(p['instance_ids']), p['diff_proxy_port'], p['base_seed'])")
  OUT="$OUTBASE/$ARM/shard_$k"; DUMP="$OUTBASE/$ARM/dumps_shard_$k"
  dlog="$OUTBASE/logs/${ARM}_shard_${k}_driver.log"
  mkdir -p "$OUT" "$DUMP"
  echo "[d shard=$k] ids=$IIDS pport=$PPORT seed=$BSEED $(date -u +%FT%TZ)" >&2
  LUMO_PROXY_FORCE_SEED="$BSEED" \
  $PY $DRIVER \
    --subset "$SUBSET" --only "$IIDS" \
    --out-root "$OUT" --runtime container \
    --endpoint "http://127.0.0.1:${PORT}/v1" \
    --model "$MODEL" --model-name "$TAG" \
    --repo-cache "$REPO_CACHE" --eval-mode skip \
    --agent-wall-s $AGENT_WALL_S --qwen-max-wall $QWEN_MAX_WALL \
    --max-session-turns $MAX_TURNS \
    --proxy-script "$PROXY_SCRIPT" --proxy-port "$PPORT" \
    --proxy-dump-dir "$DUMP" --proxy-tool-choice "" \
    > "$dlog" 2>&1 &
  PIDS+=($!)
  sleep 3
done
echo "[d] waiting on ${#PIDS[@]} shards: ${PIDS[*]}" >&2
rc_all=0
for pid in "${PIDS[@]}"; do wait "$pid" || rc_all=$?; done
T_END=$(date +%s)
printf '{"arm":"%s","clamp":%s,"wall_start_epoch":%s,"wall_end_epoch":%s,"wall_seconds":%s}\n' \
  "$ARM" "$CLAMP" "$T_START" "$T_END" "$((T_END-T_START))" > "$OUTBASE/$ARM/arm_timing.json"
echo "[d] all shards done rc_all=$rc_all wall=$((T_END-T_START))s $(date -u +%FT%TZ)" >&2
$PY runs/k_gate_c46/merge_predictions.py "$OUTBASE/$ARM" "$NSHARD" "$OUTBASE/$ARM/predictions.jsonl" >&2 || true
exit 0
