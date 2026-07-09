#!/usr/bin/env bash
# Drive ONE Opus-teacher SWE episode through the (already-running) adapter, in the
# official per-instance container. Isolated per-episode out-root + dumps dir so the
# extractor/score-aggregator glob them cleanly. The adapter is a persistent bg server
# on --port; this script only runs the episode (foreground-bounded chunk).
#   usage: run_one_episode.sh <instance_id> [adapter_port] [qwen_max_wall_s] [agent_wall_s]
set -uo pipefail
cd /home/mark/qwen_diffusion
export SWE_DOCKER_CMD=docker
PILOT=runs/swe_datagen_s1/pilot_opus
IID="${1:?instance_id}"
PORT="${2:-30050}"
QWEN_WALL="${3:-480}"
AGENT_WALL="${4:-520}"
GEN="$PILOT/gen"
OUT="$GEN/$IID"
DUMPS="$GEN/dumps_$IID"
rm -rf "$OUT" "$DUMPS"; mkdir -p "$OUT" "$DUMPS"
SUB="$GEN/subset_$IID.json"
.venv/bin/python - "$IID" "$PILOT/batch/dataset.json" "$SUB" <<'PY'
import json,sys
iid,ds,out=sys.argv[1:4]
json.dump({"dataset_name":ds,"split":"train","instance_ids":[iid]},open(out,"w"))
PY

t0=$(date +%s)
echo "[ep] -> $IID $(date -u +%FT%TZ) (qwen_wall=${QWEN_WALL}s agent_wall=${AGENT_WALL}s)"
.venv/bin/python scripts/run_swe_bench_qwen_code.py \
  --subset "$SUB" --out-root "$OUT" --runtime container \
  --endpoint "http://127.0.0.1:$PORT/v1" \
  --model claude-opus-adapter --model-name datagen-opus-teacher \
  --repo-cache runs/stage_c_driver/repo_cache --eval-mode skip \
  --agent-wall-s "$AGENT_WALL" --qwen-max-wall "${QWEN_WALL}s" --max-session-turns 50 \
  --proxy-port 30060 --proxy-dump-dir "$DUMPS" \
  --container-name-prefix swe_ep_opus --proxy-tool-choice ""
rc=$?
t1=$(date +%s)
# hard cleanup of any lingering episode container (belt)
docker ps -q --filter "name=swe_ep_opus" | xargs -r docker rm -f >/dev/null 2>&1 || true
PB=$(.venv/bin/python -c "import json,glob;fs=glob.glob('$OUT/*/predictions.jsonl');rows=[json.loads(l) for f in fs for l in open(f) if l.strip()];print(len(rows[0].get('model_patch','') or rows[0].get('prediction','') or '') if rows else 'NOPRED')" 2>/dev/null || echo "NA")
echo "[ep] <- $IID rc=$rc wall=$((t1-t0))s patch_bytes=$PB $(date -u +%FT%TZ)"
echo "EPISODE_DONE $IID rc=$rc wall=$((t1-t0))s"