#!/usr/bin/env bash
# Run the NEXT undone Opus-teacher episode(s) for the tranche, sequentially, in the
# official per-instance container via the already-running adapter (:30050). Bounded
# to fit under the 600s Bash-tool cap: episode agent-wall 520s / qwen-max-wall 480s
# (matches pilot run_one_episode defaults). Runs ONE episode; opportunistically runs
# a 2nd only if the first finished in <60s (guarantees the 2nd clears 600s too).
# Resumable + idempotent via a per-episode EP_DONE marker. Records a progress row
# (with the adapter usage-log ts window for per-episode token attribution) to
# opus_tranche1/gen_progress.jsonl.
#   usage: gen_next.sh [max_eps] [qwen_wall_s] [agent_wall_s]
set -uo pipefail
cd /home/mark/qwen_diffusion
export SWE_DOCKER_CMD=docker
HERE=runs/swe_datagen_s1
T=$HERE/opus_tranche2
PORT=30050; PPORT=30060
GEN="$T/gen"; mkdir -p "$GEN"
DS="$T/batch/dataset.json"
PROG="$T/gen_progress.jsonl"; touch "$PROG"
MAXEPS="${1:-1}"; QWEN_WALL="${2:-480}"; AGENT_WALL="${3:-520}"
PYBIN=/home/mark/qwen_diffusion/.venv/bin/python

# adapter must be up
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 || { echo "FATAL adapter not ready on :$PORT"; exit 4; }

run_one() {
  local IID="$1"
  local OUT="$GEN/$IID" DUMPS="$GEN/dumps_$IID" SUB="$GEN/subset_$IID.json"
  rm -rf "$OUT" "$DUMPS"; mkdir -p "$OUT" "$DUMPS"
  "$PYBIN" - "$IID" "$DS" "$SUB" <<'PY'
import json,sys
iid,ds,out=sys.argv[1:4]
json.dump({"dataset_name":ds,"split":"train","instance_ids":[iid]},open(out,"w"))
PY
  local ts0 tstart; ts0=$(date +%s.%N); tstart=$(date +%s)
  echo "[ep] -> $IID $(date -u +%FT%TZ) (qwen_wall=${QWEN_WALL}s agent_wall=${AGENT_WALL}s)"
  "$PYBIN" scripts/run_swe_bench_qwen_code.py \
    --subset "$SUB" --out-root "$OUT" --runtime container \
    --endpoint "http://127.0.0.1:$PORT/v1" \
    --model claude-opus-adapter --model-name datagen-opus-teacher \
    --repo-cache runs/stage_c_driver/repo_cache --eval-mode skip \
    --agent-wall-s "$AGENT_WALL" --qwen-max-wall "${QWEN_WALL}s" --max-session-turns 50 \
    --proxy-port $PPORT --proxy-dump-dir "$DUMPS" \
    --container-name-prefix swe_ep_opus --proxy-tool-choice ""
  local rc=$?; local ts1 tend; ts1=$(date +%s.%N); tend=$(date +%s)
  # belt: hard-clean any lingering opus episode container (safe: episodes run
  # strictly sequentially in this tranche, so no concurrent opus episode exists).
  docker ps -q --filter "name=swe_ep_opus" | xargs -r docker rm -f >/dev/null 2>&1 || true
  local PB
  PB=$("$PYBIN" -c "import json,glob;fs=glob.glob('$OUT/*/predictions.jsonl');rows=[json.loads(l) for f in fs for l in open(f) if l.strip()];print(len(rows[0].get('model_patch','') or rows[0].get('prediction','') or '') if rows else -1)" 2>/dev/null || echo -1)
  echo "[ep] <- $IID rc=$rc wall=$((tend-tstart))s patch_bytes=$PB $(date -u +%FT%TZ)"
  "$PYBIN" - "$PROG" "$IID" "$rc" "$((tend-tstart))" "$PB" "$ts0" "$ts1" <<'PY'
import json,sys
prog,iid,rc,wall,pb,ts0,ts1=sys.argv[1:8]
row={"instance_id":iid,"rc":int(rc),"wall_s":int(wall),"patch_bytes":int(pb),
     "ts0":float(ts0),"ts1":float(ts1)}
open(prog,"a").write(json.dumps(row)+"\n")
PY
  touch "$OUT/EP_DONE"
}

t0=$(date +%s); ran=0
for _ in $(seq 1 "$MAXEPS"); do
  # pick next undone id whose image is present
  NEXT=""
  while read -r iid; do
    [ -z "$iid" ] && continue
    [ -f "$GEN/$iid/EP_DONE" ] && continue
    slug1776="${iid/__/_1776_}"
    docker image inspect "swebench/sweb.eval.x86_64.${slug1776}:latest" >/dev/null 2>&1 || continue
    NEXT="$iid"; break
  done < "$T/target_ids.txt"
  [ -z "$NEXT" ] && { echo "NO_MORE_READY_IDS"; break; }
  # budget guard: start an episode ONLY if a FULL agent-wall still fits under the
  # 600s Bash-tool cap (585s soft ceiling). This packs several short episodes per
  # foreground call while GUARANTEEING no episode is ever SIGKILLed mid-run by the
  # cap (which would orphan a container + lose the prediction). No truncation: every
  # launched episode gets its full agent_wall.
  el=$(( $(date +%s) - t0 )); need=$(( AGENT_WALL + 40 ))
  if [ "$ran" -ge 1 ] && [ $(( 585 - el )) -lt "$need" ]; then echo "STOP_BUDGET el=${el}s need=${need}s"; break; fi
  run_one "$NEXT"; ran=$((ran+1))
done

# summary
DONE=$(ls -d "$GEN"/*/EP_DONE 2>/dev/null | wc -l)
TOTAL=$(grep -cve '^$' "$T/target_ids.txt")
echo "GEN_CHUNK ran=$ran done_total=$DONE/$TOTAL wall=$(( $(date +%s)-t0 ))s"
