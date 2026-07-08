#!/usr/bin/env bash
# Tight smoke: validate adapter->proxy->driver->container chain end-to-end on ONE
# already-present image with small caps. NOT a data source (goes to smoke/ dir).
set -uo pipefail
cd /home/mark/qwen_diffusion
export SWE_DOCKER_CMD=docker
PILOT=runs/swe_datagen_s1/pilot_opus
PARENT_PID="${1:?parent claude pid for token}"
IID="${2:-dask__dask-10342}"
PORT=30050; PPORT=30061
# Capture the user's long-lived Claude Code OAuth token from the parent process env
# (read-only; never written to disk). Export as ANTHROPIC_AUTH_TOKEN for the adapter.
TOK=$(tr '\0' '\n' < /proc/${PARENT_PID}/environ 2>/dev/null | grep '^CLAUDE_CODE_OAUTH_TOKEN=' | head -1 | cut -d= -f2-)
[ -z "$TOK" ] && { echo "NO TOKEN from parent $PARENT_PID"; exit 3; }
export ANTHROPIC_AUTH_TOKEN="$TOK"
export OPUS_ADAPTER_USAGE_LOG="$PILOT/smoke/usage_adapter.jsonl"
rm -rf "$PILOT/smoke"; mkdir -p "$PILOT/smoke"

# start adapter
.venv/bin/python scripts/opus_openai_adapter.py --backend anthropic \
  --host 127.0.0.1 --port $PORT --anthropic-model claude-opus-4-8 \
  --served-model claude-opus-adapter --max-tokens-floor 2048 --timeout 300 \
  > "$PILOT/smoke/adapter.log" 2>&1 &
ADAPTER_PID=$!
cleanup(){ kill "$ADAPTER_PID" 2>/dev/null || true; docker ps -q --filter "name=swe_ep_smoke" | xargs -r docker rm -f >/dev/null 2>&1 || true; }
trap cleanup EXIT
for i in $(seq 1 20); do curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break; sleep 0.5; done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 || { echo "adapter not ready"; exit 4; }
echo "[smoke] adapter up on :$PORT"

# tight single-instance subset
.venv/bin/python - "$IID" "$PILOT/batch/dataset.json" "$PILOT/smoke/subset.json" <<'PY'
import json,sys
iid,ds,out=sys.argv[1:4]
json.dump({"dataset_name":ds,"split":"train","instance_ids":[iid]},open(out,"w"))
PY

timeout 480 .venv/bin/python scripts/run_swe_bench_qwen_code.py \
  --subset "$PILOT/smoke/subset.json" \
  --out-root "$PILOT/smoke/gen/shard_0" \
  --runtime container \
  --endpoint "http://127.0.0.1:$PORT/v1" \
  --model claude-opus-adapter --model-name datagen-opus-teacher \
  --repo-cache runs/stage_c_driver/repo_cache \
  --eval-mode skip \
  --agent-wall-s 240 --qwen-max-wall 200s --max-session-turns 6 \
  --proxy-port $PPORT \
  --proxy-dump-dir "$PILOT/smoke/gen/dumps_shard_0" \
  --container-name-prefix swe_ep_smoke \
  --proxy-tool-choice "" 2>&1 | tail -25
rc=$?
echo "[smoke] driver rc=$rc"
echo "=== dumps ==="; ls -la "$PILOT/smoke/gen/dumps_shard_0" 2>/dev/null | head
echo "=== adapter usage rows ==="; wc -l "$PILOT/smoke/usage_adapter.jsonl" 2>/dev/null
echo "=== per_task artifacts ==="; ls -R "$PILOT/smoke/gen/shard_0/verified/per_task" 2>/dev/null | head -20
echo "=== predictions ==="; cat "$PILOT/smoke/gen/shard_0/verified/predictions.jsonl" 2>/dev/null | head -c 500; echo
echo "=== adapter errors (429/400/502) ==="; grep -iE "429|400|502|error|rate_limit" "$PILOT/smoke/adapter.log" 2>/dev/null | head
echo "SMOKE_DONE rc=$rc"
