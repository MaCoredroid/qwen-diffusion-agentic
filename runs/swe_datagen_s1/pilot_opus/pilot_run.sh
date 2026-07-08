#!/usr/bin/env bash
# N=10 OPUS-TEACHER PILOT — full pipeline, CPU/docker/API only (NO GPU, NO orch).
#   pull missing images -> start Opus adapter -> drive 10 episodes SEQUENTIALLY in
#   the official per-instance containers via the adapter -> stop adapter -> score
#   through the source-matched harness -> extract keepers into pilot_opus/keepers.
# Token is the user's long-lived Claude Code OAuth token (greenlit Opus-as-teacher);
# captured read-only from the parent process env, never written to disk.
#   usage: pilot_run.sh   (expects ANTHROPIC_AUTH_TOKEN already exported)
set -uo pipefail
cd /home/mark/qwen_diffusion
export SWE_DOCKER_CMD=docker
PILOT=runs/swe_datagen_s1/pilot_opus
GEN="$PILOT/gen"
PORT=30050; PPORT=30060
LOG="$PILOT/pilot_run.log"
: > "$LOG"
say(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

[ -z "${ANTHROPIC_AUTH_TOKEN:-}" ] && { say "FATAL: ANTHROPIC_AUTH_TOKEN unset"; echo "FAIL no_token" > "$PILOT/PIPELINE_DONE"; exit 3; }
export OPUS_ADAPTER_USAGE_LOG="$PILOT/usage_adapter.jsonl"
: > "$OPUS_ADAPTER_USAGE_LOG"
mkdir -p "$GEN"

ADAPTER_PID=""
cleanup(){
  [ -n "$ADAPTER_PID" ] && kill "$ADAPTER_PID" 2>/dev/null || true
  docker ps -q --filter "name=swe_ep_opus" | xargs -r docker rm -f >/dev/null 2>&1 || true
}
trap cleanup EXIT

# 1) PULL the batch's images (reuses present, pulls the 5 missing) --------------
say "PULL start"
bash runs/swe_datagen_s1/datagen_pull.sh "$PILOT/batch" >> "$LOG" 2>&1
say "PULL done"

# 2) START adapter (anthropic backend, Opus 4.8) -------------------------------
say "adapter start :$PORT"
.venv/bin/python scripts/opus_openai_adapter.py --backend anthropic \
  --host 127.0.0.1 --port $PORT --anthropic-model claude-opus-4-8 \
  --served-model claude-opus-adapter --max-tokens-floor 2048 --timeout 300 \
  > "$PILOT/adapter.log" 2>&1 &
ADAPTER_PID=$!
for i in $(seq 1 40); do curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break; sleep 0.5; done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 || { say "FATAL adapter not ready"; echo "FAIL adapter" > "$PILOT/PIPELINE_DONE"; exit 4; }
say "adapter ready"

# 3) GEN — drive the 10 episodes sequentially through the adapter --------------
say "GEN start (10 episodes, sequential, ~900s wall each)"
.venv/bin/python scripts/run_swe_bench_qwen_code.py \
  --subset "$PILOT/batch/shard_0.json" \
  --out-root "$GEN/shard_0" \
  --runtime container \
  --endpoint "http://127.0.0.1:$PORT/v1" \
  --model claude-opus-adapter --model-name datagen-opus-teacher \
  --repo-cache runs/stage_c_driver/repo_cache \
  --eval-mode skip \
  --agent-wall-s 900 --qwen-max-wall 840s --max-session-turns 50 \
  --proxy-port $PPORT \
  --proxy-dump-dir "$GEN/dumps_shard_0" \
  --container-name-prefix swe_ep_opus \
  --proxy-tool-choice "" >> "$LOG" 2>&1
GENRC=$?
say "GEN done rc=$GENRC"

# 4) STOP adapter (free the token before scoring) ------------------------------
kill "$ADAPTER_PID" 2>/dev/null || true; ADAPTER_PID=""
sleep 2

# 5) SCORE — source-matched harness (all 10 are swe_gym -> fork) ---------------
say "SCORE start"
bash runs/swe_datagen_s1/datagen_score.sh "$PILOT/batch" "$GEN" 4 >> "$LOG" 2>&1
SCORERC=$?
REPORT="$PILOT/batch/score/datagen-eval.batch.json"
say "SCORE done rc=$SCORERC report=$REPORT"

# 6) KEEPERS — extract resolved episodes into pilot_opus/keepers ---------------
say "KEEPERS start"
.venv/bin/python runs/swe_datagen_s1/extract_keepers.py \
  "$PILOT/batch" batch "$GEN" "$REPORT" "$PILOT/keepers" \
  '{"temperature":null,"top_p":null,"top_k":null,"seed_base":null,"generator":"opus-4.8-adapter"}' \
  >> "$LOG" 2>&1
KEEPRC=$?
say "KEEPERS done rc=$KEEPRC"

echo "DONE gen=$GENRC score=$SCORERC keepers=$KEEPRC" > "$PILOT/PIPELINE_DONE"
say "PIPELINE COMPLETE"
