#!/usr/bin/env bash
# Self-bounded DETACHED orchestrator for the ENVELOPE-CORRECTED 20-instance yield
# probe (stage0_swegym_probe_v2). ONE model server at a time, docker serialized:
# it FIRST waits for the concurrently-running ladder (stage_c_n5v3) to finish
# (its ORCH_DONE.txt, or its orchestrator process to be gone) so we never boot a
# 2nd GPU server or overlap docker with it. THEN, serially:
#   B. GEN    (boot ONE stock-AR server @concurrency 4 at the reference envelope;
#              4 driver shards concurrent; sample GPU)           [needs GPU]
#   C. SCORE  (official SWE-Bench-Fork harness over 20 preds)    [docker only]
#   D. REPORT (corrected yield + greedy-vs-envelope table)       [CPU]
# The PULL stage is skipped: all 20 prebuilt instance images are already
# pulled/tagged/cached from v1. Each stage is self-bounded (probe_gen_v2.sh has
# GPU preflight/boot/settle deadlines + a cleanup trap; the scorer has per-eval
# timeouts). Writes incremental STATUS_*.txt + a final ORCH_DONE_V2.txt.
set -uo pipefail
cd /home/mark/qwen_diffusion
export SUDO_ASKPASS=/tmp/claude-1000/-home-mark/8f712353-03d0-4607-ac3a-cba8072f9d36/scratchpad/askpass.sh

RUN=runs/stage0_swegym_probe_v2
LADDER_DONE=runs/stage_c_n5v3/ORCH_DONE.txt
LADDER_WAIT_S=${LADDER_WAIT_S:-16200}   # 4.5h cap on waiting for the ladder
mkdir -p "$RUN/logs"
LOG="$RUN/logs/pipeline.log"
exec >>"$LOG" 2>&1
DONE="$RUN/ORCH_DONE_V2.txt"

echo "==== V2 YIELD PROBE ORCH START $(date -u +%FT%TZ) ===="
free -g | awk 'NR<=2'; df -h /home/mark | tail -1

# ---- [A] serialize behind the ladder (single-GPU / docker discipline) --------
echo "======== [WAIT-LADDER] $(date -u +%FT%TZ) ========"
wstart=$SECONDS
while :; do
  if [[ -f "$LADDER_DONE" ]]; then
    echo "[wait] ladder ORCH_DONE present ($(cat "$LADDER_DONE")); proceeding"
    break
  fi
  if ! pgrep -f "pipeline_orch_v3.sh" >/dev/null 2>&1 && ! pgrep -f "run_v3_arm.sh" >/dev/null 2>&1; then
    echo "[wait] no ladder orchestrator/arm process alive and no ORCH_DONE -> treat as finished/aborted; GPU preflight will gate the boot"
    break
  fi
  if [[ $((SECONDS - wstart)) -gt $LADDER_WAIT_S ]]; then
    echo "[wait] LADDER WAIT TIMEOUT after ${LADDER_WAIT_S}s -> abort (GPU still held)"
    echo "status=ladder_wait_timeout $(date -u +%FT%TZ)" > "$RUN/STATUS_gen.txt"
    echo "aborted=ladder_wait_timeout $(date -u +%FT%TZ)" > "$DONE"
    exit 1
  fi
  sleep 30
done
# small settle margin so the ladder's final GPU release + report write completes
sleep 20

echo "======== [B] GEN (envelope) $(date -u +%FT%TZ) ========"
SHARDS="driver_shard0 driver_shard1 driver_shard2 driver_shard3" \
  MAX_NUM_SEQS=4 AGENT_WALL_S=900 QWEN_MAX_WALL=840s MAX_TURNS=50 \
  SCOPE=stage0_probe_ar_v2 SWE_EMPTY_PATCH_RETRIES=1 \
  bash $RUN/probe_gen_v2.sh
grc=$?
echo "[orch] gen rc=$grc $(date -u +%FT%TZ)"
echo "gen rc=$grc done=$(date -u +%FT%TZ)" > "$RUN/STATUS_gen.txt"
sleep 5

echo "======== [C] SCORE $(date -u +%FT%TZ) ========"
MAXW=2 RUN_ID=probe20env bash $RUN/probe_score_v2.sh
src=$?
echo "[orch] score rc=$src $(date -u +%FT%TZ)"
echo "score rc=$src done=$(date -u +%FT%TZ)" > "$RUN/STATUS_score.txt"

echo "======== [D] REPORT $(date -u +%FT%TZ) ========"
RUN_ID=probe20env MAX_NUM_SEQS=4 SWE_EMPTY_PATCH_RETRIES=1 .venv/bin/python $RUN/build_report_v2.py
rrc=$?
echo "[orch] report rc=$rrc $(date -u +%FT%TZ)"
echo "report rc=$rrc done=$(date -u +%FT%TZ)" > "$RUN/STATUS_report.txt"

echo "==== V2 YIELD PROBE ORCH DONE $(date -u +%FT%TZ) gen=$grc score=$src report=$rrc ===="
echo "done=$(date -u +%FT%TZ) gen=$grc score=$src report=$rrc" > "$DONE"
