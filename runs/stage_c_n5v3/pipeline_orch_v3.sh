#!/usr/bin/env bash
# Self-bounded DETACHED orchestrator for the ENVELOPE-CORRECTED 4-arm N=5 v3 run.
# ONE server at a time. Runs ar -> mergedar -> diffusion -> diffstock, each via
# run_v3_arm.sh (preflight/boot/settle deadlines + cleanup trap => self-bounded,
# cannot hang unbounded), then OFFICIAL docker scoring (per-eval timeouts), then
# build_report_v3.py. Writes incremental per-arm STATUS + a final ORCH_DONE.txt.
set -uo pipefail
cd /home/mark/qwen_diffusion
export SUDO_ASKPASS=/tmp/claude-1000/-home-mark/8f712353-03d0-4607-ac3a-cba8072f9d36/scratchpad/askpass.sh

ROOT=runs/stage_c_n5v3
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/pipeline.log"
exec >>"$LOG" 2>&1
DONE="$ROOT/ORCH_DONE.txt"

echo "==== V3 PIPELINE ORCH START $(date -u +%FT%TZ) ===="

for ARM in ar mergedar diffusion diffstock; do
  echo "[orch] ==== launching arm=$ARM $(date -u +%FT%TZ) ===="
  bash runs/stage_c_n5v3/run_v3_arm.sh "$ARM"
  rc=$?
  echo "[orch] arm=$ARM rc=$rc $(date -u +%FT%TZ)"
  echo "arm=$ARM rc=$rc done=$(date -u +%FT%TZ)" > "$ROOT/STATUS_${ARM}.txt"
  sleep 5
done

echo "[orch] ==== scoring (official docker) $(date -u +%FT%TZ) ===="
bash runs/stage_c_n5v3/score_all_v3.sh
echo "[orch] score rc=$? $(date -u +%FT%TZ)"

echo "[orch] ==== build_report_v3.py $(date -u +%FT%TZ) ===="
.venv/bin/python runs/stage_c_n5v3/build_report_v3.py > runs/stage_c_n5v3/report_table.txt 2>&1
echo "[orch] build_report rc=$? $(date -u +%FT%TZ)"

echo "==== V3 PIPELINE ORCH DONE $(date -u +%FT%TZ) ===="
date -u +%FT%TZ > "$DONE"
