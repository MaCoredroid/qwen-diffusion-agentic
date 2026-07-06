#!/usr/bin/env bash
# OFFICIAL docker swebench-harness scoring of the 20 v3 predictions (4 arms x 5).
# Each arm scored as its own run_id (the 5 instance_ids repeat across arms). Runs
# the harness as root via `sudo -A` with a LOCAL dataset JSON (no HF under root).
# Report + logs chowned back to the host user after each arm.
set -uo pipefail
cd /home/mark/qwen_diffusion
export SUDO_ASKPASS="${SUDO_ASKPASS:?export SUDO_ASKPASS first}"

SCRATCH="/tmp/claude-1000/-home-mark/8f712353-03d0-4607-ac3a-cba8072f9d36/scratchpad"
DATASET_JSON="$SCRATCH/swe_verified_n5.json"       # 5 full SWE-bench_Verified records
HH=runs/stage_c_n5/local_eval/.venv-harness/bin/python
IDS="django__django-11119 django__django-12754 django__django-13741 pytest-dev__pytest-8399 sympy__sympy-13757"
SCORE=runs/stage_c_n5v3/scoring
mkdir -p "$SCORE"

for ARM in ar mergedar diffusion diffstock; do
  PRED=runs/stage_c_n5v3/$ARM/verified/predictions.jsonl
  RID=n5v3_$ARM
  echo "==== SCORE arm=$ARM pred=$PRED run_id=$RID $(date -u +%FT%TZ) ====" >&2
  if [[ ! -s "$PRED" ]]; then echo "[score] MISSING/EMPTY predictions: $PRED" >&2; continue; fi
  ( cd "$SCORE" && sudo -A env HF_HUB_OFFLINE=1 "/home/mark/qwen_diffusion/$HH" -m swebench.harness.run_evaluation \
      -d "$DATASET_JSON" -s test \
      -p "/home/mark/qwen_diffusion/$PRED" -id "$RID" \
      -i $IDS -n swebench \
      --cache_level env --clean False --max_workers 1 --timeout 1800 \
      --report_dir "/home/mark/qwen_diffusion/$SCORE" ) 2>&1 | tail -18
  sudo -A chown -R "$(id -u):$(id -g)" "$SCORE" 2>/dev/null || true
  echo "[score] arm=$ARM report:" >&2
  ls -la "$SCORE"/*."$RID".json 2>/dev/null || echo "  (no report json found)" >&2
done
echo "==== V3 SCORING COMPLETE $(date -u +%FT%TZ) ====" >&2
