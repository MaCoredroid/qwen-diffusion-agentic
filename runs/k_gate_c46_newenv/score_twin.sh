#!/usr/bin/env bash
# OFFICIAL docker swebench-harness scoring for the iteration-2 twin@K1+clamp arm's
# merged 48-instance predictions. Same harness invocation as the iteration-1
# score_all.sh (local dataset JSON, --cache_level env, same instance list); sudo-free
# (member of the docker group). run_id c46_twinK1 so the gate report builders find it.
# Runs with NO model server up (arm torn down first).
#   usage: score_twin.sh <pred.jsonl> <report_dir> <max_workers>
set -uo pipefail
cd /home/mark/qwen_diffusion

PRED="${1:?pred}"; RDIR="${2:?report_dir}"; MW="${3:-4}"
DATASET_JSON="runs/k_gate_c46/inputs/swe_verified_c46.json"
HH=runs/stage_c_n5/local_eval/.venv-harness/bin/python
RID="c46_twinK1"
mkdir -p "$RDIR"

if [[ ! -s "$PRED" ]]; then echo "[score] MISSING/EMPTY predictions: $PRED" >&2; exit 3; fi
IDS=$(.venv/bin/python -c "import json;print(' '.join([r['instance_id'] for r in json.load(open('$DATASET_JSON'))]))")

echo "==== SCORE arm=twinK1(iter2) pred=$PRED run_id=$RID mw=$MW $(date -u +%FT%TZ) ====" >&2
( cd "$RDIR" && HF_HUB_OFFLINE=1 "/home/mark/qwen_diffusion/$HH" -m swebench.harness.run_evaluation \
    -d "/home/mark/qwen_diffusion/$DATASET_JSON" -s test \
    -p "/home/mark/qwen_diffusion/$PRED" -id "$RID" \
    -i $IDS -n swebench \
    --cache_level env --clean False --max_workers "$MW" --timeout 1800 \
    --report_dir "/home/mark/qwen_diffusion/$RDIR" ) 2>&1 | tail -30
echo "[score] twinK1 report:" >&2
ls -la "$RDIR"/*."$RID".json 2>/dev/null || echo "  (no report json found)" >&2
