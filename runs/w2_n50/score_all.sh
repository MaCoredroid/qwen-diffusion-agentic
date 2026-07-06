#!/usr/bin/env bash
# OFFICIAL docker swebench-harness scoring for ONE arm's merged 50-instance
# predictions. Byte-identical harness invocation to the gate's gate_score.sh
# (local dataset JSON, sudo -A root, --cache_level env) scaled to N=50 with
# --max-workers (MAX_JOBS cap). Runs with NO model server up (arms torn down).
#   usage: score_all.sh <arm> <pred.jsonl> <report_dir> <max_workers>
set -uo pipefail
cd /home/mark/qwen_diffusion
export SUDO_ASKPASS="${SUDO_ASKPASS:?export SUDO_ASKPASS}"

ARM="${1:?arm}"; PRED="${2:?pred}"; RDIR="${3:?report_dir}"; MW="${4:-4}"
SCRATCH="/tmp/claude-1000/-home-mark/8f712353-03d0-4607-ac3a-cba8072f9d36/scratchpad"
DATASET_JSON="$SCRATCH/swe_verified_n50.json"
HH=runs/stage_c_n5/local_eval/.venv-harness/bin/python
RID="w2n50_${ARM}"
mkdir -p "$RDIR"

if [[ ! -s "$PRED" ]]; then echo "[score] MISSING/EMPTY predictions: $PRED" >&2; exit 3; fi
IDS=$(.venv/bin/python -c "import json;print(' '.join(json.loads(open('$DATASET_JSON').read()) and [r['instance_id'] for r in json.load(open('$DATASET_JSON'))]))")

echo "==== SCORE arm=$ARM pred=$PRED run_id=$RID mw=$MW $(date -u +%FT%TZ) ====" >&2
( cd "$RDIR" && sudo -A env HF_HUB_OFFLINE=1 "/home/mark/qwen_diffusion/$HH" -m swebench.harness.run_evaluation \
    -d "$DATASET_JSON" -s test \
    -p "/home/mark/qwen_diffusion/$PRED" -id "$RID" \
    -i $IDS -n swebench \
    --cache_level env --clean False --max_workers "$MW" --timeout 1800 \
    --report_dir "/home/mark/qwen_diffusion/$RDIR" ) 2>&1 | tail -30
sudo -A chown -R "$(id -u):$(id -g)" "$RDIR" 2>/dev/null || true
echo "[score] arm=$ARM report:" >&2
ls -la "$RDIR"/*."$RID".json 2>/dev/null || echo "  (no report json found)" >&2
