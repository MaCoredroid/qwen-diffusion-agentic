#!/usr/bin/env bash
# BATCH SCORE — official verifiable reward for one batch (docker-only; NO GPU up).
# Merge the C shard predictions into one predictions.jsonl, then filter with the
# SWE-Gym SWE-Bench-Fork harness (@242429c + fork_reuse_prebuilt.patch so it
# REUSES the pulled instance images instead of rebuilding). Byte-identical
# toolchain to runs/stage0_swegym_probe_v2/probe_score_v2.sh, parametrized per
# batch. Emits <batchdir>/score/<model>.<run_id>.json (the resolved_ids report).
#   usage: datagen_score.sh <batchdir> <gen_root> [max_workers]
set -uo pipefail
cd /home/mark/qwen_diffusion
export SUDO_ASKPASS="${SUDO_ASKPASS:?export SUDO_ASKPASS}"
V1=runs/stage0_swegym_probe
VENVPY="$V1/.venv-swegym/bin/python"          # reuse the patched fork venv
PY=.venv/bin/python

BATCHDIR="${1:?batchdir}"; GEN_ROOT="${2:?gen_root}"; MAXW="${3:-4}"
DS="$BATCHDIR/dataset.json"
RUN_ID="$(basename "$BATCHDIR")"
SCORE="$BATCHDIR/score"; mkdir -p "$SCORE"
PRED="$SCORE/all_predictions.jsonl"

# 1) aggregate predictions from all shards (one row per instance_id)
$PY - "$GEN_ROOT" "$PRED" <<'PY'
import sys, glob, json, os
gen, pred = sys.argv[1], sys.argv[2]
rows = {}
for f in glob.glob(os.path.join(gen, "*", "verified", "predictions.jsonl")):
    for l in open(f):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l); rows[r["instance_id"]] = r
with open(pred, "w") as o:
    for iid in sorted(rows):
        o.write(json.dumps(rows[iid]) + "\n")
print(f"aggregated {len(rows)} predictions -> {pred}")
PY

# instance ids that actually produced a (possibly empty) prediction row
IDS=$($PY -c 'import json,sys;print(" ".join(json.loads(l)["instance_id"] for l in open(sys.argv[1]) if l.strip()))' "$PRED")
NPRED=$(wc -l < "$PRED" 2>/dev/null || echo 0)
echo "==== BATCH SCORE START $(date -u +%FT%TZ) preds=$NPRED maxw=$MAXW rid=$RUN_ID ====" >&2
[[ "$NPRED" -eq 0 ]] && { echo "[score] no predictions to score" >&2; exit 1; }

t0=$(date +%s)
( cd "$SCORE" && timeout 21600 sudo -A env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    "/home/mark/qwen_diffusion/$VENVPY" -m swebench.harness.run_evaluation \
    --dataset_name "/home/mark/qwen_diffusion/$DS" --split train \
    --predictions_path "/home/mark/qwen_diffusion/$PRED" \
    --instance_ids $IDS \
    --run_id "$RUN_ID" \
    --cache_level instance --clean False --max_workers "$MAXW" --timeout 1800 ) 2>&1 | tail -25
t1=$(date +%s)
sudo -A chown -R "$(id -u):$(id -g)" "$SCORE" 2>/dev/null || true
REPORT=$(ls "$SCORE"/*."$RUN_ID".json 2>/dev/null | head -1)
echo "{\"score_wall_s\": $((t1-t0)), \"report\": \"${REPORT}\", \"maxw\": $MAXW}" > "$SCORE/timing.json"
echo "[score] report=$REPORT wall=$((t1-t0))s" >&2
$PY -c "import sys,json;d=json.load(open(sys.argv[1]));print('resolved',len(d.get('resolved_ids',[])),'/',d.get('completed_instances'),'completed of',d.get('submitted_instances'),'submitted; empty',d.get('empty_patch_instances'),'errors',d.get('error_instances'))" "$REPORT" 2>/dev/null || true
echo "==== BATCH SCORE DONE $(date -u +%FT%TZ) ====" >&2
