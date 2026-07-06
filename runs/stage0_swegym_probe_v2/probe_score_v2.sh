#!/usr/bin/env bash
# Stage 0 phase 2 PROBE v2 — SCORE stage. Aggregate the v2 (envelope) shard
# predictions into one predictions.jsonl, then official-filter via the SAME
# SWE-Bench-Fork harness the greedy v1 run used (@242429c, patched to reuse the
# pre-pulled instance images; NO from-scratch build). Reuses v1's .venv-swegym +
# dataset verbatim so scoring is byte-for-byte the same toolchain as greedy — only
# the predictions differ. NO GPU server up. Emits score/<model>.<rid>.json + timing.
set -uo pipefail
cd /home/mark/qwen_diffusion
RUN=runs/stage0_swegym_probe_v2
V1=runs/stage0_swegym_probe
export SUDO_ASKPASS="${SUDO_ASKPASS:?export SUDO_ASKPASS}"
VENVPY=$V1/.venv-swegym/bin/python          # reuse the patched fork venv from v1
DS=$RUN/artifacts/swegym_probe20_dataset.json
MAXW=${MAXW:-2}
RUN_ID=${RUN_ID:-probe20env}
SCORE=$RUN/score
mkdir -p "$SCORE"

# 1) aggregate predictions from all shards
PRED=$SCORE/all_predictions.jsonl
.venv/bin/python - "$RUN" "$PRED" <<'PY'
import sys,glob,json,os
run,pred=sys.argv[1],sys.argv[2]
rows={}
for f in glob.glob(os.path.join(run,"gen","*","verified","predictions.jsonl")):
    for l in open(f):
        l=l.strip()
        if not l: continue
        r=json.loads(l); rows[r["instance_id"]]=r
with open(pred,"w") as o:
    for iid in sorted(rows): o.write(json.dumps(rows[iid])+"\n")
print(f"aggregated {len(rows)} predictions -> {pred}")
PY

IDS=$(.venv/bin/python -c "import json;print(' '.join(json.load(open('$RUN/artifacts/subset_probe20.json'))['instance_ids']))")
NPRED=$(wc -l < "$PRED" 2>/dev/null || echo 0)
echo "==== PROBE SCORE v2 START $(date -u +%FT%TZ) preds=$NPRED maxw=$MAXW rid=$RUN_ID ====" >&2
[[ "$NPRED" -eq 0 ]] && { echo "[score] no predictions to score" >&2; exit 1; }

t0=$(date +%s)
( cd "$SCORE" && timeout 14400 sudo -A env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 "/home/mark/qwen_diffusion/$VENVPY" \
    -m swebench.harness.run_evaluation \
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
cat "$REPORT" 2>/dev/null | .venv/bin/python -c "import sys,json;d=json.load(sys.stdin);print('resolved', d.get('resolved_instances'),'/',d.get('completed_instances'),'completed of',d.get('submitted_instances'),'submitted; empty', d.get('empty_patch_instances'),'errors',d.get('error_instances'))" 2>/dev/null || true
echo "==== PROBE SCORE v2 DONE $(date -u +%FT%TZ) ====" >&2
