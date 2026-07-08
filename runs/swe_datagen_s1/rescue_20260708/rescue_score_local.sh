#!/usr/bin/env bash
# LOCAL-ONLY rescue re-score: score ONLY the gym ids whose fork image is ALREADY
# present locally (retag driver/xingyaoww -> fork key; NEVER pull). Pull-free so it
# is fast, bounded, and cannot block on 900s image pulls or starve the live orch.
# Reuses the pre-built corrected dataset_gym.json in each rescue batch dir.
#   usage: rescue_score_local.sh <batch_dirname> [max_workers] [run_tag]
set -uo pipefail
cd /home/mark/qwen_diffusion
BATCH="${1:?batch dirname}"; MAXW="${2:-6}"; RUNTAG="${3:-local}"
HERE=runs/swe_datagen_s1
BR="$HERE/batches/$BATCH"
RD="$HERE/rescue_20260708/$BATCH"
PY=/home/mark/qwen_diffusion/.venv/bin/python
FORK_PY=/home/mark/qwen_diffusion/runs/stage0_swegym_probe/.venv-swegym/bin/python
D=docker
SOURCES="$BR/sources.json"
DS_GYM="$RD/dataset_gym.json"
mkdir -p "$RD/score/parts/fork"
PRED="$RD/score/all_predictions.jsonl"
PRED_GYM="$RD/score/pred_gym_${RUNTAG}.jsonl"
IDS_FILE="$RD/score/gym_ids_${RUNTAG}.txt"
IMGLOG="$RD/image_local_${RUNTAG}.log"
: > "$IMGLOG"

echo "==== RESCUE-LOCAL $BATCH START $(date -u +%FT%TZ) maxw=$MAXW tag=$RUNTAG ====" >&2

# 1) aggregate predictions from gen shards (one row per instance_id)
$PY - "$BR/gen" "$PRED" <<'PYEOF'
import sys, glob, json, os
gen, pred = sys.argv[1], sys.argv[2]
rows={}
for f in glob.glob(os.path.join(gen,"*","verified","predictions.jsonl")):
    for l in open(f):
        l=l.strip()
        if not l: continue
        r=json.loads(l); rows[r["instance_id"]]=r
with open(pred,"w") as o:
    for iid in sorted(rows): o.write(json.dumps(rows[iid])+"\n")
print(f"[rescue-local] aggregated {len(rows)} predictions", file=sys.stderr)
PYEOF

# 2) filter to gym preds whose fork image is LOCAL (retag if driver/xingyaoww present)
$PY - "$PRED" "$SOURCES" "$PRED_GYM" "$IDS_FILE" <<'PYEOF'
import json,sys,subprocess
pred,sources,outg,idsf=sys.argv[1:5]
try: src=json.load(open(sources))
except Exception: src={}
li=set(l.strip() for l in subprocess.run(
    ["docker","images","--format","{{.Repository}}:{{.Tag}}"],
    capture_output=True,text=True).stdout.splitlines())
def ensure_local(iid):
    fork=f"sweb.eval.x86_64.{iid}:latest"
    drv=f"swebench/sweb.eval.x86_64.{iid.replace('__','_1776_')}:latest"
    xsr=f"xingyaoww/sweb.eval.x86_64.{iid.replace('__','_s_')}:latest"
    if fork in li: return True
    for k in (drv,xsr):
        if k in li:
            subprocess.run(["docker","tag",k,fork],check=False)
            return True
    return False
rows=[json.loads(l) for l in open(pred) if l.strip()]
gym=[r for r in rows if src.get(r["instance_id"],"swe_gym")=="swe_gym"]
keep=[r for r in gym if ensure_local(r["instance_id"])]
with open(outg,"w") as g:
    for r in keep: g.write(json.dumps(r)+"\n")
open(idsf,"w").write("\n".join(r["instance_id"] for r in keep)+"\n")
ne=sum(1 for r in keep if (r.get("model_patch") or "").strip())
print(f"[rescue-local] gym={len(gym)} local_scorable={len(keep)} nonempty={ne} "
      f"skipped_nonlocal={len(gym)-len(keep)}", file=sys.stderr)
PYEOF

GYM_IDS=$(tr '\n' ' ' < "$IDS_FILE")
NGYM=$(wc -w <<<"$GYM_IDS")
if [[ "$NGYM" -eq 0 ]]; then echo "[rescue-local] no local gym preds -> nothing to do" >&2; exit 0; fi

# 3) run the fork harness over the LOCAL gym predictions
RUN_ID="${BATCH}_rescue_${RUNTAG}"
echo "[rescue-local] fork harness over $NGYM ids $(date -u +%FT%TZ)" >&2
( cd "$RD/score/parts/fork" && timeout 21600 env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    "$FORK_PY" -m swebench.harness.run_evaluation \
    --dataset_name "/home/mark/qwen_diffusion/$DS_GYM" --split train \
    --predictions_path "/home/mark/qwen_diffusion/$PRED_GYM" \
    --instance_ids $GYM_IDS \
    --run_id "$RUN_ID" \
    --cache_level instance --clean False --max_workers "$MAXW" --timeout 1800 ) 2>&1 | tail -15

FORK_REP=$(ls "$RD"/score/parts/fork/*."${RUN_ID}".json 2>/dev/null | grep -v timing | head -1)
echo "[rescue-local] report=$FORK_REP" >&2
echo "==== RESCUE-LOCAL $BATCH DONE $(date -u +%FT%TZ) ====" >&2
