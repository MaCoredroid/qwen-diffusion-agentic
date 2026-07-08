#!/usr/bin/env bash
# RESCUE re-score of discarded SWE-Gym episodes (2026-07-07 run) — NO new generation.
# Re-runs the SWE-Bench-Fork harness over the EXISTING on-disk gym predictions using a
# dataset_gym.json rebuilt with the FIXED build_batch_dataset.py (env_setup_commit
# dropped when None). Writes reports under rescue_20260708/<batch>/score/. Does NOT
# touch attempts.jsonl / frontier.json / keepers. Mirrors datagen_score.sh gym path.
#   usage: rescue_score.sh <batch_dirname> [max_workers]
set -uo pipefail
cd /home/mark/qwen_diffusion
BATCH="${1:?batch dirname}"; MAXW="${2:-6}"
HERE=runs/swe_datagen_s1
BR="$HERE/batches/$BATCH"
RD="$HERE/rescue_20260708/$BATCH"
PY=/home/mark/qwen_diffusion/.venv/bin/python
FORK_PY=/home/mark/qwen_diffusion/runs/stage0_swegym_probe/.venv-swegym/bin/python
D=docker
SOURCES="$BR/sources.json"
DS_GYM="$RD/dataset_gym.json"          # corrected (already built)
mkdir -p "$RD/score/parts/fork"
PRED="$RD/score/all_predictions.jsonl"
PRED_GYM="$RD/score/pred_gym.jsonl"
IDS_FILE="$RD/score/gym_ids.txt"
IMGLOG="$RD/image_ensure.log"
: > "$IMGLOG"

echo "==== RESCUE $BATCH START $(date -u +%FT%TZ) maxw=$MAXW ====" >&2

# 1) aggregate predictions from gen shards (one row per instance_id) -------------
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
print(f"[rescue] aggregated {len(rows)} predictions", file=sys.stderr)
PYEOF

# 2) filter to gym preds + emit gym id list -------------------------------------
$PY - "$PRED" "$SOURCES" "$PRED_GYM" "$IDS_FILE" <<'PYEOF'
import json,sys
pred,sources,outg,idsf=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]
try: src=json.load(open(sources))
except Exception: src={}
rows=[json.loads(l) for l in open(pred) if l.strip()]
gym=[r for r in rows if src.get(r["instance_id"],"swe_gym")=="swe_gym"]
with open(outg,"w") as g:
    for r in gym: g.write(json.dumps(r)+"\n")
open(idsf,"w").write("\n".join(r["instance_id"] for r in gym)+"\n")
ne=sum(1 for r in gym if (r.get("model_patch") or "").strip())
print(f"[rescue] gym preds={len(gym)} nonempty={ne}", file=sys.stderr)
PYEOF

GYM_IDS=$(tr '\n' ' ' < "$IDS_FILE")
NGYM=$(wc -w <<<"$GYM_IDS")
if [[ "$NGYM" -eq 0 ]]; then echo "[rescue] no gym preds -> nothing to do" >&2; exit 0; fi

# 3) ensure the fork instance image for each gym id (retag driver key, else pull) -
for iid in $GYM_IDS; do
  FORK_KEY="sweb.eval.x86_64.${iid}:latest"
  slug_1776="${iid/__/_1776_}"; DRV_KEY="swebench/sweb.eval.x86_64.${slug_1776}:latest"
  slug_s="${iid/__/_s_}"; XSRC="xingyaoww/sweb.eval.x86_64.${slug_s}:latest"
  if $D image inspect "$FORK_KEY" >/dev/null 2>&1; then
    echo "have_fork $iid" >> "$IMGLOG"
  elif $D image inspect "$DRV_KEY" >/dev/null 2>&1; then
    $D tag "$DRV_KEY" "$FORK_KEY" && echo "retag_drv $iid" >> "$IMGLOG"
  else
    if timeout 900 $D pull "$XSRC" >/dev/null 2>&1; then
      $D tag "$XSRC" "$FORK_KEY"; echo "pulled $iid" >> "$IMGLOG"
    else
      echo "PULL_FAILED $iid" >> "$IMGLOG"
    fi
  fi
done
echo "[rescue] image ensure: $(sort "$IMGLOG" | awk '{print $1}' | sort | uniq -c | tr '\n' ' ')" >&2

# 4) run the fork harness over the EXISTING gym predictions ----------------------
RUN_ID="${BATCH}_rescue"
echo "[rescue] fork harness over $NGYM ids $(date -u +%FT%TZ)" >&2
( cd "$RD/score/parts/fork" && timeout 21600 env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    "$FORK_PY" -m swebench.harness.run_evaluation \
    --dataset_name "/home/mark/qwen_diffusion/$DS_GYM" --split train \
    --predictions_path "/home/mark/qwen_diffusion/$PRED_GYM" \
    --instance_ids $GYM_IDS \
    --run_id "$RUN_ID" \
    --cache_level instance --clean False --max_workers "$MAXW" --timeout 1800 ) 2>&1 | tail -20

# 5) collect the fork sub-report + write a compact per-batch summary -------------
FORK_REP=$(ls "$RD"/score/parts/fork/*."${RUN_ID}".json 2>/dev/null | grep -v timing | head -1)
$PY - "$BATCH" "$PRED_GYM" "${FORK_REP:-}" "$IMGLOG" "$RD/summary.json" <<'PYEOF'
import json,sys
batch,pred_gym,rep,imglog,out=sys.argv[1:6]
gym=[json.loads(l) for l in open(pred_gym) if l.strip()]
nonempty={r["instance_id"] for r in gym if (r.get("model_patch") or "").strip()}
empty_pred={r["instance_id"] for r in gym} - nonempty
img={}
for l in open(imglog):
    p=l.split()
    if len(p)==2: img[p[1]]=p[0]
pull_failed=[i for i,s in img.items() if s=="PULL_FAILED"]
res={"resolved_ids":[],"unresolved_ids":[],"empty_patch_ids":[],"error_ids":[]}
if rep:
    d=json.load(open(rep))
    for k in res: res[k]=sorted(set(d.get(k,[]) or []))
scored=set(sum(res.values(),[]))
unscorable=[i for i in nonempty if i not in scored]
summ={
 "batch_id":batch,
 "gym_preds_total":len(gym),
 "gym_preds_nonempty":len(nonempty),
 "gym_preds_empty_on_disk":len(empty_pred),
 "report":rep or None,
 "resolved":len(res["resolved_ids"]),
 "unresolved":len(res["unresolved_ids"]),
 "empty_patch":len(res["empty_patch_ids"]),
 "error":len(res["error_ids"]),
 "resolved_ids":res["resolved_ids"],
 "unresolved_ids":res["unresolved_ids"],
 "empty_patch_ids":res["empty_patch_ids"],
 "error_ids":res["error_ids"],
 "pull_failed_ids":pull_failed,
 "nonempty_unscored_ids":unscorable,
}
json.dump(summ,open(out,"w"),indent=1)
print(f"[rescue] {batch}: resolved={summ['resolved']} unresolved={summ['unresolved']} "
      f"empty={summ['empty_patch']} error={summ['error']} "
      f"pull_failed={len(pull_failed)} nonempty_unscored={len(unscorable)}")
PYEOF
echo "==== RESCUE $BATCH DONE $(date -u +%FT%TZ) ====" >&2
