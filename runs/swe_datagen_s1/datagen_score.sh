#!/usr/bin/env bash
# BATCH SCORE — DUAL-SOURCE verifiable reward for one batch (docker-only; NO GPU up).
# Merge the C shard predictions into one predictions.jsonl, then score each id with
# the harness that MATCHES its image provenance (belt-lever 2026-07-07):
#   * swe_gym ids     -> SWE-Bench-Fork 2.0.13 (@242429c + fork_reuse_prebuilt.patch),
#                        --cache_level instance, over dataset_gym.json. (unchanged)
#   * swe_verified ids -> OFFICIAL swebench 4.1.0 (the W2-proven path; namespace
#                        `swebench` == the pulled swebench/sweb.eval.x86_64.<slug_1776>
#                        images), --cache_level env, over dataset_verified.json.
# Both harnesses reuse the prebuilt instance images pulled this cycle (not yet rmi'd)
# and emit the SAME report schema (resolved_ids/unresolved_ids/empty_patch_ids/
# error_ids). We write each sub-report under score/parts/{fork,official}/ and MERGE
# them into the single canonical score/<eval>.<run_id>.json the ledger reads
# (ledger._load_report globs score/*.json NON-recursively, so parts/ is invisible).
#
# BACKWARD-COMPAT: no sources.json (a pre-belt batch, e.g. an in-flight cycle) ->
# every predicted id is treated as swe_gym and dataset_gym.json falls back to
# dataset.json, reproducing the original single-harness behavior byte-for-byte.
#   usage: datagen_score.sh <batchdir> <gen_root> [max_workers]
set -uo pipefail
cd /home/mark/qwen_diffusion
export SUDO_ASKPASS="${SUDO_ASKPASS:-}"
export SWE_DOCKER_CMD="${SWE_DOCKER_CMD:-docker}"
# Privilege wrapper for the harness (reaches docker via the docker group -> NO sudo).
# The swebench harness talks to /var/run/docker.sock directly; a docker-group user
# runs it fine and its report json is written by THIS process (my user) -> readable.
# Override to 'sudo -A' only on a host lacking the group. Empty default = run as me.
PRIV="${DATAGEN_PRIV:-}"
V1=runs/stage0_swegym_probe
FORK_PY="$V1/.venv-swegym/bin/python"                        # SWE-Bench-Fork (SWE-Gym)
OFFICIAL_PY=runs/stage_c_n5/local_eval/.venv-harness/bin/python  # swebench 4.1.0 (Verified)
PY=.venv/bin/python

BATCHDIR="${1:?batchdir}"; GEN_ROOT="${2:?gen_root}"; MAXW="${3:-4}"
DS="$BATCHDIR/dataset.json"
DS_GYM="$BATCHDIR/dataset_gym.json"; [[ -s "$DS_GYM" ]] || DS_GYM="$DS"
DS_VER="$BATCHDIR/dataset_verified.json"
SOURCES="$BATCHDIR/sources.json"
RUN_ID="$(basename "$BATCHDIR")"
SCORE="$BATCHDIR/score"; mkdir -p "$SCORE" "$SCORE/parts/fork" "$SCORE/parts/official"
PRED="$SCORE/all_predictions.jsonl"

# 1) aggregate predictions from all shards (one row per instance_id) ------------
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

NPRED=$(wc -l < "$PRED" 2>/dev/null || echo 0)
echo "==== BATCH SCORE START $(date -u +%FT%TZ) preds=$NPRED maxw=$MAXW rid=$RUN_ID ====" >&2
[[ "$NPRED" -eq 0 ]] && { echo "[score] no predictions to score" >&2; exit 1; }

# 2) partition predicted ids by source (sources.json absent -> all swe_gym) ------
GYM_IDS=$($PY - "$PRED" "$SOURCES" swe_gym <<'PY'
import json,sys
pred, sources, want = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    src = json.load(open(sources))
except Exception:
    src = {}
ids=[json.loads(l)["instance_id"] for l in open(pred) if l.strip()]
print(" ".join(i for i in ids if src.get(i, "swe_gym") == want))
PY
)
VER_IDS=$($PY - "$PRED" "$SOURCES" swe_verified <<'PY'
import json,sys
pred, sources, want = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    src = json.load(open(sources))
except Exception:
    src = {}
ids=[json.loads(l)["instance_id"] for l in open(pred) if l.strip()]
print(" ".join(i for i in ids if src.get(i, "swe_gym") == want))
PY
)
NGYM=$(wc -w <<<"$GYM_IDS"); NVER=$(wc -w <<<"$VER_IDS")
echo "[score] partition: gym=$NGYM verified=$NVER $(date -u +%FT%TZ)" >&2

# 2b) SOURCE-FILTERED prediction files — one per harness. This is the dual-source
# fix (2026-07-07): swebench's get_dataset_from_preds validates that EVERY id in the
# predictions file is present in the (single-source) dataset BEFORE it applies the
# --instance_ids/-i filter (official run_evaluation.py get_dataset_from_preds; fork
# ditto). So handing the merged all_predictions.jsonl (which straddles BOTH sources)
# to a single-source harness makes it raise "Some prediction IDs not found in
# dataset!" and abort before running a single container -> no sub-report -> every id
# falls through to no_prediction in the ledger. We must feed each harness ONLY its
# own source's predictions. sources.json absent -> every id is swe_gym (pred_gym ==
# all_predictions, pred_ver empty), reproducing the pre-belt single-harness path.
PRED_GYM="$SCORE/pred_gym.jsonl"; PRED_VER="$SCORE/pred_ver.jsonl"
$PY - "$PRED" "$SOURCES" "$PRED_GYM" "$PRED_VER" <<'PY'
import json, sys
pred, sources, out_gym, out_ver = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
try:
    src = json.load(open(sources))
except Exception:
    src = {}
rows = [json.loads(l) for l in open(pred) if l.strip()]
with open(out_gym, "w") as g, open(out_ver, "w") as v:
    for r in rows:
        (v if src.get(r["instance_id"], "swe_gym") == "swe_verified" else g).write(
            json.dumps(r) + "\n")
print(f"[score] filtered predictions: gym={sum(1 for r in rows if src.get(r['instance_id'],'swe_gym')!='swe_verified')} "
      f"verified={sum(1 for r in rows if src.get(r['instance_id'],'swe_gym')=='swe_verified')}", file=sys.stderr)
PY

t0=$(date +%s)

# 3) SWE-Gym ids -> fork harness (dataset_gym.json), report to parts/fork ---------
if [[ "$NGYM" -gt 0 ]]; then
  echo "[score:gym] fork harness over $NGYM ids $(date -u +%FT%TZ)" >&2
  ( cd "$SCORE/parts/fork" && timeout 21600 $PRIV env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
      "/home/mark/qwen_diffusion/$FORK_PY" -m swebench.harness.run_evaluation \
      --dataset_name "/home/mark/qwen_diffusion/$DS_GYM" --split train \
      --predictions_path "/home/mark/qwen_diffusion/$PRED_GYM" \
      --instance_ids $GYM_IDS \
      --run_id "${RUN_ID}_gym" \
      --cache_level instance --clean False --max_workers "$MAXW" --timeout 1800 ) 2>&1 | tail -15
fi

# 4) Verified ids -> OFFICIAL harness (dataset_verified.json), report to parts/official
if [[ "$NVER" -gt 0 && -s "$DS_VER" ]]; then
  echo "[score:verified] official harness over $NVER ids $(date -u +%FT%TZ)" >&2
  ( cd "$SCORE/parts/official" && timeout 21600 $PRIV env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
      "/home/mark/qwen_diffusion/$OFFICIAL_PY" -m swebench.harness.run_evaluation \
      -d "/home/mark/qwen_diffusion/$DS_VER" -s test \
      -p "/home/mark/qwen_diffusion/$PRED_VER" -id "${RUN_ID}_ver" \
      -i $VER_IDS -n swebench \
      --cache_level env --clean False --max_workers "$MAXW" --timeout 1800 \
      --report_dir "/home/mark/qwen_diffusion/$SCORE/parts/official" ) 2>&1 | tail -15
fi
t1=$(date +%s)
$PRIV chown -R "$(id -u):$(id -g)" "$SCORE" 2>/dev/null || true

# 5) MERGE the two sub-reports into the canonical report the ledger reads --------
FORK_REP=$(ls "$SCORE"/parts/fork/*."${RUN_ID}_gym".json 2>/dev/null | grep -v timing | head -1)
VER_REP=$(ls "$SCORE"/parts/official/*."${RUN_ID}_ver".json 2>/dev/null | grep -v timing | head -1)
MERGED="$SCORE/datagen-eval.${RUN_ID}.json"
$PY - "$MERGED" "${FORK_REP:-}" "${VER_REP:-}" <<'PY'
import json, sys
out = sys.argv[1]
parts = [p for p in sys.argv[2:] if p]
LISTS = ["resolved_ids", "unresolved_ids", "empty_patch_ids", "error_ids",
         "completed_ids", "incomplete_ids", "submitted_ids"]
COUNTS = ["total_instances", "submitted_instances", "completed_instances",
          "resolved_instances", "unresolved_instances", "empty_patch_instances",
          "error_instances"]
agg = {k: [] for k in LISTS}
cnt = {k: 0 for k in COUNTS}
srcs = []
for p in parts:
    try:
        d = json.load(open(p))
    except Exception as e:
        print(f"[merge] skip {p}: {e}", file=sys.stderr); continue
    srcs.append(p)
    for k in LISTS:
        agg[k].extend(d.get(k, []) or [])
    for k in COUNTS:
        v = d.get(k)
        if isinstance(v, int):
            cnt[k] += v
rep = {k: sorted(set(agg[k])) for k in LISTS}
rep.update(cnt)
rep["_merged_from"] = srcs
rep["schema_version"] = "datagen_dual_merge_v1"
json.dump(rep, open(out, "w"), indent=1)
print(f"[merge] resolved={len(rep['resolved_ids'])} unresolved={len(rep['unresolved_ids'])} "
      f"empty={len(rep['empty_patch_ids'])} error={len(rep['error_ids'])} "
      f"from {len(srcs)} sub-report(s)")
PY

# 5b) observability guard: a whole-harness failure (report absent) would let the
# affected ids fall through to no_prediction in the ledger. That is sub-kill and
# self-heals (best-of-k re-draws them next cycle), but must be VISIBLE in the log.
if [[ "$NGYM" -gt 0 && -z "${FORK_REP:-}" ]]; then
  echo "[score] WARN: fork harness produced NO report for $NGYM gym ids -> they will "\
"record as no_prediction (re-drawable under best-of-k). Check parts/fork/." >&2
fi
if [[ "$NVER" -gt 0 && -z "${VER_REP:-}" ]]; then
  echo "[score] WARN: OFFICIAL harness produced NO report for $NVER verified ids -> they "\
"will record as no_prediction (re-drawable under best-of-k). Check parts/official/." >&2
fi

# 6) timing + one-line summary --------------------------------------------------
echo "{\"score_wall_s\": $((t1-t0)), \"report\": \"${MERGED}\", \"maxw\": $MAXW, \"n_gym\": $NGYM, \"n_verified\": $NVER}" > "$SCORE/timing.json"
echo "[score] report=$MERGED wall=$((t1-t0))s (gym=$NGYM ver=$NVER)" >&2
$PY -c "import sys,json;d=json.load(open(sys.argv[1]));print('resolved',len(d.get('resolved_ids',[])),'unresolved',len(d.get('unresolved_ids',[])),'empty',len(d.get('empty_patch_ids',[])),'errors',len(d.get('error_ids',[])))" "$MERGED" 2>/dev/null || true
echo "==== BATCH SCORE DONE $(date -u +%FT%TZ) ====" >&2
