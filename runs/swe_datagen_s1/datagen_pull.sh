#!/usr/bin/env bash
# BATCH PULL — env acquisition for one batch (docker-only; NO GPU server up).
# Serially pull+tag every instance image in <batchdir>/subset.json via the reused
# SWE-Gym pull_and_tag.sh (xingyaoww source -> fork key + driver key). Disk-floor
# guarded: abort the batch if df avail drops under the floor (the orchestrator
# will halt + flag). Records per-instance rows to <batchdir>/pull.jsonl.
#   usage: datagen_pull.sh <batchdir>
set -uo pipefail
cd /home/mark/qwen_diffusion
export SUDO_ASKPASS="${SUDO_ASKPASS:-}"
export SWE_DOCKER_CMD="${SWE_DOCKER_CMD:-docker}"   # docker-group host: plain docker
HERE=runs/swe_datagen_s1
BATCHDIR="${1:?batchdir}"
OUT="$BATCHDIR/pull.jsonl"
PULL_DISK_FLOOR_GB="${PULL_DISK_FLOOR_GB:-300}"
: > "$OUT"

df_avail_gb() { df -B1 --output=avail /home/mark | tail -1 | awk '{printf "%.0f", $1/1e9}'; }

IDS=$(.venv/bin/python -c 'import json,sys;print(" ".join(json.load(open(sys.argv[1]))["instance_ids"]))' "$BATCHDIR/subset.json")
N=$(wc -w <<<"$IDS")
# per-id source (swe_gym|swe_verified) from build_batch_dataset; absent -> swe_gym.
src_of() { .venv/bin/python -c '
import json,sys
p=sys.argv[1]; iid=sys.argv[2]
try:
    print(json.load(open(p)).get(iid,"swe_gym"))
except Exception:
    print("swe_gym")' "$BATCHDIR/sources.json" "$1"; }
echo "==== BATCH PULL START $(date -u +%FT%TZ) n=$N floor=${PULL_DISK_FLOOR_GB}GB ====" >&2
i=0; failed=0
for iid in $IDS; do
  i=$((i+1))
  avail=$(df_avail_gb)
  if [[ "$avail" -lt "$PULL_DISK_FLOOR_GB" ]]; then
    echo "[pull] DISK FLOOR HIT avail=${avail}GB < ${PULL_DISK_FLOOR_GB}GB -> abort batch" >&2
    echo "{\"aborted\":\"disk_floor\",\"avail_gb\":$avail}" >> "$OUT"
    echo "==== BATCH PULL ABORTED_DISK_FLOOR $(date -u +%FT%TZ) ====" >&2
    exit 7
  fi
  SRC=$(src_of "$iid")
  echo "[pull $i/$N] $iid source=$SRC avail=${avail}GB $(date -u +%FT%TZ)" >&2
  timeout 1800 bash "$HERE/pull_and_tag.sh" "$iid" "$OUT" "$SRC" \
    || { echo "[pull] $iid FAILED (env_unavailable; continuing)" >&2; failed=$((failed+1)); }
done
echo "==== BATCH PULL DONE $(date -u +%FT%TZ) rows=$(wc -l < "$OUT") failed=$failed avail=$(df_avail_gb)GB ====" >&2
exit 0
