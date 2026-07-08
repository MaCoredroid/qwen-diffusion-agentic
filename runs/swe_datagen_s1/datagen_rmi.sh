#!/usr/bin/env bash
# BATCH RMI — reclaim disk after a batch is scored + keepers extracted. Removes
# the THREE tags pull_and_tag.sh created for each instance (xingyaoww source,
# fork key, driver key). Since the 3 tags share one image id, the underlying
# layers free once the last tag is removed. Keeper trajectories are already
# persisted to keepers/ (text) — the image is disposable. Reports freed GB.
#   usage: datagen_rmi.sh <batchdir>
set -uo pipefail
cd /home/mark/qwen_diffusion
export SUDO_ASKPASS="${SUDO_ASKPASS:-}"
D="${SWE_DOCKER_CMD:-docker}"   # docker-group host: plain docker (override to 'sudo -A docker' where absent)
BATCHDIR="${1:?batchdir}"

df_avail_gb() { df -B1 --output=avail /home/mark | tail -1 | awk '{printf "%.0f", $1/1e9}'; }
before=$(df_avail_gb)

IDS=$(.venv/bin/python -c 'import json,sys;print(" ".join(json.load(open(sys.argv[1]))["instance_ids"]))' "$BATCHDIR/subset.json")
n=0
echo "==== BATCH RMI START $(date -u +%FT%TZ) avail=${before}GB ====" >&2
for iid in $IDS; do
  slug_s="${iid/__/_s_}"; slug_1776="${iid/__/_1776_}"
  for tag in "xingyaoww/sweb.eval.x86_64.${slug_s}:latest" \
             "sweb.eval.x86_64.${iid}:latest" \
             "swebench/sweb.eval.x86_64.${slug_1776}:latest"; do
    $D rmi -f "$tag" >/dev/null 2>&1 && n=$((n+1)) || true
  done
done
after=$(df_avail_gb)
echo "==== BATCH RMI DONE $(date -u +%FT%TZ) removed_tags=$n freed=$((after-before))GB avail=${after}GB ====" >&2
echo "{\"removed_tags\":$n,\"freed_gb\":$((after-before)),\"avail_gb_after\":$after}" > "$BATCHDIR/rmi.json"
