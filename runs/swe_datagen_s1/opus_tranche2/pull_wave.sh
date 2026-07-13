#!/usr/bin/env bash
# Throttled, budget-bounded, DUAL-SOURCE image pull for the Opus tranche-2. Unlike
# tranche-1 (pure SWE-Gym, hardcoded swe_gym source), tranche-2 mixes 78 SWE-Gym +
# 38 Verified ids, so each pull is routed to its per-id source from batch/sources.json
# (swe_gym -> xingyaoww/...; swe_verified -> official swebench/... which IS the driver
# key). Both sources land the SAME driver key swebench/sweb.eval.x86_64.<slug_1776>,
# so the present-check (inspect driver key) and idempotent skip work uniformly.
#
# WAVE-BOUNDED: pulls only the FIRST WAVE_N ids of target_ids.txt (wave-1 ~30). Later
# waves (the rest) are pulled between gen chunks, not here. CONCURRENCY <=3. Disk-floor
# guarded (abort under PULL_DISK_FLOOR_GB). Wall-budget bounded to fit the 600s tool cap.
# Idempotent: a present driver key is reused instantly. Records rows to opus_tranche2/pull.jsonl.
#   usage: pull_wave.sh [wave_n] [budget_s] [max_jobs]
set -uo pipefail
cd /home/mark/qwen_diffusion
export SWE_DOCKER_CMD="${SWE_DOCKER_CMD:-docker}"
export SUDO_ASKPASS="${SUDO_ASKPASS:-}"
HERE=runs/swe_datagen_s1
T=$HERE/opus_tranche2
SRCMAP="$T/batch/sources.json"
WAVE_N="${1:-30}"
BUDGET="${2:-560}"
MAXJOBS="${3:-3}"
PULL_DISK_FLOOR_GB="${PULL_DISK_FLOOR_GB:-300}"
OUT="$T/pull.jsonl"; touch "$OUT"
t0=$(date +%s)

df_avail_gb() { df -B1 --output=avail /home/mark | tail -1 | awk '{printf "%.0f", $1/1e9}'; }
src_of() { .venv/bin/python -c '
import json,sys
try: print(json.load(open(sys.argv[1])).get(sys.argv[2],"swe_gym"))
except Exception: print("swe_gym")' "$SRCMAP" "$1"; }
drv_present() { local iid="$1"; local slug="${iid/__/_1776_}"
  docker image inspect "swebench/sweb.eval.x86_64.${slug}:latest" >/dev/null 2>&1; }

# wave-1 id list = first WAVE_N non-empty lines
mapfile -t WAVE < <(sed '/^$/d' "$T/target_ids.txt" | sed -n "1,${WAVE_N}p")

echo "==== PULLWAVE-2 START $(date -u +%FT%TZ) wave_n=${#WAVE[@]} budget=${BUDGET}s maxjobs=$MAXJOBS floor=${PULL_DISK_FLOOR_GB}GB avail=$(df_avail_gb)GB ====" >&2
present=0; launched=0; skipped_budget=0
for iid in "${WAVE[@]}"; do
  [ -z "$iid" ] && continue
  if drv_present "$iid"; then present=$((present+1)); continue; fi
  # disk floor guard
  avail=$(df_avail_gb)
  if [ "$avail" -lt "$PULL_DISK_FLOOR_GB" ]; then
    echo "[pullwave] DISK FLOOR HIT avail=${avail}GB < ${PULL_DISK_FLOOR_GB}GB -> stop" >&2; break; fi
  # wall budget guard (leave margin so in-flight pulls can drain under the cap)
  el=$(( $(date +%s) - t0 ))
  if [ "$el" -ge "$BUDGET" ]; then skipped_budget=$((skipped_budget+1)); continue; fi
  # throttle to <=MAXJOBS concurrent
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do wait -n 2>/dev/null || break; done
  SRC=$(src_of "$iid")
  echo "[pullwave] $iid source=$SRC launch el=${el}s avail=${avail}GB $(date -u +%FT%TZ)" >&2
  ( timeout 900 bash "$HERE/pull_and_tag.sh" "$iid" "$OUT" "$SRC" \
      || echo "[pullwave] $iid FAILED (env_unavailable)" >&2 ) &
  launched=$((launched+1))
done
# drain
wait
# recount wave-1 presence
missing=0; wpresent=0
for iid in "${WAVE[@]}"; do
  [ -z "$iid" ] && continue
  if drv_present "$iid"; then wpresent=$((wpresent+1)); else missing=$((missing+1)); fi
done
echo "PULLWAVE-2 done wall=$(( $(date +%s)-t0 ))s wave_n=${#WAVE[@]} present_now=$wpresent launched_this_call=$launched budget_deferred=$skipped_budget remaining_missing=$missing avail=$(df_avail_gb)GB"
