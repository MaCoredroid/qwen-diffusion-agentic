#!/usr/bin/env bash
# Background infra: pull+tag the 6 pilot images missing the driver key. Bounded
# per-image timeout; the IPv6-flaky pydantic-4882 goes LAST with a tight bound so
# its hang never blocks the others. Writes a PULL_MISSING_DONE marker on exit.
set -uo pipefail
cd /home/mark/qwen_diffusion
export SWE_DOCKER_CMD=docker
HERE=runs/swe_datagen_s1
PILOT=$HERE/pilot_opus
OUT="$PILOT/pull_missing.jsonl"; : > "$OUT"
MARK="$PILOT/PULL_MISSING_DONE"; rm -f "$MARK"
# id  per-image-timeout-seconds
run() { local iid="$1" to="$2"; echo "[pullmiss] $iid start (to=${to}s) $(date -u +%FT%TZ)" >&2
  timeout "$to" bash "$HERE/pull_and_tag.sh" "$iid" "$OUT" swe_gym
  echo "[pullmiss] $iid rc=$? $(date -u +%FT%TZ)" >&2; }
run pandas-dev__pandas-47475 480
run pandas-dev__pandas-47493 480
run getmoto__moto-4867 300
run getmoto__moto-4874 300
run dask__dask-10342 420
run pydantic__pydantic-4882 150
echo "PULL_MISSING_DONE $(date -u +%FT%TZ)" > "$MARK"
echo "[pullmiss] ALL DONE $(date -u +%FT%TZ)" >&2