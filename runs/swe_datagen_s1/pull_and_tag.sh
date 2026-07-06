#!/usr/bin/env bash
# Pull one prebuilt SWE-Gym instance image from xingyaoww and re-tag it to BOTH
# local keys the toolchain expects:
#   * fork scorer  : sweb.eval.x86_64.<instance_id>:latest        (SWE-Bench-Fork test_spec.instance_image_key)
#   * qwen driver  : swebench/sweb.eval.x86_64.<slug_1776>:latest (run_swe_bench_qwen_code._container_image_for)
# Appends a JSONL row: {instance_id, slug_s, src, pull_s, size_bytes, status}.
#   $1 = instance_id (e.g. facebookresearch__hydra-1006)
#   $2 = out jsonl (appended)
set -uo pipefail
IID="${1:?instance_id}"
OUT="${2:?out jsonl}"
export SUDO_ASKPASS="${SUDO_ASKPASS:?export SUDO_ASKPASS}"
D="sudo -A docker"

slug_s="${IID/__/_s_}"
slug_1776="${IID/__/_1776_}"
SRC="xingyaoww/sweb.eval.x86_64.${slug_s}:latest"
FORK_KEY="sweb.eval.x86_64.${IID}:latest"
DRV_KEY="swebench/sweb.eval.x86_64.${slug_1776}:latest"

emit() {  # status pull_s size
  python3 - "$IID" "$slug_s" "$SRC" "$2" "$3" "$1" "$OUT" <<'PY'
import json,sys
iid,slug,src,pull_s,size,status,out=sys.argv[1:8]
with open(out,"a") as f:
    f.write(json.dumps({"instance_id":iid,"slug_s":slug,"src":src,
        "pull_s":float(pull_s),"size_bytes":int(size),"status":status})+"\n")
PY
}

if $D image inspect "$FORK_KEY" >/dev/null 2>&1; then
  sz=$($D image inspect "$FORK_KEY" --format '{{.Size}}' 2>/dev/null || echo 0)
  $D tag "$FORK_KEY" "$DRV_KEY" 2>/dev/null || true
  emit cached 0 "$sz"; echo "[pull] $IID CACHED size=$sz" >&2; exit 0
fi

t0=$(date +%s.%N)
if ! $D pull "$SRC" >&2 2>&1; then
  emit pull_failed 0 0; echo "[pull] $IID PULL_FAILED" >&2; exit 1
fi
pull_s=$(python3 -c "import sys;print(f'{$(date +%s.%N)-$t0:.1f}')")
sz=$($D image inspect "$SRC" --format '{{.Size}}' 2>/dev/null || echo 0)
$D tag "$SRC" "$FORK_KEY"
$D tag "$SRC" "$DRV_KEY"
emit ok "$pull_s" "$sz"; echo "[pull] $IID OK pull_s=$pull_s size=$sz" >&2
