#!/usr/bin/env bash
# Pull one prebuilt SWE instance image and re-tag it to the local keys the
# toolchain expects. DUAL-SOURCE (belt-lever 2026-07-07):
#
#   * swe_gym (default): pull xingyaoww/sweb.eval.x86_64.<slug_s> and tag to BOTH
#       - fork scorer : sweb.eval.x86_64.<instance_id>            (fork test_spec key)
#       - qwen driver : swebench/sweb.eval.x86_64.<slug_1776>     (driver + rmi key)
#   * swe_verified: the OFFICIAL image swebench/sweb.eval.x86_64.<slug_1776> IS the
#     driver key AND the official-harness instance-image key (namespace=swebench),
#     so we pull it directly. We ALSO tag the fork key (harmless) so rmi + any
#     fork path find it uniformly.
#
# Appends a JSONL row: {instance_id, slug_s, src, pull_s, size_bytes, status, source}.
#   $1 = instance_id (e.g. facebookresearch__hydra-1006 | django__django-16263)
#   $2 = out jsonl (appended)
#   $3 = source: swe_gym (default) | swe_verified
set -uo pipefail
IID="${1:?instance_id}"
OUT="${2:?out jsonl}"
SOURCE="${3:-swe_gym}"
export SUDO_ASKPASS="${SUDO_ASKPASS:-}"
D="${SWE_DOCKER_CMD:-docker}"   # docker-group host: plain docker (override to 'sudo -A docker' where absent)

slug_s="${IID/__/_s_}"
slug_1776="${IID/__/_1776_}"
FORK_KEY="sweb.eval.x86_64.${IID}:latest"
DRV_KEY="swebench/sweb.eval.x86_64.${slug_1776}:latest"
if [[ "$SOURCE" == "swe_verified" ]]; then
  SRC="$DRV_KEY"                                    # official image == driver key
else
  SRC="xingyaoww/sweb.eval.x86_64.${slug_s}:latest"  # SWE-Gym prebuilt
fi

emit() {  # status pull_s size
  python3 - "$IID" "$slug_s" "$SRC" "$2" "$3" "$1" "$OUT" "$SOURCE" <<'PY'
import json,sys
iid,slug,src,pull_s,size,status,out,source=sys.argv[1:9]
with open(out,"a") as f:
    f.write(json.dumps({"instance_id":iid,"slug_s":slug,"src":src,
        "pull_s":float(pull_s),"size_bytes":int(size),"status":status,
        "source":source})+"\n")
PY
}

# Already have the driver key (both sources ultimately need it) -> reuse.
if $D image inspect "$DRV_KEY" >/dev/null 2>&1; then
  sz=$($D image inspect "$DRV_KEY" --format '{{.Size}}' 2>/dev/null || echo 0)
  $D tag "$DRV_KEY" "$FORK_KEY" 2>/dev/null || true
  emit cached 0 "$sz"; echo "[pull] $IID CACHED size=$sz source=$SOURCE" >&2; exit 0
fi

t0=$(date +%s.%N)
if ! $D pull "$SRC" >&2 2>&1; then
  emit pull_failed 0 0; echo "[pull] $IID PULL_FAILED src=$SRC source=$SOURCE" >&2; exit 1
fi
pull_s=$(python3 -c "import sys;print(f'{$(date +%s.%N)-$t0:.1f}')")
sz=$($D image inspect "$SRC" --format '{{.Size}}' 2>/dev/null || echo 0)
# Ensure BOTH keys exist regardless of source ($SRC may already be one of them).
$D tag "$SRC" "$FORK_KEY" 2>/dev/null || true
$D tag "$SRC" "$DRV_KEY"  2>/dev/null || true
emit ok "$pull_s" "$sz"; echo "[pull] $IID OK pull_s=$pull_s size=$sz source=$SOURCE" >&2
