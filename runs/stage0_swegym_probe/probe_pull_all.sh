#!/usr/bin/env bash
# Stage 0 phase 2 PROBE — PULL stage. Serially pull+tag all 20 prebuilt SWE-Gym
# instance images (env acquisition). NO GPU server up (docker-heavy, serialized
# per RAM discipline). Records per-instance pull_s + size to pull/pull.jsonl.
set -uo pipefail
cd /home/mark/qwen_diffusion
RUN=runs/stage0_swegym_probe
export SUDO_ASKPASS="${SUDO_ASKPASS:?export SUDO_ASKPASS}"
OUT=$RUN/pull/pull.jsonl
: > "$OUT"
IDS=$(.venv/bin/python -c "import json;print(' '.join(json.load(open('$RUN/artifacts/subset_probe20.json'))['instance_ids']))")
echo "==== PROBE PULL START $(date -u +%FT%TZ) ====" >&2
i=0
for iid in $IDS; do
  i=$((i+1))
  echo "[pull $i/20] $iid $(date -u +%FT%TZ)" >&2
  timeout 1200 bash "$RUN/pull_and_tag.sh" "$iid" "$OUT" || echo "[pull] $iid FAILED (continuing)" >&2
  df -h /home/mark | tail -1 >&2
done
echo "==== PROBE PULL DONE $(date -u +%FT%TZ) rows=$(wc -l < "$OUT") ====" >&2
