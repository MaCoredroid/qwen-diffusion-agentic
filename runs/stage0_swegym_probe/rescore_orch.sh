#!/usr/bin/env bash
set -uo pipefail
cd /home/mark/qwen_diffusion
RUN=runs/stage0_swegym_probe
export SUDO_ASKPASS="${SUDO_ASKPASS:?}"
LOG=$RUN/logs/rescore.log
exec >>"$LOG" 2>&1
echo "######## RESCORE START $(date -u +%FT%TZ) ########"
MAXW=2 RUN_ID=probe20 bash $RUN/probe_score.sh
echo "[rescore] score rc=$?"
RUN_ID=probe20 MAX_NUM_SEQS=4 .venv/bin/python $RUN/build_report.py
echo "[rescore] report rc=$?"
echo "######## RESCORE DONE $(date -u +%FT%TZ) ########"
