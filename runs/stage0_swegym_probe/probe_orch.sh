#!/usr/bin/env bash
# Stage 0 phase 2 PROBE — master orchestrator. Chains, SERIALLY (RAM discipline:
# ONE heavy phase at a time; docker-heavy pull/score never overlap the GPU server):
#   A. PULL   (docker pull+tag 20 prebuilt SWE-Gym images; NO server)
#   B. GEN    (boot ONE stock-AR server @concurrency 4; 4 driver shards; sample GPU)
#   C. SCORE  (official SWE-Bench-Fork harness over 20 predictions; NO server)
#   D. REPORT (measured table + price)
# Each stage is self-bounded (per-instance timeouts / boot+settle deadlines + traps).
set -uo pipefail
cd /home/mark/qwen_diffusion
RUN=runs/stage0_swegym_probe
export SUDO_ASKPASS="${SUDO_ASKPASS:?export SUDO_ASKPASS}"
LOG=$RUN/logs/orch.log
exec >>"$LOG" 2>&1

echo "############ PROBE ORCH START $(date -u +%FT%TZ) ############"
free -g | awk 'NR<=2'; df -h /home/mark | tail -1

echo "======== [A] PULL $(date -u +%FT%TZ) ========"
bash $RUN/probe_pull_all.sh; echo "[orch] pull rc=$?"

echo "======== [B] GEN $(date -u +%FT%TZ) ========"
SHARDS="driver_shard0 driver_shard1 driver_shard2 driver_shard3" \
  MAX_NUM_SEQS=4 AGENT_WALL_S=900 QWEN_MAX_WALL=840s MAX_TURNS=50 SCOPE=stage0_probe_ar \
  bash $RUN/probe_gen.sh; echo "[orch] gen rc=$?"
sleep 5

echo "======== [C] SCORE $(date -u +%FT%TZ) ========"
MAXW=2 RUN_ID=probe20 bash $RUN/probe_score.sh; echo "[orch] score rc=$?"

echo "======== [D] REPORT $(date -u +%FT%TZ) ========"
RUN_ID=probe20 MAX_NUM_SEQS=4 .venv/bin/python $RUN/build_report.py; echo "[orch] report rc=$?"

echo "############ PROBE ORCH DONE $(date -u +%FT%TZ) ############"
