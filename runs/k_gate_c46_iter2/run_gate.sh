#!/usr/bin/env bash
# C46 RE-GATE ITERATION-2 (#129) — detached, self-bounded orchestrator.
# Mirrors the iteration-1 C46 entry gate (runs/k_gate_c46/launch.sh + launch_ar.sh),
# fused into ONE self-bounded runner. Iteration-2 deltas (ONLY these):
#   (a) diffusion arm = models/qwen3.5-9b-fastdllm-mswe2-S-twinK1-vllm-bf16 served FLARE
#       hybrid_clean K=1 (32768) WITH the CERTIFIED read-clamp shim active (cert 7ae55d4,
#       runs/k_gate_c46/proxy_readclamp.py, LUMO_PROXY_READCLAMP_LIMIT=100).
#   (b) AR arm    = models/qwen3.5-9b-fastdllm-mswe-S-iter2-vllm-bf16 AR-decoded (the SAME
#       fold the iteration-2 KILL-T1 anchor gate served) at the same frozen envelope.
#   (c) ctx-overflow truth-telling labels active (52ffcc2): report has a distinct
#       ctx_overflow_deaths bucket (kept OUT of clean_exit0 / empty_patches).
#   (d) output = runs/k_gate_c46_iter2/.
#
# SEQUENCE (ONE server at a time; caged via systemd-run --user --scope):
#   1. verify 48 official swebench images present, GPU idle preflight
#   2. twin+clamp diffusion arm (gmu 0.74 seqs 4) -> completion -> teardown + GPU settle
#   3. AR arm (gmu 0.85 seqs 4) -> completion -> teardown + GPU settle
#   4. OFFICIAL docker scoring both (servers DOWN)
#   5. gate reports: twin single-arm resolve@1 vs >=12/46 entry bar + ctx_overflow
#      accounting (build_report.py); McNemar twin-vs-AR paired (build_ar_paired_report.py)
#
# MEMORY-BUDGET RULE: the diffusion arm's gmu (0.74) is NEVER the AR arm's (0.85). The
# GDN align-cache lives OUTSIDE the KV pool; the iteration-1 gate's measured
# gmu/concurrency (twin 0.74/4, AR 0.85/4) is authoritative — do not copy across arms.
#
# [state] lines per arm (eps done / N, wall seconds) every 60s. STOP-file:
# runs/k_gate_c46_iter2/STOP (touch it to abort gracefully — server torn down, exit 9).
# Detach with:  setsid bash runs/k_gate_c46_iter2/run_gate.sh &   (pidfile gate.pid)
# Docker via the docker group (plain `docker`, no sudo/askpass on this host).
set -uo pipefail
cd /home/mark/qwen_diffusion
export SWE_DOCKER_CMD="docker"

ROOT=runs/k_gate_c46_iter2
PLAN=runs/k_gate_c46/shard_plan.json          # SAME frozen 48-instance pool as iteration-1
IMAGES=data/swe_kraise_c46_pool/images.txt
C=4
GPU_CEIL=8000
N=48
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/run_gate.log"
exec >>"$LOG" 2>&1
PIDFILE="$ROOT/gate.pid"
echo $$ > "$PIDFILE"
STOP="$ROOT/STOP"

stamp(){ date -u +%FT%TZ; }
status(){ echo "$*" > "$ROOT/GATE_STATUS.txt"; echo "[gate] $*"; }
gpu_used()  { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
gpu_capps() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . ; }

ARM_PID=""; MON_PID=""
cleanup() {
  local rc=$?
  [[ -n "$MON_PID" ]] && kill "$MON_PID" 2>/dev/null || true
  [[ -n "$ARM_PID" ]] && kill "$ARM_PID" 2>/dev/null || true
  systemctl --user stop c46i2_diff_server.scope 2>/dev/null || true
  systemctl --user stop c46i2_ar_server.scope 2>/dev/null || true
  pkill -TERM -f 'qwen35_9b_flare_hybrid_serve|runcage_ar|EngineCore|vllm' 2>/dev/null || true
  local dl=$((SECONDS+90))
  while :; do local u; u=$(gpu_used); [[ "$u" -lt "$GPU_CEIL" ]] && break; [[ $SECONDS -gt $dl ]] && break; sleep 4; done
  rm -f "$PIDFILE"
  echo "[gate] EXIT rc=$rc $(stamp)"
}
trap cleanup EXIT

# episode-count monitor for a given arm subdir; emits [state] lines to the gate log.
monitor() {
  local arm=$1 t0=$2
  while :; do
    if [[ -f "$STOP" ]]; then echo "[gate] STOP-file seen in monitor (arm=$arm) $(stamp)"; break; fi
    local done wall
    done=$(find "$ROOT/$arm" -path '*/verified/per_task/*/runner_metadata.json' 2>/dev/null | wc -l)
    wall=$(( $(date +%s) - t0 ))
    echo "[state] arm=$arm eps=${done}/${N} wall=${wall}s $(stamp)"
    sleep 60
  done
}

run_arm() {
  # run_arm <arm_label> <runner.sh> ; sets ARM_PID/MON_PID, waits, tears the monitor down.
  local arm=$1 runner=$2 t0
  t0=$(date +%s)
  monitor "$arm" "$t0" & MON_PID=$!
  bash "$ROOT/$runner" "$C" "$ROOT" "$PLAN" & ARM_PID=$!
  wait "$ARM_PID"; local rc=$?; ARM_PID=""
  kill "$MON_PID" 2>/dev/null || true; MON_PID=""
  local done; done=$(find "$ROOT/$arm" -path '*/verified/per_task/*/runner_metadata.json' 2>/dev/null | wc -l)
  echo "[state] arm=$arm FINAL eps=${done}/${N} wall=$(( $(date +%s) - t0 ))s rc=$rc $(stamp)"
  return $rc
}

echo "==================== C46 RE-GATE ITER2 START $(stamp) ===================="

# 0. STOP pre-check
[[ -f "$STOP" ]] && { status "ABORT: STOP-file present at launch $(stamp)"; exit 9; }

# 1. image verification (all 48 must be present; pull is a separate prior step)
status "verifying 48 images present $(stamp)"
docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep 'sweb.eval' | sort -u > "$ROOT/logs/have_images.txt"
miss=0
while read -r img; do
  grep -qxF "$img" "$ROOT/logs/have_images.txt" || { echo "[gate] MISSING image: $img"; miss=$((miss+1)); }
done < "$IMAGES"
if [[ "$miss" -ne 0 ]]; then status "ABORT: $miss/48 images missing $(stamp)"; exit 1; fi
echo "[gate] all 48 images present $(stamp)"

# preflight: GPU idle by memory (gnome-shell baseline compute-app tolerated)
u=$(gpu_used)
if [[ "$u" -ge "$GPU_CEIL" ]]; then status "PREFLIGHT ABORT: GPU busy ${u}MiB $(stamp)"; exit 1; fi
echo "[gate] preflight clear ${u}MiB $(stamp)"

# 2. TWIN+CLAMP diffusion arm (server down + settled inside run_arm_twin.sh at exit)
[[ -f "$STOP" ]] && { status "STOP before twin arm $(stamp)"; exit 9; }
status "twin+clamp diffusion arm (mswe2-S-twinK1 iter2, clamp=100, gmu0.74 c=$C, cap 75) $(stamp)"
run_arm diffusion run_arm_twin.sh; TWIN_RC=$?
echo "[gate] twin arm rc=$TWIN_RC $(stamp)"
DPRED="$ROOT/diffusion/predictions.jsonl"
[[ ! -s "$DPRED" ]] && { status "ANOMALY twin predictions missing/empty $(stamp)"; exit 2; }
sleep 5

# 3. AR arm (server down + settled inside run_arm_ar.sh at exit)
[[ -f "$STOP" ]] && { status "STOP before AR arm $(stamp)"; exit 9; }
status "AR arm (mswe-S-iter2 fold AR-decode, gmu0.85 c=$C, cap 75) $(stamp)"
run_arm ar run_arm_ar.sh; AR_RC=$?
echo "[gate] AR arm rc=$AR_RC $(stamp)"
APRED="$ROOT/ar/predictions.jsonl"
[[ ! -s "$APRED" ]] && { status "ANOMALY AR predictions missing/empty $(stamp)"; exit 2; }
sleep 5

# 4. OFFICIAL docker scoring both (servers DOWN; one heavy job class at a time)
status "scoring twin+clamp arm $(stamp)"
bash "$ROOT/score_twin.sh" "$DPRED" "$ROOT/diffusion/scoring" "$C"
echo "[gate] twin score rc=$? $(stamp)"
status "scoring AR arm $(stamp)"
bash "$ROOT/score_ar.sh" "$APRED" "$ROOT/ar/scoring" "$C"
echo "[gate] AR score rc=$? $(stamp)"

# 5. gate reports: twin single-arm (>=12/46 + ctx_overflow) + McNemar twin-vs-AR paired
status "building gate reports $(stamp)"
.venv/bin/python "$ROOT/build_report.py" "$ROOT" > "$ROOT/build_report_stdout.txt" 2>&1
echo "[gate] twin report rc=$? $(stamp)"
.venv/bin/python "$ROOT/build_ar_paired_report.py" "$ROOT" > "$ROOT/build_ar_paired_stdout.txt" 2>&1
echo "[gate] paired report rc=$? $(stamp)"

echo "==================== C46 RE-GATE ITER2 DONE $(stamp) ===================="
status "DONE $(stamp)"; date -u +%FT%TZ > "$ROOT/GATE_DONE.txt"
