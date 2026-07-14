#!/usr/bin/env bash
# C46 UNDER THE NEW ENVELOPE (#141) — detached, self-bounded orchestrator.
# Mirrors the iteration-2 gate (runs/k_gate_c46_iter2/run_gate.sh) but is a
# SINGLE-ARM run: only the twin@K1 diffusion arm, served gate-ON with the W-2
# causal draft-verify path (VLLM_FASTDLLM_W1_DRAFT_VERIFY=1). The two paired
# comparators (AR 12/48 and twin gate-OFF 1/48) are BANKED in
# runs/k_gate_c46_iter2/ar_paired_report.json and are NEVER re-run.
#
# Deltas vs iteration-2 (ONLY these):
#   (a) TWIN ARM ONLY (no AR arm; comparators banked).
#   (b) twin server boots VLLM_FASTDLLM_W1_DRAFT_VERIFY=1 (W-2 causal path);
#       run_arm_twin.sh hard-asserts the gate is ON from the server log before
#       episodes fan out, and this orchestrator asserts the W-2 causal-verify
#       source marker is present in the pinned engine before boot.
#   (c) CERTIFIED read-clamp proxy active (limit=100, cert 7ae55d4) as before.
#   (d) output = runs/k_gate_c46_newenv/.
#   (e) after teardown+scoring, parse_w1_telemetry.py records the live w1
#       counters (spans/toks/vfwd/rej/arej) + model_forwards + wall so the report
#       computes live blended tok/fwd + wall/episode vs the banked gate-OFF twin.
#
# SEQUENCE (ONE server; caged via systemd-run --user --scope):
#   1. verify 48 images present + GPU-idle preflight + W-2 engine-source assert
#   2. twin gate-ON+clamp diffusion arm (gmu 0.74 seqs 4) -> teardown + GPU settle
#   3. OFFICIAL docker scoring (server DOWN)
#   4. parse w1 telemetry (gate-ON log vs banked gate-OFF log)
#   5. build_report.py: resolve@1 vs >=12/46 + McNemar vs BOTH banked comparators
#      + ctx_overflow buckets + arej-must-be-0 + speed covariates
#
# MEMORY-BUDGET RULE: diffusion gmu 0.74 / seqs 4 (the GDN align-cache lives
# OUTSIDE the KV pool; authoritative, not the AR arm's 0.85).
#
# [state] lines (eps done / 48, wall) every 60s. STOP-file:
# runs/k_gate_c46_newenv/STOP (touch to abort gracefully -> server torn down, exit 9).
# Detach with:  setsid bash runs/k_gate_c46_newenv/run_gate.sh &   (pidfile gate.pid)
# Docker via the docker group (plain `docker`, no sudo/askpass on this host).
set -uo pipefail
cd /home/mark/qwen_diffusion
export SWE_DOCKER_CMD="docker"

ROOT=runs/k_gate_c46_newenv
PLAN=runs/k_gate_c46/shard_plan.json          # SAME frozen 48-instance pool
IMAGES=data/swe_kraise_c46_pool/images.txt
BANKED=runs/k_gate_c46_iter2/ar_paired_report.json
GATE_OFF_LOG=runs/k_gate_c46_iter2/logs/diffusion_server.log
GATE_OFF_WALL=14943                           # banked twin gate-OFF wall (throughput.twin)
FLARE_SRC=/home/mark/shared/vllm_p2_pr42406/vllm/v1/worker/gpu/model_states/qwen3_5_flare.py
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
  systemctl --user stop c46ne_diff_server.scope 2>/dev/null || true
  pkill -TERM -f 'qwen35_9b_flare_hybrid_serve|EngineCore|vllm' 2>/dev/null || true
  local dl=$((SECONDS+90))
  while :; do local u; u=$(gpu_used); [[ "$u" -lt "$GPU_CEIL" ]] && break; [[ $SECONDS -gt $dl ]] && break; sleep 4; done
  rm -f "$PIDFILE"
  echo "[gate] EXIT rc=$rc $(stamp)"
}
trap cleanup EXIT

# episode-count monitor for the twin arm; emits [state] lines to the gate log.
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

echo "==================== C46 NEW-ENVELOPE (gate-ON W-2) START $(stamp) ===================="

# 0. STOP pre-check
[[ -f "$STOP" ]] && { status "ABORT: STOP-file present at launch $(stamp)"; exit 9; }

# 1a. W-2 engine-source assert: the pinned engine must carry the causal-verify
# redesign (proves the compiled path is W-2, not the W-1d bidirectional read).
if grep -q "RUNG W-2 (byte-faithful redesign)" "$FLARE_SRC" 2>/dev/null; then
  echo "[gate] W-2 engine-source assert PASS ($FLARE_SRC) $(stamp)"
else
  status "ABORT: W-2 causal-verify marker absent from engine source $FLARE_SRC $(stamp)"; exit 1
fi

# 1b. image verification (all 48 present; pull is a separate prior step)
status "verifying 48 images present $(stamp)"
docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep 'sweb.eval' | sort -u > "$ROOT/logs/have_images.txt"
miss=0
while read -r img; do
  grep -qxF "$img" "$ROOT/logs/have_images.txt" || { echo "[gate] MISSING image: $img"; miss=$((miss+1)); }
done < "$IMAGES"
if [[ "$miss" -ne 0 ]]; then status "ABORT: $miss/48 images missing $(stamp)"; exit 1; fi
echo "[gate] all 48 images present $(stamp)"

# preflight: GPU idle by memory (gnome-shell baseline tolerated)
u=$(gpu_used)
if [[ "$u" -ge "$GPU_CEIL" ]]; then status "PREFLIGHT ABORT: GPU busy ${u}MiB $(stamp)"; exit 1; fi
echo "[gate] preflight clear ${u}MiB $(stamp)"

# 2. TWIN gate-ON + clamp diffusion arm (server down + settled inside run_arm_twin.sh)
[[ -f "$STOP" ]] && { status "STOP before twin arm $(stamp)"; exit 9; }
status "twin gate-ON+clamp diffusion arm (W-2 causal draft-verify, clamp=100, gmu0.74 c=$C, cap 75) $(stamp)"
run_arm diffusion run_arm_twin.sh; TWIN_RC=$?
echo "[gate] twin arm rc=$TWIN_RC $(stamp)"
# PERMANENT DORMANCY GUARD (RUNG W-2b): fail-closed if the arm aborted because the
# copy draft-verify fired zero spans across episode 1 (a byte-identical placebo of
# the banked gate-OFF twin). Never score / bank a dormant-gate run.
if [[ -f "$ROOT/DORMANT_GATE.txt" ]]; then
  status "ABORT: DORMANT_GATE — copy verify fired no spans in episode 1 (placebo run prevented) $(stamp)"
  echo "[state] DORMANT_GATE $(cat "$ROOT/DORMANT_GATE.txt")"; exit 8
fi
DPRED="$ROOT/diffusion/predictions.jsonl"
[[ ! -s "$DPRED" ]] && { status "ANOMALY twin predictions missing/empty $(stamp)"; exit 2; }
sleep 5

# 2b. gate-ON log evidence (post-run): confirm the gate booted ON + w1 counters fired
GRC=$(grep -m1 'FLARE W-1b copy draft-and-verify gate:' "$ROOT/logs/diffusion_server.log" 2>/dev/null || true)
W1CNT=$(grep -c 'w1\[on=True' "$ROOT/logs/diffusion_server.log" 2>/dev/null || echo 0)
echo "[gate] gate-ON evidence: '$GRC'  w1[on=True lines=$W1CNT $(stamp)"

# 3. OFFICIAL docker scoring (server DOWN)
status "scoring twin gate-ON+clamp arm $(stamp)"
bash "$ROOT/score_twin.sh" "$DPRED" "$ROOT/diffusion/scoring" "$C"
echo "[gate] twin score rc=$? $(stamp)"

# 4. w1 telemetry: live tok/fwd + wall/episode gate-ON vs banked gate-OFF; arej audit
status "parsing w1 draft-verify telemetry $(stamp)"
.venv/bin/python "$ROOT/parse_w1_telemetry.py" \
  "$ROOT/logs/diffusion_server.log" "$GATE_OFF_LOG" \
  "$ROOT/diffusion/arm_timing.json" "$GATE_OFF_WALL" "$N" \
  "$ROOT/w1_telemetry.json" > "$ROOT/w1_telemetry_stdout.txt" 2>&1
echo "[gate] w1 telemetry rc=$? $(stamp)"

# 5. gate report: single-arm resolve@1 vs >=12/46 + McNemar vs BOTH banked comparators
status "building gate report $(stamp)"
.venv/bin/python "$ROOT/build_report.py" "$ROOT" > "$ROOT/build_report_stdout.txt" 2>&1
echo "[gate] report rc=$? $(stamp)"

echo "==================== C46 NEW-ENVELOPE (gate-ON W-2) DONE $(stamp) ===================="
status "DONE $(stamp)"; date -u +%FT%TZ > "$ROOT/GATE_DONE.txt"
