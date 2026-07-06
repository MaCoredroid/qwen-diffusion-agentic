#!/usr/bin/env bash
# Stage-C N=5 v3 ENVELOPE-CORRECTED arm on the ALIGNED runtime
# (episode-in-official-container). IDENTICAL experimental design to
# runs/stage_c_n5v2/run_v2_arm.sh EXCEPT the reference envelope is FORCED per
# request through the driver's own proxy:
#     LUMO_PROXY_FORCE_TEMPERATURE=0.6  TOP_P=0.95  TOP_K=20  SEED=1234
# The driver's _start_proxy() Popen inherits this env (no env= override), so the
# proxy overwrites qwen-code's greedy-ish bodies with the envelope + a per-request
# reproducible seed (base+counter). This is the certified anti-degenerate regime
# the flywheel SWE campaigns run under (banked in stage_c_n5v2/report.md); v2 ran
# GREEDY (temp 0), which triggered the documented loop-halt degenerate regime.
#
# Also sets SWE_EMPTY_PATCH_RETRIES=1 -> the state-conditional empty-patch
# re-drive mitigation for the known temp-0.6 tool-call-free-terminal flake.
#
#   $1 = arm  in {ar, mergedar, diffusion, diffstock}
#
# ONE arm per invocation, ONE server at a time; bounded server lifecycle (trap
# kills the active user scope on ANY exit), RAM cage via systemd-run --user
# --scope, GPU pre-flight before boot, GPU settle after. out-root -> stage_c_n5v3.
set -uo pipefail
cd /home/mark/qwen_diffusion

ARM="${1:?usage: run_v3_arm.sh <ar|mergedar|diffusion|diffstock>}"

ROOT=runs/stage_c_n5v3
SUBSET=runs/stage_c_n5/subset_n5.json
REPO_CACHE=runs/stage_c_driver/repo_cache        # unused by container runtime; kept for arg
DRIVER=scripts/run_swe_bench_qwen_code.py
PY=.venv/bin/python
AGENT_WALL_S=900
QWEN_MAX_WALL=840s
MAX_TURNS=50

mkdir -p "$ROOT/logs"

# --- docker-as-sudo (askpass never echoes the password) --------------------
export SWE_DOCKER_CMD="sudo -A docker"
export SUDO_ASKPASS="${SUDO_ASKPASS:?export SUDO_ASKPASS to the askpass helper before running}"

# --- reference envelope, FORCED per-request via the driver's proxy ----------
export LUMO_PROXY_FORCE_TEMPERATURE=0.6
export LUMO_PROXY_FORCE_TOP_P=0.95
export LUMO_PROXY_FORCE_TOP_K=20
export LUMO_PROXY_FORCE_SEED=1234
# re-drive mitigation for the temp-0.6 tool-call-free-terminal flake
export SWE_EMPTY_PATCH_RETRIES="${SWE_EMPTY_PATCH_RETRIES:-1}"

case "$ARM" in
  ar)
    PORT=9951; PPORT=30031; SCOPE=stage_c_n5v3_ar
    MODEL=qwen3.5-9b-ar; TAG=n5v3-stock-ar; BOOT_DL=600
    LAUNCH='MAX_MODEL_LEN=32768 GPU_UTIL=0.85 PORT=9951 bash runs/stage_c_driver/runcage_ar.sh'
    ;;
  mergedar)
    PORT=9953; PPORT=30033; SCOPE=stage_c_n5v3_mergedar
    MODEL=qwen3.5-9b-mergedar; TAG=n5v3-merged-ar; BOOT_DL=600
    # v3 mergedar runcage: NO server-side temp-0 override; envelope via proxy only.
    LAUNCH='MAX_MODEL_LEN=32768 GPU_UTIL=0.85 PORT=9953 SERVED_NAME=qwen3.5-9b-mergedar bash runs/stage_c_n5v3/runcage_mergedar_v3.sh'
    ;;
  diffusion)
    PORT=9952; PPORT=30032; SCOPE=stage_c_n5v3_diff
    MODEL=qwen3.5-9b-flare-hybrid-clean; TAG=n5v3-diffusion; BOOT_DL=900
    LAUNCH='MAX_MODEL_LEN=32768 GPU_UTIL=0.85 PORT=9952 HF_HUB_OFFLINE=1 bash /home/mark/shared/lumoFlyWheel_codex_fork/scripts/qwen35_9b_flare_hybrid_serve.sh'
    ;;
  diffstock)
    # 4TH ARM: diffusion-on-STOCK-conversion (B@1000 two-stream, NO RL-v2). SAME
    # flare-hybrid engine as the `diffusion` arm but pointed at the stock-faithful
    # export; force VLLM_QWEN3_5_FLARE_MASK=248077 + CANVAS_LENGTH=32 so the served
    # engine config is byte-identical to the diffusion boot (see v2 run_v2_arm.sh).
    PORT=9954; PPORT=30034; SCOPE=stage_c_n5v3_diffstock
    MODEL=arm4-diffusion-stock; TAG=n5v3-diffstock; BOOT_DL=900
    LAUNCH='MODEL_DIR=/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-b1000-stock-vllm-bf16 SERVED_MODEL_NAME=arm4-diffusion-stock VLLM_QWEN3_5_FLARE_MASK=248077 CANVAS_LENGTH=32 MAX_MODEL_LEN=32768 GPU_UTIL=0.85 PORT=9954 HF_HUB_OFFLINE=1 bash /home/mark/shared/lumoFlyWheel_codex_fork/scripts/qwen35_9b_flare_hybrid_serve.sh'
    ;;
  *) echo "bad arm: $ARM" >&2; exit 2 ;;
esac

slog=$ROOT/logs/${ARM}_server.log
dlog=$ROOT/logs/${ARM}_driver.log

ACTIVE_SCOPE=""
cleanup() {
  if [[ -n "$ACTIVE_SCOPE" ]]; then
    echo "[cleanup] stopping $ACTIVE_SCOPE" >&2
    systemctl --user stop "${ACTIVE_SCOPE}.scope" 2>/dev/null || true
    sleep 3
  fi
  sudo -A docker ps -q --filter "name=swe_ep_" 2>/dev/null | xargs -r sudo -A docker rm -f >/dev/null 2>&1 || true
}
trap cleanup EXIT

gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }

preflight() {
  local deadline=$((SECONDS+600))
  while :; do
    local u; u=$(gpu_used)
    if [[ "$u" -lt 3000 ]]; then echo "[preflight] GPU ${u} MiB < 3000, clear" >&2; return 0; fi
    if [[ $SECONDS -gt $deadline ]]; then echo "[preflight] TIMEOUT GPU ${u} MiB" >&2; return 1; fi
    echo "[preflight] GPU ${u} MiB busy, waiting..." >&2; sleep 10
  done
}

wait_ready() {  # $1 port  $2 scope  $3 boot_deadline_s
  local port=$1 scope=$2 dl=$3
  local deadline=$((SECONDS+dl))
  while :; do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "[ready] :${port} up" >&2; return 0; fi
    if ! systemctl --user is-active --quiet "${scope}.scope"; then
      echo "[ready] scope ${scope} died before ready" >&2; return 2; fi
    if [[ $SECONDS -gt $deadline ]]; then echo "[ready] :${port} BOOT TIMEOUT" >&2; return 1; fi
    sleep 5
  done
}

echo "==== ARM=$ARM START $(date -u +%FT%TZ) runtime=container eval=skip ENVELOPE temp=0.6/top_p=0.95/top_k=20/seed=1234 retries=$SWE_EMPTY_PATCH_RETRIES ====" >&2
preflight || { echo "[$ARM] preflight failed" >&2; exit 1; }
echo "[$ARM] launching server scope=$SCOPE port=$PORT" >&2
ACTIVE_SCOPE="$SCOPE"
systemd-run --user --scope --unit="$SCOPE" \
  -p MemoryMax=22G -p MemorySwapMax=4G \
  bash -c "$LAUNCH" > "$slog" 2>&1 &
wait_ready "$PORT" "$SCOPE" "$BOOT_DL" || { echo "[$ARM] server not ready; aborting" >&2; exit 1; }

echo "[$ARM] driver start $(date -u +%FT%TZ)" >&2
$PY $DRIVER \
  --subset "$SUBSET" \
  --out-root "$ROOT/$ARM" \
  --runtime container \
  --endpoint "http://127.0.0.1:${PORT}/v1" \
  --model "$MODEL" --model-name "$TAG" \
  --repo-cache "$REPO_CACHE" \
  --eval-mode skip \
  --agent-wall-s $AGENT_WALL_S --qwen-max-wall $QWEN_MAX_WALL \
  --max-session-turns $MAX_TURNS \
  --proxy-port $PPORT \
  --proxy-dump-dir "$ROOT/dumps_${ARM}" \
  --proxy-tool-choice "" \
  > "$dlog" 2>&1
rc=$?
echo "[$ARM] driver done rc=$rc $(date -u +%FT%TZ)" >&2

echo "[$ARM] stopping server scope=$SCOPE" >&2
systemctl --user stop "${SCOPE}.scope" 2>/dev/null || true
ACTIVE_SCOPE=""
sleep 5
settle=$((SECONDS+180))
while :; do u=$(gpu_used); [[ "$u" -lt 3000 ]] && { echo "[$ARM] GPU settled ${u} MiB" >&2; break; }
  [[ $SECONDS -gt $settle ]] && { echo "[$ARM] GPU settle timeout ${u} MiB" >&2; break; }; sleep 5; done
echo "==== ARM=$ARM END rc=$rc $(date -u +%FT%TZ) ====" >&2
exit $rc
