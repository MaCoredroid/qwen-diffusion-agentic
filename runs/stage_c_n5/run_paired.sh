#!/usr/bin/env bash
# Stage-C N=5 paired run: 5 Tier0 instances x 2 arms, ONE server at a time.
# AR arm (all 5) -> kill -> diffusion arm (all 5) -> kill. Server lifecycle is
# bounded inside this script (trap kills the active scope on ANY exit), RAM cage
# via systemd-run --user --scope, GPU pre-flight before each boot.
set -uo pipefail
cd /home/mark/qwen_diffusion

ROOT=runs/stage_c_n5
SUBSET=$ROOT/subset_n5.json
REPO_CACHE=runs/stage_c_driver/repo_cache
DRIVER=scripts/run_swe_bench_qwen_code.py
PY=.venv/bin/python
AGENT_WALL_S=900
QWEN_MAX_WALL=840s
MAX_TURNS=50

ACTIVE_SCOPE=""
cleanup() {
  if [[ -n "$ACTIVE_SCOPE" ]]; then
    echo "[cleanup] stopping $ACTIVE_SCOPE" >&2
    systemctl --user stop "${ACTIVE_SCOPE}.scope" 2>/dev/null || true
    sleep 3
  fi
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

run_arm() {  # $1 arm  $2 port  $3 model  $4 tag  $5 proxyport  $6 launcher_cmd  $7 scope  $8 boot_dl
  local arm=$1 port=$2 model=$3 tag=$4 pport=$5 launcher=$6 scope=$7 boot_dl=$8
  local slog=$ROOT/logs/${arm}_server.log
  local dlog=$ROOT/logs/${arm}_driver.log
  echo "==== ARM=$arm START $(date -u +%FT%TZ) ====" >&2
  preflight || { echo "[$arm] preflight failed" >&2; return 1; }
  echo "[$arm] launching server scope=$scope port=$port" >&2
  ACTIVE_SCOPE="$scope"
  systemd-run --user --scope --unit="$scope" \
    -p MemoryMax=22G -p MemorySwapMax=4G \
    bash -c "$launcher" > "$slog" 2>&1 &
  wait_ready "$port" "$scope" "$boot_dl" || { echo "[$arm] server not ready; aborting arm" >&2; cleanup; ACTIVE_SCOPE=""; return 1; }
  echo "[$arm] driver start $(date -u +%FT%TZ)" >&2
  $PY $DRIVER \
    --subset "$SUBSET" \
    --out-root "$ROOT/$arm" \
    --endpoint "http://127.0.0.1:${port}/v1" \
    --model "$model" --model-name "$tag" \
    --repo-cache "$REPO_CACHE" \
    --eval-mode mock \
    --agent-wall-s $AGENT_WALL_S --qwen-max-wall $QWEN_MAX_WALL \
    --max-session-turns $MAX_TURNS \
    --proxy-port $pport \
    --proxy-dump-dir "$ROOT/dumps_${arm}" \
    --proxy-tool-choice "" \
    > "$dlog" 2>&1
  local rc=$?
  echo "[$arm] driver done rc=$rc $(date -u +%FT%TZ)" >&2
  echo "[$arm] stopping server scope=$scope" >&2
  systemctl --user stop "${scope}.scope" 2>/dev/null || true
  ACTIVE_SCOPE=""
  sleep 5
  local settle=$((SECONDS+180))
  while :; do local u; u=$(gpu_used); [[ "$u" -lt 3000 ]] && { echo "[$arm] GPU settled ${u} MiB" >&2; break; }
    [[ $SECONDS -gt $settle ]] && { echo "[$arm] GPU settle timeout ${u} MiB" >&2; break; }; sleep 5; done
  echo "==== ARM=$arm END $(date -u +%FT%TZ) ====" >&2
}

# ---- AR arm (stock AR :9951) ----
AR_LAUNCH='MAX_MODEL_LEN=32768 GPU_UTIL=0.85 PORT=9951 bash runs/stage_c_driver/runcage_ar.sh'
run_arm ar 9951 qwen3.5-9b-ar \
  "qwen3.5-9b-ar::qwen-code-0.19.2::stage-c-n5" \
  30021 "$AR_LAUNCH" stage_c_n5_ar 600

# ---- Diffusion arm (FLARE hybrid-clean :9952) ----
DIFF_LAUNCH='MAX_MODEL_LEN=32768 GPU_UTIL=0.85 PORT=9952 HF_HUB_OFFLINE=1 bash /home/mark/shared/lumoFlyWheel_codex_fork/scripts/qwen35_9b_flare_hybrid_serve.sh'
run_arm diffusion 9952 qwen3.5-9b-flare-hybrid-clean \
  "qwen3.5-9b-flare-hybrid-clean::qwen-code-0.19.2::stage-c-n5" \
  30022 "$DIFF_LAUNCH" stage_c_n5_diff 900

echo "==== PAIRED RUN COMPLETE $(date -u +%FT%TZ) ====" >&2
