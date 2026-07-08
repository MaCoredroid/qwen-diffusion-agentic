#!/usr/bin/env bash
# BATCH GEN — self-generation for one batch. Boots ONE stock-AR Qwen3.5-9B server
# in the RAM cage @ max_num_seqs=C, drives C concurrent qwen-code shards over the
# batch (episode-in-official-container, native qwen3_xml, REFERENCE ENVELOPE
# forced per-request via each shard's proxy, empty-patch re-drive=1), samples GPU
# util, stops the server + settles. Self-bounded: GPU preflight/boot deadlines +
# a trap that stops the scope and reaps swe_ep_ containers on ANY exit. Ported
# from runs/stage0_swegym_probe_v2/probe_gen_v2.sh, parametrized per batch, with
# --skip-existing so a resumed batch does not re-run finished instances.
#   usage: datagen_gen.sh <batchdir> <gen_root> <concurrency>
set -uo pipefail
cd /home/mark/qwen_diffusion
export SUDO_ASKPASS="${SUDO_ASKPASS:-}"
# docker-group host -> plain docker (no sudo available non-interactively). Override
# to 'sudo -A docker' + SUDO_ASKPASS only where the user lacks the docker group.
export SWE_DOCKER_CMD="${SWE_DOCKER_CMD:-docker}"
HERE=runs/swe_datagen_s1
PY=.venv/bin/python
DRIVER=scripts/run_swe_bench_qwen_code.py
# Runcage selector — which server launcher to boot for this batch.
# TEACHER SWAP 2026-07-08: the DEFAULT is now the Qwen3.6-27B NVFP4 + native-MTP
# teacher (runcage_27b.sh), certified by the three-gate acceptance run
# gate_27b_20260708T181604Z (ALL PASS: format schema-equiv 0 mismatches;
# arg-grounding 63/64=98.4% source-verbatim, 0 malformed; live 4/4 resolved;
# MTP A/B 1.59x @ 0.93 accept). datagen_gen.sh is invoked fresh per cycle, so this
# default (or a single RUNCAGE_SCRIPT export by the orchestrator) takes effect at
# the next cycle boundary.
#   ONE-LINE ROLLBACK to the stock-AR 9B teacher:
#     export RUNCAGE_SCRIPT=runcage_ar_probe.sh   # (also relaunch orch with C=4)
RUNCAGE_SCRIPT="${RUNCAGE_SCRIPT:-runcage_27b.sh}"

# Teacher-coupled knobs, keyed off the runcage selection so the swap flips ALL of
# them together: the served-model name the driver must REQUEST (must equal the
# launcher's --served-model-name or vLLM 404s), the keeper-provenance teacher
# label, and the CERTIFIED server primitives that differ from runcage_27b.sh's bare
# defaults. The 27B FROZEN/certified config (bootprobe_27b/FROZEN_CONFIG.json,
# boot_server.sh) is: TRITON_ATTN (decode), MAX_NUM_SEQS=2, NUM_SPEC_TOKENS=1,
# KV_CACHE_DTYPE=auto (-> fp8_e4m3 from ckpt), KV_OFFLOAD_GB=0. Offload stays OFF:
# the boot-probe froze it (it drops expandable_segments and SHRINKS the on-GPU fp8
# KV pool 83012->77550 tok — a net loss when 2x32k already fit; HOST-RAM-safest).
case "$RUNCAGE_SCRIPT" in
  runcage_27b.sh)
    DRIVER_MODEL="${DRIVER_MODEL:-qwen3.6-27b-nvfp4}"            # == runcage_27b SERVED_NAME
    DRIVER_MODEL_NAME="${DRIVER_MODEL_NAME:-datagen-27b-nvfp4-mtp-env}"
    ATTENTION_BACKEND="${ATTENTION_BACKEND:-TRITON_ATTN}"       # frozen/certified backend
    KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"                    # ckpt -> fp8_e4m3
    KV_OFFLOAD_GB="${KV_OFFLOAD_GB:-0}"                         # boot-probe froze offload OFF
    ;;
  *)  # stock-AR 9B (rollback) / MTP-9B — those launchers hardcode their own KV/attn
    DRIVER_MODEL="${DRIVER_MODEL:-qwen3.5-9b-ar}"
    DRIVER_MODEL_NAME="${DRIVER_MODEL_NAME:-datagen-stockAR-env}"
    ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
    KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
    KV_OFFLOAD_GB="${KV_OFFLOAD_GB:-}"
    ;;
esac

BATCHDIR="${1:?batchdir}"; GEN_ROOT="${2:?gen_root}"; C="${3:-4}"
PORT="${PORT:-9951}"
SCOPE="${SCOPE:-datagen_ar}"
AGENT_WALL_S="${AGENT_WALL_S:-900}"
QWEN_MAX_WALL="${QWEN_MAX_WALL:-840s}"
MAX_TURNS="${MAX_TURNS:-50}"          # design allows 50->75; keep 50 (envelope probe cap)
BOOT_DL="${BOOT_DL:-600}"
# GPU preflight clear-threshold. This is NOT "GPU must be empty" — this host runs a
# GNOME desktop whose compositor (gnome-shell) holds a PERSISTENT ~3.9 GiB of VRAM
# that never releases. The old hardcoded 3600 MiB gate passed cycle-1 by 13 MiB
# (3587 clear) but the desktop crept to ~3923 MiB by cycle-2, so cycles 2-3
# preflight-TIMED-OUT (600s) and the server NEVER booted -> every instance recorded
# no_prediction. The gate's real job is to catch a LEAKED model server (a 9B AR
# server holds ~18-27 GiB), so the bar must sit ABOVE the desktop baseline and well
# BELOW a leaked server. 8000 MiB tolerates the desktop (+~4 GiB creep headroom) and
# still trips on any leaked vLLM. vLLM boots fine on top: gpu_util=0.85 leaves ~15%
# (~4.9 GiB) free, which absorbs the ~3.9 GiB desktop (cycle-1 proved boot at 3587).
PREFLIGHT_MAX_MIB="${PREFLIGHT_MAX_MIB:-8000}"
mkdir -p "$GEN_ROOT" "$BATCHDIR/logs"

# --- reference envelope, FORCED per-request via each shard driver's proxy ------
export LUMO_PROXY_FORCE_TEMPERATURE="${LUMO_PROXY_FORCE_TEMPERATURE:-0.6}"
export LUMO_PROXY_FORCE_TOP_P="${LUMO_PROXY_FORCE_TOP_P:-0.95}"
export LUMO_PROXY_FORCE_TOP_K="${LUMO_PROXY_FORCE_TOP_K:-20}"
export LUMO_PROXY_FORCE_SEED="${LUMO_PROXY_FORCE_SEED:-1234}"
export SWE_EMPTY_PATCH_RETRIES="${SWE_EMPTY_PATCH_RETRIES:-1}"

gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
ACTIVE_SCOPE=""; SAMPLER_PID=""
cleanup() {
  [[ -n "$SAMPLER_PID" ]] && kill "$SAMPLER_PID" 2>/dev/null || true
  if [[ -n "$ACTIVE_SCOPE" ]]; then
    echo "[cleanup] stopping $ACTIVE_SCOPE" >&2
    systemctl --user stop "${ACTIVE_SCOPE}.scope" 2>/dev/null || true; sleep 3
  fi
  $SWE_DOCKER_CMD ps -q --filter "name=swe_ep_" 2>/dev/null | xargs -r $SWE_DOCKER_CMD rm -f >/dev/null 2>&1 || true
}
trap cleanup EXIT

preflight() {
  local deadline=$((SECONDS+600))
  while :; do
    local u; u=$(gpu_used)
    [[ "$u" -lt "$PREFLIGHT_MAX_MIB" ]] && { echo "[preflight] GPU ${u} MiB clear (<${PREFLIGHT_MAX_MIB} MiB; ~desktop baseline)" >&2; return 0; }
    [[ $SECONDS -gt $deadline ]] && { echo "[preflight] TIMEOUT ${u} MiB (>=${PREFLIGHT_MAX_MIB}; leaked server?)" >&2; return 1; }
    echo "[preflight] GPU ${u} MiB busy (>=${PREFLIGHT_MAX_MIB})..." >&2; sleep 10
  done
}
wait_ready() {
  local deadline=$((SECONDS+BOOT_DL))
  # Grace before the first liveness check: `systemd-run --user --scope &` returns
  # before the transient scope unit is registered, so an immediate check races the
  # unit's "activating" window and mis-reads not-yet-registered as death. 5s lets
  # systemd register + start the scope (a --scope goes active the moment its process
  # forks). Fixes a flaky false "[ready] scope died" that aborted a fully-healthy boot.
  sleep 5
  while :; do
    curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && { echo "[ready] :${PORT} up" >&2; return 0; }
    # Bail ONLY on a definitively dead scope (failed/inactive), NOT on "activating".
    local st; st=$(systemctl --user show -p ActiveState --value "${SCOPE}.scope" 2>/dev/null)
    [[ "$st" == "failed" || "$st" == "inactive" ]] && { echo "[ready] scope died (state=$st)" >&2; return 2; }
    [[ $SECONDS -gt $deadline ]] && { echo "[ready] BOOT TIMEOUT" >&2; return 1; }
    sleep 5
  done
}

N=$(.venv/bin/python -c 'import json,sys;print(len(json.load(open(sys.argv[1]))["instance_ids"]))' "$BATCHDIR/subset.json")
echo "==== BATCH GEN START $(date -u +%FT%TZ) n=$N C=$C envelope=$LUMO_PROXY_FORCE_TEMPERATURE/$LUMO_PROXY_FORCE_TOP_P/$LUMO_PROXY_FORCE_TOP_K retries=$SWE_EMPTY_PATCH_RETRIES ====" >&2
preflight || exit 1
# GPU_UTIL vs desktop-VRAM drift: vLLM hard-fails at boot unless free >= util*total
# (cycles 4-5 burned: free 26.34 GiB < 0.85*31.33 after desktop crept to ~4.6 GiB).
# Derive util from measured free with 1800 MiB margin (vLLM's torch-visible free runs
# ~1 GiB under nvidia-smi's), cap at 0.85, hard-floor at the boot-probe-certified 0.74.
read -r GPU_USED_MIB GPU_TOTAL_MIB < <(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F', *' 'NR==1{print $1, $2}')
GPU_UTIL=$(python3 -c "print(f'{min(0.85, ($GPU_TOTAL_MIB - $GPU_USED_MIB - 1800) / $GPU_TOTAL_MIB):.2f}')")
python3 -c "exit(0 if $GPU_UTIL >= 0.74 else 1)" || { echo "[gen] GPU_UTIL=$GPU_UTIL below certified 0.74 floor (desktop holds ${GPU_USED_MIB}MiB) — refusing boot" >&2; exit 1; }
echo "[gen] booting server scope=$SCOPE port=$PORT gpu_util=$GPU_UTIL runcage=$RUNCAGE_SCRIPT seqs=$C spec=${NUM_SPEC_TOKENS:-1} attn=${ATTENTION_BACKEND:-launcher-default} kv=${KV_CACHE_DTYPE:-launcher-default} kv_offload_gb=${KV_OFFLOAD_GB:-launcher-default} model=$DRIVER_MODEL (used=${GPU_USED_MIB}MiB total=${GPU_TOTAL_MIB}MiB)" >&2
ACTIVE_SCOPE="$SCOPE"
# Pass the teacher-coupled certified primitives through the cage to the launcher.
# For the 9B launchers these envs are unused (they hardcode their own KV/attn), so
# passing empty strings is harmless; for runcage_27b.sh they lock the FROZEN config.
systemd-run --user --scope --unit="$SCOPE" -p MemoryMax=22G -p MemorySwapMax=4G \
  bash -c "MAX_NUM_SEQS=$C MAX_MODEL_LEN=32768 GPU_UTIL=$GPU_UTIL PORT=$PORT NUM_SPEC_TOKENS='${NUM_SPEC_TOKENS:-1}' ATTENTION_BACKEND='$ATTENTION_BACKEND' KV_CACHE_DTYPE='$KV_CACHE_DTYPE' KV_OFFLOAD_GB='$KV_OFFLOAD_GB' bash $HERE/$RUNCAGE_SCRIPT" \
  > "$BATCHDIR/logs/gen_server.log" 2>&1 &
wait_ready || { echo "[gen] server not ready" >&2; exit 1; }

( while systemctl --user is-active --quiet "${SCOPE}.scope"; do
    echo "$(date -u +%FT%TZ),$(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    sleep 5
  done ) >> "$GEN_ROOT/gpu_util.csv" 2>/dev/null &
SAMPLER_PID=$!

echo "[gen] launching $C shards concurrently $(date -u +%FT%TZ)" >&2
pids=()
for ((k=0; k<C; k++)); do
  SUBSET="$BATCHDIR/shard_${k}.json"
  [[ -s "$SUBSET" ]] || { echo "[gen] shard $k subset missing/empty, skip" >&2; continue; }
  PPORT=$((30030+k))
  # best-of-k: use THIS shard's per-cycle base_seed (1234 + cycle*1e6 + k*1e5)
  # as the proxy's forced seed BASE, so a re-attempt of an instance in a LATER
  # cycle draws a DISTINCT seed (=> a distinct rollout). Falls back to the global
  # LUMO_PROXY_FORCE_SEED pin if the shard json has no base_seed.
  SHARD_SEED=$(.venv/bin/python -c 'import json,sys;print(json.load(open(sys.argv[1])).get("base_seed",""))' "$SUBSET" 2>/dev/null)
  SHARD_SEED="${SHARD_SEED:-$LUMO_PROXY_FORCE_SEED}"
  LUMO_PROXY_FORCE_SEED="$SHARD_SEED" $PY $DRIVER \
    --subset "$SUBSET" \
    --out-root "$GEN_ROOT/shard_${k}" \
    --runtime container \
    --endpoint "http://127.0.0.1:${PORT}/v1" \
    --model "$DRIVER_MODEL" --model-name "$DRIVER_MODEL_NAME" \
    --repo-cache runs/stage_c_driver/repo_cache \
    --eval-mode skip \
    --skip-existing \
    --agent-wall-s "$AGENT_WALL_S" --qwen-max-wall "$QWEN_MAX_WALL" \
    --max-session-turns "$MAX_TURNS" \
    --proxy-port "$PPORT" \
    --proxy-dump-dir "$GEN_ROOT/dumps_shard_${k}" \
    --container-name-prefix "swe_ep_s${k}" \
    --proxy-tool-choice "" \
    > "$GEN_ROOT/shard_${k}.driver.log" 2>&1 &
  pids+=($!)
  echo "[gen] shard=$k pid=${pids[-1]} pport=$PPORT seed_base=$SHARD_SEED n_ids=$(.venv/bin/python -c "import json;print(len(json.load(open('$SUBSET'))['instance_ids']))")" >&2
  sleep 2
done

rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[gen] all shards done rc=$rc $(date -u +%FT%TZ)" >&2

kill "$SAMPLER_PID" 2>/dev/null || true; SAMPLER_PID=""
systemctl --user stop "${SCOPE}.scope" 2>/dev/null || true; ACTIVE_SCOPE=""
sleep 5
settle=$((SECONDS+180))
while :; do u=$(gpu_used); [[ "$u" -lt "$PREFLIGHT_MAX_MIB" ]] && { echo "[gen] GPU settled ${u} MiB" >&2; break; }
  [[ $SECONDS -gt $settle ]] && { echo "[gen] settle timeout ${u} MiB" >&2; break; }; sleep 5; done
echo "==== BATCH GEN END rc=$rc $(date -u +%FT%TZ) ====" >&2
exit $rc
