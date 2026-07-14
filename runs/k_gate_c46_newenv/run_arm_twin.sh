#!/usr/bin/env bash
# C46-UNDER-NEW-ENVELOPE — twin@K1 diffusion arm, gate-ON (W-2 causal draft-verify)
# WITH the CERTIFIED read-clamp shim. Clone of the iteration-2 twin arm
# (runs/k_gate_c46_iter2/run_arm_twin.sh). The ONLY delta vs that frozen gate-OFF arm
# is the W-2 gate:
#   * LAUNCH adds VLLM_FASTDLLM_W1_DRAFT_VERIFY=1 — the W-2 causal fixed-width
#     block-commit draft-verify path (engine pin qwen3_5-flare-modelstate @ b92af2d,
#     LOCAL, never pushed; W-2 commit 786ed3d). BIDIR_PROBE stays serve-script default 1.
#   * After boot we HARD-ASSERT from the server log that the gate is ON
#     ("FLARE W-1b copy draft-and-verify gate: True") before any episode fans out.
# Everything else is IDENTICAL to the banked gate-OFF twin arm (the paired comparator):
#   - --proxy-script runs/k_gate_c46/proxy_readclamp.py  (certified shim, 7ae55d4)
#   - LUMO_PROXY_READCLAMP_LIMIT=$CLAMP (default 100 = the certified injected value)
# Model = models/qwen3.5-9b-fastdllm-mswe2-S-twinK1-vllm-bf16 (iteration-2 twin@plain).
# Envelope frozen IDENTICAL: mask 248077, max_model_len 32768, gmu 0.74, max_num_seqs 4,
# temp 0.6/top_p 0.95/top_k 20 (NO pp), per-shard base seeds, turn cap 75, empty-patch
# re-drive 1. Writes ARM=diffusion so the gate report builder finds it.
# MEMORY-BUDGET RULE: gmu 0.74 (NOT the AR arm's 0.85) — the GDN align-cache lives
# OUTSIDE the KV pool; the diffusion gmu/concurrency is authoritative.
# Docker via the docker group (plain `docker`, no sudo/askpass on this host).
#   usage: run_arm_twin.sh <concurrency> <outbase> <shard_plan.json> [clamp_limit]
set -uo pipefail
cd /home/mark/qwen_diffusion

C="${1:?concurrency}"; OUTBASE="${2:?outbase}"; PLAN="${3:?shard_plan.json}"; CLAMP="${4:-100}"
ARM=diffusion
SUBSET=runs/k_gate_c46/inputs/subset_c46.json
REPO_CACHE=runs/stage_c_driver/repo_cache
DRIVER=scripts/run_swe_bench_qwen_code.py
PROXY_SCRIPT=runs/k_gate_c46/proxy_readclamp.py
PY=.venv/bin/python
AGENT_WALL_S=1500          # 25 min hard agent wall (turn cap 75 at ~17s/turn)
QWEN_MAX_WALL=1440s
MAX_TURNS=75               # design §2.1 turn cap
GPU_CEIL=8000
export SWE_DOCKER_CMD="docker"

# ---- frozen diffusion envelope (v3) + re-drive; presence_penalty DROPPED --------
export LUMO_PROXY_FORCE_TEMPERATURE=0.6
export LUMO_PROXY_FORCE_TOP_P=0.95
export LUMO_PROXY_FORCE_TOP_K=20
export SWE_EMPTY_PATCH_RETRIES=1
unset LUMO_PROXY_FORCE_PRESENCE_PENALTY 2>/dev/null || true
unset LUMO_PROXY_FORCE_MIN_P 2>/dev/null || true
# ---- THE CERTIFIED READ-CLAMP: inject `limit` on read_file calls that drop it ----
export LUMO_PROXY_READCLAMP_LIMIT="$CLAMP"

PORT=9952; SCOPE=c46ne_diff_server; BOOT_DL=900
MODEL=qwen3.5-9b-flare-hybrid-clean; TAG=c46ne-twinK1-gateON-clamp-diffusion
PROXY_PORT_KEY=diff_proxy_port
# iteration-2 twin@plain export + explicit mask (manifest stores mask top-level, launcher
# parse resolves None otherwise); frozen serve config from the diffusion arm. The ONLY
# new-envelope delta: VLLM_FASTDLLM_W1_DRAFT_VERIFY=1 (the W-2 causal draft-verify gate).
LAUNCH='VLLM_FASTDLLM_W1_DRAFT_VERIFY=1 MODEL_DIR=/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-mswe2-S-twinK1-vllm-bf16 MASK_TOKEN_ID=248077 MAX_MODEL_LEN=32768 GPU_UTIL=0.74 MAX_NUM_SEQS=4 PORT=9952 HF_HUB_OFFLINE=1 bash /home/mark/shared/lumoFlyWheel_codex_fork/scripts/qwen35_9b_flare_hybrid_serve.sh'

mkdir -p "$OUTBASE/logs" "$OUTBASE/$ARM"
slog="$OUTBASE/logs/${ARM}_server.log"
monlog="$OUTBASE/logs/${ARM}_monitor.log"

gpu_used()  { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
gpu_capps() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . ; }

ACTIVE_SCOPE=""; MONPID=""
cleanup() {
  [[ -n "$MONPID" ]] && kill "$MONPID" 2>/dev/null || true
  pkill -TERM -f "run_swe_bench_qwen_code.py .*${OUTBASE}/${ARM}" 2>/dev/null || true
  if [[ -n "$ACTIVE_SCOPE" ]]; then
    echo "[cleanup] stopping $ACTIVE_SCOPE" >&2
    systemctl --user stop "${ACTIVE_SCOPE}.scope" 2>/dev/null || true
    pkill -TERM -f 'qwen35_9b_flare_hybrid_serve|EngineCore|vllm' 2>/dev/null || true
    sleep 3
    local sdl=$((SECONDS+120))
    while :; do local cn cu; cn=$(gpu_capps); cu=$(gpu_used)
      [[ "$cu" -lt "$GPU_CEIL" ]] && { echo "[cleanup] GPU settled capps=$cn ${cu}MiB" >&2; break; }
      [[ $SECONDS -gt $sdl ]] && { echo "[cleanup] GPU settle TIMEOUT capps=$cn ${cu}MiB" >&2; pkill -KILL -f 'qwen35_9b_flare_hybrid_serve|EngineCore|vllm' 2>/dev/null||true; break; }
      sleep 5; done
  fi
  docker ps -q --filter "name=swe_ep_" 2>/dev/null | xargs -r docker rm -f >/dev/null 2>&1 || true
}
trap cleanup EXIT

preflight() {
  local dl=$((SECONDS+600))
  while :; do local n u; n=$(gpu_capps); u=$(gpu_used)
    [[ "$u" -lt "$GPU_CEIL" ]] && { echo "[preflight] clear capps=$n ${u}MiB (gnome baseline tolerated)" >&2; return 0; }
    [[ $SECONDS -gt $dl ]] && { echo "[preflight] TIMEOUT capps=$n ${u}MiB" >&2; return 1; }
    echo "[preflight] busy capps=$n ${u}MiB" >&2; sleep 10; done
}

wait_ready() {
  local port=$1 scope=$2 dl=$3
  local deadline=$((SECONDS+dl)) grace=$((SECONDS+40)) seen=0
  while :; do
    curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1 && { echo "[ready] :${port} up" >&2; return 0; }
    local st; st=$(systemctl --user is-active "${scope}.scope" 2>/dev/null || true)
    if [[ "$st" == active || "$st" == activating ]]; then seen=1
    elif [[ $seen -eq 1 ]]; then echo "[ready] scope $scope died state=$st" >&2; return 2
    elif [[ $SECONDS -gt $grace ]]; then echo "[ready] scope $scope never active state=$st" >&2; return 2; fi
    [[ $SECONDS -gt $deadline ]] && { echo "[ready] :${port} BOOT TIMEOUT" >&2; return 1; }
    sleep 5; done
}

echo "==== C46-ITER2 ARM=$ARM C=$C CLAMP=$CLAMP START $(date -u +%FT%TZ) plan=$PLAN ====" >&2
preflight || { echo "[$ARM] preflight failed" >&2; exit 1; }

echo "[$ARM] launching server scope=$SCOPE port=$PORT (iter2 twin@K1 export, mask=248077, gmu0.74)" >&2
ACTIVE_SCOPE="$SCOPE"
systemd-run --user --scope --unit="$SCOPE" \
  -p MemoryMax=24G -p MemorySwapMax=6G \
  bash -c "$LAUNCH" > "$slog" 2>&1 &
wait_ready "$PORT" "$SCOPE" "$BOOT_DL" || { echo "[$ARM] server not ready" >&2; exit 1; }

# ---- HARD GATE-ON ASSERT (W-2 causal draft-verify) --------------------------------
# The engine logs "FLARE W-1b copy draft-and-verify gate: <bool>" at sampler build
# (before /v1/models goes ready). Assert it is True; fail-closed if the gate is OFF
# (a gate-OFF boot here would silently duplicate the banked comparator). The W-2 CAUSAL
# path itself is code-gated on _w1_on (qwen3_5_flare.py L819-847, "RUNG W-2 byte-faithful
# redesign"); per-episode w1[on=True vfwd=..] verify counters (parsed post-run) are the
# live proof the causal verify forward actually fires.
gate_dl=$((SECONDS+60))
while :; do
  if grep -q "FLARE W-1b copy draft-and-verify gate: True" "$slog" 2>/dev/null; then
    echo "[$ARM] GATE-ON ASSERT PASS: $(grep -m1 'draft-and-verify gate:' "$slog")" >&2; break
  fi
  if grep -q "FLARE W-1b copy draft-and-verify gate: False" "$slog" 2>/dev/null; then
    echo "[$ARM] GATE-ON ASSERT FAIL: gate booted OFF — aborting (would duplicate banked gate-OFF twin)" >&2; exit 5
  fi
  [[ $SECONDS -gt $gate_dl ]] && { echo "[$ARM] GATE-ON ASSERT TIMEOUT: no gate line in server log after 60s" >&2; exit 5; }
  sleep 3
done

# ---- END-TO-END PROXY SMOKE: route a REAL prior divergence chat dump through the
# certified shim against the LIVE iter-2 server; assert a well-formed clamped stream.
SMOKE_DUMP=runs/k_gate_c46/diffusion/dumps_shard_1/chat_0001.json
if [[ -s "$SMOKE_DUMP" ]]; then
  echo "[$ARM] proxy smoke: clamp a real read through the shim (dump=$SMOKE_DUMP)" >&2
  LUMO_PROXY_READCLAMP_LIMIT="$CLAMP" $PY "$PROXY_SCRIPT" --host 127.0.0.1 --port 33099 \
    --upstream "http://127.0.0.1:${PORT}/v1" --max-tokens 8192 > "$OUTBASE/logs/smoke_proxy.log" 2>&1 &
  SMOKE_PID=$!
  sleep 3
  $PY - "$SMOKE_DUMP" "$OUTBASE/logs/smoke_result.json" <<'PYEOF' > "$OUTBASE/logs/smoke_stdout.txt" 2>&1
import json, sys, urllib.request
dump, outp = sys.argv[1], sys.argv[2]
body = json.load(open(dump))
req = urllib.request.Request("http://127.0.0.1:33099/v1/chat/completions",
                             data=json.dumps(body).encode(), headers={"Content-Type":"application/json"}, method="POST")
raw = urllib.request.urlopen(req, timeout=300).read().decode("utf-8", "replace")
open(outp, "w").write(raw)
tcs=[]; args=""; content=""; saw_done=False
for line in raw.splitlines():
    line=line.strip()
    if not line.startswith("data:"): continue
    p=line[5:].strip()
    if p=="[DONE]": saw_done=True; continue
    if not p: continue
    o=json.loads(p)
    for ch in (o.get("choices") or []):
        d=ch.get("delta") or {}
        if d.get("content"): content+=d["content"]
        for tc in d.get("tool_calls") or []:
            fn=tc.get("function") or {}
            if fn.get("name"): tcs.append(fn["name"])
            if fn.get("arguments"): args+=fn["arguments"]
print("SMOKE tool_calls=", tcs)
print("SMOKE args=", args[:300])
print("SMOKE content_len=", len(content), "saw_done=", saw_done)
ok = saw_done and (len(tcs) > 0 or len(content) > 0)
if "read_file" in tcs:
    print("SMOKE read_file limit present:", '"limit"' in args)
print("SMOKE_OK" if ok else "SMOKE_FAIL")
sys.exit(0 if ok else 3)
PYEOF
  SMOKE_RC=$?
  kill "$SMOKE_PID" 2>/dev/null || true
  cat "$OUTBASE/logs/smoke_stdout.txt" >&2
  if [[ "$SMOKE_RC" -ne 0 ]]; then echo "[$ARM] PROXY SMOKE FAILED rc=$SMOKE_RC — aborting before episodes" >&2; exit 4; fi
  echo "[$ARM] proxy smoke PASS" >&2
else
  echo "[$ARM] proxy smoke SKIPPED (no dump at $SMOKE_DUMP); shim independently CERTIFIED (7ae55d4)" >&2
fi

( while :; do
    echo "$(date -u +%FT%TZ) $(free -m | awk '/Mem:/{printf \"mem_used=%sM mem_avail=%sM\",$3,$7} /Swap:/{printf \" swap_used=%sM\",$3}') gpu=$(gpu_used)MiB capps=$(gpu_capps)"
    sleep 20
  done ) >> "$monlog" 2>&1 &
MONPID=$!

NSHARD=$($PY -c "import json;print(len(json.load(open('$PLAN'))['shards']))")
echo "[$ARM] fan-out over $NSHARD shards (clamp=$CLAMP)" >&2
T_START=$(date +%s)
declare -a PIDS=()
for ((k=0;k<NSHARD;k++)); do
  read IIDS PPORT BSEED < <($PY -c "
import json
p=json.load(open('$PLAN'))['shards'][$k]
print(','.join(p['instance_ids']), p['$PROXY_PORT_KEY'], p['base_seed'])")
  OUT="$OUTBASE/$ARM/shard_$k"
  DUMP="$OUTBASE/$ARM/dumps_shard_$k"
  dlog="$OUTBASE/logs/${ARM}_shard_${k}_driver.log"
  mkdir -p "$OUT" "$DUMP"
  echo "[$ARM shard=$k] n=$(echo $IIDS | tr ',' ' ' | wc -w) pport=$PPORT seed=$BSEED $(date -u +%FT%TZ)" >&2
  LUMO_PROXY_FORCE_SEED="$BSEED" \
  $PY $DRIVER \
    --subset "$SUBSET" --only "$IIDS" \
    --out-root "$OUT" \
    --runtime container \
    --endpoint "http://127.0.0.1:${PORT}/v1" \
    --model "$MODEL" --model-name "$TAG" \
    --repo-cache "$REPO_CACHE" \
    --eval-mode skip \
    --agent-wall-s $AGENT_WALL_S --qwen-max-wall $QWEN_MAX_WALL \
    --max-session-turns $MAX_TURNS \
    --proxy-script "$PROXY_SCRIPT" \
    --proxy-port "$PPORT" \
    --proxy-dump-dir "$DUMP" \
    --proxy-tool-choice "" \
    > "$dlog" 2>&1 &
  PIDS+=($!)
  sleep 3
done

echo "[$ARM] waiting on ${#PIDS[@]} shards: ${PIDS[*]}" >&2
rc_all=0
for pid in "${PIDS[@]}"; do wait "$pid" || rc_all=$?; done
T_END=$(date +%s)
echo "[$ARM] all shards done rc_all=$rc_all $(date -u +%FT%TZ)" >&2
printf '{"arm":"%s","concurrency":%s,"clamp":%s,"wall_start_epoch":%s,"wall_end_epoch":%s,"wall_seconds":%s}\n' \
  "$ARM" "$C" "$CLAMP" "$T_START" "$T_END" "$((T_END-T_START))" > "$OUTBASE/$ARM/arm_timing.json"

kill "$MONPID" 2>/dev/null || true; MONPID=""

echo "[$ARM] stopping server scope=$SCOPE" >&2
systemctl --user stop "${SCOPE}.scope" 2>/dev/null || true
ACTIVE_SCOPE=""
sleep 5
settle=$((SECONDS+180))
while :; do n=$(gpu_capps); u=$(gpu_used)
  [[ "$u" -lt "$GPU_CEIL" ]] && { echo "[$ARM] GPU settled capps=$n ${u}MiB" >&2; break; }
  [[ $SECONDS -gt $settle ]] && { echo "[$ARM] GPU settle timeout capps=$n ${u}MiB" >&2; break; }; sleep 5; done

$PY runs/k_gate_c46/merge_predictions.py "$OUTBASE/$ARM" "$NSHARD" "$OUTBASE/$ARM/predictions.jsonl" >&2
echo "arm=$ARM C=$C clamp=$CLAMP rc_all=$rc_all done=$(date -u +%FT%TZ)" > "$OUTBASE/logs/STATUS_${ARM}.txt"
echo "==== C46-ITER2 ARM=$ARM END rc=$rc_all $(date -u +%FT%TZ) ====" >&2
exit 0
