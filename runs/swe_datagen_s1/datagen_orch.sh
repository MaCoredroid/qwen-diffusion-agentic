#!/usr/bin/env bash
# ============================================================================
# STAGE-1 SWE-Gym DATA-GEN ORCHESTRATOR — image-cycling, self-bounded, resumable.
# ============================================================================
# Goal (task/design §1): harvest 600-1000 (floor 400) VERIFIED-CORRECT SWE-Gym
# trajectories for the SFT pool, best-of-1 @ measured envelope yield 0.25, over
# DISTINCT instances (coverage > repeats), teacher = stock Qwen3.5-9B AR.
#
# Images (~3.5-4GB) cannot all coexist on the 3TB disk, so we CYCLE per batch:
#   [pull ~BATCH_SIZE images]  (docker-only, disk-floor guarded)
#     -> [gen episodes @ c=C]  (ONE AR server in the RAM cage; no docker eval up)
#     -> [score/filter batch]  (official fork docker harness; NO GPU server up)
#     -> [KEEP resolved trajectories -> keepers/keepers.jsonl]  (CPU)
#     -> [docker rmi the batch images]  (reclaim disk)
#     -> [record attempts + update STATUS + re-check stop conditions] -> next
# ONE heavy job-class at a time (pull|gen|score|rmi never overlap): pull+score+rmi
# are docker-only with NO model server; gen boots exactly one caged AR server.
#
# RESUME: attempts.jsonl is the source of truth; ledger.py nextbatch draws the
# next batch as frontier-order MINUS already-attempted. Kill/restart is safe.
#
# STOP conditions (ledger.py state): keepers>=TARGET (DONE_TARGET) | frontier
# exhausted (DONE_EXHAUSTED) | rolling-200 yield < 0.10 (KILL_YIELD_COLLAPSE,
# halt+flag). A per-run MAX_CYCLES backstop bounds a single invocation.
#
# DRY-RUN: DATAGEN_DRYRUN=1 swaps pull/gen/score/rmi for mock_batch.py stubs
# (no GPU, no docker); keeper-extraction + ledger + stop-conditions run FOR REAL.
#
# LAUNCH: build + launch detached, then the monitor watches externally, e.g.
#   setsid bash runs/swe_datagen_s1/datagen_orch.sh >runs/swe_datagen_s1/logs/orch.log 2>&1 &
#   echo $! > runs/swe_datagen_s1/orch.pid
# ============================================================================
set -uo pipefail
cd /home/mark/qwen_diffusion
export SUDO_ASKPASS="${SUDO_ASKPASS:-/tmp/claude-1000/-home-mark/8f712353-03d0-4607-ac3a-cba8072f9d36/scratchpad/askpass.sh}"

ROOT=runs/swe_datagen_s1
PY=.venv/bin/python
FORK_PY=runs/stage0_swegym_probe/.venv-swegym/bin/python   # has swebench + spec map

# ---- knobs (all overridable) -----------------------------------------------
TARGET="${TARGET:-1000}"
FLOOR="${FLOOR:-400}"
BATCH_SIZE="${BATCH_SIZE:-50}"
C="${C:-4}"
KILL_YIELD="${KILL_YIELD:-0.10}"
KILL_WINDOW="${KILL_WINDOW:-200}"
MAX_CYCLES="${MAX_CYCLES:-100}"
DISK_FLOOR_GB="${DISK_FLOOR_GB:-300}"
DATAGEN_DRYRUN="${DATAGEN_DRYRUN:-0}"
ENVELOPE_JSON="${ENVELOPE_JSON:-{\"temperature\":0.6,\"top_p\":0.95,\"top_k\":20,\"seed_base\":1234}}"

FRONTIER="$ROOT/frontier.json"
ATTEMPTS="$ROOT/attempts.jsonl"
KEEPERS="$ROOT/keepers/keepers.jsonl"
STATUS="$ROOT/DATAGEN_STATUS.txt"
DONE="$ROOT/DATAGEN_DONE.txt"
KILLF="$ROOT/DATAGEN_KILL.txt"
mkdir -p "$ROOT/logs" "$ROOT/batches" "$ROOT/keepers"
# RESUME-CRITICAL: create attempts.jsonl only if absent; NEVER truncate on launch.
[[ -f "$ATTEMPTS" ]] || : > "$ATTEMPTS"
[[ -f "$KEEPERS" ]] || : > "$KEEPERS"

stamp(){ date -u +%FT%TZ; }
df_avail_gb(){ df -B1 --output=avail /home/mark | tail -1 | awk '{printf "%.0f", $1/1e9}'; }

# ---- cleanup trap: never leave a server scope or episode container behind ----
cleanup(){
  systemctl --user stop datagen_ar.scope 2>/dev/null || true
  sudo -A docker ps -q --filter "name=swe_ep_" 2>/dev/null | xargs -r sudo -A docker rm -f >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "==================== DATAGEN ORCH START $(stamp) dryrun=$DATAGEN_DRYRUN ===================="
echo "[cfg] TARGET=$TARGET FLOOR=$FLOOR BATCH=$BATCH_SIZE C=$C kill_yield=$KILL_YIELD/win=$KILL_WINDOW disk_floor=${DISK_FLOOR_GB}GB max_cycles=$MAX_CYCLES"
free -g | awk 'NR<=2'; df -h /home/mark | tail -1

# ---- frontier (build once; idempotent; firewall-asserting) ------------------
if [[ ! -s "$FRONTIER" ]]; then
  echo "[frontier] building (fork venv for spec-map screen) $(stamp)"
  BP="$FORK_PY"; [[ -x "$BP" ]] || BP="$PY"
  "$BP" "$ROOT/build_frontier.py" || { echo "[frontier] BUILD FAILED (firewall/eligibility) $(stamp)"; exit 2; }
fi
echo "[frontier] n=$($PY -c 'import json;print(json.load(open("'"$FRONTIER"'"))["n_frontier"])')"

# ============================================================================
# main cycle loop
# ============================================================================
cycle=0
while :; do
  # ---- stop-condition check BEFORE spending a cycle -------------------------
  ST=$("$PY" "$ROOT/ledger.py" state "$ATTEMPTS" "$KEEPERS" "$FRONTIER" "$STATUS" \
        --target "$TARGET" --floor "$FLOOR" --kill-yield "$KILL_YIELD" --kill-window "$KILL_WINDOW")
  VERDICT=$("$PY" -c 'import json,sys;print(json.loads(sys.argv[1])["verdict"])' "$ST")
  echo "[state] $ST"
  case "$VERDICT" in
    DONE_TARGET)    echo "[stop] TARGET reached $(stamp)"; echo "DONE_TARGET $(stamp) $ST" > "$DONE"; break;;
    DONE_EXHAUSTED) echo "[stop] frontier exhausted $(stamp)"; echo "DONE_EXHAUSTED $(stamp) $ST" > "$DONE"; break;;
    KILL_YIELD_COLLAPSE)
        echo "[KILL] rolling yield < $KILL_YIELD over $KILL_WINDOW attempts $(stamp)"
        echo "KILL_YIELD_COLLAPSE $(stamp) $ST" > "$KILLF"; break;;
  esac
  cycle=$((cycle+1))
  GEN_RC=0   # reset per cycle; only the REAL gen branch overwrites it
  if [[ $cycle -gt $MAX_CYCLES ]]; then echo "[stop] MAX_CYCLES=$MAX_CYCLES backstop $(stamp)"; break; fi

  # ---- select the next batch (resume-aware) --------------------------------
  BID="batch_$(printf '%04d' "$cycle")_$(date -u +%Y%m%dT%H%M%SZ)"
  BATCHDIR="$ROOT/batches/$BID"; GEN_ROOT="$BATCHDIR/gen"
  mkdir -p "$BATCHDIR" "$GEN_ROOT"
  "$PY" "$ROOT/ledger.py" nextbatch "$FRONTIER" "$ATTEMPTS" "$BATCH_SIZE" > "$BATCHDIR/batch_ids.txt"
  NIDS=$(grep -c . "$BATCHDIR/batch_ids.txt" || echo 0)
  echo "======== [CYCLE $cycle] $BID n=$NIDS $(stamp) ========"
  if [[ "$NIDS" -eq 0 ]]; then echo "[cycle] no unattempted ids -> exhausted"; echo "DONE_EXHAUSTED $(stamp)" > "$DONE"; break; fi

  # ---- disk floor guard ----------------------------------------------------
  AV=$(df_avail_gb)
  echo "[disk] avail=${AV}GB floor=${DISK_FLOOR_GB}GB"
  if [[ "$AV" -lt "$DISK_FLOOR_GB" ]]; then
    echo "[HALT] disk under floor BEFORE pull ($AV<$DISK_FLOOR_GB) $(stamp)"
    echo "HALT_DISK_FLOOR $(stamp) avail=${AV}GB" > "$KILLF"; break
  fi

  # ---- build the batch dataset + subsets -----------------------------------
  BASE_SEED=$(( 1234 + cycle * 1000000 ))
  "$PY" "$ROOT/build_batch_dataset.py" "$BATCHDIR" "@$BATCHDIR/batch_ids.txt" "$C" "$BASE_SEED" \
      > "$BATCHDIR/logs_build_dataset.txt" 2>&1 || { echo "[cycle] dataset build failed"; continue; }

  if [[ "$DATAGEN_DRYRUN" == "1" ]]; then
    # ---- DRY-RUN: mock the docker/GPU-heavy steps ------------------------
    echo "[dryrun] mock pull (noop)"
    echo "[dryrun] mock gen"
    "$PY" "$ROOT/mock_batch.py" gen "$BATCHDIR" "$GEN_ROOT" "$C"   > "$BATCHDIR/logs_gen.txt" 2>&1
    echo "[dryrun] mock score"
    "$PY" "$ROOT/mock_batch.py" score "$BATCHDIR" "$GEN_ROOT"      > "$BATCHDIR/logs_score.txt" 2>&1
  else
    # ---- REAL: pull -> gen (one caged server) -> score -------------------
    echo "[pull] $(stamp)"
    PULL_DISK_FLOOR_GB="$DISK_FLOOR_GB" bash "$ROOT/datagen_pull.sh" "$BATCHDIR" > "$BATCHDIR/logs_pull.txt" 2>&1 || {
      echo "[pull] aborted (disk floor) -> halt"; echo "HALT_DISK_FLOOR $(stamp)" > "$KILLF"; break; }
    echo "[gen] $(stamp)"
    bash "$ROOT/datagen_gen.sh" "$BATCHDIR" "$GEN_ROOT" "$C" > "$BATCHDIR/logs_gen.txt" 2>&1
    GEN_RC=$?
    echo "[gen] rc=$GEN_RC $(stamp)"
    echo "[score] $(stamp)"
    bash "$ROOT/datagen_score.sh" "$BATCHDIR" "$GEN_ROOT" "$C" > "$BATCHDIR/logs_score.txt" 2>&1
    echo "[score] rc=$? $(stamp)"
  fi

  # ---- KEEP resolved trajectories (real extraction either way) -------------
  echo "[keepers] extract $(stamp)"
  REPORT=$(ls "$BATCHDIR/score/"*."$BID".json 2>/dev/null | head -1)
  [[ -z "$REPORT" ]] && REPORT=$(ls "$BATCHDIR/score/"*.json 2>/dev/null | grep -v timing | head -1)
  "$PY" "$ROOT/extract_keepers.py" "$BATCHDIR" "$BID" "$GEN_ROOT" "$REPORT" "$ROOT/keepers" "$ENVELOPE_JSON" \
      > "$BATCHDIR/logs_keepers.txt" 2>&1
  tail -1 "$BATCHDIR/logs_keepers.txt"

  # ---- record attempts (drives resume + rolling yield) ---------------------
  # If gen FAILED (rc!=0 -> preflight timeout / server never booted), the whole
  # batch is INFRA-INVALID: flag every row so it is excluded from yield, the kill
  # window and coverage, and the ids stay re-drawable. The kill must judge the
  # TEACHER, never our infra failure.
  REC_FLAGS=()
  if [[ "${GEN_RC:-0}" -ne 0 && "$DATAGEN_DRYRUN" != "1" ]]; then
    echo "[record] gen rc=$GEN_RC -> recording batch INFRA-INVALID (excluded from yield/kill; ids re-drawable)"
    REC_FLAGS=(--infra-invalid "gen_rc=${GEN_RC}")
  fi
  "$PY" "$ROOT/ledger.py" record "$BATCHDIR" "$BID" "$ATTEMPTS" "${REC_FLAGS[@]}"

  # ---- rmi the batch images (reclaim disk) ---------------------------------
  if [[ "$DATAGEN_DRYRUN" == "1" ]]; then
    echo "[dryrun] mock rmi (noop)"
  else
    bash "$ROOT/datagen_rmi.sh" "$BATCHDIR" >> "$BATCHDIR/logs_pull.txt" 2>&1 || true
  fi

  # ---- update STATUS + loop ------------------------------------------------
  "$PY" "$ROOT/ledger.py" state "$ATTEMPTS" "$KEEPERS" "$FRONTIER" "$STATUS" \
      --target "$TARGET" --floor "$FLOOR" --kill-yield "$KILL_YIELD" --kill-window "$KILL_WINDOW" >/dev/null
  echo "[status] $(cat "$STATUS")"
  echo "[disk] avail=$(df_avail_gb)GB after cycle $cycle"
done

echo "[final] $(cat "$STATUS" 2>/dev/null)"
echo "==================== DATAGEN ORCH END $(stamp) ===================="
