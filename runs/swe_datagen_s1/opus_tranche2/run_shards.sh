#!/usr/bin/env bash
# ============================================================================
# TRANCHE-2 DETACHED SHARD RUNNER  (task-#125 RUN step; launch-only discipline)
# ----------------------------------------------------------------------------
# Drives the 116 coverage-targeted Opus-teacher episodes in 4 shards of ~29,
# each shard a WINDOW of target_ids.txt (windows drive gen_next.sh's target-order
# pick, and let us pull+rmi per shard to hold disk). Per shard:
#   (a) PULL the shard window's images (<=3 concurrent, disk-floor guarded).  We
#       window rather than prefix-pull (pull_wave.sh's mode) precisely BECAUSE we
#       rmi each scored shard's images (step e) — a prefix pull would re-fetch the
#       images we just deleted and defeat the disk hold. Same driver (pull_and_tag.sh),
#       same <=3 concurrency + disk-floor guard as pull_wave.sh, just windowed.
#   (b) GEN the shard's episodes SEQUENTIALLY via gen_next.sh (full walls; detached
#       so no 600s tool cap). gen_next picks the next undone-present id in target
#       order == this window's ids (prior windows are EP_DONE; later windows unpulled).
#   (c) SCORE the finished shard in a docker wave via the tranche-1 dual-source
#       scoring path (datagen_score.sh), isolated to a per-shard gen-root+batchdir
#       (symlinks) so RUN_ID + report + no_prediction accounting are per-shard.
#   (d) EXTRACT keepers ISOLATED into opus_tranche2/keepers/ (never production
#       keepers.jsonl — promotion is a separate MONITORED step). Writes the shard
#       keepers_extract_summary.json (extract_keepers.py -> <batchdir>/...).
#   (e) docker rmi the scored shard's images to hold disk.
#   (f) emit a [state] line: shard n/4, eps done, keepers so far, $ spent, disk free.
#
# GUARDS (checked every episode unless noted):
#   * STOP-file opus_tranche2/STOP  -> abort cleanly.
#   * $-spent (sum usage_adapter.jsonl, Opus 4.8 pricing) >= $230 -> stop.
#   * pool-projection (334 base + keepers-so-far) >= 400 -> log FLOOR reached and
#     keep running to shard end, then stop. (keepers only change at shard-end
#     extraction, so this is a shard-boundary check by construction.)
#   * adapter dead -> restart ONCE; twice-dead -> stop with [state] ADAPTER_DOWN.
#   * a shard whose scoring is MAJORITY no_prediction -> [state] INFRA_SUSPECT + STOP
#     (never spend $ into an infra hole).
# ============================================================================
set -uo pipefail
cd /home/mark/qwen_diffusion
export SWE_DOCKER_CMD=docker
export SUDO_ASKPASS="${SUDO_ASKPASS:-}"

T=runs/swe_datagen_s1/opus_tranche2
HERE=runs/swe_datagen_s1
PY=/home/mark/qwen_diffusion/.venv/bin/python
PORT=30050
CLAUDE_PID=15765
ULOG=$T/usage_adapter.jsonl
LOG=$T/orch.log
PROG=$T/gen_progress.jsonl
KEEPERS=$T/keepers
SRCMAP=$T/batch/sources.json
TARGETS=$T/target_ids.txt
ENVELOPE='{"temperature":null,"top_p":null,"top_k":null,"seed_base":null,"generator":"opus-4.8-adapter"}'

SHARD_SIZE=29
NSHARDS=4
POOL_BASE=334
POOL_FLOOR=400
BUDGET_USD=230
PULL_DISK_FLOOR_GB=${PULL_DISK_FLOOR_GB:-300}
QWEN_WALL=${QWEN_WALL:-480}
AGENT_WALL=${AGENT_WALL:-520}

mkdir -p "$KEEPERS"
touch "$PROG"

say(){ echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }
state(){ echo "[state] $(date -u +%FT%TZ) $*" >> "$LOG"; }

df_avail_gb(){ df -B1 --output=avail /home/mark | tail -1 | awk '{printf "%.0f", $1/1e9}'; }
keepers_count(){ [ -f "$KEEPERS/keepers.jsonl" ] && grep -cve '^$' "$KEEPERS/keepers.jsonl" || echo 0; }

# --- $-spent: Opus 4.8 pricing over the isolated tranche-2 usage log -----------
# input $5.00/1M, output $25.00/1M, cache-read $0.50/1M (0.1x), cache-write $6.25/1M (1.25x)
spent_usd(){ "$PY" - "$ULOG" <<'PY'
import json,sys
P_IN,P_OUT,P_CR,P_CW=5.0/1e6,25.0/1e6,0.50/1e6,6.25/1e6
tot=0.0
try:
    for l in open(sys.argv[1]):
        l=l.strip()
        if not l: continue
        r=json.loads(l)
        tot += (r.get("uncached_input_tokens",0)*P_IN + r.get("cache_read_input_tokens",0)*P_CR
                + r.get("cache_creation_input_tokens",0)*P_CW + r.get("completion_tokens",0)*P_OUT)
except FileNotFoundError:
    pass
print(f"{tot:.2f}")
PY
}

src_of(){ "$PY" -c '
import json,sys
try: print(json.load(open(sys.argv[1])).get(sys.argv[2],"swe_gym"))
except Exception: print("swe_gym")' "$SRCMAP" "$1"; }

# --- adapter health + restart-once policy -------------------------------------
adapter_alive(){ curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; }
RESTARTS=0
restart_adapter(){
  setsid bash "$T/adapter_up.sh" "$CLAUDE_PID" "$PORT" "/home/mark/qwen_diffusion/$ULOG" \
      >> "$T/adapter.log" 2>&1 < /dev/null &
  sleep 1
  # record real (reparented) adapter pid
  local rp=""
  for p in $(pgrep -f 'opus_openai_adapter.py'); do
    if tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -q "OPUS_ADAPTER_USAGE_LOG=.*opus_tranche2/usage_adapter.jsonl"; then rp=$p; fi
  done
  [ -n "$rp" ] && echo "$rp" > "$T/adapter.pid"
  for i in $(seq 1 40); do adapter_alive && break; sleep 1; done
}
check_adapter(){   # returns 1 (stop) if the adapter cannot be kept alive
  adapter_alive && return 0
  RESTARTS=$((RESTARTS+1))
  if [ "$RESTARTS" -gt 1 ]; then state "ADAPTER_DOWN twice-dead (restarts=$RESTARTS) -> stop"; return 1; fi
  say "adapter dead -> restart attempt $RESTARTS"
  restart_adapter
  if adapter_alive; then say "adapter restarted OK (attempt $RESTARTS, pid=$(cat $T/adapter.pid 2>/dev/null))"; return 0; fi
  state "ADAPTER_DOWN restart_attempt_$RESTARTS failed to bind -> stop"; return 1
}

# --- per-shard score (isolated) -----------------------------------------------
# Builds a per-shard gen-root (symlinks to the window's episode dirs that produced
# a prediction) + per-shard batchdir (symlinks to the shared dataset files) so
# datagen_score.sh aggregates ONLY this shard's predictions, RUN_ID is unique, and
# each harness gets only its own source's rows. Echoes the merged report path.
score_shard(){
  local SH=$1; shift
  local wids=("$@")
  local SB="$T/score_shard$SH" SG="$T/gen_shard$SH"
  rm -rf "$SB" "$SG"; mkdir -p "$SB" "$SG"
  local f
  for f in dataset.json dataset_gym.json dataset_verified.json sources.json; do
    ln -sfn "/home/mark/qwen_diffusion/$T/batch/$f" "$SB/$f"
  done
  local iid
  for iid in "${wids[@]}"; do
    [ -f "$T/gen/$iid/verified/predictions.jsonl" ] && ln -sfn "/home/mark/qwen_diffusion/$T/gen/$iid" "$SG/$iid"
  done
  bash "$HERE/datagen_score.sh" "$SB" "$SG" 4 >> "$LOG" 2>&1
  echo "$SB/score/datagen-eval.score_shard$SH.json"
}

# --- majority-no_prediction infra check ---------------------------------------
# no_pred = shard ids that HAVE a prediction but appear in NONE of the report
# buckets (resolved/unresolved/empty/error). Majority no_pred => harness/infra hole.
majority_no_pred(){   # <report> <gen_shard_dir> <window-ids...>  -> prints "NOPRED SCORED VERDICT"
  local rep="$1" sg="$2"; shift 2
  "$PY" - "$rep" "$sg" "$@" <<'PY'
import json,sys,glob,os
rep=sys.argv[1]; sg=sys.argv[2]; wids=sys.argv[3:]
try: d=json.load(open(rep))
except Exception: d={}
scored=set()
for k in ("resolved_ids","unresolved_ids","empty_patch_ids","error_ids"):
    scored |= set(d.get(k,[]) or [])
# predicted = window ids that had a non-empty prediction file
pred=set()
for iid in wids:
    fs=glob.glob(os.path.join(sg,iid,"verified","predictions.jsonl"))
    for fp in fs:
        for l in open(fp):
            if l.strip(): pred.add(iid)
nopred = pred - scored
verdict = "INFRA_SUSPECT" if (len(pred)>0 and len(nopred) > len(pred)//2) else "OK"
print(f"{len(nopred)} {len(pred & scored)} {verdict}")
PY
}

# ============================================================================
mapfile -t TIDS < <(sed '/^$/d' "$TARGETS")
PRESENT0=0
for iid in "${TIDS[@]}"; do slug=${iid/__/_1776_}; docker image inspect "swebench/sweb.eval.x86_64.${slug}:latest" >/dev/null 2>&1 && PRESENT0=$((PRESENT0+1)); done
state "START runner pid=$$ shards=$NSHARDS shard_size=$SHARD_SIZE n_targets=${#TIDS[@]} present_images=$PRESENT0 budget=\$$BUDGET_USD pool_base=$POOL_BASE floor=$POOL_FLOOR spent=\$$(spent_usd) keepers=$(keepers_count) disk=$(df_avail_gb)GB"

STOP_NOW=0
POOL_HIT=0
for SH in $(seq 0 $((NSHARDS-1))); do
  [ "$STOP_NOW" -eq 1 ] && break
  [ -f "$T/STOP" ] && { state "STOP_FILE at shard $SH start -> abort"; break; }

  lo=$((SH*SHARD_SIZE)); WIN=("${TIDS[@]:$lo:$SHARD_SIZE}")
  say "==== SHARD $SH window ids=${#WIN[@]} (target idx $lo..$((lo+${#WIN[@]}-1))) $(date -u +%FT%TZ) ===="

  # (a) PULL the window (<=3 concurrent, disk-floor guarded, idempotent skip) -----
  say "pull shard $SH window (<=3 concurrent, floor=${PULL_DISK_FLOOR_GB}GB)"
  for iid in "${WIN[@]}"; do
    slug=${iid/__/_1776_}
    docker image inspect "swebench/sweb.eval.x86_64.${slug}:latest" >/dev/null 2>&1 && continue
    avail=$(df_avail_gb)
    if [ "$avail" -lt "$PULL_DISK_FLOOR_GB" ]; then say "DISK FLOOR $avail<$PULL_DISK_FLOOR_GB -> stop pull"; break; fi
    while [ "$(jobs -rp | wc -l)" -ge 3 ]; do wait -n 2>/dev/null || break; done
    src=$(src_of "$iid")
    ( timeout 900 bash "$HERE/pull_and_tag.sh" "$iid" "$T/pull.jsonl" "$src" \
        || say "pull $iid FAILED (env_unavailable)" ) >> "$LOG" 2>&1 &
  done
  wait
  npres=0; for iid in "${WIN[@]}"; do slug=${iid/__/_1776_}; docker image inspect "swebench/sweb.eval.x86_64.${slug}:latest" >/dev/null 2>&1 && npres=$((npres+1)); done
  say "shard $SH images present=$npres/${#WIN[@]} disk=$(df_avail_gb)GB"

  # (b) GEN the shard's episodes sequentially -------------------------------------
  while :; do
    NEXT=""
    for iid in "${WIN[@]}"; do
      [ -f "$T/gen/$iid/EP_DONE" ] && continue
      slug=${iid/__/_1776_}
      docker image inspect "swebench/sweb.eval.x86_64.${slug}:latest" >/dev/null 2>&1 || continue
      NEXT="$iid"; break
    done
    [ -z "$NEXT" ] && break
    # ---- per-episode GUARDS ----
    [ -f "$T/STOP" ] && { state "STOP_FILE mid-shard $SH -> abort"; STOP_NOW=1; break; }
    SP=$(spent_usd)
    if awk "BEGIN{exit !($SP>=$BUDGET_USD)}"; then state "BUDGET_STOP spent=\$$SP >= \$$BUDGET_USD (shard $SH)"; STOP_NOW=1; break; fi
    check_adapter || { STOP_NOW=1; break; }
    # ---- run ONE episode (full walls) ----
    bash "$T/gen_next.sh" 1 "$QWEN_WALL" "$AGENT_WALL" >> "$LOG" 2>&1
  done

  # count episodes done in this window
  EPDONE=0; for iid in "${WIN[@]}"; do [ -f "$T/gen/$iid/EP_DONE" ] && EPDONE=$((EPDONE+1)); done

  # (c) SCORE the shard (docker wave, isolated) -----------------------------------
  say "score shard $SH ($EPDONE episodes) $(date -u +%FT%TZ)"
  REP=$(score_shard "$SH" "${WIN[@]}")
  RES=0; [ -f "$REP" ] && RES=$("$PY" -c "import json,sys;print(len(json.load(open(sys.argv[1])).get('resolved_ids',[])))" "$REP" 2>/dev/null || echo 0)
  read -r NOPRED NSCORED NPVERDICT < <(majority_no_pred "$REP" "$T/gen_shard$SH" "${WIN[@]}")
  say "shard $SH score: resolved=$RES scored=$NSCORED no_prediction=$NOPRED verdict=$NPVERDICT report=$REP"

  # (d) EXTRACT keepers ISOLATED ---------------------------------------------------
  "$PY" "$HERE/extract_keepers.py" "$T/score_shard$SH" "score_shard$SH" "$T/gen_shard$SH" "$REP" "$KEEPERS" "$ENVELOPE" >> "$LOG" 2>&1 || say "extract_keepers rc=$? (shard $SH)"

  # (e) rmi the scored shard's images ---------------------------------------------
  for iid in "${WIN[@]}"; do
    slug=${iid/__/_1776_}
    docker rmi -f "swebench/sweb.eval.x86_64.${slug}:latest" >/dev/null 2>&1 || true
    docker rmi -f "sweb.eval.x86_64.${iid}:latest" >/dev/null 2>&1 || true
  done

  # (f) [state] line ---------------------------------------------------------------
  KE=$(keepers_count)
  state "shard $((SH+1))/$NSHARDS eps_done=$EPDONE resolved=$RES keepers=$KE spent=\$$(spent_usd) disk=$(df_avail_gb)GB no_pred=$NOPRED verdict=$NPVERDICT"

  # INFRA hole -> stop before spending into it
  if [ "$NPVERDICT" = "INFRA_SUSPECT" ]; then state "INFRA_SUSPECT shard $SH majority no_prediction ($NOPRED>$((NSCORED))) -> STOP"; STOP_NOW=1; break; fi
  # pool floor -> finish this shard (done) then stop
  if [ $((POOL_BASE + KE)) -ge "$POOL_FLOOR" ]; then say "POOL FLOOR reached: $POOL_BASE+$KE>=$POOL_FLOOR"; state "POOL_FLOOR_REACHED keepers=$KE (pool=$((POOL_BASE+KE))) -> stop after shard $((SH+1))"; POOL_HIT=1; break; fi
  [ "$STOP_NOW" -eq 1 ] && break
done

state "RUNNER_DONE stop_now=$STOP_NOW pool_hit=$POOL_HIT keepers=$(keepers_count) spent=\$$(spent_usd) disk=$(df_avail_gb)GB $(date -u +%FT%TZ)"
