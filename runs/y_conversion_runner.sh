#!/usr/bin/env bash
# SECTION Y conversion runner — full-trajectory AR-self-distillation @12288, SEGMENTED,
# DETACHED, CAGED, with BOTH standing mid-training guards (the pilots' lessons made binding).
#
# X.2 KILLED (KILL-T1 49->0/63 + loop-halt canary 3->6/8) because a read-arg-ONLY splice
# corroded the broad arg-emission conditional AND the KL guard (single-turn, non-read) was
# necessary-but-insufficient. Y fixes the SIGNAL (whole-trajectory AR-greedy CE @12288,
# uniform weight 1.0) and makes the GUARDS standing:
#   (1) KL-to-base 0.05 trip on a HELD probe that now INCLUDES tool-call turns (binding
#       lesson 3) -- every 100-step segment; HARD TRIP -> stop at that checkpoint.
#   (2) KILL-T1 matched-20 exact_args CANARY on the segment checkpoint -- every 100-step
#       segment (binding lesson 2, the guard KL missed in X.2); TRIP on net-loss > 3 vs the
#       banked 49/63 anchor -> stop at the PRIOR checkpoint.
# Both fire at step-0 (KL base==base ~0; KILL-T1 base diffusion-decode vs the 49 anchor) to
# prove the instruments are wired BEFORE any training. (The 8-ep loop canary stays a
# post-training gate, the finisher's job -- binding lesson 4.)
#
# Distillation is SINGLE-STREAM QLoRA-4bit + chunked-CE @12288 (Y.2 demonstrated max) with a
# <=4096 two-stream denoise-preservation co-step every step (Y.5 risk-1). Bit-exact segmented
# resume so the GPU is FREE between segments for the two guards. STOP-file aborts. pidfile.
set -u
ROOT=/home/mark/qwen_diffusion
cd "$ROOT"
export QWEN_DIFFUSION_ROOT="$ROOT"
ENV_PY="$ROOT/.venv-fastdllm/bin/python"   # trainer + KL probe
EVAL_PY="$ROOT/.venv/bin/python"           # KILL-T1 diffusion canary (eval_flare_northstar)

SEED="${SEED:-71201}"
HORIZON="${HORIZON:-400}"                   # Y.3 screen {200,400}; X.2 grounding saturates ~200-300, guards halt earlier if it drifts; cosine fully decays at 400
SEG="${SEG:-100}"                           # binding lesson 2: guards every 100-step segment
BLOCK="${BLOCK:-12288}"                     # Y.2 demonstrated single-stream max
DENOISE="${DENOISE:-1536}"                  # Y.5 risk-1 two-stream denoise-preservation slice (<=4096); memory-fit on 32GB alongside the 12288 distill; 0 disables
KL_CAP="${KL_CAP:-0.05}"
KILLT1_NETLOSS_CAP="${KILLT1_NETLOSS_CAP:-3}"   # binding lesson 2: net-loss > 3 -> trip
KILLT1_ANCHOR="${KILLT1_ANCHOR:-49}"

MERGED="$ROOT/models/qwen3.5-9b-fastdllm-mswe-S-iter2-merged"
DATA="$ROOT/data/swe_y_ar_distill/y_train.json"
KL_PROBE="$ROOT/runs/kraise_reconvert_iter2_y/y_kl_probe.json"
ANCHOR="$ROOT/runs/swe_sft_arm1_iter2/anchor_gate/mswe_S_matched20/ar-vllm-guided/turns.jsonl"
OUT="$ROOT/runs/kraise_reconvert_iter2_y/mswe2_S_y_ardistill_h${HORIZON}_seed${SEED}"
CANARY="$ROOT/runs/kraise_reconvert_iter2_y/canary"
PIDFILE="$ROOT/runs/y_conversion.pid"
STOPFILE="$ROOT/runs/y_conversion.STOP"
TLOG="$OUT/train.log"
KLLOG="$OUT/kl_to_base.jsonl"
KTLOG="$OUT/killt1_canary.jsonl"

mkdir -p "$OUT" "$CANARY"
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT
: > "$TLOG"

ts(){ date -Iseconds; }
log(){ echo "[y-conv $(ts)] $*" | tee -a "$TLOG"; }
gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
wait_gpu_idle(){ local dl=$((SECONDS+300)); while :; do local u; u=$(gpu_used); [[ "$u" -lt 4000 ]] && return 0; [[ $SECONDS -gt $dl ]] && { log "WARN gpu not idle (${u}MiB)"; return 0; }; sleep 5; done; }

# ---------- preflight ----------
[[ -d "$MERGED" ]] || { log "FAIL merged base missing $MERGED"; exit 1; }
[[ -f "$DATA" ]] || { log "FAIL dataset missing $DATA"; exit 1; }
[[ -f "$KL_PROBE" ]] || { log "FAIL kl probe missing $KL_PROBE"; exit 1; }
[[ -f "$ANCHOR" ]] || { log "FAIL anchor turns missing $ANCHOR"; exit 1; }
U=$(gpu_used); [[ "${U:-99999}" -lt 4000 ]] || { log "FAIL GPU busy ${U}MiB"; exit 1; }
log "PREFLIGHT ok gpu=${U}MiB HORIZON=$HORIZON SEG=$SEG BLOCK=$BLOCK DENOISE=$DENOISE KL_CAP=$KL_CAP KILLT1_NETLOSS_CAP=$KILLT1_NETLOSS_CAP seed=$SEED"
log "Y-DESIGN: whole-trajectory AR-greedy CE @$BLOCK, uniform weight 1.0 (X.1/X.2 narrow patches DROPPED) + standing KL + KILL-T1 canary"

# ---------- KL probe (s2 kit; GPU exclusive; returns 3 on trip) ----------
run_kl(){
  local step="$1"; local adapter="$2"
  wait_gpu_idle
  log "[kl] probe step=$step adapter=${adapter:-<base>}"
  set +e
  systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=8G -- \
    "$ENV_PY" "$ROOT/scripts/s2_kl_probe.py" \
      --base "$MERGED" --adapter "$adapter" \
      --probe "$KL_PROBE" --step "$step" --kl-cap "$KL_CAP" --out "$KLLOG" >>"$TLOG" 2>&1
  local rc=$?
  log "[kl] step=$step rc=$rc $(tail -1 "$KLLOG" 2>/dev/null)"
  return $rc
}

# ---------- KILL-T1 matched-20 canary (offline diffusion hybrid_clean + mcnemar vs anchor) ----------
# echoes NETLOSS=<int> ; returns 3 if net-loss > KILLT1_NETLOSS_CAP (binding lesson 2)
run_killt1(){
  local step="$1"; local adapter="$2"    # adapter empty => base (step-0 anchor sanity)
  wait_gpu_idle
  local sub="$CANARY/step${step}"; mkdir -p "$sub"
  log "[killt1] matched-20 canary step=$step adapter=${adapter:-<base>}"
  set +e
  # base anchor (step-0): a NON-EXISTENT adapter path makes the eval load base-only
  # (eval_flare_northstar_hybrid_clean.py:306 adapter=None when path missing) -> the
  # unmodified base twin@K1 diffusion decode vs the banked anchor. Real checkpoints pass
  # the checkpoint dir (no-merge default, the X.2 KILL-T1 precedent).
  local adp="$adapter"; [[ -z "$adp" ]] && adp="$ROOT/runs/kraise_reconvert_iter2_y/__base_anchor_no_adapter__"
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 "$EVAL_PY" "$ROOT/scripts/eval_flare_northstar_hybrid_clean.py" \
    --base-model "$MERGED" --adapter "$adp" --out-dir "$sub" \
    --episode-limit 20 --min-turns 3 --max-turns 6 --block-size 32 \
    --max-new-tokens 384 --temperature 0.0 --seed 20260701 >>"$TLOG" 2>&1
  local brc=$?
  if [[ $brc -ne 0 ]]; then log "[killt1] BATTERY FAILED rc=$brc step=$step"; return 2; fi
  local post="$sub/diffusion_hybrid_forced_grammar_seq_values/turns.jsonl"
  [[ -s "$post" ]] || { log "[killt1] MISSING $post"; return 2; }
  "$ENV_PY" "$ROOT/scripts/swe_sft_arm1_anchor_mcnemar.py" \
    --pre "$ANCHOR" --post "$post" --anchor "$KILLT1_ANCHOR" \
    --out "$sub/killt1_m20_mcnemar.json" >>"$TLOG" 2>&1
  local netloss; netloss=$("$ENV_PY" -c "import json;print(json.load(open('$sub/killt1_m20_mcnemar.json'))['mcnemar']['net_loss_b_minus_c'])" 2>/dev/null)
  local raw; raw=$("$ENV_PY" -c "import json;print(json.load(open('$sub/killt1_m20_mcnemar.json'))['post_sft']['exact_args'])" 2>/dev/null)
  echo "{\"step\":$step,\"net_loss\":${netloss:-null},\"post_exact_args\":\"${raw:-NA}\",\"anchor\":$KILLT1_ANCHOR}" >> "$KTLOG"
  log "[killt1] step=$step post_exact_args=${raw:-NA} net_loss=${netloss:-NA} (cap $KILLT1_NETLOSS_CAP)"
  if [[ -n "$netloss" && "$netloss" -gt "$KILLT1_NETLOSS_CAP" ]]; then return 3; fi
  return 0
}

# ---------- one bit-exact resumable train segment to STOP_AT_STEP ----------
latest_ckpt(){ ls -d "$OUT"/checkpoint-* 2>/dev/null | sed 's#.*checkpoint-##' | sort -n | tail -1; }
run_segment(){
  local stop="$1"
  wait_gpu_idle
  log "SEGMENT -> stop=$stop (resume=auto)"
  set +e
  systemd-run --user --scope -p MemoryMax=30G -p MemorySwapMax=8G -- \
    env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 \
    "$ENV_PY" "$ROOT/scripts/y_distill_train.py" \
      --model "$MERGED" --dataset "$DATA" --output-dir "$OUT" \
      --block-size "$BLOCK" --horizon "$HORIZON" --stop-at-step "$stop" \
      --save-steps "$SEG" --logging-steps 5 --lr 1e-5 --warmup-ratio 0.03 \
      --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 --seed "$SEED" \
      --denoise-slice "$DENOISE" --denoise-weight 1.0 --logits-chunk 2048 \
      --metrics "$OUT/metrics.jsonl" --resume auto >>"$TLOG" 2>&1
  local rc=$?
  log "SEGMENT stop=$stop exited rc=$rc"
  return $rc
}

# ================= STEP-0 SANITY: prove BOTH instruments are wired BEFORE training =========
[[ -f "$STOPFILE" ]] && { log "STOP-file present at start; abort"; exit 1; }
log "STEP-0 KL sanity (base==base => KL~0)"
run_kl 0 ""
log "STEP-0 KILL-T1 anchor (base diffusion-decode vs banked $KILLT1_ANCHOR => net-loss ~0)"
run_killt1 0 ""; s0=$?
[[ $s0 -eq 3 ]] && log "NOTE step-0 base already >3 net-loss vs anchor (instrument reads base drift); proceeding (baseline reference)"

# ================= SEGMENTED TRAIN LOOP with BOTH guards between segments =================
prev=0
for (( stop=SEG; stop<=HORIZON; stop+=SEG )); do
  [[ -f "$STOPFILE" ]] && { log "STOP-file present -> halt before stop=$stop"; break; }
  run_segment "$stop"; src=$?
  if [[ $src -ne 0 ]]; then log "FAIL segment rc=$src at stop=$stop; halting"; touch "$STOPFILE"; break; fi
  now=$(latest_ckpt); [[ -z "$now" ]] && { log "FAIL no checkpoint after stop=$stop"; touch "$STOPFILE"; break; }
  ck="$OUT/checkpoint-$now"

  # GUARD 1: KL-to-base
  run_kl "$now" "$ck"; klrc=$?
  if [[ $klrc -eq 3 ]]; then
    log "HARD TRIP: KL>$KL_CAP at checkpoint-$now -> STOP at this checkpoint (promote target = checkpoint-$now)"
    touch "$STOPFILE"; break
  fi
  # GUARD 2: KILL-T1 matched-20 canary (net-loss > cap -> stop at PRIOR checkpoint)
  run_killt1 "$now" "$ck"; ktrc=$?
  if [[ $ktrc -eq 3 ]]; then
    log "HARD TRIP: KILL-T1 net-loss > $KILLT1_NETLOSS_CAP at checkpoint-$now -> STOP at PRIOR checkpoint (promote target = checkpoint-$prev)"
    touch "$STOPFILE"; break
  elif [[ $ktrc -eq 2 ]]; then
    log "WARN KILL-T1 canary instrument error at checkpoint-$now (non-trip); continuing but FLAG for finisher"
  fi
  prev="$now"
done

log "Y conversion loop DONE (last checkpoint = checkpoint-$(latest_ckpt)); GPU idle on exit"
wait_gpu_idle
exit 0
