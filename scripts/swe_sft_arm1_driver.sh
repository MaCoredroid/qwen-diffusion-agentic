#!/usr/bin/env bash
# SWE-SFT arm-1 (M_swe_S) -- SELF-VERIFYING DETACHED LAUNCHER.
#
# AMENDED (monitor GPU->training-handover decision, resolution 2): drives the
# AR-STYLE SINGLE-STREAM CAUSAL QLoRA trainer (scripts/swe_sft_arm1_qlora_train.py),
# NOT the two-stream FLARE trainer. Rationale + evidence: the two-stream path
# materialises [2L, vocab] logits and OOM'd above block 8192; the plain-AR-SFT-
# then-reconvert path is certified by #29 (convert_after_rl_result.md, b019b86:
# McNemar zero net-loss, 2 seeds) which enforces train==serve parity at the
# CONVERSION stage, not during SFT. See swe_tuning_campaign_design.md STATUS block.
#
# Steps:
#   1. PREFLIGHT  -- refuses to run while the GPU is busy (never fights the datagen)
#                    and requires host-RAM headroom.
#   2. BLOCK LADDER -- the design block_size 32768 (and 24576) OOM even single-stream
#      (measured: the O(L^2) eager full-attention AND the GDN torch fallback are the
#      walls). This trainer installs SDPA causal attention (this process only) + the
#      fla fused GDN kernel (FASTDLLM_GDN_KERNEL=fla) + chunked CE, which lifts the
#      feasible block from the two-stream ~8192 to ~12288. MEASURED dry-probe peaks
#      (RTX 5090, 31.3 GiB usable, worst-case longest-8 truncated to block):
#          32768 -> OOM ; 24576 -> OOM ; 16384 -> 29.4 GiB (thin ~1.9 GiB margin,
#          rejected for an unattended run on a live-desktop GPU) ; 12288 -> 24.8 GiB
#          (~6.5 GiB margin, STABLE) ; 8192 -> 21.4 GiB.
#      So CANDIDATES default to the robustly-stable set; the ladder re-confirms the
#      largest fits (2-step caged dry fwd/bwd) before committing.
#   3. FULL DATASET -- (re)build the 334-row LMFlow pretok dataset at the chosen block
#      (left-truncation keeps the final edit-and-verify turns).
#   4. DETACHED LAUNCH -- setsid + systemd-run --scope MemoryMax=22G, resumable
#      checkpoints every 100 steps (erosion sweep points), pidfile, metrics jsonl,
#      cleanup trap, auto-resume from the latest checkpoint.
#
# Faithful chunked-resume: fixed cosine HORIZON so a killed+resumed run reproduces a
# single continuous run's schedule; deterministic seeded data-index schedule; the
# trainer restores adapter+optimizer+scheduler+rng(torch/cuda/py/np)+step and writes
# a resume manifest. STOP_AT_STEP=HORIZON => runs to the horizon; SAVE_STEPS=100 emits
# {100,200,300,400} checkpoints.
#
# Env: HORIZON (400), SEED (71101), CANDIDATES ("12288 8192 6144"),
#      GPU_FREE_MIB (3000), FORCE_BLOCK (skip ladder), LOGITS_CHUNK (2048).
set -euo pipefail
cd /home/mark/qwen_diffusion

HORIZON="${HORIZON:-400}"
SEED="${SEED:-71101}"
CANDIDATES="${CANDIDATES:-12288 8192 6144}"
GPU_FREE_MIB="${GPU_FREE_MIB:-3000}"
LOGITS_CHUNK="${LOGITS_CHUNK:-2048}"
ARM_DIR="$PWD/runs/swe_sft_arm1"
OUT="${OUT:-$ARM_DIR/Aswe_S_step${HORIZON}_seed${SEED}}"
DATA_FULL="$PWD/data/swe_sft_pool/lmflow_pretok"
PIDFILE="$ARM_DIR/train.pid"
METRICS="$ARM_DIR/metrics.jsonl"
TRAINLOG="$OUT/train.log"
PY="$PWD/.venv-fastdllm/bin/python"
TRAINER="scripts/swe_sft_arm1_qlora_train.py"
BUILDER="runs/swe_datagen_s1/build_swe_sft_lmflow_pretok.py"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FASTDLLM_GDN_KERNEL=fla
mkdir -p "$ARM_DIR" "$OUT"

ts() { date -Iseconds; }
say() { echo "[arm1-driver $(ts)] $*"; }

# ---------- 1. PREFLIGHT (never fight the datagen for the GPU) ----------
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [[ "$USED" -ge "$GPU_FREE_MIB" ]]; then
  say "ABORT: GPU busy (${USED} MiB >= ${GPU_FREE_MIB}). datagen still holds the GPU; handover not complete. Not launching."
  exit 9
fi
FREEG=$(free -g | awk '/Mem/{print $7}')
if [[ "${FREEG:-0}" -lt 8 ]]; then
  say "ABORT: host RAM headroom ${FREEG}G < 8G."
  exit 8
fi
say "preflight OK: gpu_used=${USED}MiB host_free=${FREEG}G gdn_kernel=fla"

# ---------- 2. BLOCK LADDER (2-step caged dry fwd/bwd, worst-case longest-8) ----------
choose_block() {
  if [[ -n "${FORCE_BLOCK:-}" ]]; then echo "$FORCE_BLOCK"; return; fi
  for B in $CANDIDATES; do
    say "probing block_size=$B (2-step caged dry fwd/bwd, longest-8)..." >&2
    local SM="$PWD/data/swe_sft_pool/lmflow_pretok_probe${B}"
    rm -rf "$SM"; mkdir -p "$SM"
    "$PY" "$BUILDER" --out-dir "data/swe_sft_pool/lmflow_pretok_probe${B}" --max-len "$B" --limit-longest 8 >/dev/null 2>&1
    local PLOG="$ARM_DIR/probe_${B}.log"
    set +e
    timeout 600 systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=4G -- \
      env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FASTDLLM_GDN_KERNEL=fla \
      "$PY" "$TRAINER" --dataset "data/swe_sft_pool/lmflow_pretok_probe${B}/swe_sft_train.json" \
        --output-dir "$ARM_DIR/_probe${B}" --block-size "$B" --horizon "$HORIZON" \
        --stop-at-step 2 --save-steps 100 --logging-steps 1 --logits-chunk "$LOGITS_CHUNK" \
        --max-train-samples 8 --longest-first > "$PLOG" 2>&1
    local RC=$?
    set -e
    rm -rf "$ARM_DIR/_probe${B}" "$SM"
    if [[ $RC -eq 0 ]] && grep -q "'loss'" "$PLOG" && ! grep -qi "out of memory" "$PLOG"; then
      local PK=$(grep -oE "'peak_gib': [0-9.]+" "$PLOG" | tail -1)
      say "block_size=$B FEASIBLE (2 steps, finite loss, no OOM; ${PK})." >&2
      echo "$B"; return
    fi
    say "block_size=$B rejected (rc=$RC; oom=$(grep -qi 'out of memory' "$PLOG" && echo yes || echo no))." >&2
  done
  echo ""
}
BLOCK="$(choose_block)"
if [[ -z "$BLOCK" ]]; then say "ABORT: no candidate block fit VRAM. Widen CANDIDATES downward."; exit 7; fi
say "CHOSEN block_size=$BLOCK"

# ---------- 3. FULL DATASET at the chosen block ----------
rm -rf "$DATA_FULL"; mkdir -p "$DATA_FULL"
"$PY" "$BUILDER" --out-dir "data/swe_sft_pool/lmflow_pretok" --max-len "$BLOCK" 2>&1 | tail -15
say "full 334-row dataset built at block_size=$BLOCK"

# ---------- 4. DETACHED LAUNCH ----------
say "launch: HORIZON=$HORIZON block=$BLOCK chunk=$LOGITS_CHUNK out=$OUT (auto-resume from latest checkpoint)"
: > "$METRICS"
setsid bash -c "
  echo \$\$ > '$PIDFILE'
  trap 'rm -f \"$PIDFILE\"' EXIT
  exec systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=4G -- \
    env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FASTDLLM_GDN_KERNEL=fla \
    '$PY' '$TRAINER' \
      --dataset '$DATA_FULL/swe_sft_train.json' --output-dir '$OUT' \
      --block-size '$BLOCK' --horizon '$HORIZON' --stop-at-step '$HORIZON' \
      --save-steps 100 --save-total-limit 6 --logging-steps 5 --logits-chunk '$LOGITS_CHUNK' \
      --seed '$SEED' --resume auto --metrics '$METRICS'
" > "$TRAINLOG" 2>&1 &

sleep 3
say "DETACHED. pidfile=$PIDFILE pid=$(cat "$PIDFILE" 2>/dev/null) log=$TRAINLOG metrics=$METRICS"
say "monitor: tail -f $TRAINLOG ; jq . $METRICS ; ls $OUT/checkpoint-*"
