#!/usr/bin/env bash
# SWE-SFT arm-1 (M_swe_S) — SELF-VERIFYING DETACHED LAUNCHER.
#
# Run this ONCE the 27B datagen has released the GPU (handover complete). It:
#   1. PREFLIGHT — refuses to run while the GPU is busy (does NOT fight the live
#      27B datagen teacher) and requires host-RAM headroom.
#   2. FEASIBILITY LADDER — the design's block_size=32768 is INFEASIBLE for the
#      two-stream FLARE trainer on the 32 GB 5090 (measured OOM at 16384: the
#      trainer concatenates clean+noisy to length 2L so logits are [2L, vocab]
#      ~16 GB and it builds an O((2L)^2) mask). So we MEASURE the largest block
#      in CANDIDATES that survives a 2-step bounded smoke under the memory cage,
#      and use it. Left-truncation keeps the final edit turns (highest-value SWE
#      targets). Retention per block is recorded by the dataset builder.
#   3. FULL DATASET — (re)build the 323-row LMFlow pretok dataset at the chosen
#      block.
#   4. DETACHED LAUNCH — setsid + systemd-run --scope MemoryMax=22G, resumable
#      checkpoints every 100 steps (the erosion sweep points), pidfile, metrics
#      jsonl tailer, cleanup trap, auto-resume from the latest checkpoint.
#
# Faithful-chunking: MAX_STEPS == HORIZON (fixed cosine horizon) so a killed+
# resumed run reproduces a single continuous run bit-for-bit (proven in
# convert-after-RL step-2). SAVE_STEPS=100 emits {100,200,300,400} checkpoints.
#
# Env: HORIZON (default 400), SEED (71101), CANDIDATES ("12288 8192 6144"),
#      GPU_FREE_MIB (3000), FORCE_BLOCK (skip ladder, use this block).
set -euo pipefail
cd /home/mark/qwen_diffusion

HORIZON="${HORIZON:-400}"
SEED="${SEED:-71101}"
CANDIDATES="${CANDIDATES:-12288 8192 6144}"
GPU_FREE_MIB="${GPU_FREE_MIB:-3000}"
ARM_DIR="$PWD/runs/swe_sft_arm1"
OUT="${OUT:-$ARM_DIR/Aswe_S_step${HORIZON}_seed${SEED}}"
DATA_FULL="$PWD/data/swe_sft_pool/lmflow_pretok"
PIDFILE="$ARM_DIR/train.pid"
METRICS="$ARM_DIR/metrics.jsonl"
TRAINLOG="$OUT/train.log"
mkdir -p "$ARM_DIR" "$OUT"

ts() { date -Iseconds; }
say() { echo "[arm1-driver $(ts)] $*"; }

# ---------- 1. PREFLIGHT (never fight the datagen for the GPU) ----------
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [[ "$USED" -ge "$GPU_FREE_MIB" ]]; then
  say "ABORT: GPU busy (${USED} MiB >= ${GPU_FREE_MIB}). 27B datagen still holds the GPU; handover not complete. Not launching."
  exit 9
fi
FREEG=$(free -g | awk '/Mem/{print $7}')
if [[ "${FREEG:-0}" -lt 8 ]]; then
  say "ABORT: host RAM headroom ${FREEG}G < 8G."
  exit 8
fi
say "preflight OK: gpu_used=${USED}MiB host_free=${FREEG}G"

# ---------- 2. FEASIBILITY LADDER ----------
choose_block() {
  if [[ -n "${FORCE_BLOCK:-}" ]]; then echo "$FORCE_BLOCK"; return; fi
  for B in $CANDIDATES; do
    say "probing block_size=$B (2-step bounded smoke)..." >&2
    local SM="$PWD/data/swe_sft_pool/lmflow_pretok_probe${B}"
    rm -rf "$SM"; mkdir -p "$SM"
    .venv-fastdllm/bin/python runs/swe_datagen_s1/build_swe_sft_lmflow_pretok.py \
      --out-dir "data/swe_sft_pool/lmflow_pretok_probe${B}" --max-len "$B" --limit-longest 6 >/dev/null 2>&1
    local PLOG="$ARM_DIR/probe_${B}.log"
    set +e
    timeout 600 systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=4G -- \
      env STOP_AT_STEP=2 HORIZON="$HORIZON" BLOCK_SIZE="$B" SEG_SEED="$SEED" MAX_TRAIN_SAMPLES=6 \
          SAVE_STEPS=100 LOGGING_STEPS=1 \
          DATASET_DIR="$SM" OUTPUT_DIR="$ARM_DIR/_probe${B}" \
          bash scripts/swe_sft_arm1_segment.sh > "$PLOG" 2>&1
    local RC=$?
    set -e
    rm -rf "$ARM_DIR/_probe${B}" "$SM"
    if [[ $RC -eq 0 ]] && grep -q "'loss'" "$PLOG" && ! grep -qi "out of memory" "$PLOG"; then
      say "block_size=$B FEASIBLE (2 steps, finite loss, no OOM)." >&2
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
.venv-fastdllm/bin/python runs/swe_datagen_s1/build_swe_sft_lmflow_pretok.py \
  --out-dir "data/swe_sft_pool/lmflow_pretok" --max-len "$BLOCK" 2>&1 | tail -20
say "full dataset built at block_size=$BLOCK"

# ---------- 4. DETACHED LAUNCH ----------
RESUME=""
if [[ -d "$OUT" ]]; then
  LATEST=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sed 's#.*checkpoint-##' | sort -n | tail -1 || true)
  [[ -n "${LATEST:-}" ]] && RESUME="$OUT/checkpoint-$LATEST" && rm -rf "$RESUME/adapter_model"
fi
say "launch: HORIZON=$HORIZON block=$BLOCK resume=${RESUME:-<fresh>} out=$OUT"

# metrics tailer: parse HF loss/lr/step lines from the train log into jsonl
cat > "$ARM_DIR/_metrics_tailer.py" <<'PY'
import json,re,sys,time,os
log,out=sys.argv[1],sys.argv[2]
pat=re.compile(r"\{'loss': ([0-9.eE+-]+), 'grad_norm': ([0-9.eE+naN-]+), 'learning_rate': ([0-9.eE+-]+),.*?'epoch': ([0-9.eE+-]+)\}")
seen=0
while True:
    if os.path.exists(log):
        with open(log) as f:
            lines=f.readlines()
        for ln in lines[seen:]:
            m=pat.search(ln)
            if m:
                rec={'t':time.time(),'loss':float(m.group(1)),'lr':float(m.group(3)),'epoch':float(m.group(4))}
                with open(out,'a') as g: g.write(json.dumps(rec)+"\n")
        seen=len(lines)
    time.sleep(15)
PY

: > "$METRICS"
export FASTDLLM_RESUMABLE_CKPT=1
setsid bash -c "
  echo \$\$ > '$PIDFILE'
  trap 'rm -f \"$PIDFILE\"' EXIT
  exec systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=4G -- \
    env STOP_AT_STEP='$HORIZON' HORIZON='$HORIZON' BLOCK_SIZE='$BLOCK' SEG_SEED='$SEED' \
        MAX_TRAIN_SAMPLES=323 SAVE_STEPS=100 SAVE_TOTAL_LIMIT=6 LOGGING_STEPS=5 \
        RESUME_FROM_CHECKPOINT='${RESUME}' \
        DATASET_DIR='$DATA_FULL' OUTPUT_DIR='$OUT' \
        bash scripts/swe_sft_arm1_segment.sh
" > "$TRAINLOG" 2>&1 &
setsid .venv-fastdllm/bin/python "$ARM_DIR/_metrics_tailer.py" "$TRAINLOG" "$METRICS" >/dev/null 2>&1 &

sleep 3
say "DETACHED. pidfile=$PIDFILE pid=$(cat "$PIDFILE" 2>/dev/null) log=$TRAINLOG metrics=$METRICS"
say "monitor: tail -f $TRAINLOG ; jq . $METRICS ; ls $OUT/checkpoint-*"
