#!/usr/bin/env bash
# X.1 KILL-T1 — FULL matched-20 exact_args battery (design §2.3), the pre-gate guard for the
# C46 X.1 gate. Runs the CANONICAL anchor battery `eval_flare_northstar_hybrid_clean.py`
# (FLARE hybrid_clean K=1, offline == served by A6 cert) on the X.1 twin (base+adapter,
# no-merge, exactly the step4_cert precedent), then pairs per-turn exact_args vs the BANKED
# iteration-2 lineage anchor (twin@K1's own matched-20, 49/63) with gold_sha256 identity.
# PASS bar (§2.3 / #29): McNemar net-loss NOT significant (p>=0.05) AND raw >= anchor-3.
# GPU idle on exit (the python process exits, releasing the device).
set -uo pipefail
cd /home/mark/qwen_diffusion
PY=.venv/bin/python
OUTD=runs/kraise_reconvert_iter2_x1/battery/killt1_m20
ANCHOR=runs/swe_sft_arm1_iter2/anchor_gate/mswe_S_matched20/ar-vllm-guided/turns.jsonl
BASE=models/qwen3.5-9b-fastdllm-mswe-S-iter2-merged
ADAPTER=runs/kraise_reconvert_iter2_x1/mswe2_S_x1_readground_step800_seed81101
mkdir -p "$OUTD"
echo "==== X.1 KILL-T1 matched-20 START $(date -u +%FT%TZ) ===="

# 1. candidate battery: X.1 twin hybrid_clean matched-20 (63 turns / 20 episodes)
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $PY scripts/eval_flare_northstar_hybrid_clean.py \
  --base-model "$BASE" \
  --adapter "$ADAPTER" \
  --out-dir "$OUTD" \
  --episode-limit 20 --min-turns 3 --max-turns 6 \
  --block-size 32 --max-new-tokens 384 --temperature 0.0 --seed 20260701
BRC=$?
echo "[killt1-m20] battery rc=$BRC $(date -u +%FT%TZ)"
[[ "$BRC" -ne 0 ]] && { echo "[killt1-m20] BATTERY FAILED"; exit 3; }

POST="$OUTD/diffusion_hybrid_forced_grammar_seq_values/turns.jsonl"
[[ -s "$POST" ]] || { echo "[killt1-m20] MISSING candidate turns.jsonl at $POST"; exit 3; }

# 2. paired McNemar vs the banked iter2 lineage anchor (49/63)
$PY scripts/swe_sft_arm1_anchor_mcnemar.py \
  --pre "$ANCHOR" --post "$POST" --anchor 49 \
  --out "$OUTD/killt1_m20_mcnemar.json"
MRC=$?
echo "[killt1-m20] mcnemar rc=$MRC $(date -u +%FT%TZ)"
echo "==== X.1 KILL-T1 matched-20 DONE rc=$MRC $(date -u +%FT%TZ) ===="
exit $MRC
