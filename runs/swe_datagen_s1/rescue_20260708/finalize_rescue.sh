#!/usr/bin/env bash
# Run ONLY in the quiet window right after a fresh orch [status] line.
# Flip poisoned rows -> true rescue verdicts (race-safe) + extract keepers from the
# newly-resolved episodes + print the corrected ledger. Writes to a SCRATCH status
# file (never the orch's STATUS).
set -euo pipefail
cd /home/mark/qwen_diffusion/runs/swe_datagen_s1
PY=/home/mark/qwen_diffusion/.venv/bin/python
TS=$(date -u +%Y%m%dT%H%M%SZ)
ENV_JSON='{"temperature":0.6,"top_p":0.95,"top_k":20,"seed_base":1234}'

echo "==== FINALIZE RESCUE $TS ===="
# 0) backups
cp -a attempts.jsonl "attempts.jsonl.bak_pre_rescue_flip_${TS}"
cp -a keepers/keepers.jsonl "keepers/keepers.jsonl.bak_pre_rescue_${TS}"
N_ATT_PRE=$(wc -l < attempts.jsonl)
N_KEEP_PRE=$(wc -l < keepers/keepers.jsonl)
echo "[pre] attempts=$N_ATT_PRE keepers=$N_KEEP_PRE"

# 1) FLIP (race-safe; re-reads immediately before atomic rename)
$PY rescue_20260708/apply_rescue.py --apply

# 2) EXTRACT KEEPERS from the rescued resolved episodes (idempotent instance_id dedup)
for b in batch_0001_20260707T024659Z batch_0011_20260707T101832Z batch_0012_20260707T110240Z; do
  REP=$(ls rescue_20260708/$b/score/parts/fork/*"${b}"_rescue*.json 2>/dev/null | grep -v timing | head -1)
  echo "[keepers] $b report=$REP"
  $PY extract_keepers.py "batches/$b" "$b" "batches/$b/gen" "$REP" "keepers" "$ENV_JSON"
done

# 3) CORRECTED LEDGER (scratch status file; read-only over attempts/keepers/frontier)
$PY ledger.py state attempts.jsonl keepers/keepers.jsonl frontier.json \
    "rescue_20260708/corrected_status_${TS}.txt" \
    --target 1000 --floor 400 --kill-yield 0.10 --kill-window 200

N_ATT_POST=$(wc -l < attempts.jsonl)
N_KEEP_POST=$(wc -l < keepers/keepers.jsonl)
echo "[post] attempts=$N_ATT_POST (was $N_ATT_PRE; monotonic=$([ "$N_ATT_POST" -ge "$N_ATT_PRE" ] && echo yes || echo NO)) keepers=$N_KEEP_POST (was $N_KEEP_PRE)"
echo "[distinct keeper ids] $($PY -c "import json;print(len({json.loads(l)['instance_id'] for l in open('keepers/keepers.jsonl')}))")"
echo "==== FINALIZE DONE $(date -u +%FT%TZ) ===="
