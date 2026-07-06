#!/usr/bin/env bash
# W2 N=50 THE RUN — race-free orchestrator (SHARDED path).
#
# WHY THIS FILE EXISTS: the earlier w2_orch.sh chains w2_arm.sh, whose shared
# mkdir-based worker-claim had a non-atomic race that double-ran instances
# (RACE_DIAGNOSIS_HANDOFF.md; prior agent died mid-debug). The fix is structural:
# gen_shard_plan.py pre-partitions the 50 frozen-pool instances into C DISJOINT
# fixed lists (round-robin over pool order), and run_arm.sh drives each shard with
# `--only <its ids>`. There is NO shared queue and NO claim => the race is gone by
# construction. BOTH arms use the SAME instance->shard assignment + per-shard base
# seed, so every instance is attempted under matched sampling on both arms
# (paired-McNemar validity). Use THIS, not w2_orch.sh, for the real run.
#
# ONE model server at a time; scoring serialized AFTER both arms (one heavy job
# class at a time). Detach with: setsid bash runs/w2_n50/run_all.sh &>run.log &
set -uo pipefail
cd /home/mark/qwen_diffusion
export SUDO_ASKPASS=/tmp/claude-1000/-home-mark/8f712353-03d0-4607-ac3a-cba8072f9d36/scratchpad/askpass.sh

ROOT=runs/w2_n50
PLAN="$ROOT/shard_plan.json"
C=4
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/run_all.log"
exec >>"$LOG" 2>&1
status(){ echo "$*" > "$ROOT/W2_STATUS.txt"; echo "[run_all] $*"; }
stamp(){ date -u +%FT%TZ; }

echo "==================== W2 RUN START $(stamp) ===================="

# 0. deterministic shard plan (idempotent; ARM-independent; committed integrity)
.venv/bin/python "$ROOT/gen_shard_plan.py" "$ROOT/subset_n50.json" "$C" 1234 "$PLAN" || exit 2

# 1. AR arm (stock-AR @ gmu 0.85, c=4)
status "AR arm running (gmu0.85 c=4) $(stamp)"
bash "$ROOT/run_arm.sh" ar "$C" "$ROOT" "$PLAN"; echo "[run_all] AR rc=$? $(stamp)"
sleep 5

# 2. FRESH diffusion boot-probe at the frozen config under MAX long-context load
echo "[run_all] ==== DIFFUSION FRESH BOOT-PROBE (c=4 gmu0.74, 20k prompt) $(stamp) ===="
status "diffusion boot-probe $(stamp)"
bash runs/stage_c_n5v3_gate/boot_probe_diffusion.sh 0.74 4 32768 20000
PROBE=runs/stage_c_n5v3_gate/boot_probe_result_g0.74_c4.json
PROBE_OK=$(.venv/bin/python -c 'import json,sys;d=json.load(open(sys.argv[1]));print("true" if (d.get("boot_ok") and d.get("no_allocation_failure")) else "false")' "$PROBE" 2>/dev/null || echo false)
echo "[run_all] boot-probe no_allocation_failure=$PROBE_OK $(stamp)"
sleep 5

# 3. diffusion arm (ONLY if the fresh probe confirmed the frozen config)
if [[ "$PROBE_OK" == "true" ]]; then
  status "diffusion arm running (gmu0.74 c=4) $(stamp)"
  bash "$ROOT/run_arm.sh" diffusion "$C" "$ROOT" "$PLAN"; echo "[run_all] diffusion rc=$? $(stamp)"
else
  echo "[run_all] !!! BOOT-PROBE FAILED — diffusion arm SKIPPED (frozen config not confirmed) !!!"
  echo "boot_probe_failed $(stamp)" > "$ROOT/ANOMALY_bootprobe.txt"
  status "ANOMALY diffusion boot-probe failed; arm SKIPPED $(stamp)"
fi
sleep 5

# 4. OFFICIAL docker scoring (both arms; NO server up)
status "scoring AR $(stamp)"
bash "$ROOT/score_all.sh" ar "$ROOT/ar/predictions.jsonl" "$ROOT/ar/scoring" "$C"; echo "[run_all] score ar rc=$? $(stamp)"
if [[ "$PROBE_OK" == "true" ]]; then
  status "scoring diffusion $(stamp)"
  bash "$ROOT/score_all.sh" diffusion "$ROOT/diffusion/predictions.jsonl" "$ROOT/diffusion/scoring" "$C"; echo "[run_all] score diffusion rc=$? $(stamp)"
fi

# 5. paired report (McNemar + throughput + covariates + per-repo)
.venv/bin/python "$ROOT/build_report.py" "$ROOT" > "$ROOT/build_report_stdout.txt" 2>&1
echo "[run_all] report rc=$? $(stamp)"
echo "==================== W2 RUN DONE $(stamp) ===================="
status "DONE $(stamp)"; date -u +%FT%TZ > "$ROOT/W2_DONE.txt"
