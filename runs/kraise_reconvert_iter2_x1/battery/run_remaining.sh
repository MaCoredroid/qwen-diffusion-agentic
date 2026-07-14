#!/usr/bin/env bash
# Sequential X.1 battery: waits for the GPU to free, then runs arm(ii) baseline twinK1,
# arm(iii) X.1 AR-mode, and the KILL-T1 spot — one server at a time, each self-torn-down.
set -uo pipefail
cd /home/mark/qwen_diffusion
ROOT=/home/mark/qwen_diffusion
BATT=$ROOT/runs/k_gate_c46/x1_battery.sh
KILLT1=$ROOT/runs/k_gate_c46/x1_killt1_launch.sh
OUTD=$ROOT/runs/kraise_reconvert_iter2_x1/battery
X1=$ROOT/models/qwen3.5-9b-fastdllm-mswe2-S-x1-vllm-bf16
BASE=$ROOT/models/qwen3.5-9b-fastdllm-mswe-S-twinK1-vllm-bf16

gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
wait_idle(){
  local dl=$((SECONDS+900))
  while :; do local u; u=$(gpu_used); [[ "$u" -lt 3000 ]] && { echo "[orch] GPU idle ${u}MiB"; return 0; }
    [[ $SECONDS -gt $dl ]] && { echo "[orch] TIMEOUT waiting for GPU idle (${u}MiB)"; return 1; }
    sleep 10; done
}

echo "[orch] ===== ARM (ii) baseline twinK1 (hybrid_clean) ====="
wait_idle || exit 1
bash "$BATT" "$BASE" hybrid_clean baseK1 64 "$OUTD/baseK1.json" 2>&1 | sed 's/^/[ii] /'
echo "[orch] arm(ii) done rc=${PIPESTATUS[0]}"

echo "[orch] ===== ARM (iii) X.1 AR-mode (careful_live_grammar) ====="
wait_idle || exit 1
bash "$BATT" "$X1" careful_live_grammar x1ar 64 "$OUTD/x1ar.json" 2>&1 | sed 's/^/[iii] /'
echo "[orch] arm(iii) done rc=${PIPESTATUS[0]}"

echo "[orch] ===== KILL-T1 SPOT (X.1 hybrid_clean, 3 matched turns exact_args) ====="
wait_idle || exit 1
bash "$KILLT1" "$X1" hybrid_clean "$OUTD/killt1.json" 2>&1 | sed 's/^/[kt1] /'
echo "[orch] killt1 done rc=${PIPESTATUS[0]}"

echo "[orch] ALL REMAINING DONE"
