#!/usr/bin/env bash
# Gate battery runner (gates2). ONE heavy vLLM process inside the RAM cage.
# Usage: GATE=on|off  RESET_APC=0|1  BENCH_REF=... BENCH_OUT=... EP_START/EP_END/BENCH_ONLY ...  cage_run.sh
# GATE=on  -> VLLM_QWEN3_5_FLARE_CANONICAL_PUBLISH=1, eager (no cudagraph; capture not cg-safe)
# GATE=off -> canonical publish unset, eager (controlled baseline)  [cudagraph if CG=1]
set -euo pipefail
cd /home/mark/qwen_diffusion
source /home/mark/qwen_diffusion/runs/p2_engine_nevertrain/env.sh   # CUDA_HOME, BIDIR=1, CUDAGRAPH=1, backend
# force eager for BOTH arms (fair A/B; capture path is not cudagraph-safe)
unset VLLM_FLARE_CUDAGRAPH
export VLLM_QWEN3_5_FLARE=1
if [ "${GATE:-off}" = "on" ]; then
  export VLLM_QWEN3_5_FLARE_CANONICAL_PUBLISH=1
else
  unset VLLM_QWEN3_5_FLARE_CANONICAL_PUBLISH || true
fi
echo "[cage] GATE=${GATE:-off} CANONICAL_PUBLISH=${VLLM_QWEN3_5_FLARE_CANONICAL_PUBLISH:-unset} CUDAGRAPH=${VLLM_FLARE_CUDAGRAPH:-unset} BIDIR=${VLLM_FLARE_BIDIR_PROBE:-unset} RESET_APC=${BENCH_RESET_APC:-0} REF=${BENCH_REF:-default} OUT=${BENCH_OUT:-default}"
exec /home/mark/qwen_diffusion/.venv-vllm-p2-main/bin/python \
  /home/mark/qwen_diffusion/runs/p2_engine_battery_v3b/run_battery_v3b.py
