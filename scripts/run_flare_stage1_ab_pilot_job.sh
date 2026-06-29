#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mark/qwen_diffusion"
MODE="${1:?usage: run_flare_stage1_ab_pilot_job.sh diffusion|two_stream OUTPUT_DIR}"
OUTPUT_DIR="${2:?usage: run_flare_stage1_ab_pilot_job.sh diffusion|two_stream OUTPUT_DIR}"

case "${MODE}" in
    diffusion|two_stream) ;;
    *)
        printf 'MODE must be diffusion or two_stream, got %s\n' "${MODE}" >&2
        exit 2
        ;;
esac

mkdir -p "${OUTPUT_DIR}"
TRAIN_LOG="${OUTPUT_DIR}/train.log"
MEM_LOG="${OUTPUT_DIR}/gpu_memory_mib.log"
RUNTIME_JSON="${OUTPUT_DIR}/pilot_runtime.json"

rm -f "${MEM_LOG}"
(
    while true; do
        printf '%s ' "$(date +%s)"
        nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ','
        sleep 1
    done
) > "${MEM_LOG}" &
MONITOR_PID=$!

cleanup() {
    kill "${MONITOR_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

START_TS="$(date +%s)"
set +e
env \
    BUILD_CURRICULUM=0 \
    DATASET_DIR="${DATASET_DIR:-${ROOT}/data/flare_stage1_ab_pilot_train}" \
    MODEL_PATH="${ROOT}/models/qwen3.5-9b-fastdllm-init" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    MAX_STEPS="${MAX_STEPS:-200}" \
    MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-256}" \
    BLOCK_SIZE="${BLOCK_SIZE:-1536}" \
    LEARNING_RATE="${LEARNING_RATE:-1e-5}" \
    GRAD_ACCUM="${GRAD_ACCUM:-1}" \
    SAVE_STEPS="${SAVE_STEPS:-${MAX_STEPS:-200}}" \
    SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}" \
    LORA_R="${LORA_R:-8}" \
    LORA_ALPHA="${LORA_ALPHA:-16}" \
    LORA_DROPOUT="${LORA_DROPOUT:-0.05}" \
    LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}" \
    OVERWRITE_CACHE=1 \
    TRAIN_BD_SIZE="${TRAIN_BD_SIZE:-32}" \
    SEED="${SEED:-20260701}" \
    DATA_SEED="${DATA_SEED:-${SEED:-20260701}}" \
    GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}" \
    LOGGING_STEPS="${LOGGING_STEPS:-1}" \
    FASTDLLM_FLARE_TWO_STREAM="$([[ "${MODE}" == "two_stream" ]] && printf 1 || printf 0)" \
    FLARE_TWO_STREAM="$([[ "${MODE}" == "two_stream" ]] && printf 1 || printf 0)" \
    FASTDLLM_FLARE_DEBUG="$([[ "${MODE}" == "two_stream" ]] && printf '%s' "${FASTDLLM_FLARE_DEBUG:-1000}" || printf 0)" \
    "${ROOT}/scripts/run_fastdllm_qwen35_9b_agentic_qlora_pilot.sh" \
    > "${TRAIN_LOG}" 2>&1
STATUS=$?
set -e
END_TS="$(date +%s)"
cleanup
trap - EXIT

"${ROOT}/.venv-fastdllm/bin/python" - "${MODE}" "${OUTPUT_DIR}" "${TRAIN_LOG}" "${MEM_LOG}" "${RUNTIME_JSON}" "${START_TS}" "${END_TS}" "${STATUS}" <<'PY'
import json
import sys
from pathlib import Path

mode, out_dir, train_log, mem_log, runtime_json, start_ts, end_ts, status = sys.argv[1:]
peak_mib = None
samples = 0
for line in Path(mem_log).read_text(encoding="utf-8").splitlines():
    parts = line.split()
    if len(parts) < 2:
        continue
    try:
        value = int(parts[1])
    except ValueError:
        continue
    peak_mib = value if peak_mib is None else max(peak_mib, value)
    samples += 1

payload = {
    "mode": mode,
    "output_dir": out_dir,
    "train_log": train_log,
    "memory_log": mem_log,
    "start_ts": int(start_ts),
    "end_ts": int(end_ts),
    "wall_seconds": int(end_ts) - int(start_ts),
    "exit_status": int(status),
    "gpu_peak_memory_mib": peak_mib,
    "gpu_memory_samples": samples,
}
Path(runtime_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

exit "${STATUS}"
