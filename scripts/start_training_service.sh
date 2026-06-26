#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 UNIT_NAME SCRIPT_PATH LOG_PATH EXIT_CODE_PATH" >&2
  exit 2
fi

ROOT="/home/mark/qwen_diffusion"
unit="$1"
script_path="$2"
log_path="$3"
exit_code_path="$4"

MEMORY_HIGH="${TRAINING_MEMORY_HIGH:-27G}"
MEMORY_MAX="${TRAINING_MEMORY_MAX:-28G}"
MEMORY_SWAP_MAX="${TRAINING_MEMORY_SWAP_MAX:-4G}"

mkdir -p "$(dirname "${log_path}")" "$(dirname "${exit_code_path}")"
rm -f "${exit_code_path}"

if systemctl --user is-active --quiet "${unit}.service"; then
  echo "${unit}.service is already active" >&2
  exit 1
fi

systemctl --user reset-failed "${unit}.service" >/dev/null 2>&1 || true

systemd-run --user --collect \
  --unit="${unit}" \
  --property=WorkingDirectory="${ROOT}" \
  --property=MemoryAccounting=yes \
  --property=MemoryHigh="${MEMORY_HIGH}" \
  --property=MemoryMax="${MEMORY_MAX}" \
  --property=MemorySwapMax="${MEMORY_SWAP_MAX}" \
  --property=OOMScoreAdjust=500 \
  --property=OOMPolicy=stop \
  --property=KillMode=control-group \
  "${ROOT}/scripts/run_and_record_exit.sh" "${script_path}" "${log_path}" "${exit_code_path}"

echo "started ${unit}.service"
echo "log=${log_path}"
echo "exit_code_file=${exit_code_path}"
