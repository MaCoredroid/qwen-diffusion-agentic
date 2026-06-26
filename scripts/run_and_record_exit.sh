#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 SCRIPT_PATH LOG_PATH EXIT_CODE_PATH" >&2
  exit 2
fi

script_path="$1"
log_path="$2"
exit_code_path="$3"

set +e
bash "${script_path}" > "${log_path}" 2>&1
rc=$?
set -e

printf '%s\n' "${rc}" > "${exit_code_path}"
exit "${rc}"
