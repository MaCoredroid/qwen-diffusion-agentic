#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

QWEN_CODE_BIN="${QWEN_CODE_BIN:-./node_modules/.bin/qwen}"
PROXY_HOST="${PROXY_HOST:-127.0.0.1}"
PROXY_PORT="${PROXY_PORT:-30001}"
UPSTREAM_BASE_URL="${UPSTREAM_BASE_URL:-http://127.0.0.1:30000/v1}"
MODEL="${MODEL:-qwen3.6-27b-teacher}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-128}"
PROMPT="${1:-Reply exactly QWEN_CODE_PROXY_OK}"

if [[ ! -x "${QWEN_CODE_BIN}" ]]; then
  echo "Missing Qwen Code binary: ${QWEN_CODE_BIN}" >&2
  echo "Run: npm install" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
cleanup() {
  if [[ -n "${proxy_pid:-}" ]]; then
    kill "${proxy_pid}" 2>/dev/null || true
    wait "${proxy_pid}" 2>/dev/null || true
  fi
  rm -rf "${tmpdir}"
}
trap cleanup EXIT

python3 scripts/qwen_code_sglang_proxy.py \
  --host "${PROXY_HOST}" \
  --port "${PROXY_PORT}" \
  --upstream "${UPSTREAM_BASE_URL}" \
  --max-tokens "${MAX_OUTPUT_TOKENS}" \
  >"${tmpdir}/proxy.log" 2>&1 &
proxy_pid="$!"

for _ in $(seq 1 50); do
  if curl -fsS "http://${PROXY_HOST}:${PROXY_PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done

QWEN_CODE_MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS}" \
"${QWEN_CODE_BIN}" \
  --bare \
  --auth-type openai \
  --openai-api-key dummy \
  --openai-base-url "http://${PROXY_HOST}:${PROXY_PORT}/v1" \
  --model "${MODEL}" \
  --max-tool-calls 0 \
  --max-wall-time 60s \
  --output-format json \
  --system-prompt "Reply only with the requested answer." \
  --exclude-tools read_file \
  --exclude-tools edit \
  --exclude-tools notebook_edit \
  --exclude-tools run_shell_command \
  -p "${PROMPT}"
