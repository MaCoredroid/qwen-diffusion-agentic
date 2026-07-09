#!/usr/bin/env bash
# Launch the Opus OpenAI-shim adapter as a persistent background server for the
# pilot. Reads the user's long-lived Claude Code OAuth token READ-ONLY from the
# parent claude process env (/proc/<pid>/environ); the token is exported ONLY into
# this adapter process's env and is NEVER printed or written to disk.
#   usage: adapter_up.sh <claude_pid> <port> <usage_log>
set -uo pipefail
cd /home/mark/qwen_diffusion
CLAUDE_PID="${1:?claude pid}"; PORT="${2:-30050}"; ULOG="${3:?usage log}"
TOK=$(tr '\0' '\n' < /proc/${CLAUDE_PID}/environ 2>/dev/null | grep '^CLAUDE_CODE_OAUTH_TOKEN=' | head -1 | cut -d= -f2-)
[ -z "$TOK" ] && { echo "NO TOKEN from claude pid $CLAUDE_PID"; exit 3; }
export ANTHROPIC_AUTH_TOKEN="$TOK"
export OPUS_ADAPTER_USAGE_LOG="$ULOG"
export OPUS_ADAPTER_CACHE="${OPUS_ADAPTER_CACHE:-1}"
export OPUS_ADAPTER_MAX_RETRIES="${OPUS_ADAPTER_MAX_RETRIES:-8}"
exec .venv/bin/python scripts/opus_openai_adapter.py --backend anthropic \
  --host 127.0.0.1 --port "$PORT" --anthropic-model claude-opus-4-8 \
  --served-model claude-opus-adapter --max-tokens-floor 2048 --timeout 300
