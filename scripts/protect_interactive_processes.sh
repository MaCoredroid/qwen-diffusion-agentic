#!/usr/bin/env bash
set -euo pipefail

if [ "${EUID}" -ne 0 ]; then
  echo "Run as root: sudo bash $0" >&2
  exit 1
fi

OOM_ADJ="${OOM_ADJ:--900}"
MEM_LOW_BYTES="${MEM_LOW_BYTES:-1073741824}"   # 1 GiB best-effort reclaim protection
MEM_MIN_BYTES="${MEM_MIN_BYTES:-268435456}"    # 256 MiB hard minimum protection

mapfile -t pids < <(
  {
    pgrep -f '^[t]mux( |$)|[t]mux: (server|client)' || true
    pgrep -f '/[u]sr/bin/codex|@[o]penai/codex|mcp/[s]erver.bundle.mjs' || true
  } | sort -n | uniq
)

if [ "${#pids[@]}" -eq 0 ]; then
  echo "No tmux/Codex processes found."
  exit 0
fi

declare -A cgroups=()

for pid in "${pids[@]}"; do
  [ -e "/proc/${pid}" ] || continue
  cmd="$(tr '\0' ' ' < "/proc/${pid}/cmdline" | sed 's/[[:space:]]*$//')"
  cg="$(cut -d: -f3 < "/proc/${pid}/cgroup")"
  echo "${OOM_ADJ}" > "/proc/${pid}/oom_score_adj" || true
  cgroups["${cg}"]=1
  printf 'protected pid=%s oom_score_adj=%s cgroup=%s cmd=%s\n' "${pid}" "$(cat "/proc/${pid}/oom_score_adj")" "${cg}" "${cmd:-?}"
done

for cg in "${!cgroups[@]}"; do
  cg_dir="/sys/fs/cgroup${cg}"
  [ -d "${cg_dir}" ] || continue
  if [ -w "${cg_dir}/memory.low" ]; then
    echo "${MEM_LOW_BYTES}" > "${cg_dir}/memory.low" || true
  fi
  if [ -w "${cg_dir}/memory.min" ]; then
    echo "${MEM_MIN_BYTES}" > "${cg_dir}/memory.min" || true
  fi
  printf 'cgroup=%s memory.low=%s memory.min=%s\n' \
    "${cg}" \
    "$(cat "${cg_dir}/memory.low" 2>/dev/null || echo '?')" \
    "$(cat "${cg_dir}/memory.min" 2>/dev/null || echo '?')"
done
