#!/usr/bin/env bash
# Stage-A smoke: serve the FLARE diffusion hybrid_clean /v1 endpoint (:9952) via
# the certified flywheel launcher, INSIDE the RAM cage (applied by the
# systemd-run --scope wrapper the caller uses). ONE heavy process at a time.
#
# Same launcher + engine + weights as the A6/A7 byte-parity cert (pin
# @b5fcb3d-class, gate OFF = Stage-3 shipped config, mask suppression + bidir
# probe ON). The ONE deviation from the cert: MAX_MODEL_LEN is raised 4096->8192
# so a real qwen-code agentic turn (system prompt + tool schemas + tool results)
# fits. This does not touch the decode path; the FLARE audit counters
# (decode_mode / projected_value_tokens_exact) are unaffected.
set -euo pipefail
cd /home/mark/qwen_diffusion
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
export PORT=${PORT:-9952}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
exec /home/mark/shared/lumoFlyWheel_codex_fork/scripts/qwen35_9b_flare_hybrid_serve.sh
