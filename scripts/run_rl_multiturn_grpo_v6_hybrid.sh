#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv-fastdllm/bin/python}"
INPUT_JSONL="${INPUT_JSONL:-$ROOT/data/rl_multiturn_v5_mixed_pool/episodes.jsonl}"
ADAPTER="${ADAPTER:-$ROOT/runs/rl_multiturn_grpo_v2/from_selected_base_g4_step300/adapter_model}"
OUT_DIR="${OUT_DIR:-$ROOT/runs/rl_multiturn_grpo_v6/from_v2_hybrid_mixed35_kl005_g4_step300}"

MAX_STEPS="${MAX_STEPS:-300}"
GROUP_SIZE="${GROUP_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
KL_TO_BASE_COEFF="${KL_TO_BASE_COEFF:-0.05}"
KL_EARLY_STOP_WINDOW="${KL_EARLY_STOP_WINDOW:-50}"
KL_EARLY_STOP_MEAN_THRESHOLD="${KL_EARLY_STOP_MEAN_THRESHOLD:-0.05}"
RETENTION_PROBE_EVERY_STEPS="${RETENTION_PROBE_EVERY_STEPS:-50}"
RETENTION_PROBE_LIMIT="${RETENTION_PROBE_LIMIT:-10}"
RETENTION_COLLAPSE_FLEX_ACCURACY="${RETENTION_COLLAPSE_FLEX_ACCURACY:-0.65}"
EPISODE_LIMIT="${EPISODE_LIMIT:-398}"

exec "$PYTHON" scripts/rl_multiturn_grpo_pilot.py \
  --input-jsonl "$INPUT_JSONL" \
  --out-dir "$OUT_DIR" \
  --adapter "$ADAPTER" \
  --episode-limit "$EPISODE_LIMIT" \
  --min-turns 2 \
  --max-turns 6 \
  --max-steps "$MAX_STEPS" \
  --group-size "$GROUP_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --mixed-episode-groups \
  --decode-policy hybrid_clean \
  --hybrid-grammar-topk 256 \
  --temperature 0.0 \
  --top-p 0.95 \
  --kl-to-base-coeff "$KL_TO_BASE_COEFF" \
  --kl-early-stop-window "$KL_EARLY_STOP_WINDOW" \
  --kl-early-stop-mean-threshold "$KL_EARLY_STOP_MEAN_THRESHOLD" \
  --retention-probe-every-steps "$RETENTION_PROBE_EVERY_STEPS" \
  --retention-probe-limit "$RETENTION_PROBE_LIMIT" \
  --retention-collapse-flex-accuracy "$RETENTION_COLLAPSE_FLEX_ACCURACY"
