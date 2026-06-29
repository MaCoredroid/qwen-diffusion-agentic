#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/mark/qwen_diffusion
PY="$ROOT/.venv-lmeval/bin/python"
EVAL="$ROOT/fast-dllm/v2/eval.py"
TASK_PATH="$ROOT/tasks/phaseA_retention"
BASE_MODEL="$ROOT/models/qwen3.5-9b-fastdllm-init"
CKPT_PARENT="$ROOT/runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300"
CKPT_ADAPTER="$CKPT_PARENT/checkpoint-275/adapter_model"
RUN_ROOT="$ROOT/runs/phaseA_retention_snapshot_n5_t384_nocache_matrix"
LOG_DIR="$RUN_ROOT/logs"
STATUS="$RUN_ROOT/status.tsv"

mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_ALLOW_CODE_EVAL=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1

if [[ ! -f "$STATUS" ]]; then
  printf 'condition\ttask\tstarted_at\tfinished_at\texit_code\toutput_path\tlog_path\n' > "$STATUS"
fi

run_cell() {
  local condition="$1"
  local task_alias="$2"
  local task_name="$3"
  local tokenizer_path="$4"
  local adapter_arg="$5"
  local out_dir="$RUN_ROOT/$condition/$task_alias"
  local log_file="$LOG_DIR/${condition}_${task_alias}.log"
  local started
  local finished
  local rc

  mkdir -p "$out_dir"
  started="$(date -Is)"
  {
    printf 'condition=%s\n' "$condition"
    printf 'task_alias=%s\n' "$task_alias"
    printf 'task_name=%s\n' "$task_name"
    printf 'started_at=%s\n' "$started"
    printf 'use_block_cache=False\n'
    printf 'max_new_tokens=384\n'
    printf 'threshold=0.9\n'
    printf 'bd_size=32\n'
    printf 'small_block_size=8\n'
    printf 'tokenizer_path=%s\n' "$tokenizer_path"
    printf 'adapter_arg=%s\n\n' "$adapter_arg"
  } >> "$log_file"

  set +e
  /usr/bin/time -p "$PY" "$EVAL" \
    --model fast_dllm_v2 \
    --model_args "model_path=$BASE_MODEL,tokenizer_path=$tokenizer_path,max_new_tokens=384,show_speed=True,threshold=0.9,bd_size=32,small_block_size=8,use_block_cache=False,temperature=0.0$adapter_arg" \
    --tasks "$task_name" \
    --include_path "$TASK_PATH" \
    --limit 5 \
    --batch_size 1 \
    --device cuda:0 \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --confirm_run_unsafe_code \
    --log_samples \
    --output_path "$out_dir" >> "$log_file" 2>&1
  rc=$?
  set -e

  finished="$(date -Is)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$condition" "$task_alias" "$started" "$finished" "$rc" "$out_dir" "$log_file" >> "$STATUS"

  if [[ "$rc" -ne 0 ]]; then
    printf 'Cell failed: condition=%s task=%s exit_code=%s log=%s\n' \
      "$condition" "$task_alias" "$rc" "$log_file" >&2
    exit "$rc"
  fi
}

run_cell diff_init_no_adapter mbpp phaseA_mbpp_first20 "$BASE_MODEL" ""
run_cell diff_init_no_adapter gsm8k phaseA_gsm8k_first20 "$BASE_MODEL" ""
run_cell diff_init_no_adapter ifeval phaseA_ifeval_first20 "$BASE_MODEL" ""

run_cell checkpoint_275_adapter mbpp phaseA_mbpp_first20 "$CKPT_PARENT" ",adapter_path=$CKPT_ADAPTER,merge_adapter=True"
run_cell checkpoint_275_adapter gsm8k phaseA_gsm8k_first20 "$CKPT_PARENT" ",adapter_path=$CKPT_ADAPTER,merge_adapter=True"
run_cell checkpoint_275_adapter ifeval phaseA_ifeval_first20 "$CKPT_PARENT" ",adapter_path=$CKPT_ADAPTER,merge_adapter=True"

printf 'completed_at=%s\n' "$(date -Is)" > "$RUN_ROOT/COMPLETE"
