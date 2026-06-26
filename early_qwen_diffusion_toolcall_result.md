# Early Qwen Diffusion Tool-Call Result

Date: 2026-06-26

## Goal

Create an early Qwen-family diffusion model artifact that has actually been
trained and evaluated on the agentic/tool-call direction.

This is a plumbing milestone, not a successful agentic model.

## Model

- Base: `/home/mark/qwen_diffusion/models/qwen2.5-1.5b-fastdllm-init`
- Method: Fast-dLLM v2 masked-diffusion LoRA
- Adapter: `/home/mark/qwen_diffusion/runs/fastdllm_qwen25_1p5b_toolcall_lora_clean_smoke`
- LoRA target modules: `q_proj,k_proj,v_proj,o_proj`
- LoRA rank/alpha/dropout: `16/32/0.05`

## Data

Seed data came from the local normalized public tool-call set:

- `NousResearch/hermes-function-calling-v1`
- `glaiveai/glaive-function-calling-v2`
- `Team-ACE/ToolACE`

The first strict run used only Hermes examples because they already use the Qwen
`<tool_call>{...}</tool_call>` shape.

Generated local files:

- Train: `/home/mark/qwen_diffusion/data/fastdllm_toolcall_train/train_toolcall.json`
- Eval: `/home/mark/qwen_diffusion/data/toolcall_eval/fastdllm_toolcall_smoke.jsonl`

The corrected split uses 96 train examples and 8 held-out eval examples.

## Training Run

Command shape:

```bash
OUTPUT_DIR=/home/mark/qwen_diffusion/runs/fastdllm_qwen25_1p5b_toolcall_lora_clean_smoke \
MAX_STEPS=300 \
MAX_TRAIN_SAMPLES=96 \
BLOCK_SIZE=1024 \
scripts/run_fastdllm_qwen25_1p5b_toolcall_lora_smoke.sh
```

Training metrics:

- steps: 300
- epoch: 3.125
- train loss: 1.3588
- train runtime: 91.31 seconds
- train steps/sec: 3.285
- train samples/sec: 3.285

## Held-Out Eval

Eval command shape:

```bash
.venv-fastdllm/bin/python scripts/eval_fastdllm_toolcall_smoke.py \
  --limit 8 \
  --max-new-tokens 192 \
  --adapter runs/fastdllm_qwen25_1p5b_toolcall_lora_clean_smoke \
  --out runs/fastdllm_qwen25_1p5b_toolcall_lora_clean_smoke/trained_toolcall_eval_heldout.jsonl
```

Baseline, no adapter:

- records: 8
- valid strict tool JSON: 0/8
- any parsed `<tool_call>`: 0/8
- exact tool-name set: 0/8
- contains all gold tool names: 0/8
- mentions any gold tool name: 0/8
- generated tokens/sec: 56.53

Trained adapter:

- records: 8
- valid strict tool JSON: 0/8
- any parsed `<tool_call>`: 0/8
- exact tool-name set: 0/8
- contains all gold tool names: 0/8
- mentions any gold tool name: 0/8
- generated tokens/sec: 104.76

## What Happened

The adapter trained and changed behavior, but the held-out generations did not
produce schema-valid tool calls. On an earlier overlapping train-probe, outputs
often moved toward JSON-like text and sometimes included plausible function
names, but still omitted `<tool_call>` tags and malformed braces/quotes. On the
correct held-out split, even loose gold-name mention was 0/8.

This means the current 1.5B Fast-dLLM LoRA path is enough to prove local
training/eval plumbing, but not enough to preserve agentic symbolic structure.

## Next Decision

Do not spend much more time on generic LoRA over mixed public tool-call data for
the 1.5B lab model. The next useful experiment should make structure easier:

- train on synthetic one-call examples first, then multi-call examples
- include explicit AR-teacher distillation from Qwen3.6-27B
- add constrained structural decoding or repair for `<tool_call>` spans
- evaluate train-set overfit and held-out split separately
- then move the same loop to Qwen3.5-9B once the teacher/eval path is stable
