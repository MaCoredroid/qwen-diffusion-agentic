# Multi-Turn SFT Warm-Start Gate

## Corpus

- Builder: `scripts/build_multiturn_sft_warmstart.py`
- Dataset: `data/multiturn_sft_warmstart/lmflow_dataset/train_agentic_mix.json`
- Final mix: `98` rows = `49` audited tool-call rows + `49` retention rows
- Raw accepted self-generated tool-call turns: `36/50`
- Oversampling focus after expansion: complex value `12`, long episode `6`, long-episode final stop `4`
- Value projection audit: `0` projected value tokens; `zero_projected_value_tokens_verified=1`

## SFT

- Base model: `models/qwen3.5-9b-fastdllm-init`
- Start adapter: `runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000`
- Output adapter: `runs/multiturn_sft_warmstart_qwen35_9b_sft400`
- QLoRA: `r=8`, `alpha=16`, `dropout=0.05`, target modules `q_proj,k_proj,v_proj,o_proj`
- Steps: `400`
- Block size: `1024`
- Learning rate: `5e-6`
- Template: `fast_dllm_v2_native`
- Final training loss: `2.934233522415161`
- Final logged step loss: `2.6679`
- Runtime: `966.7889` seconds

The 1536 and 2048 token attempts OOMed. The completed run uses left truncation at 1024; label-retention checks showed no zero-label examples, but partial truncation on four long rows.

## GSM8K Retention Gate

- Gate script: `scripts/eval_flare_stage1_ab_diffusion.py`
- Gate output: `runs/multiturn_sft_warmstart_qwen35_9b_sft400_gsm8k_gate/summary.json`
- Sampler: full-context generation, fresh blocks, mask-ban path
- GSM8K file: `data/phaseA_retention/gsm8k_main_test_first20.jsonl`
- Fewshot file: `data/phaseA_retention/gsm8k_main_train_first5.jsonl`
- Decode config: `block_size=32`, `small_block_size=32`, `max_new_tokens=256`, `threshold=0.9`, `temperature=0.0`, `top_p=0.95`
- Disjointness: full-hash overlap `0`, content-hash overlap `0`
- Result: strict `2/20 = 0.10`, flex `2/20 = 0.10`
- Required gate: `>=0.70`

## Decision

FAIL. RL is blocked and was not started.

