# REPRODUCE_V2.md

This is the external reproduction guide for the current winning standalone diffusion system:

- Base: `Qwen/Qwen3.5-9B`
- Conversion foundation: B@1000 two-stream Fast-dLLM adapter
- Campaign base: Run-1 copy-grounded adapter
- Agentic adapter: RL-v2 diffu-GRPO adapter
- Serving mode: `hybrid-clean`, meaning grammar/FSM-forced structural tokens are bulk committed and every parameter value token is decoded sequentially with model forwards

The promoted serving row is label-free and audited:

| slice | exact_args | episode exact | valid | sec/turn | forwards/turn | value projection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| matched-20 | 47/63 | 13/20 | 63/63 | 3.904 | 56.83 | 0 |
| never-train BFCL/API-Bank | 83/184 | 19/60 | 184/184 | 2.123 | 24.62 | 0 |

## 0. Sampler-Pinning Rule

Treat the sampler implementation as part of the model result. Every gate report must record:

- repo git commit
- script sha256
- sampler function name
- adapter path
- dataset path and, where available, manifest hash
- decode flags
- generated-token/value-projection audit file

For GSM8K retention and block-quality continuity, the valid sampler is:

```text
scripts/eval_flare_stage1_ab_diffusion.py::full_context_sample_one
```

Do not substitute `scripts/measure_block_quality_curve.py` for that gate. That script is a mutable-remask fixed-K diagnostic and is not the continuity sampler. The prior S1 gate failure came from a report labeling mistake: the command path was mutable-remask while the report text called it legacy.

For tool-call serving gates, the value-projection invariant is hard:

```text
projected_value_tokens_exact == 0
parallel_commit_forced_tokens_counter == 0
wave1_projected_tokens == 0
wave1_value_tokens_counter == 0
wave2_forced_tokens_counter == 0
zero_forward_rows == 0
```

## 1. Hardware, Repos, and Environments

Validated hardware:

```text
GPU: NVIDIA GeForce RTX 5090, 32607 MiB
Driver: 595.71.05
```

Validated local Python stacks:

```text
.venv-fastdllm: Python 3.10.20, torch 2.12.1+cu130, CUDA 13.0, transformers 4.53.1, datasets 2.14.6, peft 0.19.1
.venv-vllm:     Python 3.12.13, torch 2.11.0+cu130, CUDA 13.0, vLLM 0.23.0
```

Clone the two repos and pin the exact source state used for this guide:

```bash
export ROOT=/home/mark/qwen_diffusion
export FLYWHEEL=/home/mark/shared/lumoFlyWheel_codex_fork

git clone git@github.com:MaCoredroid/qwen-diffusion-agentic.git "$ROOT"
cd "$ROOT"
git checkout 782b4411356ab6519380f6f77676a12f61c10e0f
git submodule update --init --recursive

git clone git@github.com:MaCoredroid/Lumo_FlyWheel-qwen-diffusion.git "$FLYWHEEL"
git -C "$FLYWHEEL" checkout b91184d08cd73ffc6dd11a22a52288358cd63845
```

If your SSH remote alias differs, replace `git@github.com:...` with your GitHub remote. The local fork remote used in this run was `github.com-primary:MaCoredroid/Lumo_FlyWheel-qwen-diffusion.git`.

Create the training/eval environment:

```bash
cd "$ROOT"
python3.10 -m venv .venv-fastdllm
source .venv-fastdllm/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.12.1+cu130 torchvision torchaudio
python -m pip install -r fast-dllm/v2/requirements.txt
python -m pip install -e fast-dllm/v2
python -m pip install transformers==4.53.1 datasets==2.14.6 peft==0.19.1 accelerate bitsandbytes safetensors
deactivate
```

Install `jq` as a system CLI for the acceptance checks:

```bash
sudo apt-get update
sudo apt-get install -y jq
```

Create the vLLM environment for the AR parity route:

```bash
cd "$ROOT"
python3.12 -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.11.0+cu130
python -m pip install vllm==0.23.0
deactivate
```

Download the exact base model snapshot:

```bash
cd "$ROOT"
.venv-fastdllm/bin/huggingface-cli download Qwen/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a
```

## 2. Script Pins

Current source commit for the qwen-diffusion scripts: `782b4411356ab6519380f6f77676a12f61c10e0f`.

Current source commit for the lumoFlyWheel export/serving scripts: `b91184d08cd73ffc6dd11a22a52288358cd63845`.

Script sha256 pins:

| script | sha256 |
| --- | --- |
| `scripts/init_qwen35_fastdllm_candidate.py` | `65b94c94e82d30096222880854cee703e569c2e25e2b16f811417d05074896b3` |
| `scripts/materialize_qwen35_fastdllm_weights.py` | `8ee0c6c4e43072f6931dadaf563636a3d69155f790adccb257a4c0e399fb01e4` |
| `scripts/check_qwen35_diffusion_readiness.py` | `82cd93ff5cd48af0d30978be7d6aa586248940217a906245acd584fbec25ba8e` |
| `scripts/build_flare_stage1_ab_pilot_data.py` | `b49422f7331e2968110b14432d59b7b1fa0365d8fa90806733df145a4e463d61` |
| `scripts/run_flare_stage1_ab_pilot_job.sh` | `498c20cb849575c71f0b1fc66bd7c2b8aae5c03281472ce31af139ff9c6fb23c` |
| `scripts/run_fastdllm_qwen35_9b_agentic_qlora_pilot.sh` | `a635dea556a86b7e94565afb53d45a8aecfda8ea85ed9a48cba2c778e2f5c54e` |
| `scripts/eval_flare_stage1_ab_diffusion.py` | `eaa78d7a9abfb32b7ab73c7753cf87026741e372ef13a1c0f8e44ead79b5e503` |
| `scripts/build_flare_redesign_run1_copy_mix.py` | `22bb2585448a4fa0f1e9fc4791247b21c9f15846628e8c4db7408180e68a4e7f` |
| `scripts/run_flare_redesign_run1.sh` | `bd9ce05f90fbbcca9703dcdcdc6bdc38b6219f639c7898f8bcbb5807876e1579` |
| `scripts/eval_flare_redesign_run1.sh` | `d4083959e87b4b87541c4ecd1d6f9d4f5778bc5b204ddccb90b894a90d16118a` |
| `scripts/build_rl_multiturn_v2_pool.py` | `9ded5be9b781bf1f0e0c5eb5554c59ff5b77352a4c67ce93c9727245bf568602` |
| `scripts/run_rl_multiturn_grpo_v2_pilot.sh` | `f2dcf0b0262a2266893159344bc56055ef9bafe3c1e630ebc91bb38fb8ef888f` |
| `scripts/rl_multiturn_grpo_pilot.py` | `ea71fb89a1fa021be34d7aab95996679719a3ebb0e7bb5178331e7c1bbb8355f` |
| `scripts/eval_flare_northstar_matched.py` | `4cda3acf752c093a0ee3d3e1208c2cdc5deb064b027d984cdf54e8fa93b6203f` |
| `scripts/eval_flare_northstar_hybrid_clean.py` | `a4c66751008390ec44ff4fbb7d025352dc71ba21a005948411883818b908b1f3` |
| `scripts/audit_value_projection_tokens.py` | `7b203e3e8e2a7a7bbfa6f831be295543c728b08d9228bd241f0f07e35a620b40` |
| `$FLYWHEEL/scripts/export_qwen35_9b_fastdllm_vllm.py` | `6d507ec9ba3308ff7e0f600bc0b5ec7c4ff96f66eff4e4e92175d42af7a119d5` |
| `$FLYWHEEL/scripts/qwen35_9b_host_vllm_serve.sh` | `66fe88c7db972435010b1dfb159979de80476e288d3e2b55e9f26d5e7a6e618b` |
| `$FLYWHEEL/scripts/qwen35_9b_p1a_parity_gate.sh` | `a3420b064488d563bb3a37af64e2ccb9a75865fa10f32c39a7ef21fa5278d232` |

Historical sampler run pins:

| gate | sampler | git hash at run | notes |
| --- | --- | --- | --- |
| Run-1 GSM8K/block-quality continuity | `scripts/eval_flare_stage1_ab_diffusion.py::full_context_sample_one` | `48d7b3dca1f9122be6ecce87454b0d308ae67b68` | legacy full-context/fresh-block/mask-ban |
| RL-v2 matched-20 careful | `scripts/eval_flare_northstar_matched.py::run_diffusion -> scripts/eval_fastdllm_toolcall_cases.py::full_context_sample` | `bdc8001730c5c64443f8047e53e1bc20200a233a` | generated-token audit required |

## 3. Base Model Conversion

Create the Fast-DLLM candidate with the mask token and materialized Qwen3.5 weights:

```bash
cd "$ROOT"
mkdir -p runs/reproduce_v2

.venv-fastdllm/bin/python scripts/init_qwen35_fastdllm_candidate.py \
  --raw-model Qwen/Qwen3.5-9B \
  --out-dir models/qwen3.5-9b-fastdllm-init \
  --bd-size 32 \
  --mask-token '|<MASK>|' \
  --overwrite

.venv-fastdllm/bin/python scripts/materialize_qwen35_fastdllm_weights.py \
  --raw-model Qwen/Qwen3.5-9B \
  --candidate-dir models/qwen3.5-9b-fastdllm-init \
  --write

.venv-fastdllm/bin/python scripts/check_qwen35_diffusion_readiness.py \
  --root "$ROOT" \
  --model Qwen/Qwen3.5-9B \
  --candidate-model-path models/qwen3.5-9b-fastdllm-init \
  --training-python .venv-fastdllm/bin/python \
  --metadata-python .venv-fastdllm/bin/python \
  --json-out runs/reproduce_v2/readiness.json
```

Expected manifest:

```text
models/qwen3.5-9b-fastdllm-init/conversion_manifest.json
raw_model=Qwen/Qwen3.5-9B
bd_size=32
mask_token=|<MASK>|
mask_token_id=248077
bridge_status=implemented
gdn_mode=option_a_causal_gdn_v0
has_weights=true
```

## 4. B@1000 Two-Stream Conversion Foundation

Build the Stage-1 A/B pilot data:

```bash
cd "$ROOT"
.venv-fastdllm/bin/python scripts/build_flare_stage1_ab_pilot_data.py \
  --out-dir data/flare_stage1_ab_pilot_train \
  --gsm8k-train 160 \
  --mbpp-train 96 \
  --seed 20260701
```

Train the two-stream B adapter for 1000 steps:

```bash
cd "$ROOT"
DATASET_DIR="$ROOT/data/flare_stage1_ab_pilot_train" \
MODEL_PATH="$ROOT/models/qwen3.5-9b-fastdllm-init" \
MAX_STEPS=1000 \
MAX_TRAIN_SAMPLES=256 \
BLOCK_SIZE=1024 \
TRAIN_BD_SIZE=32 \
LEARNING_RATE=1e-5 \
GRAD_ACCUM=1 \
SAVE_STEPS=1000 \
SAVE_TOTAL_LIMIT=1 \
LORA_R=8 \
LORA_ALPHA=16 \
LORA_DROPOUT=0.05 \
LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj \
SEED=20260701 \
DATA_SEED=20260701 \
scripts/run_flare_stage1_ab_pilot_job.sh \
  two_stream \
  runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000
```

The runner sets:

```text
FASTDLLM_FLARE_TWO_STREAM=1
FLARE_TWO_STREAM=1
FASTDLLM_FLARE_DEBUG=1000
```

Validated compute budget on a single RTX 5090:

```text
wall_seconds=18945
train_runtime=18922.0906
gpu_peak_memory_mib=29612
train_loss=3.9429622707366945
```

Export the B@1000 clean stream into the vLLM-loadable Qwen3.5 conditional wrapper:

```bash
cd "$ROOT"
.venv-fastdllm/bin/python "$FLYWHEEL/scripts/export_qwen35_9b_fastdllm_vllm.py" \
  --official-model "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a" \
  --converted-model "$ROOT/models/qwen3.5-9b-fastdllm-init" \
  --adapter "$ROOT/runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000" \
  --output "$ROOT/models/qwen3.5-9b-fastdllm-b1000-vllm-bf16" \
  --overwrite
```

Expected export manifest:

```text
models/qwen3.5-9b-fastdllm-b1000-vllm-bf16/lumo_export_manifest.json
schema=lumo.qwen35_9b_fastdllm_vllm_export.v1
official_model=.../models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
converted_model=models/qwen3.5-9b-fastdllm-init
adapter=runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000
replacement_count=427
mapped_text_tensors=427
lora_merge_count=32
lora_scale=2.0
lora r=8
lora_alpha=16
lora_dropout=0.05
lora_targets=q_proj,k_proj,v_proj,o_proj
```

Run the P1a AR parity gate:

```bash
cd "$ROOT"
REPO="$FLYWHEEL" \
HF_PY="$ROOT/.venv-fastdllm/bin/python" \
VLLM_PY="$ROOT/.venv-vllm/bin/python" \
RUN_DIR="$ROOT/runs/p1a_qwen35_9b_ar_parity_reproduce_v2" \
PORT=9951 \
MODEL=qwen3.5-9b-fastdllm-b1000-bf16 \
"$FLYWHEEL/scripts/qwen35_9b_p1a_parity_gate.sh"
```

B@1000 GSM8K note:

- The historical target for the B@1000 two-stream conversion gate was `13/20 = 0.65` on GSM8K first20.
- The corrected drift post-mortem established that the current reproducible legacy continuity point for the exact saved B@1000 adapter is `11/20 = 0.55`, while the downstream Run-1 adapter is the first currently reproducible `>=0.65` checkpoint.
- Do not use mutable-remask rows to judge B@1000. If B@1000 is below `11/20` on the pinned legacy sampler, stop. If Run-1 does not clear `>=13/20`, stop.

The B@1000 continuity eval command is:

```bash
cd "$ROOT"
FASTDLLM_FLARE_GDN_ROUTE=route_i \
FASTDLLM_GDN_KERNEL=torch \
.venv-fastdllm/bin/python scripts/eval_flare_stage1_ab_diffusion.py \
  --base-model models/qwen3.5-9b-fastdllm-init \
  --adapter-b runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000 \
  --model-names B_two_stream \
  --train-path data/flare_stage1_ab_pilot_train/train_agentic_mix.json \
  --heldout-nll data/flare_stage1_ab_pilot_train/heldout_nll.jsonl \
  --gsm8k-path data/phaseA_retention/gsm8k_main_test_first20.jsonl \
  --gsm8k-fewshot-path data/phaseA_retention/gsm8k_main_train_first5.jsonl \
  --out-dir runs/reproduce_v2/b1000_legacy_fullctx_gsm8k \
  --generation-tasks gsm8k \
  --generation-limit 20 \
  --full-context-generation \
  --fresh-generation-blocks \
  --block-size 32 \
  --small-block-size 32 \
  --max-new-tokens 256 \
  --threshold 0.9 \
  --temperature 0.0 \
  --top-p 0.95 \
  --mask-id 248077 \
  --stop-token-id 248046 \
  --skip-nll
```

Check:

```bash
jq '.models.B_two_stream.generation.summary.gsm8k' \
  runs/reproduce_v2/b1000_legacy_fullctx_gsm8k/summary.json
```

## 5. Run-1 Copy-Grounding

Build the copy/retention/public mix and train the campaign base:

```bash
cd "$ROOT"
ENV_PY="$ROOT/.venv-fastdllm/bin/python" \
DATASET_DIR="$ROOT/data/flare_redesign_run1_copy_retention_mix" \
OUTPUT_DIR="$ROOT/runs/flare_redesign_run1_copy_grounded_qwen35_9b" \
MODEL_PATH="$ROOT/models/qwen3.5-9b-fastdllm-init" \
MAX_STEPS=400 \
MAX_TRAIN_SAMPLES=5055 \
BLOCK_SIZE=512 \
TRAIN_BD_SIZE=32 \
GRAD_ACCUM=1 \
LEARNING_RATE=1e-5 \
SAVE_STEPS=100 \
SAVE_TOTAL_LIMIT=4 \
LOGGING_STEPS=5 \
LORA_R=16 \
LORA_ALPHA=32 \
LORA_DROPOUT=0.05 \
LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj \
DISABLE_GROUP_TEXTS=0 \
TRUNCATION_SIDE=left \
CONVERSATION_TEMPLATE=fast_dllm_v2_native \
VALUE_SPAN_LOSS_WEIGHT=2.0 \
VALUE_SPAN_MASK_PROB=1.0 \
SEED=71101 \
DATA_SEED=71101 \
scripts/run_flare_redesign_run1.sh
```

The wrapper exports the required two-stream/copy-grounding schedule:

```text
FASTDLLM_FLARE_TWO_STREAM=1
FLARE_TWO_STREAM=1
FASTDLLM_FLARE_GDN_ROUTE=route_i
FASTDLLM_FLARE_MASK_RATE_MIN=0.3
FASTDLLM_FLARE_MASK_RATE_MAX=0.8
FASTDLLM_FLARE_ADAPTIVE_COPY_SCHEDULE=1
FASTDLLM_FLARE_HIGH_ENTROPY_MASK_RATE_MIN=0.02
FASTDLLM_FLARE_HIGH_ENTROPY_MASK_RATE_MAX=0.12
FASTDLLM_BATCH_FLARE_NOISY_GDN=1
FASTDLLM_OPTIMIZE_FLARE_CLEAN_GDN=1
FASTDLLM_GDN_KERNEL=fla
```

The underlying trainer defaults to `LR_SCHEDULER_TYPE=cosine`. This is the r16/cosine Run-1 envelope.

Expected data manifest:

```text
data/flare_redesign_run1_copy_retention_mix/train_agentic_mix.manifest
count=5055
seed=71101
source_counts:
  synthetic-copy=2048
  retention=2560
  public-toolcall-pool=447
eval_overlap_removed=64
copy_span_count=7363
block_size=512
truncation_side=left
```

Validated compute budget:

```text
train_runtime=2068.1572 seconds
train_samples=3578
train_loss=4.390680780410767
```

Run the Run-1 eval wrapper:

```bash
cd "$ROOT"
ENV_PY="$ROOT/.venv-fastdllm/bin/python" \
OUT_ROOT="$ROOT/runs/reproduce_v2/run1_eval" \
TAUS="0.99" \
GSM8K_PARALLEL_COMMIT_THRESHOLD=0.9 \
scripts/eval_flare_redesign_run1.sh \
  "$ROOT/runs/flare_redesign_run1_copy_grounded_qwen35_9b"
```

Run the pinned legacy GSM8K continuity gate directly when producing a report:

```bash
cd "$ROOT"
FASTDLLM_FLARE_GDN_ROUTE=route_i \
FASTDLLM_GDN_KERNEL=torch \
.venv-fastdllm/bin/python scripts/eval_flare_stage1_ab_diffusion.py \
  --base-model models/qwen3.5-9b-fastdllm-init \
  --adapter-b runs/flare_redesign_run1_copy_grounded_qwen35_9b \
  --model-names Run1 \
  --train-path data/flare_redesign_run1_copy_retention_mix/train_agentic_mix.json \
  --heldout-nll data/flare_stage1_ab_pilot_train/heldout_nll.jsonl \
  --gsm8k-path data/phaseA_retention/gsm8k_main_test_first20.jsonl \
  --gsm8k-fewshot-path data/phaseA_retention/gsm8k_main_train_first5.jsonl \
  --out-dir runs/reproduce_v2/run1_legacy_fullctx_gsm8k \
  --generation-tasks gsm8k \
  --generation-limit 20 \
  --full-context-generation \
  --fresh-generation-blocks \
  --block-size 32 \
  --small-block-size 32 \
  --max-new-tokens 256 \
  --threshold 0.9 \
  --temperature 0.0 \
  --top-p 0.95 \
  --mask-id 248077 \
  --stop-token-id 248046 \
  --skip-nll
```

Expected Run-1 gates:

```text
GSM8K corrected legacy strict: 14/20 to 15/20 = 0.70 to 0.75
Minimum to proceed: 13/20 = 0.65
Copy-span isolation at careful value_tpf=1.0: 41/41 copied args exact
Native held/public copy quality improved versus B@1000
```

Historical copy gates from the accepted red-team:

```text
copyspan_isolation arg8_tau099: copy_args_exact=41/41, value_tpf=1.0
native proxy heldout copy args: 41/52 = 0.788
native proxy public copy args: 55/60 = 0.917
```

## 6. RL-v2 diffu-GRPO

Build the v2 public training pool with explicit leak filtering against the frozen eval batteries:

```bash
cd "$ROOT"
.venv-fastdllm/bin/python scripts/build_rl_multiturn_v2_pool.py \
  --local-native-jsonl data/multiturn_sft_warmstart/public_train_multicall_native.jsonl \
  --out-jsonl data/rl_multiturn_v2_public_pool/episodes.jsonl \
  --manifest-json data/rl_multiturn_v2_public_pool/manifest.json \
  --rejected-jsonl data/rl_multiturn_v2_public_pool/rejected.jsonl \
  --target-count 240 \
  --min-count 150 \
  --local-first-count 120 \
  --min-turns 2 \
  --max-turns 6 \
  --max-tools 12 \
  --toolace-dataset Team-ACE/ToolACE \
  --toolace-split train \
  --max-toolace-stream-rows 50000 \
  --prompt-tokenizer-path models/qwen3.5-9b-fastdllm-b1000-vllm-bf16 \
  --chat-template-path "$FLYWHEEL/docker/chat_templates/qwen3-openai-codex.jinja" \
  --max-prompt-tokens 3500 \
  --max-new-tokens 384
```

Expected pool manifest:

```text
records=240
turns=512
episode_set_hash=5ffd1ad5ef7151bc746b83f41f94a9337919cae6cfa05b0a33c08ae926a7a5ed
source_family_counts:
  ToolACE-derived=188
  public-native-warmstart=52
selected_overlap_counts:
  fingerprint=0
  id=0
  public_eval_hash=0
  user_text=0
```

Launch RL-v2 from Run-1:

```bash
cd "$ROOT"
PYTHON="$ROOT/.venv-fastdllm/bin/python" \
INPUT_JSONL="$ROOT/data/rl_multiturn_v2_public_pool/episodes.jsonl" \
ADAPTER="$ROOT/runs/flare_redesign_run1_copy_grounded_qwen35_9b" \
OUT_DIR="$ROOT/runs/rl_multiturn_grpo_v2/from_selected_base_g4_step300" \
MAX_STEPS=300 \
GROUP_SIZE=4 \
LEARNING_RATE=5e-6 \
KL_TO_BASE_COEFF=0.05 \
RETENTION_PROBE_EVERY_STEPS=50 \
RETENTION_PROBE_LIMIT=5 \
EPISODE_LIMIT=240 \
scripts/run_rl_multiturn_grpo_v2_pilot.sh
```

RL-v2 fixed config:

```text
base adapter=runs/flare_redesign_run1_copy_grounded_qwen35_9b
policy=diffusion careful + live Qwen-native grammar + audited reward
group_size=4
max_steps=300
learning_rate=5e-6
KL_TO_BASE_COEFF=0.05
KL positions=parameter-value/free assistant tokens only
grammar-forced structural tokens masked out of policy loss
retention probes every 50 steps, N=5
seed=20260702
```

Validated compute/training accounting:

```text
elapsed_hours=3.98
nonzero_advantage_steps=214/300
zero_advantage_steps=86/300
value/free policy tokens=13858
grammar-forced structural tokens masked=58886
mean_step_reward=0.9556
mean_reward_last20=0.9796
max_KL_to_base=0.07482
max_grad_norm=43.746
trainable_params=18112512
```

Expected RL-v2 gates:

```text
GSM8K retention strict/flex: 13/20 = 0.65, pass
matched-20 careful exact_args: 44/63
matched-20 episode_exact: 11/20
matched-20 valid_tool_json: 62/63
matched-20 forwards/turn: 95.508
matched-20 sec/turn: 6.686
value projection: 0
```

Run the matched-20 careful gate explicitly:

```bash
cd "$ROOT"
.venv-fastdllm/bin/python scripts/eval_flare_northstar_matched.py \
  --backend diffusion \
  --input-jsonl data/toolcall_eval_native/flare_scaleup_native_58.jsonl \
  --out-dir runs/rl_multiturn_grpo_v2/from_selected_base_g4_step300_matched20 \
  --episode-limit 20 \
  --min-turns 3 \
  --max-turns 6 \
  --prompt-tokenizer-path models/qwen3.5-9b-fastdllm-b1000-vllm-bf16 \
  --chat-template-path "$FLYWHEEL/docker/chat_templates/qwen3-openai-codex.jinja" \
  --base-model models/qwen3.5-9b-fastdllm-init \
  --adapter runs/rl_multiturn_grpo_v2/from_selected_base_g4_step300/adapter_model \
  --tokenizer-path models/qwen3.5-9b-fastdllm-b1000-vllm-bf16 \
  --no-merge-adapter \
  --block-size 32 \
  --small-block-size 32 \
  --max-new-tokens 384 \
  --max-extra-tokens 12 \
  --threshold 0.9 \
  --top-p 0.95 \
  --temperature 0.0 \
  --diffusion-condition baseline_careful \
  --diffusion-structural-only \
  --diffusion-output-name diffusion_careful

.venv-fastdllm/bin/python scripts/audit_value_projection_tokens.py \
  --rows runs/rl_multiturn_grpo_v2/from_selected_base_g4_step300_matched20/diffusion_careful/turns.jsonl \
  --tokenizer models/qwen3.5-9b-fastdllm-b1000-vllm-bf16 \
  --out-json runs/rl_multiturn_grpo_v2/from_selected_base_g4_step300_matched20/diffusion_careful/projection_value_audit.json \
  --out-jsonl runs/rl_multiturn_grpo_v2/from_selected_base_g4_step300_matched20/diffusion_careful/projection_value_audit.jsonl
```

The adapter used by the winning hybrid serving mode is:

```text
runs/rl_multiturn_grpo_v2/from_selected_base_g4_step300/adapter_model
```

## 7. Hybrid-Clean Serving

Hybrid-clean serving is the constrained lane:

```text
FSM grammar forced tokens: bulk committed without model forwards
Parameter values: sequential, K=1 per value token, chain-rule preserving
Free/non-forced tokens: sequential
Value projection: forbidden
```

Run matched-20:

```bash
cd "$ROOT"
.venv-fastdllm/bin/python scripts/eval_flare_northstar_hybrid_clean.py \
  --input-jsonl data/toolcall_eval_native/flare_scaleup_native_58.jsonl \
  --out-dir runs/hybrid_forced_grammar_seq_values_v2/matched20 \
  --episode-limit 20 \
  --min-turns 3 \
  --max-turns 6 \
  --prompt-tokenizer-path models/qwen3.5-9b-fastdllm-b1000-vllm-bf16 \
  --chat-template-path "$FLYWHEEL/docker/chat_templates/qwen3-openai-codex.jinja" \
  --base-model models/qwen3.5-9b-fastdllm-init \
  --adapter runs/rl_multiturn_grpo_v2/from_selected_base_g4_step300/adapter_model \
  --tokenizer-path models/qwen3.5-9b-fastdllm-b1000-vllm-bf16 \
  --no-merge-adapter \
  --block-size 32 \
  --max-new-tokens 384 \
  --top-p 0.95 \
  --temperature 0.0 \
  --grammar-topk 256
```

Audit matched-20:

```bash
cd "$ROOT"
.venv-fastdllm/bin/python scripts/audit_value_projection_tokens.py \
  --rows runs/hybrid_forced_grammar_seq_values_v2/matched20/diffusion_hybrid_forced_grammar_seq_values/turns.jsonl \
  --tokenizer models/qwen3.5-9b-fastdllm-b1000-vllm-bf16 \
  --out-json runs/hybrid_forced_grammar_seq_values_v2/matched20/diffusion_hybrid_forced_grammar_seq_values/projection_value_audit.json \
  --out-jsonl runs/hybrid_forced_grammar_seq_values_v2/matched20/diffusion_hybrid_forced_grammar_seq_values/projection_value_audit.jsonl
```

Run never-train BFCL/API-Bank:

```bash
cd "$ROOT"
.venv-fastdllm/bin/python scripts/eval_flare_northstar_hybrid_clean.py \
  --input-jsonl data/toolcall_eval_native/flare_nevertrain_bfcl_apibank.jsonl \
  --out-dir runs/hybrid_broaden_nevertrain_v2/nevertrain_bfcl_apibank60 \
  --episode-limit 60 \
  --min-turns 1 \
  --max-turns 8 \
  --prompt-tokenizer-path models/qwen3.5-9b-fastdllm-b1000-vllm-bf16 \
  --chat-template-path "$FLYWHEEL/docker/chat_templates/qwen3-openai-codex.jinja" \
  --base-model models/qwen3.5-9b-fastdllm-init \
  --adapter runs/rl_multiturn_grpo_v2/from_selected_base_g4_step300/adapter_model \
  --tokenizer-path models/qwen3.5-9b-fastdllm-b1000-vllm-bf16 \
  --no-merge-adapter \
  --block-size 32 \
  --max-new-tokens 384 \
  --top-p 0.95 \
  --temperature 0.0 \
  --grammar-topk 256
```

Audit never-train:

```bash
cd "$ROOT"
.venv-fastdllm/bin/python scripts/audit_value_projection_tokens.py \
  --rows runs/hybrid_broaden_nevertrain_v2/nevertrain_bfcl_apibank60/diffusion_hybrid_forced_grammar_seq_values/turns.jsonl \
  --tokenizer models/qwen3.5-9b-fastdllm-b1000-vllm-bf16 \
  --out-json runs/hybrid_broaden_nevertrain_v2/nevertrain_bfcl_apibank60/diffusion_hybrid_forced_grammar_seq_values/projection_value_audit.json \
  --out-jsonl runs/hybrid_broaden_nevertrain_v2/nevertrain_bfcl_apibank60/diffusion_hybrid_forced_grammar_seq_values/projection_value_audit.jsonl
```

Expected hybrid-clean results:

```text
matched-20:
  exact_args=47/63
  valid_tool_call=63/63
  exact_tool_sequence=63/63
  episode_exact=13/20
  forwards_per_turn=56.83
  tokens_per_forward=1.36
  sec_per_turn=3.904
  speedup_vs_careful_wall=1.71x

never-train BFCL/API-Bank:
  exact_args=83/184
  valid_tool_call=184/184
  exact_tool_sequence=133/184
  episode_exact=19/60
  forwards_per_turn=24.62
  sec_per_turn=2.123
  speedup_vs_careful_wall=1.59x
```

## 8. Acceptance Battery

Run the summary checks:

```bash
cd "$ROOT"
jq '{exact_args, episode_exact, valid_tool_call, exact_tool_sequence, schema_ok, sec_per_turn, forwards_per_turn, tokens_per_forward}' \
  runs/hybrid_forced_grammar_seq_values_v2/matched20/summary.json

jq '{exact_args, episode_exact, valid_tool_call, exact_tool_sequence, schema_ok, sec_per_turn, forwards_per_turn, tokens_per_forward}' \
  runs/hybrid_broaden_nevertrain_v2/nevertrain_bfcl_apibank60/summary.json
```

Expected:

```json
{
  "matched20": {
    "exact_args": 47,
    "episode_exact": 13,
    "valid_tool_call": 63,
    "exact_tool_sequence": 63,
    "sec_per_turn": 3.904,
    "forwards_per_turn": 56.83
  },
  "nevertrain": {
    "exact_args": 83,
    "episode_exact": 19,
    "valid_tool_call": 184,
    "exact_tool_sequence": 133,
    "sec_per_turn": 2.123,
    "forwards_per_turn": 24.62
  }
}
```

Run the zero-projection checks:

```bash
cd "$ROOT"
jq '.totals | {
  rows,
  exact_args,
  zero_projected_value_tokens_verified,
  projected_value_tokens_exact,
  parallel_commit_forced_tokens_counter,
  wave1_projected_tokens,
  wave1_value_tokens_counter,
  wave2_forced_tokens_counter,
  zero_forward_rows,
  exact_rows_dependent_on_projected_values,
  reported_model_value_tokens,
  true_xml_value_tokens,
  verification_mode
}' \
  runs/hybrid_forced_grammar_seq_values_v2/matched20/diffusion_hybrid_forced_grammar_seq_values/projection_value_audit.json

jq '.totals | {
  rows,
  exact_args,
  zero_projected_value_tokens_verified,
  projected_value_tokens_exact,
  parallel_commit_forced_tokens_counter,
  wave1_projected_tokens,
  wave1_value_tokens_counter,
  wave2_forced_tokens_counter,
  zero_forward_rows,
  exact_rows_dependent_on_projected_values,
  reported_model_value_tokens,
  true_xml_value_tokens,
  verification_mode
}' \
  runs/hybrid_broaden_nevertrain_v2/nevertrain_bfcl_apibank60/diffusion_hybrid_forced_grammar_seq_values/projection_value_audit.json
```

Expected matched-20 audit:

```text
rows=63
exact_args=47
zero_projected_value_tokens_verified=1
projected_value_tokens_exact=0
parallel_commit_forced_tokens_counter=0
wave1_projected_tokens=0
wave1_value_tokens_counter=0
wave2_forced_tokens_counter=0
zero_forward_rows=0
exact_rows_dependent_on_projected_values=0
reported_model_value_tokens=2846
true_xml_value_tokens=2061
verification_mode=no_projection_events
```

Expected never-train audit:

```text
rows=184
exact_args=83
zero_projected_value_tokens_verified=1
projected_value_tokens_exact=0
parallel_commit_forced_tokens_counter=0
wave1_projected_tokens=0
wave1_value_tokens_counter=0
wave2_forced_tokens_counter=0
zero_forward_rows=0
exact_rows_dependent_on_projected_values=0
reported_model_value_tokens=2932
true_xml_value_tokens=1397
verification_mode=no_projection_events
```

Acceptance fails if any value-projection counter is nonzero, if `zero_forward_rows` is nonzero, or if the sampler path is not `diffusion_hybrid_forced_grammar_seq_values`.

## 9. Compute Budget Summary

| stage | GPU | wall time | peak memory | output |
| --- | --- | ---: | ---: | --- |
| Base conversion/materialization | CPU/GPU optional | minutes to tens of minutes, IO-bound | model-size dependent | `models/qwen3.5-9b-fastdllm-init` |
| B@1000 two-stream | 1x RTX 5090 32GB | 18,945 s / 5.26 h | 29,612 MiB | `runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000` |
| B@1000 vLLM export | CPU + disk IO | tens of minutes, shard IO-bound | CPU RAM/model-size dependent | `models/qwen3.5-9b-fastdllm-b1000-vllm-bf16` |
| Run-1 copy-grounding | 1x RTX 5090 32GB | 2,068 s / 34.5 min | fits 32GB with QLoRA | `runs/flare_redesign_run1_copy_grounded_qwen35_9b` |
| RL-v2 GRPO | 1x RTX 5090 32GB | 14,342 s / 3.98 h | fits 32GB with QLoRA | `runs/rl_multiturn_grpo_v2/from_selected_base_g4_step300/adapter_model` |
| Hybrid matched-20 eval | 1x RTX 5090 32GB | about 4 s/turn | eval only | `runs/hybrid_forced_grammar_seq_values_v2/matched20` |
| Hybrid never-train eval | 1x RTX 5090 32GB | about 2.1 s/turn | eval only | `runs/hybrid_broaden_nevertrain_v2/nevertrain_bfcl_apibank60` |

## 10. Stop Conditions

Stop and debug before continuing if any of these occur:

- base conversion manifest lacks `mask_token_id=248077` or `has_weights=true`
- B@1000 export manifest has `replacement_count != 427` or `lora_merge_count != 32`
- Run-1 GSM8K corrected legacy strict is `<13/20`
- RL-v2 GSM8K retention strict or flex is `<13/20`
- RL-v2 matched-20 careful has nonzero projected value tokens
- hybrid-clean matched-20 exact_args is materially below `47/63`
- hybrid-clean never-train exact_args is materially below `83/184`
- any hybrid audit counter indicates value projection or zero-forward value emission

If a gate moves by one row on GSM8K first20, rerun once and report both seeds/artifacts. If a sampler implementation or script hash differs, do not compare the score to the table above until the report identifies the sampler path and the reason for the change.
