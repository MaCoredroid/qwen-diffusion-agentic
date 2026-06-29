# Qwen3.5-9B Diffusion Pilot Readiness

Date: 2026-06-26

## Scope

This is the first guarded step toward the real Qwen3.5-9B diffusion/QLoRA loop.
It checks whether the current Fast-dLLM stack can load, train, and save a
converted text-only Qwen3.5-9B block-diffusion candidate on local hardware.

The Alienware RTX 5080 is reserved for user work during this pass, so the active
smoke training was run only on the local RTX 5090. Teacher/student flow remains
async: Qwen3.6-27B can label or repair data offline, then the 9B student can
train later without the teacher endpoint running.

## Current Result

Current readiness gate: `ready=true`.

Confirmed:

- `models/qwen3.5-9b-fastdllm-init` now has a runnable text-only Qwen3.5
  Fast-DLLM bridge.
- The bridge advertises:
  - `FAST_DLLM_QWEN3_5_BRIDGE_STATUS="implemented"`
  - `FAST_DLLM_QWEN3_5_GDN_MODE="option_a_causal_gdn_v0"`
- Raw Qwen3.5-9B shards are cached locally.
- Remapped candidate safetensor shards are materialized locally.
- Meta-tensor key audit is exact:
  - model keys: 427
  - index keys: 427
  - missing keys: 0
  - unexpected keys: 0
- Candidate tokenizer has a real special `|<MASK>|` token:
  - token id: 248077
  - encodes as `[248077]`
- The guarded QLoRA launcher now passes the readiness preflight.

The remaining warning is expected: the raw upstream Qwen3.5 tokenizer does not
ship with a single `|<MASK>|` token. The local converted candidate adds it.

## Added Artifacts

- `scripts/build_agentic_diffusion_curriculum.py`
  - merges public tool-call examples, synthetic one-call examples, synthetic
    tool-result traces, and successful Qwen Code repo-edit diffs.
- `scripts/check_qwen35_diffusion_readiness.py`
  - inspects training packages, Qwen3.5 config/tokenizer metadata, raw and
    candidate weight caches, Fast-DLLM bridge status, candidate config, and
    dataset availability.
- `scripts/init_qwen35_fastdllm_candidate.py`
  - creates the local text-only Qwen3.5 Fast-DLLM candidate with `bd_size`,
    Fast-DLLM `auto_map`, and a real special `|<MASK>|` token.
- `scripts/materialize_qwen35_fastdllm_weights.py`
  - writes remapped text-only safetensor shards for the candidate.
- `scripts/run_fastdllm_qwen35_9b_agentic_qlora_pilot.sh`
  - guarded QLoRA launcher; it exits before training if readiness regresses.
- `data/qwen35_9b_diffusion_curriculum/train_agentic_mix.json`
  - 303 de-duplicated conversation examples.
- `data/qwen35_9b_loss_smoke/train_loss_smoke.json`
  - tiny short supervised set used only to verify nonzero loss/gradients.
- `runs/qwen35_9b_diffusion_readiness.json`
  - latest saved preflight result.
- `models/qwen3.5-9b-fastdllm-init`
  - local converted text-only candidate with config/tokenizer, modeling bridge,
    and remapped weights.

## Bridge Status

The bridge is a v0 implementation for making the 9B loop trainable, not the
final serving design.

Current behavior:

- Full-attention layers receive the block-diffusion mask.
- GDN/linear-attention layers remain causal.
- During MDM training, the noisy and clean halves are processed independently
  through GDN layers to avoid target-stream leakage from the noisy stream.
- The model loop now honors Transformers gradient checkpointing, which is
  required for 512-token 9B QLoRA training to fit on the local RTX 5090.
- The CausalLM class now exposes the `sample_with_top_p` helper expected by the
  Fast-dLLM batch sampler.
- Optional batch-label instrumentation is available with
  `FASTDLLM_DEBUG_LABELS=N`; it prints pre/post-MDM non-ignored label counts for
  the first `N` model forwards.
- The first target is LoRA/QLoRA training observability, not optimized sampling.

Future work:

- GDN boundary-state snapshots for block serving/cache behavior.
- Sampler integration for the Qwen3.5 candidate.
- Larger block/window sweeps after strict tool-call metrics are stable.

## Weight Materialization

Raw checkpoint analysis:

- raw keys: 775
- text/LM-head keys kept for the candidate: 427
- vision keys dropped: 333
- MTP keys dropped for the first text-only pilot: 15
- raw total size from index: 19,306,216,416 bytes
- raw shards: 4

Candidate materialization:

- candidate safetensor shards: 4
- candidate weight-map keys: 427
- candidate size: about 17 GB
- MTP weights are intentionally excluded from the first text-only pilot.

## Curriculum Build

Command:

```bash
python3 scripts/build_agentic_diffusion_curriculum.py
```

Result:

- total examples: 303
- public tool-call: 96
- synthetic one-call: 192
- synthetic tool-result: 10
- Qwen3.6 teacher repo-edit diffs: 5

This is intentionally small. It is a pilot corpus to make the diffusion training
loop observable, not the final data mix.

## Readiness Command

Command:

```bash
python3 scripts/check_qwen35_diffusion_readiness.py \
  --json-out runs/qwen35_9b_diffusion_readiness.json
```

Current result: `ready=true`.

Useful confirmed training stack:

- Python 3.10.20
- PyTorch 2.12.1+cu130
- Transformers 4.53.1
- PEFT 0.19.1
- bitsandbytes 0.49.2
- DeepSpeed 0.19.2
- LMFlow 0.0.9

Qwen3.5-9B target architecture:

- `model_type`: `qwen3_5`
- hidden size: 4096
- layers: 32
- linear-attention/GDN layers: 24
- full-attention layers: 8
- pattern: 3 GDN layers followed by 1 full-attention layer

## Smoke Training

### Agentic Curriculum Load Smoke

Command shape:

```bash
BUILD_CURRICULUM=0 \
MAX_STEPS=5 \
MAX_TRAIN_SAMPLES=16 \
BLOCK_SIZE=32 \
GRAD_ACCUM=1 \
LORA_R=8 \
LORA_ALPHA=16 \
LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj \
OUTPUT_DIR=/home/mark/qwen_diffusion/runs/fastdllm_qwen35_9b_agentic_qlora_pilot_step5 \
scripts/run_fastdllm_qwen35_9b_agentic_qlora_pilot.sh
```

Result:

- model loaded from the remapped 9B candidate shards
- QLoRA adapters attached
- optimizer step path works
- adapter checkpoints saved
- loss was `0.0`

The zero loss is a data/windowing issue: with `BLOCK_SIZE=32`, the first chunks
from the real agentic curriculum are prompt/tool-schema tokens and contain no
assistant labels. Token probes confirm valid assistant labels exist later in the
examples.

### 512-Token Grouped Curriculum Smoke

An initial `BLOCK_SIZE=512` grouped run exposed a real memory issue: the bridge
advertised gradient checkpointing, but the model layer loop did not call the
checkpoint function. After patching the layer loop, the same 512-token grouped
run fit on the RTX 5090:

- max steps: 5
- max train samples: 16
- train runtime: 7.5777 seconds
- train samples/second: 0.66
- loss: `0.0`

This confirmed that 512-token 9B QLoRA can fit locally, but global grouping
still samples prompt-heavy windows that miss assistant labels.

### Agentic Left-Truncated Loss Smoke

Command shape:

```bash
BUILD_CURRICULUM=0 \
DISABLE_GROUP_TEXTS=1 \
TRUNCATION_SIDE=left \
MAX_STEPS=5 \
MAX_TRAIN_SAMPLES=16 \
BLOCK_SIZE=512 \
GRAD_ACCUM=1 \
LORA_R=8 \
LORA_ALPHA=16 \
LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj \
OUTPUT_DIR=/home/mark/qwen_diffusion/runs/fastdllm_qwen35_9b_agentic_qlora_pilot_b512_left_step5 \
scripts/run_fastdllm_qwen35_9b_agentic_qlora_pilot.sh
```

Result:

- trainable params: 9,056,256
- total params: 8,962,859,520
- global steps: 5
- train loss: 5.71199951171875
- final logged grad norm: 19.994108200073242
- runtime: 7.4951 seconds
- adapter saved under
  `runs/fastdllm_qwen35_9b_agentic_qlora_pilot_b512_left_step5`

This is the first real agentic-curriculum learning signal for the 9B diffusion
candidate. It uses a crude but effective pilot recipe: per-example windows,
left truncation, and 512-token sequences so the assistant/tool-call labels are
preserved.

### 100-Step Mixed Agentic Pilot

Command shape:

```bash
BUILD_CURRICULUM=0 \
DISABLE_GROUP_TEXTS=1 \
TRUNCATION_SIDE=left \
MAX_STEPS=100 \
MAX_TRAIN_SAMPLES=128 \
BLOCK_SIZE=512 \
GRAD_ACCUM=1 \
LORA_R=8 \
LORA_ALPHA=16 \
LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj \
OUTPUT_DIR=/home/mark/qwen_diffusion/runs/fastdllm_qwen35_9b_agentic_qlora_pilot_b512_left_step100 \
scripts/run_fastdllm_qwen35_9b_agentic_qlora_pilot.sh
```

Result:

- global steps: 100
- train samples: 128
- train loss: 5.716379795074463
- runtime: 139.1164 seconds
- train samples/second: 0.719
- train steps/second: 0.719
- final logged grad norm: 20.084775924682617
- late loss logs include lower chunks around `4.2265`, `4.3312`, and `5.1101`
- adapter saved under
  `runs/fastdllm_qwen35_9b_agentic_qlora_pilot_b512_left_step100`

This is the first longer nonzero 9B diffusion QLoRA run on the mixed agentic
curriculum.

### Short Loss-Smoke Training

Command shape:

```bash
BUILD_CURRICULUM=0 \
DATASET_DIR=/home/mark/qwen_diffusion/data/qwen35_9b_loss_smoke \
MAX_STEPS=5 \
MAX_TRAIN_SAMPLES=5 \
BLOCK_SIZE=32 \
GRAD_ACCUM=1 \
LORA_R=8 \
LORA_ALPHA=16 \
LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj \
OUTPUT_DIR=/home/mark/qwen_diffusion/runs/fastdllm_qwen35_9b_loss_smoke_step5 \
scripts/run_fastdllm_qwen35_9b_agentic_qlora_pilot.sh
```

Result:

- trainable params: 9,056,256
- total params: 8,962,859,520
- trainable fraction: 0.1010%
- global steps: 5
- train loss: 7.07149658203125
- final logged grad norm: 62.266544342041016
- runtime: 3.8434 seconds
- adapter and checkpoints saved under
  `runs/fastdllm_qwen35_9b_loss_smoke_step5`

This proves the current 9B bridge can load, attach QLoRA adapters, run backward,
update, and save with nonzero gradients on the local RTX 5090.

## Tiny Strict Eval

The eval scripts were updated to resolve model-specific mask/stop token ids and
to optionally use the same `fast_dllm_v2` conversation template as training.

Command shape for the trained adapter:

```bash
PYTHONPATH=/home/mark/qwen_diffusion/scripts:/home/mark/qwen_diffusion/fast-dllm/third_party \
.venv-fastdllm/bin/python scripts/eval_fastdllm_toolcall_cases.py \
  --base-model models/qwen3.5-9b-fastdllm-init \
  --adapter runs/fastdllm_qwen35_9b_agentic_qlora_pilot_b512_left_step5 \
  --no-merge-adapter \
  --conversation-template fast_dllm_v2 \
  --block-size 32 \
  --small-block-size 8 \
  --max-new-tokens 64 \
  --eval synthetic_onecall_2:data/toolcall_eval/synthetic_onecall_smoke.jsonl:runs/fastdllm_qwen35_9b_agentic_qlora_pilot_b512_left_step5_eval/synthetic_onecall_2.jsonl:2
```

Diffusion init, same 2-case slice:

- ok generations: 2/2
- exact sequence: 0/2
- exact arguments: 0/2
- valid tool JSON: 0/2
- unresolved-mask examples: 0/2
- errors: 0/2
- generated tokens/sec: 14.5063
- max CUDA allocated: 16.975 GiB

5-step left-truncated adapter, same 2-case slice:

- ok generations: 2/2
- exact sequence: 0/2
- exact arguments: 0/2
- valid tool JSON: 0/2
- unresolved-mask examples: 0/2
- errors: 0/2
- generated tokens/sec: 13.0997
- max CUDA allocated: 17.009 GiB

Interpretation:

- This is a successful train-plus-eval plumbing result.
- It is not yet a useful agentic model result.
- Five QLoRA steps are enough to prove the path and produce nonzero loss, but
  not enough to recover strict `<tool_call>` behavior.
- The generated samples are still format-poor and miss the required tools.

### 100-Step Mixed Adapter Eval

Command shape:

```bash
PYTHONPATH=/home/mark/qwen_diffusion/scripts:/home/mark/qwen_diffusion/fast-dllm/third_party \
.venv-fastdllm/bin/python scripts/eval_fastdllm_toolcall_cases.py \
  --base-model models/qwen3.5-9b-fastdllm-init \
  --adapter runs/fastdllm_qwen35_9b_agentic_qlora_pilot_b512_left_step100 \
  --no-merge-adapter \
  --conversation-template fast_dllm_v2 \
  --block-size 32 \
  --small-block-size 8 \
  --max-new-tokens 96 \
  --eval synthetic_onecall_8:data/toolcall_eval/synthetic_onecall_smoke.jsonl:runs/fastdllm_qwen35_9b_agentic_qlora_pilot_b512_left_step100_eval/synthetic_onecall_8.jsonl:8 \
  --eval public_onecall_8:data/toolcall_eval/public_onecall_hermes_smoke.jsonl:runs/fastdllm_qwen35_9b_agentic_qlora_pilot_b512_left_step100_eval/public_onecall_8.jsonl:8 \
  --eval public_multicall_4:data/toolcall_eval/public_multicall_hermes_smoke.jsonl:runs/fastdllm_qwen35_9b_agentic_qlora_pilot_b512_left_step100_eval/public_multicall_4.jsonl:4 \
  --eval synthetic_toolresult_4:data/toolcall_eval/synthetic_toolresult_smoke.jsonl:runs/fastdllm_qwen35_9b_agentic_qlora_pilot_b512_left_step100_eval/synthetic_toolresult_4.jsonl:4
```

Results:

- synthetic one-call, 8 cases:
  - ok generations: 8/8
  - exact sequence: 0/8
  - exact arguments: 0/8
  - valid tool JSON: 0/8
  - unresolved-mask examples: 0/8
  - errors: 0/8
  - generated tokens/sec: 13.0695
- public Hermes one-call, 8 cases:
  - ok generations: 8/8
  - exact sequence: 0/8
  - exact arguments: 0/8
  - valid tool JSON: 0/8
  - unresolved-mask examples: 0/8
  - errors: 0/8
  - generated tokens/sec: 13.1469
- public Hermes multi-call, 4 cases:
  - ok generations: 4/4
  - exact sequence: 0/4
  - exact arguments: 0/4
  - valid tool JSON: 0/4
  - unresolved-mask examples: 0/4
  - errors: 0/4
  - generated tokens/sec: 13.0117
- synthetic tool-result, 4 cases:
  - ok generations: 4/4
  - exact sequence: 0/4
  - exact arguments: 0/4
  - valid tool JSON: 0/4
  - unresolved-mask examples: 0/4
  - errors: 0/4
  - generated tokens/sec: 12.9426

Interpretation:

- The sampler and strict eval harness are stable for the 9B candidate.
- The 100-step adapter does not yet recover strict tool-call format.
- Failures are format collapse and missing calls, not unresolved diffusion masks
  or runtime errors.
- The next improvement should be data/objective/decoding focused rather than
  just more mixed steps.

## Synthetic-Only Probe

I tried a focused synthetic one-call warmup because the mixed run was too
diffuse. The first attempt exposed a preprocessing bug rather than a model-size
limit:

- direct dataset:
  `data/synthetic_onecall_train/train_synthetic_onecall.json`
- rebuilt curriculum format:
  `data/qwen35_9b_synthetic_onecall_curriculum/train_agentic_mix.json`
- block size: 704, chosen because all synthetic one-call examples are `<=696`
  tokens under `fast_dllm_v2` and 704 is divisible by the 32-token diffusion
  block.

Raw label checks before the fix:

- first 16 rebuilt synthetic examples all have assistant labels inside the
  704-token window
- valid assistant label counts range from 27 to 53 in those first 16 examples
- a direct one-batch model forward on the first synthetic example gives nonzero
  loss: `5.870865345001221`

Failed pre-fix training results:

- `runs/fastdllm_qwen35_9b_synthetic_onecall_b704_step100`
  - 100 steps
  - train loss: `0.0`
  - grad norm: `0.0`
- `runs/fastdllm_qwen35_9b_synthetic_onecall_curriculum_b704_step5`
  - 5 steps
  - train loss: `0.0`
  - grad norm: `0.0`

Root cause and fix:

- Hugging Face JSON loading expands the nested `tools` schema across all tool
  variants and fills missing properties with `None`.
- This bloated the rendered `fast_dllm_v2` prompts from roughly 530-620 tokens
  after pruning to roughly 900-1000 tokens before pruning.
- With right truncation at 704, the assistant/tool-call labels were truncated
  away before Trainer forwarded the batch.
- `fast-dllm/third_party/lmflow/tokenization/hf_decoder_model.py` now
  recursively drops `None` fields from `messages` and `tools` before calling
  `tokenizer.apply_chat_template`.
- Repro patch:
  `patches/fast-dllm-lmflow-drop-none-tools.patch`
- A one-step Trainer debug run now confirms valid labels survive:
  - run:
    `runs/fastdllm_qwen35_9b_synthetic_onecall_b704_debug_pruned_step1`
  - pre-MDM: `input_shape=(1, 704) valid_labels=[51]`
  - post-MDM: `input_shape=(2, 1408) valid_labels=[21, 30]`
  - train loss: `5.339972972869873`

Corrected 192-example pilot:

- run: `runs/fastdllm_qwen35_9b_synthetic_onecall_b704_pruned_step100`
- dataset: `data/qwen35_9b_synthetic_onecall_curriculum`
- QLoRA: r=8, alpha=16, target modules include attention and GDN projections
- steps: 100
- block window: 704 tokens, right truncation after pruning
- train loss: `5.1511626625061036`
- runtime: `172.2698s`, `0.58 steps/s`
- late logged losses mostly `3.8-4.5`

Cached sampler eval on the 100-step adapter:

- synthetic one-call holdout, 8 cases: `0/8` valid JSON, `0/8` exact sequence
- public one-call, 8 cases: `0/8` valid JSON, `0/8` exact sequence
- public multi-call, 4 cases: `0/4` valid JSON, `0/4` exact sequence
- synthetic tool-result, 4 cases: `0/4` valid JSON, `0/4` exact sequence
- no unresolved mask examples and no runtime errors

This cached eval was not a valid measure of the bridge. The current Qwen3.5
bridge does not implement layer KV cache, but the upstream Fast-dLLM sampler
assumes cache works and forwards only the current 32-token block after prompt
prefill. That loses most prompt context during generation.

Full-context sampler diagnostic:

- `scripts/eval_fastdllm_toolcall_cases.py` now has `--full-context-sampling`.
- It is slower, but it forwards the whole prompt plus current denoising block at
  every step and does not depend on unimplemented KV cache.
- 192-example, 100-step adapter:
  - holdout synthetic one-call: `0/8` valid JSON, `0/8` exact sequence
  - train-slice synthetic one-call: `2/8` valid JSON, `1/8` exact sequence,
    `0/8` exact arguments

Tiny overfit diagnostic:

- dataset: `data/qwen35_9b_synthetic_onecall_tiny8_curriculum`
- run: `runs/fastdllm_qwen35_9b_synthetic_onecall_tiny8_b704_pruned_step200`
- QLoRA: r=16, alpha=32, LR `5e-5`
- steps: 200 on 8 examples, 25 epochs
- train loss: `1.537280468940735`
- late logged losses reached `0.458-1.0928`
- cached sampler eval on the same 8 prompts: still `0/8`
- full-context sampler eval on the same 8 prompts:
  - valid JSON: `3/8`
  - exact tool sequence: `2/8`
  - exact arguments: `0/8`
  - unresolved-mask examples: `0/8`

Format-first curriculum:

- builder: `scripts/build_toolcall_format_curriculum.py`
- dataset: `data/qwen35_9b_toolcall_format_curriculum`
- eval slices:
  - `data/toolcall_eval/toolcall_format_train_smoke.jsonl`
  - `data/toolcall_eval/toolcall_format_heldout_smoke.jsonl`
- construction:
  - 32 base synthetic one-call training examples
  - 3 variants each: original prompt, single-tool prompt, explicit
    tool-name/arguments-to-JSON formatting prompt
  - 96 training instances total
- rendered length/label check under LMFlow + `fast_dllm_v2`:
  - length min/median/max: `239 / 339 / 599`
  - assistant labels min/median/max: `27 / 33 / 53`
  - zero-label rows: `0`
- one-step gate:
  - run: `runs/fastdllm_qwen35_9b_toolcall_format_b704_debug_step1`
  - pre-MDM: `valid_labels=[27]`
  - post-MDM: `valid_labels=[12, 15]`
  - loss: `5.973178386688232`
- 300-step pilot:
  - run: `runs/fastdllm_qwen35_9b_toolcall_format_b704_step300`
  - QLoRA: r=16, alpha=32, LR `5e-5`
  - block window: 704 tokens
  - train loss: `1.2748866283893585`
  - runtime: `521.1449s`
  - throughput: `0.576 steps/s`
  - late logged losses reached `0.2826-0.5508`

`scripts/eval_fastdllm_toolcall_cases.py` now also supports
`--repair-mode schema`. This keeps strict metrics unchanged and adds separate
`repaired_*` metrics using only generated text plus available tool schemas. It
is a diagnostic for punctuation/key corruption, not a substitute for strict
model quality.

Full-context eval results for the 300-step format adapter:

- format train slice, 16 cases:
  - strict valid JSON: `6/16`
  - strict exact sequence: `5/16`
  - strict exact arguments: `2/16`
  - repaired exact arguments: `14/16`
  - unresolved-mask examples: `0/16`
- format heldout-original slice, 16 cases:
  - strict valid JSON: `7/16`
  - strict exact sequence: `6/16`
  - strict exact arguments: `1/16`
  - repaired exact arguments: `8/16`
  - unresolved-mask examples: `0/16`
- shared synthetic one-call smoke, 16 cases:
  - strict valid JSON: `8/16`
  - strict exact sequence: `6/16`
  - strict exact arguments: `4/16`
  - repaired exact arguments: `13/16`
  - unresolved-mask examples: `0/16`
- public Hermes one-call, 8 cases:
  - strict valid JSON: `1/8`
  - strict exact sequence: `0/8`
  - strict exact arguments: `0/8`
  - repaired exact sequence: `7/8`
  - repaired exact arguments: `0/8`
  - unresolved-mask examples: `0/8`

Interpretation:

- The Trainer path and MDM loss now work for synthetic tool-call data.
- The current checkpoint is an early trained diffusion adapter, not an agentic
  model yet.
- The format-first run is the first Qwen3.5-9B diffusion checkpoint in this
  project with nonzero strict exact arguments on held-out synthetic tool-call
  prompts.
- Most remaining synthetic failures are JSON punctuation/name/key corruption:
  repair can recover many exact arguments, so constrained decoding is likely
  high leverage.
- Public Hermes transfer is still weak. The next data stage should mix this
  format curriculum with public/teacher-distilled Qwen3.6 labels rather than
  merely repeating synthetic format training.
- The next engineering blocker is sampling: either implement proper KV cache
  and diffusion attention for the Qwen3.5 bridge, or keep the full-context
  sampler as the correctness path while training/evaluating small curricula.
- The next training blocker is broad schema/argument generalization, especially
  public one-call and then multi-call.

Public-mix curriculum pilot:

- builder: `scripts/build_toolcall_format_public_mix.py`
- dataset: `data/qwen35_9b_toolcall_format_public_curriculum`
- construction:
  - 92 compact format-curriculum instances
  - 40 public one-call train instances
  - 10 exact public one-call Qwen3.6 teacher instances after dedupe
  - 142 training instances total
  - teacher-train eval slice:
    `data/toolcall_eval/public_onecall_teacher_train_smoke.jsonl`
  - teacher-heldout eval slice:
    `data/toolcall_eval/public_onecall_teacher_heldout_smoke.jsonl`
- rendered length/label check under LMFlow + `fast_dllm_v2`:
  - full length min/p50/p90/max: `239 / 457 / 882 / 1172`
  - assistant labels min/p50/max: `24 / 38 / 315`
  - zero-label rows before truncation: `0`
  - right truncation label preservation:
    - block 704: `23` zero-label rows
    - block 896: `6` zero-label rows
    - block 1024: `3` zero-label rows
    - block 1152: `0` zero-label rows, `1` partial row
    - block 1280: `0` zero-label rows
- one-step 896-token gate:
  - run:
    `runs/fastdllm_qwen35_9b_toolcall_format_public_b896_debug_step1`
  - pre-MDM: `valid_labels=[12]`
  - post-MDM: `valid_labels=[7, 5]`
  - loss: `7.1235`
- 300-step pilot:
  - run: `runs/fastdllm_qwen35_9b_toolcall_format_public_b896_step300`
  - QLoRA: r=16, alpha=32, LR `3e-5`
  - block window: 896 tokens
  - train loss: `1.8588300728797913`
  - runtime: `646.6325s`
  - throughput: `0.464 steps/s`
  - peak CUDA allocated/reserved during eval: `17.89 / 27.98 GiB`

Full-context eval results for the 300-step public-mix adapter:

- Qwen3.6 teacher train slice, 12 cases:
  - strict valid JSON: `1/12`
  - strict exact sequence: `0/12`
  - strict exact arguments: `0/12`
  - repaired exact sequence: `8/12`
  - repaired exact arguments: `1/12`
  - unresolved-mask examples: `0/12`
- Qwen3.6 teacher heldout slice, 8 cases:
  - strict valid JSON: `1/8`
  - strict exact sequence: `0/8`
  - strict exact arguments: `0/8`
  - repaired exact sequence: `6/8`
  - repaired exact arguments: `0/8`
  - unresolved-mask examples: `0/8`
- public Hermes one-call, 8 cases:
  - strict valid JSON: `1/8`
  - strict exact sequence: `0/8`
  - strict exact arguments: `0/8`
  - repaired exact sequence: `7/8`
  - repaired exact arguments: `1/8`
  - unresolved-mask examples: `0/8`
- shared synthetic one-call smoke, 16 cases:
  - strict valid JSON: `4/16`
  - strict exact sequence: `3/16`
  - strict exact arguments: `3/16`
  - repaired exact sequence: `16/16`
  - repaired exact arguments: `14/16`
  - unresolved-mask examples: `0/16`

Public-mix interpretation:

- The 5090 can run this 9B QLoRA diffusion pilot at 896 tokens without CPU
  offload or touching the 5080.
- The mixed public/teacher data improves repaired public tool choice and
  sequence recovery, including heldout teacher cases.
- Strict public metrics remain at zero exact sequence/arguments, so the model
  is not yet usable as an agentic tool caller.
- Synthetic strict metrics regressed versus the 704-token format-only run,
  likely because public examples are longer, the train mix is small, and some
  rows still lose labels at 896 tokens.
- Argument fidelity is now the clearest training/eval bottleneck. The next
  useful change is label-aware packing plus either constrained structural
  decoding or a data objective that overweights argument spans.

Label-aware public-mix pilot:

- builder: `scripts/build_toolcall_labelaware_public_mix.py`
- dataset: `data/qwen35_9b_toolcall_labelaware_public_curriculum`
- construction:
  - starts from the same format, public one-call, and Qwen3.6 teacher exact
    one-call sources as the previous public-mix builder
  - renders candidates with the real Qwen3.5 Fast-dLLM tokenizer and
    `fast_dllm_v2` template
  - accepts only rows whose assistant labels fully survive the configured
    896-token right-truncated training window
  - falls back from full tool schemas to gold-tool schemas only when needed
    for label preservation
- manifest:
  - raw candidates: `148`
  - accepted before dedupe: `147`
  - final deduped instances: `141`
  - rejected rows: `1`
  - source families:
    - format curriculum: `92`
    - public one-call: `39`
    - Qwen3.6 teacher exact one-call: `10`
  - chosen variants:
    - `format_curriculum:full_tools`: `92`
    - `public_train_onecall:full_tools`: `28`
    - `public_train_onecall:gold_tools`: `11`
    - `public_teacher_exact_onecall:full_tools`: `8`
    - `public_teacher_exact_onecall:gold_tools`: `2`
- rendered label audit for accepted rows:
  - length min/p50/p90/max: `239 / 450 / 779 / 890`
  - kept assistant labels min/p50/p90/max: `24 / 38 / 74 / 315`
  - accepted zero-label rows after truncation: `0`
  - accepted partial-label rows after truncation: `0`
  - rejected/variant candidates had `9` zero-label and `7` partial-label
    failures before label-aware selection
- one-step 896-token gate:
  - run:
    `runs/fastdllm_qwen35_9b_toolcall_labelaware_public_b896_debug_step1`
  - pre-MDM: `valid_labels=[33]`
  - post-MDM: `valid_labels=[15, 18]`
  - loss: `5.563186168670654`
- 300-step pilot:
  - run: `runs/fastdllm_qwen35_9b_toolcall_labelaware_public_b896_step300`
  - QLoRA: r=16, alpha=32, LR `3e-5`
  - block window: 896 tokens
  - train loss: `1.860784117380778`
  - runtime: `646.7168s`
  - throughput: `0.464 steps/s`
  - peak CUDA allocated/reserved during eval: `17.89 / 27.05 GiB`

Full-context eval results for the 300-step label-aware public-mix adapter:

- Qwen3.6 teacher train slice, 12 cases:
  - strict valid JSON: `1/12`
  - strict exact sequence: `1/12`
  - strict exact arguments: `0/12`
  - repaired exact sequence: `8/12`
  - repaired exact arguments: `1/12`
  - unresolved-mask examples: `0/12`
- Qwen3.6 teacher heldout slice, 8 cases:
  - strict valid JSON: `0/8`
  - strict exact sequence: `0/8`
  - strict exact arguments: `0/8`
  - repaired exact sequence: `6/8`
  - repaired exact arguments: `0/8`
  - unresolved-mask examples: `0/8`
- public Hermes one-call, 8 cases:
  - strict valid JSON: `1/8`
  - strict exact sequence: `1/8`
  - strict exact arguments: `0/8`
  - repaired exact sequence: `7/8`
  - repaired exact arguments: `0/8`
  - unresolved-mask examples: `0/8`
- shared synthetic one-call smoke, 16 cases:
  - strict valid JSON: `1/16`
  - strict exact sequence: `1/16`
  - strict exact arguments: `0/16`
  - repaired exact sequence: `16/16`
  - repaired exact arguments: `16/16`
  - unresolved-mask examples: `0/16`

Label-aware interpretation:

- This is the first Qwen3.5-9B diffusion checkpoint in the project with a
  nonzero strict public one-call exact sequence result.
- Full label preservation helped raw tool sequence selection, but did not solve
  argument exactness.
- Synthetic strict performance regressed relative to the format-only run and
  previous public-mix run, even though schema repair can recover the complete
  synthetic sequence and arguments. The model is generating enough semantic
  signal for repair, but not enough strict structure.
- The next data step should add argument-focused supervision rather than only
  more public full-context rows: for example explicit argument-copy variants,
  higher sampling weight for argument-rich rows, and/or structural-token loss
  weighting if we modify the trainer.
- The next decoding step should turn schema repair into constrained
  `<tool_call>` decoding so punctuation and key placement do not erase otherwise
  correct tool decisions.

Argument-focused curriculum pilot:

- builder: `scripts/build_toolcall_argument_curriculum.py`
- dataset: `data/qwen35_9b_toolcall_argument_curriculum`
- construction:
  - starts from the same label-aware original candidates as the previous run
  - keeps the original label-aware rows
  - adds public/Qwen3.6 teacher one-call argument variants:
    - exact function-call copy
    - request plus selected function plus arguments JSON
    - argument key-value reconstruction
  - all accepted rows are re-audited through the real tokenizer and
    `fast_dllm_v2` template at the 896-token right-truncated window
- manifest:
  - final deduped instances: `280`
  - label-aware originals: `147`
  - argument candidates: `153`
  - accepted argument variants: `141`
  - rejected rows: `13`
  - source families:
    - format curriculum: `92`
    - public one-call: `150`
    - Qwen3.6 teacher exact one-call: `38`
- rendered label audit for accepted rows:
  - length min/p50/p90/max: `239 / 459 / 750 / 890`
  - kept assistant labels min/p50/p90/max: `24 / 46 / 85 / 315`
  - accepted zero-label rows after truncation: `0`
  - accepted partial-label rows after truncation: `0`
- one-step 896-token gate:
  - run: `runs/fastdllm_qwen35_9b_toolcall_argument_b896_debug_step1`
  - pre-MDM: `valid_labels=[46]`
  - post-MDM: `valid_labels=[37, 9]`
  - loss: `7.094879627227783`
- 300-step pilot:
  - run: `runs/fastdllm_qwen35_9b_toolcall_argument_b896_step300`
  - QLoRA: r=16, alpha=32, LR `3e-5`
  - block window: 896 tokens
  - train loss: `1.872867987950643`
  - runtime: `646.5903s`
  - throughput: `0.464 steps/s`
  - peak CUDA allocated/reserved during eval: `17.89 / 28.64 GiB`

Full-context eval results for the 300-step argument adapter:

- Qwen3.6 teacher train slice, 12 cases:
  - strict valid JSON: `1/12`
  - strict exact sequence: `1/12`
  - strict exact arguments: `0/12`
  - repaired exact sequence: `10/12`
  - repaired exact arguments: `0/12`
  - unresolved-mask examples: `0/12`
- Qwen3.6 teacher heldout slice, 8 cases:
  - strict valid JSON: `1/8`
  - strict exact sequence: `1/8`
  - strict exact arguments: `0/8`
  - repaired exact sequence: `5/8`
  - repaired exact arguments: `0/8`
  - unresolved-mask examples: `0/8`
- public Hermes one-call, 8 cases:
  - strict valid JSON: `0/8`
  - strict exact sequence: `0/8`
  - strict exact arguments: `0/8`
  - repaired exact sequence: `6/8`
  - repaired exact arguments: `0/8`
  - unresolved-mask examples: `0/8`
- shared synthetic one-call smoke, 16 cases:
  - strict valid JSON: `1/16`
  - strict exact sequence: `2/16`
  - strict exact arguments: `1/16`
  - repaired exact sequence: `16/16`
  - repaired exact arguments: `15/16`
  - unresolved-mask examples: `0/16`

Argument-curriculum interpretation:

- Explicit argument-copy variants did not transfer to strict public argument
  correctness on original prompts.
- The run improves some teacher sequence metrics:
  teacher-train repaired exact sequence rises to `10/12`, and teacher-heldout
  strict exact sequence reaches `1/8`.
- Public original prompts regress from the label-aware checkpoint:
  strict exact sequence falls from `1/8` to `0/8`, repaired exact sequence
  falls from `7/8` to `6/8`, and strict/repaired exact arguments remain `0/8`.
- Sampled failures show right or near-right tool names, but corrupted argument
  JSON: missing colons, duplicated substrings, malformed nested arrays, dropped
  `arguments` keys, and occasional `<think>` leakage.
- The next useful step is not more direct argument-copy rows alone. It should
  be constrained `<tool_call>` decoding and/or structural-token/argument-span
  loss weighting so the model cannot destroy otherwise-correct names and values
  with invalid syntax.

Constrained tool-call decoding diagnostic:

- eval harness update: `scripts/eval_fastdllm_toolcall_cases.py`
  - adds `--constrained-tool-decoding`
  - keeps strict and `repaired_*` metrics unchanged
  - adds separate `constrained_*` metrics
  - uses only generated text, prompt messages, and available tool schemas
  - does not use gold tool calls or gold arguments
- offline rescorer: `scripts/rescore_fastdllm_toolcall_outputs.py`
  - applies the same repair/constrained scoring to existing eval JSONL files
  - avoids re-running GPU generation when the raw model outputs already exist
- constrained decoder behavior:
  - selects likely tool names from generated output
  - emits canonical `<tool_call>` JSON
  - fills schema properties from generated output first, then prompt context
  - tightens scalar extraction so schema keys inside ordinary values, such as
    `living room`, are not treated as argument keys
  - maps one lost-key string value to the only missing string property when the
    model produced the value but dropped the key

Constrained rescore results on existing 896-token checkpoints:

- label-aware public checkpoint:
  - public one-call, 8 cases:
    - strict exact sequence / args: `1/8` / `0/8`
    - repaired exact sequence / args: `7/8` / `0/8`
    - constrained exact sequence / args: `7/8` / `2/8`
  - Qwen3.6 teacher train, 12 cases:
    - repaired exact sequence / args: `8/12` / `1/12`
    - constrained exact sequence / args: `9/12` / `3/12`
  - Qwen3.6 teacher heldout, 8 cases:
    - repaired exact sequence / args: `6/8` / `0/8`
    - constrained exact sequence / args: `6/8` / `1/8`
  - synthetic one-call, 16 cases:
    - repaired exact sequence / args: `16/16` / `16/16`
    - constrained exact sequence / args: `16/16` / `15/16`
- argument-focused checkpoint:
  - public one-call, 8 cases:
    - repaired exact sequence / args: `6/8` / `0/8`
    - constrained exact sequence / args: `6/8` / `1/8`
  - Qwen3.6 teacher train, 12 cases:
    - repaired exact sequence / args: `10/12` / `0/12`
    - constrained exact sequence / args: `10/12` / `1/12`
  - Qwen3.6 teacher heldout, 8 cases:
    - repaired exact sequence / args: `5/8` / `0/8`
    - constrained exact sequence / args: `6/8` / `1/8`
  - synthetic one-call, 16 cases:
    - repaired exact sequence / args: `16/16` / `15/16`
    - constrained exact sequence / args: `16/16` / `14/16`
- public-mix checkpoint:
  - public one-call, 8 cases:
    - repaired exact sequence / args: `7/8` / `1/8`
    - constrained exact sequence / args: `7/8` / `2/8`
  - synthetic one-call, 16 cases:
    - repaired exact sequence / args: `16/16` / `14/16`
    - constrained exact sequence / args: `16/16` / `15/16`

Constrained-decoding interpretation:

- A schema-aware wrapper can turn the current diffusion checkpoints into a
  minimally usable one-call tool selector on these small slices: public
  constrained exact sequence reaches `6/8-7/8`.
- Public exact arguments become nonzero under constrained decoding (`1/8-2/8`),
  which confirms the model output contains some recoverable values.
- Exact argument quality is still too low for the project goal. The model is
  not yet reliably preserving values, nested structures, or JSON boundaries.
- The best next training change is structural-token/argument-span weighting or
  a denoising objective that directly rewards stable JSON keys, colons, quotes,
  arrays, and copied argument spans.
- The best next inference change is to move from post-hoc constrained repair to
  generation-time constraints for `<tool_call>`, `name`, `arguments`, property
  keys, and scalar/array delimiters.

## Structural Token Weighting Pilot

Date: 2026-06-26 18:22 PDT.

Change:

- `models/qwen3.5-9b-fastdllm-init/modeling.py` now has an opt-in weighted
  CausalLM loss path controlled by:
  - `FASTDLLM_STRUCTURAL_LOSS_WEIGHT`
  - `FASTDLLM_STRUCTURAL_TOKEN_IDS`
  - optional debug: `FASTDLLM_DEBUG_STRUCTURAL_LOSS`
- `scripts/fastdllm_structural_token_ids.py` derives structural token IDs from
  the candidate tokenizer instead of hardcoding Qwen IDs.
- `scripts/run_fastdllm_qwen35_9b_agentic_qlora_pilot.sh` computes and exports
  those token IDs only when `STRUCTURAL_LOSS_WEIGHT` is not `1.0`.

The 5080/Alienware host was reserved for user work during this pass, so all
training and eval below ran only on the local RTX 5090 inside bounded user
systemd scopes.

One-step gate:

- run: `runs/fastdllm_qwen35_9b_toolcall_labelaware_structw_b896_step1_gate`
- dataset: `data/qwen35_9b_toolcall_labelaware_public_curriculum`
- block size: `896`
- structural weight: `2.0`
- debug label coverage:
  - pre-MDM valid labels: `[27]`
  - post-MDM valid labels: `[8, 19]`
  - weighted structural labels: `9 / 27`
- train loss: `8.206977844238281`

300-step comparable run:

- run: `runs/fastdllm_qwen35_9b_toolcall_labelaware_structw2_b896_step300`
- dataset: `data/qwen35_9b_toolcall_labelaware_public_curriculum`
- samples: `141`
- block size: `896`
- max steps: `300`
- LR: `3e-5`
- LoRA targets:
  `q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj`
- structural weight: `2.0`
- token manifest:
  `runs/fastdllm_qwen35_9b_toolcall_labelaware_structw2_b896_step300/structural_token_ids.json`
- train loss: `2.6473990964889524`
- runtime: `646.309s`
- throughput: `0.464` steps/s

Comparable full-context eval at `max_new_tokens=96`:

- public one-call, 8 cases:
  - strict exact sequence / args: `0/8` / `0/8`
  - repaired exact sequence / args: `7/8` / `0/8`
  - constrained exact sequence / args: `7/8` / `1/8`
  - prior label-aware constrained result: `7/8` / `2/8`
- Qwen3.6 teacher train, 12 cases:
  - strict exact sequence / args: `1/12` / `1/12`
  - repaired exact sequence / args: `9/12` / `1/12`
  - constrained exact sequence / args: `9/12` / `2/12`
  - prior label-aware constrained result: `9/12` / `3/12`
- Qwen3.6 teacher heldout, 8 cases:
  - strict exact sequence / args: `0/8` / `0/8`
  - repaired exact sequence / args: `7/8` / `0/8`
  - constrained exact sequence / args: `8/8` / `1/8`
  - prior label-aware constrained result: `6/8` / `1/8`

Interpretation:

- Structural weighting improved heldout constrained tool selection
  (`6/8 -> 8/8`) but did not improve exact argument recovery.
- Public and teacher-train argument exactness regressed by one case under the
  comparable `96`-token eval cap.
- Raw strict JSON remains too weak: public and heldout strict exact sequence are
  still `0/8`.
- Do not scale this naive token-ID weighting as the next main recipe. Keep the
  hook for sweeps, but the next training target should weight/copy complete
  argument spans, add generation-time structure constraints, or train directly
  on constrained-decoder-compatible intermediate forms.

## Argument Span Weighting Pilot

Date: 2026-06-26 18:43 PDT.

Change:

- `models/qwen3.5-9b-fastdllm-init/modeling.py` now supports an opt-in
  per-label argument-span loss mask controlled by:
  - `FASTDLLM_ARGUMENT_SPAN_LOSS_WEIGHT`
  - `FASTDLLM_ARGUMENT_SPAN_START_TOKEN_IDS`
  - `FASTDLLM_ARGUMENT_SPAN_END_TOKEN_IDS`
  - optional debug: `FASTDLLM_DEBUG_ARGUMENT_SPAN_LOSS`
- `scripts/fastdllm_argument_span_token_ids.py` derives the start/end IDs from
  the candidate tokenizer:
  - start fragment: `arguments`
  - start token IDs: `[15889]`
  - end fragment: `</tool_call>`
  - end token IDs: `[248059]`
- `scripts/run_fastdllm_qwen35_9b_agentic_qlora_pilot.sh` computes and exports
  those IDs only when `ARGUMENT_SPAN_LOSS_WEIGHT` is not `1.0`.

Implementation note:

- The span mask is built from the original assistant labels before MDM masking.
- The same mask is then carried through the primary and complementary MDM label
  branches, so selected argument tokens keep their extra weight and ignored
  labels stay neutral.
- This is deliberately different from the earlier structural-token probe:
  it weights full argument payload regions, not just punctuation/key tokens.

One-step gate:

- run: `runs/fastdllm_qwen35_9b_toolcall_labelaware_argspanw2_b896_step1_gate`
- dataset: `data/qwen35_9b_toolcall_labelaware_public_curriculum`
- block size: `896`
- argument-span weight: `2.0`
- debug label coverage:
  - pre-MDM valid labels: `[27]`
  - argument-span labels: `15 / 27`
  - post-MDM valid labels: `[8, 19]`
  - weighted labels in shifted loss path: `15 / 27`
- train loss: `9.805089950561523`

300-step comparable run:

- run: `runs/fastdllm_qwen35_9b_toolcall_labelaware_argspanw2_b896_step300`
- dataset: `data/qwen35_9b_toolcall_labelaware_public_curriculum`
- samples: `141`
- block size: `896`
- max steps: `300`
- LR: `3e-5`
- LoRA targets:
  `q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj`
- argument-span weight: `2.0`
- token manifest:
  `runs/fastdllm_qwen35_9b_toolcall_labelaware_argspanw2_b896_step300/argument_span_token_ids.json`
- train loss: `3.9026308631896973`
- runtime: `646.444s`
- throughput: `0.464` steps/s

Comparable full-context eval at `max_new_tokens=96`:

- public one-call, 8 cases:
  - strict exact sequence / args: `2/8` / `2/8`
  - repaired exact sequence / args: `5/8` / `0/8`
  - constrained exact sequence / args: `6/8` / `3/8`
  - prior label-aware constrained result: `7/8` / `2/8`
  - prior structural-token constrained result: `7/8` / `1/8`
- Qwen3.6 teacher train, 12 cases:
  - strict exact sequence / args: `2/12` / `2/12`
  - repaired exact sequence / args: `7/12` / `0/12`
  - constrained exact sequence / args: `8/12` / `3/12`
  - prior label-aware constrained result: `9/12` / `3/12`
  - prior structural-token constrained result: `9/12` / `2/12`
- Qwen3.6 teacher heldout, 8 cases:
  - strict exact sequence / args: `0/8` / `0/8`
  - repaired exact sequence / args: `6/8` / `1/8`
  - constrained exact sequence / args: `6/8` / `1/8`
  - prior label-aware constrained result: `6/8` / `1/8`
  - prior structural-token constrained result: `8/8` / `1/8`

Interpretation:

- Argument-span weighting is the first 9B diffusion QLoRA run in this sequence
  to move public raw strict exact arguments above zero: `2/8`.
- It also improves public constrained exact arguments to `3/8`, beating both the
  prior label-aware and structural-token probes on that metric.
- The cost is lower constrained sequence coverage on public/train and no
  heldout constrained-sequence gain versus the original label-aware checkpoint.
- This is a better trainer-side direction than naive structural-token weighting,
  but it should be swept rather than accepted as-is. Next sweep candidates:
  argument-span weight `1.5`, `2.0`, and `3.0`; structural plus argument
  `max()` weighting; and a dataset mix with higher sampling weight for
  argument-rich public/teacher rows.
- The model still needs generation-time constraints or a stricter intermediate
  representation: even with argument-span weighting, public raw strict sequence
  is only `2/8` and heldout raw strict remains `0/8`.

## Combined Structural Plus Argument Span Weighting Pilot

Date: 2026-06-26 19:12 PDT.

This run tested the explicit next-sweep candidate of applying both
structural-token and argument-span weighting with max-combined per-token weights.
The intent was to preserve the public raw argument gains from argument-span
weighting while recovering the constrained sequence coverage seen in the
structural-token run.

One-step/debug behavior was verified through the first training batch:

- pre-MDM valid labels: `[33]`
- argument-span labels: `19 / 33`
- structural-token labels: `9 / 33`
- max-combined weighted labels in shifted loss path: `24 / 33`

300-step comparable run:

- run:
  `runs/fastdllm_qwen35_9b_toolcall_labelaware_structw2_argspanw2_b896_step300`
- dataset: `data/qwen35_9b_toolcall_labelaware_public_curriculum`
- samples: `141`
- block size: `896`
- max steps: `300`
- LR: `3e-5`
- LoRA targets:
  `q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj`
- structural-token weight: `2.0`
- argument-span weight: `2.0`
- train loss: `4.07765515645345`
- runtime: `646.5271s`
- throughput: `0.464` steps/s

Comparable full-context eval at `max_new_tokens=96`:

- public one-call, 8 cases:
  - strict exact sequence / args: `2/8` / `0/8`
  - repaired exact sequence / args: `4/8` / `0/8`
  - constrained exact sequence / args: `4/8` / `1/8`
  - argument-span-only result: strict `2/8` / `2/8`, constrained `6/8` / `3/8`
  - structural-token-only result: strict `0/8` / `0/8`, constrained
    `7/8` / `1/8`
- Qwen3.6 teacher train, 12 cases:
  - strict exact sequence / args: `1/12` / `0/12`
  - repaired exact sequence / args: `9/12` / `2/12`
  - constrained exact sequence / args: `9/12` / `3/12`
  - argument-span-only result: strict `2/12` / `2/12`, constrained
    `8/12` / `3/12`
  - structural-token-only result: strict `1/12` / `1/12`, constrained
    `9/12` / `2/12`
- Qwen3.6 teacher heldout, 8 cases:
  - strict exact sequence / args: `1/8` / `0/8`
  - repaired exact sequence / args: `5/8` / `1/8`
  - constrained exact sequence / args: `5/8` / `1/8`
  - argument-span-only result: strict `0/8` / `0/8`, constrained `6/8` / `1/8`
  - structural-token-only result: strict `0/8` / `0/8`, constrained
    `8/8` / `1/8`

Interpretation:

- The combined weight-2.0 recipe is a negative result for the current objective.
- It keeps public strict sequence at `2/8`, but loses the argument-span-only
  public strict exact-argument gain (`2/8 -> 0/8`).
- It also drops public constrained exact sequence and exact arguments to
  `4/8` and `1/8`, worse than both single-weight probes on at least one key
  metric.
- Keep argument-span-only as the active trainer-side candidate. The next sweep
  should reduce or rebalance structural pressure instead of combining both at
  full weight:
  - argument-span weight `1.5` and `3.0`
  - structural `1.25-1.5` plus argument-span `2.0`
  - higher sampling weight for argument-rich public/teacher rows
  - generation-time constraints for structure rather than forcing structure
    only through the training loss

## Argument Span Weight 1.5 Sweep

Date: 2026-06-26 19:35 PDT.

This run lowered argument-span loss pressure from `2.0` to `1.5` while leaving
structural-token weighting disabled. The goal was to see whether the public and
teacher-train constrained sequence coverage could recover without losing the
argument-copy signal from the weight-2.0 argument-span run.

First-batch/debug behavior:

- pre-MDM valid labels: `[33]`
- argument-span labels: `19 / 33`
- structural-token labels: `0 / 33`
- post-MDM valid labels: `[5, 28]`
- weighted labels in shifted loss path: `19 / 33`

300-step comparable run:

- run:
  `runs/fastdllm_qwen35_9b_toolcall_labelaware_argspanw1p5_b896_step300`
- dataset: `data/qwen35_9b_toolcall_labelaware_public_curriculum`
- samples: `141`
- block size: `896`
- max steps: `300`
- LR: `3e-5`
- LoRA targets:
  `q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj`
- argument-span weight: `1.5`
- train loss: `3.0483706188201904`
- runtime: `646.3971s`
- throughput: `0.464` steps/s

Comparable full-context eval at `max_new_tokens=96`:

- public one-call, 8 cases:
  - strict exact sequence / args: `2/8` / `1/8`
  - repaired exact sequence / args: `6/8` / `0/8`
  - constrained exact sequence / args: `7/8` / `3/8`
  - argument-span weight-2.0 result: strict `2/8` / `2/8`, constrained
    `6/8` / `3/8`
- Qwen3.6 teacher train, 12 cases:
  - strict exact sequence / args: `2/12` / `1/12`
  - repaired exact sequence / args: `8/12` / `1/12`
  - constrained exact sequence / args: `10/12` / `4/12`
  - argument-span weight-2.0 result: strict `2/12` / `2/12`, constrained
    `8/12` / `3/12`
- Qwen3.6 teacher heldout, 8 cases:
  - strict exact sequence / args: `1/8` / `1/8`
  - repaired exact sequence / args: `5/8` / `0/8`
  - constrained exact sequence / args: `6/8` / `1/8`
  - argument-span weight-2.0 result: strict `0/8` / `0/8`, constrained
    `6/8` / `1/8`

Interpretation:

- Argument-span weight `1.5` is the best balanced 300-step checkpoint so far
  for constrained decoding: public constrained exact args tie the weight-2.0
  result at `3/8`, public constrained exact sequence recovers to `7/8`, and
  teacher-train constrained sequence/args improve to `10/12` / `4/12`.
- The tradeoff is raw public exact arguments: weight `1.5` reaches `1/8`, while
  weight `2.0` reached `2/8`.
- Heldout remains weak. Weight `1.5` is the first argument-span-only sweep point
  with nonzero heldout raw exact sequence/args (`1/8` / `1/8`), but constrained
  heldout exact arguments are still only `1/8`.
- Treat `1.5` as the current balanced recipe for the next longer/data-mix run,
  while keeping `2.0` as the useful pressure point for public raw argument
  exactness. The next trainer-side sweep should test weight `3.0` only as a
  short probe, then move effort to data mix and generation-time constraints.

## Argument Span Weight 3.0 Sweep

Date: 2026-06-26 19:57 PDT.

This was the planned short high-pressure probe after weights `1.5` and `2.0`.
It used the same label-aware public curriculum, block size, LoRA targets, LR,
and 300-step budget as the prior argument-span runs, with structural-token
weighting disabled.

No-grouping one-step gate:

- run:
  `runs/fastdllm_qwen35_9b_toolcall_labelaware_argspanw3_b896_step1_gate`
- pre-MDM valid labels: `[33]`
- argument-span labels: `21 / 33`
- structural-token labels: `0 / 33`
- post-MDM valid labels: `[26, 7]`
- train loss: `12.57717227935791`

300-step comparable run:

- run:
  `runs/fastdllm_qwen35_9b_toolcall_labelaware_argspanw3_b896_step300`
- dataset: `data/qwen35_9b_toolcall_labelaware_public_curriculum`
- samples: `141`
- block size: `896`
- max steps: `300`
- LR: `3e-5`
- LoRA targets:
  `q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj`
- argument-span weight: `3.0`
- first training batch: pre-MDM valid labels `[33]`, argument-span labels
  `19 / 33`, post-MDM valid labels `[5, 28]`
- train loss: `5.558430992762248`
- runtime: `646.4547s`
- throughput: `0.464` steps/s

Comparable full-context eval at `max_new_tokens=96`:

- public one-call, 8 cases:
  - strict exact sequence / args: `1/8` / `0/8`
  - repaired exact sequence / args: `7/8` / `1/8`
  - constrained exact sequence / args: `7/8` / `1/8`
  - weight-1.5 result: strict `2/8` / `1/8`, constrained `7/8` / `3/8`
  - weight-2.0 result: strict `2/8` / `2/8`, constrained `6/8` / `3/8`
- Qwen3.6 teacher train, 12 cases:
  - strict exact sequence / args: `0/12` / `0/12`
  - repaired exact sequence / args: `10/12` / `1/12`
  - constrained exact sequence / args: `10/12` / `1/12`
  - weight-1.5 result: strict `2/12` / `1/12`, constrained `10/12` / `4/12`
  - weight-2.0 result: strict `2/12` / `2/12`, constrained `8/12` / `3/12`
- Qwen3.6 teacher heldout, 8 cases:
  - strict exact sequence / args: `1/8` / `0/8`
  - repaired exact sequence / args: `5/8` / `1/8`
  - constrained exact sequence / args: `5/8` / `1/8`
  - weight-1.5 result: strict `1/8` / `1/8`, constrained `6/8` / `1/8`
  - weight-2.0 result: strict `0/8` / `0/8`, constrained `6/8` / `1/8`

Interpretation:

- Argument-span weight `3.0` is a negative result for the current objective.
  It raises train loss substantially and regresses exact arguments on the
  public and teacher-train slices.
- The only preserved signal is constrained sequence selection on public and
  teacher-train, but exact argument recovery is worse than both weight `1.5`
  and weight `2.0`.
- The weight sweep should stop here. Keep weight `1.5` as the balanced
  longer-run default, keep weight `2.0` as the raw-public-argument comparison
  point, and spend the next iteration on data mix and generation-time
  constraints rather than stronger argument-span loss pressure.

## Model Repair Pass and Repair Curriculum Pilot

Date: 2026-06-26 20:23 PDT.

This is the first step from post-hoc deterministic repair toward a model-in-loop
constrained decoding path. The eval harness now supports:

- `--model-repair-pass`
- `--model-repair-max-new-tokens`

When enabled, the evaluator first generates the normal diffusion draft, then
asks the same checkpoint to rewrite that draft into valid Qwen `<tool_call>`
JSON using the original request and tool schema. These metrics are reported
separately as `model_repair_*`, so raw strict quality and deterministic
repair/constrained projection remain visible.

Added data builder:

- script: `scripts/build_toolcall_model_repair_curriculum.py`
- dataset:
  `data/qwen35_9b_toolcall_model_repair_curriculum`
- input repair drafts: train-slice raw outputs from five earlier 300-step
  checkpoints over `public_onecall_teacher_train_labelaware_smoke.jsonl`
- total rows: `227`
- label-aware originals: `147`
- model-repair rows accepted after 896-token label-retention gate: `80`
- source families:
  - format curriculum: `96`
  - public train one-call: `39`
  - Qwen3.6 teacher exact one-call: `12`
  - model repair: `80`
- accepted rendered length min/p50/p90/max: `239 / 591 / 840 / 890`
- accepted kept assistant labels min/p50/p90/max: `24 / 41 / 78 / 315`
- accepted zero-label and partial-label rows: `0 / 0`

One-step gate:

- run:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step1_gate`
- dataset: `data/qwen35_9b_toolcall_model_repair_curriculum`
- argument-span weight: `1.5`
- pre-MDM valid labels: `[55]`
- argument-span labels: `41 / 55`
- post-MDM valid labels: `[14, 41]`
- train loss: `9.102045059204102`

300-step comparable run:

- run:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300`
- samples: `227`
- block size: `896`
- max steps: `300`
- LR: `3e-5`
- LoRA targets:
  `q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj`
- argument-span weight: `1.5`
- first training batch: pre-MDM valid labels `[29]`, argument-span labels
  `17 / 29`, post-MDM valid labels `[1, 28]`
- train loss: `3.2297114753723144`
- runtime: `646.4761s`
- throughput: `0.464` steps/s

Comparable full-context eval at `max_new_tokens=96`, with deterministic repair,
deterministic constrained projection, and the learned model-repair pass:

- public one-call, 8 cases:
  - raw strict exact sequence / args: `3/8` / `2/8`
  - constrained exact sequence / args: `7/8` / `4/8`
  - learned model-repair exact sequence / args: `4/8` / `2/8`
  - prior argument-span-1.5 result: raw `2/8` / `1/8`, constrained
    `7/8` / `3/8`
- Qwen3.6 teacher train, 12 cases:
  - raw strict exact sequence / args: `2/12` / `2/12`
  - constrained exact sequence / args: `10/12` / `5/12`
  - learned model-repair exact sequence / args: `5/12` / `2/12`
  - prior argument-span-1.5 result: raw `2/12` / `1/12`, constrained
    `10/12` / `4/12`
- Qwen3.6 teacher heldout, 8 cases:
  - raw strict exact sequence / args: `1/8` / `0/8`
  - constrained exact sequence / args: `6/8` / `1/8`
  - learned model-repair exact sequence / args: `3/8` / `1/8`
  - prior argument-span-1.5 result: raw `1/8` / `1/8`, constrained
    `6/8` / `1/8`

Learned repair decode-length check:

- run:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300_eval96_modelrepair160`
- only `--model-repair-max-new-tokens` changed, from `96` to `160`; the first
  diffusion draft still used `--max-new-tokens 96`
- public one-call, 8 cases:
  - learned model-repair valid JSON: `4/8` vs `3/8` at 96
  - learned model-repair exact sequence / args: `2/8` / `1/8` vs `4/8` /
    `2/8` at 96
- Qwen3.6 teacher train, 12 cases:
  - learned model-repair valid JSON: `4/12` vs `3/12` at 96
  - learned model-repair exact sequence / args: `5/12` / `1/12` vs `5/12` /
    `2/12` at 96
- Qwen3.6 teacher heldout, 8 cases:
  - learned model-repair valid JSON: `4/8` vs `3/8` at 96
  - learned model-repair exact sequence / args: `2/8` / `1/8` vs `3/8` /
    `1/8` at 96
- the long schedule-style cases still hit the 160-token cap, so longer repair
  output alone does not solve the value-copy/structure problem

Single-call constrained projection check:

- code change:
  - `scripts/eval_fastdllm_toolcall_cases.py` now supports
    `--constrained-max-calls`
  - `scripts/rescore_fastdllm_toolcall_outputs.py` supports the same flag and
    rewrites fresh constrained/repaired called-name, call, and error fields
- rescore outputs:
  - `public_onecall_8_constrained_max1_rescore.jsonl`
  - `teacher_train_labelaware_12_constrained_max1_rescore.jsonl`
  - `teacher_heldout_labelaware_8_constrained_max1_rescore.jsonl`
- current best checkpoint rescored with `--constrained-max-calls 1`:
  - public one-call constrained exact sequence / args:
    `8/8` / `5/8`, up from `7/8` / `4/8`
  - Qwen3.6 teacher train constrained exact sequence / args:
    `10/12` / `5/12`, unchanged
  - Qwen3.6 teacher heldout constrained exact sequence / args:
    `7/8` / `2/8`, up from `6/8` / `1/8`
- the fixed public/heldout case was the garage-door example: the raw model
  emitted the correct `open_garage_door` call followed by an extra
  `close_garage_door` call; single-call projection kept the first schema-valid
  call and recovered exact arguments
- this is still an inference projection, not true token-level constrained
  denoising, but it is the correct one-call contract for these eval slices and
  should be reported beside raw strict metrics

Argument-diff diagnostic:

- new script:
  `scripts/diagnose_toolcall_argument_errors.py`
- purpose: compare a selected output prefix such as `constrained` against gold
  tool-call arguments by schema path, with categories for missing required
  fields, scalar value mismatches, array/object length/type mismatches,
  extra keys, and missing/extra tool calls
- current best checkpoint with `--constrained-max-calls 1`:
  - public one-call:
    - rows with exact tool sequence but wrong arguments: `3/8`
    - diff counts: `5` scalar value mismatches, `1` missing required field
    - main misses: `set_thermostat_schedule.$.schedule` missing, thermostat id
      drift `thermo123 -> ther123`, ambient-mode string drift, and
      `design_quantum_circuit.$.function_type` drift
  - Qwen3.6 teacher train:
    - rows with exact tool sequence but wrong arguments: `5/12`
    - diff counts: `5` missing required fields, `3` scalar value mismatches,
      `2` missing tool calls
    - main misses: schedule array, QCAAS software-tool flag, invoice client
      data, training-program modules/quizzes, reservation scalar-copy drift
  - Qwen3.6 teacher heldout:
    - rows with exact tool sequence but wrong arguments: `5/8`
    - diff counts: `9` missing required fields, `4` scalar value mismatches,
      `1` missing tool call
    - main misses: debug-circuit required fields, workflow milestones/team
      members/deadlines, sensor temperature/humidity, ambient-mode string drift

Interpretation of the diagnostic:

- After single-call constrained projection, the remaining argument gap is not
  mostly JSON syntax. It is value copying and required-field completion.
- The next data/training target should be hard argument-completion rows:
  partial or malformed arguments plus original request/tool schema -> complete
  exact gold call.
- Use only train-slice failures for training rows; keep heldout/public
  diagnostics as eval-only evidence.
- Avoid broad clean-repair rows that teach easy wrapper/prose cleanup. The
  prior cap-80 clean-repair run showed that can regress the main generator.

Interpretation:

- This is the strongest 300-step checkpoint so far on public raw strict
  one-call sequence and arguments (`3/8` / `2/8`), and it improves public and
  teacher-train constrained exact arguments by one case versus the prior
  argument-span-1.5 checkpoint.
- The learned model-repair pass is now nonzero and useful for tool selection:
  public `4/8`, teacher-train `5/12`, heldout `3/8` exact sequence. It is not
  yet competitive with deterministic constrained projection on exact arguments.
- Heldout raw exact arguments regress to `0/8`, so this is not a final recipe.
  The next iteration should scale and clean repair data, reduce overfitting to
  train-slice drafts, and keep deterministic constrained decoding as the
  inference fallback while learned repair improves.

## Clean Repair Curriculum Negative Probe

Date: 2026-06-26 21:14 PDT.

After the 160-token repair decode check regressed exactness, I added an
optional controlled-corruption path to
`scripts/build_toolcall_model_repair_curriculum.py`. It builds repair examples
from gold tool calls instead of only from previous model drafts.

New builder options:

- `--clean-repair-cap`
- `--clean-repair-repeat`
- `--clean-repair-sources`
- `--clean-repair-variants`
- `--clean-repair-onecall-only`

Clean-repair dataset:

- dataset:
  `data/qwen35_9b_toolcall_model_repair_clean_curriculum`
- command used: same builder as the model-repair curriculum, with
  `--clean-repair-cap 80`
- total rows: `294`
- label-aware originals: `147`
- prior raw-draft repair rows accepted: `80`
- accepted clean-repair rows: `67`
- clean variants:
  `json_only,missing_wrapper,wrong_arguments_key,truncated,prose`
- source families:
  - format curriculum: `96`
  - public train one-call: `39`
  - Qwen3.6 teacher exact one-call: `12`
  - raw model repair: `80`
  - clean model repair: `67`
- accepted rendered length min/p50/p90/max: `239 / 615 / 841 / 890`
- accepted kept assistant labels min/p50/p90/max: `24 / 43 / 79 / 315`
- accepted zero-label and partial-label rows: `0 / 0`

No-group one-step gate:

- run:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_clean_argspanw1p5_b896_nogroup_debug_step1_gate`
- pre-MDM valid labels: `[43]`
- argument-span labels: `27 / 43`
- post-MDM valid labels: `[18, 25]`
- train loss: `6.055093288421631`

300-step clean-repair run:

- run:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_clean_argspanw1p5_b896_step300`
- samples: `294`
- block size: `896`
- max steps: `300`
- LR: `3e-5`
- argument-span weight: `1.5`
- first training batch: pre-MDM valid labels `[61]`, argument-span labels
  `48 / 61`, post-MDM valid labels `[25, 36]`
- train loss: `2.613187549908956`
- runtime: `647.0471s`
- throughput: `0.464` steps/s

Comparable full-context eval at `max_new_tokens=96`:

- public one-call, 8 cases:
  - raw strict exact sequence / args: `1/8` / `0/8`
  - constrained exact sequence / args: `5/8` / `1/8`
  - learned model-repair exact sequence / args: `3/8` / `1/8`
- Qwen3.6 teacher train, 12 cases:
  - raw strict exact sequence / args: `1/12` / `0/12`
  - constrained exact sequence / args: `9/12` / `1/12`
  - learned model-repair exact sequence / args: `2/12` / `2/12`
  - deterministic repaired exact sequence / args: `10/12` / `2/12`
- Qwen3.6 teacher heldout, 8 cases:
  - raw strict exact sequence / args: `1/8` / `0/8`
  - constrained exact sequence / args: `6/8` / `1/8`
  - learned model-repair exact sequence / args: `2/8` / `0/8`

Interpretation:

- This cap-80 clean-repair mix is a negative scaling recipe. It passes the
  label gate and trains stably, but it regresses the public raw strict metric
  from `3/8` / `2/8` to `1/8` / `0/8`.
- It also harms the constrained public metric, falling from `7/8` / `4/8` to
  `5/8` / `1/8`.
- The only small positive is teacher-train deterministic repair exact arguments
  moving from `1/12` to `2/12`, but that does not carry into constrained or
  heldout metrics.
- Do not scale the cap-80 clean-repair mix as the next default. If controlled
  repair rows are used again, make them lower-weight/lower-cap, filter out
  easy wrapper/prose variants, or train them as a separate repair adapter rather
  than mixing them heavily into the main tool-call generator.

## Hard Argument Completion Negative Probe

Date: 2026-06-26 21:56 PDT.

After the argument-diff diagnostic showed that single-call constrained
projection mostly leaves value-copy and required-field completion errors, I
added an opt-in hard argument-completion path to
`scripts/build_toolcall_model_repair_curriculum.py`.

New builder options:

- `--hard-argument-cases`
- `--hard-argument-outputs`
- `--hard-argument-prefix`
- `--hard-argument-cap`
- `--hard-argument-repeat`

Hard-argument dataset:

- dataset:
  `data/qwen35_9b_toolcall_hard_argument_curriculum`
- source of hard rows:
  train-slice constrained outputs from the current best model-repair checkpoint,
  keeping only rows where constrained tool sequence was exact but arguments
  were not exact
- total rows: `239`
- label-aware originals: `147`
- prior raw-draft repair rows accepted: `80`
- hard argument-completion rows accepted: `12`
- hard argument candidates before filtering/dedupe: `15`
- source families:
  - format curriculum: `96`
  - public train one-call: `39`
  - Qwen3.6 teacher exact one-call: `12`
  - raw model repair: `80`
  - hard argument: `12`
- accepted rendered length min/p50/p90/max: `239 / 596 / 841 / 890`
- accepted kept assistant labels min/p50/p90/max: `24 / 41 / 87 / 315`
- accepted zero-label and partial-label rows: `0 / 0`

No-group one-step gate:

- run:
  `runs/fastdllm_qwen35_9b_toolcall_hardarg_argspanw1p5_b896_nogroup_debug_step1_gate`
- pre-MDM valid labels: `[27]`
- argument-span labels: `15 / 27`
- post-MDM valid labels: `[11, 16]`
- train loss: `6.3488993644714355`

300-step hard-argument run:

- run:
  `runs/fastdllm_qwen35_9b_toolcall_hardarg_argspanw1p5_b896_step300`
- samples: `239`
- block size: `896`
- max steps: `300`
- LR: `3e-5`
- argument-span weight: `1.5`
- first training batch: pre-MDM valid labels `[28]`, argument-span labels
  `16 / 28`, post-MDM valid labels `[11, 17]`
- train loss: `2.5478537392616274`
- runtime: `646.9774s`
- throughput: `0.464` steps/s

Comparable full-context eval at `max_new_tokens=96`,
`--constrained-max-calls 1`, and model repair enabled:

- run:
  `runs/fastdllm_qwen35_9b_toolcall_hardarg_argspanw1p5_b896_step300_eval96_modelrepair_max1`
- public one-call, 8 cases:
  - raw strict exact sequence / args: `0/8` / `0/8`
  - constrained exact sequence / args: `7/8` / `1/8`
  - learned model-repair exact sequence / args: `1/8` / `1/8`
- Qwen3.6 teacher train, 12 cases:
  - raw strict exact sequence / args: `1/12` / `0/12`
  - constrained exact sequence / args: `11/12` / `1/12`
  - learned model-repair exact sequence / args: `2/12` / `2/12`
- Qwen3.6 teacher heldout, 8 cases:
  - raw strict exact sequence / args: `0/8` / `0/8`
  - constrained exact sequence / args: `7/8` / `1/8`
  - learned model-repair exact sequence / args: `0/8` / `0/8`

Hard-argument constrained arg-diff diagnostic:

- public one-call:
  - rows with exact tool sequence but wrong arguments: `6/8`
  - diff counts: `5` missing required fields, `4` scalar value mismatches,
    `1` missing tool call
- Qwen3.6 teacher train:
  - rows with exact tool sequence but wrong arguments: `10/12`
  - diff counts: `10` missing required fields, `6` scalar value mismatches,
    `1` missing tool call, `1` missing gold key
- Qwen3.6 teacher heldout:
  - rows with exact tool sequence but wrong arguments: `6/8`
  - diff counts: `8` missing required fields, `6` scalar value mismatches,
    `1` missing tool call, `1` missing gold key

Interpretation:

- This hard-argument mix is also a negative main-generator scaling recipe. It
  trains stably and improves teacher-train constrained tool sequence selection
  to `11/12`, but it collapses public raw strict exactness to `0/8` / `0/8`
  and exact constrained arguments to `1/8`.
- The arg-diff diagnostic regressed versus the prior best
  (`3/8`, `5/12`, `5/8` exact-tool-sequence rows with wrong args), so the new
  rows made the argument-completion problem broader rather than narrower.
- The result suggests that replaying a few train-slice failed drafts is not
  enough to teach robust value copying. It likely overfits the repair framing
  while weakening first-pass generation.
- Keep hard argument-completion rows as a diagnostic and possible separate
  repair-adapter dataset. Do not mix this version heavily into the next main
  generator run.

## Qwen3.5 GDN Architecture Check

Date: 2026-06-26 22:05 PDT.

Standalone memo: `qwen35_gdn_vs_qwen25_research.md`.

Online and local sources agree that Qwen3.5 is not a Qwen2.5-style dense
full-attention transformer.

Rechecked on 2026-06-27 after the GDN concern came up again. The conclusion
still holds: Qwen3.5-9B and Qwen3.6-27B need the Qwen3.5/GDN bridge path, not a
plain Qwen2.5 attention-only conversion path.

Primary-source comparison:

- Qwen3.5-9B Hugging Face model card:
  `https://huggingface.co/Qwen/Qwen3.5-9B`
  - hidden layout:
    `8 x (3 x (Gated DeltaNet -> FFN) -> 1 x (Gated Attention -> FFN))`
  - raw config:
    `https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/config.json`
  - config confirms `model_type: qwen3_5`, `text_config.model_type:
    qwen3_5_text`, `full_attention_interval: 4`, and a 32-layer
    `layer_types` list repeating
    `linear_attention, linear_attention, linear_attention, full_attention`
  - Gated DeltaNet heads: `32` value heads and `16` QK heads, head dim `128`
  - Gated Attention heads: `16` Q heads and `4` KV heads, head dim `256`
  - context length: `262144` native tokens
  - MTP: trained with multi-steps
- Qwen2.5-7B Hugging Face model card:
  `https://huggingface.co/Qwen/Qwen2.5-7B`
  - architecture:
    Transformer with RoPE, SwiGLU, RMSNorm, Attention QKV bias
  - raw config:
    `https://huggingface.co/Qwen/Qwen2.5-7B/raw/main/config.json`
  - config confirms `architectures: Qwen2ForCausalLM`,
    `model_type: qwen2`, and no `layer_types`/linear-attention layout
  - attention: GQA with `28` Q heads and `4` KV heads
  - context length: `131072` tokens
- Qwen3.6-27B Hugging Face model card and config:
  `https://huggingface.co/Qwen/Qwen3.6-27B`
  `https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/config.json`
  - hidden layout:
    `16 x (3 x (Gated DeltaNet -> FFN) -> 1 x (Gated Attention -> FFN))`
  - config confirms a 64-layer `layer_types` list with the same 3:1
    `linear_attention` to `full_attention` pattern
  - Gated DeltaNet heads: `48` value heads and `16` QK heads, head dim `128`
  - Gated Attention heads: `24` Q heads and `4` KV heads, head dim `256`
  - MTP: trained with multi-steps
- Qwen3.6-27B-FP8 raw config:
  `https://huggingface.co/Qwen/Qwen3.6-27B-FP8/raw/main/config.json`
  - same `qwen3_5` / `qwen3_5_text` GDN-hybrid architecture as 27B bf16
  - `quantization_config` uses FP8 `e4m3` with dynamic activation scheme
- Qwen3.6-27B vLLM recipe:
  `https://recipes.vllm.ai/Qwen/Qwen3.6-27B`
  - dense 27B model with gated-delta-networks hybrid attention
  - native `262144` context
  - FP8 serving listed for a single 40 GB GPU and int4 serving listed for a
    single 24 GB GPU
  - MTP speculative decoding supported for low-latency serving

Local config check for `models/qwen3.5-9b-fastdllm-init`:

- `gdn_mode: option_a_causal_gdn_v0`
- `layer_types`: repeating
  `linear_attention, linear_attention, linear_attention, full_attention`
- `linear_conv_kernel_dim: 4`
- `linear_num_key_heads: 16`
- `linear_num_value_heads: 32`
- `linear_key_head_dim: 128`
- `linear_value_head_dim: 128`
- `rope_parameters`: `mrope_interleaved: true`, `partial_rotary_factor: 0.25`,
  `rope_theta: 10000000`

Implications for this experiment:

- Do not assume Qwen2.5 conversion logic is complete for Qwen3.5/Qwen3.6.
  Qwen3.5 has recurrent/linear-attention state paths in most layers, not only
  standard KV-cache full attention.
- The current Fast-DLLM bridge includes a
  `Fast_dLLM_Qwen3_5GatedDeltaNet` implementation and the current LoRA target
  list includes GDN projections:
  `in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj`, in addition to
  `q_proj,k_proj,v_proj,o_proj`.
- Keep full-context sampling as the correctness path until GDN recurrent-state
  caching is explicitly implemented and tested for block diffusion. A Qwen2.5
  KV-cache assumption is not enough.
- For Qwen3.6 teacher serving, keep using engines with explicit Qwen3.5/3.6
  support such as SGLang or recent vLLM, and use MTP/speculative decoding where
  supported. For the original 27B teacher target, the vLLM recipe confirms this
  is also a dense GDN-hybrid model, not a Qwen2.5-style transformer.
- For the student, GDN is not a reason to stop the current Qwen3.5-9B run, but
  it is a reason to treat Qwen2.5-based diffusion papers/code as architecture
  guidance rather than drop-in implementation guidance.
- Practically, keep Qwen2.5-1.5B as the cheap diffusion-objective lab only.
  Promotion runs should stay on Qwen3.5-9B because it exercises the same 3:1
  GDN/full-attention pattern as Qwen3.6-27B.

## Same-Recipe 600-Step Negative Probe

Date: 2026-06-26 22:33 PDT.

After the hard-argument mix regressed, I ran a conservative longer version of
the current best recipe instead of changing the corpus again.

Training run:

- run:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step600`
- dataset:
  `data/qwen35_9b_toolcall_model_repair_curriculum`
- samples: `227`
- block size: `896`
- max steps: `600`
- LR: `3e-5`
- argument-span weight: `1.5`
- LoRA targets:
  `q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj`
- train loss: `1.8730474283297858`
- runtime: `1293.9047s`
- throughput: `0.464` steps/s

Comparable full-context eval at `max_new_tokens=96`,
`--constrained-max-calls 1`, and model repair enabled:

- run:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step600_eval96_modelrepair_max1`
- public one-call, 8 cases:
  - raw strict exact sequence / args: `0/8` / `0/8`
  - constrained exact sequence / args: `7/8` / `3/8`
  - learned model-repair exact sequence / args: `2/8` / `0/8`
  - prior best constrained max-1 result: `8/8` / `5/8`
- Qwen3.6 teacher train, 12 cases:
  - raw strict exact sequence / args: `0/12` / `0/12`
  - constrained exact sequence / args: `12/12` / `5/12`
  - learned model-repair exact sequence / args: `1/12` / `0/12`
  - prior best constrained max-1 result: `10/12` / `5/12`
- Qwen3.6 teacher heldout, 8 cases:
  - raw strict exact sequence / args: `0/8` / `0/8`
  - constrained exact sequence / args: `7/8` / `0/8`
  - learned model-repair exact sequence / args: `2/8` / `0/8`
  - prior best constrained max-1 result: `7/8` / `2/8`

Step-600 constrained arg-diff diagnostic:

- public one-call:
  - rows with exact tool sequence but wrong arguments: `4/8`
  - diff counts: `3` missing required fields, `3` scalar value mismatches,
    `1` missing tool call
- Qwen3.6 teacher train:
  - rows with exact tool sequence but wrong arguments: `7/12`
  - diff counts: `7` missing required fields, `5` scalar value mismatches,
    `2` missing gold keys
- Qwen3.6 teacher heldout:
  - rows with exact tool sequence but wrong arguments: `7/8`
  - diff counts: `7` scalar value mismatches, `6` missing required fields,
    `1` missing tool call, `1` missing gold key

Interpretation:

- Same-recipe 600-step training is a negative scaling point. It improves
  teacher-train constrained sequence selection to `12/12`, but public exact
  arguments fall from `5/8` to `3/8`, heldout exact arguments fall from `2/8`
  to `0/8`, and learned model-repair regresses across all slices.
- Future runs need early stopping or checkpoint sweeps around `250-350` steps,
  not simply more epochs over the same small corpus.

## Checkpoint-275 Sweep

Date: 2026-06-27 00:02 PDT.

After the 600-step run regressed, I evaluated the surviving early checkpoint
from the current best run:

- adapter:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model`
- tokenizer source:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300`
- eval run:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_eval96_modelrepair_max1`

Harness change:

- `scripts/eval_fastdllm_toolcall_cases.py` now supports
  `--tokenizer-path`, so checkpoint-only PEFT adapter folders can be evaluated
  without copying tokenizer files into each checkpoint directory.
- `scripts/run_fastdllm_toolcall_checkpoint_sweep.sh` wraps the standard
  public/train/heldout eval suite for one or more checkpoint adapter folders
  and writes `checkpoint_sweep_summary.tsv` under the chosen output root.

Comparable full-context eval at `max_new_tokens=96`,
`--constrained-max-calls 1`, and model repair enabled:

- public one-call, 8 cases:
  - raw strict exact sequence / args: `3/8` / `2/8`
  - constrained exact sequence / args: `8/8` / `5/8`
  - learned model-repair exact sequence / args: `4/8` / `2/8`
  - versus final step-300: public constrained is tied, model-repair is tied
- Qwen3.6 teacher train, 12 cases:
  - raw strict exact sequence / args: `2/12` / `2/12`
  - constrained exact sequence / args: `10/12` / `5/12`
  - learned model-repair exact sequence / args: `4/12` / `3/12`
  - versus final step-300: constrained is tied, model-repair has one fewer
    exact sequence but one more exact argument
- Qwen3.6 teacher heldout, 8 cases:
  - raw strict exact sequence / args: `1/8` / `0/8`
  - constrained exact sequence / args: `8/8` / `3/8`
  - learned model-repair exact sequence / args: `3/8` / `1/8`
  - versus final step-300: constrained improves from `7/8` / `2/8` to
    `8/8` / `3/8`, model-repair is tied

Checkpoint-275 constrained arg-diff diagnostic:

- public one-call:
  - rows with exact tool sequence but wrong arguments: `3/8`
  - diff counts: `5` scalar value mismatches, `1` missing required field
- Qwen3.6 teacher train:
  - rows with exact tool sequence but wrong arguments: `5/12`
  - diff counts: `6` missing required fields, `3` scalar value mismatches,
    `2` missing tool calls, `1` missing gold key
- Qwen3.6 teacher heldout:
  - rows with exact tool sequence but wrong arguments: `5/8`
  - diff counts: `9` missing required fields, `5` scalar value mismatches

Interpretation:

- Checkpoint-275 is the active best for the current one-call objective. It ties
  final step-300 on public and teacher-train constrained exact arguments while
  improving heldout constrained exact sequence and arguments.
- The remaining gap is still argument completion, not tool selection:
  checkpoint-275 reaches constrained exact sequence `8/8` on public and
  heldout, but exact arguments are `5/8` and `3/8`.
- The next trainer run should keep checkpoint sweeps enabled and compare
  `250/275/300/325` before promoting any longer run. Use
  `scripts/run_fastdllm_toolcall_checkpoint_sweep.sh` with `RUN_DIR=...` and
  `CHECKPOINTS="250 275 300 325"` when those checkpoints are retained.

## Agentic Multi-Call And Tool-Result Eval

Date: 2026-06-27 00:24 PDT.

After promoting checkpoint-275 on one-call slices, I evaluated it on the
existing agentic slices:

- public Hermes multi-call:
  `data/toolcall_eval/public_multicall_hermes_smoke.jsonl`
- synthetic text-compatible two-step tool-result:
  `data/toolcall_eval/synthetic_toolresult_smoke.jsonl`
- synthetic OpenAI-style two-step tool-result with `role=tool`:
  `data/toolcall_eval/synthetic_toolresult_openai_smoke.jsonl`

Teacher comparison bars already in the repo:

- Qwen3.6 public multi-call:
  `11/12` exact sequence, `10/12` exact arguments
- Qwen3.6 synthetic tool-result:
  `10/10` exact sequence and arguments
- Qwen3.6 OpenAI-style required tool-result:
  `10/10` exact sequence, `8/10` exact arguments

Public multi-call eval:

- run:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_multicall_eval384_modelrepair`
- settings:
  `max_new_tokens=384`, unconstrained call count, full-context sampling,
  deterministic repair/constrained projection enabled, model-repair pass
  enabled
- raw strict exact sequence / args: `1/12` / `0/12`
- deterministic repaired exact sequence / args: `7/12` / `0/12`
- constrained exact sequence / args: `7/12` / `1/12`
- learned model-repair exact sequence / args: `1/12` / `1/12`
- repeated-call rows: raw `2/12`
- elapsed: `760.7648890018463s`

Public multi-call constrained arg-diff:

- rows with exact tool sequence but wrong arguments: `6/12`
- diff counts: `18` scalar value mismatches, `3` missing required fields,
  `2` missing tool calls
- interpretation: multi-call failure is mostly chained value copying and
  per-call argument fidelity, not choosing every tool name from scratch.

Synthetic text-compatible tool-result eval:

- run:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_toolresult_eval160_modelrepair_max1`
- settings:
  `max_new_tokens=160`, `--constrained-max-calls 1`,
  full-context sampling, deterministic repair/constrained projection enabled,
  model-repair pass enabled
- raw strict exact sequence / args: `5/10` / `3/10`
- deterministic repaired exact sequence / args: `9/10` / `1/10`
- constrained exact sequence / args: `10/10` / `8/10`
- learned model-repair exact sequence / args: `2/10` / `1/10`
- constrained arg-diff:
  `2` exact-tool-sequence rows with wrong arguments, both scalar value
  mismatches

Synthetic OpenAI-style tool-result eval:

- run:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_openai_toolresult_eval160_modelrepair_max1`
- settings:
  `max_new_tokens=160`, `--constrained-max-calls 1`,
  full-context sampling, deterministic repair/constrained projection enabled,
  model-repair pass enabled
- raw strict exact sequence / args: `6/10` / `6/10`
- deterministic repaired exact sequence / args: `9/10` / `4/10`
- constrained exact sequence / args: `10/10` / `9/10`
- learned model-repair exact sequence / args: `5/10` / `5/10`
- constrained arg-diff:
  `1` exact-tool-sequence row with wrong arguments, both scalar value
  mismatches inside `send_email`

Interpretation:

- This is the first strong agentic next-action signal for the diffusion
  checkpoint. Under constrained max-1 projection, checkpoint-275 reaches
  teacher-level exact sequence on both tool-result slices and exceeds the
  recorded Qwen3.6 OpenAI-style exact-argument bar (`9/10` vs `8/10`), although
  the teacher run used a stricter OpenAI-native serving path.
- Multi-call remains the clearest agentic gap. The model can often identify the
  tool chain under deterministic projection (`7/12` sequence), but exact
  arguments collapse to `1/12`.
- Next training data should add multi-call continuation rows and chained
  argument-copy rows, not more one-call wrapper repair. The target is to move
  public multi-call constrained exact arguments above `1/12` without regressing
  the current tool-result `10/10` sequence behavior.

## Multi-Call Continuation Curriculum Negative Probe

Date: 2026-06-26 23:16 PDT.

I tested the next obvious corpus change from the agentic eval: add multi-call
continuation and chained argument-copy rows on top of the active model-repair
curriculum.

Builder and dataset:

- builder:
  `scripts/build_toolcall_multicall_curriculum.py`
- dataset:
  `data/qwen35_9b_toolcall_multicall_curriculum`
- base curriculum:
  `data/qwen35_9b_toolcall_model_repair_curriculum/train_agentic_mix.json`
- public source:
  `data/fastdllm_toolcall_train/train_toolcall.json`
- public multi-call records found: `56`
- generated candidate rows: `416`
- accepted multi-call rows after 896-token label-retention gate: `240`
- base rows retained: `227`
- final training rows: `467`
- final assistant call-count mix:
  - one-call: `311`
  - two-call: `116`
  - three-call: `38`
  - four-call: `2`
- final audit summary:
  - chosen length min/p50/p90/max: `239/722/866/895`
  - chosen kept-labels min/p50/p90/max: `24/55/104/315`
  - zero-after-truncation rows: `0`
  - partial-after-truncation rows: `0`

Training:

- one-step gate:
  `runs/fastdllm_qwen35_9b_toolcall_multicall_argspanw1p5_b896_step1_gate`
  - train loss: `9.725969314575195`
  - runtime: `9.2679s`
  - adapter saved successfully
- 300-step run:
  `runs/fastdllm_qwen35_9b_toolcall_multicall_argspanw1p5_b896_step300`
  - block size: `896`
  - argument-span weight: `1.5`
  - LoRA targets:
    `q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj`
  - dataset grouping disabled
  - max train samples: `467`
  - train loss: `2.3689375511805215`
  - runtime: `2574.0471s`
  - throughput: `0.117` steps/s
  - retained checkpoints: `checkpoint-275`, `checkpoint-300`

Public multi-call eval, checkpoint-275:

- run:
  `runs/fastdllm_qwen35_9b_toolcall_multicall_argspanw1p5_b896_ckpt275_multicall_eval384_modelrepair`
- settings:
  `max_new_tokens=384`, unconstrained call count, full-context sampling,
  deterministic repair/constrained projection enabled, model-repair pass
  enabled
- raw strict exact sequence / args: `0/12` / `0/12`
- deterministic repaired exact sequence / args: `3/12` / `0/12`
- constrained exact sequence / args: `4/12` / `1/12`
- learned model-repair exact sequence / args: `3/12` / `1/12`
- constrained arg-diff:
  - rows with exact tool sequence but wrong arguments: `3/12`
  - diff counts: `19` scalar value mismatches, `3` missing required fields,
    `2` missing tool calls

Public multi-call eval, checkpoint-300:

- run:
  `runs/fastdllm_qwen35_9b_toolcall_multicall_argspanw1p5_b896_ckpt300_multicall_eval384_nomodelrepair`
- settings:
  `max_new_tokens=384`, unconstrained call count, full-context sampling,
  deterministic repair/constrained projection enabled, model-repair pass
  disabled for the cheaper final-check
- raw strict exact sequence / args: `0/12` / `0/12`
- deterministic repaired exact sequence / args: `5/12` / `0/12`
- constrained exact sequence / args: `5/12` / `0/12`
- constrained arg-diff:
  - rows with exact tool sequence but wrong arguments: `5/12`
  - diff counts: `21` scalar value mismatches, `3` missing required fields,
    `2` missing tool calls

Comparison to the active best:

- active best checkpoint:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model`
- active-best public multi-call constrained sequence / args: `7/12` / `1/12`
- new multi-call ckpt-275 constrained sequence / args: `4/12` / `1/12`
- new multi-call ckpt-300 constrained sequence / args: `5/12` / `0/12`

Interpretation:

- Negative result for the main generator. Directly mixing full multi-call,
  exact-plan, and continuation rows increases the task surface but does not
  improve multi-call exact arguments; it regresses constrained tool-sequence
  recovery versus the active best.
- The failure is still value copying. The diagnostic remains dominated by
  scalar value mismatches, with some missing complex required payloads.
- Do not promote or scale this multi-call curriculum as-is.
- Next multi-call work should be more targeted:
  - train on smaller per-call argument-repair/continuation windows instead of
    full original requests when the source request is long
  - add retrieval/extraction supervision for exact scalar spans before asking
    the model to emit the whole chain
  - consider a separate repair adapter for multi-call argument completion
  - keep the active best model-repair + argument-span-1.5 checkpoint as the
    comparison point

## Separate Multi-Call Repair Adapter Probe

Date: 2026-06-27 00:55 PDT.

After the full-chain multi-call continuation mix regressed the main generator, I
kept the active generator checkpoint fixed and trained a separate repair adapter
against saved first-pass outputs.

New scripts:

- `scripts/build_toolcall_multicall_repair_curriculum.py`
- `scripts/eval_fastdllm_toolcall_repair_outputs.py`

Dataset:

- path:
  `data/qwen35_9b_toolcall_multicall_repair_curriculum`
- rows: `434`
- composition: `120` base model-repair rows plus `314` accepted public
  multi-call repair rows
- candidate pool: `896` controlled corruptions from `56` public train
  multi-call records
- accepted repair audit:
  - rendered length min/p50/p90/max: `593 / 789 / 884 / 892`
  - kept labels min/p50/p90/max: `60 / 76 / 107 / 115`
  - accepted zero-label rows: `0`
  - accepted partial-label rows: `0`

Training:

- one-step gate:
  `runs/fastdllm_qwen35_9b_toolcall_multicall_repair_argspanw1p5_b896_step1_gate`
  - train loss: `11.608933448791504`
  - runtime: `9.2858s`
- 100-step repair adapter:
  `runs/fastdllm_qwen35_9b_toolcall_multicall_repair_argspanw1p5_b896_step100`
  - block size: `896`
  - argument-span weight: `1.5`
  - GDN-aware LoRA targets:
    `q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj`
  - train loss: `3.8560502338409424`
  - runtime: `859.6796s`
  - throughput: `0.116` steps/s

Eval setup:

- base generator drafts:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_multicall_eval384_modelrepair/public_multicall_12.jsonl`
- cases:
  `data/toolcall_eval/public_multicall_hermes_smoke.jsonl`
- repair decode:
  `max_new_tokens=384`, full-context sampling, no adapter merge,
  deterministic schema repair and constrained projection scored after the
  learned repair output

Eval on raw `assistant` drafts:

- output:
  `runs/fastdllm_qwen35_9b_toolcall_multicall_repair_argspanw1p5_b896_step100_eval/public_multicall_12_assistant.jsonl`
- input draft sequence / args: `1/12` / `0/12`
- learned repair sequence / args: `3/12` / `2/12`
- learned repair plus constrained projection sequence / args: `5/12` / `3/12`
- arg-diff on repair-constrained output:
  - rows with exact tool sequence but wrong arguments: `2/12`
  - diff counts: `14` scalar value mismatches, `3` missing required fields,
    `3` missing tool calls

Eval on `constrained_assistant` drafts:

- output:
  `runs/fastdllm_qwen35_9b_toolcall_multicall_repair_argspanw1p5_b896_step100_eval/public_multicall_12_constrained_draft.jsonl`
- input draft sequence / args: `7/12` / `1/12`
- learned repair sequence / args: `3/12` / `1/12`
- learned repair plus constrained projection sequence / args: `5/12` / `2/12`
- arg-diff on repair-constrained output:
  - rows with exact tool sequence but wrong arguments: `3/12`
  - diff counts: `18` scalar value mismatches, `2` missing required fields,
    `1` missing tool call

Comparison to active best:

- active best multi-call constrained projection: `7/12` sequence / `1/12`
  arguments
- separate repair adapter, raw-draft repair-constrained: `5/12` / `3/12`
- separate repair adapter, constrained-draft repair-constrained: `5/12` /
  `2/12`

Interpretation:

- Partial positive: a separate repair adapter can recover additional exact
  arguments without modifying the active first-pass generator.
- Not promotable: both repair paths reduce exact tool-sequence recovery versus
  the active constrained projection.
- The remaining gap is still scalar value copying/extraction. The repair model
  is improving syntax and sometimes arguments, but it rewrites the call chain
  too aggressively.
- Next repair work should constrain repair to preserve the existing tool
  sequence, or train on per-call scalar extraction windows where the tool name
  and argument keys are fixed.

## Fixed-Sequence Repair Adapter Probe

Date: 2026-06-27 01:30 PDT.

The previous separate repair adapter improved exact arguments on a few rows but
lost tool order. I then removed sequence-corrupting training variants and trained
a narrower fixed-sequence repair adapter whose prompt explicitly says to keep the
same function names, order, and call count.

New/updated scripts:

- `scripts/build_toolcall_sequence_repair_curriculum.py`
- `scripts/eval_fastdllm_toolcall_repair_outputs.py`
  - added `--repair-prompt-mode preserve_sequence`

Dataset:

- path:
  `data/qwen35_9b_toolcall_sequence_repair_curriculum`
- rows: `274`
- source records: `56` public train multi-call records
- candidate count: `784`
- accepted count: `274`
- accepted variants:
  `empty_args`, `null_args`, `wrong_scalar`, `drop_first_arg`, `mixed_null`,
  `mixed_wrong`, `gold_skeleton`
- accepted audit:
  - rendered length min/p50/p90/max: `599 / 785 / 875 / 895`
  - kept labels min/p50/p90/max: `60 / 76 / 107 / 115`
  - accepted zero-label rows: `0`
  - accepted partial-label rows: `0`

Training:

- one-step gate:
  `runs/fastdllm_qwen35_9b_toolcall_sequence_repair_argspanw1p5_b896_step1_gate`
  - train loss: `10.315794944763184`
  - runtime: `9.298s`
- 100-step adapter:
  `runs/fastdllm_qwen35_9b_toolcall_sequence_repair_argspanw1p5_b896_step100`
  - block size: `896`
  - argument-span weight: `1.5`
  - GDN-aware LoRA targets:
    `q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj`
  - train loss: `3.77995418548584`
  - runtime: `855.6122s`
  - throughput: `0.117` steps/s

Eval:

- input drafts:
  active-best `constrained_assistant` outputs from
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_multicall_eval384_modelrepair/public_multicall_12.jsonl`
- output:
  `runs/fastdllm_qwen35_9b_toolcall_sequence_repair_argspanw1p5_b896_step100_eval/public_multicall_12_constrained_draft.jsonl`
- settings:
  `max_new_tokens=384`, full-context sampling,
  `--repair-prompt-mode preserve_sequence`, deterministic schema repair and
  constrained projection scored after the learned repair output
- input constrained draft sequence / args: `7/12` / `1/12`
- learned fixed-sequence repair sequence / args: `1/12` / `0/12`
- learned repair plus constrained projection sequence / args: `3/12` / `1/12`
- repair-constrained arg-diff:
  - rows with exact tool sequence but wrong arguments: `2/12`
  - diff counts: `17` scalar value mismatches and `6` missing tool calls

Interpretation:

- Negative result. The prompt and training data asked the adapter to preserve
  sequence, but the learned repair still damaged tool order and did not improve
  exact arguments over the active constrained draft.
- The model-based second-stage repair lane is not yet reliable for multi-call
  outputs. It tends to rewrite the call chain even when the prompt forbids it.
- Next multi-call work should move sequence preservation out of free-form model
  repair and into deterministic/generation-time constraints, or reduce the
  learned task to per-field scalar extraction where the output schema is not a
  full tool-call chain.

## Sequence-Preserving Deterministic Projection

Date: 2026-06-27 02:10 PDT.

After the fixed-sequence learned repair adapter damaged tool order, I moved the
same idea into deterministic projection over an already selected draft. This is
not a new trained checkpoint. It preserves the function names, order, and call
count from a selected output field, including repeated calls, then refills each
call's arguments from the per-call parsed JSON plus schema-guided context
extraction.

Updated scripts:

- `scripts/eval_fastdllm_toolcall_cases.py`
  - added `sequence_preserving_constrained_tool_call_text`
  - added per-call argument normalization and repeated-token cleanup
  - fixed enum matching so enum values such as `bright` do not match inside
    words such as `brightness`
  - added key matching for underscore/space variants such as `error_budget`
    versus "error budget"
- `scripts/rescore_fastdllm_toolcall_outputs.py`
  - added `--text-field`
  - added `--sequence-preserving-constrained`

Important implementation note:

- An initial v2 attempt let natural-language string extraction override parsed
  string arguments too aggressively. That kept exact arguments at `1/12` and
  increased scalar mismatches. The retained v3 behavior uses parsed string
  values first, while still allowing context extraction to fix enums and numeric
  keys.

Eval setup:

- cases:
  `data/toolcall_eval/public_multicall_hermes_smoke.jsonl`
- active-best generator output:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_multicall_eval384_modelrepair/public_multicall_12.jsonl`
- best deterministic rescore output:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_multicall_eval384_modelrepair/public_multicall_12_sequence_preserve_rescore_v3.jsonl`
- argument diff:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_multicall_eval384_modelrepair/public_multicall_12_sequence_preserve_v3_argdiff.jsonl`

Results:

- active-best constrained draft baseline: `7/12` exact sequence, `1/12` exact
  arguments
- sequence-preserving deterministic projection over `constrained_assistant`:
  `7/12` exact sequence, `4/12` exact arguments
- same projection over raw `assistant`: `6/12` exact sequence, `3/12` exact
  arguments
- best v3 arg-diff:
  - rows with exact tool sequence but wrong arguments: `3/12`
  - diff counts: `14` scalar value mismatches, `3` missing required fields,
    `2` missing tool calls

Interpretation:

- Positive result for the multi-call lane. The active checkpoint remains the
  same, but deterministic projection recovers three additional exact-argument
  rows without sacrificing the `7/12` tool-sequence score.
- The best setting is to start from the active constrained draft, not the raw
  assistant draft. Raw projection improves arguments but loses one exact tool
  sequence.
- This should be promoted as the current public multi-call reporting path and
  as the blueprint for generation-time constrained decoding. The remaining
  failures are still mostly scalar-copy issues, plus a few missing complex
  required payloads and missing calls.

## Multi-Call Gap Curriculum

Date: 2026-06-27.

After contextual scalar projection, the remaining public multi-call failures are
not just scalar copies. The unresolved cases include missing calls, wrong
call order/tool choice, and complex required payloads such as arrays or objects.

New builder:

- `scripts/build_toolcall_multicall_gap_curriculum.py`

Output:

- `data/qwen35_9b_toolcall_multicall_gap_curriculum/train_agentic_mix.json`
- `data/qwen35_9b_toolcall_multicall_gap_curriculum/train_agentic_mix.audit.jsonl`
- `data/qwen35_9b_toolcall_multicall_gap_curriculum/train_agentic_mix.manifest`

Build result:

- source records: `56` public train multi-call records
- candidates: `199`
- accepted rows: `181`
- rejected rows: `18`
- accepted mix: `137` missing-call recovery rows and `44` complex
  array/object extraction rows
- block size: `896`
- truncation side: `right`
- max request excerpt: `1200` chars
- accepted rendered length min/p50/p90/max: `405 / 728 / 842 / 894`
- accepted kept assistant labels min/p50/p90/max: `25 / 45 / 98 / 194`
- accepted zero-label rows after truncation: `0`
- accepted partial-label rows after truncation: `0`

Interpretation:

- Positive CPU data-path result. Shortening missing-call prompts to call-local
  excerpts raised accepted rows from `160` to `181` and reduced rejected rows
  from `39` to `18` without accepting any truncated-label rows.
- This is not a new trained checkpoint or promoted model score.
- Use it as a staged repair/extraction lane first. Do not mix it heavily into
  the main generator until a one-step QLoRA gate and a short adapter preserve
  the active public multi-call sequence path.

### First Gap Adapter Probe

Additional tooling:

- `scripts/build_toolcall_multicall_gap_eval_cases.py`
  - produced `data/toolcall_eval/public_multicall_gap_eval.jsonl`
  - held-out rows: `38`
  - missing-call rows: `31`
  - complex-extraction rows: `7`
- `scripts/run_fastdllm_qwen35_9b_agentic_qlora_pilot.sh`
  - now supports `LORA_MODEL_PATH` to continue training from an existing PEFT
    adapter.

Training:

- standalone one-step gate from diffusion init:
  - run:
    `runs/fastdllm_qwen35_9b_toolcall_multicall_gap_argspanw1p5_b896_step1_gate`
  - labels: pre-MDM `[29]`, post-MDM `[25, 4]`
  - train loss: `9.249107360839844`
- standalone 50-step adapter from diffusion init:
  - run:
    `runs/fastdllm_qwen35_9b_toolcall_multicall_gap_argspanw1p5_b896_step50`
  - train loss: `7.142361106872559`
- one-step continuation gate from active checkpoint-275:
  - run:
    `runs/fastdllm_qwen35_9b_toolcall_multicall_gap_from_ckpt275_argspanw1p5_b896_step1_gate`
  - labels: pre-MDM `[29]`, post-MDM `[25, 4]`
  - train loss: `3.570006847381592`
- 50-step continuation from active checkpoint-275:
  - run:
    `runs/fastdllm_qwen35_9b_toolcall_multicall_gap_from_ckpt275_argspanw1p5_b896_step50`
  - train loss: `2.550190029144287`
  - runtime: `107.9005s`

Held-out gap eval with full-context sampling and constrained max-1 projection:

- active checkpoint-275:
  - raw valid JSON: `5/38`
  - raw exact sequence / arguments: `13/38` / `9/38`
  - constrained exact sequence / arguments: `37/38` / `26/38`
- checkpoint-275 plus 50 gap steps:
  - raw valid JSON: `8/38`
  - raw exact sequence / arguments: `11/38` / `6/38`
  - constrained exact sequence / arguments: `38/38` / `23/38`
- by kind:
  - missing-call constrained arguments regressed from `24/31` to `20/31`
  - complex-extract constrained arguments improved from `2/7` to `3/7`

Interpretation:

- Negative promotion result. The continuation makes syntax a bit better and
  helps one complex-payload constrained row, but it regresses the dominant
  missing-call lane and overall exact arguments.
- Do not promote
  `runs/fastdllm_qwen35_9b_toolcall_multicall_gap_from_ckpt275_argspanw1p5_b896_step50`.
- Keep `LORA_MODEL_PATH` and the gap eval builder. The next experiment should
  split missing-call recovery from complex-payload extraction rather than mixing
  both as a short free-form continuation.

### Complex-Only Split Probe

I split out only the complex-payload extraction lane after the mixed gap probe
regressed missing-call recovery.

Data:

- train:
  `data/qwen35_9b_toolcall_multicall_complex_extract_curriculum`
- held-out eval:
  `data/toolcall_eval/public_multicall_complex_extract_eval.jsonl`
- train rows: `44`
- train rejected rows: `0`
- train zero-label / partial-label accepted rows: `0` / `0`
- held-out eval rows: `7`

Training:

- one-step continuation gate from active checkpoint-275:
  - run:
    `runs/fastdllm_qwen35_9b_toolcall_complex_extract_from_ckpt275_argspanw1p5_b896_step1_gate`
  - labels: pre-MDM `[55]`, post-MDM `[50, 5]`
  - train loss: `4.448453426361084`
- 25-step continuation from active checkpoint-275:
  - run:
    `runs/fastdllm_qwen35_9b_toolcall_complex_extract_from_ckpt275_argspanw1p5_b896_step25`
  - train loss: `3.2026946640014646`
  - runtime: `54.1621s`

Held-out complex eval:

- active checkpoint-275:
  - raw valid JSON: `1/7`
  - raw exact sequence / arguments: `1/7` / `1/7`
  - constrained exact sequence / arguments: `7/7` / `2/7`
  - constrained schema-valid / required-present: `3/7` / `3/7`
- checkpoint-275 plus 25 complex steps:
  - raw valid JSON: `1/7`
  - raw exact sequence / arguments: `1/7` / `0/7`
  - constrained exact sequence / arguments: `7/7` / `3/7`
  - constrained schema-valid / required-present: `4/7` / `5/7`

Interpretation:

- Mixed result. Complex-only continuation improves constrained complex-payload
  extraction and required-field coverage by a small amount, but raw exact
  arguments regress.
- Do not promote
  `runs/fastdllm_qwen35_9b_toolcall_complex_extract_from_ckpt275_argspanw1p5_b896_step25`.
- Next work should move complex array/object recovery into generation-time
  constrained decoding or a per-field extraction/acceptance policy.

## Decision

The first infrastructure milestone is complete: the Qwen3.5-9B diffusion
candidate is no longer just a scaffold, the local 5090 can run 896-token QLoRA
training against it, and the resulting adapter can be scored by the strict
tool-call eval harness.

Do not treat the current checkpoint as an agentic model yet. The next run should
train long enough to move strict tool-call metrics, and the data path should keep
assistant/tool-call labels in-window.

Recommended next run:

- keep the 5080 idle for the user's 3-hour reservation
- use the local 5090
- use the LMFlow `None`-field sanitizer and require a nonzero-label smoke
  before every training run
- evaluate Qwen3.5 diffusion checkpoints with `--full-context-sampling` until
  the bridge has real KV-cache support
- replace blind truncation with label-aware packing that keeps tool schemas,
  user request, and assistant/tool-call labels in one window
- expand the public/Qwen3.6 teacher mix only through label-aware packing
- keep constrained `<tool_call>` decoding in eval and prototype generation-time
  constraints next
- use argument-span weight `1.5` as the current balanced default for the next
  longer run, with weight `2.0` retained as the comparison point for raw public
  argument exactness
- do not use argument-span weight `3.0` as the next scaling recipe; it regressed
  public and teacher-train exact argument recovery
- keep the model-repair pass in the eval suite and keep the checkpoint-275
  model-repair + argument-span-1.5 adapter as the active best for now
- for one-call eval slices, report `--constrained-max-calls 1` constrained
  metrics beside the uncapped constrained metrics; this recovers the correct
  one-call contract without changing raw strict scores
- keep `--model-repair-max-new-tokens 96` as the comparable eval default for
  now; the 160-token check slightly improved valid JSON but regressed or failed
  to improve exact sequence and exact arguments
- do not scale the cap-80 clean-repair curriculum mix; it trained stably but
  regressed raw, constrained, and heldout exactness versus the previous
  model-repair curriculum
- do not scale the current hard argument-completion mix as the next main
  generator recipe; it improved teacher-train constrained tool sequence but
  regressed public raw exactness and exact argument recovery
- next 5090 pilot should target nonzero strict public one-call exact sequence
  plus better argument exactness while preserving the synthetic gains from the
  format-only run; label preservation and explicit argument-copy rows alone are
  not enough
- use the argument-diff diagnostic to build future hard argument-completion
  rows from train-slice failures only, but keep them lower-weight, staged, or
  in a separate repair adapter until they stop hurting the main generator
- do not scale the same model-repair corpus to 600 steps as the next default;
  it overfits/regresses public and heldout exact arguments despite improving
  teacher-train constrained sequence selection
- add checkpoint sweep/early-stopping selection around `250-350` steps before
  longer runs
- use checkpoint-275 from
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300` as
  the active comparison point: public constrained `8/8` / `5/8`,
  teacher-train constrained `10/12` / `5/12`, heldout constrained `8/8` /
  `3/8`
- use `qwen35_9b_diffusion_ckpt275_agentic_scorecard.md` as the current
  comparable checkpoint summary for the active 9B diffusion adapter
- use `scripts/run_fastdllm_toolcall_checkpoint_sweep.sh` after future runs to
  make checkpoint promotion evidence repeatable; the script now includes
  public multi-call and synthetic tool-result slices when
  `INCLUDE_AGENTIC_SLICES=1`, plus contextual public multi-call projection when
  `INCLUDE_CONTEXTUAL_PROJECTION=1`
- add public multi-call and synthetic tool-result slices to every promoted
  checkpoint report. Current active checkpoint-275 reaches sequence-preserving
  public multi-call constrained `7/12` / `4/12`, and the separate scalar
  repair adapter improves the two-stage multi-call path to `7/12` / `5/12`;
  text-compatible tool-result constrained `10/10` / `8/10`, and OpenAI-style
  tool-result constrained `10/10` / `9/10`
- do not scale the first multi-call continuation curriculum as-is; it regressed
  public multi-call constrained sequence recovery to `4/12-5/12` and did not
  improve exact arguments
- the next corpus change is now built and gated:
  `qwen35_9b_multicall_scalar_curriculum_result.md` records a 1184-row
  one-call scalar extraction/argument-repair curriculum from public multi-call
  training rows, with zero rejected rows, zero label-loss rows, and a successful
  one-step QLoRA gate on the local 5090
- the staged/lower-weight scalar-mix main-generator experiment has now been
  run and should not be promoted:
  `qwen35_9b_modelrepair_scalar_mix_result.md` records a 355-row mix with 227
  model-repair rows plus 128 balanced scalar rows. It trained cleanly, but
  checkpoint-275 regressed sequence-preserving public multi-call from the
  active `7/12` / `4/12` path to `4/12` / `2/12`, and public one-call
  constrained arguments fell from `5/8` to `3/8`.
- the first scalar adapter test is positive but procedural:
  `qwen35_9b_multicall_scalar_adapter_result.md` records a 100-step scalar
  adapter that improves public multi-call exact arguments from `4/12` to `5/12`
  under sequence-preserving constrained projection. Treat it as a staged repair
  signal and generation-time decoding blueprint, not as a first-pass generator
  promotion.
- the 300-step extension of the same scalar adapter trained cleanly but did not
  improve the public multi-call top line: checkpoint-275 and checkpoint-300
  both stayed at `7/12` sequence and `5/12` arguments. Do not spend more 5090
  time on more steps for this exact scalar curriculum.
- contextual scalar projection over the scalar-repair outputs is the first
  positive next-step result after that plateau:
  `scripts/rescore_scalar_repair_contextual_projection.py` uses call-local
  request evidence for quoted IDs, date/time fields, and explicit missing
  required scalar fields. On the same public multi-call slice it improves the
  direct constrained postprocessed path from `7/12` sequence and `4/12`
  arguments to `7/12` sequence and `7/12` arguments, with zero exact-sequence
  rows left wrong on arguments. This is a deterministic projection prototype,
  not a model-only metric, and it now reaches the same top line without running
  the scalar-repair adapter. Cross-slice projection checks are neutral on
  one-call and tool-result slices, so this should be treated as a public
  multi-call scalar-grounding fix rather than a broad metric inflator.
- next multi-call work should keep the active generator fixed and move scalar
  correction into generation-time constrained decoding, per-field extraction,
  or a separate lightweight repair stage. If scalar rows are retried in the
  generator, use a much lower ratio with stronger one-call replay and require
  an early public multi-call sequence gate before running the full scorecard.
- the next gap-focused curriculum is built:
  `qwen35_9b_multicall_gap_curriculum_result.md` records a 181-row
  missing-call and complex-payload extraction set from public train multi-call
  records. Use it first for staged repair/extraction or generation-time
  constrained decoding, not as a heavy main-generator mix.
- the first 50-step checkpoint-275 gap continuation should not be promoted:
  it regresses held-out gap exact arguments from `9/38` to `6/38` raw and from
  `26/38` to `23/38` under constrained projection, despite a small syntax gain.
- the first 25-step complex-only continuation also should not be promoted:
  constrained exact arguments improve from `2/7` to `3/7`, but raw exact
  arguments regress from `1/7` to `0/7`.
- complex-context constrained decoding is now the promoted fix for the
  complex-payload lane:
  `scripts/eval_fastdllm_toolcall_cases.py` extracts conservative array/object
  payloads from request evidence such as markdown tables, bullets, and inline
  lists. On the 7-row held-out complex eval it improves active checkpoint-275
  constrained exact arguments from `2/7` to `7/7`; the trained complex adapter
  also reaches `7/7` after this decoder, so it is not worth promoting. On the
  public multi-call active path, the top line remains `7/12` sequence and
  `7/12` arguments after contextual scalar projection, while schema/required
  coverage improves to `12/12`.
- guarded sequence-planner projection is now the promoted missing-call
  diagnostic:
  `scripts/rescore_toolcall_sequence_planner_projection.py` uses numbered
  request lists, markdown table sections, and tool schema evidence to propose a
  multi-call order, but only replaces outputs that already have at least two
  calls. It improves the active public multi-call path from `7/12` sequence and
  `7/12` arguments to `11/12` sequence and `10/12` arguments after
  segment-local scalar extraction, with neutral checks on public one-call,
  Qwen3.6 teacher one-call, and both tool-result slices. This is still a
  deterministic projection result, not model-only learning.
- train-only sequence-planner distillation is now built and gated:
  `scripts/build_toolcall_sequence_planner_distill_curriculum.py` uses public
  training multi-call rows only. The deterministic planner matches gold
  sequence on `27/56` train rows and exact arguments on `2/56`; after the
  strict block-size-`896` full-label gate, `13` sequence-selected gold-target
  rows remain. A one-step QLoRA gate from active checkpoint-275 completed on
  the local 5090 under `MemoryMax=28G`, with train loss
  `1.42819082736969`. Result note:
  `qwen35_9b_sequence_planner_distill_curriculum_result.md`. Treat this as a
  small replay component and data-path validation, not a promoted checkpoint.
- compact-schema sequence-planner recovery was also tested. Compact/request-only
  prompts recover `18` fully labeled rows at block size `896` and `21` at block
  size `1024`; compact/instruction prompts recover `17` and `20`. Both
  1024-token one-step gates fit on the local RTX 5090 under the same cgroup
  cap. Loss was `4.3695` for compact/request-only and `2.6395` for
  compact/instruction. This is a row-recovery/data-shaping result, not a
  promoted checkpoint; compact rows need a full-schema eval gate before any
  longer continuation.
- compact/instruction checkpoint-1 passed the first full-schema eval gate but
  is still not promoted. It matches active constrained/projected top lines on
  public one-call (`8/8` sequence, `5/8` arguments), public multi-call
  contextual projection (`7/12`, `7/12`), guarded sequence-planner projection
  (`11/12`, `10/12`), synthetic tool-result (`10/10`, `8/10`), and
  OpenAI-style tool-result (`10/10`, `9/10`). Teacher-heldout constrained exact
  arguments improve to `4/8`, but raw public multi-call remains weak at `1/12`
  sequence and `0/12` arguments with repeated-call failures. Treat this as a
  safe compact-row gate only, not a new active checkpoint.
- compact/instruction checkpoint-25 should not be promoted. The 25-step
  continuation trained cleanly on the 20 compact/instruction rows at block size
  `1024`, but one-call full-schema eval regressed public constrained exact
  arguments from active `5/8` to `4/8`, teacher-train constrained arguments
  from `5/12` to `4/12`, and teacher-heldout constrained arguments to `2/8`.
  The sweep was stopped after the one-call failure to save 5090 time. This
  pushes the next work toward GDN-specific architecture/sampler ablations
  instead of more planner-row scaling.
- first GDN-specific LoRA family gate is complete:
  `qwen35_gdn_lora_ablation_gate_result.md` records base-start one-step gates
  for GDN-only, attention-only, and mixed adapters on the same model-repair
  curriculum. All three instantiate, train, and save under the local 5090
  cgroup cap. This proves the ablation branches are technically runnable, but
  the identical one-step loss is not a quality ranking.
- follow-up 25-step base-start branch gate is complete:
  mixed attention+GDN LoRA reached the best training loss (`7.9278`) and best
  raw public one-call sequence count (`2/8`), attention-only reached the best
  constrained sequence count (`7/8`), and all three remained at `0/8` raw exact
  arguments and `1/8` constrained exact arguments. This does not challenge the
  active checkpoint-275 public one-call top line (`8/8` constrained sequence,
  `5/8` constrained exact arguments). Treat the result as a reason to move from
  target-family ablation to GDN masking/state ablation.
- first GDN state-isolation probe is complete:
  `FAST_DLLM_QWEN3_5_GDN_MODE=option_a_noisy_block_isolation_v0` was added to
  the local Qwen3.5 Fast-DLLM bridge. The mode resets the noisy MDM stream at
  diffusion-block boundaries while leaving the clean `x_0` stream causal and
  leaving generation/eval forwards causal. A one-step smoke trained and saved;
  the 25-step mixed LoRA run reached `2/8` raw exact sequence and `7/8`
  constrained exact sequence on the public one-call cheap gate, but had much
  worse train loss (`10.9004`) and no argument gain (`0/8` raw, `1/8`
  constrained). Do not extend this exact variant; use it as evidence that hard
  noisy-block reset is too blunt.
- first clean-state GDN injection probe is complete:
  `FAST_DLLM_QWEN3_5_GDN_MODE=option_a_clean_state_injection_v0` records clean
  `x_0` recurrent state at diffusion-block boundaries and initializes each
  noisy block from the previous clean boundary state. A one-step smoke trained
  and saved with loss `7.3922`; the 25-step mixed LoRA run reached train loss
  `8.0212`, public one-call constrained exact sequence `7/8`, and constrained
  exact arguments `1/8`. This is much healthier than hard noisy-block reset,
  but it still does not challenge active checkpoint-275 or improve arguments.
  Keep the mode for objective-pairing probes, not as a promoted checkpoint.
- clean-state plus mild structural objective probe is complete:
  with `STRUCTURAL_LOSS_WEIGHT=1.25` and `ARGUMENT_SPAN_LOSS_WEIGHT=1.5`, the
  one-step smoke trained and saved with loss `7.6829`; the 25-step run reached
  train loss `8.2204`, public one-call constrained exact sequence `7/8`, and
  constrained exact arguments `0/8`. This regresses the single constrained
  argument hit from clean-state-only, so do not scale naive structural-token
  loss pairing.
- clean-state local dual-pass GDN probe is complete:
  `FAST_DLLM_QWEN3_5_GDN_MODE=option_a_clean_state_dualpass_v0` adds a reverse
  local noisy-block pass and averages it with the clean-state forward pass. The
  one-step smoke trained and saved with loss `8.3058`; the 25-step run reached
  train loss `8.9085`, public one-call raw exact sequence `0/8`, constrained
  exact sequence `6/8`, and constrained exact arguments `1/8`. Do not scale this
  exact dual-pass variant.
- clean-state value-copy objective probe is complete:
  `scripts/fastdllm_value_copy_token_ids.py` derives scalar argument-value token
  IDs from the selected curriculum and the launcher exposes
  `VALUE_COPY_LOSS_WEIGHT`. On the model-repair curriculum it found `227` tool
  calls, `869` scalar values, `241` unique scalar values, and `410` token IDs.
  With `VALUE_COPY_LOSS_WEIGHT=2.0` and `ARGUMENT_SPAN_LOSS_WEIGHT=1.5`, the
  one-step smoke trained and saved with loss `8.1665`; the 25-step run reached
  train loss `8.8879`, public one-call raw exact sequence `0/8`, constrained
  exact sequence `7/8`, and constrained exact arguments `1/8`. Do not scale
  naive token-ID value weighting as-is.
- clean-state aligned value-span objective probe is complete:
  `FASTDLLM_VALUE_SPAN_LOSS_WEIGHT` and
  `FASTDLLM_VALUE_SPAN_TOKEN_IDS` boost scalar argument-value token IDs only
  while labels are inside the derived `arguments ... </tool_call>` span, and
  the launcher exposes `VALUE_SPAN_LOSS_WEIGHT`, `VALUE_SPAN_TOKEN_IDS`, and
  `VALUE_SPAN_TOKEN_MANIFEST`. On the same model-repair curriculum, the
  manifest again found `227` tool calls, `869` scalar values, `241` unique
  scalar values, and `410` token IDs. With `VALUE_SPAN_LOSS_WEIGHT=2.0` and
  `ARGUMENT_SPAN_LOSS_WEIGHT=1.5`, the one-step smoke trained with loss
  `7.9921` and confirmed `18` aligned value-span labels in the first batch; the
  25-step run reached train loss `8.1482`, public one-call raw exact sequence
  `1/8`, constrained exact sequence `7/8`, and constrained exact arguments
  `0/8`. This is cleaner than global value-copy but still not a promotion
  candidate; move next to teacher-KL or full argument-span denoising.
- clean-state argument-span mask-forcing probe is complete:
  `FASTDLLM_ARGUMENT_SPAN_MASK_PROB` forces a sampled subset of labels inside
  the derived `arguments ... </tool_call>` span into the masked MDM branch, and
  the launcher exposes `ARGUMENT_SPAN_MASK_PROB`. With
  `ARGUMENT_SPAN_MASK_PROB=1.0`, the smoke forced `41/41` argument-span labels,
  train loss was `12.8202`, the 25-step run reached train loss `9.9601`, and
  public one-call constrained exact sequence / arguments were `6/8` / `0/8`.
  With `ARGUMENT_SPAN_MASK_PROB=0.5`, the smoke forced `24/41` argument-span
  labels, train loss was `11.1944`, the 25-step run reached train loss
  `7.7339`, and public one-call constrained exact sequence / arguments were
  `7/8` / `1/8`. A 25-step p=`0.5` continuation from active checkpoint-275
  trained cleanly to train loss `2.1003`, but public one-call constrained exact
  arguments regressed from active `5/8` to `2/8`. Keep the hook; do not promote
  hard argument-span mask forcing without a teacher-KL/full-span target. A
  gentler p=`0.1` continuation from active checkpoint-275 at LR `5e-6` forced
  `5/41` argument-span labels in the smoke, trained cleanly to train loss
  `2.2841`, and reached public one-call constrained exact sequence / arguments
  `7/8` / `3/8`. This improves over p=`0.5` on arguments but still misses the
  active `8/8` / `5/8` baseline, so stop hard-mask sweeps here.
- hard clean-repair/full-span replay from active checkpoint-275 is complete and
  negative:
  `data/qwen35_9b_toolcall_model_repair_clean_hard24_curriculum` contains 246
  rows: 147 original label-aware rows, 80 prior model-repair rows, and 19
  accepted clean-repair rows. The 25-step continuation
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_clean_hard24_argspanw1p5_b896_ckpt275_lr5e6_step25`
  used LR `5e-6`, clean-state injection, and argument-span weight `1.5`, and
  trained cleanly to loss `2.0963`. The public one-call gate regressed to raw
  exact sequence / arguments `2/8` / `1/8` and constrained exact sequence /
  arguments `7/8` / `3/8`, below active checkpoint-275's `8/8` / `5/8`
  constrained baseline. Do not promote or scale broad repair/full-span replay
  without changing the target to teacher-KL or a span-local objective.
- grounded one-call constrained projection is promoted as the current
  sampler-side one-call reporting path, with the active trained adapter still
  checkpoint-275:
  `scripts/eval_fastdllm_toolcall_cases.py` now extracts request-evidence
  weekly schedules, ID-like strings, and low-confidence contextual strings
  before trusting corrupted parsed scalar values. CPU-only rescoring over the
  existing active outputs improves public one-call constrained exact sequence /
  arguments from `8/8` / `5/8` to `8/8` / `8/8`, improves Qwen3.6 teacher-train
  one-call from `10/12` / `5/12` to `10/12` / `6/12`, and improves
  teacher-heldout one-call from `8/8` / `4/8` to `8/8` / `6/8`. Synthetic
  tool-result remains `10/10` / `8/10`, OpenAI-style tool-result remains
  `10/10` / `9/10`, and public multi-call constrained-draft sequence-preserve
  remains `7/12` / `4/12`. Result note:
  `qwen35_9b_contextual_projection_suite_result.md`.
- grounded span-fill curriculum is now built and gated:
  `scripts/build_toolcall_grounded_spanfill_curriculum.py` turns exact
  grounded projection rows into train-slice repair/span-fill examples. The
  block-`1024` dataset
  `data/qwen35_9b_toolcall_grounded_spanfill_teacher_train_b1024_curriculum`
  accepted `16` rows with full label retention; the smaller block-`896` build
  accepted `12` and rejected the longer schedule rows. A one-step continuation
  from active checkpoint-275 at LR `5e-6` fit under the 5090 cgroup cap, reached
  train loss `3.069`, and tied the active grounded constrained one-call scores:
  public `8/8` / `8/8`, teacher-train `10/12` / `6/12`, and teacher-heldout
  `8/8` / `6/8`. Raw model-only scores did not improve (`3/8` / `2/8`,
  `2/12` / `2/12`, `1/8` / `0/8`), so this adapter is not promoted. Result
  note: `qwen35_9b_grounded_spanfill_curriculum_result.md`.
- value-span mask forcing is now implemented as a narrower span-local hook:
  `FASTDLLM_VALUE_SPAN_MASK_PROB` in the local model bridge and
  `VALUE_SPAN_MASK_PROB` in the QLoRA launcher. On the grounded b1024 dataset,
  the extracted scalar-value token set has `27` token IDs from `200` scalar
  values and `18` unique scalar values. The one-step continuation
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespanmask_from_ckpt275_b1024_step1`
  used value-span mask probability `1.0`, value-span loss weight `2.0`, and
  argument-span loss weight `1.5`; the first debug batch forced `17` value
  labels while leaving whole-span forcing at `0`, and train loss was `3.5501`.
  Eval tied active grounded scores on public, teacher-train, and
  teacher-heldout one-call slices, with no raw model-only gain, so this adapter
  is not promoted.
- value-span-only 25-step continuation is complete and negative:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespanmask_only_from_ckpt275_b1024_step25`
  used argument-span loss weight `1.0`, value-span loss weight `2.0`,
  `VALUE_SPAN_MASK_PROB=1.0`, LR `5e-6`, and block size `1024`. It trained
  cleanly to loss `2.0682`, but public one-call raw regressed to `2/8` /
  `1/8`, teacher-train raw regressed to `1/12` / `1/12`, and teacher-heldout
  constrained exact sequence / arguments dropped to `7/8` / `5/8`. Keep the
  hook as infrastructure; do not scale this pressure setting.
- value-span label-only objective is now implemented:
  `FASTDLLM_VALUE_SPAN_LABEL_ONLY` in the local model bridge and
  `VALUE_SPAN_LABEL_ONLY` in the launcher. It keeps full tool-call context but
  drops non-value assistant labels before MDM masking. The one-step gate
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_from_ckpt275_b1024_step1`
  used neutral argument/value loss weights, `VALUE_SPAN_LABEL_ONLY=1`,
  `VALUE_SPAN_MASK_PROB=1.0`, LR `5e-6`, and block size `1024`. The first debug
  batch went from `55` assistant labels to `17` value-only labels, forced those
  `17`, and trained to loss `0.7236`. It tied active grounded one-call metrics
  on public, teacher-train, and teacher-heldout slices, so it is not promoted
  yet but is safer than the 25-step value-mask pressure run.
- value-span label-only 25-step continuation is complete and mixed:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_from_ckpt275_b1024_step25`
  trained cleanly to loss `0.3459` with the same label-only objective. Public
  one-call raw tied active (`3/8` / `2/8`), but constrained exact arguments
  regressed to `7/8`; teacher-train raw regressed to `1/12` / `1/12`, while
  constrained sequence improved to `11/12`; teacher-heldout raw improved to
  `2/8` / `1/8` while constrained stayed `8/8` / `6/8`. Do not promote the
  step-25 adapter. Keep the hook and next test lower update pressure or earlier
  checkpoints.
- value-span label-only 5-step continuation is also negative:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_from_ckpt275_b1024_step5`
  trained cleanly to loss `0.4866`, but public one-call raw regressed to
  `2/8` / `1/8`, teacher-train raw regressed to `1/12` / `1/12`, and
  teacher-heldout constrained exact sequence / arguments dropped to `7/8` /
  `5/8`. Do not continue full-strength `VALUE_SPAN_MASK_PROB=1.0` step sweeps;
  lower update pressure or move to teacher-KL/span distillation.
- value-span label-only lower-LR dose curve is complete:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_from_ckpt275_b1024_lr1e6_step100`
  trained from active checkpoint-275 at LR `1e-6`, block `1024`, `100` steps,
  and train loss `0.3614`. The launcher now exposes `SAVE_STEPS` and
  `SAVE_TOTAL_LIMIT`, and the planned `25/50/75/100` checkpoints were archived
  under
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_from_ckpt275_b1024_lr1e6_step100_checkpoint_archive`.
  The one-call sweep
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_from_ckpt275_b1024_lr1e6_step100_checkpoint_sweep_eval96_modelrepair_max1_onecall`
  shows checkpoint `75` as the best dose point: raw metrics tie active on
  public (`3/8` / `2/8`), teacher-train (`2/12` / `2/12`), and heldout (`1/8` /
  `0/8`), while constrained teacher-train improves to `11/12` sequence and
  `7/12` arguments and public/heldout constrained top lines remain tied.
  Checkpoint `50` also ties the active constrained top line and improves
  model-repair arguments, but checkpoint `100` shows formatting drift in raw
  valid JSON (`1/8` public, `0/12` teacher-train). Do not promote yet because no
  raw/model-only metric improves; use checkpoint `75` as the current scaling
  candidate for broader eval or a larger teacher/grounded data slice.
- synthetic-48 value-span label-only scaling probe is complete:
  active checkpoint-275 was first evaluated on 48 synthetic one-call cases,
  reaching raw exact sequence / arguments `16/48` / `11/48`, constrained
  `48/48` / `44/48`, and model-repair `25/48` / `25/48`. The grounded builder
  accepted `44` exact constrained rows into
  `data/qwen35_9b_toolcall_grounded_spanfill_synthetic_onecall48_b1024_curriculum`
  with full label retention. A 75-step continuation from active checkpoint-275
  at LR `1e-6`, block `1024`, and `VALUE_SPAN_LABEL_ONLY=1` trained to loss
  `0.3350` in
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_synth48_from_ckpt275_b1024_lr1e6_step75`.
  Checkpoint `75` is not promoted because teacher-heldout constrained exact
  sequence / arguments regressed to `7/8` / `5/8`. It is still the first branch
  to improve raw/model-only metrics: public one-call rises to `4/8` sequence and
  `3/8` arguments, and teacher-heldout rises to `2/8` and `1/8`. This makes
  scaling clean grounded value-span data the current main training direction,
  but only with an explicit preservation guard.
- synthetic-48 checkpoint-50 follow-up is complete and also not promoted:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_synth48_from_ckpt275_b1024_lr1e6_step75_checkpoint50_eval96_modelrepair_max1_onecall`
  ties active raw on public (`3/8` / `2/8`) and teacher-train (`2/12` / `2/12`)
  and preserves public constrained `8/8` / `8/8`, but teacher-heldout constrained
  still regresses to `7/8` / `5/8`. It also loses the checkpoint-75 raw gains.
  The next branch should not be a single-checkpoint promotion attempt; it should
  train a replay/preservation mix of synthetic-48 rows plus the original grounded
  teacher-train rows.
- synthetic-48 plus teacher-train replay mix is complete and not promoted:
  `scripts/build_toolcall_grounded_replay_mix.py` built
  `data/qwen35_9b_toolcall_grounded_spanfill_synth48_replay_teacher2_b1024_curriculum`
  with `44` synthetic rows plus `32` repeated teacher-train replay rows and no
  label-loss rows. A 100-step LR-`1e-6` value-span label-only continuation from
  active checkpoint-275 trained to loss `0.3505` in
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_synth48_replay_teacher2_from_ckpt275_b1024_lr1e6_step100`.
  Checkpoint `50` keeps raw gains on public (`4/8` / `3/8`) and heldout
  (`2/8` / `1/8`) but regresses constrained public args to `7/8` and heldout to
  `7/8` / `5/8`. Checkpoint `100` restores heldout constrained to active
  `8/8` / `6/8` and improves teacher-train constrained args to `7/12`, but raw
  metrics regress. The active checkpoint remains checkpoint-275. The next trainer
  branch needs explicit retention/anti-regression pressure or a split
  generator/repair design rather than more simple CE replay.
- staged retention from the replay-mix raw-gain checkpoint is complete:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_staged_retention_from_synth48replay_ckpt50_b1024_lr5e7_step50`
  starts from replay-mix checkpoint `50`, trains only on the original grounded
  teacher-train b1024 retention rows at LR `5e-7`, and was swept at checkpoints
  `24`, `40`, and `50`. Checkpoint `24` is the first broader-eval candidate:
  public raw improves over active checkpoint-275 from `3/8` / `2/8` to `4/8` /
  `3/8`, public constrained stays `8/8` / `8/8`, teacher-heldout constrained
  stays active `8/8` / `6/8`, and teacher-train constrained sequence improves
  from `10/12` to `11/12` while arguments tie at `6/12`. It is not active yet:
  raw valid JSON remains weak (`1/8` public, `0/12` teacher-train), and it still
  needs public multi-call/tool-result eval. Checkpoints `40` and `50` show
  overshoot: checkpoint `50` preserves/improves constrained projection but
  collapses raw public to `1/8` / `0/8` and raw teacher-train to `0/12` /
  `0/12`.
- staged checkpoint-24 broad agentic eval is complete:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_staged_retention_ckpt24_broad_eval96_modelrepair_agentic`.
  It confirms the one-call gain and heldout constrained guard, but it is still
  not a global replacement for active checkpoint-275. Public multi-call guarded
  sequence-planner projection reaches `11/12` sequence and `9/12` arguments
  versus active `11/12` / `10/12`. Text-compatible synthetic tool-result
  constrained recovery reaches `10/10` / `9/10` versus active `10/10` / `8/10`,
  but OpenAI-style tool-result constrained recovery is `10/10` / `8/10` versus
  active `10/10` / `9/10`. Treat checkpoint `24` as the seed for a gentler
  anti-regression scaling run that includes OpenAI-style tool-result retention,
  not as the new active checkpoint.
- checkpoint-24 anti-regression mix is complete and not promoted:
  `scripts/build_toolcall_checkpoint24_antiregression_mix.py` built
  `data/qwen35_9b_toolcall_checkpoint24_antiregression_b1024_curriculum` with
  `127` fully labeled rows: `44` synthetic grounded span-fill, `32`
  teacher-train grounded retention, `21` sequence-planner compact retention,
  `10` text tool-result retention, and `20` native OpenAI-style tool-result
  retention. `scripts/fastdllm_value_copy_token_ids.py` now extracts value
  tokens from native `assistant.tool_calls` rows. A low-LR `2e-7` continuation
  from staged checkpoint `24` trained in
  `runs/fastdllm_qwen35_9b_toolcall_checkpoint24_antiregression_mix_from_staged24_b1024_lr2e7_step80`.
  The one-call sweep was stopped after checkpoints `10`, `20`, and `40`.
  Checkpoints `10` and `20` fall back to public raw `3/8` / `2/8`, teacher-train
  constrained `10/12` / `6/12`, and heldout constrained `8/8` / `6/8`.
  Checkpoint `40` improves teacher-train constrained to `11/12` / `7/12`, but
  public raw remains `3/8` / `2/8` and heldout constrained regresses to `7/8` /
  `5/8`. Direct broad anti-regression in the same generator adapter is therefore
  a negative scaling direction; keep staged checkpoint `24` as the generator seed
  and move protection into repair/projection or sidecar routing.
- split-route sidecar scorecard is complete:
  `qwen35_9b_split_route_sidecar_scorecard.md` records the best current routed
  target from existing artifacts. The same writer emits executable gate evidence
  at `runs/qwen35_9b_split_route_sidecar_eval/route_scorecard.json` and
  `runs/qwen35_9b_split_route_sidecar_eval/route_scorecard.tsv`, plus
  `runs/qwen35_9b_split_route_sidecar_eval/route_manifest.json` for the first
  runtime-router implementation. The manifest records the base model, adapter
  roles, input case files, routed summaries, and post-processing chains.
  `--check` exits nonzero if any route gate regresses. The current route verdict
  is `PASS` across all six slices. It routes one-call and text tool-result
  slices to staged checkpoint `24`, preserving public raw `4/8` sequence and
  `3/8` arguments plus text tool-result protected `10/10` / `9/10`;
  it routes multi-call and OpenAI-style tool-result slices through active
  checkpoint-275, preserving `11/12` / `10/12` multi-call and `10/10` / `9/10`
  OpenAI-style tool-result. This is an implementation target for a
  router/sidecar, not a newly trained or promoted checkpoint.
- split-route replay runner is initialized:
  `scripts/run_qwen35_split_route_sidecar_manifest.py` validates the manifest
  and emits `runs/qwen35_9b_split_route_sidecar_eval/replay_plan/route_runner_plan.json`
  plus `runs/qwen35_9b_split_route_sidecar_eval/replay_plan/route_runner_plan.sh`.
  Current validation is `6` routes, `10` replayable steps, and `0` unknown
  post-processing steps. The generation commands are wrapped in user
  `systemd-run` memory scopes, so a full replay can be launched from the shell
  plan when the 5090 is available. The same runner now verifies replay outputs
  with `--verify-outputs --plan-json <plan.json>`; the historical-output
  verification artifact
  `runs/qwen35_9b_split_route_sidecar_eval/historical_verify_plan/route_runner_plan_verification.json`
  passes all six route gates with `0` missing summaries and `0` failed records.
- partial live replay is verified:
  `scripts/run_qwen35_split_route_sidecar_manifest.py --slice public_one_call --execute`
  regenerated the public one-call staged checkpoint-24 lane under
  `runs/qwen35_9b_split_route_sidecar_eval/live_smoke_public_onecall/`.
  The verifier passed with raw `4/8` sequence, raw `3/8` arguments, and
  protected `8/8` / `8/8`. This confirms the replay runner can regenerate and
  gate a live routed lane; the remaining routes can be replayed when we are
  ready to spend the additional model-load time.
- active-protection live replay is verified:
  `scripts/run_qwen35_split_route_sidecar_manifest.py --slice openai_style_tool_result --execute`
  regenerated the OpenAI-style tool-result route under
  `runs/qwen35_9b_split_route_sidecar_eval/live_smoke_openai_toolresult/`.
  The verifier passed with raw `6/10` sequence, raw `6/10` arguments, and
  protected `10/10` sequence / `9/10` arguments. This confirms both split-route
  adapter roles can be regenerated and gated live.
- public multi-call protection-chain live replay is verified:
  `scripts/run_qwen35_split_route_sidecar_manifest.py --slice public_multi_call_planner --execute`
  regenerated the active checkpoint-275 multi-call route under
  `runs/qwen35_9b_split_route_sidecar_eval/live_smoke_public_multicall_planner/`.
  The verifier passed with protected `11/12` sequence and `10/12` arguments
  after generation, sequence-preserving rescore, contextual projection, and
  sequence-planner projection. This confirms the longest current protection
  chain can be regenerated and gated live.
- staged text tool-result live replay is verified:
  `scripts/run_qwen35_split_route_sidecar_manifest.py --slice synthetic_text_tool_result --execute`
  regenerated the staged checkpoint-24 text tool-result route under
  `runs/qwen35_9b_split_route_sidecar_eval/live_smoke_synthetic_text_toolresult/`.
  The verifier passed with raw `6/10` sequence, raw `4/10` arguments, and
  protected `10/10` sequence / `9/10` arguments. This confirms staged
  checkpoint-24 is live-gated on both one-call and text tool-result lanes.
- teacher one-call live replay is verified:
  `scripts/run_qwen35_split_route_sidecar_manifest.py --slice teacher_train_one_call --slice teacher_heldout_one_call --execute`
  regenerated the remaining staged checkpoint-24 one-call lanes under
  `runs/qwen35_9b_split_route_sidecar_eval/live_smoke_teacher_onecall/`.
  Teacher-train passed protected `11/12` sequence / `6/12` arguments.
  Teacher-heldout passed raw `2/8` sequence, raw `1/8` arguments, and protected
  `8/8` sequence / `6/8` arguments. All `6` split-route scorecard lanes now
  have verified live replay artifacts.
- evidence policy is explicit: `1/5/25`-step runs are instrumentation and
  regression gates, not final proof that a training mechanism is dead. Promotion
  requires a raw/model-only gain plus no regression on protected
  constrained/projected metrics. The scaling context is tracked in
  `qwen35_9b_grounded_spanfill_curriculum_result.md`.
- the first low-ratio sequence-planner replay mix should not be promoted:
  `scripts/build_toolcall_modelrepair_sequence_planner_mix.py` built a
  240-row mix with 227 model-repair rows plus 13 sequence-planner rows, all
  fully labeled at block size `896`. A 100-step continuation from active
  checkpoint-275 trained cleanly with train loss `1.738793797492981`, but
  checkpoint-100 regressed the active public multi-call contextual projection
  from `7/12` sequence and `7/12` arguments to `5/12` and `4/12`, regressed
  guarded sequence-planner projection from `11/12` and `10/12` to `7/12` and
  `5/12`, and dropped OpenAI tool-result constrained exact arguments from
  `9/10` to `5/10`. Result note:
  `qwen35_9b_modelrepair_sequence_planner_mix_result.md`.

## Early Success Metrics

The first trained 9B diffusion checkpoint should be evaluated on the same gates
already used for AR Qwen3.5 and Qwen3.6 teacher:

- synthetic one-call exact sequence / exact arguments / schema pass
- public Hermes one-call exact sequence / exact arguments / schema pass
- public Hermes multi-call exact sequence / repeated-call rate
- synthetic tool-result exact next action / exact arguments
- tiny repo-edit patch/test-pass slice when the sampler can drive an agent loop
- unresolved mask count
- stop-boundary failures
- tokens/sec

Minimum early signal:

- beat the 1.5B diffusion baseline on strict tool-call metrics
- avoid repeated-call collapse on the multi-call slice
- show some nonzero strict hits on held-out tool-call cases
- produce a checkpoint summary comparable to `qwen35_9b_ar_baseline_result.md`
  and `qwen25_1p5b_diffusion_baseline_result.md`
