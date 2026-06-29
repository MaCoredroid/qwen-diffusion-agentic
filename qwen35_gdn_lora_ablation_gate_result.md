# Qwen3.5 GDN LoRA Ablation Gate Result

Date: 2026-06-27.

## Summary

This is the first architecture-ablation gate after deciding that Qwen3.5/Qwen3.6
GDN diffusion needs its own research track rather than direct Qwen2.5 recipe
transfer.

Goal: verify that the local Qwen3.5 Fast-DLLM bridge can train separate LoRA
families from the base candidate without loading the active mixed checkpoint-275
adapter, run a tiny comparable continuation to see whether the target family
affects early tool-call behavior, then test early GDN state-handling variants.

This is an ablation signal, not a promoted quality result.

## One-step Runability Gate

All three runs used:

```text
base model: models/qwen3.5-9b-fastdllm-init
starting adapter: none
dataset: data/qwen35_9b_toolcall_model_repair_curriculum
block size: 896
max train samples: 16
max steps: 1
LoRA r: 8
LoRA alpha: 16
LoRA dropout: 0.05
argument-span loss weight: 1.5
systemd cgroup: MemoryMax=28G, MemorySwapMax=4G
```

| Branch | Target modules | Trainable params | Train loss | Runtime |
| --- | --- | ---: | ---: | ---: |
| GDN-only | `in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj` | 7,090,176 | 9.102045059204102 | 2.7605s |
| attention-only | `q_proj,k_proj,v_proj,o_proj` | 1,966,080 | 9.102045059204102 | 2.6489s |
| mixed | `q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj` | 9,056,256 | 9.102045059204102 | 2.8296s |

Outputs:

```text
runs/fastdllm_qwen35_9b_gdnonly_lora_modelrepair_b896_step1_gate
runs/fastdllm_qwen35_9b_attentiononly_lora_modelrepair_b896_step1_gate
runs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_step1_gate
logs/fastdllm_qwen35_9b_gdnonly_lora_modelrepair_b896_step1_gate.log
logs/fastdllm_qwen35_9b_attentiononly_lora_modelrepair_b896_step1_gate.log
logs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_step1_gate.log
```

## 25-step Branch Gate

Shared setup:

```text
base model: models/qwen3.5-9b-fastdllm-init
starting adapter: none
dataset: data/qwen35_9b_toolcall_model_repair_curriculum
block size: 896
max train samples: 64
max steps: 25
LoRA r: 8
LoRA alpha: 16
LoRA dropout: 0.05
argument-span loss weight: 1.5
systemd cgroup: MemoryMax=28G, MemorySwapMax=4G
```

Training results:

| Branch | Output | Train loss | Runtime | Steps/s |
| --- | --- | ---: | ---: | ---: |
| GDN-only | `runs/fastdllm_qwen35_9b_gdnonly_lora_modelrepair_b896_step25` | 9.019701690673829 | 53.3787s | 0.468 |
| attention-only | `runs/fastdllm_qwen35_9b_attentiononly_lora_modelrepair_b896_step25` | 8.578798828125 | 51.7646s | 0.483 |
| mixed | `runs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_step25` | 7.927771911621094 | 54.2009s | 0.461 |

Cheap public one-call gate:

```text
eval: data/toolcall_eval/public_onecall_hermes_smoke.jsonl, first 8 rows
adapter loading: no merge
conversation template: fast_dllm_v2
block size: 32
small block size: 8
max new tokens: 96
sampling: full-context
repair mode: schema
constrained tool decoding: enabled, max calls 1
model-repair pass: disabled
```

| Branch | Raw valid | Raw exact seq | Raw exact args | Constrained valid | Constrained exact seq | Constrained exact args | Repeated rows | Tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GDN-only | 1/8 | 1/8 | 0/8 | 6/8 | 6/8 | 1/8 | 0 | 8.99 |
| attention-only | 1/8 | 1/8 | 0/8 | 7/8 | 7/8 | 1/8 | 0 | 8.91 |
| mixed | 2/8 | 2/8 | 0/8 | 6/8 | 6/8 | 1/8 | 0 | 8.72 |

Eval outputs:

```text
runs/qwen35_gdn_lora_ablation_public_onecall_gate/gdnonly_public_onecall_8.summary.json
runs/qwen35_gdn_lora_ablation_public_onecall_gate/attentiononly_public_onecall_8.summary.json
runs/qwen35_gdn_lora_ablation_public_onecall_gate/mixed_public_onecall_8.summary.json
logs/qwen35_gdn_lora_ablation_gdnonly_public_onecall_gate.log
logs/qwen35_gdn_lora_ablation_attentiononly_public_onecall_gate.log
logs/qwen35_gdn_lora_ablation_mixed_public_onecall_gate.log
```

## Noisy-block GDN State-isolation Probe

I added an environment-selectable training mode in
`models/qwen3.5-9b-fastdllm-init/modeling.py`:

```text
FAST_DLLM_QWEN3_5_GDN_MODE=option_a_noisy_block_isolation_v0
```

Default mode remains `option_a_causal_gdn_v0`. The new mode only changes
training forwards with an MDM noisy/clean split:

- noisy `x_t` half: each diffusion block is run through GDN independently, so
  recurrent state resets at block boundaries;
- clean `x_0` half: still runs through causal GDN;
- generation/eval forwards without `mdm_split_size` remain causal and
  comparable to the existing adapters.

Run setup matched the mixed target-family branch:

```text
target modules: q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj
base model: models/qwen3.5-9b-fastdllm-init
starting adapter: none
dataset: data/qwen35_9b_toolcall_model_repair_curriculum
block size: 896
LoRA r: 8
LoRA alpha: 16
LoRA dropout: 0.05
argument-span loss weight: 1.5
systemd cgroup: MemoryMax=28G, MemorySwapMax=4G
```

Training results:

| Branch | GDN mode | Output | Train loss | Runtime | Steps/s |
| --- | --- | --- | ---: | ---: | ---: |
| mixed state-isolation smoke | `option_a_noisy_block_isolation_v0` | `runs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_noisyblockiso_step1_gate` | 11.066727638244629 | 2.8837s | 0.347 |
| mixed state-isolation 25-step | `option_a_noisy_block_isolation_v0` | `runs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_noisyblockiso_step25` | 10.90037857055664 | 56.1298s | 0.445 |

Cheap public one-call gate for the 25-step adapter:

| Branch | Raw valid | Raw exact seq | Raw exact args | Constrained valid | Constrained exact seq | Constrained exact args | Repeated rows | Tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mixed state-isolation | 1/8 | 2/8 | 0/8 | 7/8 | 7/8 | 1/8 | 0 | 9.18 |

Outputs:

```text
runs/qwen35_gdn_lora_ablation_public_onecall_gate/noisyblockiso_public_onecall_8.summary.json
runs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_noisyblockiso_step1_gate
runs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_noisyblockiso_step25
logs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_noisyblockiso_step1_gate.log
logs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_noisyblockiso_step25.log
logs/qwen35_gdn_lora_ablation_noisyblockiso_public_onecall_gate.log
```

## Clean-state GDN Injection Probe

I added a less destructive state-handling mode:

```text
FAST_DLLM_QWEN3_5_GDN_MODE=option_a_clean_state_injection_v0
```

The implementation extends `torch_chunk_gated_delta_rule` so the clean `x_0`
stream can expose recurrent state at diffusion-block boundaries. During
training forwards with an MDM noisy/clean split:

- clean `x_0` half: runs through GDN causally and records recurrent boundary
  states every diffusion block;
- noisy `x_t` half: each noisy block runs through GDN in a batched block call
  initialized from the previous clean block's recurrent state;
- clean boundary states are detached before injection into the noisy blocks to
  keep the gate memory-bounded;
- generation/eval forwards without `mdm_split_size` remain causal and
  comparable to the existing adapters.

Run setup matched the mixed target-family branch:

```text
target modules: q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj
base model: models/qwen3.5-9b-fastdllm-init
starting adapter: none
dataset: data/qwen35_9b_toolcall_model_repair_curriculum
block size: 896
LoRA r: 8
LoRA alpha: 16
LoRA dropout: 0.05
argument-span loss weight: 1.5
systemd cgroup: MemoryMax=28G, MemorySwapMax=4G
```

Training results:

| Branch | GDN mode | Output | Train loss | Runtime | Steps/s |
| --- | --- | --- | ---: | ---: | ---: |
| mixed clean-state smoke | `option_a_clean_state_injection_v0` | `runs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_cleanstate_step1_gate` | 7.392234802246094 | 2.6616s | 0.376 |
| mixed clean-state 25-step | `option_a_clean_state_injection_v0` | `runs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_cleanstate_step25` | 8.021231079101563 | 50.5230s | 0.495 |

Cheap public one-call gate for the 25-step adapter:

| Branch | Raw valid | Raw exact seq | Raw exact args | Constrained valid | Constrained exact seq | Constrained exact args | Repeated rows | Tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mixed clean-state | 1/8 | 1/8 | 0/8 | 7/8 | 7/8 | 1/8 | 0 | 8.72 |

Outputs:

```text
runs/qwen35_gdn_lora_ablation_public_onecall_gate/cleanstate_public_onecall_8.summary.json
runs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_cleanstate_step1_gate
runs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_cleanstate_step25
logs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_cleanstate_step1_gate.log
logs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_cleanstate_step25.log
logs/qwen35_gdn_lora_ablation_cleanstate_public_onecall_gate.log
```

## Clean-state Plus Mild Structural Objective Probe

This probe tested the next objective-pairing idea without changing the
clean-state GDN path:

```text
FAST_DLLM_QWEN3_5_GDN_MODE=option_a_clean_state_injection_v0
STRUCTURAL_LOSS_WEIGHT=1.25
ARGUMENT_SPAN_LOSS_WEIGHT=1.5
```

The intent was to add mild pressure on tool-call delimiters and JSON punctuation
while keeping the active balanced argument-span weight.

Run setup matched the clean-state branch:

```text
target modules: q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj
base model: models/qwen3.5-9b-fastdllm-init
starting adapter: none
dataset: data/qwen35_9b_toolcall_model_repair_curriculum
block size: 896
LoRA r: 8
LoRA alpha: 16
LoRA dropout: 0.05
systemd cgroup: MemoryMax=28G, MemorySwapMax=4G
```

Training results:

| Branch | GDN mode | Objective | Output | Train loss | Runtime | Steps/s |
| --- | --- | --- | --- | ---: | ---: | ---: |
| clean-state structural smoke | `option_a_clean_state_injection_v0` | structural `1.25`, argument-span `1.5` | `runs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_cleanstate_structw1p25_argspanw1p5_step1_gate` | 7.682875156402588 | 2.7057s | 0.370 |
| clean-state structural 25-step | `option_a_clean_state_injection_v0` | structural `1.25`, argument-span `1.5` | `runs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_cleanstate_structw1p25_argspanw1p5_step25` | 8.220419921875 | 50.5339s | 0.495 |

Cheap public one-call gate for the 25-step adapter:

| Branch | Raw valid | Raw exact seq | Raw exact args | Constrained valid | Constrained exact seq | Constrained exact args | Repeated rows | Tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-state structural `1.25` + argspan `1.5` | 1/8 | 1/8 | 0/8 | 7/8 | 7/8 | 0/8 | 0 | 8.54 |

Outputs:

```text
runs/qwen35_gdn_lora_ablation_public_onecall_gate/cleanstate_structw1p25_argspanw1p5_public_onecall_8.summary.json
runs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_cleanstate_structw1p25_argspanw1p5_step1_gate
runs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_cleanstate_structw1p25_argspanw1p5_step25
logs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_cleanstate_structw1p25_argspanw1p5_step1_gate.log
logs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_cleanstate_structw1p25_argspanw1p5_step25.log
logs/qwen35_gdn_lora_ablation_cleanstate_structw1p25_argspanw1p5_public_onecall_gate.log
```

## Clean-state Local Dual-pass GDN Probe

I added a local dual-pass GDN training mode:

```text
FAST_DLLM_QWEN3_5_GDN_MODE=option_a_clean_state_dualpass_v0
```

During training forwards with an MDM noisy/clean split:

- clean `x_0` half: runs through GDN causally and records recurrent boundary
  states every diffusion block;
- noisy `x_t` half, forward pass: each noisy block runs from the previous clean
  boundary state;
- noisy `x_t` half, reverse pass: the same noisy block is reversed, run through
  GDN with local zero state, flipped back, and averaged with the forward output;
- generation/eval forwards without `mdm_split_size` remain causal and
  comparable to the existing adapters.

Run setup matched the clean-state branch:

```text
target modules: q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj
base model: models/qwen3.5-9b-fastdllm-init
starting adapter: none
dataset: data/qwen35_9b_toolcall_model_repair_curriculum
block size: 896
LoRA r: 8
LoRA alpha: 16
LoRA dropout: 0.05
argument-span loss weight: 1.5
systemd cgroup: MemoryMax=28G, MemorySwapMax=4G
```

Training results:

| Branch | GDN mode | Output | Train loss | Runtime | Steps/s |
| --- | --- | --- | ---: | ---: | ---: |
| clean-state dual-pass smoke | `option_a_clean_state_dualpass_v0` | `runs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_cleanstate_dualpass_step1_gate` | 8.305818557739258 | 3.0527s | 0.328 |
| clean-state dual-pass 25-step | `option_a_clean_state_dualpass_v0` | `runs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_cleanstate_dualpass_step25` | 8.90849624633789 | 59.6257s | 0.419 |

Cheap public one-call gate for the 25-step adapter:

| Branch | Raw valid | Raw exact seq | Raw exact args | Constrained valid | Constrained exact seq | Constrained exact args | Repeated rows | Tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-state dual-pass | 0/8 | 0/8 | 0/8 | 6/8 | 6/8 | 1/8 | 0 | 8.60 |

Outputs:

```text
runs/qwen35_gdn_lora_ablation_public_onecall_gate/cleanstate_dualpass_public_onecall_8.summary.json
runs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_cleanstate_dualpass_step1_gate
runs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_cleanstate_dualpass_step25
logs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_cleanstate_dualpass_step1_gate.log
logs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_cleanstate_dualpass_step25.log
logs/qwen35_gdn_lora_ablation_cleanstate_dualpass_public_onecall_gate.log
```

## Clean-state Value-copy Objective Probe

I added an opt-in value-copy loss hook:

```text
VALUE_COPY_LOSS_WEIGHT=2.0
ARGUMENT_SPAN_LOSS_WEIGHT=1.5
FAST_DLLM_QWEN3_5_GDN_MODE=option_a_clean_state_injection_v0
```

The launcher now derives `FASTDLLM_VALUE_COPY_TOKEN_IDS` with
`scripts/fastdllm_value_copy_token_ids.py`. The script parses assistant
`<tool_call>` JSON from the selected curriculum, extracts scalar argument
values only, tokenizes those values, and writes a manifest.

Value-copy manifest for this run:

```text
tool calls: 227
scalar values: 869
unique scalar values: 241
value token IDs: 410
```

The one-step debug hook confirmed that the first training batch had `15`
value-copy labels and `40` total weighted labels.

Run setup matched the clean-state branch:

```text
target modules: q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj
base model: models/qwen3.5-9b-fastdllm-init
starting adapter: none
dataset: data/qwen35_9b_toolcall_model_repair_curriculum
block size: 896
LoRA r: 8
LoRA alpha: 16
LoRA dropout: 0.05
value-copy loss weight: 2.0
argument-span loss weight: 1.5
systemd cgroup: MemoryMax=28G, MemorySwapMax=4G
```

Training results:

| Branch | GDN mode | Objective | Output | Train loss | Runtime | Steps/s |
| --- | --- | --- | --- | ---: | ---: | ---: |
| clean-state value-copy smoke | `option_a_clean_state_injection_v0` | value-copy `2.0`, argument-span `1.5` | `runs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_cleanstate_valuecopyw2_argspanw1p5_step1_gate` | 8.166513442993164 | 2.7501s | 0.364 |
| clean-state value-copy 25-step | `option_a_clean_state_injection_v0` | value-copy `2.0`, argument-span `1.5` | `runs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_cleanstate_valuecopyw2_argspanw1p5_step25` | 8.887884674072266 | 50.5647s | 0.494 |

Cheap public one-call gate for the 25-step adapter:

| Branch | Raw valid | Raw exact seq | Raw exact args | Constrained valid | Constrained exact seq | Constrained exact args | Repaired exact args | Repeated rows | Tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-state value-copy `2.0` + argspan `1.5` | 1/8 | 0/8 | 0/8 | 7/8 | 7/8 | 1/8 | 1/8 | 0 | 8.85 |

Outputs:

```text
runs/qwen35_gdn_lora_ablation_public_onecall_gate/cleanstate_valuecopyw2_argspanw1p5_public_onecall_8.summary.json
runs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_cleanstate_valuecopyw2_argspanw1p5_step1_gate
runs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_cleanstate_valuecopyw2_argspanw1p5_step25
logs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_cleanstate_valuecopyw2_argspanw1p5_step1_gate.log
logs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_cleanstate_valuecopyw2_argspanw1p5_step25.log
logs/qwen35_gdn_lora_ablation_cleanstate_valuecopyw2_argspanw1p5_public_onecall_gate.log
```

## Clean-state Aligned Value-span Objective Probe

I added a second opt-in value objective that reuses the scalar value token IDs,
but only applies them while the label cursor is inside the derived
`arguments ... </tool_call>` span:

```text
VALUE_SPAN_LOSS_WEIGHT=2.0
ARGUMENT_SPAN_LOSS_WEIGHT=1.5
VALUE_COPY_LOSS_WEIGHT=1.0
FAST_DLLM_QWEN3_5_GDN_MODE=option_a_clean_state_injection_v0
```

This is narrower than global value-copy weighting. The launcher now exposes
`VALUE_SPAN_LOSS_WEIGHT`, `VALUE_SPAN_TOKEN_IDS`, and
`VALUE_SPAN_TOKEN_MANIFEST`; when value-span weighting is enabled, it also
derives the argument start/end token IDs even if `ARGUMENT_SPAN_LOSS_WEIGHT`
is left at `1.0`.

Value-span manifest for this run matched the value-copy extraction:

```text
tool calls: 227
scalar values: 869
unique scalar values: 241
value token IDs: 410
```

The one-step debug hook confirmed that the first training batch had `41`
argument-span labels and `18` aligned value-span labels.

Run setup matched the clean-state branch:

```text
target modules: q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj
base model: models/qwen3.5-9b-fastdllm-init
starting adapter: none
dataset: data/qwen35_9b_toolcall_model_repair_curriculum
block size: 896
LoRA r: 8
LoRA alpha: 16
LoRA dropout: 0.05
value-span loss weight: 2.0
argument-span loss weight: 1.5
systemd cgroup: MemoryMax=28G, MemorySwapMax=4G
```

Training results:

| Branch | GDN mode | Objective | Output | Train loss | Runtime | Steps/s |
| --- | --- | --- | --- | ---: | ---: | ---: |
| clean-state value-span smoke | `option_a_clean_state_injection_v0` | value-span `2.0`, argument-span `1.5` | `runs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_cleanstate_valuespanw2_argspanw1p5_step1_gate` | 7.992103099822998 | 2.7127s | 0.369 |
| clean-state value-span 25-step | `option_a_clean_state_injection_v0` | value-span `2.0`, argument-span `1.5` | `runs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_cleanstate_valuespanw2_argspanw1p5_step25` | 8.148176574707032 | 50.5680s | 0.494 |

Cheap public one-call gate for the 25-step adapter:

| Branch | Raw valid | Raw exact seq | Raw exact args | Constrained valid | Constrained exact seq | Constrained exact args | Repaired exact args | Repeated rows | Tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-state value-span `2.0` + argspan `1.5` | 1/8 | 1/8 | 0/8 | 7/8 | 7/8 | 0/8 | 0/8 | 0 | 8.70 |

Failure pattern: constrained decoding still recovers the single tool name on
`7/8` cases, but argument values collapse to partial or wrong objects. Example:
for a thermostat scheduling case, constrained decoding produced
`{"thermostat_id": "thermo123"}` but omitted the required `schedule` array.

Decision: do not promote or scale this branch. It is better than global
value-copy on training loss (`8.1482` versus `8.8879`), but worse on exact
argument recovery (`0/8` constrained exact arguments versus `1/8`). The
infrastructure is still useful because it cleanly separates global scalar-token
weighting from argument-aligned weighting.

Outputs:

```text
runs/qwen35_gdn_lora_ablation_public_onecall_gate/cleanstate_valuespanw2_argspanw1p5_public_onecall_8.summary.json
runs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_cleanstate_valuespanw2_argspanw1p5_step1_gate
runs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_cleanstate_valuespanw2_argspanw1p5_step25
logs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_cleanstate_valuespanw2_argspanw1p5_step1_gate.log
logs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_cleanstate_valuespanw2_argspanw1p5_step25.log
logs/qwen35_gdn_lora_ablation_cleanstate_valuespanw2_argspanw1p5_public_onecall_gate.log
```

## Clean-state Argument-span Masking Objective Probe

I added an opt-in MDM corruption hook:

```text
ARGUMENT_SPAN_MASK_PROB=<0.0-1.0>
ARGUMENT_SPAN_LOSS_WEIGHT=1.5
FAST_DLLM_QWEN3_5_GDN_MODE=option_a_clean_state_injection_v0
```

The hook reuses the same `arguments ... </tool_call>` span boundaries as the
argument-span loss. Instead of only reweighting labels after random masking, it
forces a sampled subset of argument-span labels into the masked denoising branch.
`ARGUMENT_SPAN_MASK_PROB=1.0` means every argument-span label is masked together
in the main MDM branch; `0.5` means each argument-span label has a 50% forced
mask override on top of the normal diffusion mask.

One-step debug checks:

| Branch | Pre-MDM valid labels | Argument-span labels | Forced argument mask | Train loss |
| --- | ---: | ---: | ---: | ---: |
| clean-state arg-mask p=1.0 smoke | 55 | 41 | 41 | 12.820178985595703 |
| clean-state arg-mask p=0.5 smoke | 55 | 41 | 24 | 11.194357872009277 |
| checkpoint-275 arg-mask p=0.1 LR 5e-6 smoke | 55 | 41 | 5 | 2.8756701946258545 |

Training results:

| Branch | Starting adapter | Objective | Output | Train loss | Runtime | Steps/s |
| --- | --- | --- | --- | ---: | ---: | ---: |
| clean-state arg-mask p=1.0 25-step | none | arg-mask `1.0`, argspan `1.5` | `runs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_cleanstate_argmaskp1_argspanw1p5_step25` | 9.960129547119141 | 50.4887s | 0.495 |
| clean-state arg-mask p=0.5 25-step | none | arg-mask `0.5`, argspan `1.5` | `runs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_cleanstate_argmaskp05_argspanw1p5_step25` | 7.7338529968261716 | 50.5388s | 0.495 |
| checkpoint-275 arg-mask p=0.5 25-step | `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model` | arg-mask `0.5`, argspan `1.5` | `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_argmaskp05_step25` | 2.1003400611877443 | 50.5417s | 0.495 |
| checkpoint-275 arg-mask p=0.1 25-step | `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model` | arg-mask `0.1`, argspan `1.5`, LR `5e-6` | `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_argmaskp01_lr5e6_step25` | 2.2840756607055663 | 50.5133s | 0.495 |

Cheap public one-call gate:

| Branch | Raw valid | Raw exact seq | Raw exact args | Constrained valid | Constrained exact seq | Constrained exact args | Repaired exact args | Tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-state arg-mask p=1.0 | 0/8 | 0/8 | 0/8 | 6/8 | 6/8 | 0/8 | 0/8 | 8.88 |
| clean-state arg-mask p=0.5 | 1/8 | 1/8 | 0/8 | 7/8 | 7/8 | 1/8 | 0/8 | 9.08 |
| checkpoint-275 arg-mask p=0.5 continuation | 2/8 | 2/8 | 1/8 | 8/8 | 8/8 | 2/8 | 0/8 | 8.61 |
| checkpoint-275 arg-mask p=0.1 LR 5e-6 continuation | 1/8 | 2/8 | 1/8 | 7/8 | 7/8 | 3/8 | 0/8 | 8.63 |
| active checkpoint-275 reference | 2/8 | 3/8 | 2/8 | 8/8 | 8/8 | 5/8 | n/a | n/a |

Decision: keep the hook as useful infrastructure, but do not promote either
mask probability tested here. p=1.0 is too destructive. p=0.5 is behavior-neutral
from base and regresses the promoted checkpoint-275 public constrained exact
arguments from `5/8` to `2/8` after only 25 continuation steps at LR `3e-5`.
p=0.1 with LR `5e-6` is gentler and improves constrained exact arguments to
`3/8`, but it still misses the active `5/8` argument baseline and drops
constrained exact sequence from `8/8` to `7/8`. Move next to
teacher-KL/full-span targets rather than continuing to sweep hard mask forcing.

Outputs:

```text
runs/qwen35_gdn_lora_ablation_public_onecall_gate/cleanstate_argmaskp1_argspanw1p5_public_onecall_8.summary.json
runs/qwen35_gdn_lora_ablation_public_onecall_gate/cleanstate_argmaskp05_argspanw1p5_public_onecall_8.summary.json
runs/qwen35_gdn_lora_ablation_public_onecall_gate/ckpt275_argmaskp05_public_onecall_8.summary.json
runs/qwen35_gdn_lora_ablation_public_onecall_gate/ckpt275_argmaskp01_lr5e6_public_onecall_8.summary.json
runs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_cleanstate_argmaskp1_argspanw1p5_step25
runs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_cleanstate_argmaskp05_argspanw1p5_step25
runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_argmaskp05_step25
runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_argmaskp01_lr5e6_step25
logs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_cleanstate_argmaskp1_argspanw1p5_step25.log
logs/fastdllm_qwen35_9b_mixed_lora_modelrepair_b896_cleanstate_argmaskp05_argspanw1p5_step25.log
logs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_argmaskp05_step25.log
logs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_argmaskp01_lr5e6_step25.log
logs/qwen35_gdn_lora_ablation_cleanstate_argmaskp1_argspanw1p5_public_onecall_gate.log
logs/qwen35_gdn_lora_ablation_cleanstate_argmaskp05_argspanw1p5_public_onecall_gate.log
logs/qwen35_gdn_lora_ablation_ckpt275_argmaskp05_public_onecall_gate.log
logs/qwen35_gdn_lora_ablation_ckpt275_argmaskp01_lr5e6_public_onecall_gate.log
```

## Checkpoint-275 Hard Clean-repair Full-span Probe

After the hard-mask probes, I tested a smaller full-span/teacher-style replay
hypothesis from the active checkpoint instead of another base-start branch. The
curriculum builder used only a capped hard clean-repair slice:

```text
data/qwen35_9b_toolcall_model_repair_clean_hard24_curriculum
--clean-repair-cap 24
--clean-repair-variants missing_wrapper,wrong_arguments_key,truncated
--clean-repair-sources public_train_onecall,public_teacher_exact_onecall
--clean-repair-repeat 1
--repair-repeat 2
--block-size 896
```

Curriculum summary:

| Family | Rows |
| --- | ---: |
| original label-aware rows | 147 |
| prior model-repair rows | 80 |
| accepted clean-repair rows | 19 |
| total rows | 246 |

Continuation:

| Branch | Starting adapter | Objective | Output | Train loss | Runtime | Steps/s |
| --- | --- | --- | --- | ---: | ---: | ---: |
| checkpoint-275 hard clean-repair cap-24 25-step | `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model` | clean-state injection, argspan `1.5`, LR `5e-6`, no hard arg mask | `runs/fastdllm_qwen35_9b_toolcall_modelrepair_clean_hard24_argspanw1p5_b896_ckpt275_lr5e6_step25` | 2.0962688446044924 | 50.4933s | 0.495 |

Cheap public one-call gate:

| Branch | Raw valid | Raw exact seq | Raw exact args | Constrained valid | Constrained exact seq | Constrained exact args | Repaired exact args | Tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| checkpoint-275 hard clean-repair cap-24 continuation | 2/8 | 2/8 | 1/8 | 7/8 | 7/8 | 3/8 | 2/8 | 8.18 |
| active checkpoint-275 reference | 2/8 | 3/8 | 2/8 | 8/8 | 8/8 | 5/8 | n/a | n/a |

Decision: do not promote or extend this branch as-is. Even with a smaller,
harder clean-repair cap and LR `5e-6`, the continuation regressed active
checkpoint-275 on public one-call raw sequence, raw arguments, constrained
sequence, and constrained arguments. This confirms the earlier cap-80 result:
broad repair/full-span replay is still too destructive for the main generator
unless the target is changed to a less blunt teacher-KL or explicitly
span-local objective.

Outputs:

```text
data/qwen35_9b_toolcall_model_repair_clean_hard24_curriculum
runs/fastdllm_qwen35_9b_toolcall_modelrepair_clean_hard24_argspanw1p5_b896_ckpt275_lr5e6_step25
runs/qwen35_gdn_lora_ablation_public_onecall_gate/ckpt275_clean_hard24_lr5e6_public_onecall_8.summary.json
runs/qwen35_gdn_lora_ablation_public_onecall_gate/ckpt275_clean_hard24_lr5e6_public_onecall_8.jsonl
logs/qwen35_gdn_ckpt275_clean_hard24_lr5e6_step25_train.log
logs/qwen35_gdn_lora_ablation_ckpt275_clean_hard24_lr5e6_public_onecall_gate.log
```

## Interpretation

- GDN-only, attention-only, and mixed LoRA branches all instantiate, train, and
  save adapters successfully on the local RTX 5090.
- The identical one-step loss means this gate should not be used to rank
  quality. It only proves the ablation branches are technically runnable.
- GDN-only has much larger adapter capacity than attention-only because the
  Qwen3.5 target has 24 GDN layers and only 8 full-attention layers.
- At 25 steps, mixed has the best training loss and the best raw one-call
  sequence count, but it does not improve constrained argument exactness.
- Attention-only has the best constrained sequence count on this tiny gate, but
  it still has 0/8 raw exact arguments and only 1/8 constrained exact arguments.
- None of the base-start 25-step branches is close to the active checkpoint-275
  top line (`8/8` public one-call constrained sequence and `5/8` constrained
  exact arguments), so there is no promotion candidate.
- The noisy-block state-isolation probe is runnable and gives the same raw
  sequence count as mixed plus the same constrained sequence count as
  attention-only, but its loss is much worse and argument exactness does not
  move.
- Clean-state injection is a better state-handling hypothesis than hard reset:
  one-step loss improved from the mixed baseline's `9.1020` to `7.3922`, the
  25-step loss (`8.0212`) is close to mixed baseline (`7.9278`), and runtime is
  slightly better than the noisy-block reset probe.
- Clean-state injection did not improve the cheap behavior gate: it matched the
  attention-only constrained sequence count (`7/8`) but stayed at `0/8` raw
  exact arguments and `1/8` constrained exact arguments.
- Mild structural pressure on top of clean-state injection did not help:
  training loss regressed slightly versus clean-state-only (`8.2204` versus
  `8.0212`) and constrained exact arguments dropped from `1/8` to `0/8`.
- Local dual-pass GDN is runnable but harmful in this simple form: train loss
  regressed to `8.9085`, speed dropped to `0.419` steps/s, raw sequence fell to
  `0/8`, and constrained sequence fell to `6/8`.
- Value-copy weighting is implemented and correctly finds scalar argument-value
  labels, but the first weight-`2.0` gate does not improve model-only metrics:
  train loss regressed to `8.8879`, raw exact sequence dropped to `0/8`, and
  constrained exact arguments only matched clean-state-only at `1/8`.
- Aligned value-span weighting is also runnable and is a cleaner objective than
  global value-copy, but its first weight-`2.0` gate did not fix argument
  recovery: train loss was `8.1482`, raw exact sequence was `1/8`, and
  constrained exact arguments were `0/8`.
- Argument-span mask forcing is runnable and directly changes the MDM corruption
  pattern, but the first probabilities do not improve the promotion metric:
  p=1.0 is too hard, p=0.5 is behavior-neutral from base, and p=0.5 continuation
  from checkpoint-275 regresses public constrained exact arguments from `5/8` to
  `2/8`. Lowering to p=0.1 and LR `5e-6` recovers constrained exact arguments
  to `3/8`, but still misses the active baseline and loses one constrained
  sequence row.
- The useful signal is partial but narrowing: target-family choice alone does
  not solve GDN diffusion, hard noisy-block reset is too blunt, clean-state
  injection is viable, naive structural-token loss pairing is not the right
  argument-copy fix, reverse-pass averaging is too disruptive, token-ID value
  weighting alone is too blunt, and aggressive argument-span mask forcing damages
  the promoted adapter; even a lower-LR p=0.1 replay is not enough to preserve
  the active one-call top line.
- Smaller hard clean-repair/full-span replay from checkpoint-275 is also a
  negative main-generator result: it trains cleanly to loss `2.0963`, but
  public one-call constrained sequence/arguments fall to `7/8` / `3/8` versus
  active `8/8` / `5/8`. Treat repair rows as diagnostics or a separate repair
  adapter candidate, not as the next main-generator mix.

## Next Experiment

Do not promote any base-start 25-step branch as-is. The next GDN innovation
should be one of:

```text
1. teacher-KL/full-span reconstruction: keep
   `option_a_clean_state_injection_v0`, but move from token-ID weighting to
   teacher token distribution matching or explicit denoising of full JSON
   argument spans;
2. confidence-weighted/self-contrastive span denoising: only train on spans
   where teacher and schema parser agree, and downweight easy wrapper repair;
3. sampler-side innovation: keep the active adapter fixed and test constrained
   span fill, per-field extraction, or boundary-state cache changes before
   spending another long run on data replay.
```

Promotion is not based on training loss. A branch must tie or beat active
checkpoint-275 on the full-schema one-call, public multi-call projection, and
tool-result sweep, and improve at least one model-only/raw metric before it is
worth extending.
