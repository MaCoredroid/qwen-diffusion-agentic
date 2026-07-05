# Convert-after-RL — STEP 1 manifest: M_{t+1} merge (work-item #29)

Design: `convert_after_rl_design.md` §3 (commit `6f5d20f`). STEP 1 only: build the
diffusion-loadable merged base `M_{t+1} = merge_and_unload(init + RL-v2)` (mask token
248077 / bridge / `bd_size=32` intact) and pass the bit-exact merge sanity gate
(KILL-1 on failure). Steps 2–6 (fresh conversion, export, eval battery) are NOT run here.

## Result: gate PASS — KILL-1 NOT triggered

Bit-exact merge sanity gate (`merged == init + 2.0·(B@A)`, maxabs diff **0.0**),
on one GDN `in_proj_qkv` tensor and one attn `o_proj` tensor:

| probe tensor | shape | `merged − (init+2.0·B@A)` maxabs | LoRA delta maxabs | moved-off-init maxabs | scaling |
|---|---|---:|---:|---:|---:|
| `model.layers.0.linear_attn.in_proj_qkv` (GDN) | [8192, 4096] | **0.0** | 2.70754e-4 | 2.74658e-4 | 2.0 |
| `model.layers.3.self_attn.o_proj` (attn) | [4096, 4096] | **0.0** | 3.34638e-4 | 3.35693e-4 | 2.0 |

- `get_delta_weight` vs manual `2.0·(B@A)` (fp32) agree to ≤9.5e-7 (formula check).
- Config/bridge invariants (init + saved base): `mask_token_id==248077`,
  `bd_size==32`, `mask_token=="|<MASK>|"`, `has_weights==true` — all PASS.
- Merge math: `W += (α/r)·B@A`, α=32, r=16 ⇒ scale **2.0**; targets
  q/k/v/o_proj + in_proj_{qkv,z,a,b} + out_proj (matches export `lora_merge_count=152`).

### Merge arithmetic note (why CPU, why bit-exact)
The merge is run on **CPU** so the gate is deterministic and bit-exact. PEFT's
`get_delta_weight` returns an fp32 delta on the CPU path; `merge_and_unload` applies it
**in-place** (`weight.data += delta`), which rounds the sum back to bf16. The gate
replicates that exact in-place bf16 arithmetic ⇒ maxabs 0.0. (An out-of-place
`w_init + delta_fp32` would promote to fp32 and spuriously read ~2e-4; a CUDA bf16
recompute of `B@A` is non-deterministic at ~1 ulp. Neither is a merge fault — both are
measurement artifacts avoided by the CPU in-place path. This matches the export gate in
`real_diffusion_export_block_reconcile.md` §3a, which also read maxabs 0.0.)

## Artifacts

| role | path |
|---|---|
| merge script | `scripts/merge_adapter_into_fastdllm_candidate.py` |
| gate evidence (JSON) | `runs/convert_after_rl/step1_merge/merge_sanity_gate.json` |
| **M_{t+1} merged base (diffusion-loadable)** | `models/qwen3.5-9b-fastdllm-mtplus1-merged/` |
| init candidate (base of merge) | `models/qwen3.5-9b-fastdllm-init/` |
| RL-v2 adapter (merged in) | `runs/rl_multiturn_grpo_v2/from_selected_base_g4_step300/adapter_model/` |

## sha256 (provenance)

| file | sha256 |
|---|---|
| merge script | `0ea1d8c0a73599b5683ca34bd7b9fb3e95fed88e02933257fcb87e59fbb76f90` |
| RL-v2 `adapter_model.safetensors` | `c67f0a160c9dcb45baa886cf5f9ad1147dee5df1a98f4b8ae76101d3cd60c841` (matches design pin `c67f0a16…60c841`) |
| gate evidence JSON | `683116ee033a8a6d595b4e024c6562537beec3388f468222ccaa4c5dcc80a1ad` |
| merged `model-00001-of-00004.safetensors` | `f663baa25dc9278e26bb0d9c2faf1fbf9e63b91241eea0cda72030bf5163cc4e` |
| merged `model-00002-of-00004.safetensors` | `fb31fc6c348486eef2e6a269b1437a30a863dbf0621dba16e52ce15523661e04` |
| merged `model-00003-of-00004.safetensors` | `f009434332eec954721f288c572497494b67765733a0c6a234839582bcb3a5dd` |
| merged `model-00004-of-00004.safetensors` | `4a2cd38503362f51fde457827a9b581d4203ae8d9744d7758095c28b5c3c42a6` |
| merged `model.safetensors.index.json` | `f58e8bf82184ec0397c71ac0f33d060d5335fba771b2c9efb9a7c0a57f54c78d` |
| merged `config.json` | `bff9c5d4120377df7c2d860c1398fc864df7ce03968b37002542aa9b6644fa6e` |
| merged `conversion_manifest.json` | `25c7b793e2231b14a5cf034e13cfbbaccd8ada95f72ccdcb75a5a030d1e8b829` |

Merged base carries the init `conversion_manifest.json` forward + a `lineage` block
(derived_from_init, merged_adapter, merge formula, preserved-bridge note). Env:
`.venv-fastdllm`, peft 0.19.1, RAM-caged (`MemoryMax=22G`), CPU merge (~11 s wall).

## Not done in STEP 1 (next steps, per design)
- §3 decode-equivalence smoke (one HF hybrid-clean episode on the merged base, adapter
  absent, ≈3/4 exact) — optional confirmation the merged base == the serving weights.
- §4 STEP 2 fresh two-stream conversion → A_new; §5 export; §6 eval battery.
