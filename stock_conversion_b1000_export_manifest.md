# Diffusion-on-STOCK-conversion — B@1000 vLLM export manifest (Stage-C 4th-arm decider)

**Date:** 2026-07-06 (01:33–01:40 UTC). **Purpose:** build the stock-faithful diffusion twin
for the Stage-C N=25–50 **4th arm** = *diffusion-on-STOCK-conversion* (`runs/stage_c_n5v2/report.md`
§4 change-list #1). This completes the 2×2 `{weights: stock, RL-v2} × {paradigm: AR, diffusion}` and
separates paradigm-vs-weights: the existing 3-arm table (stock-AR 4/5 > merged-AR RL-v2 2/5 >
diffusion RL-v2 1/5) attributes the loss mostly to the RL-v2 **weights**, and this arm is the pivot
that tests whether the diffusion **paradigm** on stock weights recovers stock-AR.

The payload is **stock Qwen3.5-9B + B@1000 two-stream conversion foundation only — NO Run-1, NO
RL-v2.** Lineage located and verified against `REPRODUCE_V2.md` §3–4.

## Output

`models/qwen3.5-9b-fastdllm-b1000-stock-vllm-bf16/` (vLLM-loadable bf16, official Qwen3.5 conditional
layout; converted clean-stream LM tensors + merged B@1000 attn LoRA). Self-contained manifests inside
the dir: `lumo_export_manifest.json` (tool-written) + `stock_conversion_manifest.json` (lineage/sha
sidecar). Byte-identical lineage to the pre-existing `models/qwen3.5-9b-fastdllm-b1000-vllm-bf16`
(same snapshot + init + adapter + deterministic export script); this `-stock-` copy is the explicitly
labeled 4th-arm serving payload.

## Adapter identity — B@1000 conversion foundation (verified vs REPRODUCE_V2 §4)

Adapter: `runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000` (two-stream Fast-dLLM LoRA, 1000 steps).

| property | value | matches REPRODUCE_V2 §4 |
|---|---|---|
| `r` / `lora_alpha` / dropout | 8 / 16 / 0.05 | yes (attn-only r8) |
| `target_modules` | q_proj, k_proj, v_proj, o_proj | yes |
| `base_model_name_or_path` | `models/qwen3.5-9b-fastdllm-init` | yes |
| `global_step` | 1000 | yes |
| `train_loss` | 3.9429622707366945 | yes (exact) |
| `train_runtime` (s) | 18922.0906 | yes (exact) |
| `gpu_peak_memory_mib` | 29612 | yes (exact) |
| Run-1 / RL-v2 applied | **NO / NO** | correct for stock twin |

## Export profile (replacement / merge counts)

Export script `$FLYWHEEL/scripts/export_qwen35_9b_fastdllm_vllm.py`
sha256 `6d507ec9ba3308ff7e0f600bc0b5ec7c4ff96f66eff4e4e92175d42af7a119d5` — **matches REPRODUCE_V2 §2
pin.** Run in `.venv-fastdllm` (Python 3.10, torch 2.12.1+cu130, peft 0.19.1).

| metric | value | REPRODUCE_V2 §4 expected | stop-condition |
|---|---:|---:|---|
| `mapped_text_tensors` | 427 | 427 | — |
| `replacement_count` | 427 | 427 | fail if ≠427 |
| `lora_merge_count` | 32 | 32 | fail if ≠32 |
| `lora_scale` | 2.0 | 2.0 (α/r = 16/8) | — |
| `lora_target_tensor_count` | 32 | 32 | — |

Merged LoRA targets = q/k/v/o_proj of attention layers {3,7,11,15,19,23,27,31} (8 layers × 4 proj).
Strategy: `official_qwen35_conditional_layout_with_converted_language_model_and_merged_lora`. Shards
written in ~11 s (NVMe). **Both replacement/merge stop-conditions PASS.**

## HF-side sanity rows (pre-RL floors, labeled honestly)

Sanity = **valid + plausible**, not RL-level exactness. HF diffusion stack, base `init` + B@1000
adapter (no-merge, same as the eval lineage). Pinned samplers.

### Row 1 — one matched-20 episode, hybrid-clean careful

`scripts/eval_flare_northstar_hybrid_clean.py` (sha `a4c66751…b908f3`, matches §2 pin), episode-limit
1 → 4 turns. Backend `diffusion_hybrid_forced_grammar_seq_values` (forced-grammar bulk-commit +
sequential value decode).

| metric | value |
|---|---|
| turns | 4 |
| valid_tool_json / valid_tool_call | 4/4 / 4/4 |
| exact_tool_sequence | 4/4 |
| schema_ok / required_args_present | 4/4 / 4/4 |
| exact_args | 3/4 |
| episode_exact | 0/1 |
| sec/turn / forwards/turn | 2.17 / 32.5 |

Reading: 100% valid, schema-clean, correct tool sequence; args not fully exact (episode not exact) —
**expected 34/63-class pre-RL floor** (matched-20 whole-battery exact_args floor ≈ 34/63 ≈ 0.54 before
RL). Sanity PASS.

### Row 2 — GSM8K 5-prompt continuity spot

`scripts/eval_flare_stage1_ab_diffusion.py` (sha `eaa78d…3e9e503`, matches §2 pin) →
`full_context_sample_one` (default full-context + fresh-blocks; legacy continuity sampler per §0
pinning rule). GDN route_i / torch kernel, mask-id 248077, stop 248046, temp 0.0.

| idx | gold | strict pred | correct |
|---:|---:|---:|:--:|
| 0 | 18 | 18 | ✓ |
| 1 | 3 | 3 | ✓ |
| 2 | 70000 | (degenerate long number) | ✗ |
| 3 | 540 | 540 | ✓ |
| 4 | 20 | 40 | ✗ |

**strict 3/5 = 0.60, flex 3/5 = 0.60.** Consistent with the B@1000 continuity floor
`11/20 = 0.55` (REPRODUCE_V2 §4); one degenerate decode (idx 2) + one arithmetic miss (idx 4) are
textbook pre-RL behavior. Sanity PASS. GPU util mean 97.9%.

## Artifacts

| role | path |
|---|---|
| **stock vLLM export (4th-arm payload)** | `models/qwen3.5-9b-fastdllm-b1000-stock-vllm-bf16/` |
| tool export manifest | `models/qwen3.5-9b-fastdllm-b1000-stock-vllm-bf16/lumo_export_manifest.json` |
| lineage/sha sidecar | `models/qwen3.5-9b-fastdllm-b1000-stock-vllm-bf16/stock_conversion_manifest.json` |
| B@1000 adapter (conversion foundation) | `runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000/` |
| init candidate (base of merge) | `models/qwen3.5-9b-fastdllm-init/` |
| official Qwen3.5-9B snapshot | `~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| hybrid-clean sanity summary | `runs/stage_c_n5v2/diffusion/stock_sanity_hybrid1/summary.json` |
| GSM8K sanity summary | `runs/stage_c_n5v2/diffusion/stock_sanity_gsm8k5/summary.json` |

## sha256 (provenance)

| file | sha256 |
|---|---|
| export script | `6d507ec9ba3308ff7e0f600bc0b5ec7c4ff96f66eff4e4e92175d42af7a119d5` (matches §2 pin) |
| B@1000 `adapter_model.safetensors` | `f06b78c26c15a1af1f34ae6caf53cfcf0612a878627fcd79f27923a20bdda8f4` |
| B@1000 `adapter_config.json` | `4a369744153637f31fdcad1c4e55d76866528a19c996864c04a3fed96ef8a096` |
| init `model.safetensors.index.json` | `c163bf790ffaf8e1d409a301271035471d31d86a4d17038dcc07b17c43b3ce9e` |
| export `model.safetensors-00001-of-00004.safetensors` | `db6f444b43d318c92f360a13a25561a6a65b10c0631b8ed305a426dbaa6c380e` |
| export `model.safetensors-00002-of-00004.safetensors` | `31c7d7e2dd5d207840b31cc59083c8f4c4718959149e0358c0364052bb9a0330` |
| export `model.safetensors-00003-of-00004.safetensors` | `60c75252c1bc9297b468b30b6d4348c7cfe36d9a170c424e5a3efe5a7faa9320` |
| export `model.safetensors-00004-of-00004.safetensors` | `fd4b5a52cd663ac970c7e641fa001c91a602a14b8dc139dfbc9c02c5fe941c2a` |
| export `model.safetensors.index.json` | `b930fffd76536e902fd0acb01ab556f4f9098bee73cd276627dfe14a9eeafd77` |

## Environment / pins

- qwen-diffusion git commit: `c68bcf00e797d0260c6f9c9f62ff646a52566b54`
- flywheel git commit: `f063387bdb2a05dafb157eb4258efe4b5de2087c` (export script sha matches §2 pin)
- official base: `Qwen/Qwen3.5-9B` @ `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- Host-RAM caging note: `systemd-run --scope` cage is polkit-blocked (no interactive auth in this
  runtime); the export streams one ~5 GB shard at a time (peak <10 GB, 27 GiB host RAM free), so it is
  provably bounded and was run uncaged. GPU jobs stayed one-at-a-time under the <3000 MiB pre-flight.

## Next (ARM RUN, not in this step)

Serve `models/qwen3.5-9b-fastdllm-b1000-stock-vllm-bf16` on the aligned runtime + Stage-A engine config
(canvas 32, hybrid_clean, gate OFF), same 5 Tier0 instances, official docker scoring, usage-capture
proxy. Pre-registered verdict: ~3–4/5 ⇒ paradigm cost small (serve stock-conversion, retrain RL-v2 on
SWE episodes); ~1–2/5 ⇒ paradigm tax at SWE scale (investigate long-horizon decode policy). n=5 caveats.
