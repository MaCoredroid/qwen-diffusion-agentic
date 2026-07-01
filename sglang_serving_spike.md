# SGLang Serving Spike Wrap

Date: 2026-06-30

Decision context: the serving foundation is now our own HF route_i FLARE forward plus a per-request `RequestDiffusionState` cache, per `serving_architecture.md`. SGLang is deferred: useful later as a throughput/sample-generation path, with parity re-scored on the HF forward, but not the foundation for RL rollouts.

## Bottom Line

- **Toolchain gate: PASS.** SGLang runs on this RTX 5090 (`sm_120`) with the native Qwen3.5/GDN path. No `no kernel image` or `sm_120` crash class was seen.
- **Load gate: PARTIAL/POSITIVE.** Our 9B converted checkpoint can load and boot an SGLang server only after text-only architecture/config shims. This confirms SGLang is viable but has nontrivial integration cost for our custom FLARE checkpoint.
- **Generation under temporary shim: NOT production-stable.** The shimmed 9B server reached Uvicorn and `/v1/models`, and SGLang's startup `/generate` warmup returned 200, but a manual OpenAI completion request later crashed in prefill CUDA graph replay with `AssertionError: PCG capture stream is not set`.
- **Train-serve parity: DEFERRED.** Per the confirmed foundation decision, parity will be by construction on the HF route_i forward, with exact re-score for RL. I did not pursue SGLang parity.
- **Verdict.** SGLang is toolchain-viable on the 5090 and remains a valid deferred throughput play. It should not be the foundation.

## Gate 1: SGLang On RTX 5090 / sm_120

Environment used:

- Python: `.venv-sglang/bin/python`
- SGLang: `0.5.14`
- Torch: `2.11.0+cu130`
- Transformers: `5.8.1`
- GPU: `NVIDIA GeForce RTX 5090`, capability `(12, 0)`

Native GDN smoke:

- Launched SGLang with `Qwen/Qwen3.5-4B`, Triton attention/GDN backends, tiny context, and the repo's CUDA wrapper setup from `scripts/serve_sglang_qwen36_teacher.sh`.
- First launch with `MEM_FRACTION_STATIC=0.30` failed during cache sizing, not kernel execution: not enough memory for hybrid Mamba/linear-attention state cache.
- Relaunch with `MEM_FRACTION_STATIC=0.65`, `CONTEXT_LENGTH=512`, `MAX_RUNNING_REQUESTS=1`, Triton prefill/decode succeeded.

Important observed logs:

- `Using hybrid linear attention backend for hybrid GDN models`
- `GDN kernel dispatcher: decode=TritonGDNKernel, extend=TritonGDNKernel, verify=TritonGDNKernel packed_decode=True`
- `Tree cache initialized: source=default impl=MambaRadixCache hybrid_swa=False hybrid_ssm=True`
- API smoke worked: `/v1/models` returned the served model, and `/v1/completions` returned text for prompt `READY`.

Conclusion: SGLang itself runs on sm_120 here, including native GDN and Mamba radix cache.

## Load Probe: FLARE Qwen3.5-9B

Target artifacts:

- Base checkpoint/config: `models/qwen3.5-9b-fastdllm-init`
- Trained adapter: `runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000`
- Local architecture: `Fast_dLLM_Qwen3_5ForCausalLM`
- Underlying source: Qwen3.5-9B / Qwen3-Next style GDN hybrid, 24 linear-attention layers and 8 full-attention layers.

Direct load failed as expected:

```text
ValueError: Fast_dLLM_Qwen3_5ForCausalLM has no SGlang implementation and the Transformers implementation is not compatible with SGLang.
```

Text-only Qwen3.5 view findings:

- A raw top-level `Qwen3_5ForConditionalGeneration` view triggered SGLang's multimodal processor and failed due missing vision/preprocessor config.
- A text-only `Qwen3_5ForConditionalGeneration` view still went through `qwen_vl.py` and expected vision fields.
- SGLang registers `Qwen3_5ForConditionalGeneration`, not a standalone dense `Qwen3_5ForCausalLM`, for this family.

Temporary config/property shims that got the 9B load through:

- Register a new external architecture: `Qwen3_5TextOnlyForCausalLM`.
- Reuse SGLang's in-tree `Qwen3_5ForCausalLM` as the language backbone, but wrap it with an LM head and `LogitsProcessor`.
- Return `None` for dense-model expert-location metadata; SGLang's in-tree dense class otherwise assumes `config.num_experts`.
- Register HF `Qwen3_5TextConfig` with SGLang's external `LinearAttnModelSpec`, pointing to `sglang.srt.layers.attention.linear.gdn_backend.GDNAttnBackend`.
- Add derived config properties expected by SGLang's hybrid cache/backend path: `layers_block_type`, `linear_layer_ids`, `full_attention_layer_ids`, and `mamba2_cache_params`.
- Keep derived fields out of `config.json`; HF config reload rejects read-only property keys.
- For the shim wrapper, use normal `positions` when `forward_batch.mrope_positions` is absent.

Successful shimmed load/boot evidence:

- Weights loaded: `type=Qwen3_5TextOnlyForCausalLM`, memory usage `16.74 GB`.
- Mamba cache allocated with `MAX_MAMBA_CACHE_SIZE=1`: `ssm_state size: 0.09GB`.
- KV cache allocated: `#tokens: 512`.
- Linear attention backend initialized: `decode=triton, prefill=triton`.
- GDN dispatcher initialized: `TritonGDNKernel` for decode/extend/verify.
- Prefill and decode CUDA graph capture completed.
- Server reached Uvicorn on `127.0.0.1:30106`.
- `/v1/models` returned `fastdllm-qwen35-9b-sglang-text-wrapper`, `max_model_len=128`.

Generation caveat:

- SGLang startup warmup `/generate` returned 200.
- A manual OpenAI `/v1/completions` request with prompt `READY`, `max_tokens=3`, `temperature=0` crashed the scheduler:

```text
AssertionError: PCG capture stream is not set, please check if runtime recompilation happened
```

This is a temporary-wrapper integration failure, not an sm_120/GDN kernel failure. It still matters: a production SGLang path would need a real upstream-style text-only Qwen3.5 registration, CUDA graph handling, and LoRA/merged-weight handling before it can be trusted.

LoRA note:

- I did not pursue SGLang LoRA loading after the wrap decision. If SGLang is revived later, the safest route is likely to merge the small FLARE LoRA into a full checkpoint or add explicit LoRA target mapping for SGLang's packed modules (`q_proj/k_proj/v_proj` -> `qkv_proj`, `o_proj` unchanged).

## Parity

Deferred by decision.

The serving foundation is the HF route_i FLARE forward, so train-serve parity should be established there by construction and verified by the full-stack cache-on/cache-off instruments in `serving_architecture.md`. SGLang, if revived, should be a sample generator or throughput optimization with final logprobs re-scored by the HF forward.

## Radix Cache / Diffusion Path

Gate-1 confirmed SGLang's native Qwen3.5/GDN server initializes `MambaRadixCache` on this GPU. The shimmed 9B load was intentionally run with `DISABLE_RADIX_CACHE=1` and a tiny Mamba pool to minimize the load probe; that server initialized `ChunkCache` with `hybrid_ssm=True`.

I did not continue into SGLang block-diffusion integration because the confirmed foundation is HF own-forward. If SGLang is resumed later, the minimum integration work is:

1. Promote the temporary text-only Qwen3.5/FLARE adapter into a maintainable SGLang external model or upstream-style model registration.
2. Enable Mamba radix cache on the 9B path and size it honestly.
3. Prove generation stability without temporary wrapper CUDA-graph recompilation.
4. Treat SGLang as throughput/sample generation only until exact HF re-score confirms rollout logprobs.

## Final Verdict

SGLang is viable on this RTX 5090 and its native GDN/Mamba cache stack is real. Our custom FLARE 9B checkpoint is not plug-and-play: it needs architecture and config shims before it even loads, and the temporary shim is not generation-stable. That integration cost is acceptable for a deferred throughput project, but it is the wrong foundation for the next RL-serving step.

Stop here. Build the HF route_i serving foundation as the next clean task.
