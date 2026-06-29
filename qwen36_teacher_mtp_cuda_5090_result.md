# Qwen3.6 Teacher MTP / CUDA 5090 Result

Date: 2026-06-26

## Status

Qwen3.6-27B NVFP4 with integrated MTP now fits and serves on the local RTX 5090
through SGLang. The teacher has been verified in both a 4k context profile for
tool-call slices and an 8k context profile for Qwen Code repo-edit turns. CUDA
graph remains disabled.

CUDA graph plus MTP also fits in memory at batch 1, but SGLang 0.5.14 crashes in
the Triton hybrid-attention speculative verification path on first scheduler use.
Treat CUDA graph as blocked by runtime compatibility, not by VRAM.

## Online Reference Check

- SGLang speculative decoding docs list MTP as a supported speculative path and
  say MTP-enabled models can use built-in multi-token heads without a separate
  draft model in some cases:
  https://docs.sglang.io/docs/advanced_features/speculative_decoding
- The SGLang Qwen3-Next cookbook says Qwen3-Next ships built-in MTP layers and
  uses `NEXTN` with:
  `--speculative-num-steps 3`,
  `--speculative-eagle-topk 1`,
  `--speculative-num-draft-tokens 4`:
  https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3-Next
- The SGLang docs default `--speculative-attention-mode` to `prefill`; the local
  launcher now follows that default.
- SGLang also supports DFlash speculative decoding, but that path requires a
  matching DFlash draft checkpoint. For this Qwen3.6 local teacher, the practical
  fast path is the model's built-in MTP/NEXTN heads; DFlash flags are exposed in
  the launcher for later draft-checkpoint experiments, not used in this result.

## Working Live Profile

4k tool-call profile:

```bash
PROFILE=nvfp4 \
HOST=127.0.0.1 \
PORT=30000 \
CONTEXT_LENGTH=4096 \
CHUNKED_PREFILL_SIZE=1024 \
MEM_FRACTION_STATIC=0.84 \
MAX_RUNNING_REQUESTS=1 \
MAX_TOTAL_TOKENS=8192 \
DISABLE_RADIX_CACHE=1 \
CUDA_GRAPH_BACKEND_DECODE=disabled \
CUDA_GRAPH_BACKEND_PREFILL=disabled \
ENABLE_MTP=1 \
SPECULATIVE_ALGORITHM=NEXTN \
SPECULATIVE_NUM_STEPS=3 \
SPECULATIVE_EAGLE_TOPK=1 \
SPECULATIVE_NUM_DRAFT_TOKENS=4 \
SPECULATIVE_ATTENTION_MODE=prefill \
scripts/serve_sglang_qwen36_teacher.sh
```

8k Qwen Code repo-edit profile:

```bash
PROFILE=nvfp4 \
HOST=127.0.0.1 \
PORT=30000 \
CONTEXT_LENGTH=8192 \
CHUNKED_PREFILL_SIZE=1024 \
MEM_FRACTION_STATIC=0.84 \
MAX_RUNNING_REQUESTS=1 \
MAX_TOTAL_TOKENS=8192 \
DISABLE_RADIX_CACHE=1 \
CUDA_GRAPH_BACKEND_DECODE=disabled \
CUDA_GRAPH_BACKEND_PREFILL=disabled \
ENABLE_MTP=1 \
SPECULATIVE_ALGORITHM=NEXTN \
SPECULATIVE_NUM_STEPS=3 \
SPECULATIVE_EAGLE_TOPK=1 \
SPECULATIVE_NUM_DRAFT_TOKENS=4 \
SPECULATIVE_ATTENTION_MODE=prefill \
scripts/serve_sglang_qwen36_teacher.sh
```

Important fit lever: `DISABLE_RADIX_CACHE=1`. With radix cache on, SGLang's
hybrid GDN/Mamba cache allocator required a larger Mamba cache ratio and failed
to allocate even one request after the 27B target and MTP module were resident.

Current live tmux session:

```text
qwen36_teacher
```

Current live log:

```text
logs/qwen36_teacher_nvfp4_mtp_4k_radixoff_nograph_20260626_120834.log
```

## Verification

Endpoint:

```text
GET /v1/models -> qwen3.6-27b-teacher, max_model_len=4096
```

The 8k profile returns:

```text
GET /v1/models -> qwen3.6-27b-teacher, max_model_len=8192
```

Chat smoke:

```text
prompt: Reply with exactly: MTP_4K_OK
response: MTP_4K_OK
usage: prompt_tokens=22, completion_tokens=7, total_tokens=29
```

Observed memory after smoke:

```text
RTX 5090: about 23.7 GiB used, about 8.2 GiB free
```

Observed memory during the 8k Qwen Code repo-edit sweep:

```text
RTX 5090: about 24-25 GiB used out of 32.6 GiB
```

Useful load-time memory lines:

```text
target Qwen3_5ForConditionalGeneration: 18.81 GB
MTP Qwen3_5ForCausalLMMTP: 5.05 GB
Mamba cache: max_mamba_cache_size=1, intermediate_ssm_state_cache=1.12 GB
KV cache target: 8192 tokens, K=0.25 GB, V=0.25 GB
```

## CUDA Graph Attempts

Memory was not the blocker. CUDA graph attempts reached graph capture and server
startup, then failed during speculative verification.

Failed full graph profile:

```bash
DISABLE_RADIX_CACHE=1 \
CUDA_GRAPH_BACKEND_DECODE=full \
CUDA_GRAPH_MAX_BS_DECODE=1 \
CUDA_GRAPH_BS_DECODE=1 \
CUDA_GRAPH_BACKEND_PREFILL=disabled \
ENABLE_MTP=1 \
SPECULATIVE_ALGORITHM=NEXTN \
SPECULATIVE_NUM_STEPS=3 \
SPECULATIVE_EAGLE_TOPK=1 \
SPECULATIVE_NUM_DRAFT_TOKENS=4 \
scripts/serve_sglang_qwen36_teacher.sh
```

Failure:

```text
RuntimeError: The expanded size of the tensor (4096) must match the existing
size (4112) at non-singleton dimension 0.
```

The same error reproduced with `SPECULATIVE_ATTENTION_MODE=prefill` and with a
smaller `SPECULATIVE_NUM_STEPS=1`, `SPECULATIVE_NUM_DRAFT_TOKENS=2` tree:

```text
RuntimeError: The expanded size of the tensor (2048) must match the existing
size (2052) at non-singleton dimension 0.
```

Failed breakable graph profile:

```text
TypeError: Unsupported BCG output type:
<class 'sglang.srt.layers.logits_processor.LogitsProcessorOutput'>
```

Logs:

```text
logs/qwen36_teacher_nvfp4_mtp_cudagraph_bs1_20260626_114709.log
logs/qwen36_teacher_nvfp4_mtp_cudagraph_breakable_bs1_20260626_115015.log
logs/qwen36_teacher_nvfp4_mtp_cudagraph_bs1_steps1_20260626_115217.log
logs/qwen36_teacher_nvfp4_mtp_cudagraph_bs1_prefillmode_20260626_115657.log
```

## Takeaway

Use the 4k MTP/no-graph profile for short tool-call teacher/data loops. Use the
8k MTP/no-graph profile for Qwen Code or other tool-schema-heavy agentic turns.
Both fit comfortably enough on the 5090 with NVFP4, one running request, radix
cache disabled, and CUDA graph disabled.

For CUDA graph, next work should be upstream/runtime focused:

1. Try a newer SGLang nightly after checking for fixes around hybrid GDN,
   NEXTN/EAGLE verification, and Triton CUDA graph metadata.
2. Try FlashInfer attention only after the Blackwell/CUDA graph path is known to
   compile cleanly on this machine.
3. Keep graph capture at `bs=1` and prefill graph disabled until the verify-mask
   crash is gone.
