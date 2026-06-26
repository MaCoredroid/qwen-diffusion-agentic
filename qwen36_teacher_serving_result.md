# Qwen3.6 Teacher Serving Result

Date: 2026-06-26

## Status

The local RTX 5090 can serve a Qwen3.6-27B-class AR teacher through SGLang using
the NVFP4 checkpoint and an OpenAI-compatible endpoint.

This closes the first serving smoke gate for the eval/data loop. It is not yet a
speed-tuned serving profile.

## Working Profile

- Model: `sakamakismile/Qwen3.6-27B-NVFP4`
- Served name: `qwen3.6-27b-teacher`
- Server: SGLang `0.5.14`
- PyTorch: `2.11.0+cu130`
- `sgl_kernel`: `0.4.4`
- GPU: RTX 5090
- Context length: `2048`
- Max running requests: `1`
- Attention backend: `triton`
- FP4 GEMM backend: `cutlass`
- CUDA graph decode/prefill: `disabled`
- MTP/speculative: disabled for this smoke profile

Launch shape:

```bash
PROFILE=nvfp4 \
HOST=127.0.0.1 \
PORT=30000 \
CONTEXT_LENGTH=2048 \
CHUNKED_PREFILL_SIZE=1024 \
MEM_FRACTION_STATIC=0.84 \
MAX_RUNNING_REQUESTS=1 \
ENABLE_MTP=0 \
scripts/serve_sglang_qwen36_teacher.sh
```

The launcher defaults now encode the proven NVFP4 fallback profile. Faster
FlashInfer/CUDA-graph/MTP settings remain explicit overrides to re-test later.

## Fixes Needed

- Use the available NVFP4 checkpoint ID:
  `sakamakismile/Qwen3.6-27B-NVFP4`.
- Use `compressed-tensors` quantization because the checkpoint declares that
  quantization config.
- Build a cache-local CUDA wrapper for Python CUDA packages so FlashInfer/SGLang
  JIT sees conventional `CUDA_HOME/lib64` linker paths.
- Add the vendored CCCL/libcudacxx include path so the NVFP4 CUTLASS JIT can
  resolve `nv/target`.
- Use Triton attention and CUTLASS FP4 GEMM for the stable 5090 smoke path.

## Verification

Endpoint:

```text
GET /v1/models -> qwen3.6-27b-teacher, max_model_len=2048
```

Chat smoke:

```text
POST /v1/chat/completions with chat_template_kwargs.enable_thinking=false -> 200 OK
```

Important: Qwen3.6 emits reasoning by default in this serving path. Labeling
requests must pass top-level:

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

Tool-call teacher probe:

```text
input: data/toolcall_eval/synthetic_onecall_smoke.jsonl
output: data/toolcall_eval/synthetic_onecall_teacher_q36_nvfp4_heldout48.jsonl
records: 48
ok: 48
valid tool-call format: 48
exact tool-name set: 48
errors: 0
elapsed: 73.63s
```

The teacher produced Qwen-native `<tool_call><function=...>` formatting, so
`scripts/teacher_distill_toolcall_cases.py` now accepts both JSON-object calls
and Qwen function-tag calls when scoring tool names.

Observed memory while serving:

```text
RTX 5090: about 28.4 GiB used, about 3.7 GiB free
```

Log for the successful launch:

```text
logs/qwen36_teacher_nvfp4_20260626_111353_triton_cutlass_cccl.log
```

Logs are not committed.

## Caveats

- This is a conservative smoke profile: 2k context, one request, no CUDA graph,
  no MTP/speculative decoding.
- The earlier FlashInfer path loaded weights but failed or stalled in
  FlashInfer/CuTe/CUDA-graph setup on this local stack.
- The metric name `valid_tool_json` in the current probe summary means valid
  recognized tool-call emission; the accepted Qwen-native format is not JSON.
- This does not complete the full agentic data/eval harness. Public one-call,
  argument-level scoring, multi-call traces, Qwen3.5-9B AR baseline, and
  diffusion-target training remain next.

## Next Step

Use this teacher endpoint to build the public-data tool-call eval loop:

1. Add argument-level schema scoring for Qwen-native tool calls.
2. Run Hermes/Glaive/ToolACE one-call teacher labels.
3. Add two-step tool traces and repeated-call checks.
4. Measure Qwen3.5-9B AR baseline on the same evals.
5. Move the constrained synthetic/teacher-distilled loop to the 9B diffusion
   target.
