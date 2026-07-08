#!/usr/bin/env bash
# =============================================================================
# STAGED — NOT ACTIVE. Datagen teacher swap: Qwen3.6-27B NVFP4 + native MTP
# spec-decode, flywheel APC (lossless GDN prefix cache), capped DDR5 KV offload.
#
# Drop-in replacement for runcage_ar_probe.sh, driven the SAME way by
# datagen_gen.sh: point it via  RUNCAGE_SCRIPT=runcage_27b.sh  (see
# datagen_gen.sh line ~25). datagen_gen.sh wraps THIS script inside the
# systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=4G cage and
# passes MAX_NUM_SEQS / MAX_MODEL_LEN / GPU_UTIL / PORT / NUM_SPEC_TOKENS as env
# (its dynamic-gmu block computes GPU_UTIL = min(0.85,(total-used-1800)/total),
# hard-floored 0.74). This launcher, like runcage_ar_probe.sh, just execs
# `vllm serve` inside that cage. ONE heavy process at a time.
#
# ---- WHY these flags (provenance) --------------------------------------------
#  * NVFP4 weights   : nvidia/Qwen3.6-27B-NVFP4 (modelopt 0.45, MIXED_PRECISION,
#                      quant_method=modelopt -> vLLM quantization "modelopt_fp4"
#                      = ModelOptNvFp4Config, registered in vLLM 0.23). 2194
#                      tensors, ~21.9GB, arch Qwen3_5ForConditionalGeneration,
#                      model_type qwen3_5, vocab 248320, 64 layers /
#                      full_attention_interval 4 => 16 full-attn layers.
#  * MTP head PRESENT: 15 mtp.* tensors (mtp.fc, mtp.layers.0.*, mtp.norm,
#                      mtp.pre_fc_norm_*), mtp_num_hidden_layers=1, ALL BF16 on
#                      disk (excluded from NVFP4 quant via hf_quant_config
#                      exclude_modules ['mtp*','mtp.layers.0*'] => "mtp.fc kept
#                      bf16"). Verified from the safetensors headers on disk.
#  * spec method     : qwen3_5_mtp is the ONLY vLLM MTP path that reads
#                      mtp_num_hidden_layers (all others read
#                      num_nextn_predict_layers, which this ckpt lacks). This is
#                      the flywheel's config-E choice
#                      (scripts/swe_x86_helpers/relaunch_qwen36_E.py) and
#                      fr13_launch_native_mtp_server.sh. num_speculative_tokens
#                      default 1 (flywheel config-E; probe may raise).
#  * APC (flywheel FR13 lossless GDN prefix cache): --enable-prefix-caching
#                      --enable-chunked-prefill --mamba-cache-mode align
#                      --mamba-block-size 1024 --mamba-ssm-cache-dtype float32
#                      --gdn-prefill-backend triton. This is the exact
#                      cache-mode='align' + fp32 SSM primitive proven lossless
#                      on the 9B (runcage_ar_probe.sh) and de-risked in the
#                      flywheel (FR13_APC_EXACT_SEED_SUCCESS). align is REQUIRED
#                      for MTP+hybrid; prefix-cache 'all' mode is UNSUPPORTED
#                      with MTP, so we pin 'align' explicitly.
#  * DDR5 KV offload : --kv-offloading-size <=8 GiB, backend native. This is the
#                      vLLM 0.23 CacheConfig.kv_offloading_size flag (V1 dropped
#                      legacy --swap-space). HARD-CAPPED at 8 GiB in-code per the
#                      HOST-RAM directive (30GB host, cage MemoryMax=22G). The
#                      boot-probe MUST confirm the honored size in the startup
#                      log ("KV cache offloading ... GiB" / num_cpu_blocks).
#  * sm_120 (RTX 5090): --no-enable-flashinfer-autotune + VLLM_USE_FLASHINFER_SAMPLER=0
#                      (proven-stable on the 9B this box). NVFP4 GEMMs use the
#                      FlashInferB12xNvFp4 linear kernel automatically on sm_120.
#  * tools/format    : native qwen3_xml tool-call parser + qwen3 reasoning parser
#                      + the codex chat template (SAME as the 9B AR teacher) so
#                      the emitted keeper format is IDENTICAL BY CONSTRUCTION
#                      (STRONGER_TEACHER_PILOT.md §4). See BOOT-PROBE CHECKLIST.
#
# ---- MEMORY BUDGET (RTX 5090 32607MiB / ~31.84GiB) — do NOT copy the 9B gmu ---
#   weights (NVFP4) ~20.4 GiB  |  gmu 0.85 -> ~27.1 GiB usable -> ~6.7 GiB left
#   for {KV pool + GDN state + activations + cudagraph + MTP drafter}.
#   KV @32k, 16 full-attn layers, 64KiB/tok bf16 ~= 2.0 GiB/seq (fp8 ~1.0 GiB/seq).
#   GDN mamba state ~0.15 GiB/seq fp32 (OUTSIDE the KV pool). => bf16 KV fits ~2
#   seqs on-GPU; the 8 GiB DDR5 offload buffer absorbs preempted-seq KV so 2-3
#   seqs is workable. MAX_NUM_SEQS default 2 (probe decides 2 vs 3; fp8 KV via
#   KV_CACHE_DTYPE=fp8 is the capacity lever for 3). GMU+concurrency budgeted
#   together with measured headroom, per standing rule.
#
# ---- BOOT-PROBE CHECKLIST (must pass BEFORE this replaces the 9B in datagen) --
#   1. Server reaches READY on the 5090 (weights load, no OOM) at gmu/seqs above.
#   2. Startup log shows: MTP head loaded (qwen3_5_mtp), Mamba cache mode 'align',
#      chunked prefill on, prefix caching on, and the DDR5 KV-offload size honored
#      (<=8 GiB) — grep the log, per HOST-RAM directive.
#   3. NVFP4 + MTP + kv_offloading + align are a NEW combo here: if kv_offloading
#      is rejected alongside MTP/hybrid at boot, set KV_OFFLOAD_GB=0 and rely on
#      gmu + KV_CACHE_DTYPE=fp8 for capacity (report honestly; do not force it).
#   4. Format-equivalence cert (§4): diff one 27B keeper row vs a production
#      keeper row (extract_keepers.py) — schema + native qwen3_xml tool-call XML
#      must match EXACTLY. If the codex chat template mis-renders on the 27B
#      tokenizer (vocab 248320), switch CHAT_TEMPLATE to the ckpt's own
#      chat_template.jinja and re-cert.
#   5. ARGUMENT-GROUNDING SPOT-CHECK (NON-NEGOTIABLE): NVFP4 calibration can
#      damage verbatim copying — hand-verify a few tool-call arguments are
#      byte-exact echoes of the source (the project crux). NO promotion without it.
# =============================================================================
set -euo pipefail
cd /home/mark/qwen_diffusion

# ---- checkpoint (nvidia NVFP4, MTP-bearing) ---------------------------------
SNAP=${SNAP:-/home/mark/.cache/huggingface/hub/models--nvidia--Qwen3.6-27B-NVFP4/snapshots/0893e1606ff3d5f97a441f405d5fc541a6bdf404}
VLLM_BIN=${VLLM_BIN:-/home/mark/qwen_diffusion/.venv-vllm/bin/vllm}
# SAME codex tool template as the 9B AR teacher (format equivalence by construction).
# Boot-probe may switch to $SNAP/chat_template.jinja if the 27B tokenizer mis-renders.
CHAT_TEMPLATE=${CHAT_TEMPLATE:-/home/mark/shared/lumoFlyWheel_codex_fork/docker/chat_templates/qwen3-openai-codex.jinja}
SERVED_NAME=${SERVED_NAME:-qwen3.6-27b-nvfp4}

PORT=${PORT:-9952}
GPU_UTIL=${GPU_UTIL:-0.85}                       # datagen_gen.sh overrides with dynamic gmu
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-2}                  # probe decides 2 vs 3
QUANT=${QUANT-modelopt_fp4}                      # vLLM 0.23 NVFP4 modelopt; set QUANT= (empty) => auto-detect
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-auto}           # auto=bf16 (safest for grounding); fp8 = capacity lever
NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS:-1}            # MTP speculative tokens (flywheel config-E=1)
ATTENTION_BACKEND=${ATTENTION_BACKEND:-FLASH_ATTN}
NO_FI_AUTOTUNE=${NO_FI_AUTOTUNE:-1}              # sm_120-proven on the 9B

# ---- HARD-CAP the DDR5 KV offload at 8 GiB (HOST-RAM directive; in-code, not just default)
# DEFAULT 0 (offload OFF) — boot-probe (bootprobe_27b) FROZE this: KV offload forces
# dropping PYTORCH_CUDA_ALLOC_CONF=expandable_segments (VMM/pinned-buffer conflict),
# which SHRINKS the on-GPU fp8 KV pool 83,012->77,550 tok (2.53x->2.37x @32k) AND
# pushes cage RSS toward the 22G cap — a net LOSS when 2 full 32k seqs already fit
# on-GPU. Offload remains a VALIDATED opt-in capacity lever (set KV_OFFLOAD_GB=4|8);
# it boots cleanly with the OffloadingConnector. USER-APPROVED, HARD-CAPPED <=8 GiB.
KV_OFFLOAD_GB=${KV_OFFLOAD_GB:-0}
# Hard-cap at 8 GiB and emit an INTEGER string: --kv-offloading-size uses vLLM's
# human_readable_int parser (int() based), which REJECTS a decimal like "8.0".
KV_OFFLOAD_GB=$(python3 -c "v=min(float('${KV_OFFLOAD_GB}'), 8.0); v=v if v>0 else 0.0; print(int(v) if v==int(v) else v)")
KV_OFFLOAD_BACKEND=${KV_OFFLOAD_BACKEND:-native}

export CUDA_HOME=${CUDA_HOME:-/home/mark/qwen_diffusion/.venv-vllm/lib/python3.12/site-packages/nvidia/cu13}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
# sm_120 (RTX 5090) + FlashInfer 0.6.12 JIT is BROKEN on this box: the bundled
# nvcc is 13.2 while FlashInfer's vendored cccl/cutlass headers reject it
# ("CUDA compiler and CUDA toolkit headers are incompatible"), so the FlashInfer
# NVFP4/FP8 *linear* GEMMs, which JIT-compile at boot, FAIL and abort the engine.
# Disable those FlashInfer linear kernels so vLLM's kernel selector falls through
# to COMPILED (no-nvcc) kernels: NVFP4 -> CutlassNvFp4/MarlinNvFp4 (MarlinNvFp4
# supports cap>=7.5, verified), FP8 linear_attn projs -> CutlassFP8/PerTensorTorch.
# Boot-probe verified (bootprobe_27b): this is REQUIRED for the 27B to boot here.
export VLLM_DISABLED_KERNELS="${VLLM_DISABLED_KERNELS:-FlashInferFP8ScaledMMLinearKernel,FlashInferCutlassNvFp4LinearKernel,FlashInferTrtllmNvFp4LinearKernel,FlashInferCudnnNvFp4LinearKernel}"
# ROOT-CAUSE of the FlashInfer JIT failure: the bundled nvidia/cu13 pip package is
# INTERNALLY INCONSISTENT — nvcc is release 13.2 but its headers declare
# CUDA_VERSION=13000 (13.0). FlashInfer's vendored cccl asserts nvcc==toolkit
# (cuda_toolkit.h: !_CCCL_CUDACC_EQUAL) and hard-errors on the 13.2-vs-13.0 skew.
# The full-attention layers (head_dim 256, Qwen3.5 gated attn) + fp8 KV can ONLY be
# served by FlashInfer's batch-prefill kernel, which JIT-compiles on the first
# forward -> without this the engine dies with a 500 mid-request. cccl provides the
# documented escape hatch CCCL_DISABLE_CTK_COMPATIBILITY_CHECK; nvcc honors
# NVCC_APPEND_FLAGS, so we inject the define into every JIT nvcc call. A 13.2 nvcc
# compiling 13.0 headers is a benign minor skew for these kernels. Compiled modules
# are cached under ~/.cache/flashinfer, so only the first forward pays the JIT cost.
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:+$NVCC_APPEND_FLAGS }-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
# The FlashInfer JIT LINK step then fails "cannot find -lcudart": the nvidia/cu13
# pip wheels ship only versioned libs (libcudart.so.13) with NO unversioned dev
# symlink, and their lib dir is not on the default linker path. Provide both:
# a dir of unversioned *.so dev-symlinks (generated once into cuda_dev_links/) plus
# the cu13 lib dir on LIBRARY_PATH (link time) and LD_LIBRARY_PATH (load time).
CU13_LIB="${CUDA_HOME}/lib"
CUDA_DEVLINKS="${CUDA_DEVLINKS:-/home/mark/qwen_diffusion/runs/swe_datagen_s1/cuda_dev_links}"
# Self-heal: generate the unversioned *.so dev-symlinks if missing (this dir is a
# gitignored run output, so a fresh checkout has none). Idempotent.
if [[ ! -e "${CUDA_DEVLINKS}/libcudart.so" ]]; then
  mkdir -p "$CUDA_DEVLINKS"
  for _f in "$CU13_LIB"/*.so.13; do [[ -e "$_f" ]] || continue; _b=$(basename "$_f"); ln -sf "$_f" "${CUDA_DEVLINKS}/${_b%.so.13}.so"; done
fi
export LIBRARY_PATH="${CUDA_DEVLINKS}:${CU13_LIB}${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="${CU13_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_NO_USAGE_STATS=${VLLM_NO_USAGE_STATS:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
# KV offloading (OffloadingConnector) is INCOMPATIBLE with expandable_segments:True
# (PyTorch's CUDA VMM can remap KV virtual addresses, invalidating the pinned host
# offload buffer) unless the cumem allocator is enabled. Boot-probe verified: with
# KV_OFFLOAD_GB>0 the engine fails VllmConfig validation under expandable_segments.
# So drop it only when offload is on; keep it (fragmentation safety) when offload=0.
if python3 -c "exit(0 if float('$KV_OFFLOAD_GB')>0 else 1)"; then
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF_OFFLOAD:-}"
else
  export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
fi

SPEC_CONFIG=${SPEC_CONFIG:-"{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS}}"}

# ---- assemble optional flags -------------------------------------------------
QUANT_ARGS=(); [[ -n "$QUANT" ]] && QUANT_ARGS=(--quantization "$QUANT")
ATTN_ARGS=(); [[ -n "$ATTENTION_BACKEND" ]] && ATTN_ARGS=(--attention-backend "$ATTENTION_BACKEND")
FI_ARGS=();  [[ "$NO_FI_AUTOTUNE" == "1" ]] && FI_ARGS=(--no-enable-flashinfer-autotune)
KVOFF_ARGS=()
if python3 -c "exit(0 if float('$KV_OFFLOAD_GB')>0 else 1)"; then
  KVOFF_ARGS=(--kv-offloading-size "$KV_OFFLOAD_GB" --kv-offloading-backend "$KV_OFFLOAD_BACKEND")
fi
SPEC_ARGS=(); [[ -n "$SPEC_CONFIG" ]] && SPEC_ARGS=(--speculative-config "$SPEC_CONFIG")

echo "[runcage_27b STAGED] snap=$SNAP served=$SERVED_NAME port=$PORT gmu=$GPU_UTIL len=$MAX_MODEL_LEN seqs=$MAX_NUM_SEQS quant=${QUANT:-auto} kv=$KV_CACHE_DTYPE spec=$SPEC_CONFIG kv_offload_gib=$KV_OFFLOAD_GB($KV_OFFLOAD_BACKEND) attn=$ATTENTION_BACKEND no_fi_autotune=$NO_FI_AUTOTUNE" >&2

exec "$VLLM_BIN" serve "$SNAP" \
  --served-model-name "$SERVED_NAME" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype bfloat16 \
  "${QUANT_ARGS[@]}" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gdn-prefill-backend triton \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --mamba-block-size 1024 \
  --mamba-ssm-cache-dtype float32 \
  "${KVOFF_ARGS[@]}" \
  "${SPEC_ARGS[@]}" \
  "${ATTN_ARGS[@]}" \
  "${FI_ARGS[@]}" \
  --chat-template "$CHAT_TEMPLATE" \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3
