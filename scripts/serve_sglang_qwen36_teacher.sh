#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

SGLANG_PYTHON="${SGLANG_PYTHON:-.venv-sglang/bin/python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
PROFILE="${PROFILE:-nvfp4}"
SGLANG_MIN_VERSION="${SGLANG_MIN_VERSION:-0.5.10}"
SKIP_VERSION_CHECK="${SKIP_VERSION_CHECK:-0}"

case "${PROFILE}" in
  fp8)
    MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.6-27B-FP8}"
    QUANTIZATION="${QUANTIZATION:-fp8}"
    KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
    DEFAULT_ATTENTION_BACKEND="flashinfer"
    DEFAULT_FP4_GEMM_BACKEND="auto"
    DEFAULT_CUDA_GRAPH_BACKEND_DECODE=""
    DEFAULT_CUDA_GRAPH_BACKEND_PREFILL=""
    ;;
  nvfp4|fp4|q4)
    MODEL_PATH="${MODEL_PATH:-sakamakismile/Qwen3.6-27B-NVFP4}"
    # This NVFP4 checkpoint declares `compressed-tensors` in config.json; forcing
    # `petit_nvfp4` makes newer SGLang reject the load before weights download.
    QUANTIZATION="${QUANTIZATION:-compressed-tensors}"
    KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
    DEFAULT_ATTENTION_BACKEND="triton"
    DEFAULT_FP4_GEMM_BACKEND="cutlass"
    DEFAULT_CUDA_GRAPH_BACKEND_DECODE="disabled"
    DEFAULT_CUDA_GRAPH_BACKEND_PREFILL="disabled"
    ;;
  custom)
    MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH for PROFILE=custom}"
    QUANTIZATION="${QUANTIZATION:-}"
    KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
    DEFAULT_ATTENTION_BACKEND="flashinfer"
    DEFAULT_FP4_GEMM_BACKEND="auto"
    DEFAULT_CUDA_GRAPH_BACKEND_DECODE=""
    DEFAULT_CUDA_GRAPH_BACKEND_PREFILL=""
    ;;
  *)
    echo "Unknown PROFILE=${PROFILE}; use fp8, nvfp4, or custom." >&2
    exit 2
    ;;
esac

CONTEXT_LENGTH="${CONTEXT_LENGTH:-8192}"
LOAD_FORMAT="${LOAD_FORMAT:-}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.84}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-4}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-4096}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-${DEFAULT_ATTENTION_BACKEND}}"
PREFILL_ATTENTION_BACKEND="${PREFILL_ATTENTION_BACKEND:-${ATTENTION_BACKEND}}"
DECODE_ATTENTION_BACKEND="${DECODE_ATTENTION_BACKEND:-${ATTENTION_BACKEND}}"
CUDA_GRAPH_BACKEND_DECODE="${CUDA_GRAPH_BACKEND_DECODE:-${DEFAULT_CUDA_GRAPH_BACKEND_DECODE}}"
CUDA_GRAPH_BACKEND_PREFILL="${CUDA_GRAPH_BACKEND_PREFILL:-${DEFAULT_CUDA_GRAPH_BACKEND_PREFILL}}"
CUDA_GRAPH_MAX_BS_DECODE="${CUDA_GRAPH_MAX_BS_DECODE:-}"
CUDA_GRAPH_MAX_BS_PREFILL="${CUDA_GRAPH_MAX_BS_PREFILL:-}"
CUDA_GRAPH_BS_DECODE="${CUDA_GRAPH_BS_DECODE:-}"
CUDA_GRAPH_BS_PREFILL="${CUDA_GRAPH_BS_PREFILL:-}"
FP8_GEMM_BACKEND="${FP8_GEMM_BACKEND:-auto}"
FP4_GEMM_BACKEND="${FP4_GEMM_BACKEND:-${DEFAULT_FP4_GEMM_BACKEND}}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.6-27b-teacher}"
DTYPE="${DTYPE:-auto}"
DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-0}"
MAMBA_RADIX_CACHE_STRATEGY="${MAMBA_RADIX_CACHE_STRATEGY:-}"
MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-}"
MAMBA_FULL_MEMORY_RATIO="${MAMBA_FULL_MEMORY_RATIO:-}"
ENABLE_INT8_MAMBA_CHECKPOINT="${ENABLE_INT8_MAMBA_CHECKPOINT:-0}"
INT8_MAMBA_CKPT_SIZE="${INT8_MAMBA_CKPT_SIZE:-}"

# Qwen3.6 uses MTP in the model family. SGLang exposes this through speculative
# decoding flags when the backend/model path supports it. Enable it after the
# base profile loads cleanly; keep the toggle explicit because invalid speculative
# settings can prevent the server from starting.
ENABLE_MTP="${ENABLE_MTP:-0}"
SPECULATIVE_ALGORITHM="${SPECULATIVE_ALGORITHM:-NEXTN}"
SPECULATIVE_NUM_STEPS="${SPECULATIVE_NUM_STEPS:-3}"
SPECULATIVE_EAGLE_TOPK="${SPECULATIVE_EAGLE_TOPK:-1}"
SPECULATIVE_NUM_DRAFT_TOKENS="${SPECULATIVE_NUM_DRAFT_TOKENS:-4}"
SPECULATIVE_DFLASH_BLOCK_SIZE="${SPECULATIVE_DFLASH_BLOCK_SIZE:-}"
SPECULATIVE_ATTENTION_MODE="${SPECULATIVE_ATTENTION_MODE:-prefill}"
SPECULATIVE_DRAFT_ATTENTION_BACKEND="${SPECULATIVE_DRAFT_ATTENTION_BACKEND:-}"
SPECULATIVE_DRAFT_MODEL_PATH="${SPECULATIVE_DRAFT_MODEL_PATH:-}"
SPECULATIVE_DRAFT_MODEL_QUANTIZATION="${SPECULATIVE_DRAFT_MODEL_QUANTIZATION:-}"

if [[ ! -x "${SGLANG_PYTHON}" ]]; then
  echo "Missing SGLang Python: ${SGLANG_PYTHON}" >&2
  exit 1
fi

# SGLang/FlashInfer JIT needs nvcc, CUDA headers, and conventional toolkit
# linker paths. The local SGLang env is built from Python CUDA packages, so
# `/usr/local/cuda` may not exist and the package may expose `lib/` rather than
# the `lib64/` layout expected by generated FlashInfer build files.
if [[ -z "${CUDA_HOME:-}" ]]; then
  PYTHON_CUDA_HOME="$("${SGLANG_PYTHON}" - <<'PY'
import pathlib
import site

for site_dir in site.getsitepackages():
    candidate = pathlib.Path(site_dir) / "nvidia" / "cu13"
    if (candidate / "bin" / "nvcc").exists():
        print(candidate)
        break
PY
)"
  if [[ -n "${PYTHON_CUDA_HOME}" ]]; then
    CUDA_WRAPPER_HOME="${CUDA_WRAPPER_HOME:-${XDG_CACHE_HOME:-${HOME}/.cache}/qwen_diffusion/cuda/cu13}"
    mkdir -p "${CUDA_WRAPPER_HOME}/lib64/stubs"

    for cuda_dir in bin include nvvm; do
      if [[ -e "${PYTHON_CUDA_HOME}/${cuda_dir}" ]]; then
        ln -sfn "${PYTHON_CUDA_HOME}/${cuda_dir}" "${CUDA_WRAPPER_HOME}/${cuda_dir}"
      fi
    done

    if [[ -d "${PYTHON_CUDA_HOME}/lib" ]]; then
      while IFS= read -r lib_file; do
        ln -sfn "${lib_file}" "${CUDA_WRAPPER_HOME}/lib64/$(basename "${lib_file}")"
      done < <(find "${PYTHON_CUDA_HOME}/lib" -maxdepth 1 -mindepth 1 -type f | sort)

      if [[ ! -e "${CUDA_WRAPPER_HOME}/lib64/libcudart.so" ]]; then
        CUDART_SO="$(find "${PYTHON_CUDA_HOME}/lib" -maxdepth 1 -name 'libcudart.so.*' | sort -V | tail -n 1)"
        if [[ -n "${CUDART_SO}" ]]; then
          ln -sfn "${CUDART_SO}" "${CUDA_WRAPPER_HOME}/lib64/libcudart.so"
        fi
      fi
    fi

    if [[ ! -e "${CUDA_WRAPPER_HOME}/lib" || -L "${CUDA_WRAPPER_HOME}/lib" ]]; then
      ln -sfn lib64 "${CUDA_WRAPPER_HOME}/lib"
    fi

    if [[ -e /usr/lib/x86_64-linux-gnu/libcuda.so ]]; then
      ln -sfn /usr/lib/x86_64-linux-gnu/libcuda.so "${CUDA_WRAPPER_HOME}/lib64/stubs/libcuda.so"
    fi

    CUDA_HOME="${CUDA_WRAPPER_HOME}"
  fi
fi

if [[ -n "${CUDA_HOME:-}" ]]; then
  export CUDA_HOME
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib64/stubs:${CUDA_HOME}/lib:${LIBRARY_PATH:-}"
fi

# The SGLang NVFP4 CUTLASS JIT includes CUDA headers that depend on libcudacxx
# headers such as `nv/target`. FlashInfer vendors CCCL/libcudacxx, but SGLang's
# generated NVFP4 compile command does not add that include path.
CUDA_CCCL_INCLUDE="$("${SGLANG_PYTHON}" - <<'PY'
import pathlib
import site

for site_dir in site.getsitepackages():
    candidate = (
        pathlib.Path(site_dir)
        / "flashinfer"
        / "data"
        / "cccl"
        / "libcudacxx"
        / "include"
    )
    if (candidate / "nv" / "target").exists():
        print(candidate)
        break
PY
)"

if [[ -n "${CUDA_CCCL_INCLUDE}" ]]; then
  export CPATH="${CUDA_CCCL_INCLUDE}:${CPATH:-}"
  export CPLUS_INCLUDE_PATH="${CUDA_CCCL_INCLUDE}:${CPLUS_INCLUDE_PATH:-}"
fi

if [[ "${SKIP_VERSION_CHECK}" != "1" ]]; then
  "${SGLANG_PYTHON}" - "${SGLANG_MIN_VERSION}" <<'PY'
import sys
from packaging.version import Version
import sglang

required = Version(sys.argv[1])
current = Version(sglang.__version__)
if current < required:
    raise SystemExit(
        f"SGLang {current} found, but Qwen3.6 teacher serving should use >= {required}. "
        "Upgrade .venv-sglang or rerun with SKIP_VERSION_CHECK=1 for a smoke test."
    )
print(f"SGLang {current} ok")
PY
fi

cmd=(
  "${SGLANG_PYTHON}" -m sglang.launch_server
  --model-path "${MODEL_PATH}"
  --trust-remote-code
  --host "${HOST}"
  --port "${PORT}"
  --dtype "${DTYPE}"
  --context-length "${CONTEXT_LENGTH}"
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
  --attention-backend "${ATTENTION_BACKEND}"
  --prefill-attention-backend "${PREFILL_ATTENTION_BACKEND}"
  --decode-attention-backend "${DECODE_ATTENTION_BACKEND}"
  --fp8-gemm-backend "${FP8_GEMM_BACKEND}"
  --fp4-gemm-backend "${FP4_GEMM_BACKEND}"
  --tool-call-parser "${TOOL_CALL_PARSER}"
  --reasoning-parser "${REASONING_PARSER}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --show-time-cost
)

if [[ -n "${QUANTIZATION}" ]]; then
  cmd+=(--quantization "${QUANTIZATION}")
fi

if [[ -n "${LOAD_FORMAT}" ]]; then
  cmd+=(--load-format "${LOAD_FORMAT}")
fi

if [[ -n "${KV_CACHE_DTYPE}" && "${KV_CACHE_DTYPE}" != "auto" ]]; then
  cmd+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
fi

if [[ -n "${MAX_TOTAL_TOKENS}" ]]; then
  cmd+=(--max-total-tokens "${MAX_TOTAL_TOKENS}")
fi

if [[ -n "${CUDA_GRAPH_BACKEND_DECODE}" ]]; then
  cmd+=(--cuda-graph-backend-decode "${CUDA_GRAPH_BACKEND_DECODE}")
fi

if [[ -n "${CUDA_GRAPH_BACKEND_PREFILL}" ]]; then
  cmd+=(--cuda-graph-backend-prefill "${CUDA_GRAPH_BACKEND_PREFILL}")
fi

if [[ -n "${CUDA_GRAPH_MAX_BS_DECODE}" ]]; then
  cmd+=(--cuda-graph-max-bs-decode "${CUDA_GRAPH_MAX_BS_DECODE}")
fi

if [[ -n "${CUDA_GRAPH_MAX_BS_PREFILL}" ]]; then
  cmd+=(--cuda-graph-max-bs-prefill "${CUDA_GRAPH_MAX_BS_PREFILL}")
fi

if [[ -n "${CUDA_GRAPH_BS_DECODE}" ]]; then
  read -r -a cuda_graph_bs_decode_args <<< "${CUDA_GRAPH_BS_DECODE}"
  cmd+=(--cuda-graph-bs-decode "${cuda_graph_bs_decode_args[@]}")
fi

if [[ -n "${CUDA_GRAPH_BS_PREFILL}" ]]; then
  read -r -a cuda_graph_bs_prefill_args <<< "${CUDA_GRAPH_BS_PREFILL}"
  cmd+=(--cuda-graph-bs-prefill "${cuda_graph_bs_prefill_args[@]}")
fi

if [[ "${DISABLE_RADIX_CACHE}" == "1" ]]; then
  cmd+=(--disable-radix-cache)
fi

if [[ -n "${MAMBA_RADIX_CACHE_STRATEGY}" ]]; then
  cmd+=(--mamba-radix-cache-strategy "${MAMBA_RADIX_CACHE_STRATEGY}")
fi

if [[ -n "${MAX_MAMBA_CACHE_SIZE}" ]]; then
  cmd+=(--max-mamba-cache-size "${MAX_MAMBA_CACHE_SIZE}")
fi

if [[ -n "${MAMBA_FULL_MEMORY_RATIO}" ]]; then
  cmd+=(--mamba-full-memory-ratio "${MAMBA_FULL_MEMORY_RATIO}")
fi

if [[ "${ENABLE_INT8_MAMBA_CHECKPOINT}" == "1" ]]; then
  cmd+=(--enable-int8-mamba-checkpoint)
fi

if [[ -n "${INT8_MAMBA_CKPT_SIZE}" ]]; then
  cmd+=(--int8-mamba-ckpt-size "${INT8_MAMBA_CKPT_SIZE}")
fi

if [[ "${ENABLE_MTP}" == "1" ]]; then
  cmd+=(
    --speculative-algorithm "${SPECULATIVE_ALGORITHM}"
    --speculative-num-steps "${SPECULATIVE_NUM_STEPS}"
    --speculative-eagle-topk "${SPECULATIVE_EAGLE_TOPK}"
    --speculative-num-draft-tokens "${SPECULATIVE_NUM_DRAFT_TOKENS}"
  )

  if [[ -n "${SPECULATIVE_DFLASH_BLOCK_SIZE}" ]]; then
    cmd+=(--speculative-dflash-block-size "${SPECULATIVE_DFLASH_BLOCK_SIZE}")
  fi

  if [[ -n "${SPECULATIVE_ATTENTION_MODE}" ]]; then
    cmd+=(--speculative-attention-mode "${SPECULATIVE_ATTENTION_MODE}")
  fi

  if [[ -n "${SPECULATIVE_DRAFT_ATTENTION_BACKEND}" ]]; then
    cmd+=(--speculative-draft-attention-backend "${SPECULATIVE_DRAFT_ATTENTION_BACKEND}")
  fi

  if [[ -n "${SPECULATIVE_DRAFT_MODEL_PATH}" ]]; then
    cmd+=(--speculative-draft-model-path "${SPECULATIVE_DRAFT_MODEL_PATH}")
  fi

  if [[ -n "${SPECULATIVE_DRAFT_MODEL_QUANTIZATION}" ]]; then
    cmd+=(--speculative-draft-model-quantization "${SPECULATIVE_DRAFT_MODEL_QUANTIZATION}")
  fi
fi

echo "Launching SGLang teacher:"
printf ' %q' "${cmd[@]}"
echo
exec "${cmd[@]}"
