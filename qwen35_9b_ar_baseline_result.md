# Qwen3.5-9B AR Baseline Result

Date: 2026-06-26

## Status

Qwen3.5-9B AR baseline is measured on the same local tool-call slices used for
the Qwen3.6 teacher/reference. This run used Alienware's RTX 5080 so the local
RTX 5090 could keep serving the live Qwen3.6-27B NVFP4 MTP teacher.

Model metadata checked from Hugging Face:

```text
model: Qwen/Qwen3.5-9B
architecture: Qwen3_5ForConditionalGeneration
model_type: qwen3_5
lastModified: 2026-03-02
gated: false
```

## Hardware / Runtime

```text
host: mark-Alienware-Aurora-ACT1250
GPU: RTX 5080, 16 GB
Python: /home/mark/qwen_diffusion/.venv
PyTorch: 2.12.1+cu130
Transformers: 5.12.1
bitsandbytes: 0.49.2
load: 4-bit NF4 + double quant, bf16 compute
peak CUDA allocated: 7.75 GiB
```

The Transformers path emitted this warning:

```text
The fast path is not available ... Falling back to torch implementation.
```

So this is a correctness baseline, not a speed-optimized serving result. The
missing fast kernels are FLA and `causal-conv1d`.

## Script

The baseline uses:

```text
scripts/eval_transformers_toolcall_cases.py
```

The script can run a suite of evals after one model load via repeated
`--eval name:input:output[:limit]` arguments.

## Command

```bash
HF_HUB_DISABLE_PROGRESS_BARS=1 \
.venv/bin/python scripts/eval_transformers_toolcall_cases.py \
  --model Qwen/Qwen3.5-9B \
  --load-in-4bit \
  --bnb-4bit-use-double-quant \
  --torch-dtype bf16 \
  --max-new-tokens 1024 \
  --eval synthetic_onecall:data/toolcall_eval/synthetic_onecall_smoke.jsonl:data/toolcall_eval/qwen35_9b_bnb4_synthetic_onecall_48.jsonl \
  --eval public_onecall:data/toolcall_eval/public_onecall_hermes_smoke.jsonl:data/toolcall_eval/qwen35_9b_bnb4_public_onecall_24.jsonl \
  --eval public_multicall:data/toolcall_eval/public_multicall_hermes_smoke.jsonl:data/toolcall_eval/qwen35_9b_bnb4_public_multicall_12.jsonl \
  --eval synthetic_toolresult:data/toolcall_eval/synthetic_toolresult_smoke.jsonl:data/toolcall_eval/qwen35_9b_bnb4_synthetic_toolresult_10.jsonl
```

Log:

```text
logs/qwen35_9b_baseline_20260626_124719.log
```

## Results

| Slice | Valid calls | Exact sequence | Exact args | Schema valid | Extra/missing/repeated |
| --- | ---: | ---: | ---: | ---: | ---: |
| synthetic one-call, 48 | 48/48 | 48/48 | 48/48 | 48/48 | 0 / 0 / 0 |
| public Hermes one-call, 24 | 24/24 | 17/24 | 13/24 | 22/24 | 10 / 0 / 2 |
| public Hermes multi-call, 12 | 12/12 | 11/12 | 10/12 | 12/12 | 1 / 1 / 0 |
| synthetic tool-result, 10 | 10/10 | 10/10 | 10/10 | 10/10 | 0 / 0 / 0 |

Elapsed generation time after model load:

```text
synthetic one-call: 71.99s
public one-call: 102.32s
public multi-call: 57.55s
synthetic tool-result: 15.18s
```

## Comparison To Qwen3.6 Teacher

Qwen3.5-9B is strong enough for controlled synthetic gates and matches the
Qwen3.6 teacher on the current 12-case public multi-call slice. It is weaker on
the public one-call slice:

```text
Qwen3.6-27B public one-call exact sequence: 21/24
Qwen3.5-9B public one-call exact sequence: 17/24

Qwen3.6-27B public one-call exact arguments: 18/24
Qwen3.5-9B public one-call exact arguments: 13/24
```

The key 9B failure mode is over-calling. On the 24-case public one-call slice it
emitted extra calls in 7 records, including one repeated-call style failure:

```text
find_interactive_media_collaboration_tool
setup_interactive_media_collaboration
setup_interactive_media_collaboration
setup_interactive_media_collaboration
```

## Takeaway

Use Qwen3.6-27B as the labeler/repair teacher. Use Qwen3.5-9B as the first real
GDN-family AR baseline and diffusion target, but train/evaluate explicitly
against public one-call over-calling and repeated-action loops.

Next gate:

1. Add the 10 small code-generation tasks from the roadmap.
2. Run the current local diffusion baselines on the same tool-call slices.
3. Move the curriculum to the Qwen3.5-9B diffusion target once conversion is
   available.
