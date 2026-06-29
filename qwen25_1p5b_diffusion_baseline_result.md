# Qwen2.5 1.5B Fast-dLLM Tool-Call Baseline Result

Date: 2026-06-26

## Status

The local 1.5B Fast-dLLM lab model now has comparable strict tool-call metrics
on the same eval slices used for the Qwen3.6 teacher and Qwen3.5-9B AR
baseline.

This is not the target model. It is a cheap diffusion sampler/objective lab used
to expose failure modes before moving the loop to Qwen3.5-9B.

## Script

Added:

```text
scripts/eval_fastdllm_toolcall_cases.py
```

The script loads the local Fast-dLLM Qwen2.5-1.5B init model, optionally merges a
LoRA adapter, runs the Fast-dLLM block sampler, and scores generations through
the same `score_tool_calls` path as the AR baselines:

- valid tool JSON
- exact tool sequence
- exact tool-name multiset
- exact arguments
- schema validity
- extra, missing, and repeated calls
- unresolved mask examples
- generated tokens/sec

## Models Measured

```text
base/init:
  /home/mark/qwen_diffusion/models/qwen2.5-1.5b-fastdllm-init

public-data LoRA:
  /home/mark/qwen_diffusion/runs/fastdllm_qwen25_1p5b_toolcall_lora_clean_smoke

synthetic one-call LoRA:
  /home/mark/qwen_diffusion/runs/fastdllm_qwen25_1p5b_synthetic_onecall_lora
```

Sampling settings:

```text
block size: 32
small block size: 8
threshold: 0.9
temperature: 0
top_p: 0.95
peak CUDA allocation: about 3.34 GiB
RTX 5080 use: none
```

The Qwen3.6 teacher was stopped before these GPU-bound runs. The teacher/student
flow does not require both models to be live at the same time; saved JSONL labels
and eval outputs are enough except for future online KL/logit distillation.

## Result Table

Strict raw scoring, no repair:

| Model / prompt | Slice | Valid tool JSON | Exact sequence | Exact args | Extra / missing / repeated | Tokens/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| base init | synthetic one-call, 48 | 0/48 | 0/48 | 0/48 | 0 / 48 / 0 | 97.0 |
| base init | public one-call, 24 | 0/24 | 0/24 | 0/24 | 0 / 24 / 0 | 95.6 |
| base init | public multi-call, 12 | 0/12 | 0/12 | 0/12 | 0 / 31 / 0 | 93.5 |
| base init | synthetic tool-result, 10 | 0/10 | 0/10 | 0/10 | 0 / 10 / 0 | 94.3 |
| public-data LoRA | synthetic one-call, 48 | 0/48 | 0/48 | 0/48 | 0 / 48 / 0 | 110.3 |
| public-data LoRA | public one-call, 24 | 0/24 | 0/24 | 0/24 | 0 / 24 / 0 | 110.0 |
| public-data LoRA | public multi-call, 12 | 0/12 | 0/12 | 0/12 | 0 / 31 / 0 | 110.4 |
| public-data LoRA | synthetic tool-result, 10 | 0/10 | 0/10 | 0/10 | 0 / 10 / 0 | 110.8 |
| synthetic one-call LoRA, train-style prompt | synthetic one-call, 48 | 7/48 | 17/48 | 10/48 | 0 / 31 / 0 | 133.0 |
| synthetic one-call LoRA, train-style prompt | public one-call, 24 | 0/24 | 0/24 | 0/24 | 1 / 24 / 0 | 115.4 |
| synthetic one-call LoRA, train-style prompt | public multi-call, 12 | 0/12 | 0/12 | 0/12 | 5 / 31 / 1 | 118.5 |
| synthetic one-call LoRA, train-style prompt | synthetic tool-result, 10 | 0/10 | 0/10 | 0/10 | 1 / 10 / 0 | 122.7 |
| synthetic one-call LoRA, appended instruction | synthetic one-call, 48 | 12/48 | 13/48 | 4/48 | 12 / 31 / 6 | 125.1 |
| synthetic one-call LoRA, appended instruction | public one-call, 24 | 1/24 | 0/24 | 0/24 | 2 / 24 / 0 | 116.4 |
| synthetic one-call LoRA, appended instruction | public multi-call, 12 | 0/12 | 0/12 | 0/12 | 3 / 31 / 1 | 122.7 |
| synthetic one-call LoRA, appended instruction | synthetic tool-result, 10 | 0/10 | 0/10 | 0/10 | 6 / 10 / 1 | 121.2 |

Raw outputs and summaries are under:

```text
runs/fastdllm_qwen25_1p5b_diffusion_baseline/
```

## Interpretation

The base diffusion init produces no strict tool calls on these slices.

The public-data LoRA is also zero under strict scoring. The earlier public-data
training run was useful plumbing, but it did not learn robust tool-call
structure.

The synthetic one-call LoRA learns a narrow behavior:

- It improves synthetic exact sequence from 0/48 to 17/48.
- It gets exact arguments in 10/48 training-style held-out synthetic cases.
- It has no unresolved masks.
- It fails public one-call, public multi-call, and tool-result continuation.
- The AR-style appended instruction makes the model more likely to emit wrappers
  but increases extra/repeated calls and reduces exact arguments.

This confirms the next training loop should not spend more time on 1.5B generic
public LoRA. The useful next move is Qwen3.5-9B diffusion/QLoRA with offline
Qwen3.6 labels, structure-weighted loss, and tool-call curriculum data that
explicitly covers:

- exact JSON/tool-call wrappers
- arguments, not only tool names
- public schemas
- multi-call ordering
- tool-result continuation
- prompt-style robustness
