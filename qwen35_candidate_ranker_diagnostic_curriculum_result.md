# Qwen3.5 Candidate-Ranker Diagnostic Curriculum Result

Date: 2026-06-27

## Purpose

Test whether the 12-case public multi-call candidate-ranking artifact can be
packaged as a Fast-DLLM conversation curriculum and whether a tiny continuation
from checkpoint-275 moves the masked candidate-ranking metric.

This is diagnostic only. The corpus is built from the public 12-case eval
slice, so checkpoints trained on it are not promotable.

## Curriculum

Builder:

```bash
.venv-fastdllm/bin/python scripts/build_candidate_ranking_curriculum.py \
  --out-dir data/qwen35_9b_candidate_ranker_public12_diagnostic_curriculum \
  --block-size 1024 \
  --truncation-side left
```

Manifest:

- path:
  `data/qwen35_9b_candidate_ranker_public12_diagnostic_curriculum/train_agentic_mix.manifest`
- instances: `131`
- rejected: `0`
- hard remaining-failure rows: `48`
- tool-name rows: `31`
- candidate-ranker normal rows: `34`
- improved rows: `6`
- single-candidate rows: `12`
- label retention: no zero-label or partial-label truncation
- promotion: `promotion_allowed=false`

## One-Step Gate

Training command used `scripts/run_fastdllm_qwen35_9b_agentic_qlora_pilot.sh`
with:

- base: `models/qwen3.5-9b-fastdllm-init`
- starting adapter:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model`
- dataset:
  `data/qwen35_9b_candidate_ranker_public12_diagnostic_curriculum`
- output:
  `runs/fastdllm_qwen35_9b_candidate_ranker_public12_diag_from_ckpt275_step1_gate`
- block size: `1024`
- truncation: `left`
- max steps: `1`
- max train samples: `32`
- learning rate: `1e-6`

Result:

- checkpoint saved: yes
- train loss: `1.0847731828689575`
- train runtime: `3.1321s`
- readiness: `ready=true`

## Masked Candidate-Ranking Eval

Output:

`runs/candidate_ranking/public_multicall_qwen35_candidate_ranker_diag_ckpt1_masked_span_rank_v3_12_prefix_only.summary.json`

Delta report:

`qwen35_candidate_ranker_diagnostic_gate_result.md`

Result:

| Run | Overall | Tool names | Argument values |
| --- | ---: | ---: | ---: |
| checkpoint-275 | `80/86` | `31/31` | `49/55` |
| candidate-ranker diagnostic checkpoint-1 | `80/86` | `31/31` | `49/55` |

Delta:

- improved examples: `0`
- regressed examples: `0`
- remaining failures: `6`

## Interpretation

The candidate-ranker conversation objective is trainable and fits in the local
5090 QLoRA path, but a one-step index-selection SFT gate does not move the
masked-span candidate-ranking metric. This suggests the next useful learned
pressure should be closer to the actual failure surface:

- masked value-span CE on the selected candidate tokens
- row/table grounding examples where the target candidate is tied to a row
- a real ranker/verifier head or longer sidecar ranker training, evaluated as
  index selection rather than assumed to transfer to masked span scoring

Do not scale this exact one-step diagnostic as a promoted generator recipe.
