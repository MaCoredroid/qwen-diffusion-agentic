# Qwen3.5 Synthetic Multi-Call Planner Distill Result

Date: 2026-06-28

## Purpose

Move the successful protected planner decisions for the synthetic multi-call
failure analogues into a train-only distillation corpus, then run a cheap
one-step QLoRA gate from the active checkpoint-275 adapter.

This tests plumbing and whether a tiny synthetic planner-only continuation
moves generation. It is not a promotion run.

## Artifacts

- builder:
  `scripts/build_synthetic_multicall_planner_distill_curriculum.py`
- curriculum:
  `data/qwen35_9b_synthetic_multicall_planner_distill_curriculum/train_agentic_mix.json`
- manifest:
  `data/qwen35_9b_synthetic_multicall_planner_distill_curriculum/train_agentic_mix.manifest`
- independent overlap audit:
  `runs/synthetic_multicall_failure_analogues/synthetic_planner_distill_vs_public_multicall_overlap.json`
- one-step adapter:
  `runs/fastdllm_qwen35_9b_synthetic_multicall_planner_from_ckpt275_step1_gate/checkpoint-1/adapter_model`
- training log:
  `logs/fastdllm_qwen35_9b_synthetic_multicall_planner_from_ckpt275_step1_gate.log`
- new-adapter synthetic eval:
  `runs/fastdllm_qwen35_9b_synthetic_multicall_planner_from_ckpt275_step1_gate/synthetic_multicall_failure_analogues_8.summary.json`
- checkpoint-275 baseline synthetic eval:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/synthetic_multicall_failure_analogues_8_baseline.summary.json`
- candidate-index examples:
  `data/candidate_ranking/synthetic_multicall_failure_analogue_index_ranking.jsonl`
- diffusion-init candidate-index eval:
  `runs/candidate_ranking/synthetic_multicall_failure_analogue_diffusion_init_index_rank.summary.json`
- checkpoint-275 candidate-index eval:
  `runs/candidate_ranking/synthetic_multicall_failure_analogue_ckpt275_index_rank.summary.json`
- step-1 candidate-index eval:
  `runs/candidate_ranking/synthetic_multicall_failure_analogue_step1_index_rank.summary.json`

## Curriculum Build

Command:

```bash
.venv-fastdllm/bin/python scripts/build_synthetic_multicall_planner_distill_curriculum.py \
  --cases-jsonl data/toolcall_eval/synthetic_multicall_failure_analogues.jsonl \
  --planner-jsonl runs/synthetic_multicall_failure_analogues/sequence_planner_bad_draft_safe_seqmismatch.jsonl \
  --out-dir data/qwen35_9b_synthetic_multicall_planner_distill_curriculum \
  --target-source planner \
  --accept-mode exact_arguments \
  --tool-schema-mode compact \
  --prompt-mode instruction \
  --repeat 3 \
  --block-size 1024 \
  --truncation-side right \
  --exclude-eval-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl
```

Manifest result:

- rows: `24`
- unique synthetic cases: `8`
- deduped equivalent rows: `8`
- target rejected: `0`
- label rejected: `0`
- public multi-call eval overlap: `0`
- no eval leakage: `true`
- target exactness:
  - voice-command camera: `4/4` sequence, `4/4` arguments
  - security installation codes: `4/4` sequence, `4/4` arguments
- label retention:
  - kept-label min/p50/p90/max: `107/111/120/120`
  - zero-after-truncation: `0`
  - partial-after-truncation: `0`

Independent overlap audit against `public_multicall_hermes_smoke`:

- train records: `24`
- eval records: `12`
- exact overlap: `0`
- user overlap: `0`

## One-Step Gate

Source adapter:

```text
runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model
```

Output adapter:

```text
runs/fastdllm_qwen35_9b_synthetic_multicall_planner_from_ckpt275_step1_gate/checkpoint-1/adapter_model
```

Training ran under a systemd user scope with `MemoryMax=28G` and
`MemorySwapMax=4G`.

Training result:

- train samples: `24`
- max steps: `1`
- train loss: `1.4955649375915527`
- train runtime: `3.1357s`
- adapter saved: yes
- readiness: `ready=true`

## Generation Eval

Both adapters were evaluated on
`data/toolcall_eval/synthetic_multicall_failure_analogues.jsonl` with:

- full-context sampling
- forced `<tool_call>\n` prefix
- max new tokens: `256`
- block size: `32`
- small block size: `8`
- constrained tool decoding
- sequence-preserving constrained projection
- constrained max calls: `3`

| adapter | raw sequence | raw arguments | constrained sequence | constrained arguments |
| --- | ---: | ---: | ---: | ---: |
| checkpoint-275 baseline | `1/8` | `0/8` | `2/8` | `0/8` |
| synthetic planner step-1 | `1/8` | `0/8` | `2/8` | `0/8` |

## Interpretation

The corpus is clean and trainable, and the one-step gate saves correctly. But
one step of synthetic planner-only SFT does not move raw or constrained
generation on the synthetic analogue eval.

This is an expected negative result, not a blocker. It says the safe planner
decisions are not yet absorbed into the generator by a tiny continuation. The
next model-side attempt should either:

- use a longer but still small sweep on this corpus mixed with retention rows,
- train a candidate/tool/value selector objective instead of full generation,
- or move the successful planner rules into generation-time constraints before
asking the model to learn them.

Do not promote this adapter.

## Candidate Index-Ranking Diagnostic

The failed parallel run was repeated serially under the same user-scope memory
cap (`MemoryMax=28G`, `MemorySwapMax=4G`) to avoid loading two 9B copies on the
5090 at once.

This diagnostic asks the diffusion model to score a masked single-token answer
over candidate indices rather than generate a full tool-call chain. It checks
whether the one-step planner continuation changed a simpler internal selection
preference for the two synthetic failure families.

| adapter | overall | tool-name voice-command camera | argument-value security code |
| --- | ---: | ---: | ---: |
| diffusion init, no adapter | `6/8` | `2/4` | `4/4` |
| checkpoint-275 baseline | `7/8` | `3/4` | `4/4` |
| synthetic planner step-1 | `7/8` | `3/4` | `4/4` |

Margins:

| adapter | min | p50 | p90 | max |
| --- | ---: | ---: | ---: | ---: |
| diffusion init, no adapter | `-0.875` | `3.875` | `4.375` | `4.5` |
| checkpoint-275 baseline | `-0.25` | `4.375` | `4.625` | `4.75` |
| synthetic planner step-1 | `-0.25` | `4.375` | `4.625` | `4.75` |

The only miss for both adapters is:

```text
synthetic_voice_command_camera_003:
  target: activate_voice_command
  predicted: set_thermostat
  target margin: -0.25
```

Interpretation:

- The security-code argument-value ranking problem is already solved by the
  diffusion init on this synthetic slice.
- Checkpoint-275 shows a small real model-side lift over diffusion init on the
  voice-command camera tool-name ranking family: `2/4 -> 3/4`.
- The voice-command camera conflict is still not fully solved as a model-side
  selection preference.
- The one-step planner SFT did not move either generation behavior or masked
  index-ranking behavior, so this adapter remains a plumbing gate only.
- The next learned route should not be another one-step planner-only
  continuation. A better next probe is either a longer low-LR retention-mixed
  sweep or an explicit tool-name/value selector objective with heldout
  promotion gates.

Follow-up leave-one-out selector probe:

```text
qwen35_synthetic_candidate_index_leaveone_result.md
```

That probe holds out `synthetic_voice_command_camera_003`, trains on the other
seven synthetic selector examples, and moves the heldout masked selector choice
to correct by checkpoint-10/20/30. It still does not improve full tool-call
generation, so selector pressure should be treated as a side objective or
sidecar signal rather than a promoted generator update.
