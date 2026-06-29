# Qwen3.5 Route-Delta Train-Only Mix Result

Date: 2026-06-27

## Purpose

Build a train-only curriculum from the checkpoint-5 route-delta failure report,
without using eval or heldout rows as training data.

The target failure classes are:

- one-call sequence retention
- scalar argument grounding
- text tool-result next-action retention
- OpenAI-style tool-result argument retention

Route-delta diagnostic source:

- `qwen35_public_train_candidate_value_span_route_delta.md`
- `runs/fastdllm_qwen35_9b_candidate_value_span_public_train_from_ckpt275_step10/checkpoint-5/route_delta_vs_current_routed_target.json`

## Builder

Script:

```bash
.venv-fastdllm/bin/python scripts/build_qwen35_route_delta_trainonly_mix.py
```

Output:

`data/qwen35_9b_route_delta_trainonly_mix_curriculum`

Manifest:

`data/qwen35_9b_route_delta_trainonly_mix_curriculum/train_agentic_mix.manifest`

## Data Mix

Accepted rows: `335`

Rejected rows: `0`

Source counts:

| Source | Rows |
| --- | ---: |
| `public_train_value_span` | `173` |
| `fastdllm_toolcall_train` | `64` |
| `synthetic_onecall_train` | `48` |
| `synthetic_toolresult_openai_train` | `30` |
| `synthetic_toolresult_text_train` | `20` |

Provenance:

- `contains_eval_slice=false`
- `diagnostic_only=false`
- `promotion_allowed=true`
- no eval/heldout row from the route-delta report is used as a training row

Token audit:

- block size: `1024`
- truncation side: `left`
- p50 length: `862`
- p90 length: `1429`
- p50 kept labels: `14`
- p90 kept labels: `102`
- full labels kept for all accepted rows

## One-Step Fit Gate

Command shape:

```bash
systemd-run --user --scope --quiet -p MemoryMax=28G -p MemorySwapMax=4G \
  env BUILD_CURRICULUM=0 \
  DATASET_DIR=/home/mark/qwen_diffusion/data/qwen35_9b_route_delta_trainonly_mix_curriculum \
  OUTPUT_DIR=/home/mark/qwen_diffusion/runs/fastdllm_qwen35_9b_route_delta_trainonly_mix_from_ckpt275_step1_gate \
  MAX_STEPS=1 \
  MAX_TRAIN_SAMPLES=32 \
  BLOCK_SIZE=1024 \
  LEARNING_RATE=1e-6 \
  SAVE_STEPS=1 \
  SAVE_TOTAL_LIMIT=1 \
  GRAD_ACCUM=4 \
  LORA_MODEL_PATH=/home/mark/qwen_diffusion/runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model \
  DISABLE_GROUP_TEXTS=1 \
  TRUNCATION_SIDE=left \
  VALUE_SPAN_LABEL_ONLY=1 \
  /home/mark/qwen_diffusion/scripts/run_fastdllm_qwen35_9b_agentic_qlora_pilot.sh
```

Output:

`runs/fastdllm_qwen35_9b_route_delta_trainonly_mix_from_ckpt275_step1_gate`

Result:

- start adapter: checkpoint-275
- max steps: `1`
- max train samples: `32`
- block size: `1024`
- train loss: `0.2080969214439392`
- train runtime: `10.54s`
- checkpoint saved: yes
- adapter saved: yes

## Interpretation

The route-delta train-only mix is valid and trainable. It should not be promoted
from the one-step gate. Its next use is a short checkpoint sweep with early
stopping against the exact lanes that blocked checkpoint-5 promotion.

Minimum next gates:

1. public one-call must stay `8/8` constrained sequence and arguments.
2. teacher-heldout one-call must recover at least the routed target:
   `8/8` constrained sequence and `6/8` constrained arguments.
3. public multi-call contextual projection must stay at least `8/12` sequence
   and `8/12` arguments.
4. public multi-call guarded planner must stay `11/12` sequence and `10/12`
   arguments.
5. synthetic text tool-result must stay `10/10` sequence and `9/10` arguments.
6. OpenAI-style tool-result must recover `10/10` sequence and `9/10` arguments.

Do not run a longer sweep unless the early teacher-heldout and OpenAI-style
tool-result gates stop regressing.

## One-Step Protected Eval Follow-Up

Date: 2026-06-28.

The one-step adapter was evaluated before any longer sweep:

`runs/fastdllm_qwen35_9b_route_delta_trainonly_mix_from_ckpt275_step1_gate/checkpoint-1/adapter_model`

Top-line results:

| Slice | Raw seq / args | Original constrained seq / args | Patched constrained/protected seq / args | Gate |
| --- | ---: | ---: | ---: | --- |
| public one-call | `3/8` / `2/8` | `8/8` / `8/8` | `8/8` / `8/8` | pass |
| teacher-heldout one-call | `1/8` / `0/8` | `8/8` / `6/8` | `8/8` / `6/8` | pass |
| synthetic text tool-result | `5/10` / `3/10` | `10/10` / `8/10` | `10/10` / `10/10` | pass after decoder fix |
| OpenAI-style tool-result | `6/10` / `6/10` | `10/10` / `9/10` | `10/10` / `9/10` | pass |
| public multi-call, direct constrained | `1/12` / `0/12` | `7/12` / `4/12` | `7/12` / `4/12` | fail versus checkpoint-5 |
| public multi-call, contextual projection | n/a | n/a | `7/12` / `7/12` | fail versus `8/12` / `8/12` gate |
| public multi-call, guarded planner | n/a | n/a | `11/12` / `10/12` | pass |

The synthetic text tool-result failure was not a model-only fix. It was a
constrained-decoder/projection bug: the decoder searched generated text before
tool-result context for missing string values, so it preserved `CUST99` instead
of copying `customer_id: "CUST-99"`, and it did not map
`email_subject -> subject`. The decoder now uses tool-result context first only
when explicit tool-result evidence is present, and recognizes conservative
tool-result aliases such as `email_subject`, `email_body`, `callback_date`, and
`callback_time`.

Artifacts:

- public one-call patched constrained:
  `runs/fastdllm_qwen35_9b_route_delta_trainonly_mix_from_ckpt275_step1_gate/checkpoint-1/public_onecall_8_contextfirst_projection_v3_nomodelrepair.summary.json`
- teacher-heldout patched constrained:
  `runs/fastdllm_qwen35_9b_route_delta_trainonly_mix_from_ckpt275_step1_gate/checkpoint-1/teacher_heldout_labelaware_8_contextfirst_projection_v3_nomodelrepair.summary.json`
- synthetic text patched constrained:
  `runs/fastdllm_qwen35_9b_route_delta_trainonly_mix_from_ckpt275_step1_gate/checkpoint-1/synthetic_toolresult_10_contextfirst_projection_v3_nomodelrepair.summary.json`
- synthetic text contextual projection:
  `runs/fastdllm_qwen35_9b_route_delta_trainonly_mix_from_ckpt275_step1_gate/checkpoint-1/synthetic_toolresult_10_contextual_projection_v3_nomodelrepair.summary.json`
- synthetic text patched argdiff:
  `runs/fastdllm_qwen35_9b_route_delta_trainonly_mix_from_ckpt275_step1_gate/checkpoint-1/synthetic_toolresult_10_contextfirst_projection_v3_argdiff.summary.json`
- OpenAI-style patched constrained:
  `runs/fastdllm_qwen35_9b_route_delta_trainonly_mix_from_ckpt275_step1_gate/checkpoint-1/synthetic_openai_toolresult_10_contextfirst_projection_v3_nomodelrepair.summary.json`
- public multi-call generation:
  `runs/fastdllm_qwen35_9b_route_delta_trainonly_mix_from_ckpt275_step1_gate/checkpoint-1/public_multicall_12_nomodelrepair.summary.json`
- public multi-call contextual projection:
  `runs/fastdllm_qwen35_9b_route_delta_trainonly_mix_from_ckpt275_step1_gate/checkpoint-1/public_multicall_12_contextual_projection.summary.json`
- public multi-call guarded planner:
  `runs/fastdllm_qwen35_9b_route_delta_trainonly_mix_from_ckpt275_step1_gate/checkpoint-1/public_multicall_12_sequence_planner_projection.summary.json`

Decision:

- Do not promote the one-step route-delta adapter.
- Do not launch a longer sweep of this exact mix as the next default. It
  recovers the previous teacher-heldout/OpenAI-style blocker under patched
  constrained decoding, but it loses the public multi-call intermediate signal
  that made checkpoint-5 interesting.
- Keep the context-first tool-result decoder rule as a protected-path
  improvement. It is runtime protection, not model-side learning.
- The next learned target should be narrower: train or score the public
  multi-call planner/value-selection failures directly, especially the
  voice-command camera case and the motion-detector installation-code mismatch,
  while preserving tool-result context-copy gates.
