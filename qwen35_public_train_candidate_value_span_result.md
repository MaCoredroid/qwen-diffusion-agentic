# Qwen3.5 Public-Train Candidate Value-Span Result

Date: 2026-06-27

## Purpose

Test a closer-to-failure-surface curriculum than index selection. Instead of
training the model to output the candidate index, train it to output the exact
JSON value span for argument-value examples.

## Builder Change

`scripts/build_candidate_ranking_curriculum.py` now supports:

- `--answer-mode index`: original candidate-index answer
- `--answer-mode target_text`: exact JSON target-span answer
- `--include-kinds`: choose `tool_name`, `argument_value`, or both

## Curriculum

Command:

```bash
.venv-fastdllm/bin/python scripts/build_candidate_ranking_curriculum.py \
  --examples-jsonl data/candidate_ranking/public_train_multicall_toolname_argument_ranking.jsonl \
  --delta-json qwen35_public_train_candidate_ranking_delta_result.json \
  --out-dir data/qwen35_9b_candidate_value_span_public_train_curriculum \
  --answer-mode target_text \
  --include-kinds argument_value \
  --block-size 1024 \
  --truncation-side left \
  --no-contains-eval-slice \
  --no-diagnostic-only
```

Manifest:

`data/qwen35_9b_candidate_value_span_public_train_curriculum/train_agentic_mix.manifest`

Result:

- accepted instances: `173`
- rejected labels: `0`
- skipped unusable examples: `39`
- skipped tool-name examples: `155`
- hard remaining-failure repeats: `32`
- answer mode: `target_text`
- include kinds: `argument_value`
- no zero-label or partial-label truncation
- p50 kept labels: `8`
- p90 kept labels: `11`
- contains eval slice: `false`
- promotion allowed by data provenance: `true`, subject to heldout gates

## One-Step Gate

Training output:

`runs/fastdllm_qwen35_9b_candidate_value_span_public_train_from_ckpt275_step1_gate`

Result:

- start adapter: checkpoint-275
- max steps: `1`
- max train samples: `32`
- block size: `1024`
- train loss: `3.019835948944092`
- checkpoint saved: yes

## Heldout Public-12 Candidate Ranking

Output:

`runs/candidate_ranking/public12_qwen35_candidate_value_span_public_train_ckpt1_masked_span_rank_v3_prefix_only.summary.json`

Delta report:

`qwen35_public_train_candidate_value_span_gate_result.md`

Result:

| Run | Overall | Tool names | Argument values |
| --- | ---: | ---: | ---: |
| checkpoint-275 | `80/86` | `31/31` | `49/55` |
| value-span checkpoint-1 | `80/86` | `31/31` | `49/55` |

Delta:

- improved rows: `0`
- regressed rows: `0`
- remaining failures: `6`

## Short Sweep

Because the one-step gate was neutral, a short 10-step continuation was run from
checkpoint-275 using the same public-train value-span curriculum:

Output:

`runs/fastdllm_qwen35_9b_candidate_value_span_public_train_from_ckpt275_step10`

Training:

- max steps: `10`
- save steps: `5`
- max train samples: `64`
- learning rate: `1e-6`
- block size: `1024`
- train loss: `2.2868268966674803`
- logged step-5 loss: `2.4903`
- logged step-10 loss: `2.0833`

Heldout public-12 masked candidate ranking:

| Run | Overall | Tool names | Argument values |
| --- | ---: | ---: | ---: |
| checkpoint-275 | `80/86` | `31/31` | `49/55` |
| value-span checkpoint-5 | `81/86` | `31/31` | `50/55` |
| value-span checkpoint-10 | `80/86` | `31/31` | `49/55` |

Step-5 delta:

- improved rows: `1`
- regressed rows: `0`
- remaining failures: `5`
- improved failure: public-12 invoice `invoice_id=INV-301` for call 2,
  previously predicted `INV-303`

Step-10 falls back to the checkpoint-275 candidate-ranking score, so the useful
point in this sweep is checkpoint-5, not the final checkpoint.

## Public One-Call Generation Gate

Checkpoint-5 was then tested on the cheap public one-call generation gate,
without the slower model-repair pass:

```bash
.venv-fastdllm/bin/python scripts/eval_fastdllm_toolcall_cases.py \
  --base-model models/qwen3.5-9b-fastdllm-init \
  --adapter runs/fastdllm_qwen35_9b_candidate_value_span_public_train_from_ckpt275_step10/checkpoint-5/adapter_model \
  --tokenizer-path models/qwen3.5-9b-fastdllm-init \
  --input-jsonl data/toolcall_eval/public_onecall_hermes_smoke.jsonl \
  --out-jsonl runs/fastdllm_qwen35_9b_candidate_value_span_public_train_from_ckpt275_step10/checkpoint-5/public_onecall_8_nomodelrepair.jsonl \
  --limit 8 \
  --block-size 32 \
  --small-block-size 8 \
  --max-new-tokens 96 \
  --threshold 0.9 \
  --temperature 0.0 \
  --top-p 0.95 \
  --conversation-template fast_dllm_v2 \
  --full-context-sampling \
  --repair-mode schema \
  --constrained-tool-decoding \
  --constrained-max-calls 1 \
  --no-merge-adapter
```

Comparable first-pass results:

| Run | Raw sequence | Raw arguments | Constrained sequence | Constrained arguments |
| --- | ---: | ---: | ---: | ---: |
| checkpoint-275 | `3/8` | `2/8` | `8/8` | `5/8` |
| value-span checkpoint-5 | `3/8` | `2/8` | `8/8` | `8/8` |

## Public Multi-Call Generation Gate

Checkpoint-5 was then tested on the 12-case public multi-call slice without the
slower learned model-repair pass:

```bash
systemd-run --user --scope --quiet -p MemoryMax=28G -p MemorySwapMax=4G \
  .venv-fastdllm/bin/python scripts/eval_fastdllm_toolcall_cases.py \
  --base-model models/qwen3.5-9b-fastdllm-init \
  --adapter runs/fastdllm_qwen35_9b_candidate_value_span_public_train_from_ckpt275_step10/checkpoint-5/adapter_model \
  --tokenizer-path models/qwen3.5-9b-fastdllm-init \
  --input-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --out-jsonl runs/fastdllm_qwen35_9b_candidate_value_span_public_train_from_ckpt275_step10/checkpoint-5/public_multicall_12_nomodelrepair.jsonl \
  --limit 12 \
  --block-size 32 \
  --small-block-size 8 \
  --max-new-tokens 384 \
  --threshold 0.9 \
  --temperature 0.0 \
  --top-p 0.95 \
  --conversation-template fast_dllm_v2 \
  --full-context-sampling \
  --repair-mode schema \
  --constrained-tool-decoding \
  --no-merge-adapter
```

Artifacts:

- first pass:
  `runs/fastdllm_qwen35_9b_candidate_value_span_public_train_from_ckpt275_step10/checkpoint-5/public_multicall_12_nomodelrepair.summary.json`
- sequence-preserving projection:
  `runs/fastdllm_qwen35_9b_candidate_value_span_public_train_from_ckpt275_step10/checkpoint-5/public_multicall_12_sequence_preserve.summary.json`
- contextual projection:
  `runs/fastdllm_qwen35_9b_candidate_value_span_public_train_from_ckpt275_step10/checkpoint-5/public_multicall_12_contextual_projection.summary.json`
- guarded sequence-planner projection:
  `runs/fastdllm_qwen35_9b_candidate_value_span_public_train_from_ckpt275_step10/checkpoint-5/public_multicall_12_sequence_planner_projection.summary.json`

Comparable public multi-call results:

| Stage | Checkpoint-275 sequence | Checkpoint-275 args | Checkpoint-5 sequence | Checkpoint-5 args |
| --- | ---: | ---: | ---: | ---: |
| raw | `1/12` | `0/12` | `1/12` | `0/12` |
| direct constrained | `7/12` | `4/12` | `8/12` | `5/12` |
| contextual projection | `7/12` | `7/12` | `8/12` | `8/12` |
| guarded planner projection | `11/12` | `10/12` | `11/12` | `10/12` |

Runtime:

- elapsed: `348.8s`
- generated tokens/sec: `6.84`
- CUDA max allocated: `18.3 GiB`
- CUDA max reserved: `28.8 GiB`

Row-level delta versus the active checkpoint-275 public multi-call path:

- direct constrained improves `adc48a37-6341-4ea6-972a-8ec2b5421321` on both
  sequence and arguments, with no regressions
- contextual projection improves the same row on sequence and arguments, with
  no regressions
- guarded planner projection ties checkpoint-275 exactly: no improved rows and
  no regressed rows

During this gate, `scripts/rescore_scalar_repair_contextual_projection.py` was
fixed to normalize malformed quoted scalar strings only when the cleaned value
appears in prompt context, and to prefer ID-specific context selection before
that cleanup for ID-like properties. This fixes the camera row where
`"front_door` needed cleanup in the first two calls while the third call still
needed context selection to `front_garden`.

## Remaining Split-Route Lanes

Checkpoint-5 was then run through the remaining cheap split-route lanes with the
same adapter and no learned model-repair pass:

- row-level delta report:
  `qwen35_public_train_candidate_value_span_route_delta.md`
- machine-readable delta:
  `runs/fastdllm_qwen35_9b_candidate_value_span_public_train_from_ckpt275_step10/checkpoint-5/route_delta_vs_current_routed_target.json`
- teacher-train one-call:
  `runs/fastdllm_qwen35_9b_candidate_value_span_public_train_from_ckpt275_step10/checkpoint-5/teacher_train_labelaware_12_nomodelrepair.summary.json`
- teacher-heldout one-call:
  `runs/fastdllm_qwen35_9b_candidate_value_span_public_train_from_ckpt275_step10/checkpoint-5/teacher_heldout_labelaware_8_nomodelrepair.summary.json`
- synthetic text tool-result:
  `runs/fastdllm_qwen35_9b_candidate_value_span_public_train_from_ckpt275_step10/checkpoint-5/synthetic_toolresult_10_nomodelrepair.summary.json`
- OpenAI-style tool-result:
  `runs/fastdllm_qwen35_9b_candidate_value_span_public_train_from_ckpt275_step10/checkpoint-5/synthetic_openai_toolresult_10_grounded_projection_v2_nomodelrepair.summary.json`

Six-lane comparison against the current routed target:

| Slice | Current routed raw | Current routed protected | Checkpoint-5 raw | Checkpoint-5 protected |
| --- | ---: | ---: | ---: | ---: |
| public one-call | `4/8`, `3/8` | `8/8`, `8/8` | `3/8`, `2/8` | `8/8`, `8/8` |
| teacher-train one-call | `2/12`, `2/12` | `11/12`, `6/12` | `2/12`, `2/12` | `10/12`, `6/12` |
| teacher-heldout one-call | `2/8`, `1/8` | `8/8`, `6/8` | `1/8`, `0/8` | `7/8`, `5/8` |
| public multi-call planner | direct raw `1/12`, `0/12` | `11/12`, `10/12` | direct raw `1/12`, `0/12` | `11/12`, `10/12` |
| synthetic text tool-result | `6/10`, `4/10` | `10/10`, `9/10` | `5/10`, `3/10` | `10/10`, `9/10` |
| OpenAI-style tool-result | `6/10`, `6/10` | `10/10`, `9/10` | `7/10`, `7/10` | `10/10`, `8/10` |

Each cell is `exact sequence`, `exact arguments`.

Decision:

- Do not replace the current split-route target with checkpoint-5.
- Keep checkpoint-5 as a value-span/constrained-decoding candidate because it
  improves public multi-call constrained/contextual stages and public one-call
  constrained arguments.
- Keep staged checkpoint-24 on the one-call and synthetic text tool-result
  generator lanes because checkpoint-5 is weaker on raw and teacher-heldout
  protected metrics.
- Keep checkpoint-275 active on OpenAI-style tool-result because checkpoint-5
  improves raw output but regresses protected exact arguments from `9/10` to
  `8/10`.

## Interpretation

The exact value-span objective is mechanically valid and provides more relevant
assistant-label tokens than index selection, but a one-step continuation still
does not move heldout masked value ranking. A short sweep does move the public
heldout ranking at checkpoint-5 and improves the cheap public one-call
constrained argument gate without raw regression. The broader public multi-call
gate is positive at the direct constrained and contextual-projection stages, and
ties the active protected planner score after the scalar-projection fix.

Checkpoint-5 is therefore a safe candidate for broader split-route evaluation,
but it is not a full replacement for checkpoint-275 yet: raw public multi-call
behavior is still `1/12` sequence and `0/12` arguments, and the final protected
planner only ties the active route. The remaining split-route lanes confirm the
same conclusion: checkpoint-5 is useful signal, not a promoted route.

Next options:

- use checkpoint-5 failures to build train-only row/table grounding and
  OpenAI-style tool-result anti-regression examples
- add explicit row/table grounding prompts for the remaining row-alignment
  failure modes
- add a true candidate-ranker/verifier head if index selection is meant to be
  used as a sidecar rather than transferred implicitly into generation logits
