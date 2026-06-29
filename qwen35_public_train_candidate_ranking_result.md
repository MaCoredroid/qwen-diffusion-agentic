# Qwen3.5 Public-Train Candidate Ranking Result

Date: 2026-06-27

## Purpose

Move candidate/value-ranking supervision off the public 12-case eval slice and
onto the public train multi-call slice. This creates a non-eval curriculum for
future row/time/value grounding work.

## Train-Slice Materialization

Builder:

```bash
.venv-fastdllm/bin/python scripts/materialize_conversation_toolcall_cases.py \
  --input-json data/fastdllm_toolcall_train/train_toolcall.json \
  --out-jsonl data/toolcall_eval/public_train_multicall_gold_cases.jsonl \
  --min-tool-calls 2
```

Result:

- source records: `96`
- multi-call records: `56`
- skipped one-call records: `40`
- tool-call histogram: `2:22`, `3:28`, `4:5`, `7:1`
- no eval leakage: `true`

## Schedule And Candidate Artifact

Artifacts:

- tokenized block plan:
  `runs/tool_sensitive_block_plans/public_train_multicall_gold_blocks_tokenized_with_ids.jsonl`
- sampler schedule:
  `runs/tool_sensitive_block_plans/public_train_multicall_gold_sampler_schedule_with_ids.jsonl`
- schedule with candidates:
  `runs/tool_sensitive_block_plans/public_train_multicall_gold_sampler_schedule_with_toolname_candidates.jsonl`
- ranking examples:
  `data/candidate_ranking/public_train_multicall_toolname_argument_ranking.jsonl`
- train conversation file:
  `data/candidate_ranking/public_train_multicall_toolname_argument_ranking.train.json`

Coverage:

- train records: `56`
- tool calls: `155`
- argument-value spans in block plan: `378`
- argument blocks augmented with candidates: `224`
- argument blocks with sequence candidates: `169`
- tool-name blocks augmented: `170`
- tool-name blocks with target candidate: `155`
- ranking examples: `338`
- usable ranking examples: `299`
- usable tool-name examples: `155`
- usable argument-value examples: `144`
- target missing from candidate set: `39`

## Masked Ranking Baselines

Diffusion-init baseline:

- output:
  `runs/candidate_ranking/public_train_multicall_qwen35_diffusion_init_masked_span_rank.summary.json`
- overall: `294/299`
- tool names: `154/155`
- argument values: `140/144`

Checkpoint-275 baseline:

- output:
  `runs/candidate_ranking/public_train_multicall_qwen35_ckpt275_masked_span_rank.summary.json`
- overall: `295/299`
- tool names: `155/155`
- argument values: `140/144`

Delta report:

`qwen35_public_train_candidate_ranking_delta_result.md`

Delta:

- improved rows: `2`
- regressed rows: `1`
- remaining rows after checkpoint-275: `4`
- remaining failures are all argument values: `19:00` vs `11:00`,
  `23:00` vs `22:00`, `0` vs `1`, and `mat_001` vs `mat_002`

## Train Curriculum

Builder:

```bash
.venv-fastdllm/bin/python scripts/build_candidate_ranking_curriculum.py \
  --examples-jsonl data/candidate_ranking/public_train_multicall_toolname_argument_ranking.jsonl \
  --delta-json qwen35_public_train_candidate_ranking_delta_result.json \
  --out-dir data/qwen35_9b_candidate_ranker_public_train_curriculum \
  --block-size 1024 \
  --truncation-side left \
  --no-contains-eval-slice \
  --no-diagnostic-only
```

Manifest:

`data/qwen35_9b_candidate_ranker_public_train_curriculum/train_agentic_mix.manifest`

Result:

- accepted instances: `329`
- rejected labels: `0`
- skipped unusable examples: `39`
- hard remaining-failure repeats: `32`
- tool-name rows: `154`
- no zero-label or partial-label truncation
- contains eval slice: `false`
- promotion allowed by data provenance: `true`, subject to heldout gates

## One-Step Gate

Training output:

`runs/fastdllm_qwen35_9b_candidate_ranker_public_train_from_ckpt275_step1_gate`

Result:

- start adapter: checkpoint-275
- max steps: `1`
- max train samples: `32`
- block size: `1024`
- train loss: `1.3103340864181519`
- checkpoint saved: yes

Heldout public-12 candidate-ranking eval:

- output:
  `runs/candidate_ranking/public12_qwen35_candidate_ranker_public_train_ckpt1_masked_span_rank_v3_prefix_only.summary.json`
- delta report:
  `qwen35_public_train_candidate_ranker_gate_result.md`
- checkpoint-275: `80/86` overall, `31/31` tool names, `49/55` arguments
- train-slice ranker checkpoint-1: `80/86` overall, `31/31` tool names,
  `49/55` arguments
- improved rows: `0`
- regressed rows: `0`

## Interpretation

The train-slice candidate-ranking pipeline is now reproducible and non-eval.
The current model already scores very high on masked train examples, and
checkpoint-275 mainly fixes the last train tool-name miss. The remaining public
heldout failures do not move with a one-step index-selection SFT gate.

Next training pressure should target the value itself rather than the index
instruction format:

- selected-value span CE on the target candidate tokens
- row/table grounding prompts where the model must bind candidate value to the
  correct row
- a separately evaluated verifier/ranker head if we want index selection as a
  real sidecar
