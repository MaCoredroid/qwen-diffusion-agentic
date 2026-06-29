# Qwen3.5 Live-Planner Evidence-Only Selector Sampler Result

Date: 2026-06-28

## Purpose

Remove the remaining planner-target-inclusion caveat from the live planner
arg-sketch sampler gate.

The previous live-planner sampler result used candidate sets that included the
planner target value for every argument span. This run uses evidence-derived
candidates only: request text, schemas, available tool names, path-aware table
rows, quoted strings, IDs, dates, enums/booleans, and local call sketches.

This is still a protected runtime result, not raw diffusion model promotion:
the live planner is a sidecar/post-processing route and structural JSON/tool
tags/stop boundaries are still guarded.

## Code Changes

Updated `scripts/diagnose_schedule_value_candidates.py`:

- resolve nested schemas using `json_path` / `argument_path`, so fields like
  `expense_data[3].date` and `invoice_data[2].due_date` use the correct item
  schema;
- support empty-string candidates when they are key/schema-plausible;
- add focused evidence extraction for:
  - ISO date strings;
  - booleans from enable/activate/turn-off language;
  - snake-case values such as `energy_saving`;
  - symbolic language tokens such as `Q#`;
  - common room/location/door phrases;
  - command phrases such as `Activate security cameras in away mode`;
  - capitalized target/model phrases such as `Quantinuum H-Series Emulator`;
  - markdown-table row values keyed by path and row index.

The table extraction is the important discriminator: broad evidence candidates
made the selector confuse neighboring financial table rows. Row-local pruning
reduced candidate count while preserving coverage.

## Evidence Candidate Coverage

Input live schedule:

```text
runs/tool_sensitive_block_plans/public_multicall_live_v5_sequence_planned_sampler_schedule_with_ids.jsonl
```

Evidence-only candidate schedule:

```text
runs/tool_sensitive_block_plans/public_multicall_live_v5_sequence_planned_sampler_schedule_with_candidates_evidence_v5.summary.json
```

Coverage:

- records: `12`
- argument blocks augmented: `100`
- argument blocks with sequence candidates: `100`
- argument candidate values: `643`
- argument sequence candidates: `288`
- tool-name blocks augmented: `31`
- tool-name blocks with sequence candidates: `31`
- tool-name target candidates present: `31`

Selector examples:

```text
data/candidate_ranking/public_multicall_live_v5_sequence_planned_toolname_argument_ranking_evidence_v5.summary.json
```

Coverage:

- examples: `131`
- usable examples: `131`
- argument-value examples: `100/100`
- tool-name examples: `31/31`
- target missing from candidates: `0`
- total candidate values: `341`

Delta versus earlier evidence-only attempts:

| artifact | argument sequence coverage | usable selector rows | total candidate values | selector result |
| --- | ---: | ---: | ---: | ---: |
| evidence v1 | `69/100` | `99/100` visible rows | `367` | not sufficient |
| evidence v4 | `100/100` | `131/131` | `516` | `128/131` |
| evidence v5 row-local | `100/100` | `131/131` | `341` | `131/131` |

The v4 selector misses were all row-alignment errors in the financial table
case: `expense_data[3].date`, `expense_data[1].description`, and
`invoice_data[2].due_date`. Row-local markdown table candidates fixed them.

## Selector Gate

Tournament:

```text
runs/candidate_ranking/public_multicall_live_v5_sequence_planned_evidence_v5_ckpt275_pairwise_tournament.summary.json
```

Result:

- overall: `131/131`
- argument values: `100/100`
- tool names: `31/31`
- pair comparisons: `690`
- elapsed: `278.8s`
- max allocated VRAM: `18.46 GiB`
- max reserved VRAM: `27.46 GiB`

For comparison, the broader v4 candidate set had `1480` pair comparisons and
missed `3` rows. Candidate pruning improved both quality and speed.

## Injected Schedule

Command:

```bash
.venv-fastdllm/bin/python scripts/inject_pairwise_tournament_schedule_choices.py \
  --schedule-jsonl runs/tool_sensitive_block_plans/public_multicall_live_v5_sequence_planned_sampler_schedule_with_candidates_evidence_v5.jsonl \
  --selector-jsonl runs/candidate_ranking/public_multicall_live_v5_sequence_planned_evidence_v5_ckpt275_pairwise_tournament.jsonl \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_live_v5_sequence_planned_sampler_schedule_with_evidence_pairwise_choices.jsonl \
  --include-kinds tool_name argument_value
```

Injection result:

- selectors consumed: `131`
- selectors correct: `131`
- restricted schedule items: `267`
- candidate missing items: `0`
- records: `12`

## Generation Gate

Command:

```bash
systemd-run --user --scope --quiet -p MemoryMax=28G -p MemorySwapMax=4G bash -lc 'cd /home/mark/qwen_diffusion && CUDA_VISIBLE_DEVICES=0 .venv-fastdllm/bin/python scripts/eval_fastdllm_toolcall_cases.py --base-model models/qwen3.5-9b-fastdllm-init --adapter runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model --tokenizer-path models/qwen3.5-9b-fastdllm-init --input-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl --out-jsonl runs/tool_sensitive_block_plans/public_multicall_live_v5_evidence_toolargselector_structguard_ckpt275_generation.jsonl --max-new-tokens 560 --conversation-template fast_dllm_v2 --full-context-sampling --sampler-schedule-jsonl runs/tool_sensitive_block_plans/public_multicall_live_v5_sequence_planned_sampler_schedule_with_evidence_pairwise_choices.jsonl --force-schedule-token-kinds json_key,json_structure,tool_tag --force-argument-boundary-target-tokens --force-best-candidate-sequence --force-best-tool-name-sequence --ban-argument-boundary-tokens --ban-argument-newline-tokens --stop-after-schedule-tool-calls --constrained-tool-decoding --constrained-sequence-preserving --constrained-max-calls 3 --no-merge-adapter'
```

Summary:

```text
runs/tool_sensitive_block_plans/public_multicall_live_v5_evidence_toolargselector_structguard_ckpt275_generation.summary.json
```

Result:

- records: `12`
- valid tool JSON: `12/12`
- exact tool sequence: `12/12`
- exact tool-name multiset: `12/12`
- exact arguments: `12/12`
- schema valid: `12/12`
- required args present: `12/12`
- extra calls: `0`
- missing calls: `0`
- stop-boundary trims: `8`
- elapsed: `380.7s`
- generated tokens/sec: `10.80`
- max allocated VRAM: `18.39 GiB`
- max reserved VRAM: `27.90 GiB`

Sampler counters:

- scheduled interval visits: `814`
- scheduled token visits: `863`
- forced structural/tool-tag/key token visits: `844`
- argument candidate sequence force visits: `126` intervals / `459` tokens
- tool-name sequence force visits: `152` intervals / `152` tokens
- semantic candidate model-choice events: `0`, because selector injection
  reduced tool-name and argument-value spans to singleton candidate sequences

## Audit

Command:

```bash
.venv-fastdllm/bin/python scripts/analyze_toolcall_candidate_misses.py \
  --eval-jsonl runs/tool_sensitive_block_plans/public_multicall_live_v5_evidence_toolargselector_structguard_ckpt275_generation.jsonl \
  --cases-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --schedule-jsonl runs/tool_sensitive_block_plans/public_multicall_live_v5_sequence_planned_sampler_schedule_with_evidence_pairwise_choices.jsonl \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_live_v5_evidence_toolargselector_structguard_ckpt275_candidate_miss_audit.jsonl
```

Audit result:

- failed records: `0`
- mismatches: `0`
- missing calls: `0`
- extra calls: `0`
- invalid tool blocks: `0`

## Interpretation

This closes the main caveat from the first live planner sampler replay:
planner-target inclusion is no longer needed on the public multi-call slice.

The current protected stack is:

- live planner proposes tool-call text and sensitive spans;
- evidence extractor builds request/schema-derived candidates;
- pairwise selector chooses tool names and argument values;
- diffusion sampler commits singleton semantic spans;
- deterministic guards still handle JSON structure, tool tags, and stop
  boundaries.

Remaining caveats:

- The planner is still a deterministic sidecar over live outputs, not raw model
  generation.
- The selector is still runtime/procedural, not learned into the diffusion
  model.
- Structural and stop guards remain deterministic.
- The gate is still a small 12-case public multi-call slice.

Next target: turn this evidence-only selector path into a reusable live route
manifest and evaluate it on teacher-heldout or freshly generated Qwen3.6
multi-call cases before training a selector/boundary adapter.

## Replay Route

The route is now captured by a reproducible runner:

```text
scripts/run_qwen35_live_evidence_selector_route.py
```

Plan and shell replay:

```text
runs/tool_sensitive_block_plans/live_v5_evidence_selector_route/route_plan.json
runs/tool_sensitive_block_plans/live_v5_evidence_selector_route/route_plan.sh
```

Verifier:

```bash
.venv-fastdllm/bin/python scripts/run_qwen35_live_evidence_selector_route.py --verify-existing
```

Verification output:

```text
runs/tool_sensitive_block_plans/live_v5_evidence_selector_route/route_plan_verification.json
```

Result:

- checks: `16`
- missing artifacts/summaries: `0`
- failed checks: `0`
- status: pass

The route plan covers all steps needed to replay the current public multi-call
evidence-selector path:

1. tokenize live planner spans;
2. emit sampler schedule;
3. build evidence-only candidates;
4. build selector examples;
5. run checkpoint-275 pairwise selector tournament;
6. inject selector choices;
7. run scheduled protected generation;
8. audit generated calls against gold.

GPU steps in `route_plan.sh` are wrapped in the same user `systemd-run` memory
scope used by the manual experiments.

## Follow-Up Route

The reusable non-public analogue replay is now captured separately:

```text
qwen35_synthetic_evidence_selector_route_result.md
```

It uses `scripts/run_qwen35_evidence_selector_route.py` on
`data/toolcall_eval/synthetic_multicall_failure_analogues.jsonl` and verifies
the same evidence-selector/scheduled-generation chain at `8/8` exact sequence
and `8/8` exact arguments, with the generic route verifier passing `20/20`
checks.
