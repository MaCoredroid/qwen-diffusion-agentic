# Qwen3.5 Arg-Sketch Tool+Argument Selector Sampler Result

Date: 2026-06-28

## Purpose

Use the `99/99` same-call argument-sketch selector gate as a sampler component:
inject selector choices for both tool names and argument values, then rerun the
public multi-call protected generation gate.

This tests whether the older argument-only selector path can be replaced by a
cleaner selector-owned path for all sensitive semantic spans. It is still a
protected sampler result over a gold-tokenized schedule, not raw/live diffusion
model promotion.

## Inputs

Selector tournament:

```text
runs/candidate_ranking/public_multicall_pathaware_phrase_argsketch12_ckpt275_pairwise_tournament.jsonl
```

Selector gate:

- overall: `99/99`
- argument values: `68/68`
- tool names: `31/31`

Base candidate schedule:

```text
runs/tool_sensitive_block_plans/public_multicall_gold_sampler_schedule_with_toolname_candidates_pathaware_phrase_12.jsonl
```

This schedule has path-aware phrase candidates and available-tool candidates,
but is not restricted to the pairwise tournament winners until injection.

## Injected Schedule

Command:

```bash
.venv-fastdllm/bin/python scripts/inject_pairwise_tournament_schedule_choices.py \
  --schedule-jsonl runs/tool_sensitive_block_plans/public_multicall_gold_sampler_schedule_with_toolname_candidates_pathaware_phrase_12.jsonl \
  --selector-jsonl runs/candidate_ranking/public_multicall_pathaware_phrase_argsketch12_ckpt275_pairwise_tournament.jsonl \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_gold_sampler_schedule_with_pairwise_argsketch_choices_12.jsonl \
  --include-kinds tool_name argument_value
```

Summary:

```text
runs/tool_sensitive_block_plans/public_multicall_gold_sampler_schedule_with_pairwise_argsketch_choices_12.summary.json
```

Injection result:

- selector rows consumed: `99`
- selector rows correct: `99`
- restricted schedule items: `226`
- candidate missing items: `0`
- records: `12`

Breakdown:

- tool-name selectors: `31`, restricting `152` token-level schedule items
- argument-value selectors: `68`, restricting `74` token-level schedule items

After injection, all restricted tool-name and argument-value schedule chunks
have singleton `candidate_sequence_values` from the pairwise selector.

## Generation Gate

Command:

```bash
systemd-run --user --scope --quiet -p MemoryMax=28G -p MemorySwapMax=4G bash -lc 'cd /home/mark/qwen_diffusion && CUDA_VISIBLE_DEVICES=0 .venv-fastdllm/bin/python scripts/eval_fastdllm_toolcall_cases.py --base-model models/qwen3.5-9b-fastdllm-init --adapter runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model --tokenizer-path models/qwen3.5-9b-fastdllm-init --input-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl --out-jsonl runs/tool_sensitive_block_plans/public_multicall_argsketch_toolargselector_structguard_ckpt275_generation.jsonl --max-new-tokens 560 --conversation-template fast_dllm_v2 --full-context-sampling --sampler-schedule-jsonl runs/tool_sensitive_block_plans/public_multicall_gold_sampler_schedule_with_pairwise_argsketch_choices_12.jsonl --force-schedule-token-kinds json_key,json_structure,tool_tag --force-argument-boundary-target-tokens --force-best-candidate-sequence --force-best-tool-name-sequence --ban-argument-boundary-tokens --ban-argument-newline-tokens --stop-after-schedule-tool-calls --constrained-tool-decoding --constrained-sequence-preserving --constrained-max-calls 3 --no-merge-adapter'
```

Summary:

```text
runs/tool_sensitive_block_plans/public_multicall_argsketch_toolargselector_structguard_ckpt275_generation.summary.json
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
- stop-boundary trims: `6`
- elapsed: `406.3s`
- generated tokens/sec: `10.14`
- max allocated VRAM: `18.45 GiB`
- max reserved VRAM: `28.38 GiB`

Sampler counters:

- scheduled interval visits: `1056`
- scheduled token visits: `1486`
- forced structural/tool-tag/key token visits: `956`
- argument candidate sequence force visits: `76` intervals / `307` tokens
- tool-name sequence force visits: `152` intervals / `152` tokens
- tool-name sequence model-choice count: `0`, because selector injection
  reduced each compatible tool-name span to a singleton candidate sequence
- argument sequence model-choice count: `0`, for the same reason

## Audit

Command:

```bash
.venv-fastdllm/bin/python scripts/analyze_toolcall_candidate_misses.py \
  --eval-jsonl runs/tool_sensitive_block_plans/public_multicall_argsketch_toolargselector_structguard_ckpt275_generation.jsonl \
  --cases-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --schedule-jsonl runs/tool_sensitive_block_plans/public_multicall_gold_sampler_schedule_with_pairwise_argsketch_choices_12.jsonl \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_argsketch_toolargselector_structguard_ckpt275_candidate_miss_audit.jsonl
```

Summary:

```text
runs/tool_sensitive_block_plans/public_multicall_argsketch_toolargselector_structguard_ckpt275_candidate_miss_audit.summary.json
```

Audit result:

- failed records: `0`
- mismatches: `0`
- missing calls: `0`
- extra calls: `0`
- invalid tool blocks: `0`

## Interpretation

The same-call argument-sketch selector can now own both semantic sensitive-span
classes on this public multi-call heldout slice:

- tool-name spans are restricted by selector-ranked available-tool candidates;
- argument-value spans are restricted by selector-ranked evidence candidates;
- structural JSON and tool tags are still hard guarded by the sampler;
- stop boundaries are still protected by `--stop-after-schedule-tool-calls`.

Compared with the previous argument-only phrase selector sampler gate, this
removes the known `30/31` tool-name selector gap and uses selector choices for
all `99` tool-name/argument selector rows. The generation aggregate remains
`12/12` exact sequence and arguments, with a clean miss audit.

Limits:

- The schedule is still built from gold assistant text, so this does not prove
  a live planner can choose all block boundaries.
- Structural tokens, tool tags, and stop boundaries are still protected by
  deterministic sampler guards.
- This is not raw model promotion. It is a stronger runtime-component gate and
  a clearer target for a learned selector/boundary adapter.

Next step: build the same selector interface for non-gold/live schedules, where
the planner proposes tool-call slots and argument sketches, then the selector
chooses tool names and values before the diffusion sampler commits sensitive
blocks.

## Live Planner Follow-Up

Follow-up result:

```text
qwen35_live_planner_argsketch_sampler_result.md
```

The selector-owned sampler path now runs over the live public multi-call
`sequence_planner_assistant` schedule instead of gold assistant text.

Key result:

- live planner schedule: `31` tool-name spans and `100` argument-value spans
- target-included selector gate: `131/131`
- injected schedule: `131` selectors, `267` schedule items restricted,
  `0` candidate misses
- generation: `12/12` exact tool sequence, `12/12` exact arguments,
  `12/12` valid JSON
- final audit: `0` failed records and `0` mismatches

Remaining gap: evidence-only extraction has sequence candidates for `69/100`
planned argument-value spans, so full live replay still needs planner-target
inclusion. The next target is closing that evidence-candidate coverage gap.
