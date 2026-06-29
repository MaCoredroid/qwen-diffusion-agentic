# Qwen3.5 Heldout Seed Evidence-Selector Preflight

Date: 2026-06-28

## Purpose

Move the evidence-selector route from public/synthetic curated slices onto a
clean heldout multi-call slice from the original tool-call seed corpus.

This is a preflight, not a route pass. The goal was to test whether the current
planner, evidence extractor, and checkpoint-275 pairwise selector still hold up
when the prompts are broader than the public smoke and synthetic analogue rows.

## Heldout Slice

Builder:

```text
scripts/build_heldout_seed_multicall_cases.py
```

Output:

```text
data/toolcall_eval/heldout_seed_multicall_2to3_clean.jsonl
data/toolcall_eval/heldout_seed_multicall_2to3_clean.summary.json
```

Summary:

- records: `13`
- gold tool calls: `2` to `3` per record
- call-count histogram: `{2: 7, 3: 6}`
- source: `13` Hermes rows
- exact train/eval overlaps removed: `85` filtered-train records and `20`
  eval records considered
- no exact assistant or user overlap with the filtered train/public/synthetic
  eval sources

## Planner Preflight

Request-derived planner from empty assistant text:

```text
runs/heldout_seed_multicall_2to3_clean/sequence_planner_from_empty.summary.json
```

Result:

- valid planned tool JSON: `13/13`
- exact tool-name set: `5/13`
- exact tool-name multiset: `3/13`
- exact tool sequence: `3/13`
- exact arguments: `0/13`
- all schema-valid / required args present: `5/13`

Interpretation: the public/synthetic request-derived planner does not
generalize to this broader heldout slice. Before this can become an end-to-end
route, the planner needs stronger decomposition or teacher-generated planner
targets.

## Gold-Span Evidence Preflight

To isolate the semantic selector from planner errors, the next pass used
`gold_assistant` as the span source.

Artifacts:

```text
runs/tool_sensitive_block_plans/heldout_seed_multicall_gold_evidence_selector/blocks_tokenized_with_ids.jsonl
runs/tool_sensitive_block_plans/heldout_seed_multicall_gold_evidence_selector/sampler_schedule_with_ids.jsonl
runs/tool_sensitive_block_plans/heldout_seed_multicall_gold_evidence_selector/sampler_schedule_with_candidates_evidence.jsonl
runs/tool_sensitive_block_plans/heldout_seed_multicall_gold_evidence_selector/value_candidate_diagnostic.jsonl
data/candidate_ranking/heldout_seed_multicall_gold_evidence_selector_toolname_argument_ranking_evidence.jsonl
```

Evidence candidate coverage:

- argument blocks augmented: `140`
- argument blocks with sequence candidates: `140`
- tool-name blocks augmented: `32`
- tool-name target candidates present: `32`
- diagnostic target coverage: `144/144` argument values
- selector examples: `172`
- target missing from candidates: `0`

Important code fix:

- `scripts/plan_tool_sensitive_blocks.py` now treats only top-level
  `json_path == "name"` as a tool-name value. Nested argument fields such as
  `items[0].name` and `scenarios[0].name` remain `argument_value`.

## Selector Gate

Tournament:

```text
runs/candidate_ranking/heldout_seed_multicall_gold_evidence_selector_ckpt275_pairwise_tournament.summary.json
```

Result:

- overall selector accuracy: `157/172`
- argument values: `125/140`
- tool names: `32/32`
- pair comparisons: `585`
- elapsed: `212.5s`
- max allocated VRAM: `18.30 GiB`
- max reserved VRAM: `23.76 GiB`

All `15` misses are argument-value choices. They cluster in four records:

- construction expenses: row-local amount/date/type/vendor confusion
- IoT setup: `device_002`/`smart_light` confused with neighboring device rows
- ad campaign schedule: second demographic and nested ad schedule row choices
- ticket cancellation: `full` refund confused with nearby policy options

The common failure mode is not syntax or function choice. It is list/row-local
semantic alignment inside repeated fields.

## Peer-Context Selector Follow-Up

Code changes:

- `scripts/build_candidate_ranking_examples.py`
  - argument-value groups now include `json_path`, so repeated same-key spans
    are not collapsed across array/object locations.
  - argument-value examples now include non-leaking `local_peer_arguments` and
    `same_call_peer_arguments`, excluding the current value span.
- `scripts/build_candidate_pairwise_curriculum.py`
  - pairwise prompts now include local peer argument sketches for argument
    values.
  - prompts also surface short request snippets anchored on local peer values
    and path terms, so the selector sees row/policy evidence near the options.
- `scripts/eval_fastdllm_candidate_pairwise_tournament.py`
  - tournament rows now preserve peer-context fields for miss audits.

Rebuilt examples:

```text
data/candidate_ranking/heldout_seed_multicall_gold_evidence_selector_toolname_argument_ranking_evidence_peerctx_rules.jsonl
```

Coverage:

- examples: `176`
- argument-value examples: `144`
- tool-name examples: `32`
- target missing from candidates: `0`

Best selector tournament:

```text
runs/candidate_ranking/heldout_seed_multicall_gold_evidence_selector_peerctx_rules_snippets_ckpt275_pairwise_tournament.summary.json
```

Result:

- overall: `174/176`
- argument values: `142/144`
- tool names: `32/32`
- accuracy: `0.9886`
- pair comparisons: `586`
- elapsed: `327.4s`
- max allocated VRAM: `19.10 GiB`
- max reserved VRAM: `28.16 GiB`

Delta:

- baseline selector: `157/172`, with `125/140` argument values
- peer-context selector: `173/176`, with `141/144` argument values
- peer-context plus request snippets: `174/176`, with `142/144` argument values

The row/list-local failure family is largely fixed. The remaining misses are:

- `portfolio[2].weight`: target `0.334`, predicted `0.333`; this is a rounded
  residual after two prior `0.333` weights.
- `refund_policy`: target `full`, predicted `partial`; this requires applying
  a threshold policy to a cancellation `20` days before the event.

These two misses are derived-rule failures, not evidence-coverage or row-local
alignment failures.

## Derived-Rule Sidecar and Protected Replay

Added:

```text
scripts/apply_derived_rule_selector_sidecar.py
```

The sidecar applies auditable non-gold rules after the model tournament:

- `equal_weight_residual`: for final array weights, choose the candidate that
  makes previous known weights sum closest to `1.0`.
- `percentage_range_midpoint`: for percentage ranges near a local peer label,
  choose the midpoint candidate.
- `refund_policy_threshold`: for refund enum choices, parse the cancellation
  lead time and policy thresholds from the request.

Sidecar output:

```text
runs/candidate_ranking/heldout_seed_multicall_gold_evidence_selector_peerctx_rules_snippets_derived_sidecar.jsonl
```

Result:

- model selector before rules: `174/176`
- final selector after rules: `176/176`
- argument values after rules: `144/144`
- tool names after rules: `32/32`
- rules applied: `2`
  - `equal_weight_residual`: fixes `portfolio[2].weight` from `0.333` to
    `0.334`
  - `refund_policy_threshold`: fixes `refund_policy` from `partial` to `full`

Injected schedule:

```text
runs/tool_sensitive_block_plans/heldout_seed_multicall_gold_evidence_selector/sampler_schedule_with_derived_pairwise_choices.jsonl
```

Injection summary:

- selectors consumed: `176`
- selectors correct: `176`
- restricted schedule items: `341`
- candidate missing items: `0`

Protected generation replay:

```text
runs/tool_sensitive_block_plans/heldout_seed_multicall_gold_evidence_selector/derived_toolargselector_structguard_ckpt275_generation.summary.json
```

Result:

- valid tool JSON: `13/13`
- exact tool-name set: `13/13`
- exact tool-name multiset: `13/13`
- exact tool sequence: `13/13`
- exact arguments: `13/13`
- schema valid: `13/13`
- required args present: `13/13`
- extra calls: `0`
- missing calls: `0`
- stop-boundary trims: `10`
- elapsed: `1005.6s`
- generated tokens/sec: `7.50`
- max allocated VRAM: `18.80 GiB`
- max reserved VRAM: `28.40 GiB`

Final audit:

```text
runs/tool_sensitive_block_plans/heldout_seed_multicall_gold_evidence_selector/derived_toolargselector_structguard_ckpt275_candidate_miss_audit.summary.json
```

Result:

- failed records: `0`
- mismatches: `0`
- missing calls: `0`
- extra calls: `0`
- invalid tool blocks: `0`

Interpretation: the gold-span protected route now passes this clean heldout
multi-call seed slice. This is stronger than the selector preflight because it
checks the sampler path end to end once spans and selectors are provided. It is
still not raw model promotion and not a live-planner pass: the route uses
`gold_assistant` spans and deterministic structure/stop/derived-rule sidecars.

## Interpretation

This result changes the next target.

What is working:

- tool-sensitive block tagging handles the broader heldout gold text
- evidence extraction can cover every gold semantic span on this slice
- tool-name selection is saturated at `32/32`

What is not yet working:

- request-derived planning from empty text is too weak on diverse heldout rows
- the pairwise selector still over-prefers nearby values in repeated lists,
  schedules, and row-like prose

Next technical target:

1. Add a derived-rule sidecar/evaluator for scalar residuals, numeric range
   reductions, and policy-threshold enums, or train a selector target that
   explicitly covers those rules.
2. Use Qwen3.6 teacher or a stronger planner pass to build heldout planner
   targets before another end-to-end generation replay.
3. Keep local peer/path/request-snippet context in future argument-value
   selector prompts; it is the change that removed the broad row/list-local
   miss family.

Do not count this as raw model promotion. Count it as a clean heldout
gold-span protected replay pass. The next missing piece is live planning:
request-derived planning from empty text remains `3/13` exact sequence and
`0/13` exact arguments on this slice, so teacher/decomposition planner targets
are required before this becomes an end-to-end behavior-preserving route.
