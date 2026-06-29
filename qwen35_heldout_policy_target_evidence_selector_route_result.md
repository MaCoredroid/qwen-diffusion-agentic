# Qwen3.5 Heldout Policy-Target Evidence-Selector Route Result

Date: 2026-06-28

## Purpose

Replay the protected evidence-selector route on the clean 12-row heldout
planner-policy target set, instead of the original 13-row seed slice. This
tests the next live-route scaffold after rejecting the ambiguous construction
expense row.

This is still a protected route, not raw model promotion. The route uses
policy-target assistant spans, structural forcing, constrained tool decoding,
candidate/tool-name sequence forcing, and the derived-rule selector sidecar.

## Inputs

Cases and span source:

```text
runs/planner_decomposition/heldout_seed_multicall_policy_targets.jsonl
```

The route uses the `policy_planner_assistant` field for block planning.

Route root:

```text
runs/tool_sensitive_block_plans/heldout_seed_policy_evidence_selector/
```

Adapter:

```text
runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model
```

## Route Artifacts

```text
runs/tool_sensitive_block_plans/heldout_seed_policy_evidence_selector/blocks_tokenized_with_ids.jsonl
runs/tool_sensitive_block_plans/heldout_seed_policy_evidence_selector/sampler_schedule_with_ids.jsonl
runs/tool_sensitive_block_plans/heldout_seed_policy_evidence_selector/sampler_schedule_with_candidates_evidence.jsonl
data/candidate_ranking/heldout_seed_policy_evidence_selector_toolname_argument_ranking_evidence.jsonl
runs/candidate_ranking/heldout_seed_policy_evidence_selector_ckpt275_pairwise_tournament.jsonl
runs/candidate_ranking/heldout_seed_policy_evidence_selector_derived_sidecar.jsonl
runs/tool_sensitive_block_plans/heldout_seed_policy_evidence_selector/sampler_schedule_with_derived_pairwise_choices.jsonl
runs/tool_sensitive_block_plans/heldout_seed_policy_evidence_selector/derived_toolargselector_structguard_ckpt275_generation.jsonl
runs/tool_sensitive_block_plans/heldout_seed_policy_evidence_selector/derived_toolargselector_structguard_ckpt275_candidate_miss_audit.jsonl
```

## Coverage

Block planning:

- records: `12`
- tool calls: `29`
- tool-name token blocks: `29`
- argument-value token blocks: `123`

Candidate augmentation:

- argument blocks with sequence candidates: `119/119`
- tool-name blocks with sequence candidates: `29/29`
- tool-name blocks with target candidate: `29/29`

Selector examples:

- examples: `152`
- argument-value examples: `123`
- tool-name examples: `29`
- target missing from candidates: `0`

## Selector Gate

Raw checkpoint-275 pairwise tournament:

- overall: `150/152`
- argument values: `121/123`
- tool names: `29/29`
- elapsed: `268.9s`
- max allocated VRAM: `19.10 GiB`
- max reserved VRAM: `27.74 GiB`

The two misses are the same derived-rule cases seen in the gold-span heldout
route:

- `portfolio[2].weight`: choose final equal-weight residual `0.334` instead of
  another `0.333`
- `refund_policy`: choose `full` from the cancellation lead-time policy

Derived-rule sidecar:

- model selector before rules: `150/152`
- final selector after rules: `152/152`
- argument values after rules: `123/123`
- tool names after rules: `29/29`
- rules applied: `2`
  - `equal_weight_residual`: `1`
  - `refund_policy_threshold`: `1`

Injected schedule:

- selectors consumed: `152`
- selectors correct: `152`
- restricted schedule items: `302`
- candidate missing items: `0`
- records covered: `12`

## Protected Generation

Generation output:

```text
runs/tool_sensitive_block_plans/heldout_seed_policy_evidence_selector/derived_toolargselector_structguard_ckpt275_generation.summary.json
```

Result:

- valid tool JSON: `12/12`
- exact tool-name set: `12/12`
- exact tool-name multiset: `12/12`
- exact tool sequence: `12/12`
- exact arguments: `12/12`
- schema valid: `12/12`
- required args present: `12/12`
- extra calls: `0`
- missing calls: `0`
- stop-boundary trims: `9`
- elapsed: `858.6s`
- generated tokens/sec: `7.73`
- max allocated VRAM: `18.80 GiB`
- max reserved VRAM: `28.17 GiB`

Independent audit:

```text
runs/tool_sensitive_block_plans/heldout_seed_policy_evidence_selector/derived_toolargselector_structguard_ckpt275_candidate_miss_audit.summary.json
```

Result:

- failed records: `0`
- mismatches: `0`
- missing calls: `0`
- extra calls: `0`
- invalid tool blocks: `0`

## Interpretation

The policy-target protected route now passes end to end on the accepted
heldout planner targets. Compared with the gold-span route, this is a cleaner
planner-side target because the contradictory construction-expense row is
excluded and the remaining rows have explicit teacher/gold/decomposition
policy.

What this proves:

- policy-target spans can drive tool-sensitive block planning;
- evidence candidates cover the semantic decisions needed by the route;
- tool-name selection is saturated on this slice;
- the known derived-rule sidecar is sufficient for the remaining value choices;
- scheduled protected generation can replay the policy targets with exact
  tool sequence and arguments.

What this does not prove:

- raw diffusion planning is solved;
- the model can choose the decomposition policy without the sidecar;
- the protected route should be counted as model-side promotion.

Next useful step: convert this route into training/eval pressure. The cleanest
next gate is to distill the planner-policy and selector decisions into a
model-side adapter or dynamic boundary/value adapter, then require movement in
raw or constrained-decoder metrics while keeping this protected replay as an
oracle/debugging ceiling.

Follow-up diagnostic:

```text
qwen35_heldout_policy_derived_pairwise_diagnostic_result.md
```

A focused 10-step pairwise SFT run on the two sidecar-derived misses trained
cleanly but changed `0/152` selector predictions versus checkpoint-275. The
two misses remained `portfolio[2].weight` and `refund_policy`. Treat this as a
negative result for more of the same short pairwise SFT; derived value/policy
choices need a different objective, a value adapter, or generation-time scorer.

## Follow-up: Close Guard and Structural-Key Ceiling

Result note:

```text
qwen35_heldout_policy_close_guard_scorecard_result.md
```

The public close-guard stack was moved to this heldout policy-target route.
With only the named guards:

```text
--guard-tool-call-mode
--guard-tool-json-prefix
--guard-tool-name-candidates
--guard-tool-value-candidates
```

the run reaches raw valid JSON `11/12`, exact sequence `11/12`, and exact
arguments `11/12`. The single miss is `heldout_seed_multicall_0004`, a nested
`create_campaign` case where JSON keys/structure drift inside a long
`campaign_details` array. The JSON-prefix guard rejects `83` unsafe commits but
still has `83` unsafe fallbacks because no safe top-k replacement is found.

Adding only structural skeleton forcing:

```text
--force-schedule-token-kinds json_key,json_structure
```

while keeping named mode/name/value guards reaches raw valid JSON `12/12`,
exact sequence `12/12`, exact arguments `12/12`, schema valid `12/12`, and
required args present `12/12`. The completability diagnostic reports `29/29`
complete raw JSON segments and zero invalid segments.

Interpretation: the heldout protected ceiling is now exact, but the missing
model-side target is JSON skeleton/key/structure stability on long nested
objects, followed by evidence-grounded value infill without full oracle
schedule forcing.
