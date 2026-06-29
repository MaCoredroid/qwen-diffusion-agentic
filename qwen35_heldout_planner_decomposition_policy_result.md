# Heldout Planner Decomposition Policy Result

Date: 2026-06-28

## Purpose

Turn the heldout live-planner ambiguity into an auditable policy artifact. The
gold-span protected sampler now passes the heldout slice, but live planning is
still weak:

- deterministic request planner: `3/13` exact sequence, `0/13` exact arguments
- Qwen3.6 required/native teacher: `9/13` exact sequence, `6/13` exact
  arguments

This note records which heldout rows are clean planner targets, which need
value/decomposition policy, and which should be rejected until adjudicated.

## Analyzer

Added:

```text
scripts/analyze_planner_decomposition_policy.py
```

Inputs:

```text
data/toolcall_eval/heldout_seed_multicall_2to3_clean.jsonl
runs/qwen36_teacher_heldout_multicall_2to3_clean_toolchoice_required.jsonl
runs/qwen36_teacher_heldout_multicall_2to3_clean_toolchoice_auto.jsonl
runs/heldout_seed_multicall_2to3_clean/sequence_planner_from_empty.jsonl
```

Output:

```text
runs/planner_decomposition/heldout_seed_multicall_policy_analysis.jsonl
runs/planner_decomposition/heldout_seed_multicall_policy_analysis.summary.json
```

Summary:

- records: `13`
- recommended target counts:
  - `teacher_required_or_gold`: `6`
  - `teacher_required_sequence_plus_value_sidecars`: `3`
  - `gold_sequence_decomposition_target`: `3`
  - `adjudicate_full_request_vs_seed_gold`: `1`
- key tags:
  - `teacher_required_exact`: `6`
  - `teacher_required_sequence_exact_args_need_normalization`: `3`
  - `teacher_required_undercalls_prompt_supported_gold`: `3`
  - `seed_gold_subset_ambiguous_teacher_overcalls_prompt_supported`: `1`
  - `split_call_policy_needed`: `1`
  - `heuristic_sequence_mismatch`: `10`
  - `heuristic_sequence_exact_args_wrong`: `3`

Interpretation:

- `6` rows are clean teacher/gold planner targets.
- `3` rows have the right teacher sequence but need value-normalization or
  request-grounding sidecars.
- `3` rows need gold/decomposition policy because Qwen3.6 undercalls a
  prompt-supported gold action.
- `1` row is genuinely ambiguous: the construction-expense prompt asks to
  record, categorize, and report; seed gold contains only the record-expense
  calls, while Qwen3.6 follows the fuller request.

## Policy Targets

Added:

```text
scripts/materialize_planner_policy_targets.py
```

Output:

```text
runs/planner_decomposition/heldout_seed_multicall_policy_targets.jsonl
runs/planner_decomposition/heldout_seed_multicall_policy_targets.rejected.jsonl
runs/planner_decomposition/heldout_seed_multicall_policy_targets.summary.json
```

Materialization policy:

- accept exact teacher-required rows as `teacher_required_exact`;
- use gold targets when teacher sequence is right but values need sidecars;
- use gold targets for prompt-supported missing/split-call decomposition rows;
- reject full-request-vs-seed-gold ambiguity pending human or explicit policy
  adjudication.

Summary:

- accepted records: `12`
- rejected records: `1`
- rejected id: `heldout_seed_multicall_0001`
- target source counts:
  - `teacher_required_exact`: `6`
  - `gold_values_for_teacher_sequence`: `3`
  - `gold_decomposition_policy`: `3`

Verification:

- accepted policy targets score `12/12` valid tool JSON
- `12/12` exact tool sequence
- `12/12` exact arguments
- `12/12` schema valid
- `12/12` required args present

## Implication

This is the planner-side counterpart to the heldout gold-span protected replay:
it gives us a clean 12-row planner target set and explicitly keeps the
ambiguous row out of the target. The next live-route step should use these
policy targets to test planner-span scheduling and selector replay, while
separately deciding whether the ambiguous construction-expense row should
follow the full request or preserve the seed-gold subset.

Do not treat this as raw model promotion. It is an adjudication scaffold for
building a behavior-preserving live planner target without silently training or
evaluating against contradictory labels.
