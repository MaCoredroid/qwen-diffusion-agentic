# Qwen3.5 Path-Aware Phrase Selector Gate Result

Date: 2026-06-28

## Purpose

Turn the earlier public-slice path-aware selector diagnostic into a cleaner
recipe component:

- propagate `json_path` / `argument_path` through train and heldout schedules;
- expand candidate extraction for user-mentioned free-form string values;
- build a promotion-eligible train-only pairwise selector curriculum;
- gate the current checkpoint-275 selector on heldout public multi-call cases;
- inject only argument-value selector choices into the protected sampler.

This is still protected-sampler evidence, not raw model promotion.

## Code Changes

- `scripts/plan_tool_sensitive_blocks.py` now records JSON paths for scalar
  keys/values, including array positions such as `invoice_data[1].client_id`.
- `scripts/emit_tool_sensitive_sampler_schedule.py` carries those paths into
  schedule rows.
- `scripts/build_candidate_ranking_examples.py` and
  `scripts/build_candidate_pairwise_curriculum.py` include paths in selector
  prompts and rows.
- `scripts/diagnose_schedule_value_candidates.py` now adds conservative
  single-quoted string candidates and model/product phrases with digits. This
  fixes user-evidence values such as `Main Control Group`,
  `YRD256 Yale Assure Lock SL`, and `ChemSimulationProject`.
- `scripts/inject_pairwise_tournament_schedule_choices.py` now supports
  `--include-kinds` and can restrict multi-chunk argument spans.

## Train-Only Artifacts

Path-aware phrase train ranking examples:

```text
data/candidate_ranking/public_train_multicall_toolname_argument_ranking_pathaware_phrase.jsonl
data/candidate_ranking/public_train_multicall_toolname_argument_ranking_pathaware_phrase.summary.json
```

Coverage:

- train records: `56`
- examples: `367`
- usable examples: `328`
- usable argument-value examples: `173`
- usable tool-name examples: `155`
- target missing from candidates: `39`

Promotion-eligible pairwise curriculum:

```text
data/qwen35_9b_public_train_pairwise_pathaware_phrase_curriculum
```

Manifest:

- accepted rows: `376`
- rejected labels: `0`
- contains eval slice: `false`
- diagnostic only: `false`
- promotion allowed by provenance: `true`
- block size: `1536`
- no zero-label or partial-label truncation

## Heldout Selector Gate

Heldout examples:

```text
data/candidate_ranking/public_multicall_toolname_argument_ranking_pathaware_phrase_12.jsonl
```

Coverage:

- records: `12`
- examples: `99`
- usable examples: `99`
- argument-value examples: `68`
- tool-name examples: `31`
- target missing from candidates: `0`

Checkpoint-275 pairwise tournament:

```text
runs/candidate_ranking/public_multicall_pathaware_phrase12_ckpt275_pairwise_tournament.summary.json
```

Result:

- overall: `98/99`
- argument values: `68/68`
- tool names: `30/31`
- pair comparisons: `616`
- max reserved VRAM: `24.37 GiB`

The single selector miss is a tool-name row. The sampler injection therefore
filters to `argument_value` selectors only.

## Protected Sampler Gate

Injected schedule:

```text
runs/tool_sensitive_block_plans/public_multicall_gold_sampler_schedule_with_pairwise_path_arg_choices_pathaware_phrase_12.jsonl
```

Injection summary:

- argument selectors: `68`
- correct argument selectors: `68`
- restricted schedule chunks: `74`
- candidate missing items: `0`
- filtered tool-name selectors: `31`

Generation:

```text
runs/tool_sensitive_block_plans/public_multicall_pathaware_phrase12_argselector_structguard_ckpt275_generation.summary.json
```

Result:

- exact tool sequence: `12/12`
- exact arguments: `12/12`
- valid JSON: `12/12`
- schema valid: `12/12`
- elapsed: `409.2s`
- max reserved VRAM: `28.19 GiB`

Final audit:

```text
runs/tool_sensitive_block_plans/public_multicall_pathaware_phrase12_argselector_structguard_ckpt275_candidate_miss_audit.summary.json
```

Audit result: `0` failed records, `0` mismatches, `0` missing/extra calls, `0`
invalid tool blocks.

## Interpretation

The selector stack now has a cleaner story:

- JSON path metadata is required for row/table arguments.
- Whole-candidate sequence restriction is the stable sampler mechanism.
- Free-form copied phrase values need evidence extraction, not just a selector.
- Tool-name and argument-value selectors should remain separate; argument
  selection is perfect on this heldout gate, while tool-name selection still has
  a known miss.

Next promotion step: train or distill on the train-only phrase-aware pairwise
curriculum, then require heldout public/teacher gates to improve without
depending on public-derived labels.

## Train-Only SFT Follow-Up

Follow-up result:

```text
qwen35_public_train_pairwise_phrase_sft_result.md
```

A conservative 10-step SFT from checkpoint-275 on the promotion-eligible
train-only pairwise phrase curriculum completed without OOM and saved
checkpoint-5/checkpoint-10. Heldout selector performance was unchanged from
checkpoint-275:

- checkpoint-275: `98/99` overall, `68/68` argument values, `30/31` tool names
- checkpoint-5: `98/99` overall, `68/68` argument values, `30/31` tool names
- checkpoint-10: `98/99` overall, `68/68` argument values, `30/31` tool names

Row-level predictions were identical on all `99` heldout selector rows. Treat
this as a no-regression training smoke, not model promotion.

## Same-Call Argument-Sketch Follow-Up

Follow-up result:

```text
qwen35_toolname_argsketch_selector_result.md
```

The heldout tool-name miss was fixed by changing selector context, not by more
SFT. Tool-name selector prompts now include a same-call argument sketch derived
from schedule rows with the same `tool_call_index`.

Heldout tournament from checkpoint-275:

```text
runs/candidate_ranking/public_multicall_pathaware_phrase_argsketch12_ckpt275_pairwise_tournament.summary.json
```

Result:

- overall: `99/99`
- argument values: `68/68`
- tool names: `31/31`
- max reserved VRAM: `27.46 GiB`

Interpretation: tool-name choice needs local call evidence. The dynamic
block-boundary recipe should either expose planned argument keys/values before
committing the tool-name span or delay the tool-name decision until that
evidence is available. This is component-gate progress, not raw model
promotion.
