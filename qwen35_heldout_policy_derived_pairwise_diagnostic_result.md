# Qwen3.5 Heldout Policy Derived-Pairwise Diagnostic Result

Date: 2026-06-28

## Purpose

Test whether the two derived-rule selector decisions from the heldout
policy-target route can be moved from deterministic sidecar logic into a
model-side pairwise selector adapter.

This is intentionally diagnostic. The source rows are heldout/eval rows, so
any checkpoint trained here is non-promotable. The question is narrower:
can short focused SFT make the model choose the sidecar-correct values at all?

## Builder Changes

Updated:

```text
scripts/build_synthetic_multicall_planner_distill_curriculum.py
scripts/build_candidate_pairwise_curriculum.py
```

Planner builder additions:

- `--planner-text-field` to target `policy_planner_assistant`
- `--contains-eval-slice`
- `--diagnostic-only`
- manifest now sets `promotion_allowed=false` for diagnostic/eval-slice data

Pairwise builder additions:

- `--only-ids`
- `--only-json-paths`
- `--only-json-keys`

These filters make it possible to build a focused curriculum for exactly the
model-side misses that the sidecar currently fixes.

## Diagnostic Corpora

Policy planner distillation corpus:

```text
data/qwen35_9b_heldout_policy_planner_distill_diagnostic_curriculum/
```

Manifest highlights:

- raw policy cases: `12`
- exact policy targets: `12/12`
- raw candidates: `24` from full + compact schemas
- accepted rows: `15`
- label rejected: `9`
- block size: `1024`
- zero/partial labels among accepted rows: `0/0`
- contains eval slice: `true`
- diagnostic only: `true`
- promotion allowed: `false`

Full heldout policy pairwise selector corpus:

```text
data/qwen35_9b_heldout_policy_pairwise_selector_diagnostic_curriculum/
```

Manifest highlights:

- accepted rows: `420`
- source examples: `152`
- include kinds: `tool_name`, `argument_value`
- block size: `1536`
- zero/partial labels: `0/0`
- promotion allowed: `false`

Focused derived-rule pairwise corpus:

```text
data/qwen35_9b_heldout_policy_derived_pairwise_diagnostic_curriculum/
```

It includes only:

- `heldout_seed_multicall_0002`, `portfolio[2].weight`
- `heldout_seed_multicall_0009`, `refund_policy`

Manifest highlights:

- accepted rows: `120`
- repeat: `20`
- both candidate orders: `true`
- block size: `1536`
- zero/partial labels: `0/0`
- promotion allowed: `false`

## Focused SFT Gate

Start adapter:

```text
runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model
```

Output:

```text
runs/fastdllm_qwen35_9b_heldout_policy_derived_pairwise_from_ckpt275_step10_diag
```

Training settings:

- max steps: `10`
- save steps: `5`
- max train samples: `120`
- block size: `1536`
- truncation side: `left`
- `DISABLE_GROUP_TEXTS=1`
- learning rate: `1e-5`
- gradient accumulation: `4`
- LoRA: `r=8`, `alpha=16`
- target modules:
  `q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,out_proj,in_proj_a,in_proj_b`
- systemd user scope:
  `MemoryMax=28G`, `MemorySwapMax=4G`

Training result:

- checkpoint-5 and checkpoint-10 saved
- train loss: `3.686075782775879`
- train runtime: `156.342s`
- max observed GPU memory during training: about `30.5 GiB`

## Selector Eval

Full heldout policy selector gate:

```text
data/candidate_ranking/heldout_seed_policy_evidence_selector_toolname_argument_ranking_evidence.jsonl
```

Evaluated checkpoint:

```text
runs/fastdllm_qwen35_9b_heldout_policy_derived_pairwise_from_ckpt275_step10_diag/checkpoint-10/adapter_model
```

Output:

```text
runs/candidate_ranking/heldout_seed_policy_evidence_selector_derived_pairwise_diag_ckpt10_tournament.summary.json
```

Result:

| run | overall | argument values | tool names |
| --- | ---: | ---: | ---: |
| checkpoint-275 baseline | `150/152` | `121/123` | `29/29` |
| derived-pairwise checkpoint-10 | `150/152` | `121/123` | `29/29` |

Row-level comparison:

- changed predictions versus checkpoint-275: `0/152`
- remaining misses: `2`
  - `heldout_seed_multicall_0002`, `portfolio[2].weight`, target `0.334`,
    predicted `0.333`
  - `heldout_seed_multicall_0009`, `refund_policy`, target `full`, predicted
    non-target candidate

## Interpretation

Short focused pairwise SFT does not internalize the two derived-rule decisions,
even when training directly on repeated in-sample examples. This is a stronger
negative signal than the broader train-only pairwise results: the issue is not
just lack of examples in a broad curriculum.

Current implication:

- keep the deterministic derived-rule sidecar for the protected route;
- do not launch more of the same short pairwise SFT expecting these rules to
  move;
- next model-side attempt should change the objective or interface, for
  example explicit verifier/logit-margin training on the derived decisions,
  a small learned value adapter with numeric/policy features, or constrained
  decoding that calls the sidecar as a generation-time value scorer rather
  than asking the base diffusion adapter to learn the rule from A/B text.

This does not block the overall route. It clarifies that derived arithmetic
and policy-threshold choices should be treated as a separate value-reasoning
component, not as ordinary pairwise selector SFT.
