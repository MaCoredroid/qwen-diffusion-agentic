# Qwen3.5 Heldout Policy Close-Guard Scorecard

Date: 2026-06-28

## Purpose

Move the public multi-call close-guard stack to the clean 12-row heldout
policy-target slice:

```text
runs/planner_decomposition/heldout_seed_multicall_policy_targets.jsonl
```

This is the accepted heldout planner-policy target set. The row targets use
`gold_assistant == policy_planner_assistant`, so the standard evaluator can
score against `gold_assistant` directly.

## Lean Named Guard Stack

Run:

```text
runs/tool_sensitive_block_plans/heldout_seed_policy_evidence_selector/
  derived_pairwise_mode_prefix_name_value_closeguard_ckpt275_generation.jsonl
```

Settings:

```text
--guard-tool-call-mode
--guard-tool-json-prefix
--guard-tool-name-candidates
--guard-tool-value-candidates
--stop-after-schedule-tool-calls
```

Result:

| metric | score |
|---|---:|
| raw valid JSON | `11/12` |
| raw exact tool-name set | `11/12` |
| raw exact tool sequence | `11/12` |
| raw exact arguments | `11/12` |
| raw schema valid | `11/12` |
| raw required args present | `11/12` |
| constrained exact sequence | `12/12` |
| constrained exact arguments | `2/12` |
| close-token deferrals | `6` |
| JSON-prefix rejected tokens | `83` |
| JSON-prefix unsafe fallbacks | `83` |
| elapsed | `1091.1s` |
| max CUDA allocated / reserved | `18.80 GiB / 27.59 GiB` |

Diagnostic:

```text
runs/tool_sensitive_block_plans/heldout_seed_policy_evidence_selector/
  derived_pairwise_mode_prefix_name_value_closeguard_completability.json
```

The single miss is `heldout_seed_multicall_0004`, a nested `create_campaign`
case. The raw text corrupts JSON keys/structure inside a long
`campaign_details` array. The guard rejects `83` unsafe scheduled commits but
eventually allows unsafe fallback because no safe top-k replacement is found.

Target fallback ablation:

```text
runs/tool_sensitive_block_plans/heldout_seed_policy_evidence_selector/
  derived_pairwise_mode_prefix_name_value_closeguard_targetfallback_limit4.jsonl
```

Result: still `3/4`. Target fallback fires `0` times, so it does not address
the already-corrupted nested JSON prefix.

## Structural-Key Guard Ceiling

Run:

```text
runs/tool_sensitive_block_plans/heldout_seed_policy_evidence_selector/
  derived_pairwise_mode_prefix_name_value_jsonstruct_closeguard_ckpt275_generation.jsonl
```

Settings:

```text
--guard-tool-call-mode
--guard-tool-json-prefix
--guard-tool-name-candidates
--guard-tool-value-candidates
--force-schedule-token-kinds json_key,json_structure
--stop-after-schedule-tool-calls
```

Result:

| metric | score |
|---|---:|
| raw valid JSON | `12/12` |
| raw exact tool-name set | `12/12` |
| raw exact tool sequence | `12/12` |
| raw exact arguments | `12/12` |
| raw schema valid | `12/12` |
| raw required args present | `12/12` |
| constrained exact sequence | `12/12` |
| constrained exact arguments | `2/12` |
| generic key/structure forced tokens | `1246` |
| value candidate forced tokens | `725` |
| tool-name candidate forced tokens | `141` |
| JSON-prefix rejected tokens | `0` |
| JSON-prefix unsafe fallbacks | `0` |
| elapsed | `859.5s` |
| max CUDA allocated / reserved | `18.80 GiB / 27.81 GiB` |

Completability diagnostic:

```text
runs/tool_sensitive_block_plans/heldout_seed_policy_evidence_selector/
  derived_pairwise_mode_prefix_name_value_jsonstruct_closeguard_completability.json
```

Result:

| field | complete JSON segments | invalid segments | exact sequence | exact arguments |
|---|---:|---:|---:|---:|
| raw assistant | `29/29` | `0` | `12/12` | `12/12` |
| constrained assistant | `29/29` | `0` | `12/12` | `2/12` |

## Interpretation

The close guard and named candidate guards transfer to heldout, but the lean
stack is not enough for long nested JSON objects. The failure class is
structural skeleton drift, not value grounding: keys and punctuation inside a
large array can become unrecoverably invalid before the closing guard matters.

For this heldout slice:

- name/value candidate protection covers the semantic decisions;
- tool-call close protection prevents premature `</tool_call>` closure;
- JSON-prefix checking catches bad commits but cannot repair an already-broken
  nested prefix without a safe replacement;
- forcing `json_key,json_structure` closes the structural gap and reaches the
  protected ceiling of `12/12`.

Next model-side target: teach or predict the skeleton/key/structure path rather
than forcing it as an oracle. The practical next experiment is
skeleton-conditioned value infill:

1. keep tags, tool names, JSON keys, and punctuation as planned skeleton;
2. train argument-value slots with candidate/evidence supervision;
3. add on-policy AR-teacher KL/top-k labels on student diffusion states;
4. promote only if raw or constrained metrics move without relying on full
   schedule-key/structure forcing.
