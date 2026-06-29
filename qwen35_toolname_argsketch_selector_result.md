# Qwen3.5 Tool-Name Argument-Sketch Selector Result

Date: 2026-06-28

## Purpose

Test whether the remaining heldout tool-name selector miss is caused by missing
local call context rather than model capacity or insufficient SFT.

This is a selector/context gate. It is not raw diffusion generation promotion.

## Change

The path-aware selector prompt now includes a same-call argument sketch for
`tool_name` rows. The sketch is derived from schedule rows with the same
`tool_call_index` and lists argument keys plus target text.

Example for the previous heldout miss:

```text
Span kind: tool_name
Tool call index: 1
JSON key: name
JSON path: name
Same-call argument sketch:
- temperature: 72
- location: "living room"
Candidates:
0: "set_thermostat"
1: "activate_security_cameras"
```

Code paths updated:

- `scripts/build_candidate_ranking_examples.py`
- `scripts/build_candidate_pairwise_curriculum.py`
- `scripts/eval_fastdllm_candidate_pairwise_tournament.py`

## Artifacts

Heldout examples:

```text
data/candidate_ranking/public_multicall_toolname_argument_ranking_pathaware_phrase_argsketch_12.jsonl
```

Coverage:

- records: `12`
- examples: `99`
- usable examples: `99`
- argument-value examples: `68`
- tool-name examples: `31`
- target missing from candidates: `0`

Train examples:

```text
data/candidate_ranking/public_train_multicall_toolname_argument_ranking_pathaware_phrase_argsketch.jsonl
```

Coverage:

- records: `56`
- examples: `367`
- usable examples: `328`
- usable argument-value examples: `173`
- usable tool-name examples: `155`
- target missing from candidates: `39`

Promotion-eligible curricula:

```text
data/qwen35_9b_public_train_pairwise_toolname_pathaware_phrase_argsketch_curriculum
data/qwen35_9b_public_train_pairwise_pathaware_phrase_argsketch_curriculum
```

Manifests:

- tool-name only: `240` accepted rows, `0` rejected labels,
  `contains_eval_slice=false`, `promotion_allowed=true`
- tool-name plus argument-value: `616` accepted rows, `0` rejected labels,
  `contains_eval_slice=false`, `promotion_allowed=true`

## Heldout Gate

Checkpoint-275 tournament:

```text
runs/candidate_ranking/public_multicall_pathaware_phrase_argsketch12_ckpt275_pairwise_tournament.summary.json
```

Result:

- overall: `99/99`
- argument values: `68/68`
- tool names: `31/31`
- pair comparisons: `616`
- elapsed: `248.6s`
- max allocated VRAM: `18.47 GiB`
- max reserved VRAM: `27.46 GiB`

Delta versus the prior path-aware phrase selector gate:

| gate | overall | argument values | tool names |
| --- | ---: | ---: | ---: |
| path-aware phrase | `98/99` | `68/68` | `30/31` |
| path-aware phrase + same-call argument sketch | `99/99` | `68/68` | `31/31` |

The previous miss on case `c483f963-8a29-4ff0-a684-89be0d0f2843`, call index
`1`, is fixed: the selector chooses `set_thermostat` over
`activate_security_cameras` once it can see `temperature: 72` and
`location: "living room"`.

## Interpretation

The remaining tool-name miss was a context/boundary problem. More SFT on the
old prompt did not move any heldout rows, while adding same-call argument
context fixed the miss without training.

Implication for the behavior-preserving AR-to-diffusion recipe:

- tool-name blocks should not be treated as independent scalar spans;
- the boundary policy should expose a local call sketch before committing a
  tool name, or delay tool-name commitment until enough argument evidence is
  available;
- selector prompts and future learned selector adapters should carry
  `tool_call_index`, `json_path`, and same-call argument evidence;
- this is still component evidence. Model promotion still requires raw or
  constrained-decoder movement on heldout public/teacher gates.

## Sampler Follow-Up

Follow-up result:

```text
qwen35_argsketch_toolargselector_sampler_result.md
```

The `99/99` arg-sketch tournament was injected into the protected sampler for
both `tool_name` and `argument_value` spans:

```text
runs/tool_sensitive_block_plans/public_multicall_gold_sampler_schedule_with_pairwise_argsketch_choices_12.summary.json
```

Injection summary:

- selectors consumed: `99`
- selectors correct: `99`
- restricted schedule items: `226`
- candidate missing items: `0`

Generation result:

```text
runs/tool_sensitive_block_plans/public_multicall_argsketch_toolargselector_structguard_ckpt275_generation.summary.json
```

- exact tool sequence: `12/12`
- exact arguments: `12/12`
- valid JSON: `12/12`
- schema valid: `12/12`

Final audit:

```text
runs/tool_sensitive_block_plans/public_multicall_argsketch_toolargselector_structguard_ckpt275_candidate_miss_audit.summary.json
```

Audit result: `0` failed records, `0` mismatches, `0` missing/extra calls, and
`0` invalid tool blocks.

Interpretation: the selector-owned semantic-span path now covers tool names and
argument values for this heldout slice. This remains protected/gold-schedule
evidence, not raw diffusion generation promotion.
