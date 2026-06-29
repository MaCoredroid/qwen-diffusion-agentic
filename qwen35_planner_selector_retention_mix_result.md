# Qwen3.5 Planner/Selector/Retention Mix Result

Date: 2026-06-28

## Purpose

Create a promotion-relevant train-only recipe after the heldout diagnostics:
mix planner-pressure rows, selector/value rows, and retention rows without
training on public or heldout eval prompts.

This is a corpus and trainability result, not a promoted model result.

## Builder

Script:

```text
scripts/build_qwen35_planner_selector_retention_mix.py
```

The builder combines three train-only sources, performs token-label retention
audits at the final training block size, and can remove eval overlaps before
writing the final conversation JSON.

The new overlap controls are:

```text
--exclude-eval-jsonl
```

For each candidate row, the builder fingerprints both the user prompt and the
full user/assistant pair. Matching rows are removed before the token audit and
recorded in:

```text
eval_overlap_removed.jsonl
```

## Sources

Clean output:

```text
data/qwen35_9b_planner_selector_retention_mix_nooverlap_curriculum
```

Inputs:

- retention:
  `data/qwen35_9b_route_delta_trainonly_mix_curriculum`
- planner:
  `data/qwen35_9b_toolcall_sequence_planner_distill_no_public_multicall_smoke_curriculum`
- selector:
  `data/qwen35_9b_public_train_pairwise_pathaware_phrase_argsketch_curriculum`

Build settings:

- retention cap: `192`
- planner repeat: `2`
- selector cap: `160`
- block size: `1536`
- truncation side: `left`
- require full labels: `true`

## Overlap Audit

The first draft contained eval leakage:

- candidate rows: `382`
- exact/user overlaps: `5`
- all five overlaps were against `data/toolcall_eval/public_multicall_hermes_smoke.jsonl`
- all five removed rows came from the retention source

The cleaned build removed those rows before writing training data:

- final rows: `377`
- removed eval overlaps: `5`
- rejected rows after token-label audit: `0`
- zero-label rows after truncation: `0`
- partial-label rows after truncation: `0`

Clean overlap audit:

```text
runs/planner_selector_retention_mix_nooverlap_audit/train_vs_public_synthetic_heldout.json
```

Result:

- train records: `377`
- eval records: `45`
- exact overlap count: `0`
- user overlap count: `0`

The eval files checked were:

- `data/toolcall_eval/public_multicall_hermes_smoke.jsonl`
- `data/toolcall_eval/synthetic_multicall_failure_analogues.jsonl`
- `data/toolcall_eval/heldout_seed_multicall_2to3_clean.jsonl`
- `runs/planner_decomposition/heldout_seed_multicall_policy_targets.jsonl`

## Corpus Composition

Manifest:

```text
data/qwen35_9b_planner_selector_retention_mix_nooverlap_curriculum/train_agentic_mix.manifest
```

Source counts:

- route-delta retention: `187`
- sequence planner: `30`
- pairwise selector: `160`

Token-label audit:

- sequence length: min `410`, p50 `882`, p90 `1401`, max `1726`
- full labels: min `3`, p50 `5`, p90 `104`, max `440`
- kept labels match full labels for all accepted rows

## One-Step Trainability Gate

Start adapter:

```text
runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model
```

Output:

```text
runs/fastdllm_qwen35_9b_planner_selector_retention_mix_nooverlap_from_ckpt275_step1_gate
```

Run settings:

- max steps: `1`
- max train samples: `32`
- block size: `1536`
- truncation side: `left`
- learning rate: `1e-6`
- gradient accumulation: `4`
- LoRA: `r=8`, `alpha=16`
- target modules:
  `q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,out_proj,in_proj_a,in_proj_b`
- `VALUE_SPAN_LABEL_ONLY=1`
- systemd user scope:
  `MemoryMax=28G`, `MemorySwapMax=4G`

Training result:

- checkpoint saved
- train loss: `1.4096273183822632`
- train runtime: `16.2331s`
- train samples/sec: `0.246`

## Short Heldout Sweep

Training run:

```text
runs/fastdllm_qwen35_9b_planner_selector_retention_mix_nooverlap_from_ckpt275_step10
```

Settings:

- start adapter:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model`
- max steps: `10`
- saved checkpoints: `5`, `10`
- max train samples: `377`
- block size: `1536`
- truncation side: `left`
- learning rate: `1e-6`
- gradient accumulation: `4`
- LoRA: `r=8`, `alpha=16`
- `VALUE_SPAN_LABEL_ONLY=1`
- systemd user scope:
  `MemoryMax=28G`, `MemorySwapMax=4G`

Training result:

- train loss: `0.5359438180923461`
- train runtime: `156.0962s`
- train samples/sec: `0.256`
- observed training VRAM during run: about `30.5 GiB`

Heldout eval target:

```text
runs/planner_decomposition/heldout_seed_multicall_policy_targets.jsonl
```

Eval settings:

- full-context sampling
- forced `<tool_call>\n` prefix
- max new tokens: `900`
- block size: `32`
- small block size: `8`
- constrained tool decoding
- sequence-preserving constrained projection
- constrained max calls: `3`
- no protected sampler schedule
- no selector/sidecar injection

Outputs:

```text
runs/fastdllm_qwen35_9b_planner_selector_retention_mix_nooverlap_from_ckpt275_step10/checkpoint5_policy_targets_forcedprefix.summary.json
runs/fastdllm_qwen35_9b_planner_selector_retention_mix_nooverlap_from_ckpt275_step10/checkpoint10_policy_targets_forcedprefix.summary.json
```

Comparison:

| run | raw valid | raw exact seq | raw exact args | constrained name set | constrained seq | constrained args | extra calls | missing calls | repeated calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| checkpoint-275 baseline | `0/12` | `0/12` | `0/12` | `6/12` | `5/12` | `0/12` | `7` | `22` | `7` |
| mixed checkpoint-5 | `0/12` | `0/12` | `0/12` | `7/12` | `5/12` | `0/12` | `1` | `23` | `1` |
| mixed checkpoint-10 | `0/12` | `0/12` | `0/12` | `5/12` | `3/12` | `0/12` | `4` | `23` | `4` |
| heldout planner diagnostic checkpoint-25 | `1/12` | `0/12` | `0/12` | `7/12` | `6/12` | `0/12` | `2` | `23` | `1` |

Row-level constrained sequence movement:

- checkpoint-5 keeps baseline hits `0004`, `0005`, `0011`, gains `0006` and
  `0012`, but loses `0009` and `0010`; aggregate remains `5/12`.
- checkpoint-10 keeps only `0004`, `0005`, and `0011`; it loses `0009` and
  `0010` versus checkpoint-275 and adds no new constrained sequence hits.

Eval runtime:

- checkpoint-5: `891.26s`, `5282` generated tokens, `5.93` tokens/sec
- checkpoint-10: `932.60s`, `5596` generated tokens, `6.00` tokens/sec

## Interpretation

The corpus gives a clean trainable substrate:

- it is train-only with explicit eval-overlap filtering;
- it preserves all accepted labels at `1536` tokens;
- it trains from checkpoint-275 without immediate memory/runtime failure on the
  local RTX 5090.

The short sweep does not prove model-side improvement:

- checkpoint-5 ties the checkpoint-275 constrained sequence aggregate but swaps
  row-level successes and still has `0/12` exact arguments;
- checkpoint-10 regresses constrained sequence to `3/12`;
- neither checkpoint improves raw valid JSON, raw exact sequence, or exact
  arguments.

Do not scale this exact recipe unchanged. It reduces extra/repeated raw calls,
which is useful signal, but the missing-call count and exact-argument gap remain
unchanged or worse. The next branch should change the objective/data balance:
more explicit tool-sequence planning pressure, less selector-only low-label
mass, and a separate value/argument grounding objective before running longer
than 10 steps.

## Planner-Heavy Follow-Up Substrate

After the first short sweep failed the heldout gate, a revised corpus was built
with the same overlap protections but a different balance:

```text
data/qwen35_9b_plannerheavy_selectorlight_retention_mix_nooverlap_curriculum
```

Build changes:

- retention cap stays `192`
- planner repeat increases from `2` to `6`
- selector cap decreases from `160` to `80`
- block size remains `1536`
- truncation side remains `left`

Composition:

- total accepted rows: `357`
- route-delta retention: `187`
- sequence planner: `90`
- pairwise selector: `80`
- removed eval overlaps: `5`, all from retention
- rejected rows after token-label audit: `0`
- zero-label / partial-label rows after truncation: `0` / `0`

Independent overlap audit:

```text
runs/plannerheavy_selectorlight_retention_mix_nooverlap_audit/train_vs_public_synthetic_heldout.json
```

Result:

- train records: `357`
- eval records: `45`
- exact overlap count: `0`
- user overlap count: `0`

One-step gate:

```text
runs/fastdllm_qwen35_9b_plannerheavy_selectorlight_retention_mix_nooverlap_from_ckpt275_step1_gate
```

Result:

- checkpoint saved
- train loss: `1.3001604080200195`
- train runtime: `16.2254s`
- train samples/sec: `0.247`

Interpretation: this revised substrate is clean and trainable. It is not yet a
model result. The next paid eval should be a `5/10` step heldout policy-target
run only if we accept the premise that the previous mix was underweighting
planner rows and overweighting low-label selector rows.

### Planner-Heavy Heldout Sweep

Training run:

```text
runs/fastdllm_qwen35_9b_plannerheavy_selectorlight_retention_mix_nooverlap_from_ckpt275_step10
```

Training result:

- saved checkpoints: `5`, `10`
- train loss: `0.428307843208313`
- train runtime: `156.1422s`
- train samples/sec: `0.256`
- observed training VRAM during run: about `30.5 GiB`

Heldout eval outputs:

```text
runs/fastdllm_qwen35_9b_plannerheavy_selectorlight_retention_mix_nooverlap_from_ckpt275_step10/checkpoint5_policy_targets_forcedprefix.summary.json
runs/fastdllm_qwen35_9b_plannerheavy_selectorlight_retention_mix_nooverlap_from_ckpt275_step10/checkpoint10_policy_targets_forcedprefix.summary.json
```

Comparison against checkpoint-275 and the earlier balanced mix:

| run | raw valid | raw exact seq | raw exact args | constrained name set | constrained seq | constrained args | extra calls | missing calls | repeated calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| checkpoint-275 baseline | `0/12` | `0/12` | `0/12` | `6/12` | `5/12` | `0/12` | `7` | `22` | `7` |
| balanced checkpoint-5 | `0/12` | `0/12` | `0/12` | `7/12` | `5/12` | `0/12` | `1` | `23` | `1` |
| balanced checkpoint-10 | `0/12` | `0/12` | `0/12` | `5/12` | `3/12` | `0/12` | `4` | `23` | `4` |
| planner-heavy checkpoint-5 | `0/12` | `0/12` | `0/12` | `7/12` | `4/12` | `1/12` | `3` | `21` | `3` |
| planner-heavy checkpoint-10 | `0/12` | `0/12` | `0/12` | `7/12` | `4/12` | `0/12` | `5` | `22` | `5` |

Row-level notes:

- planner-heavy checkpoint-5 gets the first constrained exact-argument hit on
  this heldout policy gate: `heldout_seed_multicall_0006`;
- that checkpoint also loses constrained sequence aggregate versus
  checkpoint-275 (`4/12` vs `5/12`), so it is not a promotion;
- planner-heavy checkpoint-10 keeps constrained sequence at `4/12` but loses
  the exact-argument hit, so the value signal is not stable under more steps;
- both planner-heavy checkpoints keep raw valid JSON and raw exact sequence at
  `0/12`.

Interpretation: the planner-heavy balance changes behavior and exposes a real
argument-value signal, but it still breaks sequence retention. The next recipe
should not simply increase planner repeat again. It should combine:

- sequence anti-regression rows for the checkpoint-275 constrained-sequence
  hits;
- explicit planner rows for missing-call reduction;
- a separate value/argument grounding objective focused on rows like `0006`;
- public one-call/tool-result retention gates before any promotion.

## Next Gates

Minimum gates before considering promotion:

- heldout policy targets, forced prefix, no protected selector:
  constrained exact sequence must beat checkpoint-275's `5/12` and exact
  arguments must move above `0/12`;
- public multi-call guarded planner route:
  must not regress the active protected `11/12` sequence and `10/12` argument
  line;
- public one-call and tool-result routes:
  must not regress current exact sequence/argument retention lanes;
- masked candidate-ranking:
  must beat or tie checkpoint-275 on heldout public argument values.
