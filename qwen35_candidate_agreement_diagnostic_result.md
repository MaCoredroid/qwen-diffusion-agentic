# Qwen3.5 Candidate Agreement Diagnostic

Date: 2026-06-28

## Purpose

Add a behavior-retention diagnostic before scaling more diffusion training:
score the same candidate choices with a reference model and a diffusion
checkpoint, then compare target accuracy, prediction agreement, regressions,
improvements, and margins.

This is the first local version of the AR-agreement / introspective-consistency
metric discussed in the behavior-preserving recipe. The final version should use
the true agentic AR Qwen reference. This first smoke validates the scoring loop
using the converted Fast-DLLM init as the reference because local Transformers
cannot currently instantiate raw Qwen3.5.

## Code

Added:

```text
scripts/eval_qwen_ar_diffusion_candidate_agreement.py
```

Modes:

- `--mode ar`: score candidate-index choices with an AR model by causal
  log-probability of the answer index.
- `--mode fastdllm`: score candidate-index choices with Fast-DLLM by masked
  log-probability of the answer index.
- `--mode fastdllm_causal`: score candidate-index choices with a converted
  Fast-DLLM model by causal next-token log-probability. This is the current
  local AR-proxy path because the converted init uses raw Qwen3.5 text weights
  and the model runs with causal masks outside the diffusion training/scoring
  path.
- `--mode compare`: compare two score JSONL files.

Compile gate:

```bash
.venv-fastdllm/bin/python -m py_compile \
  scripts/eval_qwen_ar_diffusion_candidate_agreement.py
```

Result: passed.

## True AR Reference Blocker

Attempted local AR Qwen3.5-9B scoring:

```bash
.venv-fastdllm/bin/python scripts/eval_qwen_ar_diffusion_candidate_agreement.py \
  --mode ar \
  --examples-jsonl data/candidate_ranking/synthetic_multicall_failure_analogue_index_ranking.jsonl \
  --out-jsonl runs/candidate_agreement/synthetic8_qwen35_ar_limit4.jsonl \
  --limit 4 \
  --min-candidates 2 \
  --ar-model /home/mark/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --load-in-4bit \
  --local-files-only
```

It failed before model load:

```text
ValueError: checkpoint has model type `qwen3_5` but Transformers does not
recognize this architecture.
```

The local raw Qwen3.5 snapshot is present, but the installed Transformers build
does not know `qwen3_5` and the snapshot does not provide a usable AutoModel
remote-code mapping for this environment. Options:

- upgrade/use a Transformers build that supports Qwen3.5,
- add/register a local text-only Qwen3.5 AR loader,
- or route AR scoring through SGLang if logprobs are available.

## Smoke: Converted Init vs Fixed `bd_size=16` Adapter

Reference score file:

```text
runs/candidate_agreement/synthetic8_fastdllm_init_limit4.jsonl
```

Candidate score file:

```text
runs/candidate_agreement/synthetic8_bd16_ckpt5_fastdllm_limit4.jsonl
```

Comparison:

```text
runs/candidate_agreement/synthetic8_fastdllm_init_vs_bd16_ckpt5_limit4.summary.json
```

Slice:

- `data/candidate_ranking/synthetic_multicall_failure_analogue_index_ranking.jsonl`
- first `4` examples with at least `2` candidates
- all four are `voice_command_camera` tool-name choices

Result:

| metric | converted init | fixed `bd_size=16` ckpt5 |
|---|---:|---:|
| accuracy | `2/4` | `3/4` |
| prediction agreement | `3/4` | `3/4` |
| candidate regressions vs init | n/a | `0` |
| candidate improvements vs init | n/a | `1` |

Row-level:

- `synthetic_voice_command_camera_001`: init predicts `set_thermostat`, bd16
  predicts correct `activate_voice_command`.
- `synthetic_voice_command_camera_002`: both correct.
- `synthetic_voice_command_camera_003`: both predict `set_thermostat`, still
  wrong.
- `synthetic_voice_command_camera_004`: both correct.

Interpretation:

- The diagnostic loop works and produces a useful behavior-drift table.
- The fixed `bd_size=16` branch has a small positive candidate-selection signal
  over the converted init on this tiny synthetic slice.
- This is not promotion evidence. It is only a smoke test for the new metric.

## Heldout Policy Candidate Smoke

The same diagnostic was run on the heldout policy evidence-selector candidate
file:

```text
data/candidate_ranking/heldout_seed_policy_evidence_selector_toolname_argument_ranking_evidence.jsonl
```

Settings:

- first `40` usable examples with at least `2` candidates
- `36` argument-value choices
- `4` tool-name choices
- reference: converted Fast-DLLM init, no adapter
- candidate: fixed `bd_size=16` checkpoint-5 adapter

Outputs:

```text
runs/candidate_agreement/heldout_policy_fastdllm_init_limit40.summary.json
runs/candidate_agreement/heldout_policy_bd16_ckpt5_limit40.summary.json
runs/candidate_agreement/heldout_policy_fastdllm_init_vs_bd16_ckpt5_limit40.summary.json
```

Result:

| metric | converted init | fixed `bd_size=16` ckpt5 |
|---|---:|---:|
| overall accuracy | `38/40` | `38/40` |
| argument-value accuracy | `34/36` | `34/36` |
| tool-name accuracy | `4/4` | `4/4` |
| prediction agreement | n/a | `40/40` |
| regressions vs init | n/a | `0` |
| improvements vs init | n/a | `0` |
| median target margin | `3.75` | `4.25` |

The two shared misses are both on `heldout_seed_multicall_0002`:

- `growth_rate`: target `6`, predicted `3`
- `weight`: target `0.334`, predicted `0.333`

Interpretation:

- The fixed `bd_size=16` adapter does not change candidate choices on this
  first heldout policy subset.
- It does slightly raise the median target margin, but that is not enough to
  claim behavior improvement.
- The remaining misses are derived numeric/policy decisions, not simple
  candidate extraction. This reinforces the need for explicit AR-reference
  behavior metrics and/or a tool-result/policy reasoning target, not more
  uniform value-span masking.
- Next useful diagnostic run is true AR-vs-diffusion once the AR reference
  loader is available, plus a full 88-example nontrivial heldout sweep if the
  margin signal remains interesting.

## Heldout Policy Full Nontrivial Sweep

The full nontrivial heldout policy candidate set was then run with
`--min-candidates 2` and no limit.

Outputs:

```text
runs/candidate_agreement/heldout_policy_fastdllm_init_nontrivial.summary.json
runs/candidate_agreement/heldout_policy_bd16_ckpt5_nontrivial.summary.json
runs/candidate_agreement/heldout_policy_fastdllm_init_vs_bd16_ckpt5_nontrivial.summary.json
```

The compare script was fixed during this run to avoid dropping duplicate
semantic keys; it now compares by row index plus semantic key. Compile gate
passed after the fix.

Result:

| metric | converted init | fixed `bd_size=16` ckpt5 |
|---|---:|---:|
| overlap | `88` | `88` |
| overall accuracy | `84/88` | `85/88` |
| argument-value accuracy | `71/75` | `72/75` |
| tool-name accuracy | `13/13` | `13/13` |
| prediction agreement | n/a | `87/88` |
| regressions vs init | n/a | `0` |
| improvements vs init | n/a | `1` |
| median target margin | `3.625` | `3.75` |

One changed prediction:

- `heldout_seed_multicall_0012`, `total_tax_amount`: converted init predicts
  `720`, fixed `bd_size=16` predicts correct `160`.

Remaining fixed `bd_size=16` misses:

- `heldout_seed_multicall_0002`, `weight`: target `0.334`, predicted `0.333`
- `heldout_seed_multicall_0002`, `growth_rate`: target `6`, predicted `3`
- `heldout_seed_multicall_0009`, `refund_policy`: target `full`, predicted
  `partial`

Interpretation:

- The fixed `bd_size=16` adapter has a small positive value-choice signal over
  the converted init on the full nontrivial heldout policy candidate set:
  `+1` correct argument choice, `0` regressions.
- This agrees with the heldout generation result: fixed `bd_size=16` helps
  route/order and slightly improves some masked value choices, but it is not
  sufficient for exact argument grounding in generated tool calls.
- The remaining candidate misses are derived/normative choices, not simple
  string copying. They should feed a policy/arithmetic reasoning target or
  AR-reference distillation, not another uniform value-span mask run.

## Causal Fast-DLLM AR-Proxy Sweep

Because raw Qwen3.5 AR loading is blocked in the current Transformers build,
`fastdllm_causal` mode was added and run over the same heldout nontrivial set.
This uses the converted Qwen3.5 Fast-DLLM model in its normal causal eval path,
not masked candidate scoring.

Outputs:

```text
runs/candidate_agreement/heldout_policy_fastdllm_init_causal_nontrivial.summary.json
runs/candidate_agreement/heldout_policy_bd16_ckpt5_causal_nontrivial.summary.json
runs/candidate_agreement/heldout_policy_fastdllm_init_causal_vs_bd16_ckpt5_causal_nontrivial.summary.json
```

Result:

| metric | causal converted init | causal fixed `bd_size=16` ckpt5 |
|---|---:|---:|
| overlap | `88` | `88` |
| overall accuracy | `84/88` | `85/88` |
| argument-value accuracy | `71/75` | `72/75` |
| tool-name accuracy | `13/13` | `13/13` |
| prediction agreement | n/a | `87/88` |
| regressions vs init | n/a | `0` |
| improvements vs init | n/a | `1` |
| median target margin | `3.625` | `3.75` |

The causal AR-proxy sweep matches the masked-choice sweep on predictions. The
same changed row improves:

- `heldout_seed_multicall_0012`, `total_tax_amount`: converted init predicts
  `720`, fixed `bd_size=16` predicts correct `160`.

The same three misses remain:

- `heldout_seed_multicall_0002`, `weight`: target `0.334`, predicted `0.333`
- `heldout_seed_multicall_0002`, `growth_rate`: target `6`, predicted `3`
- `heldout_seed_multicall_0009`, `refund_policy`: target `full`, predicted
  `partial`

Interpretation:

- We now have a working local causal reference proxy even without raw
  Transformers Qwen3.5 support.
- The fixed `bd_size=16` adapter does not appear to damage causal candidate
  preferences on this heldout policy set; it gives a small `+1` value-choice
  lift.
- Since masked and causal candidate-choice summaries match here, the larger
  generation failure is likely not just local candidate preference. It is the
  end-to-end generation process: route/order, JSON grammar, stop boundaries,
  and committing/copying values inside generated tool-call spans.
