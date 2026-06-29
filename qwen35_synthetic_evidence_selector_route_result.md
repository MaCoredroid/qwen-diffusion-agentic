# Qwen3.5 Synthetic Evidence-Selector Route Result

Date: 2026-06-28

## Purpose

Move the public multi-call evidence-selector route onto a non-public synthetic
analogue slice before training another selector or boundary adapter.

The target slice is:

```text
data/toolcall_eval/synthetic_multicall_failure_analogues.jsonl
```

It has `8` cases and `24` gold tool calls:

- `4` voice-command camera cases, where the right answer is a repeated
  `activate_voice_command` call instead of the direct camera API.
- `4` security installation-code cases, where lock, detector, and alarm codes
  must stay scoped to their matching calls.

This is still protected runtime evidence, not raw diffusion model promotion:
the planner proposes the target text/spans, structural JSON/tool tags and stop
boundaries are guarded, and the selector narrows semantic spans before
generation.

## Runner

Added a reusable route runner:

```text
scripts/run_qwen35_evidence_selector_route.py
```

Default route:

```bash
.venv-fastdllm/bin/python scripts/run_qwen35_evidence_selector_route.py \
  --execute \
  --verify-existing
```

The runner writes:

```text
runs/tool_sensitive_block_plans/synthetic_multicall_failure_evidence_selector/route_plan.json
runs/tool_sensitive_block_plans/synthetic_multicall_failure_evidence_selector/route_plan.sh
runs/tool_sensitive_block_plans/synthetic_multicall_failure_evidence_selector/route_plan_execution.json
runs/tool_sensitive_block_plans/synthetic_multicall_failure_evidence_selector/route_plan_verification.json
```

GPU steps are wrapped in:

```text
systemd-run --user --scope --quiet -p MemoryMax=28G -p MemorySwapMax=4G
```

## Evidence Coverage

Planner input:

```text
runs/synthetic_multicall_failure_analogues/sequence_planner_bad_draft_safe_seqmismatch.jsonl
```

Evidence candidate schedule:

```text
runs/tool_sensitive_block_plans/synthetic_multicall_failure_evidence_selector/sampler_schedule_with_candidates_evidence.summary.json
```

Coverage:

- records: `8`
- argument blocks augmented: `60`
- argument blocks with sequence candidates: `60`
- argument sequence candidates: `128`
- tool-name blocks augmented: `24`
- tool-name blocks with sequence candidates: `24`
- tool-name target candidates present: `24`

Selector examples:

```text
data/candidate_ranking/synthetic_multicall_failure_evidence_selector_toolname_argument_ranking_evidence.summary.json
```

Coverage:

- examples: `84`
- usable examples: `84`
- argument-value examples: `60/60`
- tool-name examples: `24/24`
- target missing from candidates: `0`

## Selector Gate

Tournament:

```text
runs/candidate_ranking/synthetic_multicall_failure_evidence_selector_ckpt275_pairwise_tournament.summary.json
```

Result:

- overall: `84/84`
- argument values: `60/60`
- tool names: `24/24`
- pair comparisons: `120`
- elapsed: `28.3s`
- max allocated VRAM: `17.47 GiB`
- max reserved VRAM: `19.90 GiB`

Injected schedule:

```text
runs/tool_sensitive_block_plans/synthetic_multicall_failure_evidence_selector/sampler_schedule_with_evidence_pairwise_choices.summary.json
```

Result:

- selectors consumed: `84`
- selectors correct: `84`
- restricted schedule items: `165`
- candidate missing items: `0`

## Generation Gate

Generation summary:

```text
runs/tool_sensitive_block_plans/synthetic_multicall_failure_evidence_selector/evidence_toolargselector_structguard_ckpt275_generation.summary.json
```

Result:

- valid tool JSON: `8/8`
- exact tool sequence: `8/8`
- exact arguments: `8/8`
- schema valid: `8/8`
- required args present: `8/8`
- extra calls: `0`
- missing calls: `0`
- stop-boundary trims: `3`
- elapsed: `142.2s`
- generated tokens/sec: `13.60`
- max allocated VRAM: `17.81 GiB`
- max reserved VRAM: `28.49 GiB`

The repeated-call counter reports `4` rows because the voice-command family
intentionally calls `activate_voice_command` twice; exact sequence and exact
arguments are the decisive metrics for this slice.

Audit:

```text
runs/tool_sensitive_block_plans/synthetic_multicall_failure_evidence_selector/candidate_miss_audit.summary.json
```

Result:

- failed records: `0`
- mismatches: `0`
- missing calls: `0`
- extra calls: `0`
- invalid tool blocks: `0`

Route verifier:

```text
runs/tool_sensitive_block_plans/synthetic_multicall_failure_evidence_selector/route_plan_verification.json
```

Result:

- checks: `20`
- missing artifacts/summaries: `0`
- failed checks: `0`
- status: pass

## Interpretation

The evidence-selector route now generalizes from the public multi-call smoke to
the synthetic analogue slice that targets the two active failure families.

This is useful because it separates three questions:

1. Candidate proposal: evidence extraction can cover all semantic spans on this
   slice without target injection.
2. Candidate selection: the checkpoint-275 pairwise selector can choose every
   tool name and argument value under the same-call sketch prompt.
3. Scheduled generation: once semantic spans are singleton-constrained and
   structure/stop are guarded, the diffusion sampler can emit exact tool-call
   traces.

It does not yet prove learned raw behavior. The next promotion gate should be a
larger teacher/fresh multi-call slice and then a model-side adapter experiment
whose raw or constrained-decoder metrics improve without relying entirely on
runtime post-processing.
