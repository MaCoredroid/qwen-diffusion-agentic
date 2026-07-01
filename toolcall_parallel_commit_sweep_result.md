# Tool-Call Parallel-Commit Sweep Result

Date: 2026-07-01

## Setup

Command family: `scripts/eval_fastdllm_toolcall_cases.py` with base `models/qwen3.5-9b-fastdllm-init`, adapter
`runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model`, full-context
sampling, `runs/causal_value_span_decisive/schedules/raw_baseline_merged_schedule.jsonl`, raw lane value/name/candidate
forcing off, temp 0.0, guards on:

- `--guard-tool-call-mode`
- `--guard-tool-json-prefix`
- `--tool-prefix-guard-mode hermes_json`
- `--force-tool-call-prefix`
- `--stop-after-schedule-tool-calls`

Slices:

- heldout policy-target 12: `runs/planner_decomposition/heldout_seed_multicall_policy_targets.jsonl`
- public multicall 12: `data/toolcall_eval/public_multicall_hermes_smoke.jsonl`

Baseline exact-argument targets from `runs/causal_value_span_decisive`: heldout `7/12`, public `10/12`.

Implementation note: the Hermes JSON prefix guard has no practical full allowed-token enumerator, so confidence is the
selected grammar-safe token's full-softmax probability after mask-token ban. This is the same implemented Fast-dLLM
confidence-threshold proxy used by the harness. At low taus, `zero_fallbacks` means the stale-logit same-forward lane
could not safely commit and the sampler fell back to the existing guarded path; those low-tau `tokens/forward` numbers
are therefore optimistic lane counters, not end-to-end speedups. Tau `0.99` has zero fallbacks, so its tpf is the clean
ceiling measurement.

## Results

| tau | slice | valid_JSON | exact_sequence | exact_arguments | tokens/forward | structural_tpf | value_tpf | forced% | unsafe_blocked | zero_fallbacks | elapsed_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.99 | heldout | 12/12 | 10/12 | 6/12 | 1.092 | 1.132 | 1.007 | 3.6% | 0 | 0 | 300.5 |
| 0.99 | public | 12/12 | 11/12 | 10/12 | 1.126 | 1.175 | 1.016 | 6.0% | 0 | 0 | 230.3 |
| 0.95 | heldout | 8/12 | 6/12 | 3/12 | 1.109 | 1.135 | 1.031 | 2.5% | 22 | 20 | 419.3 |
| 0.95 | public | 9/12 | 8/12 | 5/12 | 1.228 | 1.285 | 1.089 | 7.0% | 24 | 22 | 238.7 |
| 0.90 | heldout | 3/12 | 1/12 | 1/12 | 1.091 | 1.122 | 1.083 | 1.7% | 46 | 41 | 431.0 |
| 0.90 | public | 6/12 | 4/12 | 1/12 | 1.266 | 1.357 | 1.150 | 6.8% | 38 | 34 | 239.6 |
| 0.80 | heldout | 2/12 | 0/12 | 0/12 | 1.136 | 1.298 | 1.101 | 2.2% | 64 | 60 | 323.6 |
| 0.80 | public | 5/12 | 1/12 | 0/12 | 1.293 | 1.379 | 1.299 | 5.2% | 44 | 39 | 243.1 |
| 0.70 | heldout | 0/12 | 0/12 | 0/12 | 1.082 | 1.151 | 1.125 | 0.9% | 59 | 59 | 580.4 |
| 0.70 | public | 1/12 | 0/12 | 0/12 | 1.327 | 1.535 | 1.255 | 5.0% | 50 | 46 | 247.6 |
| 0.50 | heldout | 0/12 | 0/12 | 0/12 | 1.378 | 1.810 | 1.211 | 1.6% | 62 | 61 | 299.6 |
| 0.50 | public | 0/12 | 0/12 | 0/12 | 1.476 | 1.783 | 1.406 | 4.1% | 48 | 46 | 242.5 |

## Readout

No requested tau holds both baseline exact-argument targets. Tau `0.99` holds public at `10/12`, but heldout drops from
baseline `7/12` to `6/12`; the lost row is `heldout_seed_multicall_0010` (`exact_sequence` and `exact_arguments` both
flip from true to false versus the stored raw baseline).

The clean no-fallback ceiling point is therefore tau `0.99`: heldout `1.092` tpf and public `1.126` tpf. Structural
tokens are only `1.132-1.175` tpf, while value spans are effectively sequential at `1.007-1.016` tpf. The hypothesized
tool-call boilerplate bulk-commit regime did not appear.

Conclusion: on this 9B target tool-call distribution, same-forward confident-run parallel commit does not produce a
quality-preserving speedup beyond roughly `1.0-1.1x`. Lower taus increase lane tpf in places, but quality collapses and
malformed generations often run long. No promotion was performed.
