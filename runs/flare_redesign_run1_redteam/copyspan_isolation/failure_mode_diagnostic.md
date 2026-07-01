# Copy-Span Failure-Mode Diagnostic

No new training was run. This diagnostic compares existing `arg32_tau050` parallel-copy failures against the `arg8_tau099` careful copy control on the same 41 labeled copy spans.

## Breakdown

| bucket | count | fraction of 41 parse-level failures | CoDD implication |
| --- | ---: | ---: | --- |
| Value preserved, scaffold/alignment broken | 19 | 46.3% | Neither A nor B; grammar/scaffold failure |
| B: single/first-token copy corruption | 22 | 53.7% | CoDD unlikely to help these value failures |
| A: first-token OK, later-token inconsistency | 0 | 0.0% | No evidence for factorization barrier |

Among true value-token failures only: B = 22/22, A = 0/22.

## Readout

The cheap diagnostic does not support the CoDD premise for these failures. The value failures are first-token/copy-circuit disruptions, not multi-token factorization errors with a correct first token. A large separate bucket copies the value text exactly somewhere in the malformed output but loses the tool-call scaffold, which is also outside CoDD's joint-value-prior scope.

Artifacts:

- `failure_mode_diagnostic.json`
- `arg32_tau050.jsonl`
- `arg8_tau099.jsonl`
- `copyspan_blocks_12.jsonl`
