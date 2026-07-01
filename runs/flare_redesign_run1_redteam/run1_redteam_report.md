# Run 1 Red-Team Confound Report

No PASS/FAIL verdict is declared here. This report only resolves the requested confounds.

## Matched Native Baseline Control

| tau | base heldout args/valid | copy heldout args/valid | base public args/valid | copy public args/valid |
| --- | ---: | ---: | ---: | ---: |
| 0.00 | 0/12, 0/12 | 0/12, 0/12 | 0/12, 0/12 | 0/12, 0/12 |
| 0.50 | 0/12, 0/12 | 0/12, 1/12 | 0/12, 0/12 | 0/12, 1/12 |
| 0.70 | 0/12, 2/12 | 0/12, 2/12 | 0/12, 3/12 | 0/12, 3/12 |
| 0.80 | 0/12, 1/12 | 0/12, 7/12 | 0/12, 5/12 | 0/12, 9/12 |
| 0.90 | 0/12, 2/12 | 0/12, 9/12 | 3/12, 6/12 | 3/12, 10/12 |
| 0.95 | 2/12, 4/12 | 3/12, 10/12 | 3/12, 7/12 | 8/12, 11/12 |
| 0.99 | 3/12, 6/12 | 4/12, 10/12 | 8/12, 9/12 | 8/12, 11/12 |

The pre-copy adapter does not reproduce the old guarded 7/12 heldout and 10/12 public baseline under this raw native matched route; this isolates the raw native eval/policy route as a confound.

## GSM8K Retention

Validated fullctx/fresh/mask-ban retention on copy-grounded: strict 15/20 = 0.75, flex 15/20 = 0.75.
Prior validated B1000 summary artifact: strict 13/20 = 0.65, flex 11/20 = 0.55.
Discarded: `runs/flare_redesign_run1_eval/gsm8k_first20.summary.json` because it used the broken phaseA_fewshot/fastdllm_anywhere path.

## Copy-Span Isolation

| cell | record exact | copy arg exact | value tokens/forward | value tokens/forwards |
| --- | ---: | ---: | ---: | ---: |
| arg32_tau000 | 0/12 | 0/41 | 13.49 | 499/37 |
| arg32_tau050 | 0/12 | 0/41 | 1.23 | 439/356 |
| arg8_tau099 | 12/12 | 41/41 | 1.00 | 499/499 |

Schedule metadata: arg8 and arg32 both carry `denoise_steps=8` for argument_value tokens; in the active parallel-commit path, actual few-step pressure is measured by value-forward visits. The arg32/tau0.00 cell gives high copy value TPF but no syntax/exactness; arg8/tau0.99 preserves copy exactly but is effectively 1 value token/forward.

## Native Copy-vs-Derived Proxy at tau 0.99

| run | record exact/valid | copy arg exact | derived arg exact | value tokens/forward |
| --- | ---: | ---: | ---: | ---: |
| base_heldout | 3/12, 6/12 | 19/52 | 14/32 | 1.004 |
| run1_heldout | 4/12, 10/12 | 41/52 | 18/32 | 1.001 |
| base_public | 8/12, 9/12 | 49/60 | 5/11 | 1.002 |
| run1_public | 8/12, 11/12 | 55/60 | 5/11 | 1.000 |

Artifacts are listed in `run1_redteam_report.json`.
