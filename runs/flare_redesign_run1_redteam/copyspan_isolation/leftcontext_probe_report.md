# FLARE Left-Context-First Copy Probe

No new training. Reused the Run-1 copy-grounded checkpoint and the 12-record copy-span slice.
The left-context-first condition decoded scaffold/key/tool-name/tag structure through the normal careful path,
then allowed same-forward parallel commit only for `argument_value` schedule intervals.

## Required Conditions

| Condition | Run | Copy exact | Single-token exact | Value TPF | Value tokens/forwards | Valid tool JSON | Record exact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| (0) CONTROL careful | `arg8_tau099` | 41/41 | 1/1 | 1.000 | 499/499 | 12/12 | 12/12 |
| (1) ALL-PARALLEL | `arg32_tau050` | 0/41 | 0/1 | 1.233 | 439/356 | 0/12 | 0/12 |
| (2) LEFT-CONTEXT-FIRST, values-only parallel | `leftctx_value_arg32_tau050` | 29/41 | 1/1 | 1.496 | 178/119 | 12/12 | 5/12 |

## Left-Context Threshold Check

| Run | Copy exact | Single-token exact | Value TPF | Value tokens/forwards | Valid tool JSON | Record exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `leftctx_value_arg32_tau050` | 29/41 | 1/1 | 1.496 | 178/119 | 12/12 | 5/12 |
| `leftctx_value_arg32_tau070` | 36/41 | 1/1 | 1.187 | 178/150 | 12/12 | 9/12 |
| `leftctx_value_arg32_tau080` | 40/41 | 1/1 | 1.092 | 178/163 | 12/12 | 11/12 |
| `leftctx_value_arg32_tau085` | 40/41 | 1/1 | 1.047 | 178/170 | 12/12 | 11/12 |

## Single-Token Subset

Best speed/quality left-context point (`leftctx_value_arg32_tau080`): 1/1 exact.
The single-token span is `run1_copyspan_0007` `priority` = `normal` (1 token), exact=True.

## Verdict

H2 circuit disruption is not confirmed. Committing left context before value spans rescues copy quality from 0/41 under all-parallel to 29/41 at matched tau 0.50 and to 40/41 at tau 0.80 while value TPF remains >1.
The single-token copy value is exact in the left-context-first condition, so the previous first/single-token corruption was primarily decode-order dependent, not an unavoidable masked-right-context copy-circuit break.
This is a strong H1 result, but not a full speed gate pass: the best >1 TPF point is still 40/41 rather than the 41/41 careful baseline.
The remaining tau 0.80 miss is multi-token: `run1_copyspan_0006` `revision_id` target `rev-t214bv4rzc` (9 tokens).
