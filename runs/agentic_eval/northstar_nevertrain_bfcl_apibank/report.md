# FLARE Never-Train BFCL/API-Bank Matched Eval

## Corrected Headline

The old `88/184` corrected diffusion row does not survive when rerun with the same strict clean-wave flags and the 19a50b7 generated-token audit. The strict rerun is `9/184` exact-args and `0/60` episode-exact.

| Backend | exact_args | episode exact | exact_seq | valid_xml | schema_ok | sec/turn | forwards/turn | audit | note |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| AR vLLM FR13 guided | 77/184 | 19/60 | 126/184 | 184/184 | 184/184 | 0.591 | n/a | n/a |  |
| Diffusion clean waves strict | 9/184 | 0/60 | 97/184 | 142/184 | 91/184 | 2.695 | 33.353 | projected_token_records_x_generated_token_offsets / projected_value=0 | same strict flags + generated-token audit |
| Diffusion old corrected | 88/184 | 19/60 | 141/184 | 182/184 | 147/184 | 1.604 | 10.293 | n/a | NOT comparable; predates strict left-prefix/generated-token audit |
| Diffusion legacy contaminated | 181/184 | 57/60 | 184/184 | 184/184 | 184/184 | 0.875 | 0.598 | n/a | CONTAMINATED; zero-forward/value projection path |

## Reconciliation

- The strict rerun used `--two-wave-grammar-forced-only`, `--two-wave-no-project-inside-parameter-value`, projected-token recording, and generated-token offset audit.
- Audit result: `projected_value_tokens_exact=0`, `wave1_value_tokens_counter=0`, `zero_projected_value_tokens_verified=1` over 184 turns.
- Conclusion: the prior `88/184` was still not an audit-equivalent clean-waves capability number. Simple-string values do not rescue the aggregate; one-turn API/BFCL rows also mostly fail under strict waves.

## Files

- Strict rows: `runs/agentic_eval/northstar_nevertrain_bfcl_apibank/diffusion_structural_only_strict/turns.jsonl`
- Strict audit: `runs/agentic_eval/northstar_nevertrain_bfcl_apibank/diffusion_structural_only_strict/projection_value_audit.json`
- AR-guided rows: `runs/agentic_eval/northstar_nevertrain_bfcl_apibank/ar-vllm-guided/turns.jsonl`
