# FLARE North-Star Matched Eval

## Corrected Headline

The previous diffusion per-call waves headline is invalid as a capability claim. Under strict structural-only projection with generated-token audit, diffusion clean waves collapse below both AR-guided and diffusion-careful.

| Backend | exact_args | episode exact | exact_seq | valid_xml | schema_ok | sec/turn | forwards/turn | audit | note |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| AR vLLM FR13 guided | 50/63 | 13/20 | 63/63 | 63/63 | 59/63 | 1.120 | n/a | n/a |  |
| Diffusion careful, no waves | 34/63 | 8/20 | 55/63 | 59/63 | 47/63 | 6.049 | 95.254 | no_projection_events / projected_value=0 | honest no-tricks diffusion baseline |
| Diffusion clean waves strict | 3/63 | 0/20 | 36/63 | 46/63 | 13/63 | 6.546 | 94.317 | projected_token_records_x_generated_token_offsets / projected_value=0 | strict flags: forced-only + no project inside values + generated-token audit |
| Diffusion waves legacy | 55/63 | 15/20 | 63/63 | 63/63 | 58/63 | 1.442 | 8.905 | n/a | CONTAMINATED; projected values/old scheduler, not a capability row |

## Reconciliation

- Matched-20 strict clean waves: `3/63` exact-args, `0/20` episode-exact, audit mode `projected_token_records_x_generated_token_offsets`, `projected_value_tokens_exact=0`.
- Diffusion-careful: `34/63` exact-args, `8/20` episode-exact, no projection events. This is the honest model-without-waves row.
- AR-guided remains `50/63` exact-args, `13/20` episode-exact. The strict clean wave mode is below AR-guided and below careful diffusion.
- Failure taxonomy: `runs/agentic_eval/northstar_matched_ar_vllm_vs_diffusion/clean_waves_failure_taxonomy.md` and `runs/agentic_eval/northstar_matched_ar_vllm_vs_diffusion/clean_waves_failure_taxonomy.jsonl`.

## Files

- AR-guided rows: `runs/agentic_eval/northstar_matched_ar_vllm_vs_diffusion/ar-vllm-guided/turns.jsonl`
- Diffusion careful rows: `runs/agentic_eval/northstar_matched_ar_vllm_vs_diffusion/diffusion_careful/turns.jsonl`
- Strict clean waves rows: `runs/agentic_eval/northstar_matched_ar_vllm_vs_diffusion/diffusion_structural_only_strict/turns.jsonl`
- Strict clean audit: `runs/agentic_eval/northstar_matched_ar_vllm_vs_diffusion/diffusion_structural_only_strict/projection_value_audit.json`
