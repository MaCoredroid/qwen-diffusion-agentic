# FLARE Scale-Up Eval

## Corrected Headline

Strict clean-waves scale-up was capped at N=20 after the first rows made the result decisive. The capped strict row is `0/20` exact-args with generated-token audit pass.

| Backend | exact_args | episode exact | exact_seq | valid_xml | schema_ok | sec/turn | forwards/turn | audit | note |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| AR-guided | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0.000 | n/a | n/a | not run on this native single-turn scale-up slice |
| Diffusion careful baseline | 20/58 | 20/58 | 36/58 | 48/58 | 34/58 | 15.038 | 228.207 | n/a | full N=58 |
| Diffusion clean waves strict | 0/20 | 0/20 | 5/20 | 15/20 | 1/20 | 16.545 | 244.950 | projected_token_records_x_generated_token_offsets / projected_value=0 | N=20 cap; first 20 rows only |
| Diffusion waves legacy | 30/58 | 30/58 | 50/58 | 52/58 | 44/58 | 8.363 | 109.569 | n/a | CONTAMINATED; old projected-value row, full N=58 |

## Files

- Strict capped rows: `runs/flare_scaleup_eval/percall_waves_tau095_structural_only/scaleup_native_20_capped.jsonl`
- Strict capped audit: `runs/flare_scaleup_eval/percall_waves_tau095_structural_only/projection_value_audit_20_capped.json`
- Capped summary: `runs/flare_scaleup_eval/percall_waves_tau095_structural_only/scaleup_native_20_capped.summary.json`
