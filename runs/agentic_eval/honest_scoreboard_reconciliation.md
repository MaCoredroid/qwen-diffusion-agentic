# Honest Scoreboard Reconciliation

All strict clean-waves rows below use the 19a50b7 generated-token offset audit. Legacy waves rows are retained only as contaminated references.

## Matched-20 Agentic

| Backend | exact_args | episode exact | exact_seq | valid_xml | schema_ok | sec/turn | forwards/turn | audit | note |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| AR-guided | 50/63 | 13/20 | 63/63 | 63/63 | 59/63 | 1.120 | n/a | n/a |  |
| Diffusion careful | 34/63 | 8/20 | 55/63 | 59/63 | 47/63 | 6.049 | 95.254 | no_projection_events / projected_value=0 |  |
| Diffusion clean waves strict | 3/63 | 0/20 | 36/63 | 46/63 | 13/63 | 6.546 | 94.317 | projected_token_records_x_generated_token_offsets / projected_value=0 |  |

## Never-Train BFCL/API-Bank

| Backend | exact_args | episode exact | exact_seq | valid_xml | schema_ok | sec/turn | forwards/turn | audit | note |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| AR-guided | 77/184 | 19/60 | 126/184 | 184/184 | 184/184 | 0.591 | n/a | n/a |  |
| Diffusion careful | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0.000 | n/a | n/a | not run; monitor prioritized matched-20 careful |
| Diffusion clean waves strict | 9/184 | 0/60 | 97/184 | 142/184 | 91/184 | 2.695 | 33.353 | projected_token_records_x_generated_token_offsets / projected_value=0 |  |

## Scale-Up Native

| Backend | exact_args | episode exact | exact_seq | valid_xml | schema_ok | sec/turn | forwards/turn | audit | note |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| AR-guided | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0.000 | n/a | n/a | not run on this slice |
| Diffusion careful | 20/58 | 20/58 | 36/58 | 48/58 | 34/58 | 15.038 | 228.207 | n/a | full N=58 |
| Diffusion clean waves strict | 0/20 | 0/20 | 5/20 | 15/20 | 1/20 | 16.545 | 244.950 | projected_token_records_x_generated_token_offsets / projected_value=0 | N=20 cap |

## Verdict

- The old north-star diffusion quality claim is invalid as a capability claim.
- Never-train reconciliation: old `88/184` falls to strict audit-clean `9/184`; it was not the same mode.
- Matched-20 capability baseline: diffusion-careful is `34/63`, while strict waves are `3/63`; clean waves are worse than the no-waves model.
- Scale-up strict clean waves were capped at N=20 and are `0/20`; this is sufficient to stop spending compute on the tail.
- Root cause from taxonomy: clean waves are audit-clean but the wave/order schedule corrupts values and generated-history alignment. The issue is not residual projected VALUE tokens after 19a50b7.
