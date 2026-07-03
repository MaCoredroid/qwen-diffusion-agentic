# S2.0 Close-Out

Verdict: **STOP before S2.1**.

The approved teacher was the v2 adapter because v4 passed retention but regressed on matched-20 (`37/63` vs v2 `44/63`). S2.0 loaded that adapter as an active frozen PEFT adapter named `s2_teacher`; `disable_adapter` was not used.

## Teacher Value-Span Gate

| Probe | Prefix reveal | Context cap | Top-1 | Span exact | Cropped rows | Verdict |
|---|---:|---:|---:|---:|---:|---|
| `v2_heldout_value_top1_fullseq` | 0.50 | 4096 | 164/688 = 0.238 | 28/84 = 0.333 | 0/12 | FAIL |
| `v2_heldout_value_top1_prefix90` | 0.90 | 4096 | 100/180 = 0.556 | 65/84 = 0.774 | 0/12 | FAIL |

The initial 1024-token run also failed (`178/688 = 0.259`) but cropped 11/12 rows, so it is diagnostic only. The full-context 4096-token runs are the stop evidence.

## Nested-View Validators

- `validate_flare_two_stream_forward.py`: PASS, including the new nested-view GDN state-discipline check.
- `validate_gdn_state_snapshot.py`: PASS on GDN layers `0,2`, including same-seed/read-only/repeatability/noisy-state sensitivity checks.

## Decision

Per `s1s2_speed_training_recipe.md`, teacher value-span top-1 below about 60 percent means the DSCD KL target is not trustworthy on the cliff mode. S2.1 corpus generation and S2 Round 1 training were not started.
