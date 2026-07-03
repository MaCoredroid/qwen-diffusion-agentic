# S1 Phase Close-Out

## Status

S1 is closed.

The S1 founding premise dissolved after the eval-drift rebaseline. Run-1 passes
the GSM8K anchor on the corrected legacy full-context scale:

- Run-1 corrected legacy anchor/careful: 14/20 = 0.70.
- Anchor floor: 0.65.

Therefore Run-1 remains the campaign base.

## What Failed

The extended-training result is still real, but it means erosion rather than
budget scaling:

| State | Corrected legacy GSM8K |
| --- | ---: |
| Run-1 | 14/20 = 0.70 |
| S1c-final | 9/20 = 0.45 |
| S1-final | 6/20 = 0.30 |

S1/S1c damaged the corrected careful/legacy scale. No r128 escalation or S2
budget-scaling conclusion should be based on the retired mutable-remask anchor.

## Root Cause

`scripts/run_s1_gate_eval.sh` called `scripts/measure_block_quality_curve.py`
as the hard anchor gate. That script uses the mutable-remask fixed-K sampler.

The S1 report then mislabeled the result as the validated legacy sampler. The
underlying artifact contradicted the label:

- Report label: "validated legacy sampler".
- Actual report line:
  "Sampler: mutable-remask fixed-K full-context fresh blocks, mask-token banned".

The mutable-remask sampler re-masks and resamples previously visible positions,
so it is not the GDN-state/serving continuity scale. It is now classified as an
invalid third-scale diagnostic.

## Rule

Every future gate report must record:

- the exact sampler function called,
- the script path,
- the git commit hash of that script,
- the command-line decode flags,
- the resolved model/tokenizer sentinel IDs,
- and whether the path is the approved continuity harness or an exploratory
  diagnostic.

Gate reports that omit this harness pinning should be treated as provisional.

## References

- Corrected rebaseline: `runs/eval_drift_rebaseline/report.md`
- S1 mislabeled gate report, now superseded: `runs/s1_budget_retrain_r64_qwen35_9b_gate/s1_gate_report.md`
- S1c mutable-remask anchor report, now superseded: `runs/s1c_run1_envelope_2000_qwen35_9b_anchor_probes/report.md`

