# S1 Gate Summary

Superseded anchor note: the "validated legacy sampler" label below was wrong. The actual
first-stage command in `scripts/run_s1_gate_eval.sh` called
`scripts/measure_block_quality_curve.py`, whose report identifies the sampler as
mutable-remask fixed-K. That mutable-remask scale is not the legacy continuity scale.
Use `runs/eval_drift_rebaseline/report.md` for the corrected legacy full-context
rebaseline.

Status: FAIL.

The hard block-mode anchor failed, so the gate stopped before the block speed curve, GSM8K careful decode gate, and frozen native copy-arg battery.

## Anchor Trend

| Adapter | Correct | Examples | Strict accuracy |
|---|---:|---:|---:|
| checkpoint-1000 | 2 | 20 | 0.10 |
| checkpoint-1500 | 5 | 20 | 0.25 |
| final checkpoint-2000 | 5 | 20 | 0.25 |

Gate threshold: GSM8K strict accuracy >= 0.65 at B=32, K=32 using the validated legacy sampler.

## Evidence

- Final anchor summary: `block_quality_curve/s1_gsm8k_k_curve.summary.json`
- Checkpoint-1000 trend summary: `block_quality_curve_ckpt1000/s1_ckpt1000_gsm8k_anchor.summary.json`
- Checkpoint-1500 trend summary: `block_quality_curve_ckpt1500/s1_ckpt1500_gsm8k_anchor.summary.json`
- Gate log: `logs/s1_gate_eval.log`
- Trend logs: `logs/s1_ckpt1000_anchor.log`, `logs/s1_ckpt1500_anchor.log`
