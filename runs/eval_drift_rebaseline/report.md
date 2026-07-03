# Eval Drift Rebaseline

## Verdict

The apparent anchor drift was a harness-scale mixup, not a weight-scale failure.

The correct continuity harness is the validated legacy full-context/fresh-block/mask-ban sampler in
`scripts/eval_flare_stage1_ab_diffusion.py`. That is also the careful retention gate for this GSM8K
slice, so the corrected anchor and careful table are one scale.

The mutable-remask fixed-K sampler in `scripts/measure_block_quality_curve.py` is not the continuity
anchor. It re-masks and resamples previously visible positions, which breaks the GDN-state discipline
needed for the serving/cache interpretation. Its numbers should be treated as invalid third-scale
diagnostics, not gate evidence.

## Diff Pin

Reference commit: `43f8366 Measure block quality curve anchor gate`.

Findings:

- `scripts/measure_block_quality_curve.py` has no code diff from `43f8366` to this rebaseline.
- `scripts/eval_flare_stage1_ab_diffusion.py` changed only the tokenizer load fallback after `43f8366`:
  adapters with `tokenizer.json` still load their adapter tokenizer; tokenizer-less checkpoint dirs fall
  back to the base tokenizer. This is not the scale shift.
- `scripts/run_s1_gate_eval.sh` was added at `7f347e0` and invokes
  `scripts/measure_block_quality_curve.py` as the first hard anchor gate.
- `runs/s1_budget_retrain_r64_qwen35_9b_gate/s1_gate_report.md` incorrectly describes that gate as
  "validated legacy sampler", but the actual artifact
  `runs/s1_budget_retrain_r64_qwen35_9b_gate/block_quality_curve/s1_gsm8k_k_curve.report.md`
  states: "Sampler: mutable-remask fixed-K full-context fresh blocks, mask-token banned".

Therefore the S1 anchor failure report mixed labels: the command path was mutable-remask, while the
human-readable report called it legacy.

## Corrected Table

Config for all rows:

- GSM8K first20, Phase-A 5-shot
- `block_size=32`, `small_block_size=32`
- `threshold=0.9`, `temperature=0.0`, `top_p=0.95`
- `max_new_tokens=256`
- `full_context_generation=True`, `fresh_generation_blocks=True`
- `FASTDLLM_FLARE_GDN_ROUTE=route_i`, `FASTDLLM_GDN_KERNEL=torch`

| Model state | Adapter | Corrected legacy anchor / careful strict | Flex | Seconds | Generated tokens | Unresolved masks |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| init-base | none | 0/20 = 0.00 | 0/20 = 0.00 | 708.209 | 5120 | 0 |
| B@1000 | `runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000` | 11/20 = 0.55 | 11/20 = 0.55 | 370.977 | 2670 | 0 |
| Run-1 | `runs/flare_redesign_run1_copy_grounded_qwen35_9b` | 14/20 = 0.70 | 14/20 = 0.70 | 412.860 | 2845 | 0 |
| S1-final | `runs/s1_budget_retrain_r64_qwen35_9b` | 6/20 = 0.30 | 6/20 = 0.30 | 267.820 | 2178 | 0 |
| S1c-final | `runs/s1c_run1_envelope_2000_qwen35_9b` | 9/20 = 0.45 | 9/20 = 0.45 | 316.654 | 2550 | 0 |

Artifacts:

- `runs/eval_drift_rebaseline/careful_init_b1000_run1/summary.json`
- `runs/eval_drift_rebaseline/careful_s1final_s1cfinal/summary.json`
- `runs/eval_drift_rebaseline/logs/careful_init_b1000_run1.log`
- `runs/eval_drift_rebaseline/logs/careful_s1final_s1cfinal.log`

## Stability Check

B@1000 reproduces the `43f8366` legacy current-check result exactly:

- `43f8366` legacy current-check: 11/20 strict, 11/20 flex.
- Current rebaseline: 11/20 strict, 11/20 flex.

Run-1 remains on the high careful scale, though one row lower than the saved red-team artifact:

- Saved Run-1 validated fullctx/fresh/mask-ban artifact: 15/20 strict, 15/20 flex.
- Current rebaseline: 14/20 strict, 14/20 flex.

S1-final reproduces its prior careful failure exactly:

- Prior S1-final careful artifact: 6/20 strict, 6/20 flex.
- Current rebaseline: 6/20 strict, 6/20 flex.

S1c-final careful is 9/20. So the S1/S1c damage is real on the corrected continuity scale.

## Historical Number Map

Continuity / valid legacy scale:

- B@1000 legacy full-context: current reproducible point is 11/20 = 0.55.
- Older saved B@1000 strict artifact `13/20 = 0.65` remains historical context, but the
  reproducible continuity point is 11/20 under both `43f8366` and this rebaseline.
- Run-1 validated fullctx/fresh/mask-ban: old 15/20, current 14/20.
- S1-final careful: old/current 6/20.
- S1c-final careful: current 9/20.

Invalid mutable-remask third scale:

- `runs/block_quality_curve_b1000/anchor_b32_k32_gsm20.summary.json`: B@1000 5/20 = 0.25.
- `runs/s1_budget_retrain_r64_qwen35_9b_gate/block_quality_curve/*`: S1 ckpt1000 2/20,
  ckpt1500 5/20, final 5/20.
- `runs/s1c_run1_envelope_2000_qwen35_9b_anchor_probes/*`: Run-1 3/20, S1c ckpt400 5/20,
  ckpt1000 6/20, ckpt2000 6/20.

These mutable-remask rows should not drive gate verdicts or budget-scaling conclusions.

