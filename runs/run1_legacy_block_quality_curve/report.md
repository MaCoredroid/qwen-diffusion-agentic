# Run-1 Corrected-Legacy Block-Quality Curve

## Headline

Run-1 remains the campaign base on the corrected legacy continuity scale. On the GSM8K first20 slice, legacy chunk widths K=32 and K=16 hold the 0.65 anchor at 14/20, K=8 improves to 15/20, and K=4 falls to 11/20. This is the cheap S2 baseline; it is not an S2 go/no-go.

Important interpretation: `K` here is `small_block_size` in `scripts/eval_flare_stage1_ab_diffusion.py::full_context_sample_one`. It is the validated legacy sampler continuity scale, not the mutable-remask fixed-denoise diagnostic from `scripts/measure_block_quality_curve.py`.

## Results

| K (`small_block_size`) | strict | flex | elapsed s | tokens/s | denoise forwards | generated tpf | unresolved masks |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 14/20 (0.70) | 14/20 (0.70) | 410.2 | 6.94 | 2793 | 1.019 | 0 |
| 16 | 14/20 (0.70) | 15/20 (0.75) | 394.4 | 6.98 | 2701 | 1.020 | 0 |
| 8 | 15/20 (0.75) | 15/20 (0.75) | 425.7 | 6.92 | 2886 | 1.020 | 0 |
| 4 | 11/20 (0.55) | 11/20 (0.55) | 399.2 | 7.01 | 2732 | 1.024 | 0 |

Commit accounting from sampler metrics:

| K | natural commits | forced commits | selected mask tokens | mean denoise forwards/example |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 2105 | 744 | 0 | 139.65 |
| 16 | 2038 | 720 | 0 | 135.05 |
| 8 | 2207 | 739 | 0 | 144.30 |
| 4 | 2108 | 689 | 0 | 136.60 |

## Harness Pin

- Sampler: `scripts/eval_flare_stage1_ab_diffusion.py::full_context_sample_one` at line 582; call site line 693.
- Git commit called: `48d7b3dca1f9122be6ecce87454b0d308ae67b68`.
- Script SHA-256: `eaa78d7a9abfb32b7ab73c7753cf87026741e372ef13a1c0f8e44ead79b5e503`.
- Base model: `models/qwen3.5-9b-fastdllm-init`.
- Adapter: `runs/flare_redesign_run1_copy_grounded_qwen35_9b`.
- Slice: `data/phaseA_retention/gsm8k_main_test_first20.jsonl`; few-shot path `data/phaseA_retention/gsm8k_main_train_first5.jsonl`.
- Decode flags: `--block-size 32 --small-block-size {32,16,8,4} --max-new-tokens 256 --threshold 0.9 --temperature 0.0 --top-p 0.95 --generation-limit 20 --generation-batch-size 1`.
- Sentinels: `mask_id=248077`, `stop_token_ids=[248046, 248044, 248045]`.
- Env: `FASTDLLM_FLARE_GDN_ROUTE=route_i`, `FASTDLLM_GDN_KERNEL=torch`, `FASTDLLM_BATCH_FLARE_NOISY_GDN=1`, `FASTDLLM_OPTIMIZE_FLARE_CLEAN_GDN=1`.

## Readout

The corrected legacy baseline supports K=32/16/8 as quality-safe operating points on this 20-example slice, with K=8 the best observed point. K=4 is below the anchor and should not be used as a promoted baseline. The measured generated-token-per-denoise-forward remains approximately 1.02 for all four rows, so this legacy curve is a quality curve, not evidence of a new speed lane.
