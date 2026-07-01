# FLARE profile and threshold sweep

## Scope

- Do not promote.
- Matched bf16, no 4-bit.
- Mock/easy tau2 tasks: `create_task_1_with_env_assertions`, `update_task_with_message_history`.
- Diffusion backend with HF FLARE cache, allocator `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Per-Forward Profile

Cached noisy read forward, block size 32:

| prefix tokens | mean read ms | attention | GDN | MLP | other | peak reserved GiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 55.88 | 6.0% / 3.29 ms | 71.6% / 39.25 ms | 16.4% / 8.98 ms | 6.0% / 3.27 ms | 16.971 |
| 8192 | 55.83 | 8.5% / 4.65 ms | 69.6% / 38.10 ms | 16.2% / 8.86 ms | 5.8% / 3.16 ms | 17.557 |

Clean advance forward at 8192 prefix: 57.58 ms, with GDN 70.4%, MLP 16.6%, attention 8.5%, other 4.5%.

Interpretation: the dominant per-forward cost is the block-wide GDN path, then MLP. Attention is not the main wall-time cost even at 8k context. The mean read time is essentially unchanged from 1k to 8k prompt, so this run is not memory-capacity or KV-attention-memory bound; the evidence points to compute/kernel time in block-wide GDN+MLP as the bottleneck. No hardware-counter bandwidth profile was run.

## Cost Driver

At the previous mock baseline, denoise-forwards/token was 0.989. That means the sampler is still effectively committing about one token per block-wide denoise read. With block size 32, each committed token pays for a 32-position forward. This confirms the speed gap is dominated by the one-token-per-block-wide-forward behavior.

## Threshold Pareto

| block | threshold | binary | action | db | env | tok/s | denoise/tok | cache-advance/tok |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 0.9 | 0.167 | 0.333 | 0.167 | 0.667 | 8.517 | 0.989 | 0.718 |
| 32 | 0.8 | 0.167 | 0.333 | 0.167 | 0.667 | 9.236 | 0.948 | 0.670 |
| 32 | 0.6 | 0.000 | 0.167 | 0.000 | 0.667 | 8.848 | 0.804 | 0.754 |
| 32 | 0.4 | 0.000 | 0.167 | 0.000 | 0.500 | 8.512 | 0.664 | 0.862 |
| 64 | 0.8 | 0.000 | 0.167 | 0.000 | 0.500 | 8.404 | 0.964 | 0.364 |
| 64 | 0.6 | 0.000 | 0.167 | 0.000 | 0.500 | 9.386 | 0.794 | 0.378 |

## Takeaways

- Near-free zone: threshold 0.8 at block 32. It preserves the 0.333 action score and 1/6 binary score while improving speed from 8.52 to 9.24 tok/s, an 8.4% speedup.
- Lossy zone: thresholds 0.6 and 0.4 reduce denoise-forwards/token but lose the one binary pass and cut action score to 0.167.
- Bigger block size is not a free win here. Block 64 at 0.8 and 0.6 loses binary capability and is dominated or only marginally faster than block 32 while still far below AR speed.
- The biggest lever is still committing multiple correct tokens per denoise forward. Threshold tuning alone found only a small free-speedup zone; aggressive thresholding changes outputs but does not recover the 10x speed target.

## Artifacts

- `runs/agentic_eval/flare_profile_forward_prefix1024_b32.json`
- `runs/agentic_eval/flare_profile_forward_prefix8192_b32.json`
- `runs/agentic_eval/tau2_mock_diffusion9b_threshold0.9_b32.summary.json`
- `runs/agentic_eval/tau2_mock_diffusion9b_threshold0.8_b32.summary.json`
- `runs/agentic_eval/tau2_mock_diffusion9b_threshold0.6_b32.summary.json`
- `runs/agentic_eval/tau2_mock_diffusion9b_threshold0.4_b32.summary.json`
- `runs/agentic_eval/tau2_mock_diffusion9b_threshold0.8_b64.summary.json`
- `runs/agentic_eval/tau2_mock_diffusion9b_threshold0.6_b64.summary.json`
