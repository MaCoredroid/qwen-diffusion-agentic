# FLARE Agentic Phase 2 Native Mix-v2 Endgame Eval

Adapter: `runs/flare_agentic_phase2/two_stream_native_mix_v2_from_init_s1024_step1000`

Training summary:
- Fresh from init, native-format mix-v2, two-stream, block 1024, 1000 steps.
- Final train loss: 3.4846.
- Runtime: about 89 min.
- Training peak: 30080 MiB.

Tool-call eval settings:
- Native tool-call slices.
- Full-context corrected sampler, fresh 32-mask blocks, mask-id ban, stop fix.
- Gold stripped from generation.
- Live grammar decoder: native JSON/function-call grammar, strict live unsafe fallback = 0.

## Tool-Call Results

| Mode | Slice | Valid JSON | Exact sequence | Exact args |
| --- | --- | ---: | ---: | ---: |
| Raw native | public one-call | 2/8 | 2/8 | 1/8 |
| Raw native | public multicall | 2/12 | 2/12 | 1/12 |
| Raw native | teacher heldout | 1/8 | 1/8 | 0/8 |
| Raw native | total | 5/28 | 5/28 | 2/28 |
| Live grammar | public one-call | 8/8 | 7/8 | 5/8 |
| Live grammar | public multicall | 12/12 | 11/12 | 10/12 |
| Live grammar | teacher heldout | 8/8 | 7/8 | 4/8 |
| Live grammar | total | 28/28 | 25/28 | 19/28 |

Comparison anchors:
- B@1000 transfer raw native: 0/28 exact args.
- B@1000 transfer live grammar: 19/28 exact args, 28/28 valid JSON.
- Native mix-v2 live grammar matches transfer live grammar, but does not beat it.

Live exact-arg per-example status:

| Slice | Exact-arg pass indices |
| --- | --- |
| public one-call | 1, 3, 4, 6, 8 |
| public multicall | 2, 4, 5, 6, 7, 8, 9, 10, 11, 12 |
| teacher heldout | 5, 6, 7, 8 |

GPU utilization:
- Raw native eval monitor: mean 97.9%, p50 99%, max 100%, peak 31628 MiB.
- Live grammar eval monitor: mean 97.6%, p50 98%, max 100%, peak 31890 MiB.

## Retention

Retention eval settings:
- `scripts/eval_flare_stage1_ab_diffusion.py`
- `generation_limit=20`, GSM8K first20 and MBPP first20.
- Full-context corrected diffusion sampler, `temperature=0`, `threshold=0.9`.

| Metric | Native mix-v2 | Anchor |
| --- | ---: | ---: |
| GSM8K strict | 12/20 = 0.60 | floor 0.65; B@1000 was 0.70 |
| GSM8K flex | 12/20 = 0.60 | floor 0.65 |
| MBPP raw pass@1 | 5/20 = 0.25 | AR reference in script: 0.20 |
| Unresolved masks | 0 | 0 required |

Retention GPU utilization:
- Mean 94.1%, max 100%, peak 15190 MiB.
- Throughput: 3908 generated tokens in 491.8s = 7.95 tok/s.

Interpretation for review:
- Native training modestly lifts raw native exact args from 0/28 transfer to 2/28, but raw remains structurally weak.
- Live grammar remains the dominant lever: 2/28 raw -> 19/28 live, with valid JSON guaranteed.
- Native mix-v2 did not improve the live-constrained headline over transfer (19/28 == 19/28).
- GSM8K retention misses the 0.65 floor at 0.60, so this adapter should not be promoted without red-team approval.
