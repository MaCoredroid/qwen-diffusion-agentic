# FLARE Scale-Up Native Eval

Slice: 58 leak-checked native records.
Sources: {'/home/mark/qwen_diffusion/runs/flare_scaleup_eval/heldout_seed_multicall_run1_exactuser_clean.jsonl': 58}
Leak check: exact_instance=0, user=0 against all Run-1 train records; same-tool/all-value=0 against near-leak scope `copy_synth` (2048 records).

| Condition | exact_args | exact_seq | valid_json | blended TPF | sec/rec | total wall |
|---|---:|---:|---:|---:|---:|---:|
| Baseline careful | 20/58 | 36/58 | 48/58 | 1.018 | 15.039 | 872.263s |
| Per-call waves tau 0.95 | 30/58 | 50/58 | 52/58 | 1.958 | 8.363 | 485.04s |

## Headline

- exact_args delta (per-call - baseline): 10 / 58
- honest wall speedup: 1.798x
- per-call misses: 28/58
- per-call miss value split: {'copy': 185, 'derived': 78}
- value force counters: {'forced_schedule_token_visits': 0, 'tool_value_candidate_force_token_visits': 0, 'wave1_value_tokens': 0, 'wave2_forced_tokens': 0, 'parallel_commit_forced_tokens': 0, 'wave1_projected_tokens': 5999, 'wave1_forced_tokens': 5999}
- full per-call miss list: `/home/mark/qwen_diffusion/runs/flare_scaleup_eval/scaleup_eval_report.json`

## B-Only Misses

- idx 17 id `heldout_seed_run1clean_0017`: valid=True seq=True copy/derived={'copy': 9} names=['get_campaign_metrics', 'calculate_campaign_roi']
