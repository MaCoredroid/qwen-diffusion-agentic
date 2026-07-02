# RL Multi-Turn Env Smoke

No training was run. This smoke exercises public episode loading, eval-battery filtering, careful+live-grammar rollout, audited reward scoring, and GRPO advantage calculation.

- Episode: `85f6c398-69c7-4df2-aed1-29d614a93a26`
- Turns: 3
- Exact args: 2/3
- Mean reward: 0.922
- Audit: mode `no_projection_events`, projected_value_tokens_exact=0, verified=1
- Live grammar token visits: 178; parallel-commit forwards: 0.
- Leak filter: kept 12/12 rows; rejected 0.

Reward design follows the taxonomy: format/schema and wrong-value terms dominate direct misses, with exact-args retained as the gate and episode-level accounting available for generated-history compounding.
