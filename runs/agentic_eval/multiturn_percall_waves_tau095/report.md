# FLARE Multiturn Per-Call Waves Eval

Slice: 12 generated-history episodes, 38 turns, from `/home/mark/qwen_diffusion/data/toolcall_eval_native/flare_scaleup_native_58.jsonl`.
Prompting: previous sampled assistant tool call plus synthetic `<tool_response>` is appended before the next turn.
Stop: `</tool_call>` added as a stop token so each turn measures one tool call.

| Condition | exact_args | episode exact | valid_json | model forwards/turn | gen tok/turn | blended TPF | sec/turn | prefix hits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline careful | 20/38 | 1/12 | 38/38 | 71.263 | 73.737 | 1.035 | 5.079 | 25/26 |
| Per-call waves tau 0.95 | 35/38 | 10/12 | 38/38 | 6.079 | 74.132 | 12.195 | 1.347 | 26/26 |

## Headline

- End-to-end turn/episode wall speedup: 3.769x
- Paired exact-args delta (per-call - baseline): 15 / 38 turns
- Per-call only exact args: 15; baseline only: 0
- Episode-level exact-args delta: 9 / 12 episodes
- Per-call only exact episodes: 9; baseline only: 0
- TPF accounting: numerator is all generated visible tokens; denominator is model denoise forwards. Grammar-projected scaffold tokens count in generated tokens but consume no model forward.
- Baseline raw TPF: 2802 generated tokens / 2708 denoise forwards = 1.035
- Per-call raw TPF: 2817 generated tokens / 231 denoise forwards = 12.195
- Baseline timed split: prefill/cache-reset 0.159, decode 0.841
- Per-call timed split: prefill/cache-reset 0.547, decode 0.453
- Schedule build overhead: baseline 0.023s, per-call 0.021s
- Grammar projection overhead in per-call: 0.216s
- Per-call value force counters: {"forced_schedule_token_visits": 0.0, "parallel_commit_forced_tokens": 0.0, "tool_value_candidate_force_token_visits": 0.0, "wave1_forced_tokens": 2553.0, "wave1_projected_tokens": 2553.0, "wave1_value_tokens": 0.0, "wave2_forced_tokens": 0.0}

Full JSON summary: `/home/mark/qwen_diffusion/runs/agentic_eval/multiturn_percall_waves_tau095/summary.json`
Per-turn rows: `/home/mark/qwen_diffusion/runs/agentic_eval/multiturn_percall_waves_tau095/turns.jsonl`
