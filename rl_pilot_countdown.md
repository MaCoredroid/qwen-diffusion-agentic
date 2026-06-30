# Countdown RL de-risk pilot

Date: 2026-06-30.

Objective: cheaply answer whether online diffu-GRPO on Countdown is practical on the torch-path single RTX 5090, and whether it moves RAW Countdown accuracy. This is a feasibility pilot, not a promoted system.

Settled constraints from the handoff:
- Torch GDN path is default. FLA is closed for this scale.
- Rollouts and constrained eval use diffusion plus a live grammar decoder.
- Reward is reasoning-gym Countdown strict success: `score_answer(expression, row) == 1.0`.
- GRPO advantage is group-relative `(reward - group_mean) / group_std`; zero-std groups produce zero update.
- Policy-gradient logprobs are computed under the constrained distribution, with invalid grammar tokens masked out before normalization.
- Track RAW vs CONSTRAINED; PROTECTED is not used.
- Do not promote. Report to monitor for red-team.

Implementation:
- Script: `scripts/rl_pilot_countdown.py`
- Default base: `models/qwen3.5-9b-fastdllm-init`
- Default warm start: `runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000`
- Dataset: `reasoning-gym` Countdown, default toy config `4` numbers, values `1..20`, target `10..100`.
- Grammar: token-level prefix FSM over digits, `+-*/()`, exactly the given number multiset, balanced parentheses, stop only after all numbers are used.
- Update: QLoRA continuation of the warm-start LoRA adapter; one prompt per step, `G` completions per prompt.

Status:
- Implementation scaffold added.
- Smoke checks:
  - `py_compile` passed for `scripts/rl_pilot_countdown.py`.
  - One-step end-to-end smoke completed under `runs/rl_pilot_countdown/smoke` with rollout/eval/save/metrics.
  - Targeted streamed-backward smoke forced nonzero advantages over 12 constrained policy tokens: finite loss, finite grad, peak allocation ~13.65 GiB.
- Full feasibility measurement pending.
