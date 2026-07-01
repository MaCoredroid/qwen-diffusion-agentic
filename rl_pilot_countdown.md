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
- Feasibility measurement complete.

## 200-step pilot result

Command:

```bash
.venv-fastdllm/bin/python scripts/rl_pilot_countdown.py \
  --out-dir runs/rl_pilot_countdown/pilot_g4_step200_eval16 \
  --max-steps 200 \
  --train-size 256 \
  --eval-size 16 \
  --group-size 4 \
  --max-new-tokens 32 \
  --eval-max-new-tokens 32 \
  --eval-every 50 \
  --log-every 10
```

Artifacts:
- `runs/rl_pilot_countdown/pilot_g4_step200_eval16/summary.json`
- `runs/rl_pilot_countdown/pilot_g4_step200_eval16/metrics.jsonl`
- `runs/rl_pilot_countdown/pilot_g4_step200_eval16/adapter_model`

Throughput and memory:
- 200 train steps, `G=4`.
- Total wall-clock including evals: `1284.8s` (~21.4 min).
- Train step average: `4.96s/step`.
- Rollout average: `1.37s/step`.
- Update average over all steps: `3.59s/step`.
- Zero-advantage steps: `146/200`.
- Nonzero-update steps: `54/200`; nonzero-update average step time `14.68s`, update time `13.30s`.
- Peak PyTorch allocated: `14.61 GiB`; peak reserved `15.42 GiB`.
- Peak `nvidia-smi` memory: `22369 MiB`.
- Mean GPU util: `65.94%`; max `100%`.

16-row held-out eval during the run:

| step | RAW strict | CONSTRAINED strict | gap |
| ---: | ---: | ---: | ---: |
| 0 | 0/16 | 0/16 | 0 |
| 50 | 0/16 | 0/16 | 0 |
| 100 | 0/16 | 2/16 | +2 |
| 150 | 1/16 | 2/16 | +1 |
| 200 | 0/16 | 2/16 | +2 |

The step-150 RAW hit was transient and disappeared by step 200.

## 32-row confirmation eval

Same eval seed/config, eval-only:

| adapter | RAW strict | CONSTRAINED strict | gap |
| --- | ---: | ---: | ---: |
| B@1000 baseline | 0/32 | 1/32 | +1 |
| Countdown RL step-200 | 0/32 | 5/32 | +5 |

Artifacts:
- Baseline: `runs/rl_pilot_countdown/eval32_base_B1000/summary.json`
- Step-200 adapter: `runs/rl_pilot_countdown/eval32_pilot_step200/summary.json`

## Gate read

Systems feasibility: PASS. Online constrained diffu-GRPO fits comfortably in 32GB on the torch path. It is not memory-bound. Pace is tolerable for a pilot, but nonzero reward updates are slow because the implementation backprops constrained token likelihoods one at a time to stay under memory.

Learning signal: PARTIAL. The constrained policy moved on Countdown (`1/32 -> 5/32` strict), so reward/rollout/update plumbing is not inert. RAW did not move durably (`0/32 -> 0/32`; transient `1/16` at step 150 only). The RAW-vs-constrained gap widened, exactly the decoder-dependence risk called out in the RL design doc.

Decision: do not promote and do not claim the full gate is green. The pilot says online RL is practical, but constrained-only RL is insufficient for the promotable RAW lane. The next decision belongs to the monitor: either proceed to the fuller build specifically because it adds RAW internalization/self-distillation, or pivot lighter if the widened raw gap is unacceptable.
