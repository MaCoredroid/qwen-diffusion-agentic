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

## Stage 1 redo — fast serving + graded reward + dual-term (2026-07-01)

Directive: re-run Countdown RL de-risk before Stage 2, now using the fast HF serving foundation, graded reward, and
dual-term loss. Public data only (`reasoning-gym` Countdown). Protected lane unused.

Code checkpoints:
- `c0607c3` — FLARE prefix reuse cache checkpoint.
- `5cf7010` — cached graded dual-term Countdown RL pilot.
- `5d910a3` — batched exact re-score cleanup without per-token CPU syncs.

Implementation changes:
- Rollouts/eval sample through `RequestDiffusionState` cached route-I FLARE serving.
- Policy logprobs are still differentiated only by the training forward exact re-score, with the Countdown grammar mask
  re-applied before normalization.
- Reward is graded: exact success = `1.0`; parseable all-number wrong expressions get bounded inverse-distance credit
  instead of tying as strict zero.
- Dual term: `L = L_RL(constrained policy) + lambda_raw * L_raw_internalize`, where raw internalization is full-vocab CE
  over verified-correct constrained rollout tokens. No gold answer CE, no private data.

Primary run (`lambda_raw=2.0`):

```bash
.venv-fastdllm/bin/python scripts/rl_pilot_countdown.py \
  --out-dir runs/rl_pilot_countdown/stage1_cached_graded_dual_lam2_g4_step200_eval16 \
  --max-steps 200 \
  --train-size 256 \
  --eval-size 16 \
  --group-size 4 \
  --max-new-tokens 32 \
  --eval-max-new-tokens 32 \
  --eval-every 50 \
  --log-every 10 \
  --rescore-micro-batch-size 4 \
  --lambda-raw 2.0
```

Efficiency:

| run | avg s/step | nonzero-update s/step | rollout s/step | train tok/s | rollout tok/s | GPU mean | peak alloc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old strict/cache-off pilot | 4.96 | 14.68 | 1.37 | n/a | n/a | 65.9% | 14.61 GiB |
| Stage 1 `lambda_raw=0.5` | 4.34 | 4.50 | 0.92 | 9.55 | 45.16 | 69.7% | 22.37 GiB |
| Stage 1 `lambda_raw=2.0` | 4.07 | 4.17 | 0.90 | 10.08 | 45.84 | 73.0% | 22.37 GiB |

Read: average wall-clock is only moderately faster than the old headline average because graded reward makes almost every
step a real update; the fair nonzero-update comparison is much better (`14.68s -> 4.17s`). Cached rollout itself is
`1.37s -> 0.90s`. Exact re-score/update is now the dominant cost, not serving rollout.

Zero-advantage starvation:

| run | zero-advantage steps | rate |
| --- | ---: | ---: |
| old strict/cache-off pilot | 146/200 | 73.0% |
| Stage 1 `lambda_raw=0.5` | 13/200 | 6.5% |
| Stage 1 `lambda_raw=2.0` | 9/200 | 4.5% |

Graded reward fixes the starvation failure decisively.

Held-out 16-row trajectory (`lambda_raw=2.0`):

| step | RAW strict | CONSTRAINED strict | gap | RAW graded | CONSTRAINED graded |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0/16 | 2/16 | +2/16 | 0.0413 | 0.2136 |
| 50 | 1/16 | 4/16 | +3/16 | 0.1099 | 0.3509 |
| 100 | 0/16 | 3/16 | +3/16 | 0.0469 | 0.3012 |
| 150 | 0/16 | 3/16 | +3/16 | 0.0450 | 0.2900 |
| 200 | 0/16 | 4/16 | +4/16 | 0.0475 | 0.3454 |

Secondary `lambda_raw=0.5` run:

| step | RAW strict | CONSTRAINED strict | gap | RAW graded | CONSTRAINED graded |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0/16 | 2/16 | +2/16 | 0.0413 | 0.2136 |
| 50 | 1/16 | 2/16 | +1/16 | 0.1184 | 0.2391 |
| 100 | 1/16 | 3/16 | +2/16 | 0.1088 | 0.2948 |
| 150 | 0/16 | 4/16 | +4/16 | 0.0450 | 0.3472 |
| 200 | 0/16 | 4/16 | +4/16 | 0.0475 | 0.3404 |

Dual-term read:
- Raw internalization was active, not dead: `lambda_raw=2.0` used 69/200 raw-internalization steps and 1145 raw CE
  tokens; `lambda_raw=0.5` used 64/200 and 1080 tokens.
- RAW moved transiently (`0/16 -> 1/16` at step 50, and for `lambda_raw=0.5` also step 100), and RAW graded reward
  improved transiently.
- RAW strict did **not** move durably by final step (`0/16` final in both Stage 1 runs). The final raw-vs-constrained
  gap widened to `+4/16`.

Stage 1 verdict:
- PASS: fast serving is wired into rollouts/eval, and exact training-forward re-score remains the parity spine.
- PASS: graded reward fixes zero-advantage starvation (`73% -> 4.5-6.5%`).
- PARTIAL: constrained Countdown improves (`2/16 -> 4/16`).
- NOT GREEN: dual-term raw internalization is active and causes transient RAW movement, but final RAW strict is not
  durable. Do **not** proceed to Stage 2 or promote. The next red-team question is whether raw needs a verified-success
  replay buffer / stronger raw schedule, or whether this Countdown raw lane is too noisy at 16 rows for the current term.

## Raw-rollout disambiguator (2026-07-01)

Red-team directive: self-distillation did not move RAW, so test the other raw lever before calling the gap structural:
direct RAW RL rollouts. Keep constrained-policy RL and graded reward; add no-decoder raw diffusion rollouts scored by
the same graded Countdown reward, then policy-gradient the raw tokens directly through the training-forward exact
re-score. Self-distillation is disabled for the direct raw-rollout runs (`lambda_raw=0`) so the raw update is isolated.

Code checkpoint:
- `7ce4bed` — add raw rollout GRPO to Countdown pilot.

Standard 4-number Countdown raw-rollout bound:
- Run: `runs/rl_pilot_countdown/raw_rollout_g4_micro2_step200_eval16`
- Config: standard 4-number Countdown, `G_constrained=4`, `G_raw=4`, `lambda_raw=0`, `raw_rl_weight=1.0`,
  exact re-score micro-batch 2.
- Bounded at step ~190 after the user redirected to the easier disambiguator. Held-out RAW was still zero at step 150.

| step | RAW strict | CONSTRAINED strict | RAW graded | CONSTRAINED graded |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0/16 | 2/16 | 0.0413 | 0.2136 |
| 50 | 0/16 | 2/16 | 0.0817 | 0.2304 |
| 100 | 0/16 | 0/16 | 0.0600 | 0.1391 |
| 150 | 0/16 | 1/16 | 0.0444 | 0.1882 |

Raw rollout signal was active but did not transfer to held-out RAW: raw-rollout zero-advantage was `33/192 = 17.2%`,
so `159/192` raw groups had nonzero raw advantage and trained `10,456` raw tokens before the bound.

### Easier Countdown control

Config used for both easy runs:
- 3 numbers, values `1..10`, targets `3..30`, 24 generated tokens.
- Same public `reasoning-gym` generator; train/eval seeds unchanged.
- Bounded at step 150 once RAW remained pinned.

This control cleanly separates "task too hard / too few constrained-correct samples" from "raw lever fails": constrained
baseline is high at `7/16` on the held-out easy split, while RAW baseline is `0/16`.

Self-distillation on easy Countdown (`lambda_raw=2.0`, no raw rollouts):

| step | RAW strict | CONSTRAINED strict | RAW graded | CONSTRAINED graded |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0/16 | 7/16 | 0.0375 | 0.5278 |
| 50 | 0/16 | 4/16 | 0.0546 | 0.3636 |
| 100 | 0/16 | 4/16 | 0.0540 | 0.3724 |
| 150 | 0/16 | 6/16 | 0.0475 | 0.4950 |

Self-distillation had ample correct constrained samples on this easier task: raw-internalization was active on `86/165`
logged steps and replayed `1,026` raw CE tokens. RAW strict still stayed `0/16`.

Direct raw rollouts on easy Countdown (`lambda_raw=0`, `raw_rl_weight=1.0`):

| step | RAW strict | CONSTRAINED strict | RAW graded | CONSTRAINED graded |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0/16 | 7/16 | 0.0375 | 0.5278 |
| 50 | 0/16 | 6/16 | 0.0500 | 0.4736 |
| 100 | 0/16 | 4/16 | 0.0475 | 0.3802 |
| 150 | 0/16 | 4/16 | 0.0538 | 0.3896 |

Raw rollout was also genuinely active on the easy task: raw-rollout zero-advantage was `67/153 = 43.8%`, so `86/153`
raw groups had nonzero raw advantage and trained `7,689` raw tokens. RAW strict still stayed `0/16`.

Verdict:
- Easy Countdown proves the issue is not simply "Countdown was too hard for constrained learning": constrained starts
  high (`7/16`) and remains nonzero.
- Self-distillation fails to move RAW even when verified constrained outputs are abundant.
- Direct raw rollouts also fail to move RAW even when raw rollout groups frequently have nonzero graded advantages.
- Therefore, for this FLARE diffusion sampler, the RAW structural gap is behaving like a fundamental parallel-decode
  structural corruption issue, not a missing-logit-training issue. The decoder remains structurally essential for the
  deployable lane. Do **not** proceed to Tier-B as a raw-lane-green Stage 1; escalate the promotion story around the
  constrained model-only lane and treat RAW as not currently trainable by either tested raw lever.
