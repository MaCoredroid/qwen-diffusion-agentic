# RL-v6 Hybrid-Serve Design

Status: design only. No training launched.

User stop point: monitor review is required before any v6 run.

## Starting Point

- Base adapter: `runs/rl_multiturn_grpo_v2/from_selected_base_g4_step300/adapter_model`.
- Serve policy to train against: `diffusion_hybrid_forced_grammar_seq_values`.
- Matched-20 hybrid-clean baseline: `47/63` exact_args, `13/20` episode exact, `63/63` valid, `63/63` exact_seq, `3.904` sec/turn, `56.83` forwards/turn.
- Never-train BFCL/API-Bank hybrid-clean baseline: `83/184` exact_args, `19/60` episode exact, `184/184` valid, `133/184` exact_seq, `2.123` sec/turn, `24.62` forwards/turn.
- Retention baseline from the accepted v2 gate: GSM8K corrected careful `0.65`.

Promotion target: move matched-20 hybrid-clean exact_args from `47/63` to `>=50/63` while holding retention `>=0.65` and preserving the zero-value-projection audit.

## Why v6 Is Different

The v2-v4 GRPO line trained rollouts decoded with diffusion careful plus live Qwen-native grammar. Hybrid-clean is now the serving default, so v6 must make train equal serve:

- Grammar-bulk only truly-forced structural tokens.
- Decode every value token sequentially from the raw model distribution.
- Do not project, force, or overwrite value spans.
- Exclude grammar-forced structural tokens from the policy loss.
- Apply policy loss only to parameter-value tokens and any non-forced free tokens that the model actually chose.

The current miss taxonomy says the remaining errors are not structure errors:

| class | matched-20 misses |
|---|---:|
| wrong_value | 9 |
| history_compounding | 5 |
| close_timing | 2 |
| missing_or_wrong_call | 0 |
| invalid_xml | 0 |

So the reward and replay should target values, timing, and episode compounding rather than XML scaffolding.

## Required Plumbing Changes

1. Add `decode_policy=hybrid_clean` to the multi-turn RL environment.
   - Reuse the same sampler semantics as `eval_flare_northstar_hybrid_clean.py`.
   - Emit per-turn schedule events, generated token ids, tokenizer offsets, and projection counters.
   - Hard-fail the rollout if any value token is projected or force-committed.

2. Add hybrid replay masks.
   - Mask out all truly-forced grammar tokens from GRPO policy loss.
   - Include parameter-value tokens.
   - Include non-forced close-timing/free tokens when they were sampled by the model.
   - Persist counters: `grammar_forced_tokens_masked`, `policy_value_tokens`, `policy_free_tokens`, `projected_value_tokens_exact`.

3. Audit every rollout and eval.
   - Use the tokenizer-offset value-projection audit, not the older broken counter.
   - Required invariants: `projected_value_tokens_exact=0`, `parallel_commit_forced_tokens_counter=0`, `zero_forward_rows=0`.

4. Keep the v4 KL safety mechanism.
   - `kl_to_base_coeff=0.05`, matching v2.
   - Rolling early stop: if last-50 mean KL exceeds `0.05`, stop immediately and gate.
   - Retention probes: N=10 every 50 steps.

## Training Pool

Use a mixed pool to avoid the v3 all-hard drift:

- About 35 percent easy anchor episodes solved by v2.
- Frontier episodes unsolved or partially solved by v2, oversampling the matched-20 taxonomy:
  - string-vs-object schema value shape,
  - numeric scale, rounding, and case exactness,
  - date/year exactness,
  - close timing on prefix-like values,
  - identifier consistency across turns.
- Fresh public-derived episodes not used by v2, v3, v4, or v5.
- Leak-check the pool against matched-20 and never-train batteries before training.

Group size stays modest at 4 unless the smoke shows high no-op rates.

## Reward Shape

Use the audited ToolRL reward as the base, then weight the terms around the observed misses:

- Exact value reward for scalar, string, list, and object values.
- Schema type-shape penalty when the gold wants a string phrase but the model emits an object or list.
- Numeric/date exactness penalty for scale shifts, rounded values, and year shifts.
- Case-sensitive string reward for values where case is semantically part of the target.
- Close-timing penalty when the model emits a strict prefix of the gold value and closes the parameter early.
- Episode-level compounding penalty when an early wrong identifier or result id is reused downstream.
- Low weight on structure because hybrid-clean already has `63/63` valid XML and `63/63` exact tool sequence on matched-20.

## Run Recipe

1. Smoke one episode.
   - Hybrid-clean rollout must produce valid rows with zero projected value tokens.
   - Replay mask sanity: `grammar_forced_tokens_masked > 0` and `policy_value_tokens > 0`.

2. Train for at most 300 steps from the v2 adapter.
   - Stop early on last-50 mean KL `>0.05`.
   - Probe retention every 50 steps with N=10.

3. Gate in order.
   - GSM8K corrected careful retention `>=0.65`.
   - Matched-20 hybrid-clean `>=50/63` exact_args.
   - Never-train hybrid-clean no material regression: `>=80/184` exact_args, `184/184` valid, and zero value projection.
   - Report paired deltas against the frozen v2 hybrid-clean baseline on both slices.

## Kill Criteria

- Reject the adapter if retention drops below `0.65`.
- Reject the adapter if matched-20 hybrid-clean remains below `50/63` after 300 steps.
- Reject the adapter if last-50 KL trips early and the gated quality does not improve over `47/63`.
- Reject the adapter if never-train hybrid-clean drops below `80/184` or validity drops below `184/184`.
- Reject any run with nonzero value projection, value force-commit, or zero-forward value rows.

No v6 training should start until this design is reviewed and the hybrid-clean rollout/replay plumbing is ready.
