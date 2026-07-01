# HF serving foundation Stage 0 result (2026-07-01)

Scope: prerequisite gate before any `RequestDiffusionState` cache build. Adapter:
`runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000` on
`models/qwen3.5-9b-fastdllm-init`. Cache OFF for both serving paths.

## 0a. FLARE-vs-causal serving A/B

Instrumentation: commit `5f5c900` adds `--denoise-logit-mode` to
`scripts/eval_fastdllm_toolcall_cases.py`.

- `causal_shift`: existing full-context path, `model(input_ids=x_t, use_cache=False)`,
  right-shifted logits.
- `flare_no_shift`: route_i two-stream noisy forward, cache-off, no right shift.
- `flare_shift` was added after this falsifier as the corrected route_i two-stream mode:
  noisy FLARE logits with the same +1/right shift as training.
- Both arms use the same Qwen-native live grammar decoder, gold stripped from generation,
  `fresh_generation_blocks`, `block_size=32`, `small_block_size=32`, `max_new_tokens=192`,
  greedy temp 0.

Cases: banked native one-call rows 3-8 from
`data/toolcall_eval_native/public_onecall_qwen_native_smoke.jsonl`, materialized at
`runs/stage0_serving_ab/public_onecall_short6.input.jsonl`. This short bank was used because
the long thermostat row made cache-off FLARE impractically slow; all causal outputs on this
bank are under 103 generated tokens, so the 192-token cap is not truncating the causal arm.

Artifacts:

- Causal: `runs/stage0_serving_ab/causal_shift_onecall_short6_m192/public_onecall_short6.summary.json`
- FLARE: `runs/stage0_serving_ab/flare_no_shift_onecall_short6_m192/public_onecall_short6.summary.json`

Result:

| mode | valid JSON | exact sequence | exact args | elapsed | generated tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| causal_shift | 6/6 | 6/6 | 4/6 | 52.6s | 7.93 |
| flare_no_shift | 0/6 | 0/6 | 0/6 | 1313.4s | 0.35 |

Per-row: causal exact args = `[1, 1, 0, 1, 0, 1]`; FLARE no-shift exact args =
`[0, 0, 0, 0, 0, 0]`. FLARE also produced zero valid tool-call JSON rows despite zero
unsafe grammar fallbacks.

Verdict: **FAIL for no-shift.** FLARE no-shift serving is not >= causal-shift serving on
task score. This does **not** mean the checkpoint is unusable for FLARE serving; it means
the original serving architecture head-alignment claim was wrong.

## 0a resolution: true train-matched head alignment

The true train-matched serving alignment is **route_i FLARE noisy forward plus the existing
+1/right shift**.

Code evidence in `models/qwen3.5-9b-fastdllm-init/modeling.py`:

- `_flare_two_stream_training_forward` computes `noisy_logits = self.lm_head(noisy_hidden)`.
- `_compute_flare_losses` sets `targets = labels[:, 1:]`.
- Diffusion masks are also shifted: `diff_mask0 = mask_view0[:, 1:]` and
  `diff_mask1 = mask_view1[:, 1:]`.
- The diffusion CE consumes `noisy_logits[:batch_size, :-1]` and
  `noisy_logits[batch_size:, :-1]` against those shifted labels.

So noisy hidden/logit position `i` is trained to predict token `i+1`, exactly like the clean
AR stream. The within-block bidirectional mask changes the context available to the noisy
stream; it does not change the head's target index. The correct cached serving method must
therefore produce route_i FLARE noisy logits and then apply the same shift before sampling or
logprob scoring. The no-shift branch is retained only as the Stage-0 falsifier.

## 0b. Throughput projection

Measured pilot input: `runs/rl_pilot_countdown/pilot_g4_step200_eval16`.

- B/G in pilot: one prompt x `G=4` completions, lockstep.
- Avg rollout time: `1.368s/step`.
- Avg denoise forwards: `11.79/step`.
- Aggregate denoise-forward rate at B=4: `8.62 forwards/s`.
- Avg policy tokens: `43.74/step` total across G=4, or `10.94/completion`.
- Aggregate rollout token rate at B=4: `31.96 policy tokens/s`.
- Mean GPU util: `65.94%`; peak `nvidia-smi` memory: `22369 MiB`.

Memory projection for the HF lockstep cache:

- Per-request state from the design note: about `100-150 MB/req`.
- B=16 cache cost: `1.6-2.4 GB`.
- B=32 cache cost: `3.2-4.8 GB`.
- With the measured pilot peak around `22.4 GB`, B=16 is safe; B=32 is plausible but
  context-length/fragmentation dependent. Practical memory saturation is around B=32-48,
  not B=4.

Capacity projection:

- Optimistic no-latency-growth scaling from B=4 gives about `128 tok/s` at B=16 and
  `256 tok/s` at B=32.
- Conservative torch-ceiling haircut, because the measured path already averages about 66%
  GPU util, is about `85-128 tok/s` at B=16 and `128-170 tok/s` at B=32.
- Expected first saturation: compute/torch orchestration by B~=16-32; memory after that,
  especially at longer prefixes.

Needed B and throughput:

- RL rollouts should run B = `prompts x G`, with B=16 as the minimum useful target
  (`4 prompts x G=4` or `2 prompts x G=8`) and B=32 preferred if memory holds.
- A 1k-step RL pilot at `4 prompts x G=8 x 128 rollout tokens` is about `4.1M` rollout
  tokens. Overnight in 12h needs about `95 tok/s`; one day needs about `47 tok/s`.
  The B=16/32 HF-lockstep projection clears this.
- A 10k-step run at the same scale is about `41M` rollout tokens: roughly 2.8-5.6 days
  at the conservative projected capacity. Slow, but not a reason to switch the foundation.
- Agentic eval is less naturally lockstep. A 2.5M-token full mixed eval needs only
  about `30 tok/s` to finish in one day, but heterogeneous tasks/turns can collapse active
  batch size. This is the deferred continuous-batching case, not the RL foundation case.

Throughput-only verdict after the 0a alignment resolution: **HF-LOCKSTEP-SUFFICES** for the
RL serving foundation if the cached shifted-FLARE forward measures near this projection.
Concrete trigger to layer SGLang generation on the HF spine:
measured cached HF lockstep at B=16 falls below `80 policy tok/s`, B=32 cannot fit due
to memory, adaptive per-sequence commit breaks lockstep, or the agentic eval harness shows
p50 active batch `<8` with projected wall-clock `>24h` for the target eval.

## Overall gate

Do not build the cache until this corrected alignment is carried into the foundation:
`RequestDiffusionState` architecture stays the same, but T2 must compare shifted FLARE
serving logits/logprobs against the shifted training-loss alignment.
