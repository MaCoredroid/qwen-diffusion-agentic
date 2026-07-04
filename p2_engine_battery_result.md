# P2 Engine Battery — FULL 63-turn matched-20 on the OPT-3-fixed engine (2026-07-04)

vLLM pin `d2fccab` (`qwen3_5-flare-modelstate`: GAP-5A windowed-probe + **OPT-1**
GPU-native sampling + **OPT-3 sync-scheduler fix**), export
`qwen3.5-9b-fastdllm-rlv2-vllm-bf16` (block/canvas 32, mamba 1024, align+APC),
RTX 5090, RAM cage. Greedy, seed 20260701, **uncapped** (`max_tokens=n_ref+16`).
**No harness patches** — the config-only pin fix alone. Full detail + artifacts:
`runs/p2_engine_battery_full/report.md`.

## Verdict

| step | verdict | one-line |
|---|---|---|
| **full 63-turn battery completes** | **PASS** | all 63 turns run, **zero stalls** (gt4/24/50 — the ex-stalls — complete). The FIRST honest, complete full-battery engine wall-clock. |
| OPT-3 async/stall fix delivered | **PASS** | partial-canvas stall gone; the `first_div=33` async-boundary divergence gone (0/11 breaks at pos 33); gt12/16/18/20 restored to byte-parity. |
| byte-parity == HF 63/63 (promotable) | **NOT MET (52/63)** | 11 turns diverge via the *separate, author-flagged* windowed-probe causal approx of the reference's windowed-**bidirectional** read (all `proj=0`, mid-block value logits). Not a regression. |
| exact_args == 47/63 | **DEVIATION +1 (48/63)** | independently scored; engine wins gt60 (correct where HF wrong), no losses. >= HF. |
| episode_exact 13/20 | **MET** | ties HF exactly. |
| valid 63/63 | **DEVIATION -1 (62/63)** | gt19's divergence cascaded to a non-stopping run (finish=length) -> 1 invalid. |
| verify_invariants / value_projection | **CLEAN** | 63/63 verify ok; 0/63 value-projection (label-free constrained lane). |
| temp=0.7 RL sanity | **PASS** | 5 rollouts bounded/valid/proj0, same-seed 2x byte-reproducible. |

## The two things the OPT-3 fix fixed, and the one it did not

- **FIXED — liveness:** every long turn (16 turns n_ref>=95, up to 259) and every
  ex-stall completes. The partial-canvas `valid_len 32->13` hang is gone.
- **FIXED — async-rollback divergence:** the pos-33 boundary-corruption signature is
  absent; gt12/gt16/gt18/gt20 are byte-parity again.
- **NOT fixed (separate, documented, follow-up):** the causal windowed-probe is an
  approximation of the reference's windowed-**bidirectional** read -> 11 turns
  diverge byte-for-byte in mid-block value regions. Quality stays >= HF in aggregate,
  but the engine is **not byte-identical** to HF. The "engine == HF by construction"
  claim is therefore **not** established on the full battery.

## Numbers (all 63 turns, greedy, uncapped)

| | ENGINE full-63 | HF full-63 | stock-AR-guided | stock-AR agg | M2/K3 |
|---|---:|---:|---:|---:|---:|
| exact_args | **48/63** | 47/63 | 51/63 | 124/247 | >=55/63 |
| episode_exact | **13/20** | 13/20 | — | — | — |
| valid | 62/63 | 63/63 | 63/63 | — | — |
| s/turn mean | **1.681** | 3.904 | 1.213 | 0.741 | **< 1.120** |
| s/turn p50 / p90 | 1.427 / 2.724 | — | — | — | — |
| s/turn worst | 5.361 (gt50, 207 tok) | — | — | — | — |
| denoise fwd/turn | 56.65 | 56.83 | 82.24 (tok) | 49.06 (tok) | — |
| tokens/forward | 1.360 | — | — | — | — |
| byte-parity | 52/63 | (self) | — | — | 63/63 |

Engine is **2.32x under HF** on the full battery (OPT-1's 2.27-2.36x generalizes),
but **1.39x slower than guided-AR** and **2.27x slower than stock-agg**.

## M2 / K3 adjudication (now possible for the first time)

- **Speed:** 1.681 s/turn > 1.120 -> **MISSED**. The bench's 1.250 was a short-turn
  subset (60 tok) excluding the 16 long turns + stalls; the honest full-63 (77 tok
  mean) is 1.681.
- **Quality:** 48/63 >= HF 47 but < guided-AR 51 and < the >=55 bar -> **MISSED**.
- **Net:** M2/K3 not met on either axis — but **adjudicable at all** for the first
  time (the whole point of the OPT-3 stall fix). Residual speed gap = per-forward
  compute shape (OPT-4); residual quality/parity gap = windowed-bidirectional forward.

## Next hotspot — OPT-4 (incremental KV+GDN 1-token decode), MEASURED

`torch.profiler` kernel attribution on 3 turns (kernel-level, double-counting removed):
**gemm ~62%** (MLP+proj+lm_head, computed over CL=32 query rows to read 1 probe logit)
> elementwise/copy ~21% > full-attn 6-9% > **GDN chunk path ~5%** > sampling **0.5%**
(OPT-1 confirmed). GDN dispatches the **prefill `chunk_gated_delta_rule` kernels**
(`chunk_gated_delta_rule_fwd_kernel_h`, `chunk_fwd_kernel_o`, `chunk_scaled_dot_kkt`,
`recompute_w_u`, `_causal_conv1d`) — **`fused_recurrent` is absent**. Per-forward
~18 ms GPU + ~11 ms host (eager launch overhead; `enforce_eager=True`, no CUDA graph).
**OPT-4** = incremental KV+GDN state so the probe is a genuine 1-token decode ->
`fused_recurrent` + FULL graph replay. Caveat: at batch=1 gemm is weight-bandwidth-
bound, so the win is the GDN chunk->recurrent switch + eager->graph launch removal,
not a 32x gemm cut.

## Frontier (evidence-ranked)

1. **OPT-3 follow-up — byte-EXACT windowed-BIDIRECTIONAL forward** (+ true per-request
   variable draft width): the remaining 11 divergences + the batched-rollout path.
   Restores universal parity; the forward-compute cut.
2. **OPT-4 — incremental KV/GDN 1-token decode** (`fused_recurrent` + FULL graph):
   the residual GPU-shape gap 1.681 -> guided-AR 1.213. Measured above.
3. **OPT-1 done + verified** (2.32x on full battery). **OPT-5 do NOT do** (0.5%).

## Artifacts (`runs/p2_engine_battery_full/`)

`matched20_turns.jsonl` (63 turns) · `aggregate.json` (stats + per-turn dist + 11
breaks) · `matched20_temp07{a,b}.jsonl` (RL rollouts x2) · `profile_opt4.py` +
`opt4_breakdown.json` (OPT-4 kernel breakdown) · `report.md` (full writeup).
