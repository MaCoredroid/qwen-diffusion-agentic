# P2 Engine Battery — FULL 63-turn matched-20 on the OPT-3-fixed engine (2026-07-04)

vLLM pin **`d2fccab`** (`qwen3_5-flare-modelstate`: GAP-5A windowed-probe forward +
OPT-1 GPU-native sampling + **OPT-3 sync-scheduler fix**), real export
`models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16` (block/canvas 32, mamba 1024, align+APC),
RTX 5090, one heavy process in the `systemd-run … MemoryMax=22G` cage. Greedy,
temp 0, seed 20260701, **uncapped** (`max_tokens = n_ref + 16`, no hard cap). No
harness patches — the pin fix alone (the config-only change auto-triggers via
`diffusion_config`). Two boots (ep0-9, ep10-19), per-turn incremental JSONL.

## Headline — the OPT-3 fix delivered its two targets; the battery now RUNS

**All 63 turns complete. Zero stalls.** The partial-canvas forward stall that made
19/63 turns uncompletable in the prior bench is **gone** — including the exact
reproducers (gt4/ep1-t0 110-tok now 110/110 byte-parity 3.4s; gt24/ep7-t2 completes;
gt50/ep15-t3 259-ref runs to 207 tok in 5.36s). This is the **first honest,
complete, full-battery engine wall-clock** — the precondition the bench could not
meet. **The async-rollback boundary divergence is also fixed**: the recurring
`first_div=33` signature is **absent** (0/11 breaks at pos 33), and the 4
formerly-async-divergent reproducers gt12/gt16/gt18/gt20 are now **byte-parity**.

**But byte-parity is not universal (52/63)** — a *separate*, author-flagged residual
(the causal windowed-probe approximation of the reference's windowed-**bidirectional**
read) diverges on 11 turns. Because these divergences are a different decode from
HF, the "engine == HF by byte-construction" claim is **not** established. The
independently-scored quality is nonetheless **at parity-or-better**: exact_args
**48/63 (HF 47)**, episode-exact **13/20 (= HF)**, valid **62/63**, verify_invariants
**63/63**, value_projection **0/63** (clean, label-free).

## Gate results vs the task's required checks

| task check | required | measured | verdict |
|---|---|---|---|
| (1) byte-parity per turn | **63/63** (promotable) | **52/63** | **NOT met** — 11 windowed-probe divergences (separate known issue) |
| (2) exact_args | == 47/63 | **48/63** | deviation **+1** (engine wins gt60 where HF misses); diagnosed below |
| (2) episode_exact | 13/20 | **13/20** | **met** |
| (2) valid | 63/63 | **62/63** | deviation **-1** (gt19 diverged -> over-generated -> invalid); diagnosed |
| verify_invariants | — | **63/63** | clean |
| value_projection_events | 0 | **0/63** | clean (no grammar-projection contamination) |

Per the task's "any deviation => parity broke, stop and diagnose" rule: **byte-parity
is 52/63, so the by-construction chain that would force exact_args == 47 does not
hold.** The exact_args / valid deviations are downstream of the 11 non-parity turns
(diagnosis below). This is reported, not silently promoted.

## Diagnosis — the deviations are the windowed-probe residual, NOT a regression

The 11 byte-parity breaks:

| gt | ep/t | first_div | pos % 32 | n_gen/n_ref | finish | proj | eng/hf exact | valid |
|---|---|---:|---:|---|---|---:|---|---:|
| 1 | ep0/t1 | 41 | 9 | 76/75 | stop | 0 | 0/0 | 1 |
| 19 | ep6/t0 | 19 | 19 | 85/69 | **length** | 0 | 0/0 | **0** |
| 21 | ep6/t2 | 26 | 26 | 78/79 | stop | 0 | 0/0 | 1 |
| 23 | ep7/t1 | 53 | 21 | 96/103 | stop | 0 | 0/0 | 1 |
| 24 | ep7/t2 | 34 | 2 | 98/92 | stop | 0 | 0/0 | 1 |
| 44 | ep14/t0 | 31 | 31 | 101/99 | stop | 0 | 0/0 | 1 |
| 50 | ep15/t3 | 17 | 17 | 207/259 | stop | 0 | **1/1** | 1 |
| 57 | ep18/t0 | 19 | 19 | 57/56 | stop | 0 | 0/0 | 1 |
| 58 | ep18/t1 | 47 | 15 | 78/74 | stop | 0 | 0/0 | 1 |
| 60 | ep19/t0 | 19 | 19 | 170/169 | stop | 0 | **1/0** | 1 |
| 61 | ep19/t1 | 38 | 6 | 57/48 | stop | 0 | 0/0 | 1 |

Evidence this is the windowed-**bidirectional** residual, not the fixed async/stall
bug:
1. **No pos-33 signature.** `first_div` positions are `{17,19,19,19,26,31,34,38,41,47,53}`
   — scattered mid-block value-region positions (pos%32 mostly != 0/1), *not* the
   "first denoise token after the block-0 boundary" (pos 33) that was the
   async-rollback fingerprint. The boundary-rollback corruption is fixed.
2. **All `value_projection_events = 0`** — a forward-logit approximation, not
   grammar/projection corruption. `verify_invariants` passes on all 63.
3. The pin author flagged exactly this: the causal windowed-probe is an
   approximation of the reference's windowed-**bidirectional** read that "affects
   value logits mid-block on some turns" (commit `d2fccab`, "Known residual").
4. The two eng!=hf turns are both explained by this: **gt60** — a mid-block
   divergence (`fd=19`) happened to produce the *correct* argument where HF was
   wrong (engine **+1**); **gt19** — a divergence (`fd=19`) cascaded into a
   non-stopping run to `n_ref+16` (finish=length) -> the one invalid tool call
   (valid 63->62). **gt50** diverged but still scored exact (stopped early at 207 of
   259, correct call).

**Conclusion:** the OPT-3 sync fix cleared its two named blocker classes (boundary
divergence + partial-canvas stall) completely; the residual 11-turn byte-divergence
is the *pre-existing, separate, documented* windowed-bidirectional approximation
(the batched follow-up), unrelated to OPT-1 / the read-only-denoise restore / the
IMA fix. Net quality is **>= HF** in aggregate but the model is **not byte-identical**
to the HF reference.

## Timing — the FIRST honest full-63 wall-clock (uncapped, all turns complete)

| metric | engine full-63 | HF full-63 (same turns) |
|---|---:|---:|
| s/turn **mean** | **1.681** | 3.904 |
| s/turn **p50** | **1.427** | — |
| s/turn **p90** | **2.724** | — |
| s/turn min / max | 0.512 / **5.361** | — |
| worst turn | gt50 ep15/t3 (207 tok, 190 fwd, 5.361 s) | — |
| **TRUE denoise forwards/turn** | **56.65** | 56.83 |
| tokens / forward | 1.360 | — |
| n_gen mean / median / max | 77.0 / 67 / 207 | — |
| finish stop / length | 62 / 1 | — |
| **engine speedup vs HF** | **2.32x** | — |

The 2.32x under HF on the *full* battery confirms the OPT-1 win (2.27-2.36x measured
on the bench subset) generalizes to the complete distribution, including the long
turns. Full per-turn distribution: `aggregate.json` (`per_turn`) and
`matched20_turns.jsonl`.

## Honest scoreboard + M2 / K3 adjudication

| row | exact_args | episode | valid | s/turn mean | p50 | p90 | fwd-or-tok/turn |
|---|---:|---:|---:|---:|---:|---:|---:|
| **ENGINE (OPT-3 fix), full-63** | **48/63** | **13/20** | 62/63 | **1.681** | 1.427 | 2.724 | 56.65 denoise fwd |
| HF hybrid-clean (v2), full-63 | 47/63 | 13/20 | 63/63 | 3.904 | — | — | 56.83 denoise fwd |
| stock-bf16-AR-guided, full-63 | 51/63 | — | 63/63 | 1.213 | — | — | 82.24 tok |
| stock-AR aggregate | 124/247 | — | — | 0.741 | — | — | 49.06 tok |
| **M2 / K3 target** | >=55/63 | — | — | **< 1.120** | — | — | — |

**M2 / K3 — now ADJUDICABLE for the first time, and MISSED on the honest full battery:**
- **Speed:** full-63 mean **1.681 s/turn > 1.120** -> the < 1.120 bar is **missed**.
  The bench's 1.250 was a *short-turn subset* (mean 60 tok, long turns + stalls
  excluded); the honest full-63 (mean 77 tok, incl. 16 long turns up to 259) is
  **1.681**. The engine is **2.32x under HF** but **1.39x slower than guided-AR
  (1.213)** and **2.27x slower than stock-AR-agg (0.741)**.
- **Quality:** exact_args **48/63 >= HF 47** (a clean, 0-projection constrained-lane
  number) but **< guided-AR 51** and **< the M2 >=55 bar**. Episode-exact ties HF at
  13/20.
- **Verdict:** M2/K3 **not met** on speed or quality, but — crucially — **the
  battery now completes so the bar is adjudicable at all** (the bench could not
  produce a full-63 number). The residual speed gap to AR is the per-forward compute
  **shape** (OPT-4), measured below; the residual quality gap is the
  windowed-bidirectional forward (the OPT-3 follow-up).

## temp = 0.7 rollouts (RL-rollout sanity + timing)

5 seeded rollouts (gt0/7/17/29/51), **two independent boots, same seed -> byte-reproducible**
(identical n_gen + forwards; wall within 2 ms):

| gt | n_gen | fwd | finish | valid | exact(hf) | proj | wall (run1 / run2) |
|---|---:|---:|---|---:|---|---:|---|
| 0 | 42 | 23 | stop | ok | 1(1) | 0 | 0.785 / 0.787 |
| 7 | 36 | 16 | stop | ok | 1(1) | 0 | 0.576 / 0.578 |
| 17 | 31 | 15 | stop | ok | 1(1) | 0 | 0.546 / 0.547 |
| 29 | 44 | 26 | stop | ok | 1(1) | 0 | 0.854 / 0.857 |
| 51 | 47 | 27 | stop | ok | 1(1) | 0 | 0.875 / 0.876 |

All bounded, grammar-valid, `value_projection_events = 0`, reproducible. The highly-peaked
value distributions collapse temp-0.7 onto the greedy tokens (same n_gen/parity as
greedy) — the RL contract holds. (Determinism is triply cross-checked: the battery,
and both profiler boots, all produced identical n_gen/forwards for gt7/gt25/gt35.)

## Next hotspot — OPT-4 (incremental KV+GDN 1-token decode), MEASURED

`torch.profiler` CUDA-kernel attribution on 3 turns (gt7 short / gt25 medium / gt35
long), kernel-level only (operator-dispatch rows excluded to kill double-counting).
Consistent across all three:

| family | share of GPU time | note |
|---|---:|---|
| **gemm (MLP + q/k/v/o/gate proj + lm_head)** | **~62%** | cutlass bf16 s16816 gemm, **computed over CL=32 query rows to read 1 probe logit** |
| elementwise / copy | ~21% | index_select/copy/add/mul (canvas staging) |
| full-attn (`unified_attention`) | ~6-9% | 8 full-attn layers over the context |
| **GDN / linear-attn (chunk path)** | ~5% (spread) | see below |
| norm / act / rope | ~1% | |
| **sampling / topk** | **0.5%** | **OPT-1 confirmed — the host-sampling wall is gone** |

**OPT-4 root cause confirmed by kernel names (gt35):** GDN dispatches the **prefill
`chunk_gated_delta_rule` path** — `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`,
`chunk_fwd_kernel_o`, `chunk_scaled_dot_kkt_fwd_kernel`, `recompute_w_u_fwd_kernel`,
`_causal_conv1d_fwd_kernel`, `chunk_local_cumsum`, `merge_16x16_to_64x64_inverse` —
and the single-token **`fused_recurrent` decode kernel is entirely absent** (2976
GDN-kernel calls = 118 forwards x ~25 layers, i.e. the chunk kernel every layer
every forward). Because read-only-denoise then restores GDN state, the chunk kernel's
state write is discarded — pure waste, exactly as the plan predicted.

Per-forward cost (gt35, 118 forwards): **~18 ms GPU** + **~11 ms host** (real,
unprofiled: battery 3.443 s / 118 = 29.2 ms/fwd; profiled GPU 18.0 ms/fwd). The host
component is eager kernel-launch overhead — the engine runs **`enforce_eager=True`
(no CUDA graph)**, so OPT-4 must pair the 1-token decode with the FULL decode graph
to collapse both.

**OPT-4 lever:** fold each committed clean token's KV+GDN state incrementally so the
probe is a genuine 1-token decode -> GDN hits `fused_recurrent`, the CL-wide gemm/attn
token-work collapses toward 1 row, and the FULL CUDA graph replays (removing the
~11 ms/fwd launch overhead). **Honest caveat:** at batch=1 the gemm is
weight-bandwidth-bound, so shrinking 32->1 rows will *not* give a 32x gemm win — the
clean wins are the GDN chunk->recurrent switch and the eager->graph launch-overhead
removal; the gemm benefit is bounded by weight bandwidth. This is the residual gap
between the current 1.681 s/turn and guided-AR's 1.213.

## Method notes / integrity

- Prompts byte-identical to the HF matched-20 eval (63 `prompt_sha256` + `prompt_tokens`
  verified in the source `matched20_ref.json`); parity scored token-for-token vs
  `HF.generated_token_ids`; exactness independently via `score_tool_calls`.
- The pin fix is config-only (`vllm/config/vllm.py`), guarded by `diffusion_config
  is not None`; it does not touch `hybrid_clean.py` (OPT-1), the read-only-denoise
  restore (`af21dc8`), or the IMA fix (`1e32dcd`). gt4's 110-token / 3-boundary
  byte-parity confirms the GDN read-only-denoise + boundary migration are intact
  under sync.
- Determinism: two boots of the battery + two profiler boots + two temp-0.7 boots —
  all agreeing on n_gen/forwards per turn.

## Artifacts (`runs/p2_engine_battery_full/`)

- `matched20_turns.jsonl` — 63 completed turns, incremental (byte-parity, exact,
  forwards, wall, counters, verify per turn).
- `aggregate.json` — all headline stats + full `per_turn` distribution + the 11
  parity-break records.
- `matched20_temp07a.jsonl` / `matched20_temp07b.jsonl` — the 5 seeded temp-0.7
  rollouts, twice (reproducibility).
- `profile_opt4.py` -> `opt4_breakdown.json` — the OPT-4 kernel-level forward-time
  breakdown on 3 turns (all kernels + families + per-forward GPU/host split).
- Reference: `runs/p2_engine_bench/matched20_ref.json` (63 prompt-verified HF records).
