# P2 Engine Bench — first honest matched-20 engine wall-clock (2026-07-04)

vLLM pin `58cfe2c` (`qwen3_5-flare-modelstate`: GAP-5A windowed-probe forward +
**OPT-1** GPU-native sampling), real export `qwen3.5-9b-fastdllm-rlv2-vllm-bf16`
(block/canvas 32, mamba 1024, align+APC), RTX 5090, RAM cage. Greedy.
Full detail + artifacts: `runs/p2_engine_bench/report.md`.

## Verdict

| step | verdict | one-line |
|---|---|---|
| prompts byte-faithful | **PASS** | all 63 matched-20 prompts reconstructed from the HF row's teacher-forced history; every `prompt_sha256` + `prompt_tokens` matches, 3 `gap5a_ref` records cross-check byte-for-byte. |
| OPT-1 integrity (A/B vs pre-OPT-1 `6b81154`) | **PASS** | engine output **byte-identical** OPT-1 vs pre-OPT-1 on all tested turns (incl. divergent ones) at **2.36x** speedup -> OPT-1 is a pure, behavior-preserving speedup. |
| full 63-turn battery end-to-end | **CANNOT COMPLETE** | 44/63 turns run; **19 uncompletable** (16 long n_ref>=95 + 3 short) due to a partial-canvas forward **STALL** (>10 min, non-terminating). |
| byte-parity == HF 47/63 by construction | **DOES NOT GENERALIZE** | parity holds on 35/44 completed turns; **9 diverge** (proj 0) from the GAP-5A windowed-probe *causal approximation* of the reference's windowed-*bidirectional* read. Engine exact_args on completed = **32/44** (HF 35/44 on same). NOT an optimization regression (A/B above). |
| temp=0.7 RL sanity | **PASS** | 5 rollouts bounded/valid/proj0, same-seed 2x reproducible. |

## The two blockers (both pre-OPT-1, both OPT-3 territory, both correctness)

1. **Non-universal byte-parity.** Windowed-probe fix is a *causal* approx of the
   reference's windowed-*bidirectional* read (author-flagged). Diverges on 9/44
   completed turns, systematically at the first denoise position after a block
   boundary (`first_div=33` recurs), all `value_projection_events=0`. Present
   identically at `6b81154` (A/B) -> not from OPT-1.
2. **Partial-canvas forward STALL.** When staged canvas `valid_len` drops below the
   full block width (measured **32->13** at committed ~95), one denoise forward
   never returns. Blocks all turns generating past ~90-95 tokens (16 turns) + 3
   short turns on block-aligned prompts (gt32 plen=1024=mamba block). **Grammar is
   0.7% of turn time** — the prior "O(committed^2) grammar" hypothesis is disproven.

## Numbers (44 completed turns — a short-turn subset; NOT a full-battery number)

| | engine | HF (same 44) | HF full-63 | stock-AR-guided 63 | stock-AR agg | M2 |
|---|---:|---:|---:|---:|---:|---:|
| s/turn mean | **1.250** | 2.835 | 3.904 | 1.213 | 0.741 | <1.120 |
| s/turn p50 | 1.185 | 2.756 | — | — | — | — |
| worst completed | 2.201 | — | — | — | — | — |
| denoise fwd/turn | 40.95 | 39.30 | 56.83 | 82.24 (tok) | 49.06 (tok) | — |
| exact_args | 32/44 | 35/44 | 47/63 | 51/63 | 124/247 | >=55/63 |

Engine is **2.27x** under HF on the identical completed subset (a real OPT-1 win) and
its completed-subset s/turn sits at the stock-AR-guided level — but **M2 is
unadjudicated**: the full battery cannot complete and byte-parity is not universal.
On the 35 parity-hold turns, engine exact_args == HF exactly (31/31).

## Frontier (evidence-ranked)

1. **OPT-3 — byte-EXACT windowed-BIDIRECTIONAL variable-width single-`[MASK]`
   forward.** Fixes both blockers at once (kills the divergence -> restores
   universal parity + the HF 47/63; kills the stall -> long turns + full battery run
   -> real full-battery s/turn). **#1 correctness blocker.**
2. **OPT-4** — incremental KV/GDN 1-token decode; residual GPU gap to AR, only after
   OPT-3.
3. **OPT-1 done + verified** (byte-identical, 2.36x). **OPT-5 do NOT do** (0.7%).

Artifacts: `runs/p2_engine_bench/{build_matched20_ref.py,matched20_ref.json,run_battery.py,
matched20_turns.jsonl,ab_opt1.py,ab_A_opt1.json,ab_B_preopt1.json,diag_ep1*.py,
matched20_temp07*.jsonl,report.md}`.
