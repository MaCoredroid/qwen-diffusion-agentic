# P2 Engine Battery v2 — promotable full-63 on the FINAL engine (bidir + cudagraph) (2026-07-04)

vLLM pin **`e5496cc`** (`qwen3_5-flare-modelstate`: windowed-BIDIRECTIONAL denoise
read `b7d76e2` + PIECEWISE cudagraph `e5496cc` OPT-4 part 2 + OPT-1 GPU sampling +
OPT-3 sync scheduler), export `qwen3.5-9b-fastdllm-rlv2-vllm-bf16` (block/canvas 32,
mamba 1024, align+APC), RTX 5090, RAM cage. Greedy, temp 0, seed 20260701, uncapped
(`max_tokens=n_ref+16`), two boots (ep0-9, ep10-19). FINAL engine =
`VLLM_FLARE_BIDIR_PROBE=1` + `VLLM_FLARE_CUDAGRAPH=1`. Full detail + artifacts:
`runs/p2_engine_battery_v2/report.md`.

## Verdict

| step | verdict | one-line |
|---|---|---|
| full 63-turn battery completes | **PASS** | all 63 run, zero stalls, mean **1.051 s/turn** |
| cudagraph captured (OPT-4 part 2) | **PASS** | `enforce_eager=False`, `cudagraph_mode=PIECEWISE`, 3756 PIECEWISE dispatches; **1.615x** vs eager-bidir (1.697->1.051) |
| byte-parity == HF 63/63 (promotable) | **NOT MET (58/63)** | 5 breaks {20,21,44,45,60} = unlanded 32-absolute-align + variable commit-width (OPT-4 Part 1); not a regression |
| exact_args == 47/63 | **DEVIATION +1 (48/63)** | engine wins gt60 (correct where HF wrong), no losses; >= HF |
| episode_exact 13/20 | **MET** | ties HF exactly |
| valid 63/63 | **MET** | bidir fixed the causal engine's lone gt19 invalid (62->63) |
| verify_invariants / value_projection | **CLEAN** | 63/63 verify ok; 0/63 projection |
| temp=0.7 RL sanity | **PASS** | 5 rollouts bounded/valid/proj0, byte-reproducible across 2 boots |
| never-train BFCL/API-Bank spot-check | **PASS** | 3/3 byte-parity vs HF, valid, exact, proj0 — not matched-20-specific |

## Numbers (all 63, greedy, uncapped)

| | ENGINE v2 (bidir+cudagraph) | bidir-eager anchor | causal-eager (prior) | HF | guided-AR | stock-agg | M2/K3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| exact_args | **48/63** | 48/63 | 48/63 | 47/63 | 51/63 | 124/247 | >=55/63 |
| episode_exact | **13/20** | 13/20 | 13/20 | 13/20 | — | — | — |
| valid | **63/63** | 63/63 | 62/63 | 63/63 | 63/63 | — | — |
| byte-parity | **58/63** | 58/63 | 52/63 | (self) | — | — | 63/63 |
| s/turn mean | **1.051** | 1.697 | 1.681 | 3.904 | 1.213 | 0.741 | **< 1.120** |
| s/turn p50 / p90 | 0.876 / 1.699 | — | 1.427 / 2.724 | — | — | — | — |
| s/turn worst | 4.248 (gt50, 259 tok) | — | 5.361 | — | — | — | — |
| denoise fwd/turn | 56.62 | 56.49 | 56.65 | 56.83 | 82.24 tok | 49.06 tok | — |
| per-forward ms | **18.56** | ~30 | ~29 | — | — | — | — |
| tokens/forward | 1.362 | — | 1.360 | — | — | — | — |

## Bar adjudication (mean 1.051 s/turn)

- **HF 3.904 -> BEAT** (0.269x, 3.715x speedup).
- **guided-AR 1.213 -> BEAT** (0.866x). Eager missed this (1.39x over).
- **M2/K3 1.120 -> BEAT** (0.938x). Eager missed this (1.50x over).
- **stock-agg 0.741 -> MISS** (1.418x). The residual is the per-forward compute
  **shape** (CL=32 gemm/attn + GDN chunk-vs-recurrent) = OPT-4 Part 1, which also
  closes the last 5 parity turns.

## What this run establishes, and what it does not

- **Establishes:** the PIECEWISE-cudagraph engine is **byte-neutral on the entire
  promotable set** — it reproduces the bidir-eager 58/63 parity and break-set
  {20,21,44,45,60} exactly, changing zero promotable-turn tokens (the only n_gen/fwd
  deltas vs the eager anchor are in the already-divergent tails of gt20/gt44). The
  OPT-4 part-2 speed win (1.615x, per-forward 29->18.56 ms host-overhead collapse)
  therefore lands the engine **UNDER M2, guided-AR, and HF** for the first time on
  the honest full-63, with quality **>= HF** (exact 48>=47, valid 63/63, proj 0).
- **Does NOT establish:** the strict "engine == HF by byte-construction" promotion
  gate (63/63 -> exact exactly 47). Byte-parity is 58/63; exact is 48 (engine wins
  gt60). **Not promoted.** The 5-turn residual is the coupled, documented, unlanded
  32-absolute commit alignment + per-request variable commit-width — **OPT-4 Part 1**,
  which is simultaneously the remaining forward-compute speed cut. Parity closure
  and the last speed cut land together.

## Frontier (evidence-ranked)

1. **OPT-4 Part 1 — per-request variable commit width + 32-absolute block alignment**:
   closes the last 5 parity turns (-> 63/63) AND the CL=32 forward-compute shape
   (-> under stock-agg 0.741). Single coupled lever; highest leverage remaining.
2. **Quality >=55/63** is a model-training matter (RL), not an engine defect — the
   engine byte-matches its served HF row on every parity turn.
3. OPT-1 done, OPT-3 done, **OPT-4 Part 2 (cudagraph) done + validated here**.

## Artifacts (`runs/p2_engine_battery_v2/`)

`matched20_turns.jsonl` (63) · `aggregate.json` · `matched20_temp07{a,b}.jsonl` ·
`nevertrain_ref.json` (184 sha-verified) + `nevertrain_spotcheck.jsonl` (3) ·
`smoke.jsonl` · `run_battery_v2.py` / `aggregate.py` / `build_nevertrain3_ref.py` /
`env.sh` · anchor `runs/p2_engine_bench/parity_bidir/battery_bidir.jsonl`.
