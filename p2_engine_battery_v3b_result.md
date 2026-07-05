# P2 Engine Battery v3b — PROMOTION ATTEMPT on the POST-FIX engine: 62/63, exact EXACTLY 47, NOT PROMOTED (2026-07-04)

vLLM pin **`95d8b47`** (`qwen3_5-flare-modelstate`: OPT-4 **Stage 1+2+3 landed** — 32-absolute
variable commit width + scheduler width plumbing + byte-robust bidir key window; **code default
OFF**), export `qwen3.5-9b-fastdllm-rlv2-vllm-bf16` (block/canvas 32, mamba 1024, align+APC),
RTX 5090, RAM cage. Greedy, temp 0, seed 20260701, uncapped (`max_tokens=n_ref+16`), chunked
foreground (ep0-9, ep10-19), per-turn JSONL. FINAL engine = `VLLM_FLARE_BIDIR_PROBE=1` +
`VLLM_FLARE_CUDAGRAPH=1`. Independent fresh boot of the Stage-3 engine (NOT a re-run of the
pre-fix v2/v3 pin `e5496cc`). Full detail + artifacts: `runs/p2_engine_battery_v3b/report.md`.

## Verdict

| step | verdict | one-line |
|---|---|---|
| full 63-turn battery completes | **PASS** | all 63 run, zero stalls, mean **1.053 s/turn** |
| byte-parity == HF 63/63 (promotion gate) | **NOT MET (62/63)** | lone break gt44 = path-invariant fp-residue; not a regression, not APC-class |
| exact_args == 47/63 | **MET (exactly 47)** | 0 turns eng!=hf (fix drops the pre-fix gt60 "win" 48->47) |
| episode_exact 13/20 | **MET** | ties HF |
| valid 63/63 | **MET** | |
| verify_invariants / value_projection | **CLEAN** | 63/63 verify ok; 0/63 projection |
| cudagraph captured (PIECEWISE) | **PASS** | 63/63 turns, 3798 dispatches, per-forward 18.52 ms |
| Stage-3 fix landed (58/63 -> 62/63) | **PASS** | cleared {20,21,45,60} path-robustly; no regressions |
| gt44 APC-class? (documented protocol) | **NO** | breaks identically fresh-boot (fd16,n101) -> cannot rescue to 63/63 |
| temp=0.7 RL sanity | **PASS** | 5 rollouts bounded/valid/proj0/parity, byte-reproducible across 2 boots (max wall d 4ms) |
| never-train BFCL/API-Bank spot-check | **PASS** | 3/3 byte-parity vs HF, valid, exact, proj0 |

**PROMOTED = NO.** 63/63 gate is one turn short; code default stays OFF.

## Numbers (all 63, greedy, uncapped)

| | ENGINE v3b (post-fix) | pre-fix v3/v2 (`e5496cc`) | HF | guided-AR | stock-agg | M2/K3 |
|---|---:|---:|---:|---:|---:|---:|
| byte-parity | **62/63** (break {44}) | 58/63 ({20,21,44,45,60}) | (self) | — | — | 63/63 |
| exact_args | **47/63** (== HF) | 48/63 (won gt60) | 47/63 | 51/63 | 124/247 | >=55/63 |
| episode_exact | **13/20** | 13/20 | 13/20 | — | — | — |
| valid | **63/63** | 63/63 | 63/63 | 63/63 | — | — |
| s/turn mean | **1.053** | 1.056 | 3.904 | 1.213 | 0.741 | <1.120 |
| s/turn p50 / p90 | 0.896 / 1.700 | 0.874 / 1.734 | — | — | — | — |
| s/turn worst | 4.241 (gt50, 259 tok) | 4.253 | — | — | — | — |
| TRUE denoise fwd/turn | 56.86 | 56.62 | 56.83 | 82.24 tok | 49.06 tok | — |
| per-forward ms (cudagraph) | **18.52** | 18.66 | — | — | — | — |

## Bar adjudication (mean 1.053 s/turn)

- **HF 3.904 -> BEAT** (0.270x, 3.708x speedup).
- **guided-AR 1.213 -> BEAT** (0.868x).
- **M2/K3 1.120 -> BEAT** (0.940x).
- **stock-agg 0.741 -> MISS** (1.421x). **Measured** residual-gap breakdown below; **NOT** an
  engine defect and **NOT** closable by width-narrowing.

## Residual gap to stock-agg 0.741 (MEASURED) — NOT reachable by engine plumbing at batch=1

Per-forward wall **18.52 ms** = **weight-stream floor 11.40 ms** (measured gemm device self-time,
63.5% of GPU; arithmetic cross-check 10.77 ms = 19.31 GB bf16 / 1.79 TB/s HBM, **irreducible at
bs=1**) + non-weight GPU compute **6.54 ms** + residual host/launch **0.58 ms** (cudagraph). To
hit 0.741 at 56.86 fwd/turn needs 13.03 ms/fwd (cut 5.49 ms) — only 1.64 ms above the weight
floor. **Stage-3 A/B proved the 6.54 ms non-weight compute does NOT shrink with variable width**
(18.52 vs 18.56 ms; cudagraph buckets narrow widths back to a captured bucket) — so the CL=32
shape was **not** the residual. **This supersedes the v3/section-0.G "REACHABLE via OPT-4 Part 1"
claim.** Reachable levers are orthogonal to parity/integration: (a) fewer forwards/turn (larger
parallel commit — training), (b) fp8/int8 weights (halve/quarter the floor -> ~0.68/0.51 s/turn,
a quality tradeoff), (c) batching (amortize the weight stream across concurrent requests).
Caveat: stock-agg is a stock-AR number over a different, shorter turn mix (49.06 tok/turn).

## What this establishes / does not

- **Establishes:** the OPT-4 Stage-1/2/3 fix is real and independently reproducible — 58/63 ->
  **62/63**, exact **exactly 47 (== HF, 0 wins/losses)**, valid 63/63, episode 13/20, verify
  63/63, projection 0, mean 1.053 s/turn (beats M2, guided-AR, HF). The 4 pre-fix breaks
  {20,21,45,60} are cleared **path-robustly** (clean in APC-on AND cold-prefix fresh-boot); no
  regressions; shared-clean turns byte-identical to the pre-fix run.
- **Does NOT establish:** the strict 63/63 promotion gate. The lone residual **gt44** is a proven
  **path-invariant deterministic fp-residue** (breaks identically fresh-boot; both engine and HF
  non-exact there -> quality-neutral), rooted in the block#0 fold-path fp gap the pin documents
  (HF folds 32 incl. `prompt%32` leftover; the aligned engine folds `32 - L%32` gen tokens from
  the L checkpoint — fp-close, not bit-identical). **Not promoted; code default stays OFF.**

## Frontier (evidence-ranked)

1. **gt44 byte-parity** requires matching HF's block#0 GDN fold granularity / chunk boundary
   exactly (kernel-level, explicitly deferred) — the last 1 turn. It is quality-neutral, so its
   value is the strict "engine == HF by construction" certificate, not a quality gain.
2. **Quality >=55/63** is a **model-training** (RL) matter, not an engine defect — the engine
   byte-matches its served HF row on every parity turn (exact == HF).
3. **stock-agg 0.741** is the batch-1 weight-stream floor + fixed per-forward cost (measured),
   not closable by engine plumbing; levers are training (fewer forwards/turn), quantization, or
   batching.

## Artifacts (`runs/p2_engine_battery_v3b/`)

`matched20_turns.jsonl` (63) . `aggregate.json` . `matched20_temp07{a,b}.jsonl` .
`nevertrain_spotcheck.jsonl` (3) . `parity_cert_freshboot.jsonl` (5) . `opt4_breakdown.json` .
`run_battery_v3b.py` / `aggregate.py` / `profile_v3b.py` / `env.sh` / `runcage.sh` /
`chunk1.log` / `chunk2.log`.
