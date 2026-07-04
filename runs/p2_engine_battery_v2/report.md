# P2 Engine Battery v2 — the promotable full-63 matched-20 on the FINAL engine (2026-07-04)

vLLM pin **`e5496cc`** (`qwen3_5-flare-modelstate`), real export
`models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16` (block/canvas 32, mamba 1024, align+APC),
RTX 5090, one heavy process in the `systemd-run … MemoryMax=22G MemorySwapMax=4G`
cage. Greedy, temp 0, seed 20260701, **uncapped** (`max_tokens = n_ref + 16`, no
hard cap). Two boots (ep0-9, ep10-19), per-turn incremental JSONL.

**The FINAL engine = both landed levers on:**
- `VLLM_FLARE_BIDIR_PROBE=1` — the reference-exact windowed-**BIDIRECTIONAL**
  denoise read (pin `b7d76e2`; the correct `flare_hf_cache.noisy_active_attention_mask`
  semantics), replacing the legacy windowed-CAUSAL probe approximation.
- `VLLM_FLARE_CUDAGRAPH=1` — PIECEWISE cudagraph for the block-diffusion decode
  (pin `e5496cc`, OPT-4 part 2). Boot log confirms `enforce_eager=False`,
  `cudagraph_mode=PIECEWISE`, "Capturing CUDA graphs (PIECEWISE)"; every turn
  records `cg_pw` PIECEWISE-graph dispatches > 0 (3756 total over the battery).

## Headline

**The cudagraph win lands the engine UNDER the M2 and guided-AR speed bars for
the first time** (mean **1.051 s/turn**, vs the eager engine's 1.681/1.697 that
missed both), while the bidir read holds parity at **58/63**, valid at **63/63**,
and quality at **>= HF**. But **byte-parity is not 63/63**, so the strict
"engine == HF by construction" promotion gate is **NOT met** — the residual 5
divergences are the *coupled, documented, unlanded* 32-absolute block alignment +
per-request variable commit-width work (OPT-4 Part 1). Reported, not promoted.

## Gate results vs the task's required checks

| task check | required | measured | verdict |
|---|---|---|---|
| (1) byte-parity per turn | **63/63** | **58/63** | **NOT met** — 5 breaks {20,21,44,45,60} (diagnosed below) |
| (2) exact_args | == 47/63 | **48/63** | deviation **+1** — engine wins gt60 (correct where HF is wrong) |
| (2) episode_exact | 13/20 | **13/20** | **met** (ties HF) |
| (2) valid | 63/63 | **63/63** | **met** (bidir fixed the causal engine's lone gt19 invalid) |
| verify_invariants | — | **63/63** | clean |
| value_projection_events | 0 | **0/63** | clean (label-free constrained lane) |

Per the task's "byte-parity implies all three; any deviation => stop and
diagnose" rule: **byte-parity is 58/63, so the by-construction chain that would
force exact_args == 47 does not hold.** exact_args is 48 (engine wins gt60);
valid is 63/63; episode 13/20. The one exact deviation and all 5 parity breaks
are downstream of the same unlanded alignment/variable-width residual (below).
This is reported, not silently promoted.

## Cudagraph is byte-neutral on the promotable set (proof)

The bidir-**eager** full-63 (pin `b7d76e2` validation, `parity_bidir/battery_bidir.jsonl`)
is the anchor: 58/63 parity, breaks **{20,21,44,45,60}**, exact 48, valid 63,
episode 13/20. This cudagraph-**on** battery reproduces that anchor **exactly**:

- Same parity count **58/63**, same break set **{20,21,44,45,60}**
  (`parity_matches_bidir_eager_anchor = true`).
- **All 58 parity turns are byte-identical to HF** (hence to the eager anchor).
- The only two turns whose `n_gen`/`fwd` differ from the eager anchor are gt20 and
  gt44 — and only in their **already-divergent tail** (both are non-parity in
  eager too; gt20 differs by 1 tail token, gt44's divergent trajectory differs).
  Neither is promotable in either path (both eng_exact==hf_exact==0).

So PIECEWISE cudagraph changes **zero** promotable-turn output and only perturbs
the tail of turns that already diverge from HF — exactly the SPEED-REPORT finding,
now confirmed on the full 63 under bidir.

## The 5 byte-parity breaks (all diagnosed; none a regression of a landed fix)

| gt | ep/t | first_div | pos%32 | n_gen/n_ref | finish | proj | eng/hf exact | valid | class |
|---|---|---:|---:|---|---|---:|---|---:|---|
| 20 | ep6/t1 | 20 | 20 | 75/79 | stop | 0 | 0/0 | 1 | bidir alignment regression |
| 21 | ep6/t2 | 19 | 19 | 66/79 | stop | 0 | 0/0 | 1 | APC/prefix-cache artifact |
| 44 | ep14/t0 | 16 | 16 | 103/99 | stop | 0 | 0/0 | 1 | bidir-miss (variable-width) |
| 45 | ep14/t1 | 20 | 20 | 109/110 | stop | 0 | 0/0 | 1 | bidir alignment regression |
| 60 | ep19/t0 | 19 | 19 | 170/169 | stop | 0 | **1/0** | 1 | bidir divergence — **engine WINS** |

Root cause (pin `b7d76e2` "REMAINING STRUCTURAL WORK", decisive HF attribution):
the engine commits **generation-aligned** 32-blocks while the reference commits
**32-ABSOLUTE-aligned** blocks (first block absorbs `prompt%32` leftover tokens).
Bidirectional reads are alignment-sensitive (causal reads are block-invariant),
so full byte-parity needs (a) 32-absolute commit alignment (`VLLM_FLARE_ALIGN_BLOCKS`,
scaffold only, default off) **and** (b) a per-request **variable** commit width so
the GDN fold == block_target (a fixed-32 forward over-folds the recurrent state by
the stale slots on a sub-32 block). (b) is the deferred "true per-request variable
draft width" contract — which is **also the OPT-4 forward-compute cut**: parity
closure and the remaining speed cut are coupled and land together.

Because the bidir mask fix trades block-invariance for reference-correctness, it
**fixes 8** of the 11 causal-path divergences ({1,19,23,24,50,57,58,61}) and
**regresses 2** block-invariant turns (gt20/gt45) while **3 remain** (gt21 APC,
gt44 variable-width, gt60 which the engine actually gets *right*): net 11 -> 5.
All 5 are grammar-valid, `value_projection_events = 0`, `verify_invariants` pass,
and eng_exact == hf_exact on 4/5 (engine strictly wins the 5th, gt60). Net quality
is **>= HF** everywhere.

## Timing — the first engine full-63 UNDER M2 and guided-AR

| metric | v2 (bidir+cudagraph) | bidir-eager anchor | causal-eager (prior report) | HF full-63 |
|---|---:|---:|---:|---:|
| s/turn **mean** | **1.051** | 1.697 | 1.681 | 3.904 |
| s/turn **p50** | **0.876** | — | 1.427 | — |
| s/turn **p90** | **1.699** | — | 2.724 | — |
| s/turn min / max | 0.326 / **4.248** | — | 0.512 / 5.361 | — |
| worst turn | gt50 (259 tok, 242 fwd, 4.248 s) | — | gt50 (207 tok, 5.361 s) | — |
| **TRUE denoise fwd/turn** | **56.62** | 56.49 | 56.65 | 56.83 |
| tokens / forward | 1.362 | — | 1.360 | — |
| **per-forward ms** (amortized) | **18.56** | ~30 (eager) | ~29 (eager) | — |
| per-forward ms (long-turn settled) | 18.06 | — | — | — |
| **speedup vs HF** | **3.715x** | 2.30x | 2.32x | — |

**Cudagraph win: 1.697 -> 1.051 = 1.615x** (vs the eager-bidir anchor on the same
turns), matching the SPEED REPORT's 1.64x. The per-forward host overhead collapse
(eager ~29-30 ms/fwd -> 18.56 ms/fwd) is the whole lever; TRUE fwd/turn (56.62)
and tokens/forward (1.362) are unchanged — this is pure launch-overhead removal,
not a compute-shape change (that is OPT-4 Part 1, still open).

## Bar adjudication (mean s/turn = 1.051)

| bar | value | ratio | verdict |
|---|---:|---:|---|
| HF hybrid-clean | 3.904 | 0.269x | **UNDER (beat)** |
| guided-AR (stock-bf16 matched-20) | 1.213 | 0.866x | **UNDER (beat)** ← eager missed (1.39x over) |
| **M2 / K3** | **1.120** | **0.938x** | **UNDER (beat)** ← eager missed (1.50x over) |
| stock-AR aggregate | 0.741 | 1.418x | OVER (miss) |

The engine now **beats M2 (1.120), guided-AR (1.213), and HF (3.904)** on speed —
the first time on the honest full-63. Only stock-AR-**aggregate** (0.741; a mix
dominated by short never-train turns) remains above the engine. The residual gap
to stock-agg is the per-forward GPU compute **shape** (the CL=32-wide gemm/attn +
GDN chunk-vs-recurrent), i.e. OPT-4 Part 1 — which also closes the parity residual.

## Honest scoreboard

| row | exact_args | episode | valid | s/turn mean | p50 | p90 | fwd-or-tok/turn |
|---|---:|---:|---:|---:|---:|---:|---:|
| **ENGINE v2 (bidir+cudagraph), full-63** | **48/63** | **13/20** | **63/63** | **1.051** | 0.876 | 1.699 | 56.62 denoise fwd |
| HF hybrid-clean (v2), full-63 | 47/63 | 13/20 | 63/63 | 3.904 | — | — | 56.83 denoise fwd |
| stock-bf16-AR-guided, full-63 | 51/63 | 14/20 | 63/63 | 1.213 | — | — | 82.24 tok |
| stock-AR aggregate | 124/247 | — | — | 0.741 | — | — | 49.06 tok |
| **M2 / K3 target** | >=55/63 | — | — | **< 1.120** | — | — | — |

**Quality vs the >=55 M2 aspiration:** exact_args 48/63 >= HF 47 but < guided-AR
51 and < 55. That gap is a **model-training** matter (the memory's 55/63 aspiration),
not an engine defect: the engine byte-matches its served HF row on 58/63 and only
misses full parity via the alignment/variable-width residual. The engine quality
gate is byte-parity with the served HF row (47/63-exact HF), which the engine meets
or exceeds on every parity turn.

## temp = 0.7 rollouts (RL-rollout sanity + timing)

5 seeded rollouts (gt0/7/17/29/51), **two independent boots (a/b), same seed**:

| gt | n_gen | fwd | finish | valid | exact(hf) | proj | parity | wall a / b |
|---|---:|---:|---|---:|---|---:|---|---|
| 0 | 42 | 23 | stop | ok | 1(1) | 0 | yes | 0.512 / 0.513 |
| 7 | 36 | 16 | stop | ok | 1(1) | 0 | yes | 0.378 / 0.378 |
| 17 | 31 | 15 | stop | ok | 1(1) | 0 | yes | 0.359 / 0.360 |
| 29 | 44 | 26 | stop | ok | 1(1) | 0 | yes | 0.549 / 0.550 |
| 51 | 47 | 27 | stop | ok | 1(1) | 0 | yes | 0.556 / 0.559 |

**Byte-reproducible across both boots** (identical n_gen/forwards/parity; max wall
delta 3 ms). All bounded, grammar-valid, `value_projection_events = 0`. The peaked
value distributions collapse temp-0.7 onto the greedy tokens (same n_gen/parity as
greedy) — the RL contract holds under cudagraph.

## Never-train spot-check (BFCL / API-Bank, not matched-20-specific)

3 turns spanning families, prompts reconstructed teacher-forced and **prompt_sha256
byte-verified** against the HF never-train eval (all 184 turns reconstruct;
`ALL_prompt_sha256_match = true`, 0 problems):

| gt | family | parity | n_gen/n_ref | valid | exact(hf) | proj | cg_pw | ms/f | wall |
|---|---|---|---|---:|---|---:|---:|---:|---:|
| 147 | BFCL-AST | **yes** | 54/54 | ok | 1(1) | 0 | 39 | 19.43 | 0.719 |
| 159 | API-Bank-Lv1 | **yes** | 37/37 | ok | 1(1) | 0 | 23 | 21.71 | 0.456 |
| 172 | API-Bank-Lv2 | **yes** | 38/38 | ok | 1(1) | 0 | 24 | 21.91 | 0.482 |

**3/3 byte-parity vs HF, 3/3 valid, 3/3 exact-correct, 0/3 projection, cudagraph
captured on all.** The engine byte-matches the HF reference on out-of-distribution
never-train BFCL-AST + API-Bank Lv1/Lv2 prompts too — the parity chain is **not**
matched-20-specific.

## Method notes / integrity

- Prompts byte-identical to the HF eval (63 matched-20 `prompt_sha256` +
  `prompt_tokens` verified in `runs/p2_engine_bench/matched20_ref.json`; 184
  never-train verified in `nevertrain_ref.json`). Parity scored token-for-token
  vs `HF.generated_token_ids`; exactness independently via `score_tool_calls`.
- Cudagraph confirmed live: `enforce_eager=False`, `cudagraph_mode=PIECEWISE`,
  PIECEWISE capture at boot, 3756 PIECEWISE-graph dispatches recorded across 63
  turns (every turn > 0).
- Determinism: greedy smoke (gt0/19/20/44/45/60) is byte-identical to the battery
  turns; the temp-0.7 rollouts are byte-identical across two boots.
- The pin (`e5496cc`) is editable-installed (`vllm.__file__` -> the pin source);
  no rebuild. The `enforce_eager` opt-in seam is `scripts/parity_audit_flare_engine.py`
  (`VLLM_FLARE_CUDAGRAPH`).

## Verdict

The FINAL engine (bidir + PIECEWISE cudagraph) is the strongest promotable
candidate to date: **58/63 byte-parity, valid 63/63, exact 48>=47, episode 13/20,
verify 63/63, projection 0, and mean 1.051 s/turn — the first engine to beat M2
(1.120), guided-AR (1.213), and HF (3.904) on the honest full-63.** But the strict
promotion gate (**63/63 byte-parity => exact exactly 47**) is **NOT met**: 5 turns
diverge via the coupled, unlanded 32-absolute-alignment + per-request variable
commit-width work (OPT-4 Part 1), which is also the remaining forward-compute speed
cut. **Not promoted.** The single lever that closes both the last 5 parity turns
and the residual speed gap to stock-agg is OPT-4 Part 1.

## Artifacts (`runs/p2_engine_battery_v2/`)

- `matched20_turns.jsonl` — 63 turns (parity, exact, forwards, wall, per-forward
  ms, cg_pw dispatches, counters, verify, fd tokens).
- `aggregate.json` — headline stats + full `per_turn` + the 5 parity-break records
  + anchor cross-check.
- `matched20_temp07a.jsonl` / `matched20_temp07b.jsonl` — 5 seeded temp-0.7
  rollouts, twice (reproducibility).
- `nevertrain_ref.json` (184 sha-verified never-train records) +
  `nevertrain_spotcheck.jsonl` (the 3 spot-check turns).
- `smoke.jsonl` — the 6-turn cudagraph/bidir de-risk (determinism anchor).
- `run_battery_v2.py`, `aggregate.py`, `build_nevertrain3_ref.py`, `env.sh` — drivers.
- Anchor: `runs/p2_engine_bench/parity_bidir/battery_bidir.jsonl` (bidir-eager 58/63).
