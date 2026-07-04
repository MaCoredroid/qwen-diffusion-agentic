# P2 Engine Bench — the FIRST honest matched-20 engine wall-clock (2026-07-04)

vLLM pin `58cfe2c` (`qwen3_5-flare-modelstate`: GAP-5A windowed-probe forward +
**OPT-1** GPU-native sampling), real export
`models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16` (block/canvas 32, mamba 1024, align+APC),
RTX 5090, one heavy process in the `systemd-run … MemoryMax=22G` cage. Greedy.

## Headline

The optimized engine is **fast and byte-exact on the turns it completes**
(1.250 s/turn mean over 44/63 turns — **2.27x faster than the HF stack on the same
turns**, OPT-1 verified as a pure byte-identical speedup), **but the full 63-turn
matched-20 battery cannot be run end-to-end on this engine**, for two *pre-OPT-1*
engine-forward reasons that also break the "engine == HF 47/63 by construction"
claim the acceptance inferred from 3 turns:

1. **Byte-parity to HF is NOT universal.** The GAP-5A windowed-probe fix is a
   *causal approximation* of the reference's windowed-**bidirectional** read
   (author-flagged in the acceptance doc: "byte-exact-by-construction would need
   a windowed-bidirectional mask"). It is byte-exact on the 3 originally-tested
   turns but **diverges from HF on 9 / 44 completed turns**, systematically at the
   first denoise position after a block boundary (`first_div=33` recurs). All
   divergences are `value_projection_events=0` (a forward-logit approximation, not
   grammar corruption).
2. **A partial-canvas forward STALL.** When the staged canvas `valid_len` drops
   below the full block width (measured **32 -> 13** at committed ~ 95), a single
   denoise forward **stalls indefinitely (> 10 min, non-terminating)**. This blocks
   every turn that generates past ~ 90-95 tokens (**16 turns, n_ref >= 95**) and
   also 3 short turns whose prompt length hits a block boundary (e.g. gt32,
   prompt_len = 1024 = the mamba block). **Grammar/detok is not involved** — it is
   0.7 % of turn time; the prior "O(committed^2) grammar" hypothesis is disproven.

Both are **OPT-3 territory** (the single-`[MASK]` variable-width forward) and both
are **correctness/liveness blockers, not efficiency levers**. So the answer to
"is OPT-3/OPT-4 width-narrowing now the frontier?" is **yes, emphatically — OPT-3
is the frontier and it is a correctness fix**, ahead of OPT-4 and far ahead of
OPT-5.

## Method (prompts are byte-identical to the HF matched-20 eval)

The matched-20 eval is a *generated-history* loop, so its per-turn prompts are the
HF row's own teacher-forced history. I reconstructed all 63 turn prompts from the
HF hybrid-clean row
(`runs/hybrid_forced_grammar_seq_values_v2/matched20/.../turns.jsonl`):
`prompt_0 = render_matched_prompt(...)`, then
`prompt_{k+1} = prompt_k + decode(HF.generated_token_ids_k) + tool_response_suffix(HF.payload_k)`
— and **verified every reconstruction**: all 63 `prompt_sha256` and all 63
`prompt_tokens` match the HF row exactly, and the 3 pre-tokenized `gap5a_ref`
records cross-check byte-for-byte (`prompt_ids` **and** `ref_new_ids`). The engine
then generates greedily on each exact prompt; parity is scored token-for-token vs
`HF.generated_token_ids` and exactness independently via `score_tool_calls`.
Artifacts: `matched20_ref.json` (63 verified records), `build_matched20_ref.py`,
`run_battery.py`.

## OPT-1 integrity — A/B vs pre-OPT-1 (the optimization is clean)

OPT-1 (`58cfe2c`) touches only `hybrid_clean.py`. Running the same turns on OPT-1
vs a checked-out pre-OPT-1 `hybrid_clean.py` (`6b81154`, the byte-parity baseline):

| turn | OPT-1 first_div vs HF | pre-OPT-1 first_div vs HF | ids A == B | OPT-1 wall | pre-OPT-1 wall |
|---|---|---|---|---|---|
| gt0 ep0/t0 | none | none | yes | 0.81 | 1.79 |
| gt2 ep0/t2 | none | none | yes | 1.04 | 2.41 |
| gt16 ep5/t0 | **33** | **33** | yes | 0.89 | 1.93 |
| gt18 ep5/t2 | **33** | **33** | yes | 0.84 | 1.94 |
| gt19 ep6/t0 | **18** | **18** | yes | 2.22 | 5.45 |
| gt20 ep6/t1 | **33** | **33** | yes | 1.88 | 4.59 |

**Every engine output is byte-identical between OPT-1 and pre-OPT-1** (including the
diverging turns), at **2.36x mean speedup**. => OPT-1 is a pure, behavior-preserving
speedup; it caused **zero** parity change, and the divergences are a property of the
`6b81154` forward, not of the optimization.

## Completed-turn results (44 / 63 turns, greedy)

| metric | engine (44 completed) | HF on the same 44 |
|---|---:|---:|
| s/turn mean | **1.250** | 2.835 |
| s/turn p50 | **1.185** | 2.756 |
| s/turn worst completed | 2.201 (gt19, 85 tok) | — |
| **engine speedup on identical subset** | **2.27x** | — |
| denoise forwards / turn | 40.95 | 39.30 |
| tokens / forward | 1.53 | — |
| n_gen mean / median / max | 59.9 / 58 / 87 | — |
| byte-parity holds | **35 / 44** | — |
| value_projection_events > 0 | 0 / 44 | — |
| verify_invariants ok | 44 / 44 | — |
| engine exact_args | **32 / 44** | 35 / 44 |
| engine valid_tool_call | 43 / 44 | 44 / 44 |
| engine exact == HF exact (per turn) | 41 / 44 | — |

On the **35 byte-parity-holding turns, engine exact_args == HF exact_args exactly
(31 / 31)** — where the forward is byte-exact, quality is HF's by construction. The
3 turns where engine exact != HF (gt12, gt16, gt18; HF-exact -> engine-miss) are all
windowed-probe divergences (`first_div` 65/33/33, proj 0).

**WARNING: 1.250 s/turn is a favorable subset** (mean 59.9 tok): it excludes the 16
long turns (n_ref >= 95) and the 3 stalls. A true full-battery s/turn **cannot be
produced** on this engine until the stall is fixed.

## Byte-parity spot-check — the 3 reference turns

| turn | verdict |
|---|---|
| ep0/t0 (gt0) | **byte-parity 42/42**, first_div none, stop |
| ep2/t0 (gt7) | **byte-parity 36/36**, first_div none, stop |
| ep1/t0 (gt4) | **STALLS** at committed ~ 95 (canvas valid_len 32->13) — never completes; prefix to 94 byte-identical (first_div none) |

The prior acceptance ran ep1/t0 only to a 32-token cap; run to full length it
exposes the partial-canvas stall.

## The assertion the task asked for

`exact_args == 47/63` is **NOT reproduced** on the engine, and byte-parity does not
hold across all 63 turns. This is a genuine deviation — reported per the task's
"stop and report" rule — but it is **not an optimization regression** (OPT-1 is
byte-identical, A/B above). It is the GAP-5A windowed-probe causal approximation
(9 divergent turns) plus 19 uncompletable turns (16 long + 3 short stalls). The
"engine == HF 47/63 by construction" statement in `p2_engine_acceptance_result.md`
held only for the 3 turns it was measured on and does **not** generalize.

## What did not complete (19 / 63)

- **16 long turns** (n_ref >= 95) — skipped for the mass run; each stalls when
  committed crosses ~ 90-95 (verified on ep1/t0, ep1/t1 < 94).
- **3 short stalls**: gt24 ep7/t2 (n_ref 92, plen 1594), gt32 ep10/t0
  (n_ref 34, **plen 1024 = mamba block**), gt46 ep14/t2 (n_ref 91, plen 1753).

## Root-cause diagnostic (grammar is NOT the bottleneck)

Per-step trace of ep1/t0 to the stall (`diag_ep1*.py`): steps to committed 95 are a
**flat 27 ms each**; cumulative grammar time (`legal_candidates` / `_keeps_prefix` /
`native_tool_candidate_token_ids` / `grammar.text` / `truly_forced_token`) =
**0.017 s = 0.7 %** of the 2.6 s spent; no single FSM call > 0.3 s; `decode_model_token`
(the OPT-1 path) never slow. The turn reaches 95/110 tokens in 2.6 s, then the next
forward (canvas `valid_len` 32 -> 13) never returns. => **OPT-5 (incremental
detok/FSM) is confirmed a non-issue at agentic turn lengths**; the blocker is the
engine denoise forward on a partial-width canvas.

## temp = 0.7 rollouts (RL-rollout sanity)

5 seeded rollouts (gt0/7/17/29/51): all `finish=stop`, `valid_tool_call=True`,
`value_projection_events=0`, and **same-seed 2x reproducible** (n_gen + forwards
identical). Wall 0.58-0.90 s (unchanged from greedy). Value distributions are highly
peaked so these collapse to the greedy tokens (documented); the RL contract —
bounded, grammar-valid, zero value-projection, reproducible — holds.

## Honest scoreboard

| row | exact_args | s/turn | fwd-or-tok/turn | note |
|---|---:|---:|---:|---|
| **ENGINE (OPT-1), 44 completed turns** | 32/44 (HF 35/44 on same) | **1.250** (p50 1.185, worst 2.201) | 40.95 denoise fwd | **subset (mean 60 tok); 19 turns uncompletable; parity 35/44** |
| ENGINE, 35 parity-hold turns | **31/31 == HF** | ~1.1 | — | where forward is byte-exact |
| HF hybrid-clean (v2), full 63 | 47/63 | 3.904 | 56.83 denoise fwd | reference |
| HF hybrid-clean, same 44 as engine | 35/44 | 2.835 | 39.30 | matched subset |
| stock-bf16-AR-guided, full 63 | 51/63 | 1.213 | 82.24 tok | AR bar |
| stock-AR aggregate | 124/247 | 0.741 | 49.06 tok | beyond-AR bar |
| **M2 target** | >=55/63 | **< 1.120** | — | **not adjudicated (battery can't complete)** |

The engine's completed-subset s/turn (1.250) already sits at the stock-AR-guided
level (1.213) and is 2.27x under HF — a real OPT-1 win — but M2 stays
**unadjudicated** because the battery cannot be completed and byte-parity is not
universal.

## Remaining hotspots / frontier (evidence-ranked)

1. **OPT-3 — byte-EXACT windowed-BIDIRECTIONAL variable-width single-`[MASK]`
   forward.** Fixes both blockers at once: (a) removes the causal-approximation
   divergence -> restores universal byte-parity -> restores the HF 47/63 quality;
   (b) removes the partial-canvas stall -> the long turns and the full battery can
   run -> a real full-battery s/turn. **Correctness blocker, #1 by a wide margin.**
2. **OPT-4 — incremental KV/GDN 1-token decode** (recurrent kernel + FULL graph):
   the residual GPU-shape gap to AR, only meaningful once OPT-3 lands.
3. **OPT-1 — done and verified** (byte-identical, 2.36x). **OPT-5 — do not do it**
   (0.7 % of time; disproven as a bottleneck).

## Artifacts (`runs/p2_engine_bench/`)

- `build_matched20_ref.py` -> `matched20_ref.json` (63 prompt-verified records).
- `run_battery.py` -> `matched20_turns.jsonl` (44 completed turns, incremental).
- `ab_opt1.py` -> `ab_A_opt1.json` / `ab_B_preopt1.json` (OPT-1 A/B, byte-identical).
- `diag_ep1.py` / `diag_ep1_v2.py` -> `diag_ep1_steps.jsonl` / `diag_ep1_v2.log`
  (stall root-cause: partial-canvas forward, grammar 0.7 %).
- `matched20_temp07.jsonl` / `matched20_temp07b.jsonl` (5 rollouts + reproducibility).
