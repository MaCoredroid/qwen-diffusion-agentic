# Per-Capability Conversion-Tax Table — RESULT (work-item #28)

**Verdict: the per-capability conversion tax is SMALL and BOUNDED on every class — no capability collapses.**
RL+merge is roughly capability-neutral (reasoning −2, code 0, instruction +1); the diffusion engine then costs
−1..−2 vs its own AR-served weights on each class. On the certified N=247 tool-call row the direction is the
opposite for RL+merge (a *gain*, +12), and the engine sits between stock and merged. Engine stability held: **0
CPU-pathological hangs**, `value_projection_events == 0`, all `verify.ok == True`.

Frame: the columns are the conversion pipeline read left-to-right —
**STOCK-AR** (pre-conversion Qwen3.5-9B `c202236`) → **MERGED-AR** (RL-v2 merged weights, the 136/247 `A_new`
export, served plain AR) → **ENGINE-DIFFUSION** (the *same* RL-v2 weights served through the block-diffusion engine,
pin `0b44dcc` hybrid_clean free-text). STOCK-AR and MERGED-AR share the identical offline-vLLM AR path (only the
weight dir differs), isolating "what RL+merge did to capability" from "what the diffusion engine costs on top."
Regime: **B=1 greedy** (temp 0, seed 20260701), strict deterministic scoring, **identical prompts across all three
systems**. Artifacts under `runs/conversion_tax/` (`README.md`, `summary.json`, `report.md`, per-item JSONL, scorers).

## 1. The table (raw exact counts)

| capability class | STOCK-AR | MERGED-AR | ENGINE-DIFFUSION |
|---|---:|---:|---:|
| GSM8K free-CoT (30) | **29/30** | **27/30** | **26/30** |
| CODE / MBPP-25 | **22/25** | **22/25** | **20/25** |
| INSTRUCTION-25 | **21/25** | **22/25** | **21/25** |
| _TOOL-CALL (247 turns, agentic)_ ¹ | 124/247 | 136/247 | 130/247 |

¹ Reference row — from prior certified work (`endgame_scoreboard` / `convert_after_rl` step-3); **not re-run** in
this battery. A C0 merged-AR alternate operating point = 127/247; the 136/247 shown is the promoted `A_new` AR point.

## 2. Per-class tax/gain deltas + binomial (Wilson 95%) CIs — SMALL N, read as bands not points

The N=25–30 cells are small: **every A/B/C 95% interval spans ≈20 points and all three systems overlap within a
class**, so per-class differences of ≤3 items are *within noise* — the table certifies "no collapse," not a ranked
ladder. Deltas below are raw item counts (and percentage points); the tool-call row (N=247) is the only one with
intervals tight enough to separate.

| class | N | STOCK-AR 95% CI | MERGED-AR 95% CI | ENGINE 95% CI | Δ merged−stock | Δ engine−merged | Δ engine−stock |
|---|---:|---|---|---|---|---|---|
| GSM8K | 30 | 97% [83, 99] | 90% [74, 97] | 87% [70, 95] | −2 (−7pp) | −1 (−3pp) | −3 (−10pp) |
| MBPP | 25 | 88% [70, 96] | 88% [70, 96] | 80% [61, 91] | 0 (0pp) | −2 (−8pp) | −2 (−8pp) |
| INSTR | 25 | 84% [65, 94] | 88% [70, 96] | 84% [65, 94] | +1 (+4pp) | −1 (−4pp) | 0 (0pp) |
| TOOL ¹ | 247 | 50% [44, 56] | 55% [49, 61] | 53% [46, 59] | +12 (+5pp) | −6 (−2pp) | +6 (+2pp) |

All A/B/C intervals mutually overlap → differences not individually significant. On the tool-call row the merged-AR
gain (+12/247, CI [49,61] vs stock [44,56]) is the largest and most credible single effect; engine-diffusion (130,
[46,59]) lands *between* stock and merged, consistent with the small per-class engine cost seen on A/B/C.

## 3. Engine-side audit / stability (the L0-fix check)

| engine cell | hangs | length-runaways | `value_projection_events` nonzero | all `verify.ok` |
|---|---:|---:|---:|---:|
| A (GSM8K) | 0 | 0 | 0 | True |
| B (MBPP) | 0 | 1 (idx 7) | 0 | True |
| C (INSTR) | 0 | 0 | 0 | True |

**0 CPU-pathological hangs** across both free-text engine cells swept in this battery (B, C; A reused from the
L0-fixed head). `value_projection_events == 0` across **all** engine cells; all `verify.ok == True`. The single class-B
length-runaway (idx 7) is a normal length-cap finish, not a hang. **The L0 fix held exactly as expected.**

## 4. Honest reading — where conversion+RL costs, gains, is neutral

- **Neutral / gain from RL+merge (STOCK-AR → MERGED-AR).** Code exactly neutral (0), instruction +1, reasoning −2,
  tool-call **+12** — i.e. RL+merge is capability-neutral on the general classes and *buys* the tool-call exactness it
  was trained for. No general-capability regression is detectable above noise.
- **Small bounded cost from the engine (MERGED-AR → ENGINE-DIFFUSION).** −1 (reasoning), −2 (code), −1 (instruction),
  −6/247 (tool). The diffusion engine costs −1..−2 items vs its own AR-served weights on each class; nothing collapses.
- **End-to-end (STOCK-AR → ENGINE-DIFFUSION).** Reasoning −3, code −2, instruction 0, tool **+6** — the full
  conversion is net-positive where it was optimized (tools) and within-noise-negative elsewhere.
- **Small-N caveat is load-bearing.** On N=25–30 the per-class CIs overlap; treat A/B/C as "bounded, no collapse,"
  and lean on the N=247 tool-call row for the one directional claim (RL+merge gains; engine sits between).

## 5. Provenance / scoring (documented + committed before the battery)

- **A (GSM8K)** — the 30 clean L1 prompts (`runs/l1_census/gsm8k_prompts_clean.json`, 5-shot CoT, thinking-off);
  strict last-`#### <number>` match. STOCK-AR reused from `runs/l1_baseline_b1/` (re-scored → 29), ENGINE reused from
  `runs/l0l2_final_head_verify/` (re-scored → 26). MERGED-AR run fresh (`A_merged_ar.jsonl`).
- **B (MBPP-25)** — MBPP-`sanitized` from local HF cache (offline, no download), 3-shot (prompt-split task_ids 2/3/4),
  eval = first 25 test-split by ascending task_id (11…74); scored by exec'ing the extracted code against
  `test_imports + test_list` asserts in a fresh 5 s subprocess. All 25 MBPP reference solutions self-test 25/25.
- **C (INSTRUCTION-25)** — 25 IFEval-style verifiable-constraint prompts constructed locally (`build_sets.py`, 18
  check types), one deterministic machine-check each, scored strict on the full stripped completion (same scorer for
  all three systems).
- Two scaffold fixes caught pre-battery, applied uniformly to all systems: (1) class C uses the Qwen3.5 thinking-off
  scaffold (empty `<think></think>`) — without it zero-shot prompts open a `<think>` block and ramble to the cap;
  (2) `extract_code` was made robust to the engine dropping the *opening* code fence (it emits `python\ndef…\n` ``` ```)
  — a pure extraction artifact (engine looked like 0/25 before the fix; real value 20/25).

**Artifacts** (all under `runs/conversion_tax/`): `README.md`, `prompt_sets_manifest.json`, `code_prompts.json`,
`instr_prompts.json`, `scoring.py`, `build_sets.py`, `run_ar_cell.py`, `run_engine_cell.py`, `reboot_cell.sh`,
per-item JSONL for the 7 fresh cells + the 2 reused, `aggregate.py`, `summary.json`, `report.md`.
