# S2 pilot — EVAL adjudication (spec `s2_pilot_design.md` @ 9ce9445)

**Date:** 2026-07-05 · **Repo HEAD:** 9ce9445 (`main`, in sync with origin/main) · **Author:** S2 EVAL
adjudication pass (CPU-only; no GPU process spawned).

## VERDICT: BLOCKED — NOT ADJUDICABLE (neither PASS nor KILL)

The pilot has not run. The trained adapter `A_S2`
(`runs/s2_pilot/Apilot_step400_seed90101`) **does not exist**, so none of the pre-registered
measurements (a K-gate, b retention, c tool-call, d audits) can be produced. Per the spec's evidence
discipline, **emitting any a/b/c/d number now would be fabrication — forbidden.** This document records
the honest state, the two hard blockers, and the verified preconditions, so the pilot can be executed
exactly when the blockers clear.

- **This is NOT a KILL.** KILL-a is a *measured* claim — "after the full 400-step budget, A_S2 K=2 loses
  >2 items vs K=1 on the 30-set." With no adapter and no measurement, declaring the K-factor a wall would
  invent a scientific conclusion the data does not support. The 5×-vs-AR claim is therefore **UNRESOLVED,
  not retired.**
- **This is NOT a PASS.** PASS requires measured `tok/fwd ≥ 2.0` at held accuracy on the trained adapter.
  No adapter exists. L3 is **not funded.**

## Why the pilot cannot run — two hard blockers

**Blocker 1 — no corpus, therefore no adapter (upstream data step stalled).**
The self-trajectory corpus is far below even the reduced yield floor:

| artifact | count | spec floor (§4) |
|---|---:|---|
| `runs/s2_pilot/train_gen.jsonl` (raw generations) | **66** | ~2,200 target |
| `runs/s2_pilot/s2_traj_corpus.jsonl` (audit-clean trajectories) | **31** | ~1,000 target; **700 hard floor** |

The gen loop is not alive (`ps` shows no `run_s2_gen`/`gen_loop`; GPU idle at 2,243 MiB; `gen_iter.log`
frozen at `idx 65 … fin=stop`). Training cannot start below the 700-trajectory floor, so `A_S2` cannot be
built. **The data step (Call #1's job) must be restarted, or the whole pilot stays blocked.**

**Blocker 2 — two eval scripts are missing.** The battery below references scripts that do not exist in
the tree:
- **`scripts/eval_flare_freetext_cad.py`** — the new entropy-gated CAD sampler
  (`adaptive_k_sample_one`, spec §3) that rows a/CTRL-decode/CTRL-K1 all call. **Must be authored**, and
  its §3 acceptance test (reproduce the K=1 free-text baseline **byte-exactly**: 26/30, 0.862 tok/fwd)
  must PASS before any tok/fwd number it emits is admissible.
- **`export_qwen35_9b_fastdllm_vllm.py`** — **not found anywhere in the tree**, and **no file matches the
  pinned sha `6d507ec9…`** it is referenced by. (Needed only if any row is run engine-side; the a/b/c
  rows here are HF-stack per spec §6, so this blocks only an engine cross-check, not the primary battery.)

## Preconditions I re-verified this pass (real, CPU-only)

| check | result |
|---|---|
| Repo HEAD == spec commit | `9ce9445` ✓ |
| `eval_flare_stage1_ab_diffusion.py` sha256 (retention driver, §6) | `eaa78d7a…b5e503` ✓ matches |
| `eval_flare_northstar_hybrid_clean.py` sha256 (tool-call, §6) | `a4c66751…b908b1f3` ✓ matches |
| `audit_value_projection_tokens.py` sha256 (§6) | `7b203e3e…a620b40` ✓ matches |
| Gate set `runs/l1_census/gsm8k_prompts_clean.json` | list, **len 30** ✓ (the 26/30 anchor set) |
| **KILL-0 base merge-gate** — `models/qwen3.5-9b-fastdllm-mtplus1-merged/config.json` | `mask_token_id=248077`, `bd_size=32` ✓ (half PASS; free-text sanity episode is a GPU step, unrun) |
| `.venv-fastdllm` present (training / HF eval env) | ✓ |
| Anchor `runs/l0l2_final_head_verify/summary.json` (pin `0b44dcc`) | 26/30 strict · 0.862 emitted tok/fwd · model_chosen_K=1.0 · value_projection_events=0 · verify_ok_all=true · eng_exact_eq_hf_all path intact ✓ |

## The battery — pre-registered, execute-ready, all rows PENDING an adapter

Every a/b/c/d row below requires `A_S2`. No numbers are entered because none exist. The anchors and the
verbatim §8 thresholds are transcribed so the adjudication is fully specified in advance.

| # | measurement | anchor | PASS threshold | KILL threshold | **measured** |
|---|---|---|---|---|---|
| a | K-gate: A_S2 K=2 tok/fwd **and** accuracy (30-prompt clean, free-CoT) | 0.862 tpf · 26/30 | `tok/fwd ≥ 2.0` AND net-loss `b−c ≤ 2` AND McNemar `p ≥ 0.05` | net-loss `> 2` | **PENDING — no A_S2** |
| b | GSM8K retention, legacy full-context, N=20 | 13/20 | `≥ 13/20` | `≤ 11/20` | **PENDING — no A_S2** |
| c | tool-call spot-check, 10 matched hybrid-clean turns | C0-10 | `10/10 exact vs C0` | `≥ 2 lost` | **PENDING — no A_S2** |
| d | value-projection audits (every turns.jsonl) | 0 | all counters 0 | any nonzero ⇒ KILL-3 | **PENDING — no turns.jsonl** |
| — | training delta vs CTRL-decode | — | A_S2 K=2 > CTRL-decode at held acc | CTRL-decode `≥2.0` held ⇒ "decode-only" | **PENDING** |

**Adjudication rule when run (§8, verbatim):** PILOT PASS = a∧b∧c PASS ∧ d clean ∧ training-delta
positive ⇒ fund the full S2 build. PILOT KILL = a KILL **OR** b KILL **OR** c KILL (d clean) ⇒
reasoning-span K is a wall (or the pilot damaged a certified capability).

## The honest speed story as it stands today (unchanged by this pass; nothing measured to move it)

The pilot exists precisely because the 5× north star is stuck on **one unmoved factor**: model-chosen
reasoning tokens commit at **K = 1.0**. The final-head-verified anchor (pin `0b44dcc`,
`runs/l0l2_final_head_verify/`) is **0.862 emitted tok/fwd at K=1, 26/30 GSM8K, 30/30 clean stop, 0
value-projection events**. The corrected B=1 equation on reasoning content is
`0.86 × (10.72 AR-cudagraph ÷ 25.8 engine-free-text) = 0.36×` vs AR-cudagraph (0.47× vs eager) —
**distance to 5× ≈ 14×, entirely in the K factor.** L2 per-forward parity (25.8→~13 ms) buys at most ~2×
and is still K-bound. **Only L3 (this pilot: S2 consistency-distillation + entropy-gated adaptive K) can
raise reasoning K above 1 — and whether it can is exactly what remains UNTESTED.**

## To unblock (ordered)

1. Restart the data step (Call #1): drive `train_gen.jsonl` to ~2,200 raw → audit-filter to ≥700 clean
   trajectories (yield-floor branch), with the §4 leakage-dedupe = 0 gate/retention collisions.
2. Author `scripts/eval_flare_freetext_cad.py::adaptive_k_sample_one` and PASS its §3 byte-exact K=1
   baseline acceptance test (26/30, 0.862 tok/fwd) before admitting any tok/fwd it emits.
3. Train `A_S2` (chunked, resume-safe, KL-to-base ≤ 0.05 early-stop) → then run this battery, one caged
   GPU row at a time, and adjudicate per §8.
