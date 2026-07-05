# P2 Engine Battery — NEVER-TRAIN slice (BFCL/API-Bank, 184 turns): exact 83/184 == HF, aggregate-247 completes the endgame battery (2026-07-04)

vLLM pin **`95d8b47`** (`qwen3_5-flare-modelstate`, OPT-4 Stage 1+2+3 landed; code default
**OFF**), export `qwen3.5-9b-fastdllm-rlv2-vllm-bf16` (block/canvas 32, mamba 1024, align+APC),
RTX 5090, RAM cage. FINAL engine = `VLLM_FLARE_BIDIR_PROBE=1` + PIECEWISE cudagraph
`VLLM_FLARE_CUDAGRAPH=1` — the **exact v3b configuration**, byte-identical harness
(`run_battery_v3b.py`). Greedy, temp 0, seed 20260701, uncapped (`max_tokens = n_ref + 16`),
chunked foreground (ep0-19 / ep20-39 / ep40-59), per-turn JSONL. This is the missing
NEVER-TRAIN slice of the full 247-turn endgame battery; the matched-20 slice was v3b
(`runs/p2_engine_battery_v3b/`).

Reference = the HF **hybrid-clean (v2_hybrid_clean)** never-train row, the same 83/184 system on
the endgame scoreboard:
`runs/hybrid_broaden_nevertrain_v2/nevertrain_bfcl_apibank60/diffusion_hybrid_forced_grammar_seq_values/turns.jsonl`
(184 turns, 83 exact, 184 valid). Per-turn reference token ids are stored there
(`generated_token_ids`), so **no HF-bridge regeneration was needed** — every reference
sequence came straight from the stored HF ids.

## Prompt-hash matching discipline (done BEFORE any engine decode)

`build_nevertrain_ref.py` reconstructs all 184 prompts by teacher-forcing each HF row's own
`generated_token_ids` + stored `tool_response_payload` through the identical hybrid-clean
multi-turn loop (`eval_flare_northstar_hybrid_clean.run_hybrid`), then **verifies**
`sha256_text(prompt) == hf_row.prompt_sha256` and `len(prompt_ids) == hf_row.prompt_tokens`
for every turn:

| check | result |
|---|---|
| records reconstructed | **184/184** (60 episodes, ep0-59) |
| `ALL_prompt_sha256_match` | **true** (184/184) |
| `ALL_prompt_tokens_match` | **true** (184/184) |
| HF exact / valid embedded | 83/184 · 184/184 |
| total ref tokens | 7204 |

No approximate prompts: every engine decode ran on a byte-identical reconstruction of the prompt
the HF backend actually saw.

## Verdict

| step | verdict | one-line |
|---|---|---|
| all 184 never-train turns complete | **PASS** | zero stalls, mean **0.480 s/turn** |
| **exact_args == HF 83/184** | **MET (exactly 83)** | per-turn exact verdict == HF on **184/184** turns (0 wins, 0 losses) |
| valid tool call | **MET (184/184)** | ties HF |
| byte-parity vs HF hybrid-clean | **171/184** | 13 breaks, **all quality-neutral** (fp-residue class; 0 structural) |
| audit: value_projection_events == 0 | **CLEAN** | 184/184 turns proj 0 |
| audit: zero_forward_rows == 0 | **CLEAN** | 0 rows |
| audit: verify_invariants | **CLEAN** | 184/184 ok; all finish_reason == stop |
| cold-prefix parity certificate (13 breaks) | **PASS** | 10/13 break identically cold (deterministic fp-residue); 2/13 = APC artifact (parity restored cold); 1/13 same-fd path-sensitive tail |
| temp=0.7 OOD RL sanity (3 rollouts) | **PASS** | 3/3 bounded, valid, proj0, verify-ok, byte-reproducible (a==b) across BFCL-multi / BFCL-AST / API-Bank |

## Numbers — never-train (184, greedy, uncapped)

| | ENGINE (this run) | HF hybrid-clean (ref) | stock-bf16-AR | merged-AR |
|---|---:|---:|---:|---:|
| exact_args | **83/184 (== HF)** | 83/184 | 73/184 | 77/184 |
| valid | **184/184** | 184/184 | 184/184 | 184/184 |
| byte-parity | **171/184** | (self) | — | — |
| s/turn mean | **0.480** | 2.123 | 0.579 | 0.596 |
| s/turn p50 / p90 | 0.427 / 0.808 | — | — | — |
| s/turn worst | 1.448 (gt176) | — | — | — |
| TRUE denoise fwd/turn | **24.06** | 24.62 | 37.70 tok | 39.05 tok |
| per-forward ms (cudagraph) | 20.91 | — | — | — |

Per-family (engine exact == HF exact in **every** family): API-Bank-Lv1 7/7, API-Bank-Lv2 4/4,
BFCL-AST 12/12, BFCL-multi_turn 60/60. Byte-parity by family: BFCL-AST 12/12, API-Bank-Lv1 13/13,
API-Bank-Lv2 11/12, BFCL-multi_turn 135/147 — the fp-residue breaks concentrate in the ambiguous
file-system multi-turn scenarios.

### Never-train speed bars (engine 0.480 s/turn) — engine BEATS every AR baseline here

- **HF hybrid-clean 2.123 -> BEAT** (0.226x, **4.42x** speedup).
- **stock-bf16-AR 0.579 -> BEAT** (0.829x, 1.21x). *Unlike matched-20, the engine beats stock AR
  on this slice: never-train turns take ~24 forwards/turn (vs 57 on matched-20), so the batch-1
  weight-stream cost per turn is far lower and the diffusion turn undercuts the longer AR decode.*
- **stock-FP8-AR 0.743 -> BEAT** (1.55x). **merged-AR 0.596 -> BEAT** (1.24x).

## Byte-parity break decomposition (13 breaks) — all quality-neutral, 0 structural

Every break diverges at a **model-chosen value/name token** (the grammar-structural scaffold
`<tool_call>` / `<function=` / `<parameter=...>` / `>` always matches); `proj == 0` and
`verify == ok` on all 13; **`eng_exact == hf_exact` on all 13** (both non-exact — the fp
perturbation flips one already-wrong near-tie token, never the exact_arguments verdict). The
cold-prefix (`reset_prefix_cache` per turn) certificate on all 13 splits them:

| class | count | turns | meaning |
|---|---:|---|---|
| deterministic fp-residue (gt44 class) | **10** | 10,32,55,70,74,91,108,124,131,179 | break **byte-identically** under cold prefix -> intrinsic bf16 GDN-fold fp gap, path-invariant |
| APC cross-turn artifact | **2** | 16,130 | **byte-parity RESTORED** under cold prefix -> a `_image`/`\n` mirror near-tie contaminated across turns by the prefix cache; not intrinsic |
| fp-residue, path-sensitive tail | **1** | 36 | same first-divergence token (fd=27, `result` -> `diff`) under both, tail length differs |
| **structural / grammar** | **0** | — | none |

Illustrative divergences (engine token <-> HF ref token, both non-exact): `cd`<->`cp` (4x, file-system
command near-tie), `.txt`<->`\n`, `diff`<->`result`, `.`<->`..`, `Get`<->`Add` (API-Bank-Lv2). Under a
cold-prefix (fresh-context) config byte-parity would be **173/184**; the two APC turns are still
non-exact either way, so **exact_args stays 83/184 regardless of APC**.

Cold-prefix certificate audit: proj==0 all 13, verify ok all 13, cold exact count 0/13 = APC-on
0/13 (exact_args APC-invariant).

## temp=0.7 rollouts on never-train prompts (RL contract on OOD)

3 prompts (one per family: gt0 BFCL-multi, gt147 BFCL-AST, gt159 API-Bank-Lv1), two seeded passes
each (seed 20260701), full engine token ids stored:

| gt | family | bounded | valid | proj0 | verify | byte-repro (a==b) | n_gen/maxtok | fwd |
|---:|---|---|---|---|---|---|---|---:|
| 0 | BFCL-multi_turn | yes | yes | yes | yes | yes | 25/41 | 12 |
| 147 | BFCL-AST | yes | yes | yes | yes | yes | 54/70 | 37 |
| 159 | API-Bank-Lv1 | yes | yes | yes | yes | yes | 37/53 | 21 |

All bounded (finish==stop, no runaway), valid tool calls, zero value projection, invariants ok, and
**byte-reproducible under a fixed seed** — the RL sampling contract holds on out-of-distribution
never-train inputs.

## Aggregate-247 ENGINE — completes the endgame battery (matched-20 v3b + never-train)

| slice | exact_args | valid | byte-parity | s/turn mean | fwd/turn |
|---|---:|---:|---:|---:|---:|
| matched-20 (v3b) | 47/63 | 63/63 | 62/63 | 1.053 | 56.86 |
| never-train (this) | 83/184 | 184/184 | 171/184 | 0.480 | 24.06 |
| **AGGREGATE-247 ENGINE** | **130/247** | **247/247** | **233/247** | **0.626** | **32.43** |
| HF hybrid-clean (ref) | 130/247 | 247/247 | (self) | 2.577 | 32.84 |

**exact_args 130/247 == HF hybrid-clean 130/247 EXACTLY**, valid 247/247, on the same 247-turn mix.

### Aggregate-247 speed bars (engine 0.626 s/turn, apples-to-apples turn mix)

- **HF hybrid-clean 2.577 -> BEAT** (0.243x, **4.12x** speedup).
- **stock-bf16-AR-agg 0.741 -> BEAT** (0.845x, 1.18x). **This closes the v3b "stock-agg 0.741 MISS".**
  That MISS compared the matched-20-only engine mean (1.053) to the stock *aggregate* — a
  different, shorter turn mix. Over the **identical** 247-turn mix the engine aggregate (0.626)
  **beats** stock-AR aggregate.
- **stock-FP8-AR-agg 0.910 -> BEAT** (1.45x). **merged-AR-agg 0.739 -> BEAT** (1.18x).

Caveat: AR baselines were measured under the endgame-scoreboard vLLM-guided server harness; the
engine numbers are the FLARE adapter at batch=1. Turn mixes are now identical (247), so the
aggregate comparison is apples-to-apples on workload; the residual caveat is harness/batch.

## What this establishes / does not

- **Establishes:** the FINAL post-fix engine reproduces the HF hybrid-clean never-train row on
  quality **exactly** — exact 83/184, valid 184/184, per-turn exact verdict == HF on 184/184
  (0 wins / 0 losses), proj 0, invariants clean — while running **0.480 s/turn** (4.42x over HF,
  and faster than stock/merged AR on this slice). Completing the 247-turn battery gives engine
  **exact 130/247 == HF 130/247** at **0.626 s/turn**, which now **beats stock-AR aggregate speed
  (0.741)** — the engine is both faster than stock AR aggregate and quality-identical to the
  served HF hybrid row.
- **Does NOT change** the strict byte-parity promotion gate: byte-parity is 171/184 (233/247
  aggregate). The 13 never-train breaks are the **same fp-residue class as matched-20 gt44** —
  deterministic bf16 GDN-fold fp gaps at model-chosen near-tie tokens (10), APC cross-turn
  near-ties (2), one path-sensitive tail (1); **0 structural**, all quality-neutral (both engine
  and HF non-exact, exact_args unchanged). Byte-parity remains the kernel-level "engine == HF by
  construction" certificate; it is quality-neutral, not a quality gain.

## Artifacts (`runs/p2_engine_nevertrain/`)

`nevertrain_ref.json` (184, prompt-hash-verified) . `nevertrain_turns.jsonl` (184 greedy) .
`nevertrain_parity_cert_resetapc.jsonl` (13, cold-prefix certificate) .
`nevertrain_temp07.jsonl` (3 OOD rollouts, 2 seeded passes) . `aggregate.json` (full break
diagnosis) . `build_nevertrain_ref.py` . `aggregate.py` . `temp07_rollouts.py` . `env.sh` .
`runcage.sh` / `runcage_t07.sh` . `chunk1.log` / `chunk2.log` / `chunk3.log` / `cert.log` /
`temp07.log`. Harness reused verbatim: `runs/p2_engine_battery_v3b/run_battery_v3b.py`.
