# Per-capability conversion-tax table (#28)

Raw exact counts, B=1 greedy (temp 0, seed 20260701), strict deterministic scoring,
identical prompts across all three systems. Two class-A cells reused (see Reuse).

| capability class | STOCK-AR | MERGED-AR | ENGINE-DIFFUSION |
|---|---|---|---|
| **GSM8K free-CoT (30)** | 29/30 | 27/30 | 26/30 |
| **CODE / MBPP (25)** | 22/25 | 22/25 | 20/25 |
| **INSTRUCTION (25)** | 21/25 | 22/25 | 21/25 |
| _TOOL-CALL (247 turns, agentic)_ ¹ | 124/247 | 136/247 | 130/247 |

¹ reference row — reference row, from prior certified work (endgame_scoreboard / convert_after_rl step3); not re-run in this battery. C0 merged-AR alt point = 127/247.

Columns are the conversion pipeline: **STOCK-AR** (pre-conversion baseline) → **MERGED-AR** (RL-v2 merged weights served plain AR — the 136/247 export) → **ENGINE-DIFFUSION** (the same RL-v2 weights served through the block-diffusion engine).

## Per-cell detail

| class | system | correct | finish reasons | wrong idxs | reused | source |
|---|---|---|---|---|---|---|
| A | STOCK-AR | 29/30 | stop:30 | [12] | yes | `runs/l1_baseline_b1/ar_gsm8k_clean.jsonl` |
| A | MERGED-AR | 27/30 | stop:29, length:1 | [12, 13, 21] | no | `runs/conversion_tax/A_merged_ar.jsonl` |
| A | ENGINE-DIFFUSION | 26/30 | stop:30 | [2, 7, 12, 13] | yes | `runs/l0l2_final_head_verify/engine_gsm8k_clean_head.jsonl` |
| B | STOCK-AR | 22/25 | stop:25 | [2, 18, 24] | no | `runs/conversion_tax/B_stock_ar.jsonl` |
| B | MERGED-AR | 22/25 | stop:25 | [2, 7, 18] | no | `runs/conversion_tax/B_merged_ar.jsonl` |
| B | ENGINE-DIFFUSION | 20/25 | stop:24, length:1 | [2, 7, 14, 18, 20] | no | `runs/conversion_tax/B_engine.jsonl` |
| C | STOCK-AR | 21/25 | stop:25 | [0, 3, 20, 23] | no | `runs/conversion_tax/C_stock_ar.jsonl` |
| C | MERGED-AR | 22/25 | stop:25 | [3, 20, 23] | no | `runs/conversion_tax/C_merged_ar.jsonl` |
| C | ENGINE-DIFFUSION | 21/25 | stop:25 | [0, 3, 19, 20] | no | `runs/conversion_tax/C_engine.jsonl` |

## Engine-side audit / stability

- **class A**: hangs [], length-runaways [], value_projection_events nonzero [], all verify.ok True.
- **class B**: hangs [], length-runaways [7], value_projection_events nonzero [], all verify.ok True.
- **class C**: hangs [], length-runaways [], value_projection_events nonzero [], all verify.ok True.

**Stability summary:** 0 hangs total across the two free-text engine cells (B, C) — L0 fix held. value_projection_events: 0 across all engine cells; all verify.ok: True.

## Reuse

- **A / STOCK-AR** reused from `runs/l1_baseline_b1/ar_gsm8k_clean.jsonl` (same 30 clean GSM8K prompts, same offline-vLLM greedy).
- **A / ENGINE** reused from `runs/l0l2_final_head_verify/engine_gsm8k_clean_head.jsonl` (pin 0b44dcc free-text head).
- All other seven cells run fresh here. Scoring for every cell (reused included) is recomputed by `scoring.py`.
