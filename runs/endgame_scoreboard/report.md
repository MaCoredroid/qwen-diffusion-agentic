# Endgame Scoreboard

Rows are the user-defined comparison set with the added stock FP8 control: stock bf16 guided, stock FP8 guided, merged-AR guided, and our best diffusion hybrid-clean system.

- Stock model: `/home/mark/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- Stock precision/runtime: vLLM bf16 guided decoding and vLLM `--quantization fp8` guided decoding. The bf16 stock row is not NVFP4; NVFP4 was only the 27B teacher context.
- Hybrid selected: `v2_hybrid_clean`.
- Selection rule: highest aggregate exact_args among promoted/accepted, retention-valid, zero-value-projection hybrid candidates.
- Stock bf16 matched-20 exact_args is 51/63; merged-AR guided is 50/63. The maintains-AR bar rises to the stock result.
- Quant tax/speedup: FP8-vs-bf16 deltas are reported in `summary.json` under `quant_comparison`.
- Wall-clock: the HF hybrid-clean row runs on the HF stack (slower than vLLM AR here); the **P2 engine row now closes that column** — the same hybrid-clean (v2) system served on the vLLM FLARE engine (pin `95d8b47`, batch=1) reproduces HF exact_args EXACTLY on all three slices and is faster than every AR baseline on the aggregate. Byte-parity certificate: 233/247 aggregate (14 breaks, all quality-neutral bf16 GDN-fold fp-residue, 0 structural; strict 247/247 gate NOT met ⇒ code default OFF). Final assembled table: `endgame_table_final.md` (repo root).

## Matched-20

| row | exact_args | episode_exact | valid | sec/turn | forwards-or-steps/turn | runtime |
|---|---:|---:|---:|---:|---:|---|
| stock-bf16-AR-guided | 51/63 | 14/20 | 63/63 | 1.213 | 82.24 decode tokens/turn | vLLM bf16 guided |
| stock-FP8-AR-guided | 51/63 | 14/20 | 63/63 | 1.399 | 77.11 decode tokens/turn | vLLM fp8 guided |
| merged-AR guided | 50/63 | 13/20 | 63/63 | 1.158 | 77.62 decode tokens/turn | vLLM bf16 guided |
| OUR SYSTEM hybrid-clean (v2_hybrid_clean) | 47/63 | 13/20 | 63/63 | 3.904 | 56.83 denoise forwards/turn | HF diffusion hybrid-clean |
| OUR ENGINE hybrid-clean (v2_hybrid_clean) | 47/63 | 13/20 | 63/63 | 1.053 | 56.86 denoise forwards/turn | vLLM P2 engine (FLARE pin 95d8b47) — byte-parity 62/63 (gt44 fp-residue) |

## Never-Train BFCL/API-Bank

| row | exact_args | episode_exact | valid | sec/turn | forwards-or-steps/turn | runtime |
|---|---:|---:|---:|---:|---:|---|
| stock-bf16-AR-guided | 73/184 | 19/60 | 184/184 | 0.579 | 37.70 decode tokens/turn | vLLM bf16 guided |
| stock-FP8-AR-guided | 78/184 | 19/60 | 184/184 | 0.743 | 40.06 decode tokens/turn | vLLM fp8 guided |
| merged-AR guided | 77/184 | 19/60 | 184/184 | 0.596 | 39.05 decode tokens/turn | vLLM bf16 guided |
| OUR SYSTEM hybrid-clean (v2_hybrid_clean) | 83/184 | 19/60 | 184/184 | 2.123 | 24.62 denoise forwards/turn | HF diffusion hybrid-clean |
| OUR ENGINE hybrid-clean (v2_hybrid_clean) | 83/184 | 19/60 | 184/184 | 0.480 | 24.06 denoise forwards/turn | vLLM P2 engine (FLARE pin 95d8b47) — byte-parity 171/184 (13 fp-residue) |

## Aggregate

| row | exact_args | episode_exact | valid | sec/turn | forwards-or-steps/turn | runtime |
|---|---:|---:|---:|---:|---:|---|
| stock-bf16-AR-guided | 124/247 | 33/80 | 247/247 | 0.741 | 49.06 decode tokens/turn | vLLM bf16 guided |
| stock-FP8-AR-guided | 129/247 | 33/80 | 247/247 | 0.910 | 49.51 decode tokens/turn | vLLM fp8 guided |
| merged-AR guided | 127/247 | 32/80 | 247/247 | 0.739 | 48.89 decode tokens/turn | vLLM bf16 guided |
| OUR SYSTEM hybrid-clean (v2_hybrid_clean) | 130/247 | 32/80 | 247/247 | 2.577 | 32.84 denoise forwards/turn | HF diffusion hybrid-clean |
| OUR ENGINE hybrid-clean (v2_hybrid_clean) | 130/247 | 32/80 | 247/247 | 0.626 | 32.43 denoise forwards/turn | vLLM P2 engine (FLARE pin 95d8b47) — byte-parity 233/247 (14 fp-residue, 0 structural) |

**Engine vs baselines (aggregate 247):** exact **130/247 == HF hybrid-clean 130/247 EXACTLY** (and +6 vs stock-bf16-AR, +1 vs stock-FP8, +3 vs merged-AR); speed **0.626 s/turn** beats stock-bf16-AR (1.18×), stock-FP8 (1.45×), merged-AR (1.18×), HF hybrid (4.12×). Byte-parity 233/247 (cold-config 235/247); the 14 breaks are quality-neutral bf16 GDN-fold fp-residue (0 structural, proj==0, `eng_exact==hf_exact`), so exact stays 130 in every config. Strict 247/247 promotion gate NOT met ⇒ default OFF. Sources: `runs/p2_engine_battery_v3b/` (matched-20) + `runs/p2_engine_nevertrain/` (never-train); assembled `endgame_table_final.md`.

## Stock FP8 Quantization

| slice | exact_args delta | sec/turn delta | FP8 speedup vs bf16 |
|---|---:|---:|---:|
| matched20 | +0 | +0.186 | 0.867x |
| nevertrain | +5 | +0.164 | 0.779x |
| aggregate | +5 | +0.170 | 0.814x |

## Artifacts

- Stock bf16 AR-guided root: `/home/mark/qwen_diffusion/runs/endgame_stock_qwen35_ar_guided/bf16`
- Stock FP8 AR-guided root: `/home/mark/qwen_diffusion/runs/endgame_stock_qwen35_ar_guided/fp8`
- v6 gates root: `/home/mark/qwen_diffusion/runs/rl_multiturn_grpo_v6/from_v2_hybrid_mixed35_kl005_g4_step300_gates`
- Merged-AR rows: `runs/hybrid_broaden_nevertrain_v2/.../ar-vllm-guided/turns.jsonl`
- Hybrid rows: selected from v2/v6 retention-valid hybrid-clean artifacts.
