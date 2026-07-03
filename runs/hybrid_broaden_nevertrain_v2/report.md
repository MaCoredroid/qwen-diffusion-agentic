# Hybrid Clean Broadened Eval

Status: complete.

Default serving mode under test: `diffusion_hybrid_forced_grammar_seq_values`.

Adapter: `runs/rl_multiturn_grpo_v2/from_selected_base_g4_step300/adapter_model`

## Slices

| slice | source | episodes | turns |
|---|---|---:|---:|
| matched-20 | held-out native matched battery | 20 | 63 |
| never-train | BFCL/API-Bank eval-only battery | 60 | 184 |

The matched-20 AR-guided row was rerun fresh in this directory. The matched-20 careful and hybrid rows reuse the already-audited v2 promotion artifacts. The never-train AR-guided, careful, and hybrid rows were all rerun in this directory.

AR-guided launch used vLLM `0.23.0`, FR13/APC prefix cache, `max_model_len=4096`, `mamba_block_size=1024`, `gdn_prefill_backend=triton`, and `gpu_memory_utilization=0.68` because `gnome-shell` occupied about 9.6 GiB on the 5090. Quality is the main AR-guided control here; diffusion wall-speed comparisons are against the HF careful row.

## Three-Way Tables

### Matched-20

| backend | exact_args | episode exact | valid | exact_seq | schema_ok | sec/turn | forwards/turn | note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| AR-guided vLLM | 50/63 | 13/20 | 63/63 | 63/63 | 59/63 | 1.158 | n/a | fresh rerun |
| diffusion careful | 44/63 | 11/20 | 62/63 | 57/63 | 59/63 | 6.686 | 95.51 | v2 promotion row |
| diffusion hybrid-clean | 47/63 | 13/20 | 63/63 | 63/63 | 59/63 | 3.904 | 56.83 | audited zero value projection |

Paired exact-args:

- Hybrid vs careful: `+4/-1`, net `+3/63`.
- Hybrid vs AR-guided: `+0/-3`, net `-3/63`.

Speed:

- Hybrid vs careful wall: `6.686 / 3.904 = 1.71x`.
- Hybrid vs careful forwards: `95.51 / 56.83 = 1.68x`.

### Never-Train BFCL/API-Bank

| backend | exact_args | episode exact | valid | exact_seq | schema_ok | sec/turn | forwards/turn | note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| AR-guided vLLM | 77/184 | 19/60 | 184/184 | 126/184 | 184/184 | 0.596 | n/a | fresh rerun |
| diffusion careful | 77/184 | 16/60 | 173/184 | 121/184 | 165/184 | 3.373 | 43.62 | fresh rerun |
| diffusion hybrid-clean | 83/184 | 19/60 | 184/184 | 133/184 | 179/184 | 2.123 | 24.62 | fresh rerun, audited zero value projection |

Paired exact-args:

- Hybrid vs AR-guided: `+6/-0`, net `+6/184`.
- Hybrid vs careful: `+7/-1`, net `+6/184`.
- Careful vs AR-guided: `+5/-5`, net `0/184`.

Speed:

- Hybrid vs careful wall: `3.373 / 2.123 = 1.59x`.
- Hybrid vs careful forwards: `43.62 / 24.62 = 1.77x`.
- Hybrid remains slower than AR-guided vLLM wall on this engine path: `2.123 / 0.596 = 3.56x`.

## Never-Train Source Breakdown

| backend | source | exact_args | episode exact | valid | exact_seq | sec/turn | forwards/turn |
|---|---|---:|---:|---:|---:|---:|---:|
| AR-guided | API-Bank-Lv1 | 7/13 | 7/13 | 13/13 | 13/13 | 0.916 | n/a |
| AR-guided | API-Bank-Lv2 | 4/12 | 4/12 | 12/12 | 9/12 | 1.057 | n/a |
| AR-guided | BFCL-AST | 12/12 | 8/8 | 12/12 | 12/12 | 0.664 | n/a |
| AR-guided | BFCL-multi_turn | 54/147 | 0/27 | 147/147 | 92/147 | 0.524 | n/a |
| careful | API-Bank-Lv1 | 5/13 | 5/13 | 13/13 | 13/13 | 5.038 | 59.69 |
| careful | API-Bank-Lv2 | 3/12 | 3/12 | 11/12 | 10/12 | 5.169 | 61.33 |
| careful | BFCL-AST | 12/12 | 8/8 | 12/12 | 12/12 | 3.233 | 40.92 |
| careful | BFCL-multi_turn | 57/147 | 0/27 | 137/147 | 86/147 | 3.090 | 40.97 |
| hybrid-clean | API-Bank-Lv1 | 7/13 | 7/13 | 13/13 | 13/13 | 3.901 | 45.54 |
| hybrid-clean | API-Bank-Lv2 | 4/12 | 4/12 | 12/12 | 10/12 | 4.290 | 51.67 |
| hybrid-clean | BFCL-AST | 12/12 | 8/8 | 12/12 | 12/12 | 2.124 | 24.33 |
| hybrid-clean | BFCL-multi_turn | 60/147 | 0/27 | 147/147 | 98/147 | 1.789 | 20.59 |

## Audit

Never-train hybrid projection audit:

- `zero_projected_value_tokens_verified=1`
- `projected_value_tokens_exact=0`
- `parallel_commit_forced_tokens_counter=0`
- `wave1_projected_tokens=0`
- `wave1_value_tokens_counter=0`
- `wave2_forced_tokens_counter=0`
- `zero_forward_rows=0`
- `reported_model_value_tokens=2932`
- `true_xml_value_tokens=1397`

The matched-20 hybrid audit was already committed with the promotion run and reports the same zero-projection/zero-force invariant.

## Verdict

Hybrid-clean holds on unleakable data as a label-free constrained-lane serving win. The absolute exact rate is lower on BFCL/API-Bank because the slice is harder, but baseline-relative quality improves: `83/184` beats both AR-guided and careful at `77/184`, with a clean `+6/-0` paired delta versus AR-guided. It also preserves AR-guided episode exactness (`19/60`) while matching its validity (`184/184`).

The speedup against diffusion careful also holds, but is smaller than matched-20: `1.59x` wall and `1.77x` forwards on never-train, versus `1.71x` wall and `1.68x` forwards on matched-20. The remaining gap is value-length dominated, not structure dominated, which keeps DFlash/MTP-style engine work relevant for the AR-vLLM wall-clock gap.

## Files

- Matched-20 fresh AR-guided rows: `runs/hybrid_broaden_nevertrain_v2/matched20/ar-vllm-guided/turns.jsonl`
- Never-train AR-guided rows: `runs/hybrid_broaden_nevertrain_v2/nevertrain_bfcl_apibank60/ar-vllm-guided/turns.jsonl`
- Never-train careful rows: `runs/hybrid_broaden_nevertrain_v2/nevertrain_bfcl_apibank60/diffusion_careful/turns.jsonl`
- Never-train hybrid rows: `runs/hybrid_broaden_nevertrain_v2/nevertrain_bfcl_apibank60/diffusion_hybrid_forced_grammar_seq_values/turns.jsonl`
- Never-train hybrid audit: `runs/hybrid_broaden_nevertrain_v2/nevertrain_bfcl_apibank60/diffusion_hybrid_forced_grammar_seq_values/projection_value_audit.json`
