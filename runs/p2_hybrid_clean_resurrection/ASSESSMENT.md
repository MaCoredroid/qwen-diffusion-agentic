# P2 Hybrid-Clean Resurrection Assessment

Status: assessment only. No engine build launched.

Scope: standalone diffusion serving only. Drafter work and DFlash M0 are parked.

## Current Evidence

Hybrid-clean is now the default constrained-lane serving mode:

| slice | backend | exact_args | episode exact | valid | exact_seq | sec/turn | forwards/turn |
|---|---|---:|---:|---:|---:|---:|---:|
| matched-20 | AR-guided vLLM | 50/63 | 13/20 | 63/63 | 63/63 | 1.158 | n/a |
| matched-20 | diffusion careful | 44/63 | 11/20 | 62/63 | 57/63 | 6.686 | 95.51 |
| matched-20 | diffusion hybrid-clean | 47/63 | 13/20 | 63/63 | 63/63 | 3.904 | 56.83 |
| never-train BFCL/API-Bank | AR-guided vLLM | 77/184 | 19/60 | 184/184 | 126/184 | 0.596 | n/a |
| never-train BFCL/API-Bank | diffusion careful | 77/184 | 16/60 | 173/184 | 121/184 | 3.373 | 43.62 |
| never-train BFCL/API-Bank | diffusion hybrid-clean | 83/184 | 19/60 | 184/184 | 133/184 | 2.123 | 24.62 |

The quality case is credible on both slices. Hybrid-clean beats careful on matched-20 by `+4/-1` paired and beats AR-guided on never-train by `+6/-0` paired. The wall-clock gap remains an engine problem: HF hybrid-clean is still `3.37x` slower than AR-guided on matched-20 and `3.56x` slower on never-train.

## What Changes Versus Banked P2

The `p2_serving_reuse_plan.md` route remains the right substrate:

- vLLM main pinned at/after PR #42406.
- Model Runner V2 `ModelState`.
- DiffusionGemma-style canvas and commit path.
- `MambaHybridModelState` plus FR13 align-mode APC.
- `flare_hf_cache.py` as the GDN state/cache semantics spec.

The decode policy changes:

- Old P2 assumption: per-call waves with parallel value commits.
- New P2 target: hybrid-clean constrained serving.
- Bulk commit only truly-forced structural FSM tokens.
- Decode value spans strictly sequentially from the raw model distribution.
- Do not project or force value tokens.
- Keep tokenizer-offset value audits in the engine metrics path.

This is a smaller and cleaner engine target than value-parallel diffusion. It does not require proving that value conditionals can be parallelized. It does require exact state discipline when structure is committed without model sampling.

## Engine Design Delta

1. `Qwen3_5FlareModelState`
   - Same MRV2 registration plan as P2.
   - Denoise read must be read-only for GDN and conv state.
   - Clean commit must advance GDN state exactly once at block boundaries.
   - Forced structural tokens must update the request canvas and later committed prefix exactly as if they had been generated.

2. Hybrid-clean scheduler in `custom_sampler`
   - Run a per-request Qwen-native XML/tool-call FSM.
   - If the next token is truly forced by the FSM, bulk commit the maximal forced structural span.
   - Stop forced bulk at the first parameter-value token or any ambiguous/free token.
   - Decode each value/free token with one model denoise step and grammar-masked logits.
   - Never overwrite a sampled value token with a label or grammar projection.

3. APC and state publication
   - Preserve FR13 boundary-aligned commit discipline.
   - Prefix cache restore must copy fp32 GDN boundary state and conv tail verbatim.
   - Structure-only commits still need consistent request state before the next value-token forward.
   - Expose APC hit/miss/reused-token counters because upstream has no sufficient visibility.

4. Audits and observability
   - Surface `projected_value_tokens_exact`, `parallel_commit_forced_tokens_counter`, `wave1_value_tokens_counter`, `zero_forward_rows`, `true_xml_value_tokens`, and `reported_model_value_tokens`.
   - Hard-fail promotion runs if any value token is projected or force-committed.
   - Emit per-turn forwards, forced-structure count, sequential-value count, valid XML, exact_seq, schema_ok, and sec/turn.

## Effort Estimate

This is still P2-class engine work, but narrower than the original value-parallel plan.

| stage | work | estimate | gate |
|---|---|---:|---|
| P2.0 re-pin smoke | New venv, vLLM main post-PR #42406, DiffusionGemma smoke, Qwen export MRV2 default plus align/APC, read-only-denoise probe | 2-3 days | All three smoke items pass or have a concrete upstream/root-cause patch |
| P2.1 parity skeleton | `Qwen3_5FlareModelState`, read-only denoise parity, clean advance parity, one-turn HF hybrid-clean parity | 4-6 days | One matched turn byte-matches HF hybrid-clean with zero value projection |
| P2.2 FSM scheduler | Qwen-native FSM forced-structure bulk commit plus sequential value policy | 4-7 days | Matched-20 exact_args within 1 turn of HF hybrid-clean and all audits clean |
| P2.3 cache and latency | FR13 align APC counters, multi-turn cache restoration, latency profiling | 3-5 days | Hybrid-clean engine is materially faster than HF and state counters are stable |

Total: about 2-3 focused weeks to a real promotion gate if the vLLM pin is healthy. The older 5-6 week estimate included value-parallel waves and CUDA-graph/batching work; those are no longer required for the first standalone diffusion win.

## Promotion Gates

Run these gates before claiming engine-fast standalone diffusion:

1. Load and state gate
   - Qwen3.5-9B export loads under MRV2.
   - Align-mode APC is enabled and counters show nonzero reuse on repeated multi-turn prefixes.
   - Read-only denoise leaves GDN/conv state unchanged; clean commit advances once.

2. HF parity gate
   - One-turn and five-turn fixtures match HF hybrid-clean outputs at temperature 0.
   - Any nondeterminism must be explained by an explicit kernel or dtype difference.

3. Matched-20 gate
   - Exact_args `>=47/63` with v2 adapter, or `>=50/63` if an approved v6 adapter becomes the teacher.
   - Valid XML `63/63`.
   - Exact tool sequence `63/63`.
   - Zero projected value tokens.

4. Never-train gate
   - Exact_args `>=83/184` with v2 adapter, or no regression versus the selected teacher adapter.
   - Valid XML `184/184`.
   - Zero projected value tokens.

5. Latency gate
   - First promotion: faster than HF hybrid-clean by at least `1.5x` wall on matched-20.
   - North-star: at parity or better quality, close the AR-guided vLLM gap enough to justify engine continuation. The current AR-guided controls are `1.158` sec/turn on matched-20 and `0.596` sec/turn on never-train.

## Kill Criteria

- Stop if MRV2 cannot run the Qwen GDN model after five focused working days of fixes.
- Stop if read-only denoise mutates GDN or conv state.
- Stop if engine hybrid-clean loses more than one matched-20 exact turn versus HF hybrid-clean.
- Stop if any value-projection counter is nonzero.
- Stop if APC counters show silent zero reuse on the multi-turn shapes.
- Stop if engine wall time is not at least `1.5x` faster than HF hybrid-clean after the scheduler is correct.

## Recommendation

P2 is worth resurrecting as standalone diffusion serving, but only under the hybrid-clean contract. The target is no longer "parallelize values"; it is "preserve the audited hybrid-clean quality while moving the same fewer-forwards policy into the vLLM/MRV2 engine with FR13 APC."

Do not start DFlash or any drafter work. Do not start the engine build until this assessment is reviewed.
