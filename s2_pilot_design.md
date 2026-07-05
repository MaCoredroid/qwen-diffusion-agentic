# S2 Pilot — the cheapest decisive test of the L3 bet (reasoning-span K: 1 → ≥2 at held GSM8K)

Author: design sweep, 2026-07-05. Status: **DESIGN ONLY — monitor review before any GPU run. CPU-only produced.**
Campaign: `goal_5x_rollout_b1.md` (L3 is the surviving lever). Evidence base: `l1_content_mix_result.md`,
`l1_baseline_b1_result.md`, `runs/l0l2_final_head_verify/summary.json` (the free-text 26/30 anchor), banked recipe
`s1s2_speed_training_recipe.md`, root-cause `training_redesign_10x_research.md`, corrector `corrector_larger_blocks_research.md`.
Design-quality bar and provenance discipline: `convert_after_rl_design.md`. Sampler-pinning: `REPRODUCE_V2.md §0`.

---

## 0. The bet, in one paragraph

The 5×-at-B=1 north star is entirely gated on **one factor that is stuck at 1.0**: model-chosen reasoning tokens commit at
**K=1 tok/forward** (chain-rule wall; `l1_content_mix_result.md` — `denoise_forwards == model_chosen_tokens` exactly, and on
the working free-text path the engine emits **0.862 tok/fwd**, i.e. *below* 1.0, because block-diffusion re-denoises past EOS).
L2 (per-forward parity) cannot move this factor; only training can. This pilot runs the **smallest decisive test** of whether
training can lift reasoning-span committed-tokens-per-forward from 1.0 to **≥2.0 at held GSM8K accuracy**, inside **~1 GPU-day**.
It is a de-risk, not a promotion candidate: PASS greenlights the full S2 build (`s1s2_speed_training_recipe.md`); the pre-registered
KILL retires the 5×-vs-AR claim honestly.

**Verdict axis.** PASS ⇒ a parallel reasoning lane exists and is trainable on our GDN-hybrid stack ⇒ the S2/L3 program is worth
its ~53 GPU-h. KILL ⇒ reasoning-span K stays ≤1 at held exactness (or retention/tool-call safety fails) ⇒ K-factor is a wall on
this architecture ⇒ the 5× claim is retired and the campaign reverts to "0.36× today, L2 buys at most ~2×, no path to 5×."

---

## 1. Object under test — weights, lineage, exact artifacts

Per **retrain-freely** and the **certified-capability** requirement, the pilot trains on top of the **merged RL-v2 diffusion
candidate** — the weights that (a) carry the certified tool-call capability the pilot must not damage, and (b) on the L0-fixed
free-text path *do* free-CoT GSM8K (26/30), so the reasoning spans the pilot targets actually exist in the natural output.

| symbol | what it is | on disk |
|---|---|---|
| base (frozen) | `Qwen/Qwen3.5-9B @ c202236…b9a` | HF cache snapshot |
| init | Fast-dLLM candidate, mask `\|<MASK>\|` id **248077**, `bd_size=32` | `models/qwen3.5-9b-fastdllm-init` |
| **M_{t+1} (train base)** | init + RL-v2 folded, **diffusion-loadable** (mask/bridge intact), built + merge-gated by `convert_after_rl` step-1 | `models/qwen3.5-9b-fastdllm-mtplus1-merged` (**exists**) |
| RL-v2 vLLM export | the served merged-AR weights (free-text 26/30 anchor path, pin `0b44dcc`) | `models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16` |
| **A_S2 (this pilot)** | new consistency-distillation LoRA, disposable | `runs/s2_pilot/Apilot_step400_seed90101` (to be trained) |

**Why M_{t+1}-merged and not `init` (pre-RL) or A_new:** the tool-call spot-check gate (§6c) requires the certified capability
to be present in the base; it lives in the RL-v2 weights, not in `init`. Training on the merged base keeps that capability frozen
in the backbone while A_S2 (a small LoRA trained only on reasoning CoT) adds the span-consistency behavior. A_new (the
convert-after-RL re-diffusionized checkpoint) is equivalent for our purpose but M_{t+1}-merged is the exact promoted-serving base,
so it is the cleaner control point.

### Anchors this pilot is judged against (all measured, pinned)

| lane | metric | value | source |
|---|---|---|---:|
| **K-gate (primary)**: free-text GSM8K, K=1, 30-prompt clean set | strict correct | **26/30** | `runs/l0l2_final_head_verify/summary.json` |
| same, committed-tokens/forward (emitted, EOS-trimmed) | tok/fwd | **0.862** (K=1; forwards==model-chosen) | same |
| same, stability / audit | clean-stop · proj · verify | **30/30 · 0 · all-OK** | same |
| GSM8K retention (legacy full-context, N=20) | strict | **13/20 = 0.65** | `REPRODUCE_V2 §6`, `convert_after_rl` |
| tool-call (hybrid-clean matched-20) | exact_args | **47/63** (C0) | `REPRODUCE_V2 §7` |
| per-forward wall (free-text, cudagraph) | ms/fwd | **25.8** | `l0l2_final_head_verify` |

**The thing being moved = the 0.862→≥2.0 tok/fwd factor on reasoning spans, at 26/30 held (±McNemar noise).**

---

## 2. Training objective — trajectory-consistency self-distillation (precise)

**Method.** On-policy **trajectory-consistency distillation**: the teacher is the model's **own K=1 sequential free-CoT
trajectory** (offline, cached — no per-step teacher forward, which is what keeps the pilot inside 1 GPU-day). The student learns
**multi-position joint prediction**: given a block with an aggressive parallel mask, reproduce in *one* forward the tokens its own
K=1 decode committed one-at-a-time. This is the DSCD/CD4LM family (`s1s2_speed_training_recipe.md` §S2) reduced to its cheapest
faithful core — cached-target trajectory CE — with the DSCD nested-KL banked as the fallback (§11).

**Foundation preserved (two-stream FLARE, unchanged from the conversion recipe).** `FASTDLLM_FLARE_TWO_STREAM=1`, GDN
`route_i`, GDN state **read-only during denoise, advanced once at the committed block boundary** (the load-bearing
snapshot-restore verified 6/6 bit-identical in `l0l2_final_head_verify.readonly_denoise_fingerprint`), clean-stream `L_AR`
byte-identical to the AR forward, native `fast_dllm_v2_native` chat template, `TRUNCATION_SIDE=left`.

**Loss (new flag `FASTDLLM_FLARE_TRAJ_DISTILL=1`).** Answer region = the CoT continuation (prompt clean, loss-masked). Let
`y = (y_1…y_L)` be the cached K=1 self-trajectory tokens on the answer region.

```
L_pilot = L_AR(clean stream)                                   # retention anchor; unchanged; byte-identical to AR forward
        + w · Σ_{i ∈ M_S} CE( z_student(i) , y_i )             # trajectory-consistency: student marginal → own K=1 token
```

- **Masking / which positions.** `M_S = RandomSubset(answer positions)` at ratio `r_S ~ U(0.50, 0.90)` — *higher* than the
  conversion default `U(0.30,0.80)` on purpose: to force ≥2 adjacent positions masked jointly so the student must predict
  multi-position runs, not isolated infills. Short-answer protection: `L<20 → r_S ≤ 0.60`; min masks `= max(2, ceil(0.10·L))`.
  Loss is computed **only on `M_S` positions**; targets are `y_i` (the self-trajectory token), not a re-tokenized dataset string.
- **Shift.** `L_AR` is the standard next-token clean stream (causal, shift-by-one). The trajectory-CE term is a **denoiser**
  target: no shift — the student predicts the token *at* the masked position from bidirectional-within-block context.
- **Block-causality (GDN-specific, ours).** Nesting is **within-block**: masked positions may condition only on the current
  block + already-committed blocks, never later blocks (no off-policy targets the block-causal decoder can never see). This is the
  same discipline that kept `value_projection_events == 0` on the promoted path.
- **Values stay K=1 — not special-cased in training.** Free-CoT has no grammar FSM to tag numeric spans, and hand-tagging is
  fragile. Instead, values are held K=1 at **decode** by the entropy gate (§3): high-entropy numeric positions never enter the
  parallel-committed prefix. Training uses uniform trajectory-CE; the K=1-on-values guarantee is a decode-time invariant, not a
  training-time one. (Tool-call value spans remain FSM-forced K=1 regardless — §3.)
- **Numerics.** fp32 softmax; per-token aggregation; grad-clip 1.0; CE clamp 5.0; NaN-skip. No temperature term (cached-target CE,
  not soft-KL) in the primary; τ=2.0 only in the banked nested-KL fallback.

**LoRA config.** `r=16 / α=32 / dropout=0.05` (the proven RL-v2 / A_new envelope — **not** the S1 r=64 budget-retrain; this is a
de-risk, not the full dose). Targets: `q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj`
(attn + GDN, identical to RL-v2). Trained on top of `M_{t+1}-merged`; the RL-v2 tool-call capability sits frozen in the backbone.

---

## 3. Decode policy at eval — entropy-gated adaptive K (CAD), values/grammar unaffected

New sampler `scripts/eval_flare_freetext_cad.py::adaptive_k_sample_one` (a gated modification of the promoted
`_hybrid_clean_step`; sha recorded at run time per `REPRODUCE_V2 §0`).

- **Commit rule (free-text reasoning path only).** Per forward, over the block's un-committed frontier, compute per-position
  confidence `c_i = max softmax`. Commit `k = clip( |leading contiguous run with c_i ≥ γ| , 1, k_max )`, `k_max = 2` for the
  pilot. **Contiguous-prefix** commit means a single low-confidence position (a number) *blocks* the run → forces K=1 there — this
  is the mechanism that keeps values sequential. **Never remask** (preserves GDN state discipline / FR13 cache). **Native stop-ids
  only** (`248046`): stop the moment EOS is the committed token — this alone recovers the 0.862→≥1.0 EOS-overshoot waste before any
  parallelism.
- **γ sweep.** `γ ∈ {0.90, 0.95, 0.99}`, `k_max=2`. **Pinning sanity (mandatory before any sweep row):** `γ=1.0, k_max=1` must
  reproduce the K=1 free-text baseline **byte-exactly** (26/30, 0.862 tok/fwd) — if it does not, the sampler diverged from the
  promoted path and no row is comparable to the anchor.
- **Values & grammar untouched.** Tool-call turns still decode via **hybrid-clean**: structure force-committed at 0 forwards,
  value spans strictly K=1 through the grammar FSM — the adaptive-K commit applies *only* to the free-text reasoning path. So the
  "values always K=1" invariant holds two ways: FSM for tool-call values, entropy gate for free-CoT numbers.

---

## 4. Data recipe — self-generated CoT, leakage-safe

- **Prompts.** GSM8K **train** split (7,473 examples) only. The **gate set (test first-30)** and the **retention set (test
  first-20)** are the **test** split — disjoint from train by construction. Belt-and-suspenders: hash every train prompt and drop
  any whose normalized question matches a gate/retention item (expected 0 collisions; recorded in the corpus manifest). **Leakage
  rule: the 30-prompt gate set and the 20-prompt retention set never enter training data.**
- **Generation.** Sample K=1 free-CoT trajectories on `M_{t+1}` via the promoted free-text path (pin `0b44dcc`, greedy, seed
  90101). Record per-position committed token = the teacher target `y`.
- **Audit-filter (every trajectory).** Keep only turns that are (i) **strictly correct** (self-consistent teacher — we distill
  toward *correct* reasoning, not the model's mistakes), (ii) `verify.ok == True`, (iii) `value_projection_events == 0`, (iv)
  clean-stop (not length-capped / degenerate). Discard hangs/runaways (the ~20 % off-distribution instability from
  `l1_baseline_b1_result.md` is filtered here, not trained on).
- **Target size.** **~1,000 audit-clean correct trajectories** (a de-risk dose, not the S2 5,000). Yield ≈ 26/30 correct × ~54 %
  audit-clean ≈ 0.47 net → ~2,200 raw generations. **Yield floor:** if audit-clean-correct yield < 30 %, cut the target to 700 and
  proceed (a smaller clean corpus is fine for a feasibility probe; a dirty one is not).
- **No self-anchor-only risk hedge needed at this size,** but keep the CE target = the *correct* self-trajectory (filter above), so
  the corpus cannot reinforce wrong reasoning. Builder: `scripts/build_s2_traj_corpus.py`; corpus is teacher-checkpoint-bound
  (regenerate, never reuse, if the base or format changes).

---

## 5. Budget + step cap (erosion law)

- **Single round.** `MAX_STEPS = 400` optimizer steps (batch 1 × grad-accum 2 = 800 micro-forwards). **400 is the floor of the
  documented 400–600 erosion cap** (`convert_after_rl_design`, MEMORY); a de-risk takes the floor. **Do NOT extend past 600** to
  rescue a weak number — retrain-freely at a *different* step count in {300, 400, 500} instead.
- LR: WSD peak 1e-5 (warmup 0–40, stable 40–340, decay 340–400 to 1e-6); fallback `cosine_with_min_lr` (min-ratio 0.1, warmup 40).
- Save every 100 steps. Profile GPU-util at step 50 (GPU-util standard: torch-path baseline ~65 %; if util drops below, stop and
  fix the host-bound defect before continuing).
- **In-training retention guard (RL safety kit):** the runner probes GSM8K + tracks **rolling KL-to-base on the retention
  distribution; early-stop if KL exceeds 0.05** (the campaign's KL cap). This is the training-time analogue of the §6b gate.

---

## 6. Eval battery — sampler-pinned (every row records: git commit, script sha256, sampler fn, adapter+base paths, dataset+manifest hash, decode flags, value-projection audit file)

| script | pinned sha256 (re-verify at run) |
|---|---|
| `eval_flare_freetext_cad.py` (**new** — CAD adaptive-K) | record on first run; pin thereafter |
| `eval_flare_stage1_ab_diffusion.py` (GSM8K legacy retention) | `eaa78d7a9abfb32b7ab73c7753cf87026741e372ef13a1c0f8e44ead79b5e503` |
| `eval_flare_northstar_hybrid_clean.py` (tool-call spot-check) | `a4c66751008390ec44ff4fbb7d025352dc71ba21a005948411883818b908b1f3` |
| `audit_value_projection_tokens.py` | `7b203e3e8e2a7a7bbfa6f831be295543c728b08d9228bd241f0f07e35a620b40` |
| `export_qwen35_9b_fastdllm_vllm.py` | `6d507ec9ba3308ff7e0f600bc0b5ec7c4ff96f66eff4e4e92175d42af7a119d5` |

**(a) K-GATE — the primary signal (30-prompt clean set, free-text CoT).** `runs/l1_census/gsm8k_prompts_clean.json`, the identical
prompt_ids behind the 26/30 anchor. Three decode rows on **A_S2** + one control base row (§7): measure **committed-tokens/forward
(emitted, EOS-trimmed) on the reasoning answer region** and **strict GSM8K correct**. γ-sweep {0.90,0.95,0.99}, pick the highest-K
row that holds accuracy. This row is the pilot verdict.

**(b) GSM8K RETENTION ANCHOR — legacy full-context, N=20.** `eval_flare_stage1_ab_diffusion.py::full_context_sample_one`,
`--generation-tasks gsm8k --generation-limit 20 --full-context-generation --fresh-generation-blocks --block-size 32 --temperature
0.0 --mask-id 248077 --stop-token-id 248046`, `--adapter A_S2`, base `M_{t+1}-merged`. Guards general reasoning retention against
the trajectory-CE objective. **Do NOT substitute `measure_block_quality_curve.py`** (disqualified mutable-remask diagnostic).

**(c) TOOL-CALL SPOT-CHECK — the certified capability must not be damaged.** 10 matched turns from the hybrid-clean matched-20 set
(`eval_flare_northstar_hybrid_clean.py`, base `M_{t+1}-merged`, adapter A_S2, `--block-size 32 --temperature 0.0 --grammar-topk
256`). Decode policy on tool calls is **unchanged hybrid-clean** (K=1 FSM values) — the S2 LoRA is loaded but adaptive-K is off on
this path; this is a safety check that the reasoning LoRA did not perturb tool-call exact_args. Pair against the same 10 turns' C0.

**(d) AUDITS — hard, zero-tolerance (`REPRODUCE_V2 §8`).** `audit_value_projection_tokens.py` on **every** diffusion turns.jsonl
(K-gate + retention + spot-check). Required: `projected_value_tokens_exact==0`, `parallel_commit_forced_tokens_counter==0`,
`wave1_projected_tokens==0`, `wave1_value_tokens_counter==0`, `wave2_forced_tokens_counter==0`, `zero_forward_rows==0`,
`verification_mode==no_projection_events`. **Any nonzero ⇒ KILL-3** (the tok/fwd number is contaminated — this class has
manufactured every phantom win in this project).

---

## 7. Controls (report side-by-side; the training delta lives here)

| row | base | adapter | decode | isolates |
|---|---|---|---|---|
| **CTRL-decode**: untrained-adapter K=2 | M_{t+1}-merged | **none** | CAD adaptive-K (γ-swept, k_max=2) | decode-policy-only baseline — how much K comes free from adaptive commit + EOS-stop *without* training |
| **CTRL-K1**: trained-adapter K=1 | M_{t+1}-merged | A_S2 | K=1 (γ=1.0) | did training **hurt** sequential K=1 accuracy? (must stay ≈26/30) |
| **A_S2 K=2** (primary) | M_{t+1}-merged | A_S2 | CAD adaptive-K | the bet: ≥2.0 tok/fwd at held accuracy |
| anchor | RL-v2 export | — | K=1 free-text | 26/30 · 0.862 tok/fwd |

**Promotion discipline (memory: `diffusion-promotion-discipline`).** The pilot "wins on training" only if **A_S2 K=2 beats
CTRL-decode** — i.e. training either (a) reaches ≥2.0 tok/fwd at held accuracy where CTRL-decode *loses* accuracy, or (b) reaches a
strictly higher held-accuracy K than CTRL-decode. If **CTRL-decode alone** already hits ≥2.0 tok/fwd at held accuracy, the gain is
**decode-policy, not training** — report that honestly (a still-useful L3 result: adaptive-K decode suffices for the pilot bar and
S2 training is not the lever), and do not credit training.

---

## 8. Statistics + PASS / KILL thresholds (pre-registered)

30-prompt gate: `n=30`, K=1 anchor `= 26/30`. Pair the 30 prompts (K=2 vs K=1 on **the trained adapter**): `b =` (K=1 right & K=2
wrong), `c =` (K=2 right & K=1 wrong); net loss `= b − c`; two-sided exact-binomial **McNemar** on `(b,c)`.

| # | measurement | anchor | PASS | KILL / FAIL | INCONCLUSIVE |
|---|---|---|---:|---|---|
| a | K-gate: A_S2 K=2 tok/fwd **and** accuracy | 0.862 tpf · 26/30 | **tok/fwd ≥ 2.0** AND net-loss `≤ 2` AND McNemar p ≥ 0.05 | net-loss `> 2` (K=2 loses >2 items) | tok/fwd 1.5–2.0 at held acc, or net-loss=2 at p<0.05 → retrain {300,500} / one γ re-sweep |
| b | GSM8K retention N=20 | 13/20 | **≥ 13/20** | **≤ 11/20** (fell to conversion floor) | 12/20 → rerun once (single-row rule) |
| c | tool-call spot-check (10 matched) | C0-10 | **10/10 exact vs C0** (0 lost) | ≥ 2 lost | 1 lost → rerun once |
| d | value-projection audits | 0 | all counters 0 | any nonzero (run invalid) | — |
| — | training delta vs CTRL-decode | — | A_S2 K=2 > CTRL-decode at held acc | — | CTRL-decode ≥2.0 held ⇒ "decode-only" verdict (not a training win) |

**PILOT PASS (L3 greenlit):** a PASS ∧ b PASS ∧ c PASS ∧ d clean ∧ training-delta positive. ⇒ fund the full S2 build.
**PILOT KILL (5× retired):** a KILL **OR** b KILL **OR** c KILL (with d clean). ⇒ reasoning-span K is a wall (or the pilot damaged
a certified/retained capability); retire the 5×-vs-AR claim; campaign reverts to the honest "0.36× today, L2-bound ~2× ceiling."

---

## 9. Kill gates (consolidated, pre-registered)

- **KILL-0 (base sanity)** — `M_{t+1}-merged` fails its merge gate (`mask_token_id≠248077` / `bd_size≠32`, or a free-text sanity
  episode is not ≈26/30-coherent). Base is wrong; do not train. (convert_after_rl already gated this artifact; re-verify.)
- **KILL-a (the primary bet)** — after the full 400-step budget, A_S2 K=2 loses **>2 items vs K=1 on the 30-set**. The parallel
  reasoning lane does not hold exactness. **⇒ retire the 5× claim.**
- **KILL-retention** — GSM8K legacy N=20 **≤ 11/20** OR in-training rolling **KL-to-base > 0.05** unrecovered. General reasoning
  eroded. **⇒ retire.**
- **KILL-toolcall** — tool-call spot-check loses **≥2 of 10** exact-args vs C0. The pilot damaged the certified capability.
  **⇒ retire (or, if isolated to the LoRA-on-reasoning interaction, one retrain at r8/reasoning-only before re-judging).**
- **KILL-3 (contamination)** — any value-projection counter nonzero, `zero_forward_rows>0`, or sampler path ≠ the pinned CAD
  function. The tok/fwd number is invalid — **not** a capability reading.
- **INCONCLUSIVE handling** — K-gate tok/fwd in 1.5–2.0 at held accuracy after **one** retrain (different step count) + one γ
  re-sweep ⇒ report inconclusive (partial lane, cap L3 expectations at ~1.5×); do **not** extend steps past 600 to manufacture a
  pass.

---

## 10. GPU-hours + wall-clock (RTX 5090, single GPU, one process at a time)

| step | GPU-h | note |
|---|---:|---|
| Re-verify `M_{t+1}-merged` base + free-text sanity | ~0.1 | artifact exists (convert_after_rl step-1) |
| Self-trajectory corpus gen (~2,200 raw → ~1,000 clean) + audit-filter | ~3.0 | free-CoT ~3 s/turn; batch prompts to keep the 5090 fed |
| Build corpus (CPU) | ~0.1 | dedupe + manifest |
| VRAM smoke (3 steps, exact config) + train 400 steps | ~0.3 + ~1.2 | ≈ Run-1 step rate; CE-only (no per-step teacher) |
| Export A_S2 clean stream → vLLM | ~0.1 | shard IO |
| Eval (a) K-gate: A_S2 K=1 + K=2 γ-sweep{.90,.95,.99} + CTRL-decode γ-sweep, 30 prompts each | ~1.5 | free-CoT, ~5 rows + boots |
| Eval (b) GSM8K legacy N=20 (slow full-context diffusion) | ~0.4 | |
| Eval (c) tool-call spot-check 10 turns (hybrid-clean) | ~0.2 | |
| Audits (d) | ~0.0 | CPU |
| **Core total** | **~7** | |
| slack: one retrain (diff. step count) + one re-sweep | +~2 | INCONCLUSIVE path |

**Plan for ~7 GPU-h core, ~9 GPU-h with the one allowed retrain — comfortably inside the ~1 GPU-day (24 h) cap.** Wall-clock
~½–¾ day sequential (corpus gen + model loads dominate).

---

## 11. The honest prior — adverse evidence, and what (if anything) is different

**The adverse evidence is real and must be stated first.**
1. **SDTT was null on VALUE exactness.** Consistency-distillation (SDTT-class) did not conjure correct exact values in our banked
   evidence; `s1s2` carries cached-SDTT only as a *fallback*, and `training_redesign_10x` rates the paired/derived-value class
   (`C(Y|X)>0`) as **architecturally un-parallelizable at any training dose**.
2. **All-SFT-neutral history.** Off-policy SFT-on-teacher-traces has repeatedly been capability-neutral-to-negative here; the RL
   safety kit exists precisely because SFT-style objectives forget.
3. **Measured high conditional entropy on reasoning tokens.** `l1` measured **top-1 conditional ≈ 0.238** on model-chosen reasoning
   tokens — the *opposite* of the low-`C` copy spans ParallelBench says are trainable-parallel-safe. If most of the reasoning span
   is high-entropy, the entropy gate keeps it at K=1 and the tok/fwd never reaches 2.
4. **Empty-literature GDN surface.** GDN-linear-attention × diffusion is un-derisked; the one recurrent-backbone analog
   (B3D-RWKV) **cratered on GSM8K (−12.4)**. LoRA-capacity DSCD has **no published positive evidence**.

**What is different here — and why it is or is not likely to matter.**
- **Target = span consistency on reasoning content, NOT exact-value conjuring.** SDTT's null was on *manufacturing the correct
  value*. This pilot never asks the parallel step to conjure a value: values are held K=1 by the entropy gate (§3, contiguous-prefix
  commit blocks on any low-confidence numeric position). The tok/fwd ≥2 target is met by parallelizing the **low-entropy connective
  / format / restatement tokens** of the CoT ("Remaining eggs = ", "So the answer is ", operators, spaces) — the `C≈0`-ish fraction
  ParallelBench says training *can* make parallel-safe (Replace-Index <50 %→~100 %). `l1` corroborates the mechanism: ~26 % of the
  reasoning-wrapper tokens are already zero-forward grammar scaffold; the bet is that trajectory-CE moves the *next* tier of
  low-entropy connective tokens from K=1 to joint-commit. **This is a strictly easier target than GSM8K-value exactness** — and it
  is *why* the goal doc flags reasoning-span K on GSM8K accuracy as a "DIFFERENT, easier" objective than SDTT's value test.
- **Why it may still not matter (the honest hedge).** The 0.238 top-1 says the low-entropy connective mass may be too thin to reach
  an *average* of 2.0 tok/fwd even if every connective token parallelizes — the numeric/derived positions could dominate the span.
  And the GDN bidirectional-within-block copy circuit may simply refuse to commit adjacent reasoning tokens jointly (the
  empty-literature wall; our prior causal-value-span probe was INERT, i.e. uninformative, not encouraging). **Either of these lands
  the pilot in KILL-a — which is a real, valuable, honestly-reported outcome:** it localizes the 5× wall to the K-factor on this
  architecture and retires the claim on evidence rather than hope.

**Net stance.** This is a genuine coin-flip de-risk with a pre-registered kill, not an advocacy build. The one structural reason for
optimism (reasoning CoT has a real low-entropy connective fraction that ParallelBench says is trainable-parallel-safe, and values
are protected by construction) is set against four concrete adverse priors. ~1 GPU-day buys the answer either way.

---

## 12. Provenance checklist (attach to the result report)

Per row: git commit; script sha256 (re-verified vs §6 table; the new CAD sampler sha pinned on first run + the γ=1.0/k=1
byte-exact sanity certificate); base-model + adapter paths; corpus manifest hash + leakage-dedupe count (must be 0 gate/retention
collisions); decode flags (γ, k_max); value-projection audit JSON path. Record the base merge-sanity re-verification, the training
KL-to-base trace (early-stop check), and both the tok/fwd measurement and the McNemar `(b,c)`+p for the K-gate. Commit + push each
artifact to origin/main with narrated reasoning per the commit-workflow rule.
