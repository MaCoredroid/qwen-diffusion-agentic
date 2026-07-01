# Corrector-for-Larger-Blocks Research (workflow w62lg1q22, 2026-07-01)

Verified 9/9 credible mechanisms. Papers: DINGO 2505.23061, EPIC 2606.00722, LAVE 2602.00612, PRISM 2510.01384, CoDD/DCD 2410.01949, DEER 2512.15176, SpecDiff-2 2511.00606, ReMDM 2503.00307, SCOPE/D3IM 2606.01026. (1 research angle dropped mid-run on an API error; DINGO/EPIC/LAVE covered via other angles.)

# Corrector-for-Larger-Blocks: Decision-Grade Proposal

Scope: can a corrector (grammar + learned + small-AR-on-C>0-spans + remasking) let the agentic GDN-hybrid push block size materially larger at held STRICT exact-args, and is the extra speedup real or just AR cost in disguise? Backbone: Qwen3.5-9B GDN hybrid, single RTX 5090, QLoRA, batch-1. Metric of record: raw value-span exact-match (≥7/12 heldout, 10/12 public) with GSM8K ≥0.60 retention.

---

## 1. VERDICT

**A corrector buys a factor, not 10x — and only ONE corrector class actually enlarges blocks on the content that matters.** Split by what "larger block" means:

- **On C≈0 spans (copy args, IDs, paths, JSON scaffold):** blocks are already parallel-robust and Run-1-trainable to exact. Grammar/completability correctors (DINGO 2505.23061, EPIC 2606.00722, LAVE 2602.00612, our live decoder) make the *scaffold* commit in bulk (EPIC: 7.0 tokens/step on JSON) for **near-zero cost, no extra forward**. This is real and free, but it is not where the barrier lives — it does nothing for the C>0 residual that caps us.

- **On C>0 spans (start/end pairs, INV-301/302, derived/convention values, cross-call ids):** the factorization barrier is architectural and scale-independent (2602.00286). The **only** mechanism class that can enlarge a block here is **joint-modeling correction** — CoDD/DCD (2410.01949) putting the cross-token joint back into the parallel step via I-projection. Everything else (guidance 2412.10193, marginal sharpening, confidence thresholds) provably *cannot* fix a confidently-but-independently-wrong token (`end_time` peaks at 17:00 regardless of `start_time`); sharpening a factorized marginal makes it *more* confidently wrong.

- **Does the corrector re-introduce AR cost?** Partially, and honestly: yes for the two methods that actually move the C>0 needle.
  - CoDD-class joint corrector = **+1 AR (copula) forward per denoise step**. It claims "8–32x fewer *steps*" — but that is steps-vs-base-diffusion, **not wall-clock-vs-AR**. Net win exists only if (a) the copula is KV-cached so its cost amortizes and (b) it is **gated to C>0 spans only**, so you don't pay the AR forward on the C≈0 bulk.
  - Spec-decode (DEER 2512.15176, SpecDiff-2 2511.00606) = **full AR verify forward per round**; output is provably AR-exact but the diffusion model is demoted to a disposable drafter — this does **not** advance the block-diffusion-as-generator thesis. Dense-9B realized speedup ~3x, not the MoE 5.5x.

- **How much extra speedup, honestly?** The blended tool-call ceiling remains ~2–3x at strict grounding (published held-quality band is D2F 2.5x → CD4LM 5.18x; anything >5x at strict grounding is measuring the wrong metric). A joint corrector, gated and amortized, plausibly moves the C>0 spans from near-AR (K≈B) toward K≈B/2–B/3, lifting the *weighted-average* tool-call speedup from today's ~1.0–1.1x toward the 2–3x top. **It pushes toward the top of the band; it does not break past it, and it does not deliver 10x — 10x is not physically reachable at strict quality on this arch.**

**Bottom line:** grammar corrector = free but only enlarges scaffold; learned consistency head = cheap quality-recovery + the *router*, not a block multiplier; joint (CoDD) corrector = the real larger-block lever for C>0 but partially re-introduces AR cost and only nets out if gated+amortized; remasking-to-unfreeze is **disqualified on our backbone** (see §2, rank 5). Novel-surface caveat: GDN-linear-attention × diffusion × copula is empty literature (§2, §4).

---

## 2. BEST CORRECTOR DESIGN, ranked by payoff × feasibility (single-5090 QLoRA)

### Rank 1 — Grammar/completability decoder (HAVE IT) + free confidence gate `(a)`
- **Build:** keep the live label-free grammar decoder (`eval_fastdllm_toolcall_cases.py`, banked 19/28 exact-args, 28/28 valid JSON, `live_unsafe=0`) as the permanent structural base. Add DINGO-style offline token-DFA + DP-over-logits (2505.23061) for schema-forced bulk-commit, and elementwise max/entropy on logits the forward already produced as the first-order C-proxy.
- **Papers:** DINGO 2505.23061, EPIC 2606.00722, LAVE 2602.00612.
- **Overhead:** **no extra forward.** CPU-side parsing ~1–2% (DINGO), up to ~20–27% on complex grammars/few steps (EPIC/LAVE). Confidence gate is free (logits already computed).
- **Block-size gain:** widens parallel commits **only on grammar-forced/grammar-safe scaffold** (JSON envelope, keys, delimiters, function-name literals). **Zero gain on C>0 argument values.** Credit it with scaffold_tpf, never with argument-value exactness.

### Rank 2 — PRISM-style learned commit-safety / value-consistency head `(b)` — the ROUTER
- **Build:** a second sigmoid head reading the SAME final hidden state as the unmask head, trained with BCE `L=BCE(1[committed==gold], g_φ)` on the Run-1 copy-grounded trajectories (needs gold target tokens, no RL/verifier/quality-labels). Gives free per-token commit-safety at inference = **mechanism (b)**, one matmul+sigmoid on existing hiddens.
- **Paper:** PRISM 2510.01384 (Kakade/Chen). Adapter scale 0.13M–7M — skip the 250M/360-GPU-hr LLaDA recipe.
- **Overhead:** scoring is **free (one matmul, no forward).** The revise loop is not free, but here we use the head as the **gate/router**, not the corrector: it tells us *which* tokens are the confidently-but-independently-wrong C>0 residual so we spend the expensive corrector (Rank 3) only there.
- **Block-size gain:** small direct recovery (+2 to +4pp strict, concentrated at aggressive parallelism, diminishing/negative as steps grow). **Its real value is riding on existing compute to gate Rank 3** — this is exactly the overhead-balance rule ("detector must ride on compute already being done, never add a forward to decide the schedule").

### Rank 3 — CoDD I-projection joint corrector, same-model AR stream as copula, gated to C>0 — the REAL larger-block lever (Run 2)
- **Build:** at each denoise step within a block, run **one KV-cached, teacher-forced AR forward** over the block using the existing frozen same-model clean-AR stream as the copula; combine with diffusion marginals via the log-linear rule `x_i ∝ p_ar · exp(log p_dm − log p_copula)` (CoDD Alg.1/Eq.8). **Gate on C>0 spans only** (flagged coarse by grammar = `argument_value` kind, refined by Rank-2 head + confidence), so the AR forward is not paid on the C≈0 bulk.
- **Paper:** CoDD/DCD 2410.01949 (ICLR 2025). This is the ONLY joint-restoring class; corroborated as necessary by the C(Y|X) framework + 2602.00286.
- **Overhead:** **+1 AR forward per denoise step** (the honest cost). Zero new trainable weights (reuse existing AR stream) → fits the ~3 GPU-hr frozen-forward budget. Must implement copula KV-cache reuse across steps so its cost amortizes.
- **Block-size gain:** the mechanically-correct one — modeling the joint lets a block tolerate MORE co-emitted correlated tokens before corruption ("8–32x fewer steps" vs base diffusion). This is what should move `end_time`/`INV-302`/paired-value spans from K≈B toward K≈B/2–B/3. **Must be validated at strict exact-match and wall-clock, not gen-PPL/steps** — the paper's metric is confounded (copula = evaluator family) and warns net speedup is not guaranteed at few steps.

### Rank 4 — DEER-style self-spec-decode (same-model AR verifier) — latency INSURANCE, not thesis
- **Build:** keep the AR stream as an exact rejection-sampling verifier; QLoRA-align the block-diffusion drafter to it; 1 diffusion draft forward + 1 parallel AR verify forward per round, multi-token rejection. Output provably AR-exact → tool-call args AR-exact for free.
- **Papers:** DEER 2512.15176, SpecDiff-2 2511.00606.
- **Overhead:** full AR verify forward/round, amortized over accepted τ. Dense-9B ~3x (not MoE 5.5x); τ collapses exactly on high-C exact strings/paths (our crux spans), so realized speedup on arg-grounding is the weakest part of the trace.
- **Block-size gain:** raises accepted-run length (up to 32 tokens) but **demotes diffusion to drafter** — does not enlarge standalone diffusion blocks and requires shipping+running the full AR model. Adopt only if we want guaranteed AR-exact latency, not as evidence for the block-diffusion thesis.

### Rank 5 — Remasking-to-unfreeze (ReMDM / SCOPE+D3IM) — DEPRIORITIZE on our backbone
- **Papers:** ReMDM 2503.00307, SCOPE+D3IM 2606.01026.
- **Why last:** D3IM/ReMDM revise/re-mask already-committed tokens (`c_t=0`), which **forfeits the FR13 GDN prefix cache** and **leaks tentative tokens into the GDN recurrent state S_t** — a direct hit on our two hardest constraints (block-cache acceleration + state-write-corruption trap). Each corrector sweep is a **full forward**; benefit comes from MORE steps (opposite of the fewer-steps/larger-block direction). Use only as a **bounded, verifier-gated last-resort recovery** on individual blocks whose raw grounding fails, with a hard cap on extra forwards — never global.

---

## 3. ECONOMICS (tied to the overhead-balance rule)

Net speedup = parallelism_gain − corrector_overhead. Per corrector:

| Corrector | Extra forward? | Overhead | Parallelism unlocked | Net |
|---|---|---|---|---|
| Grammar + confidence (R1) | No | ~1–2% CPU (up to ~27% complex) | Scaffold bulk-commit only (EPIC 7 tok/step JSON) | **+ on scaffold, 0 on values** |
| Learned head (R2) | No (1 matmul) | Negligible | Routes the expensive corrector | **+ as router** |
| CoDD joint (R3) | **Yes, +1 AR/step** | 1 copula forward/step, KV-cached + gated | C>0 spans K≈B → K≈B/2–B/3 | **+ only if gated + amortized** |
| Spec-decode (R4) | Yes, full AR verify/round | amortized over τ | AR-exact ~3x dense | + but not the thesis |
| Remask (R5) | Yes, full forward/sweep | breaks GDN cache + S_t | none for larger blocks | **− on our backbone** |

**Break-even for R3 (the load-bearing one):** let `f` = fraction of value tokens flagged C>0 (gated), copula forward cost ≈ diffusion forward cost. You pay ≈`(1+f)` per step. Net win requires the joint to cut denoise steps on the flagged spans by more than that surcharge, i.e. `step_reduction > (1+f)`. With `f≈0.2` (C>0 is a minority — bulk is copy+scaffold), you need the copula to buy **>1.2x fewer steps on C>0 spans** to clear break-even, and materially more to reach the weighted 2–3x. This is plausible per CoDD's "8–32x fewer steps," **but that number is on softmax SEDD, not GDN, and not at strict exact-match** — hence the first experiment must measure it directly. The whole economic case collapses if the gate is loose (paying the AR forward everywhere ⇒ you've just rebuilt AR).

---

## 4. FIRST EXPERIMENT — smallest run to test "corrector enables larger blocks"

**Hypothesis:** a C>0-gated CoDD joint corrector moves the *block-size-at-held-strict-exact-args* materially right at equal-or-better wall-clock, on the Run-1 copy-grounded checkpoint.

**Asset:** Run-1 checkpoint at `/home/mark/qwen_diffusion/runs/flare_redesign_run1_eval` (copy-grounded, C≈0 values trained sharply confident). Base `models/qwen3.5-9b-fastdllm-init`; venv `.venv-fastdllm`; harness `scripts/eval_fastdllm_toolcall_cases.py`.

**Primary metric:** largest block size B on argument/value spans such that raw value-span exact-args stays ≥7/12 heldout, ≥10/12 public, `live_unsafe=0`, GSM8K ≥0.60.

**Procedure (three lanes, same eval set, temp 0 + a low-temp sample):**
1. **Baseline degradation curve (no corrector):** sweep block B ∈ {32, 64, 128, 256} on value spans; record strict exact-args, value_tpf, wall-clock at each. Establishes where C>0 corruption bites (expect a cliff on paired/derived/cross-call args). This is "block size at held strict quality, raw."
2. **Grammar+confidence control (R1):** same sweep. Expected ≈ baseline on values (~1.0–1.1x) — confirms grammar does NOT enlarge C>0 blocks (guards against crediting the free lever).
3. **Gated CoDD (R3):** same sweep, adding the same-model frozen AR stream as copula (KV-cached) via I-projection, **gated to C>0 spans** (grammar `argument_value` kind + confidence/Rank-2 head). Metric: does the held-strict-exact-args block size move right (e.g. 64→128) at ≤ baseline wall-clock?

**Instrument:** value_tpf, copula forward count + fraction gated, end-to-end wall-clock (not steps), `live_unsafe`, GSM8K retention, and a per-span breakdown isolating the C>0 residual (start/end pairs, INV-30x, cross-call ids) vs C≈0 copy.

**Kill criterion (be adversarial):** if gated-CoDD at the larger B does **not** beat smaller-B-without-corrector on wall-clock at equal strict exact-args, the corrector is just re-introducing AR cost → kill. Also kill if the gate fires on >~30% of value tokens (ungated ⇒ rebuilt AR) or if GSM8K drops <0.60.

**Empty-literature / novel-surface flags (must watch):**
- GDN linear-attention × diffusion × copula is **triple-novel** — no prior art. The only recurrent-backbone diffusion analog (B3D-RWKV) **cratered on dependency-heavy tasks** (GSM8K −12.4, MATH −25); the prior causal-value-span test came back **INERT** (byte-identical at temp 0 — uninformative, not refuted).
- The copula AR forward must advance the GDN recurrence **without leaking masked/tentative tokens into S_t** (state-write trap) and must respect the FR13 read-only-during-denoise / advance-once-at-committed-boundary discipline — unmeasured for a copula pass.
- CoDD's "8–32x" is softmax SEDD, full-sequence, gen-PPL — **none of block-diffusion, GDN, or strict exact-match is in the paper.** Treat the larger-block reading as our extrapolation to be tested here, not a paper result.

**Honest framing for the record:** this experiment tests whether the joint corrector buys a *factor* on the C>0 residual toward the 2–3x blended top. It is not a path to 10x, and a null result (CoDD doesn't beat smaller-B-AR on wall-clock at strict quality) is a real, publishable outcome that redirects effort to the spec-decode insurance lane (R4) for guaranteed AR-exact args.