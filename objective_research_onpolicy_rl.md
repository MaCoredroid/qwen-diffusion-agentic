# Agentic-phase training OBJECTIVE — deep-research verdict (workflow w9sldj0k3, 2026-06-30)

**Decision-grade (104 agents, 22 sources, 101 claims → 23 confirmed / 2 refuted / 9 synthesized; every post-2026
arXiv ID adversarially verified REAL via primary retrieval — no hallucinated IDs survived). Monitor-red-teamed.**

## Verdict: the next training lever is ON-POLICY RL with an exact-match tool reward — NOT more imitation data
The literature gives a clear, convergent answer to "what loss for the agentic phase," and it directly explains our
failure (imitation SFT tied the decoder + regressed GSM8K 0.70→0.55–0.60):
- **ToolRL (arXiv 2504.13958)** — most on-point: GRPO with a reward decomposed into **format + tool-name +
  param-name + EXACT param-VALUE match**. It shapes exact argument-value *selection by reward*, which token-CE can't.
  Reports SFT-initialization HURTS RL generalization (matches our SFT regressions).
- **RL's Razor (2509.04259, +2605.28860, 2510.18874)** — the *mechanism* for capability protection: on-policy RL is
  implicitly biased to **KL-minimal** solutions, so it forgets far less than **off-policy SFT-on-teacher-traces**
  (exactly our failed regime); forgetting is predicted by KL-to-base on the new-task distribution. This is *why* RL
  should avoid the GSM8K cratering our two SFT mixes caused.
- **d1 / diffu-GRPO (2504.12216)** — first critic-free policy-gradient RL for masked dLLMs; handles the intractable
  non-AR likelihood via a mean-field one-step estimator. Shows **SFT-then-RL is synergistic (beats either alone in
  11/12 setups)** → warm-start from our converted model, then RL.
- **GDPO (2510.08554)** — replaces diffu-GRPO's biased token-level estimator with a **sequence-level ELBO** →
  better for paired/row-aligned value coherence (our hardest slice).
- **IGPO (2509.10396)** — fixes GRPO's sparse-reward zero-advantage/zero-gradient failure via **inpainting** (matters:
  exact-match tool reward is sparse).
- **OPDLM (2606.06712)** — the on-policy version of *conversion itself*: student generates its own denoising
  trajectories, a frozen AR teacher supplies target logits → removes the train-inference mismatch we identified.
- (Q3, medium) **Structured Agent Distillation (2505.13820)** — [REASON]/[ACT] span-segmented distillation.

## The honest LIMIT (this changes the end-state design)
**Parallel-decoding theory (2602.00286):** exact decoding of *highly-correlated joint values* under parallel denoising
is **fundamentally expensive (cost exponential in within-block total correlation), and incoherence is INVISIBLE to the
Forward-KL/CE objective.** So paired/row-aligned grounding (end_time vs start_time, INV-302 vs INV-301) is **partly
structural to parallel decoding — mitigable but not fully "trainable away."** This *validates* our live decoder's
left-to-right JSON commit (reduced parallelism on high-correlation spans) as the **theoretically-indicated** complement,
not a hack. Self-correction objectives (Loopholing 2510.19304, self-correcting MDMs 2602.11590, ReMDM 2503.00307) are
the diffusion-native way to add revision.

## Red-team — gaps / risks (why this is a bet, not a recipe)
- **APPLICABILITY GAP (biggest):** every confirmed RL/grounding result is on **full-attention masked diffusion (LLaDA)
  or AUTOREGRESSIVE tool-calling**, on **verifiable math/logic rewards** (GSM8K/MATH/Countdown/Sudoku). **NONE is on a
  GDN-hybrid (linear-attention) backbone, and NONE tests RL on tool-call argument grounding.** We'd be first on both →
  the novel contribution surface AND the open risk.
- **No published recipe composes an AR→diffusion CONVERSION objective WITH an agentic exact-match RL reward on the same
  model** (OPDLM is imitation-only; d1 is RL-on-LLaDA without conversion; ToolRL is AR-only). The staged composition is
  ours to design.
- **Untested core question:** does exact-match RL reward actually move VALUE grounding (0.15 vs '15') or only the
  verifiable-reasoning tasks the dLLM-RL papers measure? Unknown.
- **THROUGHPUT (ties to FLA):** diffu-GRPO/GDPO/IGPO need **multiple forward passes / group rollouts / ELBO Monte-Carlo
  per step** — uncharacterized on a single 32GB consumer GPU under QLoRA. **This is exactly why un-parking FLA matters:
  RL is rollout-heavy; the kernel throughput win directly buys the rollouts.**
- Our own FLARE (2606.01774) / Fast-dLLM v2 anchors were NOT re-verified by this synthesis. 2 claims vote-refuted &
  excluded (2503.03595 independence claim; an over-attributed "sampling wall" framing of Loopholing — paper still real).

## Synthesized roadmap (evidence-based)
**FLA (throughput) → on-policy RL agentic phase → keep the live decoder for the structural residual.**
1. **Un-park FLA** (in progress) — buys the rollout throughput RL needs.
2. **De-risk pilot FIRST:** characterize diffu-GRPO/GDPO memory+throughput on our setup (the uncharacterized risk)
   before committing — a tiny RL loop with a ToolRL-style exact-match reward on a handful of cases.
3. **On-policy RL phase:** warm-start from the converted model (SFT-then-RL), ToolRL-style reward
   (format+name+param-name+**exact value**), GDPO sequence-level objective for paired coherence, KL-to-base for
   capability protection (RL's Razor), IGPO inpainting if reward is too sparse.
4. **Keep the live grammar decoder** as the inference-time complement for the partly-structural paired-value limit.

**Sources:** 2504.13958 (ToolRL) · 2504.12216 (d1/diffu-GRPO) · 2510.08554 (GDPO) · 2509.10396 (IGPO) · 2509.04259
+2605.28860+2510.18874 (RL's Razor) · 2606.06712 (OPDLM) · 2505.13820 (Structured Agent Distillation) · 2602.00286
(parallel-decoding limit) · 2510.19304 (Loopholing) · 2602.11590 (self-correcting MDMs) · 2503.00307 (ReMDM).
Full output: tasks/w9sldj0k3.output.

## DESIGN DECISIONS (2026-06-30, user)
1. **Constrained decoder in the loop for BOTH train and eval.** RL rollouts AND eval generate via diffusion + our
   in-house live grammar constrained decoder (label-free). Rationale: structure is always valid → the reward focuses
   purely on value-content / task-success (denser, not sparse), AND train==inference (we deploy with the decoder).
   **RL implication:** the policy is (model+decoder) — the grammar masking is part of the sampling distribution, so the
   diffu-GRPO/GDPO likelihood/advantage must be computed over the CONSTRAINED distribution (decoder is part of the
   env/policy, not a post-hoc filter). The decoder is a METHOD, not data → no contamination.
2. **RL training tasks/rewards = PUBLIC, well-created verifiable datasets ONLY** — NOT the in-house Lumo Codex-Long
   verifier pack, which is HELD-OUT EVAL (training on it = train-on-test leakage). We use the flywheel INFRASTRUCTURE
   (Codex+vLLM+verifier-runner) to run PUBLIC tasks. Dataset selection delegated to an ultracode research workflow
   (user: "just use the research, no opinion"). Held-out eval = in-house Lumo pack + SWE-bench Verified + BFCL test.
