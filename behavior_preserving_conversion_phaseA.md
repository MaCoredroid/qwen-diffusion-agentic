# Behavior-Preserving Conversion — Phase A: Behavior-Retention Snapshot

Date: 2026-06-28
Decision (lead's call): step back from tool-call argument micro-optimization to the bigger
**behavior-preserving AR→diffusion conversion** goal. Justification: iteration 1 of the causal
value-span test (see `causal_value_span_grounding_experiment.md`) showed that GIVEN correct
structure, RAW value-fill is already 7/12 heldout / 10/12 public (not ~0/12) — structure was most
of the historical wall — and the residual misses are reasoning / convention / selection / cross-call,
NOT diffusion verbatim-copy. The value problem is smaller and differently shaped than feared.

Roles unchanged: **monitor** (Claude) writes/steers/red-teams; **Codex** executes; the
raw / constrained / protected promotion discipline holds.

## The bigger goal (restated, from qwen36_diffusion_closeout_metrics.md)

A converted block-diffusion Qwen that PRESERVES the AR model's broad competence — code,
instruction-following, reasoning, tool/agentic, stop behavior — while gaining diffusion test-time
compute. Success = `DIFF_TRAIN` retention vs `AR` across the closeout battery, raw/constrained
measured separately, with a >=1.3x speed gate. We have **never measured broad retention** — all
prior work was tool-call-specific. Phase A fixes that.

## Phase A, step 1 — Behavior-Retention Snapshot (this experiment)

Question: how much of the AR Qwen's BROAD competence did the conversion preserve, RAW — and did
tool-call SFT (checkpoint-275) erode it (catastrophic forgetting)?

Measure (RAW diffusion vs AR reference; raw/constrained separate; INFERENCE-ONLY):

1. **Generic battery** on small fixed slices (sized for the RTX 5090):
   - Code: HumanEval or MBPP subset (~20-40 problems).
   - Instruction-following: IFEval subset.
   - Reasoning: GSM8K subset (cheap); optional short MMLU slice.
   Conditions: **AR reference** (Qwen3.5-9B 4-bit AR, and/or the Qwen3.6-27B SGLang teacher as the
   stronger anchor) vs **DIFF_INIT** (converted diffusion, `--no-adapter`) vs **checkpoint-275** —
   RAW (and constrained/repair where it applies).
   Gates (closeout doc): diffusion >=70% of AR on code; >=75% of AR on IFEval. Report retention ratios.

2. **Catastrophic-forgetting check**: checkpoint-275 vs DIFF_INIT on the generic battery. If
   ckpt-275 < DIFF_INIT, the tool-call SFT eroded broad behavior — a central behavior-preservation
   finding that reframes whether the current adapter line is a real conversion or a narrow overfit.

3. **Bridge de-risk / AR-agreement**: on a clean general slice, run the existing
   `scripts/eval_qwen_ar_diffusion_candidate_agreement.py` (fastdllm_causal vs fastdllm masked; vs a
   true AR path if loadable) to confirm the diffusion bridge is not silently degrading next-token
   behavior. Note explicitly that the no-GDN-cache full-context path is the correctness path.

Deliverable: a retention table {slice x (AR, DIFF_INIT raw, ckpt-275 raw [, constrained])} +
retention ratios vs gates + the forgetting verdict + the agreement summary. No promotion claims.

Guardrails: inference-only (no retraining); reuse existing AR baselines + venvs (`.venv-lmeval`,
`.venv-fastdllm`, SGLang teacher); small fixed slices; **checkpoint with the monitor** (slices, AR
reference, exact commands) BEFORE any long run — same discipline that caught the leakage/confound in
iteration 1.

## Roadmap (the bigger goal)

- **Phase A (now):** behavior-retention snapshot + bridge de-risk [this doc].
- **Phase B:** if retention is weak or ckpt-275 forgot — run a proper conversion recipe:
  OPDLM-style on-policy distillation from the 27B AR teacher (research-verified, ~0.07B tokens, fits
  the 5090) + block-wise causal attention + block-size curriculum. Put real tool/agentic traces in
  the on-policy prompt set (research caveat: OPDLM alone does NOT fix format fidelity). Measure
  retention lift vs Phase A.
- **Phase C:** closeout gates (tool-call, code-edit, then SWE-bench Verified slices) per
  `qwen36_diffusion_closeout_metrics.md`, with raw/constrained/protected separation and the speed gate.

## Phase A run note — 2026-06-28 (monitor red-team)

First battery attempt produced a BROKEN AR denominator and was stopped mid-diffusion-run.

- AR could not load 4-bit: Transformers 4.53.1 cannot load `model_type=qwen3_5`. AR now runs raw
  **bf16 in `.venv`** (Transformers 5.12.1). The diffusion conditions run via fast-dllm eval.py in
  `.venv-lmeval`. (Cross-venv; standardized lm_eval metrics keep it comparable — verify prompt parity.)
- AR scores came back ~0 (MBPP 0/20, GSM8K 0/20 strict, IFEval 1/20). Root cause from the AR gsm8k
  sample: Qwen3.5-9B emits a verbose "Thinking Process" preamble and the generation is **truncated by
  `max_gen_toks` before the final `#### answer`**, so lm_eval extraction returns `[invalid]`. The model
  reasons CORRECTLY (it computed the right answer) — this is a generation-config failure, not AR
  competence.
- Fix required before any re-run: disable thinking (`chat_template_kwargs enable_thinking=false`), raise
  `max_gen_toks` generously, verify extractors parse, **RE-VALIDATE AR gets sane numbers (GSM8K >0.6,
  MBPP >0) as a GATE**, and apply the IDENTICAL gen-config to AR and DIFF.
- Lesson: validate the REFERENCE denominator produces sane numbers before spending hours on the slow
  diffusion conditions. (Second red-team catch that saved significant compute, after the iteration-1
  leakage/confound.)
