# Causal Value-Span Grounding Experiment (Week-1 decisive test)

Date: 2026-06-28
Roles: **Monitor** (Claude) owns this plan + red-teams outcomes and never runs the
experiment. **Executor** (Codex, gpt-5.5) implements and runs it. Promotion
decisions are made by the monitor against the raw/constrained/protected discipline.

## Why (hypothesis)

The argument-value grounding wall (raw exact arguments ~0/12 on the heldout policy
slice) is most likely NOT a weight-capacity problem and NOT fixable by more SFT.
Evidence:

1. The AR Qwen3.5/3.6 GDN models copy exact arguments fine (Qwen3.6-27B teacher
   48/48 synthetic, 18/24 public; 9B 4-bit AR 13/24). Same GDN architecture.
2. The converted diffusion model's **causal and masked next-token predictions
   already agree** (candidate-agreement diagnostic) — the model "knows" the right
   token; the gap was attributed to sampler/serialization/commit behavior.
3. External research: (a) masked diffusion's per-step **factorization barrier**
   means paired values drawn in the same parallel step cannot be made mutually
   consistent; (b) **Gated DeltaNet linear-attention has a recall/copy ceiling** —
   verbatim copy is hosted by the 1-in-4 full-attention layers and is organized
   **causally** (induction heads). Parallel/bidirectional within-block denoising is
   exactly what disrupts that causal copy circuit.

**Prediction:** decoding value/ID/timestamp/path spans in a near-AR regime — tiny
block (block_size 1-2), strict left-to-right commit, full prompt (evidence) visible,
paired values decoded anchor-first, and **no value forcing** — should restore RAW
exact-argument copying. This is genuinely untried: prior runs used fixed bd_size=16,
dynamic 8/16/32 (raw 0/12), or value *forcing* (11-12/12). Letting the model
*generate* the value token-by-token causally with the source visible has not been
isolated.

This experiment is the make-or-break test of that diagnosis.

## What to run (the decisive test)

Hold model + slices fixed; vary ONLY the value-span decoding regime. Keep
**raw / constrained / protected** columns separate.

- Base model: `models/qwen3.5-9b-fastdllm-init`
- Adapter: `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model` (anchor; **DO NOT retrain or modify**)
- Tokenizer: `models/qwen3.5-9b-fastdllm-init`
- venv: `.venv-fastdllm`
- Scorer: the script's own `--eval` exact-sequence / exact-arguments / valid-JSON metrics.

Slices (bind exact files and report them):
- **Heldout policy-target 12** — the clean promotion gate (derived from
  `data/toolcall_eval/heldout_seed_multicall_2to3_clean.jsonl`; use the same
  policy-target input the close-guard scorecard used). This is the PRIMARY (clean) gate.
- **Public multi-call 12** (`data/toolcall_eval/public_multicall_hermes_smoke.jsonl`)
  as the familiar comparison. Note: 11/12 of these overlap train — fine for an
  inference-only test, but the heldout slice is the one that counts.

Conditions (identical otherwise):

1. **RAW-BASELINE** — honest raw lane. No `--guard-tool-value-candidates`, no
   `--force-best-candidate-sequence`, no `--force-selected-candidate-tokens`.
   Confirm raw exact args ~0/12. (This re-establishes the floor with the current sampler.)

2. **CAUSAL-VALUE-SPAN** — the new mechanism. For argument-value / ID / timestamp /
   path spans: block_size = 1-2, strict left-to-right commit, `--full-context-sampling`
   (prompt/evidence visible), paired values decoded anchor-first, skeleton guards ON
   (`--guard-tool-call-mode --guard-tool-json-prefix --force-tool-call-prefix
   --stop-after-schedule-tool-calls`), but **value-candidate forcing OFF** — the model
   must GENERATE the value. Prose spans keep large blocks. Measure RAW exact args /
   exact seq / valid JSON.

3. **FORCED-CEILING** (reference) — the existing guard stack WITH
   `--guard-tool-value-candidates` / `--force-best-candidate-sequence`. This is the
   known 11-12/12 upper bound and is a **protected** number — label it as such.

Secondary instrument (attribution + bridge bug check):
- `scripts/eval_qwen_ar_diffusion_candidate_agreement.py` `--mode compare` between
  `fastdllm_causal` and `fastdllm` (masked) scores on the value spans. If a true AR
  logprob path is available (27B SGLang teacher, or 9B 4-bit via
  `eval_transformers_toolcall_cases.py`), compare value-span logprobs too.

## Decision rule (what each outcome means)

- **Raw exact args materially > 0/12** (target >= 4/12, ideally trending toward the
  forced ceiling): hypothesis CONFIRMED. Proceed to Week-2 — build the tool-sensitive
  adaptive-block sampler properly (LAVE-style completable-grammar checks, anchor-first
  paired-value ordering, then AR-teacher verifier for derived values).
- **Raw args ~0/12 but FORCED-CEILING still 11-12/12**: the copy path is too damaged
  by conversion. Selection is then the legitimate mechanism → Week-2 pivots to
  constrained candidate selection (DINGO distribution-preserving) + AR-teacher (EDLM)
  verifier; reconsider keeping GDN purely causal / adding attention-layer copy capacity.
- **causal vs masked (or vs AR) logprobs DIVERGE on value spans**: suspect a silent
  GDN recurrent-state / sampler bug (cf. vLLM #39273, hybrid-GDN Qwen3.5 spec-decode
  state corruption) → debug bridge state handling BEFORE any more training.

## Implementation notes (Codex decides; monitor reviews)

- Per-span block geometry is controlled by the per-interval sampler schedule
  (`--sampler-schedule-jsonl`: each interval carries `block_size` + `denoise_steps`)
  plus `--block-size` / `--small-block-size`. Emit value-span intervals at
  `block_size` 1-2. **Prefer schedule-only (no code change).** If the base sampler
  cannot honor `block_size=1` left-to-right per value position, make the MINIMAL
  sampler change and document the diff for review.
- The schedule emitter is `scripts/emit_tool_sensitive_sampler_schedule.py`; block
  plans come from `scripts/plan_tool_sensitive_blocks.py`. Reuse them.

## Hard guardrails (do not violate)

- Inference-time experiment ONLY. Do not retrain or modify checkpoint-275.
- NEVER let gold assistant calls/args leak into the prompt or via forcing.
  `--full-context-sampling` exposes the PROMPT only (legitimate — the AR model also
  sees the prompt). The monitor will audit for leakage.
- Use overlap-clean inputs; no train/eval contamination.
- Report raw / constrained / protected SEPARATELY. Do NOT declare promotion — the
  monitor decides promotion, and only on raw or distribution-preserving-constrained
  movement.

## Deliverables for red-team

- A result note `.md`: a 3-conditions x 2-slices table (raw valid JSON / raw exact
  seq / raw exact args; plus constrained/protected where they apply), the exact
  commands, log paths, any code diff, and the per-row CAUSAL-VALUE-SPAN failures
  (which values copied vs not, and how the wrong value related to the gold — e.g.
  start vs end time, adjacent row id).
- The causal-vs-masked agreement summary on value spans.
- Codex's read of which decision-rule branch the data supports — as a PROPOSAL.

## Checkpoint before the long runs

Before launching full-context sampling (slow), Codex replies with: (a) the exact
eval input files it will use, (b) the three exact commands, (c) its implementation
approach for the causal value-span regime. The monitor red-teams the setup, THEN
Codex runs.

## Iteration 1 outcome — 2026-06-28 (monitor red-team)

raw_baseline + causal complete; forced_ceiling (protected) still completing.
Non-leakage VERIFIED: all force/candidate/target/json-prefix-fallback counters = 0 for raw and causal.

RAW exact-arguments (headline, pre-projection):
- heldout policy-12: RAW-BASELINE 7/12, CAUSAL 7/12 (seq 11/12, names 12/12)
- public multicall-12: RAW-BASELINE 10/12, CAUSAL 10/12 (seq 11/12)

Findings:
1. CAUSAL == RAW-BASELINE, byte-identical generation (empty diff both slices). The 1-token causal
   value-span manipulation was INERT at temp 0 (baseline already left-to-right across spans / value
   tokens flow through the default block path). The causal-within-span hypothesis is UNtested, not refuted.
2. RAW value-fill GIVEN correct skeleton is 7/12 (heldout) / 10/12 (public), far above the historical
   ~0/12. Most of the historical wall was STRUCTURE; given structure, raw values are already decent.

Miss taxonomy (heldout 5 misses) — NONE are broken verbatim-copy:
- 0002: convention/derived — growth_rate 0.15 vs gold 15 (percent), capitalization, weights 0.33 vs 0.333.
- 0003: wrong SELECTION from context — device_001/smart_lock vs gold device_002/smart_light.
- 0004: decomposition policy (1 call vs gold 2) + year hallucination 2024 vs 2023.
- 0007: cross-call runtime dependency — gold session_id = "use_id_from_create_trivia_game_session".
- 0009: derived/convention — event_id MAD20230615 vs MSG-150623, card spacing, refund partial vs full.
(Early forced_ceiling: even candidate-forcing does not crush these, consistent with reasoning/convention.)

PIVOT: given structure, the residual argument failures are REASONING / CONVENTION / SELECTION / CROSS-CALL,
not diffusion parallel-copy. The causal-decoding mechanism (and the GDN copy-ceiling worry) is aimed at the
wrong failure mode for these misses. Right levers: (a) candidate SELECTION from context; (b) derived-rule /
AR-teacher reasoning for policy/convention; (c) a cross-call placeholder convention; (d) an eval that
separates arguably-correct-but-nonmatching from wrong (gold penalizes plausible answers, e.g. 0.15 vs 15).
N is small (5 heldout + 2 public misses) — confirm on more rows before committing. Status: DRAMATIC, escalated.
