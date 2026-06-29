# Behavior-Preserving Agentic Diffusion Recipe

Date: 2026-06-27
Last updated: 2026-06-28

## Objective

Start from a strong agentic autoregressive Qwen model and convert it into a
block-diffusion model that preserves the AR model's behavior on tool use, coding,
planning, and stop boundaries. The payoff is diffusion-style test-time compute:
larger or smaller blocks, denoising steps, confidence thresholds, constrained
decoding, and selective re-denoising for fragile spans.

This is stricter than "train an agentic diffusion model." The target is a
behavior-preserving AR-to-diffusion conversion recipe.

Working goal, refined 2026-06-27:

1. Use a good agentic AR Qwen model as the behavioral source.
2. Convert/adapt it into a block-diffusion Qwen model without losing tool-call,
   coding, planning, and stop-boundary behavior.
3. Spend extra diffusion test-time compute only where behavior is fragile:
   tool names, JSON structure, argument values, code edit hunks, and stopping.
4. Measure raw diffusion behavior separately from protected/rescued behavior.
5. Promote a recipe only when raw behavior moves toward the AR reference, not
   merely when a post-processor hides errors.

Current goal, restated 2026-06-28:

The project is a behavior-preserving conversion recipe. Assume we already have a
good agentic AR Qwen model. The experiment is to make a diffusion/block-diffusion
version that keeps the AR model's tool calling, coding, planning, argument
grounding, and stop behavior, while gaining diffusion test-time compute controls:
block size, denoising steps, confidence thresholds, selective re-denoising, and
grammar-aware constrained sampling.

That means we should not optimize for a generic "agentic diffusion model" label.
We should optimize for AR behavior retention plus measurable diffusion-side
advantages.

Goal interpretation, 2026-06-28:

The strongest framing is not "train a diffusion model until it becomes agentic."
It is:

1. pick a strong agentic AR Qwen model as the behavior source,
2. convert it into block diffusion with minimal behavior drift,
3. spend diffusion test-time compute on fragile spans only,
4. use constraints/protection to expose where the sampler should be improved,
5. progressively move those protections into generation-time constraints or
   learned boundary/value-selection components.

The closeout question is therefore behavior retention:

```text
Can DIFF_TRAIN preserve the AR model's tool/code/stop behavior while matching
or beating AR throughput through block diffusion test-time compute?
```

## Local Terms

- Learned adapter: a LoRA/QLoRA PEFT checkpoint loaded on top of the converted
  diffusion base. It changes model weights at inference time through adapter
  deltas. Current examples include checkpoint-24 and checkpoint-275 adapters.
- Rescorer/projector: deterministic post-processing that reads a draft and
  reconstructs or repairs tool calls against tool schemas and prompt evidence.
  This is not learned unless explicitly paired with a repair adapter.
- Protected tool calling: the full guarded path: generator adapter plus
  deterministic repair/projection/planning/routing. It is useful system
  scaffolding, but it is not proof that raw diffusion generation has learned the
  behavior.
- Raw score: the diffusion model's unaided output score.
- Protected score: the score after deterministic or sidecar protection.

Promotion requires raw/model-only movement, not just a better protected score.

Plain-language answer:

- The adapter is learned. If we load a LoRA/QLoRA checkpoint, the model itself
  has been changed by trained low-rank deltas.
- The rescorer/projector/repair path is mostly not learned in the current local
  prototype. It is deterministic code that parses, compares against tool
  schemas/prompt evidence, and rewrites or selects a safer output.
- Protected tool calling is a runtime system path, not a single model behavior.
  In our current usage it means "model plus guards": adapter generation,
  schedule/prefix controls, parser/repair/projection, and route-specific
  sidecars.
- Constrained decoding is also runtime/sampler control. It masks or rejects
  invalid choices during generation. It can be combined with a learned adapter,
  but it is not itself a learned adapter unless we train a boundary predictor or
  policy network.
- A learned boundary adapter would be a separate trained component that predicts
  where blocks should end or which spans need extra compute. We do not have that
  yet; our current boundary planner is rule-based over known tool-call text.

Operational distinction, 2026-06-28:

| Component | Learned? | What it can prove |
| --- | --- | --- |
| QLoRA/LoRA adapter | yes | model weights moved toward the target behavior |
| candidate-ranker/value-span adapter | yes, if trained | model or sidecar learned to choose/copy better spans |
| deterministic rescorer/projector | no | runtime code can recover a safer output from a draft |
| constrained decoder | no, unless paired with a learned policy | sampler can avoid illegal structure during generation |
| protected tool-calling path | mixed | product path works, but raw model ability must still be measured separately |
| future boundary predictor | yes | model learned where to spend block-diffusion compute |

Short answer for the local system:

- `adapter` means learned weights: LoRA/QLoRA deltas on top of the converted
  diffusion base.
- `repairer`, `rescorer`, or `projector` currently means deterministic runtime
  code: parse the draft, compare against schemas and prompt/tool-result
  evidence, then rewrite or select a safer candidate.
- `constrained decoding` means generation-time masking/verification. It is not
  a learned adapter by itself; it changes which tokens or spans the sampler is
  allowed to commit.
- `protected tool calling` is the umbrella path: adapter output plus
  constrained decoding, parser/projection, candidate forcing, stop guards, and
  optional sidecar repair. It is a useful product path and training oracle, but
  it cannot be counted as raw diffusion-model competence.

So when we report results, use three columns:

1. raw diffusion output,
2. constrained-decoding output,
3. protected/sidecar output.

Only the first two should be used as evidence that the diffusion model itself is
becoming agentic. The third is valuable engineering, but it is a system score.

Practical implication:

- If raw improves, the generator learned something.
- If constrained decoding improves, the sampler learned to keep diffusion states
  inside valid tool/code languages.
- If only protected score improves, the system is safer, but the underlying
  diffusion model may not be closer to the AR behavior source.
- If protected score is high and raw is low, mine those protected decisions into
  future training labels or sampler constraints instead of declaring victory.

Example from 2026-06-28: a context-first tool-result constrained decoder fixed
synthetic text tool-result exact arguments from `8/10` to `10/10` by copying
explicit tool-result fields such as `email_subject` and `customer_id`. This is
a protected-path improvement, not evidence that the one-step route-delta adapter
learned better raw tool-result behavior.

## Current Evidence

The current Qwen3.5-9B scorecard is a sidecar target, not a single promoted
model:

- scorecard: `qwen35_9b_split_route_sidecar_scorecard.md`
- manifest: `runs/qwen35_9b_split_route_sidecar_eval/route_manifest.json`
- all six scorecard lanes have live replay artifacts

This is valuable because it identifies which failure classes are recoverable by
protection today:

- one-call tool-call formatting and simple arguments
- multi-call sequence recovery
- OpenAI-style tool-result continuation
- text tool-result continuation

It also shows the main gap: raw diffusion behavior still trails the AR teacher
and the protected path.

## Research Map

Closest external anchors:

- Fast-dLLM v2: AR-to-block-diffusion adaptation and the main scaffold for this
  repo. https://arxiv.org/abs/2509.26328
- OPDLM / On-Policy Distillation: directly supports the behavior-preserving
  objective by training AR-to-diffusion conversion on the student sampler's own
  trajectories while distilling targets from the frozen AR teacher.
  https://arxiv.org/abs/2606.06712
- NBDiff / Next-Token to Next-Block: reinforces the direction of a principled
  AR-to-block-diffusion path with block-size growth and AR guidance.
  https://arxiv.org/abs/2512.06776
- DINGO: constrained inference for diffusion LMs, relevant to JSON/schema/tool
  grammar constraints. https://arxiv.org/abs/2505.23061
- Constrained Decoding of Diffusion LLMs with Context-Free Grammars: directly
  relevant to JSON, code, and multi-region/out-of-order diffusion constraints.
  https://arxiv.org/abs/2508.10111
- LAVE / Lookahead-then-Verify: relevant to keeping intermediate diffusion
  states completable under a CFG, not just valid after final repair.
  https://arxiv.org/abs/2602.00612
- Swordsman: entropy-driven adaptive block partitioning, relevant to dynamic
  boundary choice. https://arxiv.org/abs/2602.04399
- AdaBlock-dLLM: training-free adaptive block sizing from confidence dynamics
  and semantic-step alignment. https://openreview.net/forum?id=0Cv9PwL7cI
- DSB: dynamic sliding block scheduling, relevant to letting active blocks move
  with confidence instead of fixed boundaries. https://arxiv.org/abs/2602.05992
- SemBlock: semantic-boundary dynamic blocks with a lightweight learned boundary
  predictor over frozen dLLM hidden states. This is closest to a learned version
  of our rule-based tool-sensitive boundary planner.
  https://arxiv.org/abs/2606.04964
- CtrlDiff: supports learned or policy-driven dynamic block prediction, but its
  published target is generic controlled generation rather than tool-call-safe
  agentic execution. https://arxiv.org/abs/2505.14455
- Anchor-based history-stable decoding: relevant to committing stable structural
  anchors across diffusion blocks. https://arxiv.org/abs/2604.08964
- DiffuCoder: diffusion-specific code generation evidence and code-oriented
  objectives. https://arxiv.org/abs/2506.20639
- DreamReasoner: block-size curriculum evidence for reasoning models.
  https://arxiv.org/abs/2606.19257
- DiffuAgent / Bitter Lesson of Diffusion LMs for Agentic Workflows: cautionary
  evidence that current dLLMs are more reliable in non-causal roles than in
  precise multi-step agent loops unless causal/tool reasoning is added to the
  denoising process. https://arxiv.org/abs/2601.12979
- Gemini Diffusion: evidence that diffusion text/code models can be fast and
  capable, but its published SWE-bench number is non-agentic single-turn editing,
  not a proof of tool-loop competence.
  https://deepmind.google/models/gemini-diffusion/

I do not see a mature public recipe for tool-call-aware dynamic block boundaries.
That looks like novel work built from adaptive block partitioning plus
grammar-constrained diffusion inference.

Research implication:

- The field has strong evidence for AR-to-block-diffusion adaptation.
- The field has growing evidence for adaptive block boundaries.
- The field has separate evidence for CFG/regex-constrained diffusion decoding.
- I do not find public evidence that these are already combined for agentic
  tool calls, stop conditions, and coding-agent harnesses. That combination is
  the novel part of this project.
- Agentic evidence argues against training a generic diffusion model and hoping
  tool use emerges. The safer path is behavior preservation from AR, then
  targeted sampler/training mechanisms for causality, exact structure, and stop
  control.

Research refresh, 2026-06-27:

- Fast-dLLM v2 and NBDiff support our core conversion premise: start from AR
  weights and adapt toward block diffusion instead of training a dLLM from
  scratch.
- AdaBlock, Swordsman, and DSB support runtime dynamic blocks, but their
  published policies are generic confidence/entropy/semantic-difficulty
  policies. They do not know that a tool name, JSON key, argument ID, or stop
  boundary is more brittle than nearby prose.
- SemBlock is closest to the learned-boundary version of our idea: it trains a
  small boundary predictor over frozen dLLM hidden states. For our project, the
  labels should be agentic spans: tool tags, function names, JSON keys, scalar
  arguments, code hunk boundaries, reasoning-step boundaries, and stop points.
- DINGO, CFG-constrained dLLM decoding, and LAVE support grammar-safe sampling.
  For tool calls this should become generation-time constrained decoding, not
  just post-hoc repair.
- Gemini Diffusion and DiffusionGemma show strong public evidence for fast
  diffusion text/code generation, including code/editing and function-calling
  claims in model documentation. Gemini Diffusion's published SWE-bench
  Verified number is explicitly non-agentic single-turn editing, so it does not
  settle the agentic-loop question.
- The DiffuAgent/Bitter Lesson result is the caution flag: current dLLMs fail
  tool-calling precision and temporal-feedback loops unless causal, precise,
  grounded reasoning is added into the denoising path.

Research refresh, 2026-06-28:

- Fast-dLLM v2 and NBDiff are still the strongest anchors for this project
  because they directly support adapting AR models into block-diffusion models
  rather than starting from scratch.
- OPDLM is especially relevant to the new goal because it frames conversion as
  post-training against the frozen AR teacher on trajectories the diffusion
  student actually visits. That is closer to behavior preservation than generic
  masked-token SFT.
- DINGO, CFG-constrained decoding, and LAVE imply that tool JSON should be
  constrained during generation. Post-hoc repair is useful for diagnostics, but
  tool-call safety belongs inside the diffusion sampler.
- AdaBlock, Swordsman, and DSB show that fixed blocks are a bad default: block
  boundaries should react to confidence, entropy, semantic difficulty, and
  currently active spans.
- SemBlock is the most relevant learned-boundary template: it trains a small
  boundary predictor on frozen dLLM hidden states. Our variant should replace
  generic semantic labels with agentic labels: tool tags, tool names, JSON keys,
  scalar values, code hunk boundaries, reasoning-step boundaries, and stop
  boundaries.
- I do not see a public SOTA recipe that combines AR-to-diffusion conversion,
  tool-call grammar constraints, dynamic block boundaries, and agentic coding
  evals. That combination is the research direction here.

Research refresh, 2026-06-28 late:

- The objective should be written as behavior-preserving conversion: given a
  strong agentic AR Qwen policy, produce a block-diffusion policy that preserves
  tool/coding behavior while adding diffusion test-time compute. This is closer
  to Fast-dLLM v2, NBDiff, and OPDLM than to generic masked-token training.
- Dynamic block boundaries are not merely a speed feature. In an agent, they are
  a correctness feature. A brittle span such as a tool name, JSON separator,
  argument ID, timestamp, code hunk header, or stop token should get smaller
  blocks, more denoising steps, and stricter grammar/candidate constraints than
  ordinary prose.
- Current SOTA pieces split into two families:
  - adaptive scheduling: AdaBlock, Swordsman, DSB, SemBlock, CtrlDiff;
  - constrained diffusion decoding: DINGO, CFG-constrained dLLM decoding, LAVE.
- The project-specific idea is to fuse those families using agentic labels:
  detect tool/code/stop regions, shrink blocks there, require grammar
  completability, and distill the chosen behavior from the AR teacher.
- A learned boundary predictor should be optional v3. The first useful version
  can be rule-based plus confidence/grammar-aware. The learned version should
  train on our AR traces and failure spans, not generic sentence boundaries.

Research refresh, 2026-06-28 parallel:

- LLaDA2.0-style Warmup-Stable-Decay strengthens the case for staged
  conversion: begin near AR behavior with small blocks, grow the diffusion
  canvas, then decay back to a practical compact block size such as `32`.
- DreamReasoner-style reasoning/coding evidence argues against jumping straight
  to large blocks. A staged `4/8 -> 16 -> mixed 8/16/32` curriculum is more
  defensible than random dynamic block choices from step one.
- CFG-constrained diffusion decoding is now a direct implementation target for
  tool calls and code. The sampler should verify that partially denoised JSON,
  tool-call, and code spans remain completable, not just repair final strings.
- I-DLM-style introspective consistency gives a concrete behavior-preservation
  metric: the diffusion model should accept and stay consistent with AR
  continuations. Add token acceptance, AR top-k/KL agreement, AST equality, and
  stop-boundary agreement before scaling a larger conversion run.
- BFCL V4 is the first external tool-call benchmark to add after the local
  heldout gate moves. SWE-bench Verified should remain the expensive final
  coding-agent gate.

Local implementation status: `scripts/eval_qwen_ar_diffusion_candidate_agreement.py`
now implements the first candidate-choice agreement diagnostic. It can score
candidate indices with AR causal logprobs, Fast-DLLM masked logprobs,
Fast-DLLM causal logprobs, and compare score files. The first smoke uses the
converted Fast-DLLM init as a reference because the installed Transformers build
cannot load raw `model_type=qwen3_5` yet. `fastdllm_causal` mode is the local
AR-proxy path: it uses the converted Qwen3.5 text model in causal eval mode,
outside diffusion training/masked scoring. The next systems dependency is still
a true AR Qwen3.5/3.6 logprob path via newer Transformers, a local registered
loader, or SGLang.

## Dynamic Tool-Sensitive Block Policy

Default block diffusion treats a whole block similarly. Agentic output should
not. Tool calls are fragile, so the sampler should spend more compute and
stronger constraints only where the output is semantically brittle.

Initial policy:

| Region | Block policy | Constraint | Reason |
| --- | --- | --- | --- |
| prose / reasoning | large block, fewer steps | none/light | fluent text tolerates small variation |
| `<tool_call>` / `</tool_call>` | tiny block | literal | boundary failure breaks the action |
| `"name"` key and function value | tiny block | tool-name enum | wrong function changes behavior |
| JSON keys | tiny block | schema-key enum | wrong key breaks schema |
| scalar arguments | small block, more steps | schema + prompt evidence | IDs, dates, paths, numbers must copy exactly |
| arrays/objects | small/medium block | JSON grammar + schema | structure must remain parseable |
| stop boundary | tiny block | literal/stop set | agent loops and extra calls start here |

Prototype:

```bash
.venv-fastdllm/bin/python scripts/plan_tool_sensitive_blocks.py \
  --input-jsonl data/toolcall_eval/public_onecall_hermes_smoke.jsonl \
  --out-jsonl runs/tool_sensitive_block_plans/public_onecall_hermes_gold_blocks.jsonl \
  --limit 8
```

This script emits a char-span block plan over known assistant/tool-call text.
With `--tokenizer-path`, it also emits non-overlapping Qwen tokenizer
`token_blocks` that sampler code can consume. It is intentionally not wired into
generation yet. Its job is to make the boundary policy concrete and auditable
before sampler integration.

Dynamic boundary design:

1. Rule-based v0: if the prompt requires tool calls, force the first
   `<tool_call>` prefix, then use tiny/small blocks over tool tags, JSON keys,
   tool names, argument values, and stop tokens.
2. Confidence-aware v1: inside the sampler, shrink or extend active blocks
   based on local confidence/entropy changes. This follows the AdaBlock,
   Swordsman, and DSB direction, but with tool-call-specific priorities.
3. Grammar-aware v2: before committing a token in a JSON/tool/code span, check
   that the partial output remains completable under the target grammar/schema.
   This follows DINGO/CFG/LAVE-style constrained diffusion decoding.
4. Learned boundary v3: train a lightweight boundary predictor from AR teacher
   traces, gold tool-call spans, reasoning-step labels, code hunk spans, and
   our own failure traces. This is SemBlock-like, but with agentic labels.

Runtime rule for choosing a block boundary:

1. Start from the largest block that is acceptable for the current mode.
2. If the next span intersects a protected region, shrink the block until the
   region is isolated or aligned:
   - tool tag,
   - tool name,
   - JSON key,
   - scalar argument value,
   - code edit hunk boundary,
   - stop or next-action boundary.
3. If confidence is low or entropy shifts sharply, shrink further or add
   denoising passes.
4. If the partial output cannot be completed under the active grammar/schema,
   reject the commit or re-denoise that span.
5. Commit large blocks only when the region is not behavior-critical or when
   the constrained checks pass.

Boundary policy interpretation:

- The current boundary planner is constrained scheduling, not a learned adapter.
- DINGO/CFG/LAVE-style methods are constrained decoding: they prevent illegal
  partial or final strings under a formal language.
- SemBlock-like boundary prediction would be a learned adapter/head/policy: it
  predicts where to place blocks or where to spend more denoising compute.
- Our novel target is a hybrid: rule-based constraints for hard tool grammars,
  learned boundary/value selection for ambiguous behavior-preserving choices,
  and AR-teacher distillation to keep the diffusion model aligned with the
  original agentic policy.

Dynamic boundary choice around tool calls:

1. Detect active mode from the prefix and prompt: prose, code, tool-call JSON,
   tool-result continuation, or stop candidate.
2. If the decoder enters a tool-call region, shrink blocks around structural
   boundaries:
   - `<tool_call>` and `</tool_call>`
   - JSON braces/brackets/commas/colons
   - `"name"` and tool-name value
   - argument keys
   - scalar argument values
   - stop tokens and next-action boundary
3. In those regions, use stronger checks:
   - grammar completable under JSON/tool schema,
   - tool name in available-tool enum,
   - argument key allowed by the chosen tool schema,
   - scalar value copied or normalized from prompt/tool evidence where required,
   - no extra call after planned stop.
4. Outside those regions, let blocks grow and use fewer denoising steps.

This is not just a speed trick. It changes the error profile. Large natural-text
blocks are fine when small wording changes are acceptable; tool calls are not
fine because one wrong delimiter, tool name, ID, or stop decision changes the
agent's behavior.

For our goal, dynamic boundaries should be tool-sensitive first and semantic
second. A semantically nice block that splits `"camera_id": "front_door"` in the
wrong place is still a bad agentic block.

Current implementation status:

- The local adapter is learned model behavior.
- The local protected path is mostly runtime control: scheduled blocks,
  candidate forcing/masking, parser/projection, stop guards, and optional
  repair sidecars.
- The local block-boundary policy is not learned yet. It is rule-based and
  eval/schedule driven.
- The intended learned upgrade is a small agentic-boundary or value-ranking
  side objective, trained from AR teacher traces and failure traces, while the
  main diffusion generator is still measured raw.

## Training Recipe Implications

The model-side recipe should be staged:

1. Preserve AR behavior under diffusion conversion with low LR and
   complementary-masking objectives.
2. Use block-size curriculum: small blocks for tool/reasoning spans first,
   larger blocks only after heldout raw scores move.
3. Distill from Qwen3.6 AR teacher asynchronously into JSONL artifacts; the
   teacher and student do not need to be live at the same time.
4. Add tool-call-sensitive loss weighting only when it improves raw heldout
   exactness, not just protected scores.
5. Promote generation-time constraints from post-hoc projection into the
   diffusion sampler.
6. Compare every checkpoint against:
   - AR teacher/reference
   - diffusion init
   - raw diffusion checkpoint
   - protected diffusion checkpoint

## Next Experiments

1. Run the tool-sensitive block planner on all six scorecard lanes and summarize
   the distribution of sensitive spans.
2. Use the existing non-generating Fast-DLLM sampler trace to identify the
   exact block/small-block override points.
3. Use the opt-in scheduled full-context sampler override as the mechanical
   integration path. The first smoke proves scheduled intervals execute, but it
   still follows a thinking/prose trajectory.
4. Use first-prefix format control before further schedule tuning. The first
   forced-prefix smoke removes thinking/prose drift but exposes malformed JSON
   and stop-boundary failures.
5. Use stop-boundary guarding plus sequence-preserving projection as a protected
   baseline. The first smoke removes extra continuation and recovers exact tool
   sequence, but exact arguments still fail on timestamp normalization.
6. Add opt-in UTC timestamp normalization to the protected baseline. The first
   smoke reaches protected exact sequence and exact arguments on one public
   multi-call case while raw strict score remains `0/1`.
7. Add true generation-time JSON key/value separator constraints so malformed
   middle calls do not require projection.
8. Use target-token schedule forcing as an alignment diagnostic. Structural
   forcing alone fails because free argument spans can emit structural
   delimiters early; oracle all-span forcing proves the schedule is aligned.
9. Add generation-time delimiter guards inside scalar value spans. First smoke
   improves protected sequence recovery but raw strict score still fails on
   value-copy corruption.
10. Add candidate-constrained value decoding for timestamps, IDs, paths, numeric
   units, and enums extracted from prompt evidence. First diagnostic: exact
   deterministic extraction is only `2/7`, but target values are present in the
   candidate sets for `7/7`; use candidates as constraints, not replacements.
11. First candidate-constrained smoke: per-position candidate masks reach raw
    valid JSON and exact sequence, but can recombine candidates (`100pp`).
    Adding selected-candidate forcing for two extractor-selected spans reaches
    raw exact arguments on one public multi-call case. This is runtime
    protection, not learned model promotion.
12. Whole-candidate sequence-consistent decoding is now implemented. It fixes
    token recombination, but model-ranked candidate choice still misses paired
    values in the smoke (`end_time` picks the start timestamp). This isolates
    semantic candidate ranking as the next learned target.
13. Evidence-selected sequence protection reaches raw exact sequence and exact
    arguments on the one-case public multi-call smoke after adding paired
    start/end datetime extraction. This is a runtime scaffold target, not a
    promoted model result.
14. Generation-time tool-name constraints are now implemented for
    length-compatible available-tool candidates. The first version fixed syntax
    but committed too early on shared token prefixes; after deferring candidate
    commitment until the compatible set is unambiguous, the public multi-call
    one-case smoke reaches raw exact tool sequence and exact arguments.
15. Train/evaluate a small candidate-ranking objective or verifier head from
    AR teacher traces: given prompt, tool, argument key, and candidate set,
    pick the behavior-preserving value.
    First artifact: `public_multicall_toolname_argument_ranking_v3_12` gives
    `86/86` usable ranker examples from the 12-case public multi-call gold
    slice, split into `31` tool-name and `55` argument-value examples.
    Masked-span checkpoint-275 baseline is `80/86` overall, `31/31` on
    tool names, and `49/55` on argument values. This keeps the next learned
    pressure on row/time argument alignment rather than pure tool-name
    recognition. `prefix_only` and
    `future_masked` context modes reproduce the same `80/86`, so this is not a
    future-token leakage artifact.
    Diffusion-init comparison is now recorded in
    `qwen35_candidate_ranking_delta_result.md`: the converted diffusion init
    gets `78/86` overall and `47/55` argument values, while checkpoint-275 gets
    `80/86` overall and `49/55` argument values. This is a small learned
    adapter lift, not a protected-decoding artifact.
    A diagnostic index-selection curriculum is recorded in
    `qwen35_candidate_ranker_diagnostic_curriculum_result.md`. It trains and
    saves in a one-step QLoRA gate, but masked candidate ranking stays unchanged
    at `80/86`; use it as sidecar evidence, not as the next promoted generator
    recipe.
    A non-eval public-train candidate-ranking pipeline is recorded in
    `qwen35_public_train_candidate_ranking_result.md`. It builds `299` usable
    train-slice ranking examples and a `329` row curriculum, but the one-step
    continuation is also neutral on public-12 heldout (`80/86`, `49/55`
    arguments). This reinforces that the next model-side pressure should be
    selected-value span CE or row-grounding, not just index-instruction SFT.
    The selected-value span version is recorded in
    `qwen35_public_train_candidate_value_span_result.md`; it trains exact JSON
    target spans for argument values and still ties checkpoint-275 after one
    step. A 10-step sweep finds an early positive checkpoint at step 5:
    public-12 heldout candidate ranking improves to `81/86` and `50/55`
    arguments, and the cheap public one-call constrained argument gate improves
    from `5/8` to `8/8` without raw regression. The focused public multi-call
    gate then improves direct constrained from `7/12` / `4/12` to `8/12` /
    `5/12`, improves contextual projection from `7/12` / `7/12` to `8/12` /
    `8/12`, and ties guarded planner projection at `11/12` / `10/12`.
    Treat this as a broader positive sidegrade, not a full promotion, because
    raw public multi-call remains `1/12` sequence and `0/12` arguments. The
    remaining split-route lanes also block promotion: teacher-heldout one-call
    and OpenAI-style tool-result protected arguments regress versus the current
    routed target.
16. Full 12-case gold scheduled/protected public multi-call ablation with
    deferred tool-name candidate commitment completed:
    `runs/tool_sensitive_block_plans/public_multicall_gold_schedule_toolname_candidate_deferred_12.summary.json`.
    It reaches raw `11/12` exact tool sequence and `3/12` exact arguments;
    constrained/protected score is `12/12` exact tool sequence and `4/12`
    exact arguments. Interpretation: structural and tool-name protection is
    now mostly working, but argument grounding, row alignment, paired IDs, and
    repeated-string cleanup are still the limiting failure class.
17. Synthetic non-eval analogues for the two active public multi-call failure
    families are now built:
    `synthetic_multicall_failure_analogue_result.md`. The pure planner reaches
    `8/8` exact sequence and `8/8` exact arguments. The current conservative
    bad-draft guard repairs only `4/8` because it refuses same-length sequence
    mismatches; the new opt-in `--use-plan-on-sequence-mismatch` repairs the
    bad drafts to `8/8` / `8/8`. The public multi-call safety ablation regresses
    the active guarded planner from `11/12` / `10/12` to `9/12` / `8/12`, so
    keep the flag as a debug option only. The follow-up targeted fix adds a
    score/margin-gated `--use-safe-plan-on-sequence-mismatch`, a camera
    voice-command conflict resolver for prompts where earlier prose says to
    execute a quoted camera command "by saying" it while a later argument list
    mentions direct camera `status/mode`, and anchored nearest-code extraction
    for installation-code scoping. The synthetic safe-gate analogue reaches
    `8/8` / `8/8`, and the public protected diagnostic reaches `12/12` /
    `12/12`. This is protected planner/projection evidence, not raw model
    promotion evidence.
18. Synthetic multi-call planner distillation is now packaged and smoke-trained:
    `qwen35_synthetic_multicall_planner_distill_result.md`. The corpus has
    `24` rows from `8` non-eval synthetic cases, exact planner targets, full
    label retention, and zero public multi-call overlap. A one-step QLoRA
    continuation from checkpoint-275 trains and saves with loss `1.4956`, but
    generation is unchanged on the synthetic analogue eval: raw `1/8` sequence
    and `0/8` arguments, constrained `2/8` and `0/8`, same as checkpoint-275.
    Do not promote this adapter. The next learned step should be a longer
    retention-mixed sweep, a candidate/tool/value selector, or generation-time
    constrained decoding integration.
19. Re-run the split-route scorecard with raw and protected metrics split out.
20. Add tool-call JSON completability as a first-class sampler diagnostic.
    `scripts/diagnose_toolcall_json_completability.py` now separates complete
    JSON, incomplete-but-completable JSON, and unrecoverable JSON states in
    generated `<tool_call>` bodies. On the 12-row heldout policy-target slice,
    raw fixed `bd_size=16`, dynamic `8,16,32`, and low-pressure arg/value
    branches all have unrecoverable JSON-prefix errors on `12/12` rows, while
    projection/repair yields complete JSON but still `0/12` exact arguments.
    Result note:
    `qwen35_toolcall_json_completability_diagnostic_result.md`.
21. Add generation-time JSON-prefix guarded commits as the first sampler-side
    primitive. `scripts/eval_fastdllm_toolcall_cases.py` now has
    `--guard-tool-json-prefix`, which keeps scheduled JSON/tool intervals
    left-to-right and checks that the active `<tool_call>` body remains
    completable before commit. The first tool-tag-only one-row smoke moves raw
    output from invalid JSON / `0/1` exact sequence without the guard to valid
    JSON / `1/1` exact sequence with the guard, while exact arguments remain
    `0/1`. Result note:
    `qwen35_toolcall_json_prefix_guard_smoke_result.md`.
22. Add tool-call mode/sentinel protection as a separate sampler primitive.
    `--guard-tool-call-mode` now hard-fills only scheduled `tool_tag` tokens
    and reports mode-force counters separately from generic structure forcing.
    The first smoke prevents the prose/thinking bypass seen with JSON-prefix
    checking alone and recovers raw valid JSON plus exact sequence on the
    one-row public multi-call case. Result note:
    `qwen35_toolcall_mode_guard_smoke_result.md`.
23. Add named value/name candidate guards and run the public-12 scheduled gate.
    `--guard-tool-value-candidates` reaches raw valid JSON `12/12`, raw exact
    sequence `11/12`, and raw exact arguments `11/12` on public multi-call 12.
    `--guard-tool-name-candidates` fixes the remaining tool-name set, but
    exposes a close-tag bug: a scheduled `</tool_call>` can be forced while the
    active JSON body is still incomplete-but-completable. Result note:
    `qwen35_tool_value_name_guard_scorecard_result.md`.
24. Add close-tag completeness under `--guard-tool-call-mode` and re-run the
    public-12 scheduled gate. Mode + JSON-prefix + name + value + close
    protection reaches raw valid JSON `12/12`, exact tool sequence `12/12`, and
    exact arguments `11/12`; the completability diagnostic reports `31/31` raw
    complete JSON segments and zero invalid segments. The only remaining raw
    miss is value grounding (`location: ""` vs `location: "home"`), not route,
    schema, or JSON closure. Result note:
    `qwen35_tool_value_name_guard_scorecard_result.md`.
25. Move the close-guard stack to the heldout policy-target route. The lean
    named guard stack reaches raw valid JSON `11/12`, exact sequence `11/12`,
    and exact arguments `11/12`; the miss is nested JSON skeleton/key drift in
    `heldout_seed_multicall_0004`. Adding only
    `--force-schedule-token-kinds json_key,json_structure` reaches raw valid
    JSON `12/12`, exact sequence `12/12`, exact arguments `12/12`, and `29/29`
    complete raw JSON segments. Result note:
    `qwen35_heldout_policy_close_guard_scorecard_result.md`.
26. Build the next model-side target as skeleton-conditioned value infill, not
    another broad final-string SFT. Artifacts should include
    `skeleton_value_slots.jsonl`, `value_candidate_bank.jsonl`,
    `student_diffusion_states.jsonl`, `ar_teacher_topk_labels.jsonl`,
    `acceptance_labels.jsonl`, and `boundary_labels.jsonl`. Losses should
    emphasize value CE/ranking, restricted AR-teacher KL on student-visited
    states, evidence span selection, accept/reject consistency, boundary kind,
    and stop behavior, with low-weight masked denoising CE for skeleton
    retention.
27. First skeleton-conditioned value-infill artifacts are built:
    `qwen35_skeleton_value_infill_artifacts_result.md`. Trainable clean source:
    `data/skeleton_value_infill/public_train_no_public_smoke/`, with `45`
    filtered records, `331` usable value slots, `711` candidate rows, `4667`
    boundary labels, and `331` value-infill train instances. The source overlap
    audit has `0` exact/user overlaps against public and heldout eval slices.
    Diagnostic heldout artifacts live in
    `data/skeleton_value_infill/heldout_policy_diagnostic/` and are explicitly
    not promotion-eligible.
28. First skeleton-conditioned value-infill QLoRA sweep is complete:
    `qwen35_skeleton_value_infill_training_gate_result.md`. The staged one-file
    dataset is
    `data/qwen35_9b_skeleton_value_infill_no_public_smoke_curriculum/`. The
    one-step checkpoint-275 continuation gate saved an adapter. The 75-step
    sweep saved checkpoint adapters at steps `25`, `50`, and `75` with final
    train loss `2.6479` and no OOM on the local RTX 5090.
29. Evaluate checkpoints `25`, `50`, and `75` before promotion. Result:
    `qwen35_skeleton_value_infill_checkpoint_eval_result.md`. All three tie
    active checkpoint-275 on public and heldout multi-call guard gates, while
    checkpoint-25 shows only a small one-call raw/model-repair improvement.
    Do not promote this standalone fixed-skeleton value-answer objective.
30. Replace standalone value emission with schedule-state selector/policy
    supervision. Result:
    `qwen35_schedule_state_selector_curriculum_gate_result.md`. New builder:
    `scripts/build_schedule_state_selector_curriculum.py`. It emits `539`
    clean train instances whose labels choose the candidate index and local
    protection policy for active argument-value spans; a one-step checkpoint-275
    QLoRA gate trains and saves successfully.
31. Do not use the schedule-state selector as free assistant JSON generation.
    The `75`-step sweep from checkpoint-275 trains and saves, but active
    checkpoint-275 and selector checkpoints `25/50/75` all score `0/16` valid
    JSON and `0/16` exact decisions on a fixed ambiguous selector slice. Treat
    the selector decision as constrained control state: prefix-force the fixed
    JSON policy fields, or score candidate/policy templates with masked or
    pairwise likelihood before injecting the chosen state into the protected
    sampler.
32. Use constrained candidate-index scoring as the current selector control
    baseline. New evaluator:
    `scripts/eval_fastdllm_schedule_state_selector_ranking.py`. In
    `index_only` mode, active checkpoint-275 reaches `312/349` top-1 and
    `334/349` target top-2 on all ambiguous schedule-state rows with `0`
    runtime errors. The selector-SFT checkpoints tie the active adapter on the
    first `64` ambiguous rows, so promote the scorer/injector pattern rather
    than those checkpoints.
33. Inject ranked selector choices into the sampler as rank-1 control first.
    New bridge: `scripts/inject_schedule_state_selector_ranking_choices.py`.
    A four-case no-public-smoke generation smoke gives rank-1 raw valid JSON
    `4/4`, exact sequence `4/4`, exact arguments `3/4`; rank-2 keeps sequence
    `4/4` but drops exact arguments to `2/4`. Do not assume top-2 fallback
    helps unless a separate rerank/repair policy chooses the final value.

Promotion gate for this line:

- raw public one-call exact arguments improve over checkpoint-24
- raw teacher-heldout exact arguments improve without protected regression
- public multi-call repeated/extra/missing-call rate decreases
- protected path remains at or above the current six-lane scorecard gates
- raw or constrained sampler output avoids unrecoverable JSON states inside
  tool-call spans before final projection/repair is applied
- tool-call mode/sentinel protection prevents prose from bypassing the active
  JSON-prefix guard
- closing `</tool_call>` sentinels are committed only when the active JSON body
  is complete, not merely prefix-completable
- public multi-call structural sampler gate reaches raw valid JSON and exact
  tool sequence `12/12` before moving the same guard stack to heldout policy
  targets
- heldout policy structural ceiling reaches `12/12` only with explicit
  `json_key,json_structure` protection; promotion requires replacing that
  oracle with model-side skeleton stability or constrained decoding that is not
  final projection
