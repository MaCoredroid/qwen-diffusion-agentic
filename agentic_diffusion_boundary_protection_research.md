# Agentic Diffusion Boundary and Protection Research Note

Date: 2026-06-28

## Working Objective

The project objective is behavior-preserving AR-to-diffusion conversion:

```text
Given a strong agentic autoregressive Qwen model, make a block-diffusion Qwen
version that preserves tool calling, code editing, planning, argument grounding,
and stop behavior, while adding diffusion test-time compute.
```

This is stricter than "train or distill an agentic diffusion model." The source
of behavior is the AR model. The diffusion model should keep that behavior and
use block size, denoising steps, confidence thresholds, constrained decoding,
and selective re-denoising as test-time compute controls.

## Component Taxonomy

| Component | Learned? | Local meaning | What it proves |
| --- | --- | --- | --- |
| LoRA/QLoRA adapter | yes | Weight deltas on the converted diffusion model | The generator weights moved |
| Candidate/value adapter | yes, if trained | A small objective or adapter for choosing/copying fragile spans | A model component learned a selection/copying behavior |
| Rescorer/projector/repairer | usually no | Deterministic code that parses a draft, checks schema/evidence, and rewrites or selects safer calls | The system can recover behavior, not that raw generation learned it |
| Constrained decoder | no by default | Sampler masks/rejects invalid tokens or spans during generation | The sampler can keep output inside a valid grammar/schema |
| Protected tool calling | mixed | Adapter output plus constraints, parser, projection, route guards, and optional sidecar repair | End-to-end path works, but it is a system score |
| Boundary predictor | future learned component | Predicts where blocks should end or where extra denoising should be spent | The model learned where generation is behavior-critical |

Interpretation:

- The adapter is learned.
- The repairer/rescorer/projector is currently runtime code, not learned.
- Constrained decoding is sampler control, not a learned adapter unless the
  constraint policy is predicted by a trained component.
- Protected tool calling is the umbrella runtime path. It may contain a learned
  adapter, but the protection itself is not automatically model learning.

Therefore every result should keep these columns separate:

1. AR reference score.
2. Raw diffusion score.
3. Constrained-decoder diffusion score.
4. Protected/sidecar score.

Only columns 2 and 3 count as direct evidence that the diffusion model/sampler
is preserving AR behavior. Column 4 is still useful because it identifies what
to turn into training labels, constraints, or a learned boundary/value policy.

## Dynamic Block Boundary Idea

Fixed block size is a poor default for agents. Ordinary prose can tolerate
larger blocks and variation; tool calls cannot. One wrong tool name, JSON key,
ID, path, timestamp, or stop boundary changes behavior.

Initial policy:

| Region | Boundary policy | Constraint policy |
| --- | --- | --- |
| prose/reasoning | larger blocks | light or none |
| tool-call tag | tiny literal block | exact literal |
| tool name | tiny block | enum over available tools |
| JSON key | tiny block | enum over schema keys |
| scalar argument | small block, more steps | schema plus prompt/tool-result evidence |
| code hunk/path | small aligned block | syntax and repository evidence |
| stop/next-action boundary | tiny block | stop set and no-extra-call guard |

Runtime rule:

1. Start with the largest block allowed by the current mode.
2. If the span intersects a tool, JSON, code, or stop-sensitive region, shrink
   or align the block boundary to isolate that region.
3. If confidence is low or entropy changes sharply, shrink further or add
   denoising passes.
4. If the partial output cannot be completed under the active grammar/schema,
   reject the commit or re-denoise the span.
5. Commit large blocks only when checks pass or the region is not behavior
   critical.

This can start rule-based. The learned version is a SemBlock-like boundary
predictor, but trained on agentic labels rather than generic sentence or
semantic units.

## SOTA Research Map

AR-to-diffusion conversion:

- Fast-dLLM v2 adapts pretrained AR models into block-diffusion LLMs with about
  1B fine-tuning tokens and reports up to 2.5x speedup while preserving quality.
  Source: https://arxiv.org/abs/2509.26328
- NBDiff / "From Next-Token to Next-Block" supports a principled path from AR
  next-token models to block diffusion with causal context and intra-block
  bidirectional attention. Source: https://arxiv.org/abs/2512.06776
- OPDLM frames AR-to-DLM conversion as post-training with on-policy
  distillation from the frozen AR teacher on student-visited trajectories,
  directly targeting train/inference mismatch and AR knowledge retention.
  Source: https://arxiv.org/abs/2606.06712
- LLaDA2.0 and related conversion recipes support staged conversion rather
  than jumping directly to large blocks: warm up from smaller blocks, stabilize
  at larger diffusion windows, then choose the serving block profile that best
  preserves behavior. Source: https://arxiv.org/abs/2512.15745
- DreamReasoner reports that reasoning is brittle under large-block diffusion
  training and argues for small-to-large block curricula. For tool agents, this
  maps to tiny protected blocks around tools/code first, then larger blocks for
  prose once retention is stable. Source: https://arxiv.org/abs/2606.19257

Adaptive/dynamic blocks:

- AdaBlock-dLLM is training-free adaptive block sizing driven by confidence
  dynamics and semantic-step alignment. Source:
  https://arxiv.org/abs/2509.26432
- Swordsman uses entropy shifts to choose block boundaries closer to semantic
  and syntactic constituents. This supports using confidence/entropy changes
  as a runtime signal, but it does not by itself solve tool-schema correctness.
  Source: https://arxiv.org/html/2602.04399v1
- SemBlock trains lightweight boundary predictors on frozen dLLM hidden states
  and uses boundary probabilities to choose dynamic block endpoints. Its labels
  are discourse, reasoning-step, and implementation spans. Source:
  https://arxiv.org/abs/2606.04964

Constrained diffusion decoding:

- DINGO is constrained inference for diffusion LLMs with regular-expression
  constraints, aimed at outputs such as fixed-schema JSON. Source:
  https://arxiv.org/abs/2505.23061
- CFG-constrained diffusion decoding extends this to context-free grammars and
  multi-region infilling, with JSON/C++ examples. Source:
  https://arxiv.org/abs/2508.10111 and https://constrained-diffusion.ai/
- LAVE-style validation is especially relevant because it treats partial
  outputs as completable or not under a formal constraint. That is the right
  abstraction for deciding whether a diffusion commit may close a tool-call
  block. Source: https://arxiv.org/abs/2602.00612

Agentic caution/evidence:

- DiffuAgent / "Bitter Lesson" reports that current dLLMs are weak agentic
  backbones for embodied and tool-calling workflows unless causal, precise, and
  grounded reasoning is added into the denoising process. Source:
  https://arxiv.org/abs/2601.12979
- Mercury, Dream-Coder, DiffuCoder, and broader diffusion-vs-AR studies give
  evidence that diffusion LLMs can be useful for coding, structured generation,
  and any-order infilling. The evidence for full multi-step tool-use and
  agentic repo-edit loops is still much thinner than the evidence for code and
  JSON-like structure. Sources: https://arxiv.org/html/2506.17298v1,
  https://arxiv.org/html/2509.01142v1, https://arxiv.org/abs/2506.20639,
  https://arxiv.org/html/2509.11252v1
- DiffusionGemma claims native function calling, coding/reasoning support, and
  fast block generation, but this is not yet a full public recipe for converting
  a Qwen agent while preserving behavior on coding-agent loops. Sources:
  https://ai.google.dev/gemma/docs/diffusiongemma/model_card and
  https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/
- BFCL V4 is the right external function-calling gate once local public/heldout
  tool-call slices stop moving. It should be used as a heldout benchmark, not
  as the first inner-loop training target. Source:
  https://gorilla.cs.berkeley.edu/leaderboard.html

Training targets beyond vanilla masked CE:

- Use masked CE only as the base denoising objective.
- Add AR-teacher KL/top-k agreement on student-visited diffusion states,
  especially tool names, JSON keys, scalar values, file paths, code hunk
  headers, and stop markers.
- Add introspective acceptance targets: committed tokens/spans should remain
  accepted by the student when reintroduced as prior context.
- Add repair-denoising targets from malformed student outputs, but only when
  the corrected span can be traced to prompt/tool-result evidence.
- Add verifier/ranker rewards for exact tool AST, schema validity, grounded
  arguments, and code tests.

Architecture-preserving conversion:

- Efficient-DLM argues for block-wise attention rather than fully
  bidirectional conversion because it better preserves pretrained AR weight
  behavior. This matches the Qwen3.5/3.6 GDN constraint: keep causal state
  across committed blocks and relax order only inside small, checked windows.
  Source: https://arxiv.org/abs/2512.14067

## Research Gap

The public literature now has pieces for:

- AR-to-block-diffusion conversion,
- adaptive block sizing,
- grammar-constrained diffusion decoding,
- diffusion code/function-call claims,
- agentic failure analyses.

I do not see a mature public recipe that combines all of these for
behavior-preserving agentic tool/coding execution. The novel recipe should be:

1. Convert a strong AR Qwen policy into block diffusion.
2. Distill on student-visited trajectories from the frozen AR teacher.
3. Use rule-based tool-sensitive blocks first.
4. Add grammar/schema-constrained diffusion decoding inside tool/code spans.
5. Mine protected-path successes into learned value and boundary objectives.
6. Promote only when raw or constrained diffusion scores improve, not when a
   post-hoc repairer hides errors.

## Local Implication

The local protected path is scaffolding and diagnostics. It should help us find
fragile spans and generate labels, but it should not be counted as final model
competence.

The next research target should be a behavior-preserving loop:

1. AR Qwen reference produces or scores traces.
2. Diffusion student samples in its own block-diffusion mode.
3. Teacher/logit/value labels correct student-visited failures.
4. Tool-sensitive blocks and grammar constraints keep fragile spans valid.
5. Evaluation reports raw, constrained, and protected scores separately.

This keeps the project aimed at a real converted agentic diffusion model rather
than a generic distilled model plus a large repair wrapper.

## Local Completability Diagnostic

The first local tool-call JSON completability diagnostic is recorded in
`qwen35_toolcall_json_completability_diagnostic_result.md`.

Key result on the 12-row heldout policy-target slice:

- raw fixed `bd_size=16`, dynamic `8,16,32`, and low-pressure arg/value outputs
  all have unrecoverable JSON-prefix errors on `12/12` rows;
- fixed `bd_size=16` raw has `5/30` complete JSON segments and `25/30` invalid
  segments;
- dynamic `8,16,32` raw has `5/38` complete segments, `2/38`
  incomplete-but-completable segments, and `31/38` invalid segments;
- low-pressure arg/value raw has `10/45` complete segments, `4/45`
  incomplete-but-completable segments, and `31/45` invalid segments;
- projection and scalar sidecars make JSON complete, but exact arguments remain
  `0/12`.

Local implication: the next meaningful innovation is not more final-string
repair. It is a diffusion commit rule that keeps partially filled tool-call
JSON completable under the active grammar/schema, shrinks blocks around
tool-call sentinels and scalar values, and separates skeleton route/order from
value infill.

## Local Prefix Guard Smoke

The first commit-time primitive is now implemented as
`--guard-tool-json-prefix` in `scripts/eval_fastdllm_toolcall_cases.py`; result
note: `qwen35_toolcall_json_prefix_guard_smoke_result.md`.

One-row public multi-call comparison with only tool tags forced:

- no guard: raw JSON invalid because a value string crosses a newline/tool-call
  boundary, raw exact sequence `0/1`;
- guard: raw JSON complete for `3/3` tool-call segments, raw exact sequence
  `1/1`;
- both: exact arguments remain `0/1`.

This supports the research thesis but also narrows it: grammar-completable
commit checks can stabilize route/structure once the model is in tool-call
mode. They do not by themselves select grounded argument values or force the
model to enter tool-call mode instead of prose. The next innovation should
combine sentinel/mode protection, grammar-prefix checks, schema-key masks, and
value-candidate infill.

## Local Mode Guard Smoke

The sentinel/mode companion is now implemented as `--guard-tool-call-mode` in
`scripts/eval_fastdllm_toolcall_cases.py`; result note:
`qwen35_toolcall_mode_guard_smoke_result.md`.

One-row public multi-call comparison:

- JSON-prefix only: model emits prose/thinking first, raw tool-call segments
  `0`, raw exact sequence `0/1`;
- mode + JSON-prefix: scheduled `tool_tag` tokens are forced as a named mode
  guard, raw complete JSON segments `3/3`, raw exact sequence `1/1`;
- exact arguments remain `0/1`.

This decomposes the protected sampler into clearer mechanisms:

1. tool-call mode/sentinel protection,
2. grammar-prefix commit checking,
3. schema/key masking,
4. value-candidate grounding.

Only the first two are implemented locally. The next frontier is schema-aware
value infill that preserves exact IDs, timestamps, paths, numbers, and paired
arguments without relying on final projection.

## Local Value/Name Guard Scorecard

Named value/name candidate guards are now implemented; result note:
`qwen35_tool_value_name_guard_scorecard_result.md`.

Public multi-call 12:

- mode + JSON-prefix + value guard: raw valid JSON `12/12`, exact sequence
  `11/12`, exact arguments `11/12`;
- adding name guard: tool-name set improves to `12/12` and constrained sequence
  improves to `12/12`, but raw valid JSON drops to `11/12` because a closing
  `</tool_call>` is committed while a string value is incomplete.

This is a useful decomposition:

- value infill is now strong enough to fix the timestamp/value miss on this
  scorecard;
- remaining raw failures are route/close-boundary mechanics, not broad JSON
  syntax or scalar extraction;
- close tags need a stricter condition than ordinary prefix-completability:
  the active JSON body must be complete before the closing sentinel is forced.

## Local Close-Guard Scorecard

Close-tag completeness is now implemented under `--guard-tool-call-mode`:
scheduled tool-call sentinel forcing is deferred when an active `<tool_call>`
body has started JSON but the JSON body is not yet complete. On public
multi-call 12 with mode + JSON-prefix + name + value + close protection:

- raw valid JSON reaches `12/12`;
- raw exact tool-name set reaches `12/12`;
- raw exact tool sequence reaches `12/12`;
- raw exact arguments reach `11/12`;
- the diagnostic reports `31/31` raw complete JSON segments and zero invalid
  segments;
- the close guard fires once, exactly on the prior truncation case.

The remaining raw miss is not syntax or route. Gold expects an empty
`location: ""` for the third voice-command call, while the model fills
`location: "home"`. This supports the current decomposition: constraints can
own grammar and closure, while learned value grounding must own benchmark-exact
argument selection.

## Heldout Structural-Key Result

On the 12-row heldout policy-target route, the same named guard stack reaches
raw valid JSON `11/12`, exact sequence `11/12`, and exact arguments `11/12`.
The miss is not a close-tag or value-choice problem. It is nested JSON skeleton
drift in `heldout_seed_multicall_0004`, where keys and punctuation inside a
large `campaign_details` array become unrecoverably invalid after `83`
JSON-prefix rejections and unsafe fallbacks.

Adding explicit `json_key,json_structure` schedule forcing while keeping named
mode/name/value guards reaches raw valid JSON `12/12`, exact sequence `12/12`,
exact arguments `12/12`, and `29/29` complete raw JSON segments. This is the
current heldout protected ceiling. It should be treated as a label source and
diagnostic, not as raw model competence.

The research implication is a two-layer behavior-preserving recipe:

1. skeleton stability: tags, tool names, JSON keys, punctuation, array/object
   boundaries, and stop points should be constrained or learned as a
   boundary/skeleton policy;
2. value grounding: argument values should be trained as evidence-conditioned
   slots under that skeleton, with candidate banks and AR-teacher correction on
   student-visited diffusion states.

## Skeleton-Conditioned Value Infill Target

Next artifacts:

- `skeleton_value_slots.jsonl`: tool sequence, JSON skeleton, `json_path`,
  tool-call index, slot range, schema type, target value, and target tokens.
- `value_candidate_bank.jsonl`: prompt/tool-result/schema/prior-call
  candidates, evidence spans, normalized values, tokenization, and path-aware
  keys.
- `student_diffusion_states.jsonl`: on-policy student states, committed prefix,
  active masks, block kind, confidence/entropy, candidate scores, and
  grammar-completability.
- `ar_teacher_topk_labels.jsonl`: restricted top-k/candidate scores from the AR
  teacher over candidate tokens, target tokens, JSON boundaries, schema tokens,
  and stop tokens.
- `acceptance_labels.jsonl`: accept correct grounded values, reject wrong
  plausible values, and reject JSON-incompletable fills after self-conditioning.
- `boundary_labels.jsonl`: labels for prose, tool tag, tool name, JSON key,
  JSON structure, argument value, code/path, stop boundary, recommended block
  size, and re-denoise/shrink flags.

Loss mix:

- low-weight masked denoising CE for skeleton retention;
- argument-value CE and whole-candidate ranking;
- restricted AR-teacher KL/top-k agreement on student-visited states;
- evidence-span selection;
- accept/reject consistency under self-conditioning;
- boundary kind/block-size prediction;
- stop-boundary loss for no-extra-call behavior.

Promotion should require raw or constrained model movement, not only protected
oracle parity.

First implementation:

- result note: `qwen35_skeleton_value_infill_artifacts_result.md`
- builder: `scripts/build_skeleton_value_infill_artifacts.py`
- trainable clean set:
  `data/skeleton_value_infill/public_train_no_public_smoke/`
- diagnostic heldout set:
  `data/skeleton_value_infill/heldout_policy_diagnostic/`

The trainable set uses the filtered no-public-smoke source and has `45`
records, `331` usable value slots, `711` candidate rows, `4667` boundary
labels, and `331` value-infill train instances. The source overlap audit checks
`85` filtered train records against `37` public/heldout eval records and finds
`0` exact overlaps and `0` user overlaps. The heldout diagnostic set has `123`
usable slots and is marked `promotion_allowed=false`.

## Hardware-Mapped Experiment Backlog

- 5090: main 9B QLoRA/sampler loop, 10/25/100-step curves, public and heldout
  tool-call scorecards, and short BFCL slices.
- GB10: memory-heavy Qwen3.6-27B FP8/NVFP4 teacher loading, long-context teacher
  scoring, export/quantization correctness, and slower async label generation.
- 5080: small eval workers, preprocessing, 1.5B/4B ablations, BFCL slices when
  available, and non-critical data generation.

Concrete next experiments:

1. Run the six-lane split-route scorecard with the close/key-structure guard
   stack reported separately as a protected structural ceiling.
2. Train on-policy AR-teacher distillation examples from student diffusion
   trajectories, with KL/top-k targets on fragile spans.
3. Compare block curricula `4/8 -> 16 -> mixed 8/16/32` against fixed `16`,
   rejecting any branch that improves sequence while collapsing arguments.
4. Train a lightweight boundary/risk head over frozen states with labels:
   prose, tool name, JSON key, value, code/path, stop.
5. Convert the current protected planner into skeleton-then-value infill
   labels, keeping raw, constrained, and protected scores separate.
