# Agentic Diffusion Frontier Research

Date: 2026-06-28

Purpose: capture online research from the parallel gpt-5.5 high subagent and
translate it into concrete next experiments for this Qwen diffusion project.

## Current Local Evidence

The protected/sidecar route proves the target behavior is reachable, but the
model has not internalized it:

- protected heldout policy-target route: `12/12` exact sequence and arguments;
- checkpoint-275 heldout policy-target baseline:
  raw `0/12`, constrained sequence `5/12`, constrained args `0/12`;
- planner-heavy checkpoint-5:
  constrained sequence `4/12`, constrained args `1/12`;
- sequence/value checkpoint-5:
  raw valid JSON `1/12`, constrained sequence `5/12`, constrained args `0/12`.
- fixed `bd_size=16` checkpoint-5:
  raw `0/12`, constrained sequence `6/12`, constrained args `0/12`;
- high-pressure arg/value branch:
  first raw exact sequence/argument signal at `1/12`, but constrained sequence
  falls to `4/12` and extra/repeated calls rise;
- lower-pressure arg/value branch:
  checkpoint-10 keeps a `1/12` raw exact sequence signal but still `0/12`
  exact arguments, with constrained sequence down to `3/12`;
- scalar repair sidecar transfer:
  schema/required-arg validity improves, but exact arguments stay `0/12`.

Implication: broad SFT mixtures can move isolated signals, but they are not
preserving sequence and argument behavior together.

## High-Signal Sources

1. [BD3-LMs / Block Diffusion Language Models](https://m-arriola.com/bd3lms/)
   - Practical AR/diffusion compromise: arbitrary length, KV caching, and
     parallel within-block sampling.
2. [Fast-dLLM](https://nvlabs.github.io/Fast-dLLM/)
   - Confidence-aware parallel decoding and block-wise cache reuse; directly
     relevant to local scaffold and commit instability.
3. [Efficient-DLM](https://arxiv.org/html/2512.14067v2)
   - AR-to-diffusion conversion via block-wise causal attention and
     position-dependent masking.
4. [DiffuLLaMA](https://arxiv.org/html/2410.17891v2)
   - AR checkpoint adaptation to diffusion through continual pretraining.
5. [SemBlock](https://arxiv.org/html/2606.04964v1)
   - Dynamic semantic block boundaries; relevant to treating tool-call spans as
     semantic blocks.
6. [When to Commit? Variable-Size Self-Contained Blocks](https://arxiv.org/abs/2604.23994)
   - Block commit criteria based on self-containedness / future-aware
     divergence.
7. [Anchor-based History-stable Decoding](https://arxiv.org/html/2604.08964v1)
   - Training-free early commit based on token stability trends.
8. [DINGO](https://arxiv.org/abs/2505.23061)
   - Dynamic-programming constrained inference for diffusion LMs; relevant to
     JSON/tool grammar.
9. [CFG Constrained Decoding for Diffusion LLMs](https://arxiv.org/html/2508.10111v1)
   - Additive infilling constraints for structured outputs such as JSON/code.
10. [DiffuCoder](https://arxiv.org/abs/2506.20639)
    - Code-focused dLLM; diffusion generation order can become a test-time
      compute/RL knob.
11. [BFCL](https://proceedings.mlr.press/v267/patil25a.html)
    - Function-call eval framing for tool calls, multi-step state, and
      agentic trajectories.
12. [Qwen3-Next / Gated DeltaNet vLLM support](https://vllm.ai/blog/2025-09-11-qwen3-next)
    - Hybrid GDN/full-attention state handling matters for remasking, caching,
      and block commits.

## Ranked Experiment Hypotheses

1. **Tool-call-aware dynamic block boundaries**

   Treat `<tool_call>`, function names, JSON keys, scalar values, delimiters,
   and `</tool_call>` as separate scheduleable spans. Start heuristic, then
   consider a lightweight boundary/value head. Commit only when self-contained
   or history-stable. This attacks the exact local failure: fragile tool-call
   spans under block diffusion.

2. **Diffusion-native JSON/tool constrained infill**

   Do not rely only on AR-style prefix constraints. The sampler should require
   every partially denoised tool-call state to be completable under the tool
   grammar, including required args, enum values, tool existence, and structural
   delimiters.

3. **Targeted AR-teacher distillation on tool-call spans**

   Use Qwen3.6 AR teacher logits/trajectories for exact tool-call
   serialization. Train response-only noising with heavier corruption on
   delimiters, field names, and value spans rather than broad assistant CE.

4. **Skeleton-then-infill tool generation**

   Generate route/name/skeleton first, then diffusion-infill argument values.
   The protected route already behaves this way; the model-side recipe should
   learn that decomposition instead of treating the whole call as one flat
   string.

5. **GDN state/cache ablation**

   Test full recompute per denoise step, snapshot recurrent state only at
   committed block boundaries, and current cache reuse. If results move
   materially, stale GDN state is part of the apparent model failure.

## Warnings

- More planner-heavy data alone is unlikely to fix arguments.
- Standard AR constrained decoding is an awkward fit for unordered diffusion
  updates.
- Tiny fixed blocks are useful as diagnostics but collapse toward AR decoding.
- Syntax validity can hide semantic argument failure, so raw JSON, tool
  sequence, arguments, and task success must stay separate.
- GDN/cache reuse can make a sampler bug look like model weakness.
- The 12-case heldout gate is useful for fast promotion decisions, but broader
  BFCL-style categories are needed before trusting an improvement.

## Parallel Research Update: 2026-06-28

Additional online research was run in parallel with the local `bd_size=16`
target-objective ablations. The high-signal additions are:

1. **LLaDA2.0 / WSD conversion**

   Source: <https://arxiv.org/html/2512.15745v2>

   LLaDA2.0 frames frontier dLLM training as systematic AR-to-diffusion
   conversion, not from-scratch agentic diffusion. Its recipe uses a
   Warmup-Stable-Decay block schedule: grow from AR-like small blocks toward a
   larger/global diffusion regime, then decay back to compact block diffusion.
   The reported common inference setting is temperature `0.0`, block size `32`,
   threshold `0.95`.

   Local implication: keep fixed `bd_size=16/32` as the practical target range
   and test WSD-style curricula only after the current route/value failure
   modes have explicit behavior-retention metrics.

2. **DreamReasoner block-size curriculum**

   Source: <https://arxiv.org/html/2606.19257v1>

   DreamReasoner-8B is initialized from Qwen3-8B-Base and reports that
   large-block training is brittle for reasoning. Its actionable signal is to
   start with very small blocks, then move to mixed larger block sizes, with
   inference staying useful across blocks `4-32`.

   Local implication: dynamic `8,16,32` was neutral in our first 10-step run,
   but the external result says the missing piece may be curriculum order, not
   just random per-batch choices. Try staged `4/8 -> 16 -> mixed 8/16/32`
   only after adding AR-agreement diagnostics.

3. **CFG constrained decoding for diffusion LMs**

   Sources:
   <https://openreview.net/forum?id=7Sph4KyeYO>,
   <https://arxiv.org/html/2508.10111v1>,
   <https://github.com/eth-sri/constrained-diffusion>

   This work handles out-of-order diffusion generation by checking whether a
   partially denoised output can still be completed under a context-free
   grammar. JSON and C++ are direct matches for tool calls and code edits.

   Local implication: the next sampler work should not be another
   post-generation JSON fixer. It should enforce completable JSON/tool states
   during diffusion updates, then report raw, constrained, and protected scores
   separately.

4. **I-DLM introspective consistency**

   Sources:
   <https://introspective-diffusion.github.io/>,
   <https://arxiv.org/html/2604.11035v1>,
   <https://github.com/Introspective-Diffusion/I-DLM>

   I-DLM argues that dLLMs often fail to accept or stay consistent with their
   own generated tokens. Its introspective strided decoding verifies prior
   tokens while advancing new tokens.

   Local implication: add behavior-retention diagnostics before scaling:
   AR-reference top-k/KL on tool spans, token acceptance on AR continuations,
   exact tool-call AST match, and stop-boundary agreement. These metrics should
   sit beside train loss and heldout exact arguments.

5. **BFCL V4 for agentic tool-call evaluation**

   Source: <https://gorilla.cs.berkeley.edu/leaderboard.html>

   BFCL V4 is current as of 2026-04-12 and explicitly covers holistic agentic
   function calling, including format sensitivity, multi-turn behavior, latency,
   and cost.

   Local implication: once the local 12-case heldout gate shows real movement,
   add a BFCL v4/v3 slice as the first external tool-call benchmark before
   spending on SWE-bench Verified.

## Parallel Research Update: Constrained Selector Control

Additional online research was run in parallel with the schedule-state selector
experiments.

1. **Constrained diffusion decoding for JSON is directly relevant**

   Sources:
   <https://arxiv.org/abs/2505.23061>,
   <https://arxiv.org/abs/2508.10111>,
   <https://github.com/eth-sri/constrained-diffusion>,
   <https://arxiv.org/html/2602.00612>

   DINGO-style regex/DFA constraints are a good fit for fixed tool-call
   envelopes and finite selector schemas. CFG-constrained diffusion decoding is
   more appropriate for nested JSON/code, because it verifies whether partial
   out-of-order denoising states remain completable. LAVE-style
   lookahead-then-verify is especially relevant to the local close-guard
   failures: do not commit a partially denoised tool-call state unless the
   grammar verifier can still complete it.

   Local implication: enforce tool-call JSON during denoising or score a small
   set of schema-valid decisions; do not rely on post-generation repair.

2. **Candidate scoring is the right interface for selector state**

   Sources:
   <https://arxiv.org/html/2502.09992v1>,
   <https://aclanthology.org/2026.eacl-long.257.pdf>,
   <https://arxiv.org/html/2601.20339v1>,
   <https://arxiv.org/html/2602.12528v1>

   LLaDA-style candidate likelihood, PADRE pseudo-likelihood, order-token
   search, and DiffuRank all support the same conclusion: for small finite
   choices, rank candidate decisions with masked likelihood instead of asking
   the dLLM to emit a structured object.

   Local result now matches that: free selector JSON gets `0/16` valid
   decisions, while constrained `candidate_index` scoring gets `312/349`
   top-1 and `334/349` top-2 on all ambiguous schedule-state rows.

3. **Agentic dLLMs need tool-call control layers**

   Sources:
   <https://arxiv.org/html/2601.12979v1>,
   <https://github.com/Coldmist-Lu/DiffuAgent>,
   <https://qwen.readthedocs.io/en/latest/framework/function_call.html>

   Agent/tool-calling evaluations for dLLMs report malformed JSON schemas and
   hallucinated API parameters as central failure modes. Qwen's own
   function-calling guidance also treats prompt/template-only tool calling as
   insufficient for production and recommends rectification.

   Local implication: the target recipe should be "AR-like behavior preserved
   under constrained diffusion control," not "diffusion model freely writes
   tool-call JSON."

## Next Local Translation

The old immediate probe, LoRA delta composition, has now failed on the
heldout policy gate: it gained occasional raw-valid syntax but collapsed the
plain `bd_size=16` sequence/order signal. The scalar repair sidecar transfer
also failed to move exact heldout arguments.

Updated local priorities:

1. Keep fixed `bd_size=16` checkpoint-5 as the route/order anchor until another
   branch beats `6/12` constrained exact sequence on the heldout policy target.
2. Build a tool-call skeleton-then-value-infill path. Route/order should be a
   separate target from argument values.
3. Implement completable JSON/tool grammar constraints in the diffusion sampler,
   starting with tool-call syntax and required-key completion.
4. Add AR-reference behavior-retention diagnostics before a larger staged
   block-size curriculum.
5. Treat remasking and scalar sidecar repair as ablations, not defaults.
