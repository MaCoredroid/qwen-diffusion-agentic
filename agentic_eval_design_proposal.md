# Agentic Eval Design Proposal

Date: 2026-07-01

## Current Gate

Do not build the harness until this design is red-teamed. This document is the
proposal checkpoint for the FLARE agentic-eval phase.

## Baselines

- **AR baseline:** real `Qwen/Qwen3.5-9B` served through SGLang as
  `qwen3.5-9b-ar`, native Qwen tool parser, `enable_thinking=false`.
- **Diffusion baseline:** current converted 9B through the validated HF
  route-I FLARE fast serving path (`RequestDiffusionState`, clean-causal
  `advance()`, train-matched +1 shifted head) plus the same native tool grammar
  decoder.
- **Quantization for the first score:** bf16 serving on both sides. This matches
  the validated diffusion HF serving path, which loads the bf16 base plus LoRA
  adapter rather than a runtime NF4 server. If we later move diffusion serving
  to runtime 4-bit, rerun the AR baseline with `PROFILE=bnb4` and treat the bf16
  numbers as invalid for the matched-quant comparison.

## Benchmark Choice

Use **tau-bench / tau2** as Tier-B first signal: public, multi-turn tool use,
stateful DB reward, and more tractable for 9B than SWE-bench Verified. SWE-bench
Verified remains the second benchmark once the tau path is running, but it should
not block the first north-star score because the 9B absolute score is expected
to be low.

## Fairness Contract

Both AR and diffusion must be driven by one harness contract:

- identical public task subset and seed;
- identical system/developer prompting, tool schemas, tool names, and tool
  result serialization;
- Qwen-native tool-call format end-to-end;
- parse Qwen-native `<tool_call><function=...>` content as canonical even when a
  backend does not populate OpenAI `message.tool_calls`;
- `enable_thinking=false`;
- identical stop policy and max-turn / max-token budgets;
- identical grammar/projection policy in the constrained lane;
- same score lanes: RAW, CONSTRAINED, PROTECTED, with promotion only from RAW or
  CONSTRAINED model-only gains;
- same quant and dtype for reported comparison;
- record every request/response, tool call, tool result, final DB-state reward,
  wall time, and backend metadata.

## Integration Approach

The harness should be backend-agnostic and speak OpenAI-compatible chat
completions or Responses-style calls:

- AR path: SGLang OpenAI server on `qwen3.5-9b-ar`.
- Diffusion path: a thin OpenAI/Responses shim around the existing in-process HF
  fast serving. The shim owns request state because diffusion serving is
  currently in-process, while most agent harnesses expect an HTTP API.
- Do not let Codex/flywheel-specific normalization change payloads differently
  per backend. If a proxy is needed, both paths must pass through the same proxy
  code and only swap upstream URLs.

## Minimal First Run After Red-Team

After approval, build the smallest harness that can run a fixed tau-bench domain
subset through both backends. Start with a smoke-sized subset, then expand only
after confirming identical prompts/tools/stops and successful DB-state scoring.

Report:

- AR score, diffusion score, and diffusion-minus-AR;
- per-lane RAW/CONSTRAINED/PROTECTED numbers;
- failure taxonomy: no tool call, invalid call, wrong tool, wrong args, bad
  tool-result handling, task-state failure, stop/turn-budget failure;
- fairness manifest: backend, model id, quant/dtype, prompt hash, tool hash,
  parser/grammar version, task subset hash.
