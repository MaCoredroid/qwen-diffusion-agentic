# Agentic Eval Smoke Result

## Scope

Minimal public tau2 mock-domain smoke run. This is a harness canary, not a
promotion result.

Tasks:

- `create_task_1_with_env_assertions`
- `update_task_with_message_history`

Lanes:

- `RAW`: parsed Qwen-native model output, no repair.
- `CONSTRAINED`: preserve already schema-valid parsed calls; otherwise apply
  schema-only `sequence_preserving_constrained_tool_call_text`, max 1 call.
- `PROTECTED`: constrained call selection plus mock tool/policy write guards.

The first fair pass kept live native grammar decoding off for all backends so
the same proxy/lane code applied to AR(a), AR(b), and diffusion.

## Multi-Turn Cache Parity

PASS: `RequestDiffusionState.reset()` advanced through an inserted tool-result
block, then cache-on and cache-off generation agreed through the served stop.

- Prompt tokens: 653
- Block size: 32
- Full prompt blocks committed on reset: 20 / 640 tokens
- Inserted tool-result token span: 485-510
- Tool-result block: 15
- Tool-result committed on reset: true
- Generated argmax flips before stop: 0
- Tokens compared before stop: 10
- Stop token: 248046
- Max logit abs diff: 0.921875 bf16 drift

Artifact: `runs/agentic_eval/multiturn_cache_parity.json`

## Scores

| Backend | RAW | CONSTRAINED | PROTECTED |
| --- | ---: | ---: | ---: |
| AR(a) real `Qwen/Qwen3.5-9B` via SGLang bf16 | 2/2 | 2/2 | 2/2 |
| AR(b) converted 9B `fastdllm_causal` bf16 | 2/2 | 2/2 | 2/2 |
| Diffusion 9B FLARE route-I cache bf16 | 0/2 | 1/2 | 0/2 |

Diffusion-minus-AR(a): RAW -2/2, CONSTRAINED -1/2, PROTECTED -2/2.

## Diffusion Failures

RAW:

- `create_task_1_with_env_assertions`: malformed/incomplete native call, no
  executed call, DB/env assertion failed.
- `update_task_with_message_history`: emitted `<status>` instead of
  `<parameter=status>`, so the parser saw only `task_id`; repeated invalid tool
  executions until turn budget.

CONSTRAINED:

- `create_task_1_with_env_assertions`: recovered by turn 3 and passed, but only
  after one invalid `create_task({})` execution and a `get_users` recovery turn.
- `update_task_with_message_history`: projection repaired the tag shape but
  chose `status=pending`, so action args and DB state failed.

PROTECTED:

- `create_task_1_with_env_assertions`: protected guard blocked the projected
  empty `create_task({})`, leaving no executed call.
- `update_task_with_message_history`: same wrong `status=pending` projection as
  constrained.

## Manifest

- Benchmark: public `sierra-research/tau2-bench` mock domain
- tau2 commit: `d8e915f7f46b56af9b14d5d0544ccc9fd5d71009`
- Task subset SHA256: `6f6ce972ca8791c5b86992e2660a7863374bd033d351b5cde2231fe10b5d2ebc`
- Policy SHA256: `50ccbefac7c9ba77ce54131d61cfc6882f74b88ace66771115f9d867ec098203`
- DB SHA256: `8729effbe2b2e1c7f4310be2c87e6c2d314db0fb5ff4b4444bb4df2a83c1d354`
- Tool schema SHA256: `bd7491d56a14cf6f5dd98690eee3e294f4e047115da14e891a1b209e0ce8325d`
- Seed: `20260701`
- Max agent turns: 3
- Max new tokens: 192
- Quant/dtype: bf16, no runtime weight quantization for all reported backends

Artifacts:

- `runs/agentic_eval/tau2_mock_openai_ar9b.jsonl`
- `runs/agentic_eval/tau2_mock_openai_ar9b.summary.json`
- `runs/agentic_eval/tau2_mock_fastdllm_ar9b.jsonl`
- `runs/agentic_eval/tau2_mock_fastdllm_ar9b.summary.json`
- `runs/agentic_eval/tau2_mock_diffusion9b.jsonl`
- `runs/agentic_eval/tau2_mock_diffusion9b.summary.json`

No promotion: diffusion does not match either AR baseline on this first
multi-turn smoke.
