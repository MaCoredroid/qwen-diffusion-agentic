# Qwen Code Official Harness Result

Date: 2026-06-26

## Status

Qwen's official Codex-like harness is available as `@qwen-code/qwen-code` and
is now installed locally as a project dev dependency. This gives us an official
agentic coding harness for later repo-edit and SWE-style evaluations.

Official package checked:

```text
package: @qwen-code/qwen-code
version: 0.19.2
node: >=22
```

Upstream reference:

```text
https://github.com/QwenLM/qwen-code
```

The installed README describes Qwen Code as an open-source terminal coding agent
with headless mode, OpenAI-compatible providers, MCP, skills, subagents, and
worktree support.

## Local Integration

Qwen Code can call OpenAI-compatible providers directly, but the local SGLang
Qwen3.6 teacher needs two compatibility adjustments:

1. Clamp output tokens. Qwen Code defaults OpenAI-compatible output to 8k when
   no explicit limit is configured, while the live teacher currently serves a 4k
   context.
2. Disable Qwen thinking via `chat_template_kwargs.enable_thinking=false`.
   Without this SGLang-specific field, Qwen3.6 spends the short smoke response
   budget on reasoning text.

Added a small local bridge:

```text
scripts/qwen_code_sglang_proxy.py
```

And a smoke runner:

```text
scripts/run_qwen_code_sglang_smoke.sh
```

The proxy forwards OpenAI Chat Completions requests to SGLang, injects
`chat_template_kwargs.enable_thinking=false`, and clamps `max_tokens`.

## Smoke Command

```bash
npm run qwen-code:smoke
```

Equivalent direct command:

```bash
scripts/run_qwen_code_sglang_smoke.sh
```

## Result

The smoke was run against the live local teacher:

```text
model: qwen3.6-27b-teacher
base URL: local proxy -> http://127.0.0.1:30000/v1
Qwen Code version: 0.19.2
tools: disabled for smoke
result: QWEN_CODE_PROXY_OK
input tokens: 934
output tokens: 7
total tokens: 941
API errors: 0
```

This proves Qwen Code can drive the local Qwen3.6 SGLang teacher for headless
agentic tasks once the compatibility bridge is in place.

## Next Gate

Use this harness for repo-edit and SWE-style work, not for the tiny deterministic
function-unit tests. The next practical Qwen Code gate is:

1. Create 5-10 tiny repo-edit tasks with tests.
2. Run Qwen Code headless with limited tools and a temporary worktree.
3. Measure patch apply rate, tests passed, tool loop failures, and stop-boundary
   behavior.
