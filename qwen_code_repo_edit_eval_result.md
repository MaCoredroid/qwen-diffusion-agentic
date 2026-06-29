# Qwen Code Repo-Edit Eval Result

Date: 2026-06-26

## Status

The local Qwen3.6-27B NVFP4 teacher can drive Qwen Code on tiny repo-edit tasks
when served with an 8k context profile and a compact Qwen Code system prompt.

Primary result on the 5-task synthetic repo-edit slice:

- initial tests failed: 5/5
- independent final tests passed after Qwen Code edits: 5/5
- changed an expected source file: 5/5
- changed unexpected files: 0/5
- nonempty diff: 5/5
- Qwen Code process exit zero: 0/5

The nonzero Qwen Code exit is a known harness caveat in this run, not a failed
patch result. We force `tool_choice=required` so Qwen3.6 emits native Qwen Code
tool calls instead of prose. After the model has already edited the repo and the
tests pass, Qwen Code keeps receiving forced tool-call turns until the
`--max-tool-calls 12` cap aborts with code 55. The evaluator therefore scores
the independently rerun final tests and expected-file-only diff as the primary
patch metric.

The Alienware RTX 5080 was not used for this run.

## Serving Profile

The 4k teacher profile is too small for real Qwen Code tool turns. The default
Qwen Code prompt plus tool schemas produced about 8.4k input tokens. A compact
system prompt reduced the first request to about 3.2k input tokens, but 4k still
left little room after tool results.

Working local profile:

```bash
PROFILE=nvfp4 \
CONTEXT_LENGTH=8192 \
MAX_TOTAL_TOKENS=8192 \
MEM_FRACTION_STATIC=0.84 \
MAX_RUNNING_REQUESTS=1 \
CHUNKED_PREFILL_SIZE=1024 \
DISABLE_RADIX_CACHE=1 \
ENABLE_MTP=1 \
SPECULATIVE_NUM_STEPS=3 \
SPECULATIVE_EAGLE_TOPK=1 \
SPECULATIVE_NUM_DRAFT_TOKENS=4 \
scripts/serve_sglang_qwen36_teacher.sh
```

Observed local RTX 5090 memory during the sweep was about 24-25 GiB used out of
32.6 GiB.

## Harness

Artifacts:

- tasks: `data/repo_edit_eval/tiny_repo_edit_5.jsonl`
- results: `data/repo_edit_eval/tiny_repo_edit_qwen_code_q36_8k_requiredall_512_tools12_5.jsonl`
- manifest: `data/repo_edit_eval/tiny_repo_edit_qwen_code_q36_8k_requiredall_512_tools12_5.manifest.json`
- work dirs: `runs/qwen_code_repo_edit_eval/work/`

Command shape:

```bash
python3 scripts/eval_qwen_code_repo_edit_cases.py \
  --out-jsonl data/repo_edit_eval/tiny_repo_edit_qwen_code_q36_8k_requiredall_512_tools12_5.jsonl \
  --qwen-timeout 180 \
  --max-wall-time 150s \
  --max-tool-calls 12 \
  --proxy-tool-choice required \
  --proxy-tool-choice-turns 0
```

The evaluator:

- creates a fresh git repo per case
- writes source and unit tests from JSONL
- verifies the seed tests fail
- runs Qwen Code through the SGLang proxy
- reruns tests independently after Qwen Code exits
- records changed files and diffs

## Per-Task Results

| task | final tests | changed files | Qwen Code exit |
| --- | --- | --- | --- |
| `slugify_text` | pass | `text_tools.py` | 55 |
| `merge_intervals` | pass | `intervals.py` | 55 |
| `parse_env_lines` | pass | `envparse.py` | 55 |
| `redact_secrets` | pass | `redact.py` | 55 |
| `word_wrap` | pass | `wraptext.py` | 55 |

Total model-loop elapsed time across the five cases was about 239 seconds.

## Takeaways

- The 5090 is enough for an 8k-context Qwen3.6 NVFP4/MTP teacher serving this
  Qwen Code micro-eval.
- Qwen3.6 can make useful repo edits through native Qwen Code tools when
  `tool_choice=required` is injected.
- The next harness improvement is to stop forcing tool calls after the first
  successful edit/test pass, or teach the proxy to switch back to auto once a
  `run_shell_command` tool result contains passing tests.
- This 5-task gate is now a practical early agentic eval for comparing AR
  teacher, 9B AR baseline, and future diffusion checkpoints before SWE-bench.
