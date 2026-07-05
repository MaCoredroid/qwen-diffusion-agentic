# Stage-A Qwen Code smoke — first end-to-end agentic-CLI run on the diffusion engine

**Date:** 2026-07-05 (RTX 5090, RAM cage, one server at a time).
**Goal:** drive the SAME planted-bug repo-edit task through **Qwen Code** (headless)
against (a) the FLARE diffusion hybrid_clean `/v1` endpoint and (b) the stock-AR
`/v1` endpoint, one server at a time. The agent must **read files, make an edit,
run the test**. Per-arm record: task completed? / turns / wall-clock / tokens;
diffusion **engine counters clean**. This closes Stage-A item **A5** (AR sanity)
and is the first real exercise of the **qwen-code <-> diffusion tool loop** (named
residual risk **R4**).

## Verdict

| | Diffusion (:9952) hybrid_clean | Stock-AR (:9951) |
|---|---|---|
| **Task completed** (independent final tests pass) | **YES** | **YES** |
| Edit correct + minimal (`return .../len + 1` -> `return .../len`) | YES | YES |
| Files changed | `mathutils.py` only | `mathutils.py` only |
| Unexpected files touched | none | none |
| read -> edit -> run-test loop closed | **YES** | **YES** |
| Qwen Code process exit | **1** (loop-detection halt *after* success) | **0** (clean) |
| Result subtype | `error_during_execution` | `success` |
| Agent turns (`num_turns`) | 7 | 5 |
| Tool calls | `read_file` -> `edit` x1 -> `run_shell_command` x4 | `read_file` -> `edit` x1 -> `run_shell_command` x1 |
| Wall-clock (qwen elapsed) | 12.98 s | 7.70 s |
| API wall (`duration_api_ms`) | 11.18 s | 6.00 s |
| Tokens (in / out / total) | 22,739 / 400 / 23,139 | 18,760 / 406 / 19,166 |

**Both arms complete the task and close the tool loop, driven identically** through
the same harness + proxy + native `qwen3_xml` tool format. The one behavioral
difference is **termination**, not capability (see Finding 1).

## Both-arms outcome, in words

- **Diffusion engine (the headline):** this is the FIRST end-to-end agentic-CLI
  run on the diffusion engine, and **it works.** Qwen Code sent an OpenAI `tools`
  array to `/v1/chat/completions`; the **A2 bridge** turned it into the FLARE
  hybrid_clean grammar; the model `read_file`'d `mathutils.py`, emitted a
  **byte-perfect `edit`** dropping the planted `+ 1`, then `run_shell_command`'d
  `python3 -m unittest test_mathutils` and observed `OK`. Independent re-run of the
  tests passes; only the expected file changed.
- **Stock-AR:** same task, same harness, same native tool format -- completes in 5
  turns and **terminates cleanly** with a final natural-language summary
  ("All tests pass. The bug was the `+ 1` ...").

## Diffusion engine counters — CLEAN (`diffusion_engine_counters.json`)

Read live from the served engine's own audit instrumentation (pin `ef97e1e`):

- **A-G1 decode-mode gate:** boot log `decode_mode=hybrid_clean` (not canvas),
  `block_size=32 bidir_probe=1 windowed_probe=1 readonly_denoise=1
  canonical_publish=False` -- i.e. the **shipped Stage-3 gate-OFF config exactly as
  the A6/A7 byte cert ran it**, plus `mask=248077` suppression active.
- **15 hybrid_clean requests** logged (14 real + 1 warmup). **Every** request went
  through the hybrid_clean grammar path -- **zero** free-text/L0 fallback, i.e. the
  A2 tools->grammar bridge engaged on every qwen-code turn (the exact-arg safety net
  behind the 47/63 cert was ON throughout).
- **Zero-value-projection tripwire held: `projected_value_tokens_exact = 0` on all
  15 requests** (0 violations).
- **`stop_reason = complete_tool_call` on all 14 real requests** -- the grammar
  closed every tool call properly (no truncation, no runaway).
- **Prefix-cache hit rate 78.1% -> 82.6%** across the run -- real cross-turn APC
  reuse inside the agentic loop.
- **0 error/traceback lines, 0 HTTP 4xx/5xx** on either server.
- `forced_token_count > 0` per request = the grammar scaffold forcing the
  tool-call structure tokens; `value_tokens` = the model's free-choice content --
  both present, confirming grammar + free content coexisting per turn.

## Findings (tool-loop breakage treated first-class)

**Finding 1 — [correctness / termination, MEDIUM] Diffusion arm never emits a
terminating free-text turn; it loops on the verify step until Qwen Code's
loop-detector halts (exit 1).** After the fix landed and `unittest` printed `OK`,
the hybrid_clean model re-issued the **identical** `run_shell_command` 4x and Qwen
Code's always-on `consecutive_identical_tool_calls` guard aborted the run. The
task still completed (edit correct, tests pass), so this is a **termination /
economics** issue, not a task-completion failure. It is **structural, not
prompt-fixable**: a second diffusion run with an explicit "as soon as tests pass,
STOP and reply in text" system prompt (`diffusion_guided.jsonl`) produced the
**same** 7-turn loop-halt. Mechanism: the A2 bridge compiles the tool-call grammar
on **every** turn that carries `tools` (Qwen Code always advertises its toolset),
and empirically **every** diffusion generation ended `stop_reason=complete_tool_call`
-- i.e. the grammar's start state effectively requires a tool call, foreclosing the
free-text "I'm done" turn the AR model uses to exit 0. The stock-AR arm, same task,
terminates cleanly in 5 turns. **This is the named R4 risk, in a benign-but-real
form.** Stage-C implications + options:
  - The loop **does** complete useful work and Qwen Code's loop-detector is a safe
    backstop, but it costs extra verify turns and yields a non-zero CLI exit, so a
    Stage-C driver must score **independent test/patch outcome**, not the CLI exit
    code (already how this harness scores).
  - Cleaner fixes to evaluate in Stage C (C1 driver): (a) let the FLARE
    hybrid_clean grammar carry a top-level `free-text | tool-call` alternation so a
    terminating assistant message is in-grammar; or (b) proxy-side, drop `tools`
    from the request once a `run_shell_command` tool result shows passing tests, so
    the final turn is ungrammared free text; or (c) accept the loop-detector as the
    terminator and post-process. (a) is the principled fix and matches
    [[native-function-format-rule]].

**Finding 2 — [test-harness robustness, LOW] Stored `qwen_stdout` is tail-truncated
(`[-6000:]`), dropping the opening `read_file` turn from the record.** The reused
`eval_qwen_code_repo_edit_cases.py` truncates the transcript, so `tool_by_name`
tallies computed off the stored stdout under-count early tools. The `read_file`
step was recovered from the proxy request dumps (`work/*_dumps/`, kept via
`--proxy-dump-dir`). Not a run defect; note for Stage-C reporting fidelity -- either
raise the cap or rely on the proxy dumps as the source of truth.

**Finding 3 — [config note, LOW] Served context raised 4096 -> 8192 for the smoke.**
A real Qwen Code turn (system prompt + tool schemas + tool results) needs ~3.5-4k
input tokens (measured `input_tokens` 3.5k first-turn, ~4k by turn 7). The A6/A7
byte cert served `max_model_len=4096`; the smoke launcher (`runcage_diffusion.sh`)
overrides `MAX_MODEL_LEN=8192`. This is a pure context-window change (does not touch
the decode path); engine counters/tripwire unaffected. Flagged so the byte-cert
regime (4096) and the agentic regime (8192) are not conflated.

## No-breakage items (things that could have broken and did not)

- OpenAI `tools` (chat path) -> FLARE grammar bridge (A2) fired for **every**
  qwen-code turn -- no `schemas={}` free-text fallback.
- vLLM `qwen3_xml` tool-call parser normalized the hybrid_clean XML into
  `message.tool_calls` that Qwen Code consumed natively (unlike the earlier SGLang
  AR baseline, which did not normalize) -- **both** vLLM arms normalize identically.
- Stock-AR booted with **cudagraph** (`FULL_AND_PIECEWISE`, not enforce-eager) on
  stock vLLM 0.23 / sm_120 for the GDN hybrid -- captured in ~1 s, no fallback
  needed.
- RAM cage + one-server-at-a-time held; GPU returned to 2579 MiB baseline between
  arms; no stray servers/ports.

## Config / provenance

- **Diffusion arm:** `runcage_diffusion.sh` -> flywheel
  `scripts/qwen35_9b_flare_hybrid_serve.sh` (synced `c9ff24da`, A1/A3/A6 fixes),
  pin `.venv-vllm-p2-main` (`b5fcb3d`-class, engine `0.1.dev200+g2665ed704`),
  export `models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16`, gate OFF (canonical_publish
  unset), FR13 align-APC, `--enforce-eager`, temp-0 default, `:9952`, served
  `qwen3.5-9b-flare-hybrid-clean`.
- **AR arm:** `runcage_ar.sh` -> stock `vllm serve` on `.venv-vllm` (0.23.0),
  **stock `Qwen/Qwen3.5-9B` @ `c202236` HF snapshot**, cudagraph on, FR13-class
  APC, `qwen3_xml` + `qwen3` reasoning + the same codex chat template, `:9951`,
  served `qwen3.5-9b-ar`.
- **Agent:** Qwen Code `@qwen-code/qwen-code@0.19.2` (already a project dev-dep at
  `node_modules/@qwen-code/qwen-code`; `node_modules/.bin/qwen`; node v22.23.1) --
  **no install needed.** Config surface = `--auth-type openai --openai-base-url
  <proxy>/v1 --openai-api-key dummy --model <served-name>`. Headless flags:
  `--bare --approval-mode yolo --max-tool-calls 30 --max-wall-time 300s
  --output-format json --system-prompt <compact> --exclude-tools agent web_fetch
  notebook_edit -p <task>`.
- **Adapter:** `scripts/qwen_code_sglang_proxy.py` (reused) as a thin OpenAI
  reverse-proxy: injects `chat_template_kwargs.enable_thinking=false`, clamps
  `max_tokens` to 512, passes `tools` through unchanged (A2 bridge runs
  server-side). Both arms driven identically.
- **Harness:** reused `scripts/eval_qwen_code_repo_edit_cases.py` (fresh git repo ->
  seed test fails -> qwen-code headless -> independent test re-run -> diff/stats),
  `--proxy-tool-choice ""` (natural/auto -- no forced tools), single toy case.
- **Toy repo (planted bug):** `toy_repo_case.jsonl` -- `mathutils.py`
  (`average()` with an extraneous `+ 1`), `test_mathutils.py` (4 unittests; 3 fail
  on the bug), `README.md`. `test_command = python3 -m unittest test_mathutils`.
  Seed test fails (exit 1) as required; fix = delete `+ 1`.

## Artifacts

- `toy_repo_case.jsonl` -- the planted-bug case (built by `build_toy_repo_case.py`).
- `runcage_diffusion.sh` / `runcage_ar.sh` -- caged serve scripts (one at a time).
- `diffusion_auto.jsonl` / `diffusion_guided.jsonl` / `ar_auto.jsonl` -- per-arm
  qwen-code result rows (+ `.manifest.json`, `.proxy.log`).
- `diffusion_engine_counters.json` -- parsed FLARE audit counters (via
  `read_counters.py`).
- `logs/diffusion_server.log` / `logs/ar_server.log` -- full server logs.
- `work/*_dumps/` -- per-request proxy payloads (transcript source of truth).

## Reproduce

    cd /home/mark/qwen_diffusion
    python3 runs/stage_a_smoke/build_toy_repo_case.py
    # Diffusion arm (one server at a time, in the RAM cage):
    systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=4G --unit=stageA_smoke_diff \
      bash runs/stage_a_smoke/runcage_diffusion.sh > runs/stage_a_smoke/logs/diffusion_server.log 2>&1 &
    #   poll http://127.0.0.1:9952/v1/models, then:
    python3 scripts/eval_qwen_code_repo_edit_cases.py \
      --input-jsonl runs/stage_a_smoke/toy_repo_case.jsonl \
      --out-jsonl runs/stage_a_smoke/diffusion_auto.jsonl \
      --work-root runs/stage_a_smoke/work/diffusion_auto \
      --endpoint http://127.0.0.1:9952/v1 --model qwen3.5-9b-flare-hybrid-clean \
      --proxy-port 30011 --proxy-tool-choice "" --qwen-timeout 360
    systemctl --user stop stageA_smoke_diff.scope
    python3 runs/stage_a_smoke/read_counters.py runs/stage_a_smoke/logs/diffusion_server.log
    # AR arm: same, with runcage_ar.sh / :9951 / qwen3.5-9b-ar / stop stageA_smoke_ar.scope
