# Stage-C N=5 paired run — Qwen Code × {stock-AR :9951, FLARE diffusion :9952}

**Date:** 2026-07-05 (RTX 5090, RAM cage, one server at a time).
**Task:** Stage-C item **C4+C5** — N=5 SWE-bench_Verified smoke, both arms, paired.
**Goal (C-G1):** prove the SWE loop *closes* on both arms — tool calls parse,
patches land, the eval classifier returns a verdict, no engine crash — and record
per-turn economics AR vs diffusion. Resolve-rate is informational at N=5.

**5 Tier0 SWE-bench_Verified instances × 2 arms, one server at a time** (all 5 on
AR -> kill -> all 5 on diffusion -> kill). AR arm 4.5 min wall (22:27->22:31Z);
diffusion arm 12.9 min (22:32->22:45Z). Clean teardown verified both arms.

**Artifacts:** `/home/mark/qwen_diffusion/runs/stage_c_n5/`
(`{ar,diffusion}/verified/per_task/<iid>/`, `paired_summary.json`, `logs/`,
`dumps_{ar,diffusion}/`, `subset_n5.json`, `run_paired.sh`).

**Instances** (5 of the 20-instance Tier0 `...20260520.json`, chosen to minimize
distinct-repo clone cost): `django-11119`, `django-12754`, `django-13741`,
`pytest-8399`, `sympy-13757` (3 django + pytest [pre-cached] + sympy).

## WARNING: Verdicts are MOCK, not docker resolve@1

docker + `swebench` are absent on this 5090, and the x86 `alienware` offload host
is **unreachable this session** (`Could not resolve hostname alienware`) — and per
the standing user decision (`USER_DECISION_LOCAL_EVAL.md`) offload is out of scope
anyway. So I could not run the real docker harness. Verdicts below are the driver's
labeled **mock** stand-in (`resolved` iff the extracted patch's changed-lines
superset the dataset gold's — extremely strict, so a genuine-but-different fix
scores `failed`). **`predictions.jsonl` is emitted for both arms** (5+5 rows) so
true resolve@1 can be scored later once a local docker/swebench harness is stood up
(in-scope Stage-C recipe work). The load-bearing paired signal here is
**behavioral**: did each arm drive Qwen Code to a real edit, in how many turns /
wall / tokens, and how did it terminate.

## Paired table

| Instance | Arm | Made edit (patch B) | Mock verdict | Turns | Wall s | s/turn | Gen tok (out) | CLI exit -> meaning |
|---|---|---|---|---|---|---|---|---|
| **django-11119** | AR | YES 485 | **resolved** | 8 | 16.9 | 2.1 | 1232 | 0 clean |
| | diff | YES 977 | failed(tests) | 50 | 426.2 | 8.5 | n/a* | **53 turn-limit** |
| **django-12754** | AR | no 0 | failed(apply) | 17 | 27.8 | 1.6 | 1992 | 0 (final turn 400) |
| | diff | no 0 | failed(apply) | 37 | 134.5 | 3.6 | 3999 | 0 (400s) |
| **django-13741** | AR | YES 465 | failed(tests) | 10 | 16.4 | 1.6 | 1288 | 0 clean |
| | diff | no 0 | failed(apply) | 14 | 112.9 | 8.1 | 2949 | 0 (final turn 400) |
| **pytest-8399** | AR | YES 1112 | failed(tests) | 37 | 84.6 | 2.3 | 6735 | 0 clean |
| | diff | no 0 | failed(apply) | 29 | 49.9 | 1.7 | 1864 | **1 loop-detector** |
| **sympy-13757** | AR | no 0 | failed(apply) | 41 | 89.7 | 2.2 | 7164 | 0 (final turn 400) |
| | diff | no 0 | failed(apply) | 19 | 41.0 | 2.2 | 1511 | **1 loop-detector** |

*\*django-11119/diff usage is null: on exit 53 qwen writes an error object (not a
result) to stdout, so per-turn tokens weren't captured; the 977B patch confirms real
edits landed. `s/turn` from wall/turns.*

**Rollup:** AR made a real edit on **3/5**, all 5 exited cleanly (exit 0),
**1/5 mock-resolved**. Diffusion made a real edit on **1/5**, only **2/5** exited
cleanly, **0/5 mock-resolved**.

## Diffusion engine counters (`paired_summary.json -> diffusion_engine_counters`)

- Boot: `decode_mode=hybrid_clean block_size=32 bidir_probe=1 windowed_probe=1
  readonly_denoise=1 canonical_publish=False route_verified=False` — the shipped
  Stage-3 gate-OFF config, same as the A6/A7 byte cert and Stage-A smoke.
- **153 hybrid_clean requests**, all on the grammar path.
- **`projected_value_tokens_exact` all-zero, 0 violations** — the zero-value-projection
  tripwire held across the whole paired run.
- Stop reasons: `complete_tool_call` 147, `max_new_tokens` 1, `None` 5.
- Prefix-cache hit-rate **88.3 -> 88.9%** — real cross-turn APC reuse inside the SWE loop.
- `n_error_lines=0`; **5 HTTP 4xx/5xx** — these are the context-ceiling 400s (Anomaly 3),
  not engine faults (`counters_clean=false` flags only the 400s + the null-usage on the
  exit-53 instance).

## Anomalies (the substance)

**1. R4 loop-detector, malign form at SWE scale (diffusion pytest-8399 &
sympy-13757, exit 1).** Both halted on `consecutive_identical_tool_calls` (qwen's
always-on guard) — pytest looped on `run_shell_command` (24 shell / 4 read), sympy
likewise (10 shell / 8 read) — and **never reached an `edit`** (empty patch). This
is worse than the Stage-A toy smoke, where the loop-detector fired *after* a correct
edit (benign). At SWE scale the diffusion agent burns its budget re-issuing
verify/inspect commands **before** landing a fix. Root cause is the one named in the
Stage-A report: the A2 tool grammar's start-state effectively requires a tool call
every turn, so the model never emits the terminating free-text "I'm done" turn that
lets AR exit cleanly — it wanders until a backstop fires.

**2. Turn-limit exhaustion (diffusion django-11119, exit 53
`FatalTurnLimitedError`).** Ran the full 50 session turns (vs AR's 8 on the same
instance) — the same non-termination — but *did* produce a 977B patch along the way.
So the tool loop works on diffusion; it just doesn't stop. Same instance, AR: 8
turns, clean exit, edit landed, the run's only mock-resolve.

**3. 32,768 context ceiling — SHARED harness limit, both arms.** With
`max_model_len=32768` and proxy `max_tokens=2048`, usable input caps at ~30,720;
once the conversation exceeds it every turn returns HTTP 400 (`...at least 30721
input + 2048 output > 32769... reduce the length`). Server-side: **AR 3x400 /
114x200; diffusion 5x400 / 152x200.** Qwen Code has no compaction, so it just keeps
erroring until it gives up empty. This confounds the long episodes — django-12754,
django-13741, sympy-13757 — on *both* arms and is **not** an engine defect. Top
actionable fix before any larger run: raise `max_model_len` (e.g. 40-48k) or drop
proxy `max_tokens`, and/or enable qwen-code compaction.

**4. Speed.** The 18x wall gap on django-11119 (426s vs 17s) is dominated by **turn
count (50 vs 8)**, not raw per-token cost — **per-turn** diffusion is 1.7-8.5s vs AR
1.6-2.3s (~1-4x).

## Go / no-go read for N=25-50

**Conditional GO, gated on two fixes first.** The loop *closes* end-to-end on both
arms (C-G1 met behaviorally: tool calls parse, patches land, verdicts return, zero
engine crash, counters clean). But two blockers would make an N=25-50 diffusion
resolve number non-creditable if run today:

1. **Context ceiling (Anomaly 3)** — shared, confounds every long episode on both
   arms. Raise `max_model_len` to 40-48k and/or enable compaction *before* the larger
   run. This is the top actionable fix and it is arm-neutral.
2. **R4 non-termination (Anomalies 1+2)** — the diffusion-specific defect. The
   Stage-A prescription (evaluate a top-level `free-text | tool-call` grammar
   alternation, or drop `tools` on post-work turns) must land so the diffusion agent
   can terminate and so it stops burning budget on pre-fix verify loops. Without it,
   diffusion's resolve@1 at N=25-50 is depressed by a termination artifact, not by
   capability.

Also required before scoring: stand up **local docker/swebench** so verdicts are real
resolve@1, not the strict mock. Do **not** launch N=25-50 for a resolve verdict until
(1)+(2)+docker are in place; the reproducing R4 findings below make the diffusion arm's
current resolve floor a harness artifact.

## Do the Stage-A findings reproduce at SWE scale?

**Yes — both reproduce, and R4 sharpens from benign to malign.**
- **Loop-detector (R4):** reproduces (2/5 diffusion episodes halt on
  `consecutive_identical_tool_calls`). At Stage-A toy scale it fired *after* a correct
  edit (benign); at SWE scale it fires *before* any edit lands (malign — empty patch).
- **Turn-count asymmetry:** reproduces and amplifies. Stage-A was 7 (diff) vs 5 (AR);
  here django-11119 is 50 (diff, turn-limit) vs 8 (AR). Diffusion consistently takes
  more turns and does not emit the clean terminating free-text turn AR does — AR exited
  clean (exit 0) on **5/5**, diffusion on only **2/5**. Same structural root cause as
  the Stage-A R4 finding.
