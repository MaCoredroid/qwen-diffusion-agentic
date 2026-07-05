# Best-of-N same-prompt (GRPO rollout) bench: FLARE hybrid-clean engine vs stock guided-AR

**Question (the *actual* GRPO rollout pattern — the one axis the batched-rollout bench left
unmeasured).** `runs/p2_batched_rollout_bench` disconfirmed the throughput-multiplier thesis on
*independent-prompt* batching (0.73–0.94x vs guided-AR). GRPO does not batch independent prompts —
it draws **N samples of the SAME prompt** (shared prefix -> APC prefix reuse both sides; the group
advantage needs N *diverse, valid* rollouts per prompt). Two load-bearing questions:

- **Q1 (throughput):** at N in {4,8,16} same-prompt, temp=0.7, per-sample distinct seeds —
  samples/sec engine vs guided-AR. AR co-batches N *identical* prefixes perfectly, so it may win
  again. Measure.
- **Q2 (signal quality — the thesis's last chance):** per group, does the engine's canvas/value
  sampling produce **more genuinely-diverse valid candidates** than AR (-> more GRPO signal per
  group), and does best-of-N lift **pass@N** more? (a) unique-output & unique-arg-set fraction,
  (b) pass@1 -> pass@N exact-args, (c) valid fraction.

## Verdict

**AR wins BOTH Q1 and Q2. The signal-quality axis does not save the thesis — it deepens the
disconfirmation.** In the same-prompt GRPO regime stock guided-AR generates rollouts **1.2-1.5x
faster** (eng/AR = 0.85x / 0.67x / 0.67x at N=4/8/16) AND generates **more GRPO signal per group**:
AR is *more* diverse than the engine at every N, and its diversity converts to more correct
rollouts on the hard prompts. **Both paradigms collapse hard at temp=0.7 on strict tool calls**
(peaked value distributions absorb the temperature — every sample ~= greedy), but the FLARE
hybrid-clean engine collapses **harder** (~1.1 unique / 16 vs AR ~1.7 / 16 overall; ~1.25 vs 2.37 on
the miss lane). The engine's *only* edge is **100% valid rollouts** (48/48 groups) vs AR's ~97% (AR
free-samples the occasional truncated/malformed call) — but a valid *identical* rollout is
zero-advantage, so this edge produces no extra GRPO signal. Engine audit is perfectly clean across
all 48 groups (**0 value-projection events, verify-OK, 100% valid**).

## Method (apples-to-apples, one boot each)

| | ENGINE | AR baseline |
|---|---|---|
| system | FLARE hybrid-clean, converted 9B (`qwen3.5-9b-fastdllm-rlv2-vllm-bf16`), vLLM pin `95d8b47` | stock `Qwen3.5-9B` snapshot `c202236`, vLLM 0.23.0 |
| decode | hybrid-clean masked-diffusion, PIECEWISE cudagraph + APC (certified v3b/nevertrain cfg) | guided: `structured_outputs = regex_from_qwen_xml_tool_schema` (the scoreboard regex) |
| gpu_mem_util | 0.62 (N=16 fits) | 0.66 |
| sampling | **temp=0.7, per-sample distinct seeds** | **temp=0.7, per-sample distinct seeds** |

- **Prompts (16, `prompts_manifest.json`, `select_prompts.py`):** never-train BFCL/API-Bank turns,
  stratified across all 4 source families **and** 8 HF-exact / 8 HF-miss (the pass@1 proxy — so
  pass@1 genuinely fails on a real fraction). prompt_len 467-1175, n_ref 23-87.
- **Group rollout (identical both sides):** for each (prompt, N) one `generate()` of N requests with
  the *same* prompt_ids and **distinct seeds** (`base+i`), APC **on** so the shared prefix is reused
  within the wave (the GRPO group). Seeds are **nested** (N=4 subset N=8 subset N=16) -> pass@N is
  monotone. APC reset between groups (cold prompt). Warmup wave per width excluded. `samples/sec = N/wall`.
- **Scoring (identical both sides, `grpo_metrics.py`):** the audited `eval_toolcall_jsonl` scorer.
  `unique-arg-set` uses the *same* `normalize_call_for_compare` coercion the exact-args comparison
  uses (two samples share a key iff exact-args would call them the same). `unique-valid-arg-set` =
  distinct valid canonical arg-sets / N (the GRPO-relevant "distinct valid rollouts").
- **Engine audit (proj==0, verify):** a monkeypatch on `_hybrid_clean_step` reads every live
  hybrid-clean decoder each step; per group we assert `max value_projection_events == 0`, forced
  tokens present, model value tokens present, `forwards == model_chosen`. Prompt integrity: manifest
  `prompt_sha256` carried on every row.

## Q1 — THROUGHPUT (samples/sec == rollouts/sec/GPU)

Micro = sum(samples) / sum(wall) over the 16 prompts at that N (weights by wall — the honest aggregate).

| N | eng micro s/s | AR micro s/s | **eng/AR** | eng mean-grp | AR mean-grp |
|---:|---:|---:|:---:|---:|---:|
|  4 | 5.53 | 6.48 | **0.85x** | 6.68 | 7.05 |
|  8 | 7.81 | 11.58 | **0.67x** | 9.48 | 12.65 |
| 16 | 8.37 | 12.56 | **0.67x** | 10.35 | 13.98 |

**Read:** same-prompt batching makes *both* faster than the independent-prompt bench (shared prefill
amortizes — eng 8.4 vs 5.7, AR 12.6 vs 7.8 at N=16), but the **ratio is unchanged-to-worse**: AR
benefits *more* from N identical prefixes (perfect co-batch at 100% util) than the hybrid, whose
per-request variable draft widths still scatter occupancy (mean batch-in-forward ~= 0.5 of N at
N=16). The two long prompts (n_ref=87) dominate wall on both sides (~3.3 eng / ~6.6 AR s/s). The
GRPO shared-prefix pattern is **not** the regime where the engine wins throughput.

## Q2 — SIGNAL QUALITY (the thesis's last chance)

`uniqOut/uniqArg/uniqValidArg` = mean unique-fraction over N (1.0 = all distinct, 1/N = total
collapse). `pass@1` = micro exact rate over all samples. `pass@N(grp)` = fraction of groups with >=1
exact sample.

**ALL 16 prompts**

| N | side | uniqOut | uniqArg | uniqValidArg | valid | pass@1 | pass@N(grp) |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 4 | engine | 0.266 | 0.266 | 0.266 | **1.000** | 0.516 | 0.562 |
| 4 | ar | **0.312** | **0.297** | **0.297** | 1.000 | 0.531 | **0.625** |
| 8 | engine | 0.141 | 0.141 | 0.141 | **1.000** | 0.516 | 0.625 |
| 8 | ar | **0.180** | **0.180** | **0.164** | 0.977 | 0.531 | 0.625 |
| 16 | engine | 0.070 | 0.070 | 0.070 | **1.000** | 0.508 | 0.625 |
| 16 | ar | **0.105** | **0.098** | **0.090** | 0.984 | **0.535** | 0.625 |

**HF-MISS lane (8 prompts — where pass@1 fails, so best-of-N *should* pay off)**

| N | side | uniqOut | uniqArg | uniqValidArg | valid | pass@1 | pass@N(grp) |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 4 | engine | 0.281 | 0.281 | 0.281 | 1.000 | 0.031 | 0.125 |
| 4 | ar | **0.375** | **0.344** | **0.344** | 1.000 | **0.062** | **0.250** |
| 8 | engine | 0.156 | 0.156 | 0.156 | 1.000 | 0.031 | 0.250 |
| 8 | ar | **0.234** | **0.234** | **0.203** | 0.953 | **0.062** | 0.250 |
| 16 | engine | 0.078 | 0.078 | 0.078 | 1.000 | 0.016 | 0.250 |
| 16 | ar | **0.148** | **0.133** | **0.117** | 0.969 | **0.070** | 0.250 |

**HF-EXACT lane (8 prompts):** both sides *identically* collapse to exactly 1 unique output / N
(uniqOut 0.25/0.125/0.0625) with pass@1 = pass@N = 1.000 at every N — on confident prompts temp=0.7
is fully absorbed; every one of the 16 samples is byte-identical to greedy. **Zero group variance ->
zero GRPO advantage** on the half of prompts the model is already sure about.

Illustrative groups at N=16 (distinct seeds verified): confident `gt154` -> engine 1 unique
arg-set / 16 exact; hard `gt14` -> **engine 2 arg-sets, 1 exact vs AR 2 arg-sets, 6 exact**; `gt161`
-> engine 1 arg-set (total collapse) vs **AR 5 arg-sets**; `gt175` -> engine 1 vs **AR 4**.

## The mechanism (measured)

1. **Peaked value distributions collapse both paradigms.** Tool-call value tokens (IDs, enum
   choices, copied args) are near-deterministic; at temp=0.7 categorical sampling almost always
   re-draws the argmax. Result: ~1 distinct rollout per group on 12/16 prompts, both sides. GRPO's
   group advantage is **structurally starved** on strict tool-call turns — the lever for signal is
   *higher temperature / graded-partial reward*, not the decode paradigm (this reconfirms the
   memory's "highly-peaked value distributions collapse onto greedy" + "graded reward killed
   zero-advantage starvation").
2. **The engine collapses *harder* than AR, not softer.** The hybrid-clean value sampler is at least
   as peaked as stock AR's, so the "engine canvas noise -> more diverse valid candidates" hypothesis
   is **refuted**: AR carries ~1.5-2x the engine's unique-output fraction on the miss lane, and that
   diversity converts to correct rollouts (`gt14`: AR 6/16 exact vs engine 1/16).
3. **The engine's one real edge is validity, not signal.** 48/48 engine groups are 100% valid tool
   calls; AR free-samples the occasional truncated/malformed call (~3-5% invalid on 2 prompts). But a
   *valid identical* rollout is zero-advantage — validity does not create GRPO signal here.

## Honest reading (5 lines)

1. **Q1: AR wins throughput in the GRPO regime too** — eng/AR 0.85x/0.67x/0.67x at N=4/8/16; the
   same-prompt shared prefix helps AR *more* (perfect co-batch) than the variable-width hybrid.
2. **Q2 does not save the thesis — it strengthens the disconfirmation:** AR is *more* diverse at
   every N and lifts miss-lane pass@1 higher, so AR generates **more** GRPO signal per group, not less.
3. **Both paradigms near-totally collapse at temp=0.7 on strict tool calls** (~1-2 unique / 16); GRPO
   on this content is signal-starved for *either* rollout generator — fix it with temperature/reward,
   not the engine.
4. **The engine's sole advantage is 100% valid stops** (vs AR ~97%); real but marginal, and it yields
   no extra advantage signal since the valid samples are identical. Engine audit: **0 projected
   values, verify-OK, 48/48 valid** — the numbers are clean.
5. **Methodology consequence:** for the GRPO rollout step of the diffusion-accelerated-RL loop,
   **stock guided-AR is the better rollout generator on both throughput and per-group signal**; the
   diffusion twin's earned role remains quality/validity at safe batch, not samples/sec or diversity.

## Artifacts

- `prompts_manifest.json` / `select_prompts.py` — the 16-prompt stratified selection (8 exact/8 miss).
- `engine_groups.jsonl` / `ar_groups.jsonl` — per-group rows (per-sample seed, exact, valid, arg-set,
  finish; group diversity/pass metrics; engine audit; GPU util/mem).
- `aggregate.json` / `aggregate.py` — the joined Q1/Q2 tables (all + hf_exact + hf_miss lanes).
- `bench_engine_bestofn.py`, `bench_ar_bestofn.py`, `grpo_metrics.py`, `runcage_*.sh`,
  `tp_engine.log`, `tp_ar.log` — harness + raw logs.
- Precondition: `runs/p2_engine_batchgates` (no cross-request contamination on the batched path).
- Scope: single RTX 5090, 9B, 16 never-train turns x N in {4,8,16}, temp=0.7 distinct seeds, RAM cage.
  Model caveat: engine=converted rlv2 diffusion, AR=stock Qwen — different weights, so pass@1~=0.52
  parity is consistent with the endgame scoreboard; the diversity/throughput findings are paradigm
  properties, not weight artifacts.
