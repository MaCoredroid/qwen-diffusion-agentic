# STRONGER_TEACHER_PILOT — priced three-way comparison for closing the keeper gap

**Status:** decision doc. Written 2026-07-08; **Opus arm now MEASURED 2026-07-09** (§0.5 below).
The original projection sections (§2) are retained verbatim as the **pre-registered prior** so the
projection-vs-measurement delta is visible and honest. The 9B numbers are measured from production
keepers. No production artifact was touched; pilot output lands in `pilot_opus/`, never
`keepers/keepers.jsonl`.

---

## 0.5 PILOT RESULT — MEASURED (2026-07-09, N=10, Opus 4.8 via Claude Code OAuth)

Full pipeline ran end-to-end: 10 stratified holdout-clean episodes (sha256-asserted, §5) → hardened
fork harness (no fetch-crash; the 5d87b4c hardening held) → 4 keepers extracted into `pilot_opus/keepers`.

**Headline (all Opus numbers are now MEASURED; API pricing is prior knowledge, labeled):**

| Metric | Measured value |
|---|---|
| **Yield** | **0.40** (4 resolved / 10) — dead-center of the 0.30–0.60 prior band |
| Yield by stratum | near-miss dregs **1/5 = 0.20**; fresh-coverage tail **3/5 = 0.60** |
| **Cache-read fraction** | **0.931** of all input tokens (16.12M read / 17.33M total) — OAuth path caches |
| Cost reduction from caching | **4.72×** (measured cached vs uncached-counterfactual, same tokens) |
| **$/episode** (cached) | **$1.91**  (uncached counterfactual $9.02) |
| **$/keeper** (cached) | **$4.78**  (uncached counterfactual $22.54) |
| Tokens (10 eps) | input 17.33M (cache-read 16.12M / cache-write 1.20M / uncached 688), output 141.7k |
| Median turns / resolved episode | ~20 (Opus is turn-efficient; range 14–45) |
| Format-equivalence gate (§4) | **PASS** — schema identical by construction; Opus keeper adds one benign additive `provenance.teacher` field; tool-entry keys (`function`/`type`) match; native qwen3_xml, `fidelity=high(all_toolcall_turns)` |

**Per-episode table** (id, stratum, resolved, patch bytes, turns, input tok, output tok, cache-read
fraction, $ cached, $ uncached-counterfactual):

| instance_id | stratum | resolved | patch B | turns | in tok | out tok | cr-frac | $ cached | $ uncab |
|---|---|---|---|---|---|---|---|---|---|
| conan-io__conan-10213 | near-miss | ✅ | 1066 | 25 | 1,344,299 | 11,186 | 0.923 | 1.55 | 7.00 |
| conan-io__conan-10408 | near-miss | ❌ empty | 0 | – | 2,841,662 | 26,360 | 0.921 | 3.36 | 14.87 |
| modin-project__modin-5507 | near-miss | ❌ | 2274 | 45 | 2,580,172 | 16,732 | 0.940 | 2.60 | 13.32 |
| pydantic__pydantic-4882 | near-miss | ❌ | 3000 | – | 2,022,719 | 16,720 | 0.945 | 2.07 | 10.53 |
| pydantic__pydantic-4911 | near-miss | ❌ empty | 2853 | 20 | 1,019,340 | 7,949 | 0.940 | 1.06 | 5.30 |
| pandas-dev__pandas-47475 | fresh-cov | ❌ | 2334 | – | 2,652,556 | 24,367 | 0.947 | 2.75 | 13.87 |
| pandas-dev__pandas-47493 | fresh-cov | ✅ | 2370 | 25 | 1,254,251 | 9,908 | 0.924 | 1.42 | 6.52 |
| getmoto__moto-4867 | fresh-cov | ✅ | 3593 | 14 | 545,276 | 3,512 | 0.920 | 0.61 | 2.81 |
| getmoto__moto-4874 | fresh-cov | ❌ | 7411 | 39 | 2,331,298 | 19,089 | 0.908 | 2.87 | 12.13 |
| dask__dask-10342 | fresh-cov | ✅ | 2020 | 17 | 734,787 | 5,845 | 0.926 | 0.83 | 3.82 |

*(turns "–" = qwen `num_turns` unrecorded on the two empty-patch and two wall-terminated unresolved
episodes.)* Raw JSON: `pilot_opus/pilot_analysis.json`; keepers: `pilot_opus/keepers/keepers.jsonl`.

### 0.5a CACHING — measured, the 4× lever is REAL on the OAuth path
The adapter was extended to (a) inject two `cache_control: ephemeral` breakpoints — one on the last
system block (caches the stable tools+system prefix; render order tools→system→messages) and one on
the last message block (incremental multi-turn prefix) — and (b) log `cache_read_input_tokens` /
`cache_creation_input_tokens` from the Anthropic usage. A direct 2-turn probe confirmed the Claude
Code OAuth path honors caching (turn-2 read 9,533/9,553 = 99.8% from cache). Across the 10 real
agentic episodes, **93.1% of all input tokens were cache-reads** (billed ~0.1×), only 688 tokens were
truly uncached, and 1.20M were cache-writes (~1.25×). Net: **4.72× cheaper than the uncached
counterfactual, MEASURED** — the projected 4× lever is confirmed, not assumed. Caching is therefore
**no longer a prerequisite-to-be-built; it is DONE and ON by default** (`OPUS_ADAPTER_CACHE=1`).

### 0.5b SCALE RECOMMENDATION — close the floor as a PARALLEL API/CPU track
Gap to floor: **89 keepers** (400 − 311 production). At measured yield 0.40 and $/keeper $4.78 cached:

| Target | keepers | Opus episodes (÷0.40) | Cost cached ($1.91/ep) | Cost uncached ($9.02/ep) |
|---|---|---|---|---|
| **Floor (400)** | 89 | **~223** | **~$425** | ~$2,010 |
| Target low (600) | 289 | ~723 | ~$1,380 | ~$6,520 |
| Target high (1000) | 689 | ~1,723 | ~$3,290 | ~$15,530 |

**Run it as a parallel track alongside the 27B GPU campaign** — Opus is API+CPU-bound (no GPU
contention). Measured episode wall was 150–481 s (mean ~330 s). Serial ≈ 20 h for the floor; at a
bounded 3–4 concurrent episodes through one adapter (respect the ≤4-container cap so the orch's
cycle-close scoring is never starved, and the account rate limit — the only thing that killed the
first pilot, now mitigated by the adapter's retry+backoff) that compresses to **~5–7 h of wall for
the 89-keeper floor**. Caching holds under concurrency (per-episode distinct prefixes → distinct
cache entries).

**Where Opus earns its $ (stratum split):** fresh-coverage tail **0.60** (3/5) vs spent near-miss
dregs **0.20** (1/5). Read: route the **fresh-coverage residual** to Opus first (best yield, lowest
$/keeper), and reserve it for the near-miss dregs only where each keeper is otherwise unattainable
(9B failed them outright — the 1/5 it does crack there is pure incremental coverage). Every keeper the
9B cannot produce, Opus produces at ~$4.78 cached.

**Blocking gates for promotion (unchanged) — status:** (i) format-equivalence cert → **PASS** (§0.5
above / §4); (ii) leakage posture → **UNCHANGED** (10 ids sha256-asserted holdout-clean, keepers
isolated in `pilot_opus/keepers`); (iii) caching enabled before bulk scale → **DONE + MEASURED**.

---

## 0. The gap we are pricing

| Milestone | Keepers | Delta from current (311) |
|---|---|---|
| **Current** (production `keepers/keepers.jsonl`, 2026-07-09T13:09Z) | 311 | — |
| **Floor** (400) | 400 | **89 more** ← the gap this doc prices |
| **Target** (600–1000) | 600–1000 | **~289–689 more** |

The campaign **self-killed** on 2026-07-08 15:20Z (`KILL_YIELD_COLLAPSE`, rolling yield 0.075 < 0.10
over the last 200 attempts) — an **honest** yield collapse on the *spent* near-miss retry stock, not a
tamper. It has since resumed on the 27B-NVFP4-MTP epoch (rolling yield ~0.128, w=149) and crept 282→311,
but the fresh-coverage 9B yield remains volatile and the floor is not guaranteed reachable on the owned
GPU in bounded time. The Opus arm below is now MEASURED (§0.5) and prices the **89-keeper floor gap** as a
parallel API/CPU track that takes no GPU from production.

---

## 1. 9B status quo — measured

**Teacher:** stock-Qwen3.5-9B-AR (qwen_code, native qwen3_xml), served on the owned RTX 5090.

| Metric | Value | Source |
|---|---|---|
| Lifetime yield | **0.171** (290 resolved / 1700 attempts) | `attempts.jsonl` |
| Rolling yield at kill | **0.075** (w=200) | `DATAGEN_STATUS.txt` |
| Verdict mix | unresolved 832 / no_prediction 453 / resolved 290 / empty_patch 106 / env_unavailable 19 | `attempts.jsonl` |
| Median tokens/resolved-episode | **610k input** (cumulative over turns), **4.7k output**, median 33 turns | keeper `trajectory_meta.usage` (n=171) |
| Marginal $ cost | **~$0** (owned GPU; electricity only) | — |

**Read:** the 9B is effectively free per token but its cheap stock is exhausted. To reach the floor at
the *observed* fresh-frontier bracket (0.075–0.17) costs **~700–1,600 more attempts** for ~120 keepers —
GPU-time-bound, not $-bound, but with **unproven fresh yield** (could be below 0.075, in which case the
floor is not reachable with 9B alone in bounded time). This is the pivot that motivated the stronger-teacher pilot.

---

## 2. Opus 4.8 teacher — projection [SUPERSEDED by §0.5; retained verbatim as the pre-registered prior]

> **Note (2026-07-09):** the pilot is DONE and MEASURED in §0.5. Everything below is the pre-measurement
> snapshot kept unedited so the projection-vs-measurement delta stays honest. Measured yield **0.40**
> landed dead-center of the 0.30–0.60 prior; measured cached $/keeper **$4.78** vs projected $1.93 (the
> projection under-counted tokens — real Opus episodes ran ~1.7M input tok, ~3× the 9B token prior used
> below). Read the projection for the reasoning, §0.5 for the truth.

**Teacher:** `claude-opus-4-8` via `scripts/opus_openai_adapter.py` (anthropic backend). The adapter is a
drop-in OpenAI `/v1` endpoint; qwen-code → proxy (dumps) → adapter → Opus. Keeper format is **identical
by construction** (§4). Pilot = 10 stratified holdout-clean instances (5 near-miss where 9B failed:
2 conan, 1 modin, 2 pydantic; 5 fresh never-attempted: 2 pandas, 2 getmoto, 1 dask), detached, self-scoring
into `pilot_opus/keepers`.

### 2a. Measured-so-far
- Adapter proven end-to-end in smoke (HTTP 200, tool_calls round-tripped, usage returned, 24 dumps).
- Claude Code OAuth gate diagnosed + fixed (system must be a block-array whose first block is exactly
  `You are Claude Code, Anthropic's official CLI for Claude.`); sampling params stripped (Opus 4.8 rejects
  `temperature`/`top_p`/`top_k` with 400); 429/5xx retry-with-backoff added.
- **Yield: NOT YET MEASURED.** 0/10 episodes executed at time of writing.

### 2b. Pricing (Claude API, prior knowledge from the `claude-api` skill — LABELED, verify at scale)
- Opus 4.8: **$5.00 / 1M input**, **$25.00 / 1M output**.
- Prompt caching: cache **read ≈ 0.1× input = $0.50/1M**; cache **write (5-min TTL) = 1.25× = $6.25/1M**.

### 2c. Cost model (uses the measured 9B token profile as a labeled prior — Opus token counts to be
re-baselined from `pilot_opus/usage_adapter.jsonl` once episodes run; Opus tokenizer differs and Opus
likely uses **fewer** turns, so these are conservative)

| Per episode (any outcome) | Uncached | With prompt caching (~90% prefix reuse) |
|---|---|---|
| median (610k in / 4.7k out) | **$3.17** | **$0.77** |
| mean (539k / 4.8k) | $2.82 | $0.70 |
| p90 (881k / 8.2k) | $4.61 | $1.15 |

| $/keeper = $/episode ÷ yield | Uncached | Cached |
|---|---|---|
| projected yield 0.30 | $10.6 | $2.58 |
| projected yield 0.40 | $7.9 | $1.93 |
| projected yield 0.50 | $6.3 | $1.55 |
| projected yield 0.60 | $5.3 | $1.29 |

**Prior on Opus yield:** frontier models resolve SWE-bench-class tasks far above the 9B; on this harder
SWE-Gym near-miss + fresh mix under our scoring a **0.30–0.60** band is the honest prior. The 5 near-miss
instances are exactly where 9B **failed**, so any Opus resolution there is a keeper otherwise unattainable.

### 2d. Total to close the gap (projected, yield 0.40, median tokens)

| | Uncached | Cached |
|---|---|---|
| ~120 to floor | **~$950** | **~$230** |
| ~320 to target (low) | ~$2,530 | ~$620 |
| ~720 to target (high) | ~$5,700 | ~$1,390 |

**The single biggest lever is prompt caching (~4× cheaper).** The pilot adapter as built does **not** inject
`cache_control` breakpoints, so the pilot measures the **uncached** ceiling. Adding caching to the adapter
before any bulk Opus run is the highest-ROI change and is a hard prerequisite before scaling Opus past the
floor. Marginal $ is real money (vs the 9B's sunk-GPU ~$0), so Opus is justified only where its yield edge
buys keepers 9B cannot.

---

## 3. 27B NVFP4 teacher — feasibility-estimated

**Checkpoint:** `sakamakismile/Qwen3.6-27B-NVFP4` already on disk (single 19.7 GB safetensors, all 64
layers, compressed-tensors nvfp4). Fits the 32 GB RTX 5090 (weights ~22 GB); fully supported by the
installed vLLM 0.23.0 / torch 2.11 stack (arch registered, GDN linear-attn path already de-risked by the
9B). **No MTP head in this checkpoint → no speculative decode** (slower serving than the FP8/bf16 official
checkpoints would allow, but those don't fit).

| Dimension | Estimate |
|---|---|
| Standup cost | **~0.5–1 day eng**: bring up NVFP4 vLLM server, point the driver's `--endpoint` at it (adapter path already generic), run a format-equivalence cert (§4). |
| Marginal $ cost | **~$0** (owned GPU) — but **GPU-contended with production datagen/vLLM**; can only run when the orch is killed/idle (standing constraint: never starve the orch). |
| Projected yield | **0.20–0.35** (between 9B and Opus; stronger base model, but **not** RL-tuned for this agentic harness, and no spec-decode). |
| Best role | A cheap yield bump above 9B **when the GPU is free**, before spending Opus $. |

**Read:** 27B is the "free-but-contended" middle. It costs engineering + GPU exclusivity, not dollars.
Its yield edge over 9B is real but unproven and smaller than Opus's; its value is avoiding per-token spend
for the bulk while beating the 9B's exhausted-stock yield.

---

## 4. Format-equivalence proof status (HARD gate)

- **By construction:** keeper rows are built by `extract_keepers.py` from the **proxy request dumps**
  (qwen-code's own OpenAI-schema request body), which are byte-for-byte independent of which model produced
  the assistant turns. Swapping only the upstream the proxy forwards to (vLLM → Opus adapter → Opus, or →
  27B vLLM) leaves the emitted format **identical by construction** (native qwen3_xml,
  schema = `messages/tools/final_patch/verify/provenance/sft`, matching `keepers/keepers.jsonl`).
- **Smoke-verified:** plumbing proven end-to-end (24 dumps, tool_calls round-tripped).
- **NOT yet certified on a real teacher keeper row:** 0 Opus/27B keepers exist yet (pipeline hasn't reached
  `extract_keepers`). **Gate:** before any pilot keeper is promoted toward SFT, diff one Opus keeper row's
  schema against a production keeper row (same `extract_keepers` machinery → same schema; risk ~zero for
  27B since it's native qwen3_xml + same tokenizer family, low for Opus). Treat this as the blocking check.

---

## 5. Leakage posture — UNCHANGED

- 10 pilot instances asserted **holdout-clean** (sha256-verified against `.eval_holdout_sha256`
  reconstruction), drawn from the **trainable frontier only** — never the 113-id holdout.
- Pilot keepers land in **`pilot_opus/keepers`**, isolated from production `keepers/keepers.jsonl`.
- No change to the holdout, the audit machinery, or the promotion discipline. Stronger teacher ≠ leakage
  regression.

---

## 6. RECOMMENDED MIX + trigger conditions

**Default posture:** keep the 9B as the free bulk producer, but do **not** assume it reaches the floor —
its cheap stock is spent. **Measure the fresh-coverage 9B yield first** (probe #94/#95, pre-registered rule).
Reserve the stronger teacher for the **hard residual** (instances 9B cannot resolve), where the yield edge
is largest and the $/keeper is justified because each keeper is otherwise unattainable.

**Triggers:**
1. **Fresh 9B yield ≥ 0.15** → continue **9B** for the ~120-to-floor (cheapest path; no teacher spend).
   Re-evaluate stronger teacher only for the target stretch.
2. **Fresh 9B yield 0.10–0.15** → continue 9B **and** stand up **27B NVFP4 when the GPU is idle** to lift
   the residual yield at ~$0 marginal, before paying Opus.
3. **Fresh 9B yield < 0.10 AND Opus pilot yield ≥ 0.30** → route the **hard residual to Opus**, **with
   prompt caching enabled in the adapter first** (~4× cost cut). Budget = (needed keepers) × ($/episode ÷
   Opus yield): floor ≈ **$230 cached / $950 uncached**; full target ≈ **$0.6k–1.4k cached / $2.5k–5.7k
   uncached**.
4. **27B standup** whenever (a) production datagen is killed/idle (GPU free) and (b) a cheap yield lift over
   9B is wanted before spending Opus $.

**Blocking gates before any promotion from a stronger teacher:** (i) format-equivalence cert on a real
keeper row (§4); (ii) leakage posture unchanged (§5); (iii) for Opus, caching enabled before bulk scale.

---

## 7. 10-LINE BRIEF

1. Gap: 282 keepers → floor 400 (~120) → target 600–1000 (~320–720); campaign self-killed on an honest 0.075 rolling-yield collapse of the spent near-miss stock.
2. 9B (measured): lifetime yield 0.171, rolling 0.075; ~$0/token but cheap stock exhausted and fresh-frontier yield still UNMEASURED.
3. Token profile (measured, 282 keepers): median 610k input / 4.7k output / 33 turns per resolved episode.
4. Opus 4.8 pilot: **MEASURED 2026-07-09** (§0.5) — 10/10 episodes, **yield 0.40** (4 keepers), **$4.78/keeper cached**, **93.1% cache-read** (4.72× cheaper), format-cert PASS, holdout-clean. Floor (89 keepers) ≈ 223 episodes ≈ **$425 cached**, runnable as a parallel API/CPU track.
5. Opus pricing (prior, labeled): $5/$25 per 1M in/out; cache read ~$0.50/1M, write ~$6.25/1M.
6. Opus cost (projected, 9B token prior): $3.17/episode uncached → $0.77 cached; $/keeper $6–11 uncached / $1.5–2.6 cached at yield 0.3–0.6.
7. Opus totals @ yield 0.4: floor ~$950 uncached / ~$230 cached; full target ~$2.5–5.7k uncached / ~$0.6–1.4k cached — CACHING is the ~4× lever and the adapter must add it before scale.
8. 27B NVFP4: on-disk, fits 32 GB, vLLM-supported, NO MTP; ~$0/token but GPU-contended; ~0.5–1 day standup; projected yield 0.20–0.35 (the free-but-contended middle).
9. Format equivalence: identical BY CONSTRUCTION + smoke-verified, but NOT yet certified on a real teacher keeper row — blocking gate before promotion. Leakage posture UNCHANGED (holdout-clean pilot, isolated keepers).
10. Recommend: measure fresh 9B yield first; 9B for floor if ≥0.15; 27B when GPU idle; Opus (caching ON) for the hard residual only when fresh 9B <0.10 and Opus pilot ≥0.30.
