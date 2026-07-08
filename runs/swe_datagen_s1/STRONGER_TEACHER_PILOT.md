# STRONGER_TEACHER_PILOT — priced three-way comparison for closing the keeper gap

**Status:** decision doc. Written 2026-07-08. The Opus arm is a **live pilot still in flight**
(docker image-pull phase, pull 4/10 at time of writing; 0/10 episodes executed, 0 pilot keepers).
Every Opus yield/cost number below the pilot line is therefore a **prior-based projection**, not a
measurement — clearly labeled as such. The 9B and token-cost numbers are **measured** from the 282
production keepers. No production artifact was touched to produce this doc; pilot output lands in
`pilot_opus/`, never `keepers/keepers.jsonl`.

---

## 0. The gap we are pricing

| Milestone | Keepers | Delta from current (282) |
|---|---|---|
| **Current** (production `keepers/keepers.jsonl`) | 282 | — |
| **Floor** (400) | 400 | **~120 more** |
| **Target** (600–1000) | 600–1000 | **~320–720 more** |

The campaign **self-killed** on 2026-07-08 15:20Z (`KILL_YIELD_COLLAPSE`, rolling yield 0.075 < 0.10
over the last 200 attempts) — an **honest** yield collapse on the *spent* near-miss retry stock, not a
tamper. The fresh-coverage frontier (pandas/getmoto/dask never-attempted ids) was swapped in live but its
yield is **unmeasured** (probe pending, task #94/#95). So "close the gap with more 9B" is not a
free continuation — it depends on a fresh yield we do not yet have.

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

## 2. Opus 4.8 teacher — PILOT IN FLIGHT (projected cost, labeled)

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
4. Opus 4.8 pilot: IN FLIGHT (docker pull 4/10; 0/10 episodes; yield NOT yet measured) — adapter+OAuth+scoring plumbing proven, self-completing into pilot_opus/.
5. Opus pricing (prior, labeled): $5/$25 per 1M in/out; cache read ~$0.50/1M, write ~$6.25/1M.
6. Opus cost (projected, 9B token prior): $3.17/episode uncached → $0.77 cached; $/keeper $6–11 uncached / $1.5–2.6 cached at yield 0.3–0.6.
7. Opus totals @ yield 0.4: floor ~$950 uncached / ~$230 cached; full target ~$2.5–5.7k uncached / ~$0.6–1.4k cached — CACHING is the ~4× lever and the adapter must add it before scale.
8. 27B NVFP4: on-disk, fits 32 GB, vLLM-supported, NO MTP; ~$0/token but GPU-contended; ~0.5–1 day standup; projected yield 0.20–0.35 (the free-but-contended middle).
9. Format equivalence: identical BY CONSTRUCTION + smoke-verified, but NOT yet certified on a real teacher keeper row — blocking gate before promotion. Leakage posture UNCHANGED (holdout-clean pilot, isolated keepers).
10. Recommend: measure fresh 9B yield first; 9B for floor if ≥0.15; 27B when GPU idle; Opus (caching ON) for the hard residual only when fresh 9B <0.10 and Opus pilot ≥0.30.
