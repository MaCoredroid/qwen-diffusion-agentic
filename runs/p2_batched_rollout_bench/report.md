# Batched RL-rollout throughput: FLARE hybrid-clean engine vs stock-AR-guided vLLM

**Question (the RL-rollout regime):** the diffusion twin is supposed to generate on-policy RL
signal at high throughput by batching rollouts. Batch-correctness is already settled
(`runs/p2_engine_batchgates`: **NO cross-request contamination** — the batched path is SAFE).
This bench answers the *quantitative* half: **rollout samples/sec/GPU, engine vs stock-AR-guided,
across batch {1,2,4,8,16}** — does the FLOP-reducing hybrid *hold or grow* its advantage with
batch (the a-priori thesis), while per-request latency stays sane?

## Verdict

**The thesis is DISCONFIRMED on this hardware/workload.** Against a *fast* guided-AR baseline the
engine is at **rollout parity at batch=1 (0.94x) and LOSES ground as batch grows — down to 0.73x
at batch=16.** The hybrid genuinely does ~10x fewer forwards/turn (4.9 vs ~50), but that FLOP
reduction does **not** convert into throughput because (1) each engine forward is ~14x costlier
(35 ms vs 2.5 ms at b16) and (2) the FLARE **sync scheduler** + per-request variable draft widths
give **poor batch occupancy** (effective 7.2/16 = **0.45** at b16; it never co-batches all 16),
whereas guided-AR co-batches near-linearly (4.83x at b16) at 100% GPU util. The engine's real,
measured wins remain elsewhere: **batching is safe**, per-turn latency is **at parity** with fast
guided-AR at low batch, rollouts are **48/48 valid** tool-call stops (guided-AR truncated 2/48),
and quality was already certified higher (scoreboard 130 vs 124 exact_args). But it is **not a
rollout-throughput multiplier vs fast guided-AR — it is ~0.7-0.9x.**

## Method (apples-to-apples)

| | ENGINE | AR baseline |
|---|---|---|
| system | FLARE hybrid-clean, converted 9B (`qwen3.5-9b-fastdllm-rlv2-vllm-bf16`), vLLM pin `95d8b47` | stock `Qwen3.5-9B` snapshot `c202236`, vLLM 0.23.0 |
| decode | hybrid-clean masked-diffusion, PIECEWISE cudagraph + APC (the certified v3b/nevertrain config) | guided: `structured_outputs = regex_from_qwen_xml_tool_schema` — **the exact `guided_tool_call_regex` the endgame scoreboard used** |
| cudagraph | on (`VLLM_FLARE_CUDAGRAPH=1`, BIDIR_PROBE=1) | on (`enforce_eager=False`) — **fast path** |
| gpu_mem_util | 0.62 | 0.66 |
| sampling | **temp=0.7 seeded** (the RL mode) + one greedy b8 point | **temp=0.7 seeded** |

- **Pool:** 48 never-train BFCL/API-Bank tool-call turns (`nevertrain_ref.json`, prompt_len band
  467-1299, nref_mean 50.8). 48 is divisible by 1/2/4/8/16 so **every batch times exactly 48 turns**.
- **Harness (identical both sides):** one boot; for each B, process the pool in **waves of B**
  concurrent requests (one `generate()` per wave; prefix cache reset per wave so waves are cold and
  independent — the RL model of many independent prompts). Warm-up wave per B excluded from timing.
  `samples/sec = 48 / total_wall`. Same per-turn `max_tokens = n_ref+16` and stop-token set both sides.
- **Why guided-AR (not plain-AR) is the fair baseline:** like the hybrid, guided decoding emits
  **one valid Qwen-native XML tool call** per turn. Plain free-form AR is not a scoreable rollout.
- **Conservative choice for the engine thesis:** AR is given its *fastest* config (offline +
  cudagraph), **not** the scoreboard's server+eager config. So the engine has to beat AR at AR's best.
- **Metrics:** turns/sec, generated tok/sec, per-forward ms (engine) / per-decode-step ms (AR),
  forwards/turn, batch occupancy, GPU util (background `nvidia-smi` sampled *across* each timed wall,
  ~5 Hz), GPU mem peak, host-RAM peak (`ru_maxrss`). Raw per-point rows in `engine_points.jsonl`,
  `ar_points.jsonl`; joined in `compare.json`.

## The two throughput curves + the ratio that matters

samples/sec == turns/sec == rollouts/sec/GPU. `eng/AR` is the headline.

| batch | eng samp/s | AR samp/s | **eng/AR** | eng tok/s | AR tok/s | eng scale | AR scale | eng fwd/turn (ms/fwd) | AR tok/turn (ms/step) | eng occ (eff) | eng util% | AR util% |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.524 | 1.625 | **0.94x** | 75 | 82 | 1.00x | 1.00x | 35.2 (18.7) | 50.3 (12.23) | 1.0 (1.00) | 88 | 100 |
| 2 | 2.248 | 2.499 | **0.90x** | 109 | 126 | 1.48x | 1.54x | 22.5 (19.7) | 50.3 (7.96) | 1.5 (0.77) | 88 | 100 |
| 4 | 3.426 | 4.103 | **0.83x** | 167 | 207 | 2.25x | 2.52x | 12.9 (22.6) | 50.5 (4.82) | 2.7 (0.68) | 87 | 100 |
| 8 | 4.948 | 6.601 | **0.75x** | 243 | 333 | 3.25x | 4.06x | 7.4 (27.2) | 50.5 (3.00) | 4.8 (0.60) | 87 | 100 |
| 16 | 5.732 | 7.846 | **0.73x** | 279 | 395 | 3.76x | 4.83x | 4.9 (35.3) | 50.3 (2.53) | 7.2 (0.45) | 84 | 100 |

Greedy engine b8 = **4.798 samp/s / 237 tok/s** ~= temp=0.7 b8 (4.948) -> **throughput is
temperature-insensitive** (confirmed; the RL sampling mode costs nothing vs greedy).

**Read of the ratio curve:** it goes the *wrong way* for the thesis — **0.94 -> 0.90 -> 0.83 -> 0.75 ->
0.73**. The hybrid does not amortize its bs=1 weight-stream floor faster than AR; AR amortizes faster
(its scale 4.83x > engine 3.76x). At *every* batch the engine is slower per rollout, and the gap widens.

## Why the FLOP reduction doesn't win (the mechanism, measured)

The hybrid really is FLOP-reducing on **forward count**: **4.9 forwards/turn at b16 vs AR's ~50 decode
steps** (~10x). But two measured effects cancel it:

1. **Per-forward cost is ~14x AR's.** Each hybrid forward processes a full block width + the GDN
   clean-state advance + the grammar FSM. `per_forward_ms` climbs **18.7 -> 35.3 ms** with batch (wider
   GEMMs), while AR's `per_decode_step_ms` *falls* **12.2 -> 2.53 ms** as cudagraph amortizes the weight
   load across the batch. 4.9 x 35.3 ms ~= 173 ms/turn of forward time ~= 50 x 2.53 ms ~= 127 ms — AR is
   simply cheaper per turn once batched.

2. **Batch occupancy collapses (the sync-scheduler straggler effect — the answer to "queueing / head-of-line?").**
   The FLARE sync scheduler co-batches denoise forwards across requests, but requests have **variable
   draft widths** (measured 3-18 tokens/commit) and finish their tool call at **very different forward
   counts**, so a synchronous wave is held by its slowest member (head-of-line/straggler). Effective
   batch-in-forward vs requested B:

   | B | mean batch-in-forward | efficiency | forwards at full width B |
   |---:|---:|---:|---:|
   | 1 | 1.0 | 1.00 | 100% |
   | 2 | 1.5 | 0.77 | 54% |
   | 4 | 2.7 | 0.68 | 38% |
   | 8 | 4.8 | 0.60 | 30% |
   | 16 | **7.2** | **0.45** | **0% (never all-16)** |

   b16 occupancy histogram (num_reqs per forward): `{1:21, 3:22, 4:51, 5:16, 10:25, 13:60, 14:1, ...}` —
   the batch is scattered across widths and **never reaches 16**. AR, whose 48 turns all take ~50 near-
   identical decode steps, co-batches near-perfectly and **pins the GPU at 100%**; the engine idles at
   **84-88%** util (the ~13% gap is host-bound FLARE per-request state management, not compute).

## Sync-scheduler behavior, honestly

- **No hard queueing / no deadlock:** at every batch all 48 rollouts finish `stop` (bounded, valid
  tool call). The scheduler co-batches genuinely (mixed commit+denoise forwards present: 11-117
  committing forwards per point).
- **Head-of-line / straggler blocking is the real cost.** Because FLARE *forces* the sync scheduler
  and requests complete at wildly different forward counts, waves stall on stragglers -> occupancy
  0.45 at b16. This is intrinsic to the variable-draft-width hybrid decode, not a harness artifact
  (the wave method is a faithful model of the sync-scheduler regime). A **continuous-batching** rollout
  driver that refills slots as requests finish would recover some occupancy — but within FLARE's sync
  constraint that recovery is bounded; it is the clearest lever if rollout throughput becomes the goal.

## Memory / capacity tradeoff (a real cost of batched hybrid rollouts)

- **AR:** flat **~21.9 GB** (0.66 gmu, half the 32 GB card) at every batch — no per-request state.
- **Engine:** the per-request GDN **snapshot/restore** state + cudagraph pools make it far more
  memory-hungry at batch. At **gmu 0.74 the b16 wave OOMs** (16 MiB alloc fails at the b16 decode
  spike, ~31.3 GB in use) — the run confirms genuine 16-way co-batching with per-request variable
  widths was scheduled before the spike. b16 only fits after **dropping gmu to 0.62** (steady ~25.5 GB;
  gmu 0.55 is too low — weights leave no KV blocks). Net: **the engine's practical rollout concurrency
  is capped tighter than AR's on a 32 GB card**, and the per-request GDN state is why.
- Host-RAM peak (inside the 22 GB cage): engine **6.91 GB**, AR **6.87 GB** — both comfortable.

## Caveats / what this does and doesn't say

- **This is a floor, not a ceiling, for the engine.** GPU util 84-88% (vs AR 100%) shows host-bound
  headroom; the still-open **OPT-4 part-1 fused_recurrent** routing (task #37) targets exactly the
  per-forward GDN overhead — lowering `per_forward_ms` would move the whole ratio curve up.
- **The engine's value proposition is not raw rollout throughput.** It is: safe batching (proven),
  **latency parity** with fast guided-AR at low batch, **48/48 valid** stops (guided-AR truncated
  2/48 at the token cap), and the certified **quality edge** (130 vs 124 exact_args aggregate). If the
  RL loop is throughput-bound, **stock guided-AR is the faster rollout generator** (1.1-1.4x at batch);
  the hybrid twin earns its place on *quality/parity at safe batch*, not on samples/sec.
- Scope: single RTX 5090, 9B, 48 moderate-length never-train turns, wave harness, temp=0.7 seeded.

## Artifacts

- `engine_throughput.json` / `engine_points.jsonl` — engine sweep (gmu 0.62; `*.gmu074.*` = the
  gmu-0.74 run where b16 OOM'd, kept as the memory-ceiling evidence).
- `ar_throughput.json` / `ar_points.jsonl` — AR-guided sweep.
- `compare.json` / `compare_table.md` — the joined ratio curve.
- `bench_engine.py`, `bench_ar_guided.py`, `gpu_sampler.py`, `make_report.py`, `runcage_*.sh`,
  `tp_engine.log`, `tp_ar.log` — harness + raw logs.
- Correctness precondition: `runs/p2_engine_batchgates` (no cross-request contamination).
