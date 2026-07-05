# P2 Engine Battery v3b — the PROMOTION ATTEMPT on the POST-FIX engine (OPT-4 Stage 1+2+3 landed) (2026-07-04)

vLLM pin **`95d8b47`** (`qwen3_5-flare-modelstate`): OPT-4 **Stage 1** (authoritative
32-absolute variable commit width + single-gate `BIDIR_PROBE`) + **Stage 2** (per-request
variable draft-width scheduler plumbing) + **Stage 3** (byte-robust bidir key window; the
one real bug the variable-width path exposed) — all landed, **code default OFF**. Export
`models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16` (block/canvas 32, mamba 1024, align+APC),
RTX 5090 / sm_120, one heavy process per boot in the `systemd-run … MemoryMax=22G
MemorySwapMax=4G` cage. Greedy, temp 0, seed 20260701, **uncapped** (`max_tokens = n_ref+16`).
FINAL engine = `VLLM_FLARE_BIDIR_PROBE=1` + `VLLM_FLARE_CUDAGRAPH=1` (PIECEWISE). Chunked
foreground: two boots (ep0-9 = gt0-31, ep10-19 = gt32-62), per-turn incremental JSONL.

This is an **independent fresh boot of the Stage-3 engine** and the explicit promotion
attempt. It is NOT a re-run of v2/v3: those were the **pre-fix** pin `e5496cc` (58/63); the
Stage-3 fix clears the {20,21,45,60} bidir/alignment breaks. v3b reproduces the Stage-3
result byte-for-byte on an independent boot.

## Headline

The Stage-3 fix takes full-63 byte-parity from the pre-fix **58/63 -> 62/63** and makes
**exact_args EXACTLY 47/63, byte-identical to HF per turn** (zero engine wins, zero losses).
**valid 63/63, episode_exact 13/20, verify_invariants 63/63, value_projection 0/63.** The
**lone remaining break is gt44** — a proven **path-invariant deterministic fp-residue**
(breaks identically under APC-on and cold-prefix fresh-boot; both engine and HF are non-exact
there, so it is quality-neutral). The strict promotion gate (**63/63 byte-parity => exact
exactly 47**) is **one turn short -> NOT MET -> NOT PROMOTED**. Code default stays **OFF**.

## Promotion gate

| gate check | required | measured | verdict |
|---|---|---|---|
| byte-parity / turn | **63/63** | **62/63** (lone break {44}) | **NOT MET** — 1 turn short |
| exact_args | **exactly 47** | **47** (0 turns eng!=hf) | **MET** |
| episode_exact | 13/20 | **13/20** (ties HF) | **MET** |
| valid | 63/63 | **63/63** | **MET** |
| verify_invariants / value_projection | clean / 0 | **63/63 / 0/63** | **MET** |
| **PROMOTED** | 63/63 => 47 | — | **NO** |

Per the task rule "byte-parity 63/63 => exact exactly 47; any deviation => stop and diagnose":
byte-parity is 62/63, so the by-construction chain that would *force* exact=47 does not close —
but exact **is** exactly 47 anyway (the lone break gt44 is non-exact for BOTH engine and HF,
so it does not move the exact count). **No deviation on the quality axes.** The single blocker
is the 1 residual parity turn.

## The Stage-3 fix landed: 58/63 -> 62/63 (delta vs pre-fix v3)

| | pre-fix v3 (`e5496cc`) | **post-fix v3b (`95d8b47`)** |
|---|---|---|
| byte-parity | 58/63, breaks **{20,21,44,45,60}** | **62/63, break {44}** |
| exact_args | **48** (engine WON gt60) | **47** (byte-matches HF on gt60) |

- **Fix cleared {20,21,45,60}** — all four now byte-match HF, **path-robustly** (clean in BOTH
  the APC-on battery AND the cold-prefix fresh-boot certificate below). No regressions
  (`new_regressions = []`).
- **gt60**: the pre-fix engine's exact-48 was an "engine wins" (correct where HF is wrong). The
  fix makes the engine **byte-identical to HF on gt60** (incl. HF's mistake), dropping exact
  48 -> **exactly 47**. This is the by-construction gate behaving as designed.
- **Shared-clean turns byte-identical:** on the 58 turns clean in both pins, `n_gen` and exact
  are identical (`shared_clean_ngen_identical=True`, `shared_clean_exact_identical=True`).

## The lone break gt44 — proven path-invariant fp-residue, NOT an APC class

| gt | ep/t | first_div | pos%32 | n_gen/n_ref | finish | proj | eng/hf exact | valid |
|---|---|---:|---:|---|---|---:|---|---:|
| 44 | ep14/t0 | 16 | 16 | 101/99 | stop | 0 | **0/0** | 1 |

**Fresh-context parity certificate** (documented protocol: one turn per fresh boot = single
request -> cold prefix cache, no cross-turn KV reuse) on the 5 pre-fix break turns:

| gt | APC-on parity | fresh-boot parity | class |
|---|---|---|---|
| 20 | clean | clean | fix-cleared, path-robust |
| 21 | clean | clean | fix-cleared, path-robust |
| 45 | clean | clean | fix-cleared, path-robust |
| 60 | clean | clean | fix-cleared, path-robust |
| **44** | **break (fd16, n101)** | **break (fd16, n101)** | **path-invariant fp-residue** |

gt44 breaks **identically** under APC-on and cold-prefix fresh-boot — same `first_div=16`, same
`n_gen=101` — so it is **NOT** an APC/prefix-cache-class artifact. The **documented APC protocol
cannot rescue it to 63/63.** This matches the pin's own docstring: on block#0 the fold *path*
differs from HF (HF folds 32 incl. `prompt%32` leftover from the `(L//32)*32` checkpoint; the
aligned engine folds `32 - L%32` gen tokens from the `L` checkpoint) — fp-close but not
bit-identical, and on gt44's specific canvas it flips one argmax at gen-16. It is deterministic
(same 101 tokens under APC-on chunk2 and fresh-boot). Both engine and HF are non-exact on gt44,
so **quality is unaffected**; it is purely a byte-parity residue.

> Note: the in-boot `BENCH_APC_OFF=1` hook (`enable_prefix_caching=False`) fails `VllmConfig`
> validation under the diffusion align-cache config, so the fresh-context certificate uses the
> v3 per-turn-fresh-boot protocol (single request per boot = cold prefix), which is the
> documented proxy.

## Timing — the first engine full-63 UNDER M2, guided-AR, and HF (holds post-fix)

| metric | v3b (post-fix) | HF full-63 |
|---|---:|---:|
| s/turn **mean** | **1.053** | 3.904 |
| s/turn **p50** | **0.896** | — |
| s/turn **p90** | **1.700** | — |
| s/turn min / max | 0.339 / **4.241** | — |
| worst turn | gt50 (259 tok, 242 fwd, 4.241 s) | — |
| **TRUE denoise fwd/turn** | **56.86** | 56.83 |
| tokens / forward | 1.361 | — |
| **per-forward ms** (amortized, cudagraph) | **18.52** | — |
| per-forward ms (long-turn settled >=80 fwd) | 18.05 | — |
| **speedup vs HF** | **3.708x** | — |
| PIECEWISE cudagraph | **63/63 turns**, 3798 dispatches | — |

## Bar adjudication (mean 1.053 s/turn)

| bar | value | ratio | verdict |
|---|---:|---:|---|
| HF hybrid-clean | 3.904 | 0.270x | **UNDER (beat)** |
| guided-AR (stock-bf16 matched-20) | 1.213 | 0.868x | **UNDER (beat)** |
| **M2 / K3** | **1.120** | **0.940x** | **UNDER (beat)** |
| stock-AR aggregate | 0.741 | 1.421x | **OVER (miss)** |

## Residual gap to stock-agg 0.741 — MEASURED; NOT reachable by engine plumbing at batch=1

**This supersedes the section-0.G/v3 "REACHABLE via OPT-4 Part 1" claim.** OPT-4 Part 1
(per-request variable commit width) is now **landed** (Stage 1+2), and the Stage-3 A/B measured
it **speed-NEUTRAL** (variable-width 18.52 vs fixed-32 18.56 ms/forward) — cudagraph buckets
narrow widths back to a captured bucket, so the CL=32 gemm/attn shape was **not** the residual.
The honest, measured per-forward decomposition (torch.profiler device self-time, current pin;
`opt4_breakdown.json`):

| component | ms/forward | note |
|---|---:|---|
| amortized per-forward **wall** (cudagraph) | **18.52** | `1000*sum(wall)/sum(fwd)` over the battery |
| — measured GPU compute (settled) | 17.94 | profiler device self-time, gt25/gt35 |
| — — **weight-stream floor** (gemm MLP+proj+lm_head) | **11.40** | **63.5%** of GPU; **irreducible at bs=1** |
| — — non-weight GPU compute (attn+GDN+norm+ew+samp) | 6.54 | the "shape" — **proven NOT width-reducible** |
| — residual host/launch (cudagraph) | 0.58 | cudagraph already near-eliminated host overhead |
| weight-stream floor — arithmetic cross-check | **10.77** | 19.31 GB bf16 / 1.79 TB/s HBM (RTX 5090) |

To hit **0.741** at 56.86 fwd/turn needs **13.03 ms/forward** (cut 5.49 ms). The bar target sits
only **1.64 ms above the weight-stream floor**, but the non-weight per-forward compute is
**6.54 ms** and Stage-3 A/B proved it does **not** shrink with variable width -> **0.741 is not
reachable by width-narrowing / engine plumbing at batch=1.** The reachable levers are orthogonal
to the parity/integration work:

1. **Fewer forwards/turn** — larger effective parallel commit (model/schedule/**training**, not
   engine plumbing).
2. **Lighter weight stream** — fp8/int8 weights halve/quarter the 11.40 ms floor -> per-forward
   ~12 / ~9 ms -> ~**0.68 / 0.51 s/turn** (would beat 0.741), a **quality tradeoff** (breaks byte-parity).
3. **Batching** — amortize the weight-stream floor across concurrent requests (serving
   throughput, not single-stream latency; matched-20 is bs=1 serial).

Caveat: stock-agg 0.741 is a **stock-AR** number over a **different, shorter** turn mix (49.06
tok/turn vs matched-20's 56.86 fwd/turn), so it is not a like-for-like single-stream comparison.

## temp = 0.7 rollouts (RL-rollout sanity + reproducibility)

5 seeded rollouts (gt0/7/17/29/51), **two independent boots (a/b), same seed 20260701**:

| gt | n_gen | fwd | finish | valid | exact(hf) | proj | parity | wall a / b |
|---|---:|---:|---|---:|---|---:|---|---|
| 0 | 42 | 23 | stop | ok | 1(1) | 0 | yes | 0.528 / 0.525 |
| 7 | 36 | 16 | stop | ok | 1(1) | 0 | yes | 0.378 / 0.377 |
| 17 | 31 | 15 | stop | ok | 1(1) | 0 | yes | 0.382 / 0.383 |
| 29 | 44 | 26 | stop | ok | 1(1) | 0 | yes | 0.570 / 0.566 |
| 51 | 47 | 27 | stop | ok | 1(1) | 0 | yes | 0.573 / 0.570 |

**Byte-reproducible across both boots** (identical n_gen/fwd/parity; max wall delta = **4 ms**).
All bounded, grammar-valid, `value_projection_events=0`, parity-full (peaked value distributions
collapse temp-0.7 onto the greedy tokens). The RL contract holds under the post-fix engine.

## Never-train spot-check (BFCL / API-Bank, not matched-20-specific)

3 turns spanning families, prompts `prompt_sha256`-verified against the HF never-train eval:

| gt | family | parity | n_gen/n_ref | valid | exact(hf) | proj | ms/f | wall |
|---|---|---|---|---:|---|---:|---:|---:|
| 147 | BFCL-AST | **yes** | 54/54 | ok | 1(1) | 0 | 19.92 | 0.737 |
| 159 | API-Bank-Lv1 | **yes** | 37/37 | ok | 1(1) | 0 | 21.52 | 0.452 |
| 172 | API-Bank-Lv2 | **yes** | 38/38 | ok | 1(1) | 0 | 21.77 | 0.479 |

**3/3 byte-parity vs HF, 3/3 valid, 3/3 exact-correct, 0/3 projection.** The parity chain is not
matched-20-specific; the fix holds on out-of-distribution never-train prompts too.

## Method notes / integrity

- **Independent fresh boot** on the Stage-3 pin `95d8b47` (editable-installed; `vllm.__file__` ->
  the pin source, no rebuild — Stage-3 diff is one Python file, +18/-2). Boot logs confirm
  `enforce_eager=False`, `cudagraph_mode=PIECEWISE`, `apc_live=True`, patched step.
- Prompts byte-identical to the HF eval (63 matched-20 `prompt_sha256`/`prompt_tokens` verified;
  184 never-train verified). Parity scored token-for-token vs `HF.generated_token_ids`; exactness
  independently via `score_tool_calls`.
- **Determinism:** gt44's break is byte-identical across the APC-on battery (chunk2) and the
  fresh-boot certificate (fd16, n101). temp-0.7 a/b byte-identical.
- GPU pre-flight (`nvidia-smi used < 2 GB`) before every boot; RAM cage on every heavy process;
  one heavy process at a time.

## Verdict

The FINAL post-fix engine (bidir + PIECEWISE cudagraph + OPT-4 Stage 1/2/3) is the strongest
promotable candidate to date: **62/63 byte-parity, exact EXACTLY 47/63 (== HF, 0 wins/losses),
valid 63/63, episode 13/20, verify 63/63, projection 0, mean 1.053 s/turn — beating M2 (1.120),
guided-AR (1.213), and HF (3.904) on the honest full-63.** But the strict **63/63** gate is **one
turn short**: the lone residual gt44 is a proven **path-invariant deterministic fp-residue** (not
an APC class, not a quality loss — both engine and HF are non-exact there), rooted in the block#0
fold-path fp gap the pin documents. **NOT PROMOTED; code default stays OFF.** The stock-agg 0.741
speed bar is **not** an engine defect and **not** closable by width-narrowing (measured); it is
the batch-1 weight-stream floor + fixed per-forward cost, addressable only by fewer forwards/turn
(training), lighter weights (quantization), or batching.

## Artifacts (`runs/p2_engine_battery_v3b/`)

`matched20_turns.jsonl` (63) . `aggregate.json` (headline + gate + delta-vs-prefix + fresh cert +
temp07 + nevertrain + measured residual-gap) . `matched20_temp07a.jsonl` / `matched20_temp07b.jsonl`
(5 rollouts x2) . `nevertrain_spotcheck.jsonl` (3) . `parity_cert_freshboot.jsonl` (5 fresh boots) .
`opt4_breakdown.json` (profiler family split) . `run_battery_v3b.py` / `aggregate.py` /
`profile_v3b.py` / `env.sh` / `runcage.sh` . `chunk1.log` / `chunk2.log`.
