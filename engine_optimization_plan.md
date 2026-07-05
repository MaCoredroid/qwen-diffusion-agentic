# FLARE hybrid_clean — Engine Optimization Plan (per-turn latency)

Synthesis of four independent CPU-only static/micro-bench audits (`graphs`,
`cache-batch`, `overhead`, `kernels`) of the vLLM FLARE `hybrid_clean` decode
path. Deduped, ranked by expected ms/turn payoff-per-effort, and split into
P0 / P1 / P2 against the two bars.

- **Author:** engine-optimization synthesis sweep. **Date:** 2026-07-04.
- **vLLM tree audited:** `/home/mark/shared/vllm_p2_pr42406`, HEAD `6b81154`
  ("single-[MASK] forward VIEW via causal-windowed probe (GAP 5A)"), working
  tree **clean**. (All four reports independently confirmed the "windowed-probe
  fix in flight" is already *committed*, one commit past the `1e32dcd` in the
  brief.) All line numbers below are at `6b81154`.
- **Method:** static analysis + one CPU microbenchmark of the pure-Python host
  loop (`overhead`). **No GPU / CUDA was touched.** Every wall-clock translation
  to GPU time is marked **UNVERIFIED**; the host-loop constants are **MEASURED**
  on CPython and re-scaled to the true vocab below. Each item carries a
  GPU-verification step to convert its estimate to a measured number.

## STATUS (2026-07-04, after the v3 PROMOTION ATTEMPT — vLLM pin `e5496cc`, tree byte-identical to v2)

> **v3 battery = THE PROMOTION ATTEMPT (NOT promoted).** The strict gate is **63/63 byte-parity ⇒ exact exactly 47**.
> The engine tree is **clean at `e5496cc` = byte-identical to v2** (OPT-4 Part 1 / Task #37 UNLANDED), so v3 is a
> faithful promotion attempt + independent 3rd boot: it **reproduces v2 exactly** (n_gen/fwd/parity/exact/first_div ALL
> identical) and adds a fresh-context parity certificate. **Measured (APC-on, v2 protocol): byte-parity 58/63** (breaks
> {20,21,44,45,60}, NOT met), **exact 48** (+1 = gt60 APC win, so exact≠47), episode 13/20, valid 63/63, verify 63/63,
> projection 0/63 ⇒ **NOT PROMOTED.** Timing reproduces v2: **s/turn mean 1.056** (p50 0.874, p90 1.734, worst 4.253),
> 56.62 TRUE fwd/turn, per-forward 18.66 ms. **Bars (1.056): HF 3.904 BEAT (0.270×) · guided-AR 1.213 BEAT (0.871×) ·
> M2 1.120 BEAT (0.943×) · stock-agg 0.741 MISS (1.425×).** **THE v3 FINDING:** byte-parity is **cache-path-dependent**;
> a fresh-context certificate (cold cache, fresh boot/turn; 57/63 measured, 6 pending under a concurrent Stage-3 GPU
> hold) localizes the **invariant structural residual to {44,45}** (both paths — gt44 variable-width, gt45 32-absolute
> align), with {20,21,60} APC-only (cross-turn prefix-cache artifacts) and {1,3,12,23,24,50,57} fresh-only. gt60's
> exact-win is an APC artifact (fresh, it copies HF's mistake → exact 0 = hf). So the robust blocker is the 2-turn set
> **{44,45}** = **OPT-4 Part 1** (variable commit width + 32-absolute align), which ALSO cuts per-forward **18.66→13.09
> ms** to reach stock-agg 0.741 (weight-stream floor 10.5 ms; 2.59 ms above floor, REACHABLE) — **parity closure and the
> last speed cut land together.** temp=0.7 (5×2 boots) byte-reproducible + never-train 3/3 byte-parity/exact vs HF.
> Source: `p2_engine_battery_v3_result.md`, `runs/p2_engine_battery_v3/report.md`; build-status §0.G; battery commit
> `55965de` (pushed origin/main). **No engine row added to the endgame scoreboard** (gate not met).
>
> **PRIOR — FINAL-engine v2 battery (bidir probe `b7d76e2` + PIECEWISE cudagraph `VLLM_FLARE_CUDAGRAPH=1`, OPT-4 Part 2):
> the strongest promotable candidate yet — engine now BEATS M2/guided-AR/HF on speed for the FIRST time on the
> honest full-63, but the strict 63/63-byte-parity gate is STILL NOT met (58/63) ⇒ NOT PROMOTED.** Required checks:
> **byte-parity 58/63** (breaks {20,21,44,45,60}, NOT met), exact_args **48** (+1, engine WINS gt60 ≥HF), episode
> **13/20** (met), valid **63/63** (bidir fixed gt19, up from 62), `value_projection 0/63`, `verify_invariants 63/63`.
> **Timing: s/turn mean 1.051** (p50 0.876, p90 1.699, worst 4.248), **56.62 TRUE fwd/turn**, per-forward **18.56 ms**
> (eager ~29 → **1.615× cudagraph win**), **3.715× under HF**. **Bar adjudication (1.051): HF 3.904 BEAT (0.269×) ·
> guided-AR 1.213 BEAT (0.866×) · M2-speed 1.120 BEAT (0.938×) · stock-AR-agg 0.741 MISS (1.418×).** The §0.E eager
> engine (1.681) missed M2 + guided-AR; **cudagraph clears both for the first time.** **M2/K3:** speed bar MET
> (1.051 < 1.120, K3 speed MET); quality axis MISSED (58/63 ≠ 63/63; exact 48 < 55) ⇒ combined M2 gate not met.
> temp=0.7 (5×2 boots) byte-reproducible + never-train spot-check (BFCL/API-Bank, sha-verified) 3/3 byte-parity/valid/
> exact — contract holds under cudagraph, not matched-20-specific. **Why 63/63 unreachable here:** bidir is the
> correct semantics; cudagraph is **byte-neutral on the promotable set** (reproduces the bidir-eager 58/63 + break-set
> exactly). The 5-turn residual + the single exact deviation are the **coupled UNLANDED OPT-4 Part 1**: 32-absolute
> commit alignment (`VLLM_FLARE_ALIGN_BLOCKS`, scaffold only) + per-request variable commit width — **parity closure
> and the last speed cut to stock-agg land together.** Break classes: gt20/gt45 bidir alignment, gt21 APC artifact,
> gt44 variable-width, gt60 engine WINS. Source: `p2_engine_battery_v2_result.md`, `runs/p2_engine_battery_v2/report.md`;
> build-status §0.F; battery commit `1acdf2e` (pushed origin/main).
>
> **PRIOR (§0.E, `d2fccab`): P0 progress:** **OPT-1 DONE + verified clean** (GPU-native batched sampling; byte-identical A/B, 2.36×).
> **OPT-2 landed** (`cg_mode` counter + fail-closed config assert). **OPT-3 core blockers LANDED** (sync
> scheduler): the full 63-turn battery now **completes end-to-end for the first time** — **zero stalls**,
> async-rollback divergence@33 gone (0/11 breaks at pos-33). **OPT-5 DISPROVEN → do NOT do** (grammar =
> 0.7% of turn). **OPT-4 is now the single next lever.**
>
> **FIRST COMPLETE full-63 engine wall-clock — engine NOT promoted; M2/K3 now ADJUDICABLE and MISSED on both
> axes.** The 63/63 byte-parity promotable gate is **NOT met — measured 52/63** (the 11 divergences are the
> *separate, author-flagged* windowed-**causal** vs reference windowed-**bidirectional** approximation, all
> `proj=0`, `first_div` scattered {17,19,19,19,26,31,34,38,41,47,53}, **none at 33**). Aggregate quality is
> **≥ HF but not byte-identical**: exact_args **48/63** (+1 vs HF 47), episode_exact **13/20** (met), valid
> **62/63** (−1). Timing: **s/turn mean 1.681** (p50 1.427, p90 2.724, worst 5.361), **56.65 TRUE denoise
> fwd/turn** (HF 56.83), **2.32× under HF**. **M2 (quality ≥55, speed <1.120) MISSED both:** speed 1.681 >
> 1.120 (1.39× *slower* than guided-AR 1.213), quality 48 < 55. **K3 speed MISSED.** But it is now measurable.
> Next lever = **OPT-4** (incremental KV+GDN 1-token decode → `fused_recurrent` + FULL graph): profiler shows
> GDN on the prefill `chunk_gated_delta_rule` path (`fused_recurrent` absent), ~18 ms GPU + ~11 ms host/forward,
> `enforce_eager`. Source: `p2_engine_battery_result.md`, `runs/p2_engine_battery_full/report.md`; build-status
> §0.E; battery commit `61d1381`.

## Bars and the current budget

| row (matched-20, full-63 unless noted) | exact | s/turn | fwd-or-tok/turn |
|---|---:|---|---|
| **ENGINE FINAL full-63 (bidir + PIECEWISE cudagraph, `e5496cc`)** | **48/63** | **1.051** (p50 0.876, p90 1.699, worst 4.248) | 56.62 TRUE denoise fwd/turn |
| ENGINE §0.E (OPT-3 eager, causal, `d2fccab`) | 48/63 | 1.681 (p50 1.427, p90 2.724, worst 5.361) | 56.65 TRUE denoise fwd/turn |
| OUR HF hybrid-clean (v2) | 47/63 | 3.904 | 56.83 denoise fwd/turn |
| stock-bf16-AR-guided | 51/63 | 1.213 | 82.24 tok/turn |
| **stock-AR aggregate** | 124/247 | **0.741** | 49.06 tok/turn |

- **M2 / K3 target:** **quality ≥55/63 AND < 1.120 s/turn** (just under guided-AR). — the P0 bar.
  **Status (FINAL engine): the SPEED axis is now MET for the first time — 1.051 < 1.120 (0.938×), also BEATING
  guided-AR 1.213 (0.866×) and HF 3.904 (0.269×) at ≥HF quality; K3 speed MET.** The eager §0.E engine (1.681)
  missed both; the PIECEWISE cudagraph (per-forward 29→18.56 ms, 1.615×) is what clears them. **The QUALITY axis is
  still MISSED** (byte-parity 58/63 ≠ 63/63; exact 48 < 55) ⇒ combined M2 gate not met, **NOT PROMOTED**. Still above
  the stock-AR aggregate 0.741 (1.418×, the beyond-AR / OPT-6 bar).
- **Beyond-AR / thesis KPI:** **< 0.741 s/turn** at equal quality via
  forwards-saved. — the P2 bar.
- The engine wall-clock is now **fully honest and complete**: OPT-1 removed the host-sampling wall (2.36×) and
  the **OPT-3 sync-scheduler fix removed the stall + async-rollback divergence** (63/63 complete, 0 stalls, 0/11
  breaks at pos-33). The residual gap is (a) **speed** — the forward is on the prefill GDN chunk path (`enforce_eager`,
  no CUDA graph), the OPT-4 target; and (b) **11/63 residual byte-parity divergences** — the separate
  windowed-**causal** vs reference windowed-**bidirectional** approximation (all `proj=0`, none at pos-33),
  which OPT-3's bidirectional refinement + OPT-4 must close to reach the ≥55 quality bar.

### Scale anchors used below (all UNVERIFIED for GPU wall-clock)

- **Vocab V = 248,320** — `vllm/transformers_utils/configs/qwen3_5.py:44`.
  **CORRECTION:** three of the four reports assumed V≈152K; the `kernels` report
  is the correct one. The `overhead` micro-bench was measured at V=152,064, so
  its host numbers are re-scaled up by ~1.6× below (this makes the dominant host
  cost *worse*, not better).
- **Canvas / block CL**: tested config CL=32 (the brief's "33"); current default
  `_DEFAULT_BLOCK = _GDN_CHUNK = 64` (`qwen3_5_flare.py:113-117`). Larger CL makes
  the width-waste findings proportionally worse.
- **F = model-chosen (non-forced) forwards/turn ≈ 50–57** (`stats.forwards`;
  reference HF row = 56.83 denoise fwd/turn). Grammar is active for *value*
  tokens too (`hybrid_clean.py:1004,1015-1026`), so most of F pays the
  grammar-active host cost.
- `grammar_topk = 256` (`hybrid_clean.py:34,613`).

---

## Consolidated root-cause map (dedupe of the four reports)

The four reports converge on **six** distinct optimizations. Cross-references
show every report finding folds into one of them.

| Opt | Root cause | Reports that found it |
|---|---|---|
| **OPT-1** | Full-vocab host-side sampling: `.tolist()` D2H + Python `_argmax` + Python full-vocab `sorted()` per model token | `graphs` F2, `cache-batch` F2, `overhead` F1+F2, `kernels` F1 |
| **OPT-2** | CUDA-graph dispatch is *eligible* for the FULL decode graph but capture is fragile/unverified; a config mismatch silently drops every step to eager | `graphs` F1 |
| **OPT-3** | Fixed `canvas_length`-wide forward: CL query rows computed, **1** probe logit read → prefill-class kernels + the GAP-5A byte-parity blocker | `cache-batch` F1, `overhead` F4, `kernels` F2, (`graphs` F1 width invariant) |
| **OPT-4** | GDN forward takes the prefill `chunk_gated_delta_rule` path (never the single-token `fused_recurrent` decode kernel); read-only-denoise then discards the state write | `kernels` F3 (mechanism unlocked once OPT-3 lands) |
| **OPT-5** | O(n²) detok/grammar: `grammar.text(committed)` re-decodes the whole committed region several times/token + O(n) Python re-scans | `cache-batch` F3, `overhead` F3 |
| **OPT-6** | Beyond-AR lever: multi-token bulk commits amortizing forwards (extend zero-forward forced commits to model-chosen high-confidence spans) | task brief; enabled by `bulk_commit_forced` infra |

Two throughput observations fold into the above: the **per-row `for i, slot`
decode loop is serial across the batch** (`qwen3_5_flare.py:1644`) — O(N) host
cost, folds into OPT-1 as a batched-topk requirement (call it OPT-1b); and the
`logits.new_zeros(num_decode, CL, V)` full-canvas logit materialization
(`qwen3_5_flare.py:1502-1517`) is a symptom of OPT-3.

## Ranking — expected ms/turn payoff per unit effort

| Rank | Opt | Expected gain (per turn) | Effort | Payoff/effort | Tier | Status |
|---|---|---|---|---|---|---|
| 1 | **OPT-1** GPU-native sampling | **~1.3–2.3 s** (host, MEASURED-scaled) → measured **2.36× A/B**, byte-identical | M | **Highest** | P0 | **DONE + verified** |
| 2 | **OPT-2** graph guard + `cg_mode` log | avoids contingent **+0.2–0.6 s** eager fallback; enables honest measurement | **S** | Very high | P0 | **landed** (no eager fallback seen; gain not triggered) |
| 3 | **OPT-3** variable single-[MASK] width (windowed-**bidirectional**) | byte-parity refinement — bidir probe LANDED (`b7d76e2`): 52→**58/63** parity, valid 62→**63/63** (gt19 fixed); residual **5/63** breaks are OPT-4-Part-1 (align+variable-width) territory | M–L | High | P0 | **BIDIR LANDED** (`e5496cc`); last 5 breaks fold into OPT-4 Part 1 |
| 4 | **OPT-5** incremental detok/FSM | ~~~15–75 ms~~ **DISPROVEN: grammar = 0.7% of turn** | L | ~~Medium~~ | ~~P1~~ | **DO NOT DO** |
| 5 | **OPT-4** incremental KV/GDN 1-wide decode | **Part 2 (PIECEWISE cudagraph) DONE** — per-forward 29→18.56 ms (1.615×), 1.681→**1.051 s/turn** (BEATS M2/guided-AR/HF); **Part 1 OPEN** = `fused_recurrent` + 32-absolute align + variable commit width (closes last 5 parity breaks AND the residual speed gap to stock-agg 0.741) | L | High | P1 | **PART 2 DONE** (`e5496cc`); **PART 1 = single next lever** |
| 6 | **OPT-6** multi-token bulk commits | fewer forwards than AR → sub-0.741 s | M–L | Beyond-AR | P2 | OPEN |

Numbers are UNVERIFIED for GPU components; the host components of OPT-1/OPT-5 are
MEASURED-CPU and re-scaled. The ranking is robust to the constants: OPT-1's host
cost alone exceeds the entire 1.120 s budget, and OPT-3 is a hard prerequisite
for *any* engine number, so those two dominate regardless.

---

# P0 — Needed to hit < 1.120 s/turn (M2)

Without all three of these there is either (a) no valid turn to measure (OPT-3),
(b) a host cost that alone blows the budget (OPT-1), or (c) an unguarded eager
fallback that silently blows it (OPT-2).

## P0-A · OPT-1 — Move full-vocab sampling onto the GPU  ✅ DONE + VERIFIED

> **DONE (vLLM pin `58cfe2c`, bench §0.D).** Landed as GPU-native batched top-k sampling. Integrity A/B vs a
> checked-out pre-OPT-1 `hybrid_clean.py` (`6b81154`): engine output **byte-identical on every turn** (incl.
> the 9 divergent ones) at **2.36× mean speedup** — a pure, behavior-preserving speedup with **zero** parity
> change. On the identical completed subset the engine is **2.27× under HF** (1.250 vs 2.835 s/turn). This
> removed the host-sampling wall as designed; the remaining blocker to a full-battery number is OPT-3, not
> sampling. Artifacts: `runs/p2_engine_bench/{ab_opt1.py,ab_A_opt1.json,ab_B_preopt1.json}`.

**Evidence.** Per model-chosen token, `decode_model_token` runs the entire
reduction in CPython over the full 248,320-wide vocab:
- `hybrid_clean.py:994` `logits = _as_float_list(logits)` → `:836-842` calls
  `probe.tolist()` (a blocking **GPU→CPU sync** materializing ~248k Python
  floats) then re-iterates `[float(x) for x in ...]`.
- `:998-1010` `[MASK]`/stop suppression indexes that full Python list.
- `:1027` `raw_top = _argmax(logits)` → `:845-852` a 248,320-iteration Python
  loop.
- `:1032` `ranked = _topk_indices(logits, grammar.grammar_topk)` → `:855-857`
  `sorted(range(248320), key=lambda i: values[i], reverse=True)[:256]` — a
  **full-vocab Python sort** on essentially every grammar-active token (value
  tokens are grammar-active, `:1004,1015-1026`).
- Driven by the **serial** per-row loop `qwen3_5_flare.py:1644`
  `for i, slot in enumerate(decode_slots_np.tolist())` → `decode_probe` →
  `decode_model_token`, so the cost is **O(N) across the decode batch**.

**MEASURED (CPython, `overhead` report, V=152,064):** `_as_float_list` 1.58 ms,
`_argmax` 1.39 ms, `_topk_indices` **21.1 ms** per call. Re-scaled to
V=248,320: ~2.6 / ~2.3 / **~35 ms** → **~40 ms per grammar-active model token**
(plus the D2H sync, UNVERIFIED +1–3 ms), ~5 ms per non-grammar token.

**AR comparison.** vLLM AR samples the whole batch on-GPU (fused argmax/top-k)
and `.item()`s only the single chosen id; constrained AR (xgrammar) applies a
precompiled bitmask on-device in C++ (~10–40 µs/token). AR pays sub-ms; this
pays tens of ms and stalls the GPU on the D2H sync.

**Fix.** Compute the reduction on-device before crossing to Python:
`vals, idx = probe.topk(grammar.grammar_topk)` over the **batched** probe rows,
scatter `-inf` for `mask_token_id` / premature-stop ids on-GPU *before* the
top-k, and pass only the ~256 `(id, logit)` candidate pairs plus the on-GPU
`raw_top` into the FSM. Change the `decode_model_token` / `legal_top_token`
contract to consume a top-k view instead of a `list[float]`; the policy stays
CPU-testable but never sees the full vocab. **Batch the top-k across all decode
rows (OPT-1b)** so the per-row Python loop no longer serializes the reduction —
collapses the O(N) host cost of the `for i, slot` loop.

**Effort:** M (one contract change; policy internals unchanged).

**Expected gain:** **~1.3–2.3 s/turn** removed from the host critical path at
F≈50–57 (host component MEASURED-scaled; the exact fraction grammar-active is
UNVERIFIED). This single item is the difference between "host loop alone exceeds
the 1.120 s budget" and not. Residual host cost after the fix: << 1 ms/token.

**GPU-verification step.** Wrap `decode_model_token` (or the whole per-row loop)
in a `torch.cuda.Event` / `time.perf_counter` counter accumulated into `stats`;
log ms/turn spent in sampling before vs after. Confirm (a) no `.tolist()` on the
[V] tensor remains in an `nsys`/`torch.profiler` trace (no full-vocab D2H), and
(b) sampling ms/turn drops from ~seconds to <10 ms on a single matched-20 turn.

## P0-B · OPT-2 — Guarantee (and instrument) the FULL decode CUDA graph

**Evidence.** The FLARE decode forward is *designed* to replay in the same FULL
decode CUDA graph as AR spec-decode — a positive result:
`decode_query_len = num_speculative_steps + num_new_sampled_tokens_per_step`
(`model_runner.py:318`), FLARE sets `num_new_sampled_tokens_per_step = 0`
(`qwen3_5_flare.py:228`) and `num_speculative_tokens = diffusion_config.canvas_length`,
so `decode_query_len == CL`; GDN is `AttentionCGSupport.UNIFORM_BATCH`
(`gdn_attn.py:83`), so `FULL_AND_PIECEWISE` survives (FULL for uniform decode).
**But** the dispatch silently falls to eager (`cg_mode = NONE`) if the per-step
`uniform_token_count` drifts off `CL`. The concrete trap (`graphs` F1 #2): the
scheduler zeroes the bonus token only if `model_config.is_diffusion`, which is
true only if **`hf_config.canvas_length`** is set (`scheduler.py:119-122`;
`config/model.py:1540-1542`), while `decode_query_len` reads
**`diffusion_config.canvas_length`** — a *different* field. If a deployment sets
one but not the other, every decode step runs `CL+1`-wide against a `CL`-captured
graph → **eager every step**.

**AR comparison.** Same dispatch path as AR uniform/spec decode; no FLARE-specific
penalty *if the width invariant holds*.

**Fix.** (a) Add a one-line per-step counter of `batch_desc.cg_mode` around
`execute_model` (`model_runner.py:1154`) so the decode loop's FULL-vs-eager
status is *measured*, not assumed. (b) Hard-assert at startup, fail-closed:
`model_config.is_diffusion == (num_new_sampled_tokens_per_step == 0)` and
`hf_config.canvas_length == diffusion_config.canvas_length`.

**Effort:** S.

**Expected gain:** avoids a contingent **+0.2–0.6 s/turn** (a 27B model running
eager launches hundreds of kernels/forward at ~5–15 ms overhead × F forwards).
Even when the graph already holds, the instrumentation is the prerequisite that
makes every other item's GPU-verification honest. UNVERIFIED whether it is
currently tripping — that is exactly what this item measures.

**GPU-verification step.** Run one matched-20 turn; assert the new counter shows
`cg_mode == FULL` on ≥99% of decode steps. Deliberately mis-set
`hf_config.canvas_length` and confirm the startup assert fires (fail-closed test).

## P0-C · OPT-3 — Variable single-[MASK] forward width (also the GAP-5A fix)  ◑ CORE LANDED (sync scheduler, `d2fccab`); full-63 battery completes; M2/K3 MISSED; bidirectional refinement open

> **BATTERY UPDATE (2026-07-04, full-63 on `d2fccab`, build-status §0.E): the sync-scheduler fix WORKS — the
> full 63-turn battery runs end-to-end for the FIRST time.** Zero stalls (ex-stalls gt4/gt24/gt50 finish),
> async-rollback divergence@33 gone (**0/11 breaks at pos-33**; gt12/16/18/20 byte-parity again). This produced
> the first honest full-battery wall-clock and made **M2/K3 adjudicable — MISSED on both axes** (speed 1.681 >
> 1.120, quality 48 < 55; **52/63** byte-parity so the 63/63 promotable gate is NOT met). **Residual (the
> bidirectional refinement, still OPT-3 territory):** 11/63 turns diverge via the windowed-**causal**
> approximation of the reference's windowed-**bidirectional** read (all `proj=0`, `first_div` scattered
> {17,19,19,19,26,31,34,38,41,47,53}, **none at pos-33** — NOT the fixed async/stall bug). The engine is quality
> **≥ HF in aggregate** (exact 48 vs 47, +1) but not byte-identical. **The speed bar is now OPT-4's job** (see
> P1-A): the forward runs the prefill `chunk_gated_delta_rule` GDN path with `fused_recurrent` absent and no
> CUDA graph (~18 ms GPU + ~11 ms host/forward, profiled). Battery commit `61d1381`.
>
> **PRIOR UPDATE (2026-07-04, pin `d2fccab`): both named defect classes were RE-DIAGNOSED to the ASYNC scheduler
> (not the windowed-probe read) and FIXED by forcing the SYNC scheduler.** A per-step harness
> (`runs/p2_engine_bench/diag_opt3.py`) replaying the reproducers showed: (A) the divergence@33 is the async
> scheduler *rolling back a committed forward* (observed `committed 6→5` then re-decode) and re-running the
> stateful hybrid_clean decoder at the boundary — a nondeterministic corruption, NOT the causal read; and
> (B) the stall is the async `num_output_placeholders` accounting drifting (the canvas is a READ window, not
> candidate output tokens) → seq_len rollback `1542→1523` + width truncation `valid_len 32→13` at committed~96
> → the next forward hangs. **Fix (`vllm/config/vllm.py`, guarded by `diffusion_config is not None`): propagate
> `diffusion_config.canvas_length` → `hf_config.canvas_length` so `is_diffusion`/`check_for_draft_tokens` is True
> (else the sync path publishes no canvas draft and stalls post-prefill), and force `async_scheduling=False`.**
> GPU-validated on a fresh boot: **gt4 (ep1/t0, 110-tok STALL) 110/110 byte-parity 2.8s**; **gt24 (ep7/t2 STALL)
> completes 2.5s**; **gt16/gt18/gt20 (the `first_div=33` divergences) full byte-parity restored**;
> gt0/gt7 acceptance turns byte-parity (no regression); **CPU suite 70 passed**; two fresh boots byte-identical
> (determinism). Zero committed-state rollbacks, constant `valid_len=32`, sampler step-time flat 0.3ms (no
> boundary spike). **Residual (SEPARATE issue, follow-up):** gt24 still shows a value-region 2-token insertion
> mid-block-1 (gen_idx 34) — the causal windowed-probe is an approximation of the reference's
> windowed-**bidirectional** read; the first-after-boundary token matches, so this is the remaining
> "windowed-bidirectional" refinement + the true per-request variable draft width (forward-compute cut), the
> **batched-rollout follow-up** (sync loses cross-request batch overlap; acceptable for the batch=1 eval regime).

> **Bench §0.D promoted this to the single frontier item and proved it is a CORRECTNESS + LIVENESS gate, not
> merely efficiency.** The §0.C causal-windowed *approximation* of the reference's windowed-**bidirectional**
> read fails two ways on the full battery, both **pre-OPT-1** (proven by the OPT-1 A/B): (1) **byte-parity
> diverges** on 9/44 completed turns (all `proj=0`), systematically at the first denoise position after a block
> boundary (`first_div=33` recurs); and (2) when the staged canvas `valid_len` drops below the full block width
> (measured **32→13 at committed ≈95**), a single denoise forward **STALLS indefinitely (>10 min)** — making
> **19/63 turns uncompletable**. A byte-EXACT windowed-**bidirectional** variable-width single-`[MASK]` forward
> fixes **both at once** (restores universal parity → HF 47/63 **and** unblocks long turns → a real full-battery
> s/turn). This is the precondition for adjudicating M2/K3 at all.

**Evidence.** The scheduler pins a **uniform** `num_spec_tokens == canvas_length`
draft width every step (`scheduler.py:986-989` `[-1]*self.num_spec_tokens`;
`config/vllm.py:516-518`), and `DraftTokensHandler.set_draft_tokens` publishes
the full `draft_tokens.shape[1] == CL` row (`spec_decode/utils.py:25`,
`model_runner.py:1479-1487`) regardless of the sampler's computed
`_hc_draft_len = tail_len+1` (`qwen3_5_flare.py:1568`). So every denoise forward
runs QKV/MLP/proj over **CL** query positions and reads exactly **one** probe
logit (`read_pos = staged-1`, `:1687-1697`); `_gather_block_logits` even
allocates `logits.new_zeros(num_decode, CL, V)` (`:1502-1517`). The
causal-windowed probe (`:626-648`) only fixes *which keys* the one probe
attends to — the other CL−1 query rows are still computed and discarded. Per the
build-status doc this is *also* the open GAP-5A byte-parity blocker: at pos-12
the fixed-32 canvas makes the probe `[MASK]` attend to ~20 trailing `[MASK]`s,
producing wrong value logits, so **no valid turn can be driven** today.

**AR comparison.** AR decode = `max_seqlen_q = 1` per request (KV cached),
hitting the decode tiling; FLARE denoise = `max_seqlen_q = CL`, prefill tiling,
with CL−1/CL of the output thrown away. Within a block of B model tokens this is
B² position-forwards vs AR's B.

**Fix.** Schedule a **per-request variable draft width** = `_hc_draft_len`
(already computed) instead of the uniform `num_spec_tokens`, so each probe
forward is exactly `[committed tail + one MASK]`. This is a scheduler /
model-runner / `DraftTokensHandler` change — the first-party comment at
`:626-648` states the async scheduler "cannot express" a variable per-request
spec width today, so it needs either a diffusion-specific (non-async) schedule
path or a variable-spec-width scheduler contract. (Cheap interim lever: shrink
`VLLM_QWEN3_5_FLARE_BLOCK`/`canvas_length` toward real tail lengths to bound the
waste — limited because it is `_GDN_CHUNK`=64-aligned.)

**Effort:** M–L (scheduler width plumbing).

**Expected gain:** (1) **Unblocks M2 at all** — turns the byte-parity gate from
FAIL to adjudicable, which is the precondition for producing *any* engine
s/turn. (2) Cuts wasted forward compute by ~CL/avg-tail ≈ **2–6×** on the
attention/MLP/GDN token-work (UNVERIFIED, GPU/model-size dependent; at low batch
partly hidden by weight-bandwidth-bound decode). This is the structural crux and
must land before OPT-4 (which builds on the 1-wide read).

**GPU-verification step.** Log `max_query_len` per denoise step (expect it to
drop from CL to `tail_len+1`) and re-run the Step-5 byte-parity harness
(`scripts/parity_audit_flare_engine.py`): confirm engine==HF token-for-token past
pos-12 with `projected_value_tokens_exact == 0`. Then capture per-forward kernel
time (nsys) before/after to quantify the width cut.

---

# P1 — Parity with stock-AR (0.741 s/turn)

Once P0 removes the host-sampling wall and produces valid 1-wide reads, the
remaining gap to AR is the residual GPU forward shape and the O(n²) host grammar.

## P1-A · OPT-4 — Incremental KV+GDN commit → true 1-token decode kernel  ◑ PART 2 (PIECEWISE cudagraph) DONE; PART 1 open

> **PART 2 DONE (vLLM pin `e5496cc`, `VLLM_FLARE_CUDAGRAPH=1`, build-status §0.F): PIECEWISE CUDA graph landed and
> confirmed live** (`enforce_eager=False`, `cudagraph_mode=PIECEWISE`, 3756 dispatches). Per-forward **29→18.56 ms
> (1.615×)**, dropping full-63 s/turn **1.681→1.051** — the engine now BEATS the M2 speed bar (1.120), guided-AR
> (1.213) and HF (3.904) for the first time at ≥HF quality, still above stock-AR-agg 0.741. **PART 1 remains the single
> next lever** and is now *coupled to parity*: routing the 1-token denoise GDN forward to `fused_recurrent` +
> **32-absolute commit alignment** (`VLLM_FLARE_ALIGN_BLOCKS`, scaffold only) + **per-request variable commit width**
> closes BOTH the last **5/63** APC-on byte-parity breaks ({20,21,44,45,60}) AND the residual forward-compute gap to the
> stock-AR aggregate. **v3 sharpened the residual (fresh-context certificate, build-status §0.G):** the invariant
> structural residual is the 2-turn set **{44,45}** (breaks in BOTH cache paths — gt44 fd16 variable-width, gt45 fd20
> 32-absolute align); {20,21,60} are APC-only cross-turn prefix-cache artifacts (gt60 "engine wins" is an APC artifact,
> not a real gain — fresh it copies HF's mistake ⇒ exact 0 = hf). So OPT-4 Part 1 = variable commit width + 32-absolute
> align closes {44,45} AND cuts per-forward **18.66→13.09 ms** to stock-agg 0.741 (weight-stream floor 10.5 ms; 2.59 ms
> above floor, REACHABLE). Parity closure and the last speed cut land together.

**Evidence.** With no active spec mask, GDN metadata build calls
`split_decodes_and_prefills(m, decode_threshold=1)` (`gdn_attn.py:213`): any row
with `query_len > 1` is classed **prefill** and dispatched to
`chunk_gated_delta_rule` (`qwen_gdn_linear_attn.py`), never the single-token
`fused_recurrent` decode kernel. Because read-only-denoise then restores GDN
state (M1 discipline), the chunk kernel's state write is **discarded** — pure
waste. Even after OPT-3 makes the read 1-wide, reaching the recurrent decode
kernel and the FULL CUDA graph on a *genuine* 1-token query requires folding
each committed clean token's KV+GDN state into the running block incrementally.

**AR comparison.** AR decode hits `fused_recurrent` (GDN) / the 1-token decode
tiling (full-attn) and replays the FULL CUDA graph. This is the parity target.

**Fix.** Append the committed tail's KV + GDN state incrementally within a block
so the next probe is a genuine 1-token decode, leaving only the trailing `[MASK]`
read-only. Must be reconciled with the read-only-denoise "advance GDN by 0"
discipline and bidirectional-denoise semantics (the `kernels`/`cache-batch`
reports flag this compatibility as UNVERIFIED).

**Effort:** L.

**Expected gain:** closes the residual GPU-side gap between an OPT-3 1-wide
forward and AR decode — moves GDN off the prefill chunk kernel and guarantees
the FULL decode graph. Magnitude UNVERIFIED; this is the item that takes the
forward from "1-wide but prefill-classed" to "AR-identical decode step."

**GPU-verification step.** Assert (nsys/trace) that GDN dispatches
`fused_recurrent` (not `chunk_gated_delta_rule`) on denoise rows, and that
`cg_mode == FULL` (via OPT-2's counter). Re-run the Step-4 read-only-denoise
bit-identity check to confirm state discipline is preserved. Compare per-forward
GPU ms to the AR decode step.

## P1-B · OPT-5 — Incremental detokenization + resumable FSM cursor  ⛔ DO NOT DO (DISPROVEN)

> **DISPROVEN by the bench §0.D — do NOT do.** The predicted O(n²) grammar cost is not real on the engine: a
> per-step trace measured **cumulative grammar time = 0.017 s = 0.7%** of turn time (steps to committed 95 are a
> flat 27 ms). The §0.C ">9 min / O(committed²) grammar" hypothesis is **false** — that pathology was actually
> the partial-canvas forward **STALL** (OPT-3 territory), misattributed to grammar. There is no host-grammar
> bottleneck to remove; effort here would be wasted. The evidence below is retained for provenance only.

**Evidence.** `grammar.text(committed)` = `tokenizer.decode(list(committed))`
over the **entire** generated region (`hybrid_clean.py:621-622`), called at
`:1001` (per model token), every iteration of `bulk_commit_forced`'s growing
prefix (`:963`), and per-candidate in `_keeps_prefix` (`:638-641`,
re-decoding `committed+[cand]` for each schema candidate). On top, the native
predicates re-scan the full text: `qwen_native_inside_parameter_value` runs 2
`re.finditer` over the whole string, `native_tool_prefix_completable` scans
char-by-char. Per value token ≈ 4 full-prefix decodes + several O(n) scans → the
prefix grows through the turn → **O(n²)/turn**.

**AR comparison.** vLLM uses an incremental detokenizer (decode only the new
suffix, cached string) and xgrammar advances a C++ FSM by exactly one token
(O(1) amortized) — no re-decode, no re-scan.

**Fix.** Keep an incremental decoded-text cache (append only the new token's
piece; never re-decode the prefix) and a resumable FSM cursor that advances from
the last position instead of re-scanning from index 0.

**Effort:** L.

**Expected gain:** **~15–75 ms/turn** at F≈50, super-linear in turn length
(O(n²)→O(n)); UNVERIFIED absolute (tokenizer-dependent). Secondary to OPT-1, but
becomes the next host bottleneck once OPT-1 lands — required for true AR parity
because AR's detok is O(1) amortized.

**GPU-verification step.** (Host-bound; CPU-measurable but verify on the engine.)
Add a per-turn timer over `grammar.text`/predicate calls; confirm total detok
ms/turn is flat vs turn length (not growing) after the change, on a long
matched-20 turn.

---

# P2 — Beyond-AR wins (forwards-saved)

## P2-A · OPT-6 — Confidence-based multi-token bulk commits amortizing forwards

**Evidence / lever.** AR is fundamentally 1 token per forward. The diffusion
path already commits **grammar-forced** tokens with **zero forwards**
(`bulk_commit_forced` / `bulk_forced_prefix`, `qwen3_5_flare.py:1678-1683`,
tracked by `hc_zero_forward_rows`) — the structural half of the thesis. The
beyond-AR extension is to commit **>1 model-chosen token per forward** on
high-agreement canvas spans: when the block-parallel denoise distribution over
the trailing canvas positions is confident/stable, accept K positions from a
single forward instead of re-probing each. This amortizes the forward over K
committed tokens, pushing forwards/turn **below** AR's ~49 tok/turn at equal
quality — the whole point of the project's `forwards-saved` KPI.

**Fix.** Add a confidence/stability gate on the already-materialized block logits
(`_gather_block_logits` produces `[num_decode, CL, V]` — the full-canvas
distribution is *already computed* today, currently discarded) to accept a run
of high-confidence positions per forward, staged through the existing
commit path. Must preserve byte-parity semantics (accept only where the
sequential decode would have produced the same token) to keep the M2 quality
gate intact — this is the same acceptance discipline as speculative decoding,
applied within the diffusion block.

**Effort:** M–L.

**Expected gain:** **< 0.741 s/turn** at equal quality — the beyond-AR bar.
Magnitude scales with the average accepted run length K (the reference already
sees ~half the tokens as zero-forward forced; extending to model-chosen spans is
additive). UNVERIFIED; gated on OPT-1 (so the per-token host cost does not
dominate) and OPT-3 (so the forward is 1-wide-correct as the acceptance
baseline).

**GPU-verification step.** Report `stats.forwards / tokens_committed` per turn
(the forwards-saved ratio) and confirm it drops below AR's 1.0 while the Step-5
byte-parity / `exact_args` quality gate stays at PARITY. Measure end-to-end
s/turn on the matched-20 battery vs the 0.741 bar.

---

# Recommended execution order

Ordered by unblock-value first, then payoff-per-effort. OPT-1 is CPU-testable and
can be *developed* in parallel from day one; OPT-3 is the gate that makes any of
it *measurable* on a real turn.

1. ~~**OPT-2 (P0-B, S)**~~ — **DONE.** `cg_mode` counter + fail-closed config assert landed; no eager
   fallback observed in the bench.
2. ~~**OPT-1 (P0-A, M)**~~ — **DONE + verified** (GPU-native batched sampling, incl. OPT-1b). Measured
   **2.36× A/B**, byte-identical; engine 2.27× under HF on the completed subset. It removed the host-sampling
   wall as designed — but because the battery can't complete (OPT-3 stall), the **M2-met checkpoint is NOT yet
   reached**: the honest number is 1.250 s/turn on a short-turn subset only.
3. ~~**OPT-3 (P0-C, M–L)**~~ — **BIDIR PROBE LANDED** (`e5496cc`, `VLLM_FLARE_BIDIR_PROBE=1`): the reference-exact
   windowed-**bidirectional** read replaced §0.E's causal approximation → byte-parity **52→58/63**, valid **62→63/63**
   (gt19 fixed). Residual: the last **5/63** breaks ({20,21,44,45,60}) are no longer a standalone OPT-3 item — they
   fold into **OPT-4 Part 1** (32-absolute align + variable commit width).
4. ~~**OPT-5 (P1-B, L)**~~ — **DROPPED.** Grammar is 0.7% of turn time (measured); no host-grammar bottleneck
   exists. Do not do.
5. **OPT-4 (P1-A, L) — PART 2 DONE, PART 1 = single next lever.** Part 2 (PIECEWISE cudagraph, `e5496cc`): per-forward
   29→18.56 ms (1.615×), **1.681→1.051 s/turn** — BEATS M2/guided-AR/HF for the first time. **Part 1 (NEXT):** route
   the 1-token denoise GDN forward to `fused_recurrent` + 32-absolute commit alignment + per-request variable commit
   width — **closes the last 5 parity breaks (→ 63/63, the promotion gate) AND the residual speed gap.**
   **Expected checkpoint: 63/63 byte-parity + 1.051 → toward ~0.741 s/turn (AR parity).**
6. **OPT-6 (P2-A, M–L)** — confidence-based multi-token bulk commits: amortize forwards below AR.
   **Expected checkpoint: < 0.741 s/turn (beyond-AR).**

**One-line rationale (updated):** OPT-2 + OPT-1 + OPT-3(sync+bidir) + OPT-4-Part-2(cudagraph) are **done** → the
FINAL engine BEATS M2/guided-AR/HF on speed (1.051 s/turn) at ≥HF quality but is **NOT PROMOTED** (58/63 byte-parity ≠
63/63; exact 48 < 55) → the single remaining lever is **OPT-4 Part 1** (`fused_recurrent` + 32-absolute align +
variable commit width), which closes BOTH the last 5 parity breaks (→ promotion) AND the residual speed gap to
stock-agg 0.741 → then amortize forwards (OPT-6) to beat AR. **OPT-5 is dropped (grammar 0.7%).**
