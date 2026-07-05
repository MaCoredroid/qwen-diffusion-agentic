# P2 Engine-Fast Diffusion Serving — Build Status & GPU Smoke Checklist

Workflow follow-on to `p2_serving_reuse_plan.md` (the reuse decision, milestones, kill criteria).
Date: 2026-07-04 (last verified 2026-07-05). Author: build+review sweep + real-export gauntlet + post-wiring acceptance +
IMA-fix / sequential-decode-rebuild acceptance + **GAP-5A forward-view fix acceptance (§0.C)**.

> **UPDATE (2026-07-05 — LOSSLESS-APC GATE BATTERY, live at HEAD `9cb5e7a`, RTX 5090, RAM cage, 5 GPU boots ~15 min; battery commit `8b98aaf`; artifacts `runs/lossless_apc/gates2/`). CACHE-ON CERTIFICATE STATUS: quality-lossless CERTIFIED (via Stage-3), strict-bitwise NOT (seam inert).**
> Two distinct findings. **(A) The canonical-publish seam (W1+W2, commit `9cb5e7a`) is STILL LIVE-INERT.** Runtime callpath
> assertion re-run at HEAD: `canonical_publish_calls`/`canonical_apply_calls` = **0** on the gt4 2048-crosser, on all 63
> matched-20 turns, and on all 184 never-train turns. gate-ON == gate-OFF **byte-identical (0/63 differing fields)**. The 3
> named wiring gaps stand (chunked-prefill guard; commit routes through `_forward_core_decode_non_spec` where there is no
> hook; `<2048` census never crosses a 1024 window). It changes **zero serving bits** → **NOT promoted, default stays OFF,
> not pushed on the vLLM pin.**
> **(B) SEPARATELY — the real news — the post-Stage-3 shipped engine is ALREADY effectively cache-lossless on the QUALITY
> axis, without the seam.** The certificate the battery targeted (cache-on quality == fresh) is **satisfied, inherited from
> Stage-3 (`95d8b47`), not produced by the inert seam.**
> **The lossless-APC numbers (measured live):**
> - **Gate 2 — matched-20 full-63 cache-on cert:** byte-parity **62/63** vs HF (lone break gt44, fd=16, path-invariant
>   fp-residue that breaks fresh too), **exact_args EXACTLY 47/63 == HF per-turn (63/63, 0 wins / 0 losses)**, episode 13/20,
>   valid 63/63, proj 0/63. Reproduces `parity_cert_freshboot.jsonl` → **cache-on == fresh MET.** gate-ON vs gate-OFF eager
>   = 0 diffs; eager vs shipped cudagraph = 0 diffs.
> - **Gate 3 — never-train full-184:** warm cache-on byte-parity **175/184**, exact **83/184 == HF**, proj 0/184. Losslessness
>   measure (engine WARM cache-on vs COLD fresh-proxy, same kernel): **174/184 byte-identical, exact 83==83 (ZERO
>   quality-affecting turns)**; the 10 differing turns are all fp-ε near-tie flips. Warm cache-on is *closer* to the warm HF
>   ref than cold (9 vs 15 breaks).
> - **Gate 5 — refold overhead:** seam inert → **0 live overhead** (0 crossings fired). Isolated primitive cost **0.374
>   ms/layer × 24 = ~9.0 ms per 1024-crossing**; gate-ON vs gate-OFF wall +2.79% (eager jitter, publish=0). **Net APC speedup
>   unchanged from the 1.23× lossy number** — the lossless number cannot be measured live until the seam fires; if wired,
>   ~9.0 ms/crossing amortizes to ~0 for the `<2048`-token turns that dominate agentic serving (crossings are rare).
> - **Gate 4 — no-regression (gate-OFF):** PASS. CPU 59/59; gate-ON==gate-OFF==shipped-cudagraph byte-identical; proj=0
>   everywhere; determinism holds at seed 20260701. WALL-RULE considered, NOT triggered (every residual explained by the
>   fold-grid / near-tie fp-residue mechanism; proj=0, verify ok, exact APC-invariant; no unexplained accumulating divergence).
>
> **Verdict.** exact_args is byte-stable and fully APC-invariant across warm/cold/gate-on/off/eager/cudagraph (matched-20
> 47/63, never-train 83/184). **cache-on quality == fresh is effectively already TRUE via Stage-3.** The **remaining gap is
> strict BITWISE losslessness on near-tie tokens** — which the inert seam targets but a *chunked* kernel cannot fully close
> anyway (rootcause Refinement 1: needs the sequential-recurrent republish design). **The Stage-3 shipped path stands and is
> the current cache-lossless-on-quality engine.** Next task fires only if strict bitwise losslessness is required beyond
> quality-lossless: complete W1+W2 (hook `_forward_core_decode_non_spec`, drop the chunked-prefill guard).
>
> **UPDATE (2026-07-05 — L0 DONE + L2 MEASURED; FINAL-HEAD CONSOLIDATED VERIFICATION on vLLM pin `0b44dcc`; certificate NOT regressed).**
> The **L0 free-text serving fix landed on the pin as `0b44dcc`** ("fix CPU-pathological hang + honor EOS like AR"). This
> re-verification re-ran the full no-regression set on that FINAL HEAD (hybrid_clean, PIECEWISE cudagraph,
> `VLLM_FLARE_BIDIR_PROBE=1`+`VLLM_FLARE_CUDAGRAPH=1`, bf16, RTX 5090, B=1, greedy seed 20260701). **Zero source edits made**
> → byte-parity certificate safe. Artifacts: `runs/l0l2_final_head_verify/` (summary.json + 5 JSONL).
>
> **(1) Free-text 30-prompt sweep (L0 gate) — FIXED.** GSM8K clean first-30, schema-free free-text: **30/30 clean stop
> (finish_reason=stop), 0 hang, 0 degenerate, single boot / 0 reboots** (pre-fix pin `95d8b47` was 24/30 clean + 3
> CPU-pathological hang + 3 degenerate-length). **GSM8K strict now 26/30** (L1 pre-fix was 0/24 — the fix restores scorable
> free-CoT instead of an unstoppable phantom tool-wrap). `value_projection_events=0`, verify.ok on all 30, cudagraph engaged
> (cg/fwd≈1.04). But this path is **K=1** (model_chosen==forwards) and **per-forward 25.8 ms** — see (4).
>
> **(2) 15-turn tool-call parity sweep (spot-gate) — NOT REGRESSED.** Turns {0-9,20,21,44,45,60}: **byte-parity 14/15**,
> the **lone break gt44 (fd=16, n=101)** — the same known path-invariant deterministic fp-residue as v3b/endgame; gt45 and
> the APC-sensitive {20,21,60} all parity-clean. `value_projection_events=0` on all 15, `eng_exact==hf` on all 15,
> verify.ok on all 15. The L0 fix is gated on `grammar.enabled` / "finished-but-no-stop-id", so tool-call decode is
> byte-identical to the pre-fix engine — **the 233/247 certificate is intact.**
>
> **(3) Read-only-denoise fingerprint — CLEAN.** 6 turns, per-forward hash of conv+ssm denoise rows (cap 4/turn):
> **6/6 bit-identical (post-restore == pre-forward), forward_wrote>0 on all (restore load-bearing), 0 leaks.** GDN state
> discipline holds. **Determinism 2×:** two independent fresh boots (battery15_B vs battery15_C) of the 15-turn sweep are
> **identical on every per-turn field** (n_gen / first_div / byte_parity / forwards / exact).
>
> **(4) Per-forward before/after + the corrected 5× factor.** Tool-call content clean per-forward **18.31 ms (fwd-weighted),
> 17.92 ms long-turn-amortized** — statistically identical to endgame **18.5** / L2 re-measure **17.8** ⇒ **L0 is
> per-forward-NEUTRAL on the tool-call path**, as designed. NEW: the now-working **free-text/reasoning per-forward = 25.8 ms**
> (cudagraph-engaged, K=1, 0.86 *emitted* tok/fwd because block-diffusion overshoots past EOS and discards the tail). This
> is the honest reasoning-content per-forward (pre-fix it hung, so was unmeasurable). **Corrected 5× equation on reasoning
> content** (ratio = committed tok/fwd × AR-ms/tok ÷ engine-ms/fwd, AR-cudagraph fairness per L2): **0.86 × (10.72 ÷ 25.8) =
> 0.36× vs AR-cudagraph** (0.47× vs AR-eager). model-chosen K=1.00 is the chain-rule wall; distance to 5× ≈ **14×**, entirely
> in K (L2 per-forward parity 25.8→~13 is at most ~2× and still K-bound). Only L3 (S2 consistency-distillation + adaptive-K
> training) can move it. Open L2 note: free-text 25.8 ms > tool-call 18.3 ms even though both engage cudagraph — pure
> block-diffusion free-gen re-denoises the full canvas per committed token with no FSM to bulk-commit/prune.
>
> **UPDATE (§0.H, vLLM pin `95d8b47` — POST-FIX PROMOTION ATTEMPT: NOT PROMOTED, 1 turn short).** OPT-4 Part 1 has now
> **landed** (Stage 1 32-absolute variable commit width + Stage 2 scheduler width plumbing + Stage 3 byte-robust bidir key
> window; **code default OFF**). v3b is an **independent fresh boot of the post-fix engine** — the real promotion attempt.
> **Measured (APC-on, full-63): byte-parity 62/63** (lone break gt44), **exact_args EXACTLY 47** (0 turns eng≠hf), episode
> **13/20**, valid **63/63**, verify **63/63**, projection **0/63**. The strict gate is **63/63 ⇒ exact exactly 47**;
> parity is 62/63 so ⇒ **NOT PROMOTED, default stays OFF.** The Stage-3 fix took byte-parity **58/63 → 62/63** and drove
> exact to **exactly 47** (pre-fix "won" gt60 at 48; post-fix it byte-matches HF incl. HF's mistake → 47), clearing
> {20,21,45,60} **path-robustly** (clean in both APC-on and cold-prefix fresh-boot), **no regressions**, shared-clean turns
> byte-identical to v3. **Lone break gt44** breaks **identically** under APC-on and per-turn fresh-boot (same fd16, n101)
> → a **path-invariant deterministic fp-residue, NOT an APC class** (the documented APC protocol cannot rescue it); both
> engine and HF are non-exact on gt44 → **quality-neutral**. Root cause = the block#0 GDN fold-path fp gap (matching HF's
> fold granularity is kernel-level, deferred). **Timing: s/turn mean 1.053** (p50 0.896, p90 1.700, worst 4.241 gt50),
> 56.86 TRUE fwd/turn, per-forward **18.52 ms**, PIECEWISE cudagraph 63/63 turns, **3.708× under HF**. **Bars (1.053): HF
> 3.904 BEAT (0.270×) · guided-AR 1.213 BEAT (0.868×) · M2 1.120 BEAT (0.940×) · stock-agg 0.741 MISS (1.421×).** **THE
> v3b CORRECTION (supersedes §0.G/v3 "0.741 REACHABLE"):** OPT-4 Part 1 (variable commit width) landed but the Stage-3
> A/B measured it **speed-NEUTRAL** (18.52 vs 18.56 ms/fwd — cudagraph buckets narrow widths back to a captured bucket),
> so CL=32 width was **not** the residual. Measured per-forward 18.52 ms = **weight-stream floor 11.40 ms** (gemm device
> self-time, 63.5% of GPU; arithmetic 10.77 ms = 19.31 GB bf16 / 1.79 TB/s HBM, **irreducible at bs=1**) + non-weight GPU
> compute 6.54 ms (**not width-reducible**) + host 0.58 ms. Bar needs 13.03 ms/fwd → **not reachable by engine plumbing at
> bs=1**; levers are fewer forwards/turn (training), fp8/int8 weights (~0.68/0.51 s/turn, quality tradeoff), or batching.
> temp=0.7 (5 rollouts × 2 boots) byte-reproducible + never-train 3/3 byte-parity/exact vs HF — contract holds post-fix.
>
> **UPDATE (ENDGAME BATTERY COMPLETE — aggregate 247, engine row now ON the scoreboard).** The missing never-train
> slice (BFCL/API-Bank, 184 turns) ran on the **exact v3b config** (`runs/p2_engine_nevertrain/`, commit `1129f86`):
> exact **83/184 == HF EXACTLY**, valid 184/184, byte-parity 171/184 (13 breaks, all quality-neutral fp-residue, 0
> structural), **0.480 s/turn**, 24.06 fwd/turn. **Aggregate-247 ENGINE** (matched-63 v3b + never-train-184): exact
> **130/247 == HF hybrid-clean 130/247 EXACTLY**, episode **32/80 == HF**, valid **247/247**, byte-parity **233/247**
> (cold-config 235/247; 14 breaks all fp-residue, **0 structural**, proj==0, `eng_exact==hf_exact` ⇒ exact stays 130),
> **0.626 s/turn**, 32.43 fwd/turn. **Quality:** engine ties HF exactly AND beats every AR baseline (+6 vs
> stock-bf16-AR 124, +1 vs stock-FP8 129, +3 vs merged-AR 127). **Speed:** 0.626 s/turn **BEATS** stock-bf16-AR-agg
> 0.741 (1.18×), stock-FP8-agg 0.910 (1.45×), merged-AR-agg 0.739 (1.18×), HF-hybrid 2.577 (4.12×) — **closes the last
> slower column of the scoreboard AND the v3b "stock-agg 0.741 MISS"** (that MISS compared matched-20-only 1.053 to a
> shorter stock mix; on the identical 247-turn mix the engine aggregate wins). The **engine row is now added to
> `runs/endgame_scoreboard/report.md`** (all three slices) — this is a "same-system, faster-serving, quality-identical"
> row, NOT a promotion: the **strict 247/247 byte-parity gate remains NOT met (233/247) ⇒ code default stays OFF**.
> Final assembled table + verdict: `endgame_table_final.md` (repo root). Never-train commit `1129f86` (pushed
> origin/main).
>
> **PRIOR (matched-20 v3b): No engine row on the scoreboard at that point** (matched-20 gate not met). Details §0.H;
> battery commit `1cb4457` (pushed origin/main).
>
> **PRIOR UPDATE (§0.G, vLLM pin `e5496cc` — THE PROMOTION ATTEMPT: NOT PROMOTED).** The v3 battery is the explicit
> promotion attempt against the strict gate (**63/63 byte-parity ⇒ exact exactly 47**) plus an independent 3rd boot.
> The engine tree is **clean at `e5496cc` = byte-identical to v2** (OPT-4 Part 1 / Task #37 UNLANDED), so this is a
> faithful re-run: it **reproduces v2 exactly** (n_gen/fwd/parity/exact/first_div ALL identical) and adds a
> **fresh-context parity certificate**. **Measured (APC-on, the v2 protocol): 58/63 byte-parity** (breaks
> {20,21,44,45,60}), **exact_args 48/63** (+1 vs HF 47 = the gt60 APC win), episode **13/20**, valid **63/63**, verify
> **63/63**, projection **0/63**. Gate needs 63/63 ⇒ exact exactly 47; parity is 58/63 so exact is 48 ⇒ **NOT
> PROMOTED.** Timing reproduces v2: **s/turn mean 1.056** (p50 0.874, p90 1.734, worst 4.253), 56.62 TRUE fwd/turn,
> per-forward 18.66 ms. **Bars (1.056): HF 3.904 → BEAT (0.270×, 3.695×) · guided-AR 1.213 → BEAT (0.871×) · M2 1.120 →
> BEAT (0.943×) · stock-AR-agg 0.741 → MISS (1.425×).** **THE v3 FINDING — byte-parity is cache-path-dependent; the
> invariant structural residual is {44,45}.** A fresh-context certificate (cold prefix+mamba cache, fresh boot per
> turn; 57/63 measured, 6 pending under a concurrent Stage-3 GPU hold, all APC-parity; `enable_prefix_caching=False`
> and `reset_prefix_cache()` are documented negative controls, both invalid here) splits the breaks: **invariant in
> BOTH paths {44,45}** (genuine OPT-4 Part 1 — gt44 fd16 variable-width, gt45 fd20 32-absolute align), **APC-only
> {20,21,60}** (cross-turn prefix-cache artifacts, resolve fresh), **fresh-only {1,3,12,23,24,50,57}** (hidden by APC).
> gt60's "engine wins" is an **APC artifact**: fresh, it byte-matches HF's 169-tok output *incl. HF's mistake* →
> eng_exact 0 = hf, exactly the gate memo's prediction. So the "58/63" headline is cache-config-specific; the robust
> promotion blocker is the 2-turn structural set **{44,45}** — single coupled lever = **OPT-4 Part 1** (variable commit
> width + 32-absolute align), which also cuts per-forward **18.66→13.09 ms** to reach stock-agg 0.741 (weight-stream
> floor 10.5 ms; target 2.59 ms above floor, REACHABLE). temp=0.7 (5 rollouts × 2 boots) byte-reproducible +
> never-train 3/3 byte-parity/exact vs HF — contract holds. **No engine row added to the endgame scoreboard** (gate not
> met). Details §0.G; battery commit `55965de` (pushed origin/main).
>
> **PRIOR UPDATE (§0.F, vLLM pin `e5496cc` = FINAL engine: bidir-probe `b7d76e2` + PIECEWISE cudagraph `VLLM_FLARE_CUDAGRAPH=1`):
> the strongest promotable candidate yet — but the strict 63/63-byte-parity promotion gate is STILL NOT met, so NOT
> PROMOTED.** The FINAL engine turns BOTH landed levers on: the reference-exact windowed-**bidirectional** denoise read
> (`VLLM_FLARE_BIDIR_PROBE=1`) and the PIECEWISE CUDA graph (OPT-4 Part 2; confirmed live — `enforce_eager=False`,
> `cudagraph_mode=PIECEWISE`, 3756 PIECEWISE dispatches). Full-63, greedy, seed 20260701, uncapped, RAM cage, two boots.
> **Required checks: byte-parity/turn 58/63** (breaks {20,21,44,45,60}) — **NOT met**, so the by-construction chain that
> would force exact=47 does not hold and the run is diagnosed, not promoted. Aggregate quality is **≥ HF**: exact_args
> **48/63** (+1 vs HF 47 — engine WINS gt60, correct where HF is wrong), episode_exact **13/20** (met), valid **63/63**
> (bidir fixed gt19's non-stopping divergence — up from 62), `value_projection 0/63`, `verify_invariants 63/63`.
> **Timing: s/turn mean 1.051** (p50 0.876, p90 1.699, min 0.326, worst 4.248 gt50/259tok), **TRUE 56.62 denoise
> fwd/turn**, per-forward **18.56 ms** (eager ~29 → **1.615× cudagraph win**), **3.715× speedup vs HF**. **Bar
> adjudication (mean 1.051): HF 3.904 → BEAT (0.269×) · guided-AR 1.213 → BEAT (0.866×) · M2-speed 1.120 → BEAT
> (0.938×) · stock-AR-agg 0.741 → MISS (1.418×).** The eager engine (§0.E, 1.681) missed M2 and guided-AR; **cudagraph
> now clears both for the FIRST time on the honest full-63** — the engine sits *below* guided-AR and *below* the M2
> speed bar at ≥HF quality, though still above stock-AR-agg. **M2/K3 adjudication:** the M2 **speed** bar (<1.120) is
> now **MET** (1.051, first time) and **K3 speed MET**; but M2's **quality** axis (≥55/63, byte-identical) is **MISSED**
> (58/63 parity ≠ 63/63; exact 48 < 55), so the combined M2 promotion gate is NOT met. temp=0.7 (5 turns × 2 boots)
> **byte-reproducible** and collapses onto greedy — RL contract holds under cudagraph; never-train spot-check (BFCL-AST
> + API-Bank Lv1/Lv2, sha-verified prompts) **3/3 byte-parity/valid/exact, 0 projection** — not matched-20-specific.
> **Why 63/63 is unreachable here:** the bidir read is the *correct* reference semantics and cudagraph is
> **byte-neutral on the entire promotable set** (reproduces the bidir-eager anchor 58/63 + break-set exactly). The 5-turn
> parity residual + the single exact deviation are the **coupled, UNLANDED** work: 32-absolute commit alignment
> (`VLLM_FLARE_ALIGN_BLOCKS`, scaffold only) + per-request **variable commit width** — which is simultaneously **OPT-4
> Part 1** (the remaining forward-compute cut to stock-agg). Parity closure and the last speed cut land together. Details
> §0.F; battery commit `1acdf2e` (pushed origin/main).
>
> **PRIOR UPDATE (§0.E, vLLM pin `d2fccab` = OPT-3 sync-scheduler fix): the FIRST COMPLETE full-63 battery now
> exists — the engine is MEASURABLE end-to-end, but NOT PROMOTED.** The OPT-3 fix closes both §0.D blockers:
> all 63 turns complete, **zero stalls**, and the async-rollback divergence@33 is gone (**0/11 breaks at
> pos-33**; gt12/16/18/20 byte-parity again). This is the first honest, complete, uncapped full-battery
> wall-clock. **But the 63/63 byte-parity promotable gate is NOT met — measured 52/63** (11 divergences are
> the *separate, author-flagged* windowed-**causal** vs reference windowed-**bidirectional** approximation,
> all `proj=0`, `first_div` scattered {17,19,19,19,26,31,34,38,41,47,53}, **none at 33** — not the fixed
> async/stall bug). Aggregate quality is **≥ HF but not byte-identical**: exact_args **48/63** (+1 vs HF 47),
> episode_exact **13/20** (met), valid **62/63** (−1; the single invalid is gt19's non-stopping divergence),
> `value_projection=0`, `verify_invariants 63/63`. Timing: **s/turn mean 1.681** (p50 1.427, p90 2.724, worst
> 5.361 gt50), **TRUE 56.65 denoise fwd/turn** (HF 56.83), **2.32× under HF**. **M2/K3 are now ADJUDICABLE for
> the first time — and MISSED on both axes:** speed 1.681 > 1.120 (1.39× *slower* than guided-AR 1.213;
> §0.D's 1.250 was a short-turn subset), quality 48 < 55. temp=0.7 RL sanity holds (2 boots byte-reproducible,
> `proj=0`). **Single next lever = OPT-4** (incremental KV+GDN 1-token decode → `fused_recurrent` + FULL CUDA
> graph): profiler shows GDN on the prefill `chunk_gated_delta_rule` path (`fused_recurrent` absent), ~18 ms
> GPU + ~11 ms host per forward, `enforce_eager`. Details §0.E; battery commit `61d1381` (pushed to origin/main).
>
> **PRIOR UPDATE (§0.D, vLLM pin `58cfe2c` = GAP-5A windowed-probe + OPT-1 GPU-native sampling): the first
> honest matched-20 engine wall-clock now exists — OPT-1 is DONE and verified clean, but the full battery
> STILL CANNOT COMPLETE and M2/K3 remain unadjudicated.** OPT-1 (P0) landed: A/B vs pre-OPT-1 `6b81154` is
> **byte-identical on every turn** (incl. divergent ones) at **2.36× speedup** — a pure, behavior-preserving
> speedup. On 44 *completed* turns (a short-turn subset, mean 60 tok) the engine runs **1.250 s/turn mean**
> (p50 1.185), **2.27× under HF** on the identical subset — a real OPT-1 win that sits at the stock-AR-guided
> level. **But the battery is NOT complete: 19/63 turns are uncompletable** due to a **partial-canvas forward
> STALL** (a single denoise forward hangs >10 min, non-terminating, when staged `valid_len` drops below block
> width — measured 32→13 at committed ≈95). And **byte-parity is NOT universal**: 35/44 completed turns hold,
> **9 diverge** (all `proj=0`) from the GAP-5A causal approximation of the reference's windowed-*bidirectional*
> read. **Both blockers are pre-OPT-1 engine-forward defects (proven by the A/B), both are OPT-3 territory, both
> are correctness/liveness — not optimization regressions.** The **prior ">9 min O(committed²) grammar"
> hypothesis is DISPROVEN**: a per-step trace shows cumulative grammar = **0.017 s = 0.7%** of turn time, so
> **OPT-5 is confirmed a non-issue and should NOT be done**; the stall is the real liveness blocker. So: M2
> (<1.120) is **UNADJUDICATED** (can't complete the battery + parity not universal); K3 **UNADJUDICATED**;
> **OPT-3 is the single frontier item** (fixes both blockers at once). Details §0.D.
>
> **PRIOR UPDATE (§0.C, vLLM pin `6b81154`): the denoise FORWARD view was CLOSED and per-turn byte-parity
> PASSED on the 3 captured `gap5a_ref` turns.** The engine hybrid_clean decode byte-matched the HF Fast_dLLM
> reference token-for-token there (ep0 **42/42**, ep2 **36/36** full to `stop`; ep1 32/32 capped),
> `value_projection_events == 0`, CPU `70 passed`, 5B IMA clear. **CORRECTED by §0.D:** that pass held on the 3
> hand-captured turns but does **NOT generalize** to the full 63-turn matched-20 battery (9/44 completed turns
> diverge; 19 can't complete). Details §0.C; the pre-fix bottom-line below is retained for provenance.

**Bottom line (PRE-§0.C: engine NOT promoted — one blocker left, now isolated to the forward):** the entire M1
write-list (`Qwen3_5FlareModelState` + ops + hybrid-clean FSM reference + parity harness + flywheel
serving surface) is implemented, CPU-tested, and committed. The gauntlet ran on the RTX 5090 (sm_120):
b1000 smoke export (§0), the REAL diffusion-trained export (§0.R — `qwen3.5-9b-fastdllm-rlv2-vllm-bf16`),
a post-wiring acceptance after blockers A/B/C were wired (§0.A), and then — after the **decode-at-scale
IMA (5B) was fixed and the sequential single-`[MASK]` decode was rebuilt on the engine (5A driver)** — a
final acceptance re-run of Steps 5-6 (§0.B). **Steps 1-4 PASS** (R1/R2/K1/K2 killed; §0.R Step-4
read-only-denoise bit-identical, 0 leaks 10/10, restore load-bearing under vLLM pin `af21dc8`). **The M1
substrate + GDN-state-discipline crux are PROVEN, and 5B is now closed** — the engine decodes real-length
turns (1041/1443/917-tok prompts, 300-tok canvas generations) without faulting. **Step 5 byte-parity
still FAILs**, now for a **single, precisely-isolated** reason:
- **5B — decode-at-scale CUDA IMA: FIXED** (vLLM pin `1e32dcd`). Root cause: `super().postprocess_state`
  (the MambaHybrid align spec-decode state copy) read the accepted draft token's intermediate GDN state
  from block-table column `src_col + (num_accepted−1)`, which assumes `num_accepted−1` **speculative**
  checkpoint columns exist — allocated only under a real `speculative_config`. The FLARE path drives the
  canvas as spec draft tokens **without** one (`num_speculative_blocks == 0`), so a commit crossing a
  mamba-block boundary indexed `src_col + (A−1)` far past the width-8 table ⇒ IMA (localized to
  `num_computed=1272` canvas / `1140` hybrid_clean, not the "first decode step" the pre-fix run guessed).
  Fix: feed the align state machine a neutral `num_accepted == 1` (a FLARE commit's final GDN state
  already lives in the running block; the real commit count is retained only for the counter). Verified:
  canvas runs the full 300-token cap with zero IMA (was faulting ~231); 66/66 CPU tests green.
- **5A — turn byte-parity: still FAIL, algorithmic, now moved into the FORWARD.** The sequential
  single-`[MASK]` **driver** was rebuilt (vLLM pin `5e2fb53`) and is correct — with 5B fixed, engine and
  HF bridge match **token-for-token through the first 12 grammar-forced tokens on the same dual-loadable
  export**, proving the FSM wiring is live. But at pos-12 (the first logit-dependent choice) the engine's
  **forward** output is wrong: top-5 logits diverge whole-distribution (ref `=num`=24.25 argmax; engine
  argmax `>`=18.25, `=num` ~16 lower — not a bf16 near-tie, and identical at shifted/raw-MASK/last-clean
  probes ⇒ not a +1-shift bug). Cause: the engine forward still runs over the **fixed 32-position
  spec-draft canvas** (`num_draft_tokens_per_req == num_spec == 32`), so the probe `[MASK]` at the tail
  attends to ~20 trailing `[MASK]`s — still a block-parallel read of a mostly-masked block, not the
  reference's exact `[tail + 1 MASK]`. The block-parallel-vs-sequential gap was moved from the driver
  into the forward, not closed.

Consequently **Step 6 is NOT ADJUDICABLE at parity** (gated on Step 5, the first hard failure): with 5A
open the engine's value logits are wrong, the grammar never sees `complete_tool_call`, and requests
over-generate to `max_new_tokens`, so a matched-20 battery is quality-meaningless and infeasibly slow.
**No engine wall-clock at real quality exists.** The only honest diffusion wall-clock remains the HF
stack (3.904 s/turn, ≈3.5× the K3 1.120 s/turn target, ≈5.3× stock-AR aggregate) — **not the engine.**
K3 remains **unadjudicable on the engine path.** No sunk-cost engine number was invented. **Remaining fix
(5A) is a scheduler / model-runner change**, not a driver change: drive a **variable single-`[MASK]`
forward width** (schedule `draft_len` spec tokens, not a fixed 32) so each probe forward is exactly
`[tail + 1 MASK]`. Details §0.B; full checklist §3.

---

## 0. GPU SMOKE GAUNTLET — RESULTS (2026-07-03, RTX 5090 / sm_120)

**Env pre-flight:** host ~30 G RAM / ~25 G avail; GPU free (gnome-shell only). Every torch/vLLM
process run one-at-a-time inside `systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=4G`
(the RAM cage killed exactly one host-RAM probe, never the session). `torch 2.11.0+cu130`,
`get_device_capability()=(12,0)` sm_120 confirmed in every smoke JSON. Editable vLLM from
`/home/mark/shared/vllm_p2_pr42406`.

**HEAD note:** the vLLM flare branch has advanced past this doc's §1 pin (`6482e1d`). Current
`qwen3_5-flare-modelstate` HEAD is **`22f660c`** — concurrent agents added `cd3ba35`
(zero-value-projection live tripwire) and `22f660c` (the M1-crux GDN read-only-denoise fix). The
qwen_diffusion fix committed alongside this update is **`1c69101`** (chat-template instruct prompts +
non-JIT NvFp4 MoE backend override in `scripts/p2_vllm_smoke.py`); other agents' uncommitted Group-4
audit-counter and harness diffs are preserved, none clobbered.

### Per-step verdict

**Step 1 — pin-venv sanity: PASS.** Editable vLLM confirmed, sm_120 detected, `VLLM_USE_V2_MODEL_RUNNER`
honored. CPU suites under the real torch: flare state-machine **21 passed** (doc said 17; concurrent
regression tests added), hybrid_clean **23 passed** (doc said 20), parity `--mode selftest` PASS,
`--mode ops-parity` **18/18** checks PASS, `--mode state-parity` PASS. (Bypassed a missing `tblib`
test-infra dep with `--noconftest`; no product code touched.)

**Step 2 — DiffusionGemma smoke on sm_120: PASS** (`logs/smoke_diffusiongemma.json`, `status=PASS`).
Coherent output `"Fast inference minimizes latency to provide real-time responses."` → **the
first-party dLLM decode path (canvas draft / per-seq-causal Triton / commit) runs correctly on this
card. R2 / K2 NOT triggered.** Two blockers fixed inline: (a) FlashInfer NvFp4-**MoE** JIT needs a
CUDA toolkit → forced a non-JIT MoE backend (`emulation`, reference-correct; `marlin` also works);
(b) the smoke fed a **raw prompt to an instruct model** → chat-template fix (raw prompt gave gibberish,
chat template gave the coherent sentence). Both fixes are in commit `1c69101`.

**Step 3 — Qwen3.5-9B under MRV2 (default + align+APC): PASS both** (`logs/smoke_qwen_default.json`,
`logs/smoke_qwen_align_apc.json`, both `status=PASS`). Both configs load and generate coherently
(`"Thinking Process:\n\n1.  **Analyze the Request"`); the align config trips **no** mamba-cache
assertion. → **#38041 (MRV2×GDN broken) is stale; #42406 (align-APC) holds. K1 NOT triggered.** Two
toolchain blockers fixed: (a) nvcc/CTK header skew — the cu13 wheel ships **nvcc 13.2** but cudart
headers report `CUDA_VERSION 13000` and cccl's `cuda_toolkit.h` hard-errors → bypassed with the
header's own sanctioned escape `NVCC_APPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK` (13.0↔13.2
ABI-compatible); (b) the FlashInfer **sampler** JIT failed to *link* (`-lcudart`/`-lcuda` absent: wheel
has `lib` not `lib64`, no `.so` dev symlink, no driver stub) → disabled it with
`VLLM_USE_FLASHINFER_SAMPLER=0` (the dense Qwen GDN path needs no FlashInfer, and native argmax is
better for step-5 byte-parity anyway). **Required env for the Qwen path:**
`CUDA_HOME=<venv>/lib/python3.12/site-packages/nvidia/cu13`,
`NVCC_APPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK`, `VLLM_USE_FLASHINFER_SAMPLER=0`, plus the
doc's `VLLM_USE_V2_MODEL_RUNNER=1` / `VLLM_ATTENTION_BACKEND=TRITON_ATTN`.

**Step 4 — read-only-denoise probe (M1 go/no-go): NOT A CLEAN GO — blocked by the checkpoint and by an
under-scoped restore, not by the substrate.** Probe: `scratchpad/step4_readonly_denoise_probe.py` —
boots the real FLARE engine in-process, monkeypatches the GDN snapshot/restore ops, and per-row
float32-fingerprints the whole conv/ssm cache around every denoise forward. Final run: 10 denoise
forwards, batch=1 (`scratchpad/step4f.log`).
- **Engine blocker fixed to even boot:** the FLARE path publishes the canvas into the spec-decode
  `draft_tokens` buffer, whose width = `num_speculative_tokens` = `diffusion_config.canvas_length`.
  With **no `diffusion_config`** (stock export ships none) the buffer is width-0 →
  `_finish_prefills` crashes (`shape mismatch: value [64] vs index [1,0]`, `qwen3_5_flare.py:742`).
  Passing `diffusion_config={canvas_length,…}` fixes it. **This is a real launcher gap:**
  `qwen35_9b_flare_hybrid_serve.sh` sets `VLLM_QWEN3_5_FLARE_BLOCK` but does **not** set
  `num_speculative_tokens`, so the launcher as-written cannot boot the FLARE path.
- **Findings (10/10 denoise forwards):** the restore makes the protected block-table boundary slot
  (**row 1**) bit-identical — the mechanism works — but every denoise forward mutates GDN rows
  **[1,2,3,4]** while `_denoise_state_rows` protects only **[1]**; rows **[2,3,4]** persist changed
  after restore (`max_rowsum_diff_after_restore = 6038.2`, nonzero on all 10 forwards). The probe's own
  bit-identical pass criterion therefore reports **STEP4: FAIL**. This empirically confirms the §2 crux
  worry ("snapshot must protect the exact physical rows the GDN kernel writes"): the current
  row-selection is **too narrow**.
- **Decisive A/B:** with `VLLM_QWEN3_5_FLARE_READONLY_DENOISE` **on** (`step4f.log`) vs **off**
  (`step4_off.log`) the committed outputs are **near-identical** (`"AustrAustr腔 Special … owl绘本征
  Sarah"` either way) → the leaked rows [2,3,4] do **not** determine the committed output on this
  checkpoint.
- **Why it's checkpoint-blocked:** the on-disk `qwen3.5-9b-fastdllm-b1000-vllm-bf16` is a **stock,
  non-diffusion-trained** export — it decodes gibberish through the canvas/denoise path regardless of
  the readonly flag. Step 4 cannot be a real quality go/no-go, and step 5's byte-parity cannot be
  meaningful, until a diffusion-trained export exists (with `canvas_length` a multiple of FLA_CHUNK 64).
  **Verdict: the sm_120 substrate is validated and the boundary-row protection mechanism works, but the
  restore's row scope must be widened to [1..4] and re-proven on a trained export before M1 can be
  called.** Not a K1/K3 kill — a fix-and-retest, gated on producing the trained checkpoint.

**Step 5 — turn byte-parity (engine vs HF): NOT RUN.** Gated behind a clean step-4 go per the §3
"do not proceed until the current step passes" rule. No `--mode turn` run was attempted; the R5
shifted-logit-capture divergence remains unmeasured.

**Step 6 — matched-20 battery / engine-vs-HF wall-clock: NOT RUN.** Consequently **there is no honest
engine-vs-HF wall-clock number** — the thesis KPI (s/turn, forwards-saved ratio) is still unmeasured on
the engine path. The only timings captured are single-prompt smoke latencies (e.g. Qwen default
`generate_seconds ≈ 0.51`, align+APC ≈ `0.48`), which are load/warm-up-dominated and **not** a valid
AR-vs-diffusion comparison.

### Remaining issues surfaced by the gauntlet (in addition to §2)
1. **Launcher cannot boot the FLARE path** — `qwen35_9b_flare_hybrid_serve.sh` must also set
   `num_speculative_tokens`/pass a `diffusion_config` with `canvas_length`, or `_finish_prefills`
   crashes at width-0. Real gap, blocks any real serve.
2. **Restore row scope is under-sized** — `_denoise_state_rows` protects only the boundary slot; the
   GDN kernel writes rows [1..4]. Widen the snapshot to the full written set and re-run the probe. (A/B
   says non-determinative on the stock export, but that is not evidence on a trained one.)
3. **No diffusion-trained vLLM export on disk** — the stock export decodes gibberish through the
   diffusion path, blocking steps 4-6 quality/parity/wall-clock gates. Produce a trained
   canvas_length-multiple-of-64 export before re-attempting M1. **→ RESOLVED in §0.R** (real export
   built); and **corrected**: the trained block is **32**, not a multiple of 64 — see §0.R.
4. **Toolchain workarounds must be baked into the launcher** — the cu13 CTK-skew flag and
   `VLLM_USE_FLASHINFER_SAMPLER=0` are required for the Qwen GDN path to build/run on this box.

---

## 0.R REAL-CHECKPOINT GAUNTLET — steps 4-6 re-run on the trained export (2026-07-03, RTX 5090)

The §0 gauntlet ran on the b1000 **smoke** export (stock, non-diffusion-trained → gibberish through the
canvas path). Steps 4-6 were re-run on the **real diffusion-trained export** to decide M1 for real.

### Export + block-config reconciliation (the pre-gauntlet decisions)
- **Real export produced:** `models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16` (19G, new dir; b1000 left
  untouched as tokenizer/AR-parity reference). It is the **RL-v2 adapter merged into init-materialized
  weights** (`W += (α/r)·B@A`, r16/α32/scale 2.0), mathematically identical to the promoted HF
  hybrid-clean eval's PEFT runtime application (`--base init --adapter …rl…v2/…step300 --no-merge`).
  The RL-v2 adapter continued FROM Run-1 so it **subsumes** Run-1; B@1000 (r8, attn-only) is a separate
  AR-parity lineage, not in the hybrid-clean delta. Export: `replacement_count=427`,
  `lora_merge_count=152` (24 GDN layers × 5 + 8 attn × 4 — **GDN in_proj/out_proj deltas merged too**,
  not just attention). Sanity gates PASS: (a) merge bit-exact vs `init + 2.0·(B@A)` (maxabs 0.0, incl.
  GDN); (b) one HF hybrid-clean episode is **coherent, not gibberish** — `exact_args=3/4`, valid/schema/
  sequence 4/4, right on the promoted 47/63 ≈ 74.6% rate. Full provenance/shas in the export's
  `conversion_manifest.json` (`real_diffusion_export_block_reconcile.md`).
- **Block size pinned `canvas_length = 32`** to match training (`bd_size=32`) and the winning HF eval
  (`set_block_size(model, 32)`) end-to-end. Engine default 64 would denoise two trained blocks per
  commit and break parity. **Stale-doc correction:** §0 said "engine default `_DEFAULT_BLOCK=32`" — the
  source is actually `_DEFAULT_BLOCK = _GDN_CHUNK = 64`. So `32 % 64 != 0` trips the engine's mid-chunk
  hazard (fp32 `chunk_states[:,-1]` boundary is a *partial* recurrent state) — an engine-side
  restore-scope concern to prove at the step-4 re-run, **not** a reason to change the trained block.
- **Launcher gap fixed:** a single `--diffusion-config '{"canvas_length":32,"max_denoising_steps":8}'`
  both sets the engine block AND (via `num_speculative_tokens` fallback) sizes the spec-decode
  `draft_tokens` buffer, fixing the width-0 `_finish_prefills` crash. `qwen35_9b_flare_hybrid_serve.sh`
  now defaults to the real export and emits the config only when the FLARE gate is on.

### Per-step verdict (real export)
| step | verdict | one-line |
|---|---|---|
| 4 — read-only-denoise probe (M1 go/no-go) | **PASS (core invariant)** | after widening restore scope, whole GDN cache **bit-identical** around every denoise forward — 0 leaks, 10/10, restore load-bearing. |
| 5 — turn byte-parity (engine vs HF) | **BLOCKED / FAIL** | 3 structural blockers persist on the real export (orphaned FSM / format mismatch / unwired turn-adapter). |
| 6 — matched-20 M2 battery / wall-clock | **BLOCKED / FAIL** | `run_turn` raises `EngineUnavailable`; canvas sampler off-paradigm + non-deterministic. **No engine-vs-HF wall-clock.** K3 unadjudicable on the engine. |

**Step 4 — PASS.** Pre-fix (committed `22f660c`, "protect slot [1] only") reproduced the leak on the
real export: `protected [1]` vs `kernel-written [1,2,3,4]`, rows [2,3,4] still changed 10/10,
`max_rowsum_diff = 5872.18`. Root cause (GPU-measured): this ModelState runs `num_spec == 0`, so the
pre-forward block-table view names only checkpoint slot [1] while the align **running-state** row
advances 1→2→3 across denoise sub-steps. **Fix (vLLM pin `af21dc8`):** snapshot in `prepare_attn`
(post-metadata, pre-forward) reading the ACTUAL `non_spec ∪ spec` state indices, protecting
`{non_spec ∪ spec} ∪ {block-table checkpoint slots}` widened by a per-anchor guard band
(`VLLM_QWEN3_5_FLARE_READONLY_BAND`, default 4). Re-run: `leak forwards = 0`, `max_rowsum_diff = 0.0`,
`changed [1,2,3,4] → [] (protected [0..5])`, `STEP4_READONLY_ON: PASS`. Live counters sane:
`read_calls=10 advance_calls=1 read_advance_ratio=10.0 residual_full_context_model_calls=0
block_size=32 route_verified=False`.
- **Caveat — A/B on/off NOT identical, and all outputs gibberish.** Committed-token sha differs across
  readonly OFF (`aa08de30…`) vs [1]-only (`f1a3a298…`) vs banded (`37509cc7…`) → the read-only scope IS
  load-bearing (changes committed tokens). But every output is gibberish because the **engine's served
  sampler is a canvas/random-renoise denoiser, a different algorithm than the trained `hybrid_clean`**.
  So "which scope is semantically correct" needs the HF `hybrid_clean` parity reference — which is
  blocked (step 5). Core M1 artifact (whole-cache bit-identical read-only denoise) PASSES; the
  A/B-on-meaningful-output check is N/A until the engine runs the trained decode paradigm.

**Step 5 — BLOCKED / FAIL** (three code-verified blockers, first found on the smoke export in
`p2_engine_parity_smoke_result.md`, **re-verified on the real export**):
- **A — engine has no hybrid_clean path.** `VLLM_QWEN3_5_FLARE=1` routes to `Qwen3_5FlareSampler`, a
  canvas/renoise **block** denoiser (random init → Gumbel sample → entropy accept / random renoise →
  commit whole block); the FSM (`vllm/v1/sample/hybrid_clean.py`) is imported by nothing on the serving
  path. It ignores `SamplingParams.temperature` and is **not even self-reproducible** — two identical
  greedy requests gave entirely different sequences (`greedy_deterministic: FALSE`). The reference is
  single-token greedy + grammar FSM with a `[MASK]` sentinel. Cannot byte-match by construction.
- **B — "same checkpoint" is undefined.** The vLLM export is stock
  `Qwen3_5ForConditionalGeneration` / `model_type=qwen3_5`, `auto_map=None`, `mask_token_id=None`,
  has `vision_config`. HF hybrid_clean needs the `Fast_dLLM_Qwen3_5` bridge + `mask_token_id` (248077)
  → **cannot load this export**; reference decode is undefined on it. A stock vLLM export always lacks
  the bridge/mask token, so this persists independent of training.
- **C — harness turn-adapter unwired.** `VllmFlareEngineAdapter.run_turn` locates the class then raises
  `EngineUnavailable` — no code path drives one shared turn and reads token-ids + block-boundary
  snapshots out of the engine.

**Step 6 — BLOCKED / FAIL, no engine wall-clock.** Requires the missing seam (blocker C); `run_turn`
raises `EngineUnavailable` and the canvas sampler is non-deterministic + off-paradigm, so `exact_args`,
`episode_exact`, TRUE forwards/turn and s/turn **cannot be produced on the engine**. The only honest
engine signal is the read/advance ratio ~10 (fewer-forwards mechanism live — substrate liveness, not
the KPI). Reference rows (matched-20, `runs/endgame_scoreboard`, NOT the engine):

| row | exact_args | episode_exact | valid | s/turn | fwd-or-tok/turn |
|---|---:|---:|---:|---:|---:|
| OUR HF hybrid-clean (v2) | 47/63 | 13/20 | 63/63 | **3.904** | 56.83 denoise fwd/turn |
| stock-bf16-AR-guided | 51/63 | 14/20 | 63/63 | **1.213** | 82.24 tok/turn |
| stock-AR aggregate | 124/247 | 33/80 | 247/247 | **0.741** | 49.06 tok/turn |

The only diffusion wall-clock is the **HF-stack** 3.904 s/turn (~3.5× the K3 1.120 target, ~5.3× stock
AR) — **not the engine**. **K3 cannot be adjudicated on the engine path** until steps 5-6 unblock. No
sunk-cost engine number was invented.

### What remains before the engine can be promoted (steps 5-6 need net-new engineering, not a re-run)
1. **Wire `hybrid_clean` onto the engine** — integrate `HybridCleanDecodePolicy` into
   `Qwen3_5FlareSampler`, or expose a forward-only logit seam so the shared `sample_hybrid_clean`
   driver sources +1-shifted block logits from the engine. Only then are both sides the same algorithm.
2. **Produce one dual-format checkpoint** byte-equal as both the `Fast_dLLM_Qwen3_5` bridge (with
   `mask_token_id`) and the vLLM `Qwen3_5ForConditionalGeneration` export.
3. **Implement the harness seam** `VllmFlareEngineAdapter.run_turn` + `snapshot_from_vllm_modelstate`
   so `--mode turn --engine vllm` drives one shared turn and emits the byte / state-snapshot report.
4. Then: matched-20 turn byte-parity (greedy, identical FSM, `projected_value_tokens_exact==0`) and the
   M2 wall-clock A/B vs guided-AR **re-baselined on the same pinned build** (R6 fairness).

### Artifacts (real gauntlet)
- vLLM pin `af21dc8` (branch `qwen3_5-flare-modelstate`) — read-only restore widening + guard band.
- `p2_engine_gauntlet_real_result.md` — steps 4-6 on the real export (this section's source).
- `real_diffusion_export_block_reconcile.md` — export build + `canvas_length=32` decision.
- `p2_engine_parity_smoke_result.md` — the smoke-export step-5 structural-blocker analysis (re-verified).
- `runs/p2_engine_gauntlet_real/` — `step4_real_probe.py`, `step4_real_on_default.json` (PASS, 0 leaks),
  `step4_real_on.json` (pre-fix FAIL), `step4_real_off.json`, `step4_measure/instrument.json`.

---

## 0.A POST-WIRING ACCEPTANCE — Steps 5-6 re-run after blockers A/B/C wired (2026-07-04, RTX 5090)

The §0.R gauntlet FAILed Steps 5-6 on three *unwired* structural blockers. Those three were then wired
and this acceptance re-ran Steps 5-6 against the wiring, on the real export
`models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16` (block/canvas 32), engine venv `.venv-vllm-p2-main`, each
GPU proc alone in the `systemd-run … MemoryMax=22G` cage. Source: `p2_engine_acceptance_result.md`.

### Wiring done (the three blockers)
- **Blocker A** (vLLM pin `qwen3_5-flare-modelstate` `e38a9ea`): `hybrid_clean` is now a **selectable
  engine decode mode** (`VLLM_QWEN3_5_FLARE_DECODE=hybrid_clean`) driving the FSM/greedy
  `HybridCleanBlockDecoder` — the previously-orphaned FSM is now invoked at both the block-decoder and
  sampler seams (the assertion the orphaned-FSM bug lacked).
- **Blocker B+C** (qwen_diffusion `ed479b3`): one **dual-loadable checkpoint** (HF-bridge loader over the
  vLLM export, so both the `Fast_dLLM_Qwen3_5` bridge and the vLLM export read the same weights) + the
  `VllmFlareEngineAdapter.run_turn` seam (the adapter now boots the real export and drives a short turn
  instead of raising `EngineUnavailable`).

### Verification (CPU wiring intact — PASS)
`pytest tests/v1/sample/test_hybrid_clean.py tests/v1/sample/test_hybrid_clean_flare_decode.py
tests/v1/worker/gpu/test_qwen3_5_flare_state_machine.py` → **61 passed** (23 hybrid_clean + 17
hybrid_clean_flare_decode + 21 flare state-machine). Editable vLLM confirmed; the `af21dc8`
read-only-denoise fix untouched; the `hybrid_clean` FSM now actually invoked on the decode path.

### Per-step verdict (post-wiring)
| step | verdict | one-line |
|---|---|---|
| pre — CPU wiring intact | **PASS** | 61/61 CPU tests green; no regression to the `af21dc8` machinery. |
| 5 — turn byte-parity (engine hybrid_clean vs HF) | **FAIL (BLOCKED)** | TWO independent GPU-confirmed hard failures: **5A** block-parallel engine logits vs sequential single-`[MASK]` reference ⇒ byte-parity impossible by construction, engine emits gibberish while HF bridge on identical weights is coherent (`" Paris"`); **5B** deterministic CUDA IMA at the first decode step on real-length (1041-tok) turns ⇒ the 3 parity turns cannot even be driven. |
| 6 — matched-20 M2 battery on the engine | **BLOCKED** | Gated on Step 5. Byte-parity fails and real turns crash, so `exact_args` / `episode_exact` / TRUE forwards-per-turn / s-per-turn **cannot be produced on the engine.** No honest engine wall-clock. |

**5A — algorithm divergence (proven from both decoders' source).** HF `sample_hybrid_clean` (sha
`a4c66751…`) decodes **sequentially, one `[MASK]` at a time**: append one mask, forward over
`[committed_clean_prefix, MASK]`, read the single last-position shifted logit — one forward per
non-forced value token; truly-forced structural tokens (`len(legal)==1`) are FSM bulk-committed with
**zero** forwards. So the logit for output position *k* is conditioned on the actual committed clean
tokens 0..k-1. The engine `Qwen3_5FlareSampler._hybrid_clean_step` instead runs **one denoise forward
over the whole 32-position canvas** and `HybridCleanBlockDecoder.decode_block` walks all positions;
positions 1..31 are conditioned on the **noisy canvas**, not the clean prefix. ⇒ For any block with >1
model-decoded token the per-position logits differ by construction. Empirically: engine `hybrid_clean`
on a working short prompt → gibberish (`"<tool_call>\n<function= .ер s ET …"`,
`engine_smoke_adapter_short_hybrid_clean.json`) with its zero-value-projection tripwire holding; the
HF-bridge forward on the **same** export → coherent top-1 `" Paris"` (`blockerB_hf_bridge_forward.json`).
Same weights, tokenizer, mask id (248077) — the difference is the decode algorithm. **Fix scope: NOT
small / NOT engine-side-trivial** — the engine must run the reference's sequential single-`[MASK]`
schedule (or expose a forward-only logit seam feeding the shared `sample_hybrid_clean` driver), which
also removes the block-parallel "fewer-forwards" mechanism for value tokens.

**5B — deterministic decode-at-scale CUDA IMA.** Turn-0/episode-0 (1041 tokens) prefills fine
(`num_computed_tokens=1041`) then faults at the first decode (1 real + 32 canvas draft tokens):
`torch.AcceleratorError: CUDA error: an illegal memory access was encountered`. Isolated across GPU
boots (`CUDA_LAUNCH_BLOCKING=1`, RAM cage):

| decode mode | read-only-denoise | mamba_block_size | prompt | result |
|---|---|---|---|---|
| — (short smoke, prior) | on | 1024 | 10 tok | **OK** |
| hybrid_clean | ON | 1024 | 1041 tok | **CRASH (IMA)** |
| canvas | **OFF** | 1024 | 1041 tok | **CRASH (IMA)** |
| canvas | on | **4096** (1041 in ONE mamba block) | 1041 tok | **CRASH (IMA)** |

⇒ The IMA is independent of decode mode, of the read-only snapshot (rules out `af21dc8`), and of the
mamba-block-1024 boundary. It lives in the **shared FLARE canvas/commit spec-decode DECODE forward over
a long multi-KV-block prefix**. Exact faulting kernel needs `compute-sanitizer` (absent) or a
`TORCH_USE_CUDA_DSA` rebuild — deferred. Engine-side, deterministic, **NOT small**.

### Step 6 — no engine wall-clock; the only honest wall-clock is the HF/stock reference
Step 6 **did not run on the engine** (blocked by Step 5): byte-parity fails and real turns crash, so no
`exact_args`, `episode_exact`, TRUE forwards/turn or s/turn exist on the engine. The only engine signal
is short-prompt substrate liveness (`read_advance_ratio ≈ 3.0`, `forced_grammar_tokens=5` FSM
bulk-commit with zero forwards, `zero_forward_rows=2`, `projected_value_tokens_exact=0`) — i.e. the
fewer-forwards + zero-value-projection mechanisms are live, but over **gibberish**. That is liveness,
not the KPI. **No sunk-cost engine number was invented.** The honest wall-clock table below is the
matched-20 reference (`runs/endgame_scoreboard`, **NOT the engine**):

| row | exact_args | episode_exact | valid | s/turn | fwd-or-tok/turn |
|---|---:|---:|---:|---:|---:|
| OUR HF hybrid-clean (v2) — diffusion, **not the engine** | 47/63 | 13/20 | 63/63 | **3.904** | 56.83 denoise fwd/turn |
| stock-bf16-AR-guided (same build) | 51/63 | 14/20 | 63/63 | **1.213** | 82.24 tok/turn |
| stock-AR aggregate | 124/247 | 33/80 | 247/247 | **0.741** | 49.06 tok/turn |

The winning HF row's forward-savings (56.83 fwd/turn vs stock-AR ~82 tok/turn) come **entirely from the
grammar-FSM bulk-commit of truly-forced structural tokens with zero forwards** — every value token is
still decoded sequentially with one forward. So the engine's block-parallel canvas is a **different
algorithm**, not a faithful accelerator of the reference; on this checkpoint it is quality-dead. A
byte-parity-and-quality engine path must run the sequential single-`[MASK]` value decode; its only
legitimate speed lever over guided-AR is the same FSM zero-forward bulk-commit, not block-parallel value
denoising. This should be reflected in the M-milestone plan.

### What remains before Steps 5-6 can pass (net-new engineering, not a re-run)
1. **Close the logit seam (5A):** drive the engine with the reference's sequential single-`[MASK]`
   forward schedule, or expose a forward-only block-logit seam feeding the shared `sample_hybrid_clean`
   driver, so both sides run the same algorithm. Precondition for any byte-parity.
2. **Fix the decode-at-scale IMA (5B):** localize with `compute-sanitizer` / device-side asserts
   (mode-, readonly-, mamba-block-independent; long multi-KV-block prefix, first decode), then fix in
   the FLARE canvas/commit spec-decode forward.
3. Then: 3-turn byte-parity (greedy, identical FSM, `projected_value_tokens_exact==0` both sides), then
   the matched-20 M2 A/B vs guided-AR **re-baselined on the same pinned build** (R6 fairness).

### Artifacts (acceptance) — `runs/p2_engine_acceptance/`
- `step5_ima_hybrid_clean_ro_on_mamba1024.log`, `step5_ima_canvas_ro_off_mamba1024.log`,
  `step5_ima_canvas_ro_on_mamba4096.{log,json}` — the three IMA isolation boots.
- `step5_scheduler_dump_at_crash.txt` — faulting decode step (`num_computed_tokens=1041`).
- `ima_mamba_block_probe.py` — parametric IMA isolation probe.
- (prior, `runs/p2_engine_gauntlet_real/`) `engine_smoke_adapter_short_hybrid_clean.json` (engine
  gibberish + live counters), `blockerB_hf_bridge_forward.json` (HF-bridge coherent `" Paris"`).
- vLLM pin `e38a9ea` (blocker A), qwen_diffusion `ed479b3` (blockers B+C), acceptance commit `589a0dd`.

---

## 0.B IMA-FIX + SEQUENTIAL-DECODE-REBUILD ACCEPTANCE — Steps 5-6 (2026-07-04, RTX 5090)

The §0.A acceptance left Steps 5-6 FAIL for two *unclosed* reasons: 5A (block-parallel vs sequential)
and 5B (decode-at-scale IMA). Since then **5B was root-caused and FIXED** (vLLM pin `1e32dcd`) and **the
sequential single-`[MASK]` decode was rebuilt on the engine** (vLLM pin `5e2fb53`, GAP-5A driver). This
section is the final acceptance re-run of Steps 5-6 on those fixes, real export
`models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16` (block/canvas 32, mamba 1024, align+APC), each GPU proc alone
in the `systemd-run … MemoryMax=22G` cage. Source: `p2_engine_acceptance_result.md` (superseding update);
qwen_diffusion commit `237fdcf`.

### Per-step verdict (post-5B-fix + 5A-rebuild)
| gap / step | verdict | one-line |
|---|---|---|
| 5B — decode-at-scale CUDA IMA | **FIXED** | align spec-decode state copy indexed non-existent speculative block-table columns; feed the align state machine a neutral `num_accepted==1`. Real turns now decode without faulting. |
| 5A — turn byte-parity (engine hybrid_clean vs HF) | **FAIL — algorithmic (not numeric)** | driver rebuilt correct (12-token forced-prefix byte-match) but the engine **forward** still reads the fixed 32-position spec-draft canvas instead of `[tail + 1 MASK]`; the probe MASK attends ~20 trailing MASKs ⇒ forward logits diverge whole-distribution at the first real choice. |
| 6 — matched-20 battery at parity | **NOT ADJUDICABLE** | gated on Step 5. 5B robustness demonstrated (engine now runs real-length turns without crashing); with 5A open the engine over-generates on wrong values, so no quality / wall-clock at parity exists. |

### 5B — root cause + fix (GPU-localized, verified)
The IMA is **not** at the first decode step (the §0.A guess): canvas mode decodes ~231 tokens then faults
at `num_computed=1272`; hybrid_clean at `1140`. `_flare_bounds_check` on every named slot/block/GDN index
tensor **passes** — the OOB is *inside a kernel*. An env-gated phase synchronize
(`VLLM_FLARE_SYNC_DEBUG=1`) pinned the last clean phase before the fault to
**`postprocess pre-super (align state copy)`** — i.e. `super().postprocess_state` (the MambaHybrid align
spec-decode state copy), firing regardless of read-only-denoise. An align-kernel input dump made it exact:
```
A=32 N=1074 src_idx=2 bs=528 needs_copy=True token_bias=13 dest_col=1
  src+bias=15  bt_stride(width)=8      <-- gather col 15 into a width-8 block table
```
- **Mechanism:** `postprocess_mamba_fused_kernel`'s temporal copy reads the accepted draft token's
  intermediate GDN state from block-table column `src_col + (num_accepted−1)`. That assumes `num_accepted−1`
  **speculative** checkpoint columns exist — allocated only when a real `speculative_config` sets
  `num_speculative_blocks`. FLARE drives the canvas as spec draft tokens **without** a `speculative_config`
  (`num_speculative_blocks == 0`), so the mamba block table has no such columns; a commit of `A` tokens
  crossing a mamba-block boundary indexes `src_col + (A−1)` (2+13=15) past the width-8 table ⇒ IMA.
- **Fix (`1e32dcd`, `Qwen3_5FlareModelState.postprocess_state`):** a FLARE commit is a single causal pass
  whose final GDN state already lives in the running block — there are **no** per-token intermediate states
  to select. `num_computed_tokens` is already advanced by `post_update` (consuming the real `num_sampled`)
  *before* `postprocess_state`, so `num_sampled` here feeds ONLY the num_accepted scatter. Feed the align
  state machine a neutral `num_accepted == 1` ⇒ the boundary migration is a plain running-block copy
  (`token_bias == 0`, in-bounds); the real commit count is retained only for the commit counter.
- **Verified:** canvas decode runs the full 300-token cap with zero IMA (was faulting ~231); the only
  `needs_copy=True` is the clean prefill boundary at `token_bias=0`. **66/66 CPU tests green** (no
  regression to the `af21dc8` read-only-denoise machinery).

### 5A — byte-parity FAIL, diagnosed ALGORITHMIC (top-5 logits, both sides)
With 5B fixed and the sequential single-`[MASK]` **driver** rebuilt (`5e2fb53`), both sides run the SAME
dual-loadable export (HF `Fast_dLLM` bridge over the vLLM export, blocker B; mask id **248077** passed to
the engine via `VLLM_QWEN3_5_FLARE_MASK`), and the engine adapter now wires tool schemas + `grammar_topk`
to the engine FSM via `SamplingParams.extra_args`. Reference produces coherent bounded tool calls
(ep0/turn0 42 tok `stop=complete_tool_call`). **Turn-0 (greedy, identical prompt/schemas/mask):** engine
matches the reference **token-for-token for the first 12 tokens, then diverges at position 12** and
degenerates. Those 12 are the tool-call scaffolding + tool name — all **grammar-forced**, so the match
proves the FSM wiring is live, NOT that the forward is correct. Pos-12 is the first logit-dependent choice:
```
decoded:  ref  "<tool_call>\n<function=initialize_qubits>\n<parameter=num_qubits>\n2\n</parameter>\n<"
          eng  "<tool_call>\n<function=initialize_qubits>\n<parameter=num_qubits>\n\n\n00\n\n..."
```
| token | text | REFERENCE (HF bridge) | ENGINE (vLLM) |
|---|---|---:|---:|
| 29 | `>` | 17.625 | **18.25 (argmax)** |
| 45334 | `=num` | **24.25 (argmax)** | 8.625 |
| 28 | `=` | 19.75 | 9.625 |
| 2334 | `num` | 12.5 | 16.125 |

This is **not** a bf16 near-tie flip (whole distribution differs) and **not** a +1-shift/position bug
(the shifted, raw-MASK, and last-clean probe positions all give the same wrong logit) — **the engine's
forward output itself is wrong.**
- **Root cause (algorithmic):** the `5e2fb53` rebuild fixed the *driver* (reads one probe logit, drives
  the chain-rule schedule) but the engine *forward* still processes the **fixed 32-position spec-draft
  canvas**. The scheduler sets `num_draft_tokens_per_req = num_spec == 32` (no variable-width spec
  schedule), so every probe forward runs over `[clean tail, MASK, MASK×(31−tail_len)]`. The FLARE denoise
  read is bidirectional, so the probe `[MASK]` at position `tail_len` attends to ~20 trailing `[MASK]`s —
  still a partial **block-parallel** read of a mostly-masked block, not the reference's exact
  `[tail + 1 MASK]`. **The block-parallel-vs-sequential gap was moved from the driver into the forward,
  not closed.**
- **Fix needed (NOT a driver change):** drive the diffusion decode with a **variable single-`[MASK]`
  forward width** — schedule `draft_len` spec tokens, not a fixed 32 — so each probe forward is exactly
  `[tail + 1 MASK]`. This is a **scheduler / model-runner** change (dynamic per-step spec-token count for
  the diffusion path), plumbed through `num_draft_tokens_per_req` / `num_spec_tokens_to_schedule` (the
  same lever standard spec-decode `dynamic_sd_lookup` uses). Until then byte-parity is impossible by
  construction.

### Step 6 — not adjudicable at parity; 5B robustness demonstrated; honest wall-clock unchanged
Per "stop at first hard failure," Step 5 is the first hard failure, so the matched-20 quality/wall-clock
battery is **not run at parity**. What IS newly true post-5B: the engine decodes **real-length turns
(prompts 1041/1443/917 tok; 300-tok canvas generations) without crashing** — the substrate is live at
scale. But with 5A open the engine's value logits are wrong, the grammar never observes a
`complete_tool_call`, and the request over-generates to `max_new_tokens` (grammar cost grows with
`committed`), so a full 63-turn battery is both quality-meaningless AND infeasibly slow. **No sunk-cost
engine KPI was invented.** The only honest diffusion wall-clock remains the HF/stock reference below
(`runs/endgame_scoreboard`, **NOT the engine** — there is still no engine row):

| row | exact_args | episode_exact | valid | s/turn | fwd-or-tok/turn |
|---|---:|---:|---:|---:|---:|
| OUR HF hybrid-clean (v2) — diffusion, **not the engine** | 47/63 | 13/20 | 63/63 | **3.904** | 56.83 denoise fwd/turn |
| stock-bf16-AR-guided (same build) | 51/63 | 14/20 | 63/63 | **1.213** | 82.24 tok/turn |
| stock-AR aggregate | 124/247 | 33/80 | 247/247 | **0.741** | 49.06 tok/turn |

**K3 cannot be adjudicated on the engine path.** The HF row's forward-savings (56.83 fwd/turn vs stock-AR
~82 tok/turn) come **entirely from grammar-FSM bulk-commit of truly-forced structural tokens with zero
forwards** — every value token is still decoded sequentially, one forward each. So the engine's
block-parallel canvas is a **different algorithm**, not a faithful accelerator; its only legitimate speed
lever over guided-AR is the same FSM zero-forward bulk-commit, not block-parallel value denoising. Reflect
in the M-milestone plan.

### Artifacts (this acceptance) — `runs/p2_engine_acceptance/`
- `byte_parity_2proc.py` — two-process byte-parity driver (one 9B per process; reference vs engine on
  identical prompt/schemas/mask; token + byte + top-k-logit diff at first divergence).
- `ima_mamba_block_probe.py`, `step5_scheduler_dump_at_crash.txt`, and the three prior §0.A IMA isolation
  boots (now superseded by the 5B fix).
- vLLM pins: `5e2fb53` (sequential single-`[MASK]` driver, GAP 5A), `1e32dcd` (5B IMA fix +
  `VLLM_QWEN3_5_FLARE_MASK` + `VLLM_FLARE_SYNC_DEBUG`). qwen_diffusion `237fdcf` (adapter passes tool
  schemas via `extra_args`; this doc). Repro: engine env of §0.A + `VLLM_FLARE_SYNC_DEBUG=1` for the
  decode-fault phase, `VLLM_FLARE_BOUNDS_CHECK=1` for the (passing) index-tensor checks.

---

## 0.C GAP-5A FORWARD-VIEW FIX — byte-parity PASS (2026-07-04, RTX 5090 / sm_120)

§0.B left Step 5 byte-parity **FAIL** for one isolated reason: the engine denoise *forward* still read
the fixed 32-position spec-draft canvas, so the probe `[MASK]` attended to ~20 trailing MASKs and the
whole logit distribution diverged at the first model-chosen token (pos 12). That forward-view gap is now
**CLOSED** (vLLM pin `6b81154`), and **turn byte-parity PASSES**. Source: `p2_engine_acceptance_result.md`
(SUPERSEDING UPDATE #2); acceptance artifacts `runs/p2_engine_acceptance/p2_full_acceptance.{py,json}`,
`p2_temp_probe.{py,json}`, and the fix commit's `gap5a_windowed_ep0_default.json` / `gap5a_windowed_ep2.json`.

### The fix (attention-view, scheduler-independent)
The "variable single-`[MASK]` width from the scheduler" plan was inert on GPU: the spec-decode canvas is
a FIXED width (`num_speculative_tokens == canvas_length`) and the **async scheduler pins that width per
step** (uniform spec-token placeholder in `AsyncScheduler._update_after_schedule`), discarding
per-request narrow widths (measured `valid_len == 32` every step); disabling async deadlocks the
diffusion bootstrap. So the width can't be narrowed from the scheduler — that plumbing was reverted.
Instead, for hybrid_clean denoise rows the fix **forces a causal mask in `prepare_attn`**
(`VLLM_FLARE_WINDOWED_PROBE`, default on) so the probe attends only to the committed prefix + itself
(trailing canvas MASKs come strictly after it in causal order; GDN linear-attn is already causal), paired
with reading the `+1`-shifted logit at the **staged tail position** (`tail_len == _hc_draft_len-1`).
`VLLM_FLARE_WINDOWED_PROBE=0` restores the old broken read for A/B. Causal is an approximation of the
reference's windowed-*bidirectional* `[tail, MASK]` read; empirically byte-exact on the parity turns.

### Per-step verdict
| step | verdict | one-line |
|---|---|---|
| pre — CPU suite | **PASS** | `70 passed` (21 flare state-machine + 23 hybrid_clean + 26 hybrid_clean_flare_decode); no regression to `af21dc8` read-only-denoise or `1e32dcd` IMA. |
| 5A — turn byte-parity | **PASS** | ep0 **42/42** & ep2 **36/36** full to `stop`, ep1 32/32 (capped), all token-identical to the HF Fast_dLLM reference; `value_projection_events == 0`; `forwards == model_chosen`, `generated == fsm + model_chosen`; `residual_full_context_model_calls == 0`; 5B IMA regression clear (ep0 does the full 32-tok block commit at `num_computed≈1073` with zero fault). |
| 6 — matched-20 quality | **byte-parity-implied = HF 47/63** | same weights + same algorithm token-for-token ⇒ engine quality **is** the HF hybrid-clean row. No independent 63-turn engine sweep was run or fabricated. |
| 6 — engine s/turn (K3 speed) | **UNADJUDICATED — correctness fix, not a speed win** | model forward ~1.3–1.8 s per 36–42-tok turn, but end-to-end is dominated by the **shared grammar-FSM host cost** (`O(committed²)`; ep1's 110-tok turn > 9 min), the same cost the HF 3.904 s/turn carries ⇒ no engine speed advantage yet. |

### Step 5 — per-turn byte-parity (single boot, `p2_full_acceptance.json`; `VLLM_QWEN3_5_FLARE_DECODE=hybrid_clean`, windowed-probe on, `boot_s=11.5`)
Engine vs the pre-captured HF Fast_dLLM reference (`gap5a_ref.json`), greedy, identical prompt/schemas/mask (id 248077):

| turn | prompt | n_gen / n_ref | first_div | finish | denoise fwd | forwards==model_chosen | generated==fsm+model | value_proj | residual_full_ctx | wall_s |
|---|---:|---:|---:|---|---:|---|---|---:|---:|---:|
| ep0/t0 | 1041 | **42 / 42** | none | stop | 24 | 23==23 ✓ | 42==19+23 ✓ | **0** | 0 | 1.83 |
| ep1/t0 | 1443 | 32 / 110* | none | length* | 19 | 18==18 ✓ | 32==14+18 ✓ | **0** | 0 | 1.47 |
| ep2/t0 | 917 | **36 / 36** | none | stop | 17 | 16==16 ✓ | 36==20+16 ✓ | **0** | 0 | 1.33 |

`*` ep1 hard-capped at 32 output tokens: its grammar-FSM cost is `O(committed²)` and the tail (~tok 60–110)
is pathologically slow (a full ep1 turn exceeds **9 min of host time**). The 32 emitted tokens are
byte-identical to the reference — the cap is a wall-clock bound, not a divergence. ep0/ep2 run FULL to
`stop`; ep0's 42/42 reproduced across two independent boots. Fewer-forwards is live and correct (ep2:
36 tokens / 16 forwards, `tokens_per_forward=2.25`; 20 forced structural tokens bulk-committed with **zero**
forwards; every value token still costs exactly one single-`[MASK]` forward, `forwards==model_chosen`).

### Step 6 — wall-clock table (honest; NO fabricated engine number)
Byte-parity ⇒ the engine runs the *same algorithm* as the winning HF row, so its matched-20 quality **is**
47/63 (reported byte-parity-implied, not re-measured). On wall-clock the fix delivers **correctness, not a
speed win**: the model *forward* is fast (~1.5 s per whole 36–42-tok turn) but end-to-end turn latency is
dominated by the **shared grammar-FSM host code** (`O(committed²)`; ep1's 110-tok turn > 9 min) — the *same*
cost the HF 3.904 s/turn carries — so there is **no full-battery engine s/turn**, and K3 speed remains
unadjudicated on the engine path. `runs/endgame_scoreboard`, **still no measured engine wall-clock row**:

| row | exact_args | episode_exact | valid | s/turn | fwd-or-tok/turn |
|---|---:|---:|---:|---:|---:|
| ENGINE (this fix) — byte-parity-implied | **= 47/63** | **= 13/20** | **= 63/63** | *forward ~1.5 s; end-to-end host-bound (unmeasured full battery)* | fewer-forwards live (ep2 36 tok / 16 fwd) |
| OUR HF hybrid-clean (v2) — reference | 47/63 | 13/20 | 63/63 | **3.904** | 56.83 denoise fwd/turn |
| stock-bf16-AR-guided (same build) | 51/63 | 14/20 | 63/63 | **1.213** | 82.24 tok/turn |
| stock-AR aggregate | 124/247 | 33/80 | 247/247 | **0.741** | 49.06 tok/turn |

The engine's only legitimate speed lever over guided-AR is the same FSM zero-forward bulk-commit (now
proven live); a *net* s/turn win needs the grammar-FSM host cost made cheap (incremental FSM state, not
re-parsing the growing prefix) — separate future work, not this fix.

### Determinism / temp>0 contract (honest)
Greedy determinism holds (fresh boot, seedA==seedB byte-identical). temp>0 fixed-seed is reproducible
(seeded per-slot `torch.Generator` in `_hc_sample_fn`); seed-diversity is real but *intermittent* (peaked
value distributions) — demonstrated at temp=0.7 (two seeds diverge at pos 33). **Caveat:** the same
greedy prompt wobbles by ~1 token (42 fresh vs 43 after cache is dirtied) from non-associative bf16
reductions over different KV/prefix-cache layouts — a **general vLLM property, not a FLARE defect**;
byte-parity is measured in the fresh per-turn condition (how both sides are captured) and reproduces
across boots. At temp>0 a near-tie can let sampling pick a grammar-illegal value token that the FSM
projects (`value_projection_events` 1) — expected under sampling, distinct from the fresh-greedy `proj==0`
invariant.

### What this changes for the plan
The M1 substrate + the sequential single-`[MASK]` decode are now **byte-parity-correct on the engine** —
the quality blocker is gone. The remaining gap to a *promotable* engine is **speed**: the fewer-forwards
mechanism is proven live (e.g. ep2: 36 tokens / 16 forwards, 20 forced tokens bulk-committed with zero
forwards), but a net s/turn win over guided-AR requires making the grammar-FSM host cost cheap
(incremental FSM state instead of re-parsing the growing prefix) — separate future work, not this fix.

---

## 0.D P2 ENGINE BENCH — first honest matched-20 engine wall-clock + OPT-1 landed (2026-07-04, RTX 5090 / sm_120)

§0.C proved per-turn byte-parity on **3 hand-captured** turns and left K3 speed unadjudicated. This bench
built the **full matched-20 battery** on the engine, landed **OPT-1** (GPU-native sampling, the P0 host-cost
item), and produced the **first honest engine wall-clock** — while surfacing that the §0.C parity does NOT
generalize. vLLM pin `58cfe2c` (GAP-5A windowed-probe forward + OPT-1; baseline pin `58cfe2c` intact, no
regression to `af21dc8`/`1e32dcd`). Real export `qwen3.5-9b-fastdllm-rlv2-vllm-bf16` (block/canvas 32, mamba
1024, align+APC), RAM cage, greedy. Source: `p2_engine_bench_result.md`; full report + artifacts
`runs/p2_engine_bench/report.md`. Bench commit `7629a21`; this doc-update commit follows.

### Prompt reconstruction (byte-faithful — PASS)
The matched-20 eval is a *generated-history* loop, so its 63 per-turn prompts are the HF row's own
teacher-forced history. All 63 prompts were reconstructed from the HF hybrid-clean row and **byte-verified**:
every `prompt_sha256` + `prompt_tokens` matches; the 3 `gap5a_ref` records cross-check `prompt_ids` AND
`ref_new_ids` exactly. The engine then ran greedy per turn with incremental per-turn JSONL.

### Per-step verdict
| step | verdict | one-line |
|---|---|---|
| prompts byte-faithful | **PASS** | all 63 matched-20 prompts reconstructed + byte-verified (sha + tokens); 3 `gap5a_ref` cross-check byte-for-byte. |
| OPT-1 integrity (A/B vs pre-OPT-1 `6b81154`) | **PASS** | engine output **byte-identical** OPT-1 vs pre-OPT-1 on every turn (incl. the divergent ones) at **2.36× speedup** → OPT-1 is a pure, behavior-preserving speedup, zero parity change. |
| full 63-turn battery end-to-end | **CANNOT COMPLETE** | 44/63 turns run; **19 uncompletable** (16 long turns n_ref≥95 + 3 short block-aligned turns, e.g. gt32 plen=1024=mamba block) due to a partial-canvas forward **STALL** (>10 min, non-terminating). |
| byte-parity == HF 47/63 by construction | **DOES NOT GENERALIZE** | parity holds on **35/44** completed turns; **9 diverge** (all `proj=0`), systematically at the first denoise position after a block boundary (`first_div=33` recurs). NOT an optimization regression (present identically at `6b81154`, A/B). |
| temp=0.7 RL sanity | **PASS** | 5 rollouts bounded/valid/`proj=0`, same-seed 2× reproducible. |

### OPT-1 — DONE and verified clean (the P0 host-cost item)
A/B vs a checked-out pre-OPT-1 `hybrid_clean.py` (`6b81154`): **engine output is byte-identical on every turn**
(including the 9 divergent ones) at **2.36× mean speedup**. So OPT-1 (full-vocab host sampling → GPU-native
batched top-k) is a pure, behavior-preserving speedup — it caused **zero** parity change; the divergences and
the stall live at the `6b81154` "parity-PASSED" baseline, i.e. they are **pre-OPT-1 engine-forward defects**,
not OPT-1 regressions. On the 35 parity-hold turns, **engine exact_args == HF exactly (31/31)**.

### The two blockers (both pre-OPT-1, both OPT-3 territory, both correctness/liveness)
1. **Partial-canvas forward STALL.** A single denoise forward **hangs indefinitely (>10 min, non-terminating)**
   when the staged canvas `valid_len` drops below the full block width (measured **32→13 at committed ≈95**).
   This makes **19/63 turns uncompletable** (16 long turns n_ref≥95 + 3 short turns on block-aligned prompts).
   Per-step trace: steps to committed 95 are a flat **27 ms**; cumulative grammar time is **0.017 s = 0.7%** of
   turn time — the prior ">9 min O(committed²) grammar" hypothesis (§0.C) is **DISPROVEN**, and **OPT-5 is
   confirmed a non-issue**. The stall, not grammar, is the liveness blocker.
2. **Non-universal byte-parity.** The GAP-5A windowed-probe fix is a *causal* approximation of the reference's
   windowed-*bidirectional* read (author-flagged). On 44 completed turns, **35 hold byte-parity, 9 diverge**
   (all `proj=0`), systematically at the first denoise position after a block boundary (`first_div=33` recurs).

### Honest numbers (44 completed turns — a short-turn subset, mean 60 tok; NOT a full-battery number)
| | engine | HF (same 44) | HF full-63 | stock-AR-guided 63 | stock-AR agg | M2 |
|---|--:|--:|--:|--:|--:|--:|
| s/turn mean | **1.250** | 2.835 | 3.904 | 1.213 | 0.741 | <1.120 |
| s/turn p50 / worst | 1.185 / 2.201 | 2.756 | — | — | — | — |
| denoise fwd/turn | 40.95 | 39.30 | 56.83 | 82.24 (tok) | 49.06 (tok) | — |
| exact_args | 32/44 | 35/44 | 47/63 | 51/63 | 124/247 | ≥55/63 |

Engine is **2.27× under HF** on the identical completed subset (a real OPT-1 win) and its completed-subset
s/turn sits at the **stock-AR-guided level (1.250 vs 1.213)** — but **M2 (<1.120) is UNADJUDICATED**: the full
battery cannot complete (stall) and byte-parity is not universal, so this 1.250 is a short-turn-subset number,
not a full-63 s/turn. **K3 remains UNADJUDICATED on the engine path.** No sunk-cost full-battery number was
invented.

### Frontier answer: OPT-3 is the single frontier item — and it is a *correctness* fix, not merely efficiency
A byte-EXACT windowed-**bidirectional** variable-width single-`[MASK]` forward (**OPT-3**) fixes **both**
blockers at once: it removes the causal-approximation divergence (restores universal parity → the HF 47/63)
**and** removes the partial-canvas stall (long turns + the full battery run → a real full-battery s/turn). It
is a correctness/liveness blocker, not merely an efficiency lever. **OPT-4** second (residual GPU gap to AR,
only after OPT-3). **OPT-1 is done + verified** (byte-identical, 2.36×). **OPT-5 should NOT be done** (grammar
is 0.7% of turn time). **OPT-2** (`cg_mode` counter + fail-closed config assert) is not yet separately
adjudicated on the engine.

### Artifacts — `runs/p2_engine_bench/`
- `build_matched20_ref.py`, `matched20_ref.json` — reconstructed + byte-verified 63-turn prompt battery.
- `run_battery.py`, `matched20_turns.jsonl` — engine greedy per-turn run with incremental JSONL.
- `ab_opt1.py`, `ab_A_opt1.json`, `ab_B_preopt1.json` — OPT-1 integrity A/B (byte-identical, 2.36×).
- `diag_ep1*.py` — the per-step stall trace (grammar = 0.7%; stall at `valid_len` 32→13).
- `matched20_temp07*.jsonl` — temp=0.7 RL sanity (5 rollouts, `proj=0`, same-seed reproducible).
- `report.md` — full report. vLLM pin `58cfe2c` (windowed-probe + OPT-1); bench commit `7629a21`.

---

## 0.E OPT-3 SYNC-SCHEDULER FIX — FIRST COMPLETE full-63 battery; engine NOT promoted (2026-07-04, RTX 5090 / sm_120)

§0.D produced the first honest engine wall-clock but the full battery **could not complete** (19/63 turns
stalled) and byte-parity was **not universal**. The OPT-3 sync-scheduler fix (vLLM pin `d2fccab`) closes
**both** liveness/rollback defects, so the P2 engine battery now runs **end-to-end for the first time** — the
honest, complete, uncapped full-battery wall-clock the prior bench could not produce. Real export
`qwen3.5-9b-fastdllm-rlv2-vllm-bf16` (block/canvas 32, mamba 1024, align+APC), RAM cage, greedy, uncapped
`n_ref+16`, no harness patches. Source: `p2_engine_battery_result.md`; full report + artifacts
`runs/p2_engine_battery_full/report.md`. Battery commit `61d1381`, pushed to origin/main; this doc-update follows.

### The OPT-3 fix works — the battery completes end-to-end
All 63 turns complete, **zero stalls** (the ex-stalls gt4/gt24/gt50 all finish), and the async-rollback
boundary divergence is gone: **0/11 breaks at pos-33**, and the 4 formerly-async-divergent turns
gt12/16/18/20 are byte-parity again. This is the first complete, uncapped full-battery run.

### Required-check results (greedy, uncapped, no harness patches) — 63/63 byte-parity NOT met ⇒ NOT promoted
| task check | required (promotable) | measured | verdict |
|---|---|---|---|
| (1) byte-parity/turn | **63/63** | **52/63** | **NOT met** |
| (2) exact_args | == 47/63 (HF) | **48/63** | deviation **+1** |
| (2) episode_exact | 13/20 | **13/20** | **met** |
| (2) valid | 63/63 | **62/63** | deviation **−1** |
| verify_invariants / value_projection | — / 0 | **63/63 / 0** | clean |

**The 63/63 byte-parity promotable claim is NOT met (52/63), so the run was diagnosed rather than reported as
a promotion.** The 11 divergences are the *separate, author-flagged* windowed-probe **causal** approximation of
the reference's windowed-**bidirectional** read (all `value_projection=0`, mid-block value logits;
`first_div` scattered at {17,19,19,19,26,31,34,38,41,47,53} — **none at pos-33**, i.e. NOT the fixed
async/stall bug). The two eng≠hf turns: **gt60** (engine **+1**, correct where HF misses) and **gt19**
(divergence → non-stopping run → the one invalid). The engine is **quality ≥ HF in aggregate but not
byte-identical** to HF.

### Timing (all 63 turns) — first honest full-battery s/turn
- s/turn **mean 1.681**, **p50 1.427**, **p90 2.724**, min 0.512, **worst 5.361** (gt50, 207 tok / 190 fwd).
- **TRUE denoise forwards/turn = 56.65** (HF 56.83), tokens/forward 1.360, **2.32× under HF**.
- Full per-turn distribution in `runs/p2_engine_battery_full/aggregate.json`.

### Honest table + M2/K3 adjudication — adjudicable for the first time, and MISSED on both axes
| row | exact | ep | valid | s/turn | fwd/turn |
|---|---:|---:|---:|---:|---:|
| **ENGINE full-63 (this fix)** | **48/63** | **13/20** | 62/63 | **1.681** | 56.65 |
| HF full-63 | 47/63 | 13/20 | 63/63 | 3.904 | 56.83 |
| stock-AR-guided | 51/63 | — | 63/63 | 1.213 | 82.24 (tok) |
| stock-AR aggregate | 124/247 | — | — | 0.741 | 49.06 (tok) |
| M2/K3 target | ≥55 | — | — | <1.120 | — |

**M2/K3 adjudicable for the first time — and MISSED on both axes:** **speed** 1.681 > 1.120 (1.39× *slower*
than guided-AR 1.213; the §0.D bench's 1.250 was a short-turn subset excluding the 16 long turns), **quality**
48 < 55. But the engine is **2.32× under HF** and, critically, now *measurable*. So M2 is **MISSED** (not
UNADJUDICATED as in §0.D), K3 speed is **MISSED**. No promotion.

### temp=0.7 RL sanity — contract holds
5 seeded rollouts (gt0/7/17/29/51): **two boots byte-reproducible**; all `finish=stop`, valid, `proj=0`;
peaked value distributions collapse to greedy. The RL contract holds.

### Next lever — OPT-4 (measured, `torch.profiler` kernel-level, 3 turns)
Kernel breakdown: gemm **~62%** (MLP+proj+lm_head, computed over CL=32 rows to read 1 probe logit) > copy
~21% > full-attn 6–9% > **GDN chunk path ~5%** > sampling 0.5% (OPT-1 confirmed). GDN runs the **prefill
`chunk_gated_delta_rule` kernels** (`chunk_gated_delta_rule_fwd_kernel_h`, `chunk_fwd_kernel_o`,
`chunk_scaled_dot_kkt`, `recompute_w_u`, `_causal_conv1d`) — **`fused_recurrent` is absent**. Per-forward
~18 ms GPU + ~11 ms host (`enforce_eager=True`, no CUDA graph). **OPT-4 = incremental KV+GDN 1-token decode →
`fused_recurrent` + FULL CUDA graph** (the gemm win is bounded by weight-bandwidth at batch=1; the real levers
are the recurrent decode kernel + graphing out the ~11 ms host per forward). This is the single next lever for
the speed bar; the residual 11-turn parity gap is the windowed-**bidirectional** refinement, which also
tightens exact_args toward the ≥55 quality bar.

### Artifacts — `runs/p2_engine_battery_full/` (committed `61d1381`, pushed to origin/main)
- `report.md` — full writeup; `matched20_turns.jsonl` — engine greedy per-turn run.
- `aggregate.json` — full per-turn timing + check distribution.
- `matched20_temp07a.jsonl`, `matched20_temp07b.jsonl` — temp=0.7 RL sanity (2 byte-reproducible boots).
- `profile_opt4.py`, `opt4_breakdown.json` — the `torch.profiler` kernel-level OPT-4 breakdown.
- vLLM pin `d2fccab` (OPT-3 sync scheduler). `p2_engine_battery_result.md` — tracked summary.

---

## 0.F P2 ENGINE BATTERY v2 — FINAL engine (bidir probe + PIECEWISE cudagraph); strongest candidate, NOT PROMOTED (2026-07-04, RTX 5090 / sm_120)

§0.E ran the OPT-3 sync-scheduler engine (windowed-**causal** probe, `enforce_eager`) and MISSED M2/K3 on both
axes (1.681 s/turn, 52/63 parity). Since then BOTH remaining levers landed and this v2 battery runs the promotable
full-63 on the **FINAL engine** = vLLM pin `e5496cc` with:
- `VLLM_FLARE_BIDIR_PROBE=1` — the reference-exact windowed-**bidirectional** denoise read (`b7d76e2`), replacing
  §0.E's causal approximation. This is the byte-parity refinement OPT-3 flagged as its residual.
- `VLLM_FLARE_CUDAGRAPH=1` — PIECEWISE CUDA graph (OPT-4 Part 2). **Confirmed live:** `enforce_eager=False`,
  `cudagraph_mode=PIECEWISE`, **3756 PIECEWISE dispatches** in the run.

Greedy, temp 0, seed 20260701, uncapped (`n_ref+16`), RAM cage, two boots (ep0-9, ep10-19). Real export
`qwen3.5-9b-fastdllm-rlv2-vllm-bf16` (block/canvas 32, mamba 1024, align+APC). Source: `p2_engine_battery_v2_result.md`;
full report + artifacts `runs/p2_engine_battery_v2/report.md`. Battery commit `1acdf2e`, pushed to origin/main; this
doc-update follows.

### Required-check results — 63/63 byte-parity NOT met ⇒ NOT PROMOTED
| task check | required (promotable) | measured | verdict |
|---|---|---|---|
| (1) byte-parity/turn | **63/63** | **58/63** (breaks {20,21,44,45,60}) | **NOT met** |
| (2) exact_args | == 47 (HF) | **48** (engine WINS gt60, ≥HF) | deviation **+1** |
| (2) episode_exact | 13/20 | **13/20** | **met** |
| (2) valid | 63/63 | **63/63** (bidir fixed gt19) | **met** |
| verify_invariants / value_projection | — / 0 | **63/63 / 0/63** | clean |

Per the stop-and-diagnose rule: byte-parity is **58/63**, so the by-construction chain that would force exact=47
does **not** hold — the run is diagnosed, **not promoted**. Aggregate quality is **≥ HF but not byte-identical**:
exact_args 48 (+1; **gt60 the engine is correct where HF misses**), valid 63/63 (bidir fixed gt19's non-stopping
divergence, up from §0.E's 62). `value_projection=0/63`, `verify_invariants=63/63`, projection `0/63` — clean.

### Timing (all 63 turns) — cudagraph win
- s/turn **mean 1.051**, p50 **0.876**, p90 **1.699**, min 0.326, **worst 4.248** (gt50, 259 tok).
- **TRUE denoise forwards/turn = 56.62**, tokens/forward 1.362, **per-forward 18.56 ms** (eager ~29 ms →
  **1.615× cudagraph win**), **speedup vs HF 3.715×**.

### Bar adjudication (mean 1.051) — clears M2/guided-AR/HF for the first time
| bar | value | engine 1.051 | verdict |
|---|---:|---:|---|
| HF hybrid-clean (v2) | 3.904 | 0.269× | **BEAT** |
| stock-bf16-AR-guided | 1.213 | 0.866× | **BEAT** |
| **M2 speed target** | **1.120** | **0.938×** | **BEAT** |
| stock-AR aggregate | 0.741 | 1.418× | **MISS** |

The §0.E eager engine (1.681) missed both M2 and guided-AR; **cudagraph now clears both for the first time on the
honest full-63** — the engine sits *below* guided-AR and *below* the M2 speed bar at ≥HF quality, still above the
stock-AR aggregate 0.741 (the beyond-AR / OPT-6 bar). **M2/K3 adjudication:** M2 **speed** bar MET (1.051 < 1.120,
first time), **K3 speed MET**; M2 **quality** axis MISSED (58/63 byte-parity ≠ 63/63; exact 48 < 55) ⇒ the combined
M2 promotion gate is **not** met and the strict 63/63-byte-parity gate is **NOT met** ⇒ NOT PROMOTED.

### Honest table
| row | exact | ep | valid | s/turn | fwd/turn |
|---|---:|---:|---:|---:|---:|
| **ENGINE FINAL (bidir + PIECEWISE cudagraph)** | **48/63** | **13/20** | **63/63** | **1.051** | 56.62 |
| ENGINE §0.E (OPT-3 eager, causal) | 48/63 | 13/20 | 62/63 | 1.681 | 56.65 |
| HF hybrid-clean (v2) | 47/63 | 13/20 | 63/63 | 3.904 | 56.83 |
| stock-bf16-AR-guided | 51/63 | — | 63/63 | 1.213 | 82.24 (tok) |
| stock-AR aggregate | 124/247 | — | — | 0.741 | 49.06 (tok) |
| M2/K3 target | ≥55 | — | — | <1.120 | — |

### temp=0.7 RL sanity + never-train spot-check — contract holds under cudagraph
- **temp=0.7:** 5 turns (gt0/7/17/29/51) × 2 boots — **byte-reproducible** (identical n_gen/fwd/parity, ≤3 ms wall
  jitter), all bounded/valid/`proj=0`; peaked value distributions collapse onto greedy. RL contract holds under
  cudagraph.
- **Never-train spot-check:** 3 turns spanning BFCL-AST + API-Bank Lv1/Lv2 (prompts sha256-verified vs HF) →
  **3/3 byte-parity, 3/3 valid, 3/3 exact, 0 projection, cudagraph active**. The parity is **not matched-20-specific**.

### Why 63/63 is unreachable in this (or any current) config — diagnosis
The bidir read is the **correct** reference semantics; cudagraph is **byte-neutral on the entire promotable set**
(reproduces the bidir-eager anchor `parity_bidir/battery_bidir.jsonl` 58/63 + the break-set exactly; the only deltas
vs eager are the already-divergent tails of gt20/gt44, neither promotable). The 5-turn parity residual + the single
exact deviation are the **coupled, documented, UNLANDED** work: **32-absolute commit alignment**
(`VLLM_FLARE_ALIGN_BLOCKS`, scaffold only) + per-request **variable commit width** — which is simultaneously
**OPT-4 Part 1** (the remaining forward-compute cut to the stock-AR aggregate). **Parity closure and the last speed
cut land together.** Break classes: **gt20/gt45** bidir alignment regressions, **gt21** APC/prefix-cache artifact,
**gt44** variable-width miss, **gt60** engine WINS (correct where HF is wrong).

### Artifacts — `runs/p2_engine_battery_v2/` (committed `1acdf2e`, pushed origin/main)
- `report.md` — full writeup; `matched20_turns.jsonl` (63) — engine greedy per-turn run.
- `aggregate.json`, `aggregate.py` — full per-turn timing + check distribution.
- `matched20_temp07a.jsonl`, `matched20_temp07b.jsonl` — temp=0.7 RL sanity (2 byte-reproducible boots).
- `nevertrain_ref.json` (184 sha-verified) + `nevertrain_spotcheck.jsonl` — never-train spot-check.
- Drivers: `run_battery_v2.py`, `build_nevertrain3_ref.py`, `env.sh`, `smoke.jsonl`.
- `scripts/parity_audit_flare_engine.py` — the `VLLM_FLARE_CUDAGRAPH` opt-in seam (default stays eager; `=1` opts
  into PIECEWISE). `p2_engine_battery_v2_result.md` — tracked top-level summary.
- vLLM pin `e5496cc` (bidir probe `b7d76e2` + PIECEWISE cudagraph OPT-4 Part 2).

---

## 0.G P2 ENGINE BATTERY v3 — THE PROMOTION ATTEMPT; NOT PROMOTED; residual localized to {44,45} (2026-07-04, RTX 5090 / sm_120)

§0.F was the strongest candidate; v3 is the **explicit promotion attempt** against the strict gate and an independent
3rd boot. The engine tree is **clean at `e5496cc` = byte-identical to v2** (OPT-4 Part 1 / Task #37 still UNLANDED), so
this is a faithful re-run, not a new engine. It **reproduces v2 exactly** (n_gen / fwd / parity / exact / first_div ALL
identical) and adds the documented **fresh-context parity certificate**, which sharpens the residual to exactly
**{44,45}**. Same protocol as §0.F: greedy, temp 0, seed 20260701, uncapped, RAM cage, export
`qwen3.5-9b-fastdllm-rlv2-vllm-bf16` (block/canvas 32, mamba 1024, align+APC), FINAL engine
`VLLM_FLARE_BIDIR_PROBE=1 + VLLM_FLARE_CUDAGRAPH=1`. Source: `p2_engine_battery_v3_result.md`; full report + 11
artifacts `runs/p2_engine_battery_v3/report.md`. Battery commit `55965de`, pushed origin/main; this doc-update follows.

### Promotion gate — 63/63 byte-parity ⇒ exact exactly 47 NOT met ⇒ NOT PROMOTED
| promotion-gate check | required | measured | verdict |
|---|---|---|---|
| byte-parity/turn | **63/63** | **58/63** (breaks {20,21,44,45,60}) | **NOT met** |
| exact_args | **exactly 47** | **48** (+1 = gt60, an APC artifact) | **deviation** |
| episode_exact | 13/20 | **13/20** | **met** |
| valid | 63/63 | **63/63** | **met** |
| verify_invariants / value_projection | clean / 0 | **63/63 / 0/63** | **met** |
| determinism vs v2 | — | n_gen/fwd/parity/exact/first_div **ALL identical** | **3rd-boot repro** |

The gate needs 63/63 ⇒ the by-construction chain that would force exact=47; parity is **58/63** so exact is **48**.
**NOT PROMOTED.**

### Numbers (APC-on full-63) — reproduces v2
| | ENGINE v3 (=v2) | HF | guided-AR | stock-agg | M2 |
|---|---:|---:|---:|---:|---:|
| exact_args | **48/63** | 47/63 | 51/63 | 124/247 | ≥55 |
| episode | 13/20 | 13/20 | — | — | — |
| valid | 63/63 | 63/63 | 63/63 | — | — |
| byte-parity | 58/63 | (self) | — | — | 63/63 |
| s/turn mean | **1.056** | 3.904 | 1.213 | 0.741 | <1.120 |
| s/turn p50 / p90 / max | 0.874 / 1.734 / 4.253 | — | — | — | — |
| denoise fwd/turn | 56.62 | 56.83 | 82.24 tok | 49.06 tok | — |
| per-forward ms | 18.66 | — | — | — | — |

### Bar adjudication (mean 1.056)
| bar | value | engine 1.056 | verdict |
|---|---:|---:|---|
| HF hybrid-clean (v2) | 3.904 | 0.270× (3.695×) | **BEAT** |
| stock-bf16-AR-guided | 1.213 | 0.871× | **BEAT** |
| **M2 speed target** | **1.120** | **0.943×** | **BEAT** |
| stock-AR aggregate | 0.741 | 1.425× | **MISS** |

### THE v3 FINDING — byte-parity is cache-path-dependent; the invariant residual is {44,45}
A **fresh-context parity certificate** (fresh boot per turn = cold prefix + mamba cache) was run alongside the APC-on
battery. `enable_prefix_caching=False` is invalid here (mamba-block-size dependency) and `reset_prefix_cache()` is a
false proxy that corrupts the un-reset GDN cache — both are documented as negative controls. 57/63 fresh boots measured
(6 pending under the concurrent Stage-3 GPU hold; all 6 are APC-parity). The two paths split the break-set:
- **invariant breaks (BOTH paths): {44,45}** — genuine OPT-4 Part 1 (gt44 fd16 variable-width; gt45 fd20 32-absolute align).
- **APC-only breaks (resolve fresh): {20,21,60}** — cross-turn prefix-cache artifacts, not denoise errors.
- **fresh-only breaks (hidden by APC): {1,3,12,23,24,50,57}**.

gt60's "engine wins" is an **APC artifact**: fresh, it byte-matches HF's 169-tok output *incl. HF's mistake* →
eng_exact 0 = hf (exactly the gate memo's prediction). So the **"58/63" headline is cache-config-specific**; the robust
promotion blocker is the 2-turn structural set **{44,45}**. Of the 57 turns measured in both paths, 45 are parity in
both, 2 break in both, ~10 flip on the cache path.

### Residual gap to 0.741 (measured) — REACHABLE
Need per-forward **18.66 → 13.09 ms** at 56.62 fwd/turn (cut 5.57 ms). The weight-stream floor is 10.5 ms, so the bar
target sits **2.59 ms above the floor — REACHABLE**. Lever = **OPT-4 Part 1**: variable commit width shrinks the CL=32
gemm/attn rows + a width-1 GDN routes to `fused_recurrent`; it also lowers fwd/turn. **It closes {44,45} AND the speed
gap to stock-agg together.**

### RL/OOD sanity — contract holds
- **temp=0.7:** 5 rollouts byte-reproducible across 2 boots, all valid / exact / proj0.
- **never-train:** 3/3 (BFCL-AST, API-Bank Lv1/Lv2) byte-parity + exact vs HF.

### Verdict
Strongest candidate to date, reproduces v2 exactly, beats HF / guided-AR / M2 on speed — but the strict gate
(63/63 ⇒ exact 47) is **NOT met**. The fresh-context certificate localizes the genuine residual to **{44,45}** and
shows the gt60 exact-win is an APC artifact. **NOT PROMOTED.** Single coupled lever = **OPT-4 Part 1** (a concurrent
Stage-3 session is actively developing it). No engine row is added to the endgame scoreboard (promotion gate not met).

### Artifacts — `runs/p2_engine_battery_v3/` (committed `55965de`, pushed origin/main; 11 empirical artifacts)
- `report.md` — full writeup; `aggregate.json` — per-turn timing + check distribution.
- APC-on full-63 battery + the fresh-context parity certificate (57/63 fresh boots) + the two negative controls
  (`enable_prefix_caching=False`, `reset_prefix_cache()`).
- temp=0.7 RL sanity (5 rollouts × 2 boots) + never-train spot-check (BFCL-AST + API-Bank Lv1/Lv2, sha-verified).
- `p2_engine_battery_v3_result.md` — tracked top-level summary.
- vLLM pin `e5496cc` = byte-identical to v2 (OPT-4 Part 1 / Task #37 UNLANDED).

---

## 0.H P2 ENGINE BATTERY v3b — PROMOTION ATTEMPT on the POST-FIX engine (OPT-4 Stage 1+2+3 landed); 62/63, exact EXACTLY 47; NOT PROMOTED (2026-07-04, RTX 5090 / sm_120)

§0.G was on the **pre-fix** pin `e5496cc` (58/63). OPT-4 Part 1 has since **landed** on pin `95d8b47` (Stage 1
32-absolute variable commit width + Stage 2 scheduler width plumbing + Stage 3 byte-robust bidir key window; **code
default OFF**). v3b is an **independent fresh boot of the post-fix engine** — the real promotion attempt. Same protocol
as §0.F/§0.G: greedy, temp 0, seed 20260701, uncapped, RAM cage, chunked foreground (ep0-9, ep10-19), export
`qwen3.5-9b-fastdllm-rlv2-vllm-bf16`, FINAL engine `VLLM_FLARE_BIDIR_PROBE=1 + VLLM_FLARE_CUDAGRAPH=1`. Source:
`p2_engine_battery_v3b_result.md`; full report + artifacts `runs/p2_engine_battery_v3b/report.md`.

### Promotion gate — 62/63 byte-parity ⇒ 63/63 NOT met ⇒ NOT PROMOTED (but exact IS exactly 47)
| promotion-gate check | required | measured | verdict |
|---|---|---|---|
| byte-parity/turn | **63/63** | **62/63** (lone break {44}) | **NOT met** — 1 turn short |
| exact_args | **exactly 47** | **47** (0 turns eng≠hf) | **met** |
| episode_exact | 13/20 | **13/20** (ties HF) | **met** |
| valid | 63/63 | **63/63** | **met** |
| verify_invariants / value_projection | clean / 0 | **63/63 / 0/63** | **met** |
| delta vs pre-fix v3 | — | fix cleared **{20,21,45,60}** path-robustly; **no regressions**; shared-clean byte-identical | **fix landed** |

The Stage-3 fix takes **58/63 → 62/63** and drops the pre-fix gt60 "engine wins" (exact 48 → **exactly 47**, byte-matching
HF incl. HF's mistake). The **lone break gt44** is a **path-invariant deterministic fp-residue**: it breaks identically
under APC-on AND cold-prefix fresh-boot (same fd16, n101), so it is **NOT an APC class** — the documented APC protocol
cannot rescue it to 63/63. gt44 is non-exact for BOTH engine and HF (quality-neutral). **NOT PROMOTED; default OFF.**

### Numbers (APC-on full-63)
| | ENGINE v3b (post-fix) | pre-fix v3/v2 | HF | guided-AR | stock-agg | M2 |
|---|---:|---:|---:|---:|---:|---:|
| byte-parity | **62/63** ({44}) | 58/63 | (self) | — | — | 63/63 |
| exact_args | **47/63** (==HF) | 48/63 | 47/63 | 51/63 | 124/247 | ≥55 |
| valid / episode | 63/63 / 13/20 | 63/63 / 13/20 | 63/63 / 13/20 | 63/63 | — | — |
| s/turn mean (p50/p90/max) | **1.053** (0.896/1.700/4.241) | 1.056 | 3.904 | 1.213 | 0.741 | <1.120 |
| denoise fwd/turn · per-fwd ms | 56.86 · **18.52** | 56.62 · 18.66 | 56.83 | 82.24 tok | 49.06 tok | — |

### Bar adjudication (mean 1.053): HF 3.904 **BEAT** (0.270×) · guided-AR 1.213 **BEAT** (0.868×) · M2 1.120 **BEAT** (0.940×) · stock-agg 0.741 **MISS** (1.421×).

### THE v3b CORRECTION — 0.741 is NOT reachable by width-narrowing (supersedes §0.G "REACHABLE")
§0.G predicted OPT-4 Part 1 would close the 0.741 gap. **It landed, and the Stage-3 A/B measured it speed-NEUTRAL**
(variable-width 18.52 vs fixed-32 18.56 ms/forward — cudagraph buckets narrow widths back to a captured bucket). The
**measured** per-forward decomposition (torch.profiler device self-time, current pin): 18.52 ms wall = **weight-stream
floor 11.40 ms** (gemm MLP+proj+lm_head, 63.5% of GPU; arithmetic 10.77 ms = 19.31 GB bf16 / 1.79 TB/s HBM,
**irreducible at bs=1**) + non-weight GPU compute **6.54 ms** (proven NOT width-reducible) + residual host **0.58 ms**.
Bar needs 13.03 ms/fwd (only 1.64 ms above the weight floor), but the 6.54 ms non-weight compute doesn't shrink → **not
reachable by engine plumbing at batch=1.** Levers are orthogonal: fewer forwards/turn (training), fp8/int8 weights
(~0.68/0.51 s/turn, quality tradeoff), or batching. stock-agg is also stock-AR over a different, shorter mix (49 tok/turn).

### RL/OOD sanity — contract holds post-fix
- **temp=0.7:** 5 rollouts (gt0/7/17/29/51) byte-reproducible across 2 boots (max wall Δ 4 ms), all valid/exact/proj0/parity.
- **never-train:** 3/3 (BFCL-AST, API-Bank Lv1/Lv2) byte-parity + exact vs HF.

### Verdict
Strongest promotable candidate to date and the **first independent certificate of the landed OPT-4 fix**: 62/63,
exact **exactly 47 (==HF, 0 wins/losses)**, valid 63/63, episode 13/20, verify 63/63, projection 0, mean 1.053 s/turn
(beats M2, guided-AR, HF). But **63/63 is one turn short** — gt44 is a proven path-invariant fp-residue (block#0 GDN
fold-path fp gap; matching HF's fold granularity is kernel-level, deferred). **NOT PROMOTED; default OFF.** No engine
row added to the endgame scoreboard (gate not met).

### Artifacts — `runs/p2_engine_battery_v3b/` + `p2_engine_battery_v3b_result.md` (tracked)
`matched20_turns.jsonl` (63) · `aggregate.json` · `matched20_temp07{a,b}.jsonl` · `nevertrain_spotcheck.jsonl` (3) ·
`parity_cert_freshboot.jsonl` (5) · `opt4_breakdown.json` (profiler split) · `run_battery_v3b.py` / `aggregate.py` /
`profile_v3b.py` / `env.sh` / `runcage.sh` / `chunk{1,2}.log`. vLLM pin `95d8b47` (Stage 1+2+3 landed, default OFF).

---

## 0.I BATCHED RL-ROLLOUT THROUGHPUT — engine vs stock-AR-guided across batch {1,2,4,8,16}; thesis DISCONFIRMED (2026-07-04, RTX 5090 / sm_120)

The prior sections adjudicated **single-turn** wall-clock (0.626 s/turn agg, beats stock-AR). This section answers the
distinct **RL-rollout regime** question: does the diffusion twin generate on-policy signal *at higher throughput* when
rollouts are **batched**? The a-priori thesis was that the FLOP-reducing hybrid (~10× fewer forwards/turn) would hold or
grow its advantage with batch. **Bench DISCONFIRMS it against a fast guided-AR baseline.** Source:
`runs/p2_batched_rollout_bench/report.md`; bench commit `58524f9` (pushed origin/main).

### Batch-correctness precondition — PASSED (bench was safe to run)
`runs/p2_engine_batchgates`: **NO cross-request contamination** on the batched path — per-request GDN snapshot/restore
state stays isolated across co-batched requests. The batched path is SAFE; only *then* was the throughput sweep run.

### The two throughput curves + the ratio that matters (samples/sec == rollouts/sec/GPU)
All on the certified engine (pin `95d8b47`, PIECEWISE cudagraph + APC, BIDIR_PROBE), RTX 5090, 22 GB RAM cage,
foreground; **48 never-train BFCL/API-Bank turns/point** (48 divisible by every batch); temp=0.7 seeded (RL mode).
**AR = stock Qwen3.5-9B guided decoding** (`regex_from_qwen_xml_tool_schema`, the exact scoreboard guidance) given its
*fastest* offline+cudagraph path — **conservative for the engine** (engine must beat AR at AR's best).

| batch | eng samp/s | AR samp/s | **eng/AR** | eng tok/s | AR tok/s | eng occ (eff) | eng util% | AR util% |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 1 | 1.524 | 1.625 | **0.94×** | 75 | 82 | 1.0 (1.00) | 88 | 100 |
| 2 | 2.248 | 2.499 | **0.90×** | 109 | 126 | 1.5 (0.77) | 88 | 100 |
| 4 | 3.426 | 4.103 | **0.83×** | 167 | 207 | 2.7 (0.68) | 87 | 100 |
| 8 | 4.948 | 6.601 | **0.75×** | 243 | 333 | 4.8 (0.60) | 87 | 100 |
| 16 | 5.732 | 7.846 | **0.73×** | 279 | 395 | 7.2 (0.45) | 84 | 100 |

The ratio goes the **wrong way**: 0.94 → 0.90 → 0.83 → 0.75 → 0.73. The engine is at rollout **parity at batch=1** and
**LOSES ground as batch grows**; AR amortizes its weight-stream floor faster (scale 4.83× at b16 vs engine 3.76×).
Greedy engine b8 = 4.798 samp/s ≈ temp=0.7 b8 → **throughput is temperature-insensitive** (RL sampling mode is free).

### Mechanism (measured) — FLOP reduction is real but cancelled twice
1. **Per-forward cost ~14× AR's.** Hybrid does ~10× fewer forwards (4.9/turn at b16 vs AR's ~50 decode steps) but
   `per_forward_ms` **climbs 18.7 → 35.3 ms** with batch (wider GEMMs) while AR's per-step **falls 12.2 → 2.53 ms**
   (cudagraph amortizes weight load). 4.9 × 35.3 ≈ 173 ms/turn forward ≈ 50 × 2.53 ≈ 127 ms — AR is cheaper per turn.
2. **Batch occupancy collapses — the STRUCTURAL limitation.** The FLARE **sync scheduler** + per-request **variable
   draft widths** (3–18 tok/commit) mean requests finish at wildly different forward counts, so a synchronous wave
   stalls on its slowest member (head-of-line/straggler). Effective batch-in-forward: **7.2/16 = 0.45 at b16 — it never
   co-batches all 16** (histogram scattered 1..14). AR co-batches near-linearly at 100% GPU util; the engine idles at
   **84–88%** (host-bound FLARE per-request state management, not compute).

### Sync-scheduler honesty + structural batch limitation
No hard queueing / no deadlock — **48/48 finish `stop`** (bounded valid tool call) at every batch. The real cost is
**head-of-line/straggler blocking** intrinsic to the variable-draft-width hybrid under FLARE's *forced* sync scheduler.
Continuous batching (refill slots as requests finish) would recover *some* occupancy but is **bounded by the forced
sync** — it cannot reach AR's near-linear co-batching. **Memory limitation:** the per-request GDN snapshot/restore state
makes **b16 OOM at gmu 0.74** (~31.3 GB spike); it fits only at gmu 0.62 (~25.5 GB). AR is flat ~21.9 GB at every batch
→ **the engine's practical rollout concurrency is capped tighter than AR's on a 32 GB card.** Host-RAM peak both ~6.9 GB.

### Verdict — the engine's value is NOT samples/sec
Throughput-bound RL loops are **faster with stock guided-AR (1.1–1.4× at batch)**. The engine earns its place on
**quality/parity at safe batch**, not rollout throughput: safe batching (proven), **latency parity** at low batch,
**48/48 valid** stops (guided-AR truncated 2/48 at the cap), and the certified **quality edge** (130 vs 124 exact_args
aggregate). This is a **floor, not a ceiling** — the 84–88% util flags host-bound headroom; **OPT-4 Part-1
fused_recurrent** (task #37) is the lever to lower `per_forward_ms` and lift the whole ratio curve.

### Artifacts — `runs/p2_batched_rollout_bench/` (bench commit `58524f9`, pushed origin/main)
`report.md` · `engine_points.jsonl` / `ar_points.jsonl` · `engine_throughput.json` / `ar_throughput.json` /
`compare.json` / `compare_table.md` · harness `bench_engine.py` / `bench_ar_guided.py` / `gpu_sampler.py` /
`make_report.py` / `runcage_*.sh` · raw `tp_engine.log` / `tp_ar.log` · b16-OOM evidence `engine_throughput.gmu074.json`
+ `tp_engine.gmu074.log` · correctness precondition `runs/p2_engine_batchgates/`.

---

## 0.J LOSSLESS-APC — Route A gate battery STOPPED at gate-1 (seam inert) + END-GOAL agentic serving bench (2026-07-05, RTX 5090 / sm_120)

The user end-goal (task #53) demands a **LOSSLESS prefix cache**: cache-on byte-identical to fresh-context
decode, so the parity certificate holds *with* cross-turn reuse (today's certificate is anchored fresh-context
as a workaround for the {20,21,60}/gt130 APC artifact class). This section runs the lossless-APC gate battery
in order, stops at the first hard failure, and then benches the **speed envelope** a lossless publish would
inherit. Sources: `runs/lossless_apc/gates/gate_results.jsonl`, `runs/lossless_apc/bench/` (bench commit
`b6586f0`, pushed origin/main); Route A design in `runs/lossless_apc/rootcause.md`.

### Probe finding (gate-0 preconditions — the primitive layer is REAL and bit-exact)
Two preconditions both PASS, proving the Route A **math** is correct:
- **CPU state-machine suite:** `pytest tests/v1/worker/gpu/test_qwen3_5_flare_state_machine.py` → **52 passed / 0
  failed (7.2 s)** (39 baseline + 13 new incl a bitwise-identity property, `torch.equal` atol=0).
- **GPU refold-parity probe** (`flare_refold_parity_probe.py`, the **real** `chunk_gated_delta_rule` kernel at
  deployed geometry H=32, K=V=128, stride=1024, fp32, 32-token commits):

  | path | max_abs vs fresh | bits differ | verdict |
  |---|---:|---:|:---:|
  | deployed **lossy** 32-fold | 5.555e-5 | 319 599 / 524 288 (**61%**) | diverges |
  | **Route A refold** | **0.0** | **0 / 524 288** | **bit-identical** |
  | **Route A publish buffer** | **0.0** | **0 / 524 288** (published=1) | **bit-identical** |

  ⇒ Route A's canonical-checkpoint refold reproduces fresh recompute **bit-for-bit** on the real kernel; the
  shipped 32-fold align path diverges on 61% of GDN state bits. **The math is done; only the wiring is missing.**

### Route A (canonical GDN state boundaries — the lossless design)
Cache only chunk-aligned GDN states computed via the **same kernel path** as fresh recompute (bitwise
losslessness by construction; attention-KV reuse was exact already). At a 1024-token checkpoint crossing the
committed tokens are re-folded through `chunk_gated_delta_rule` and the resulting canonical state — not the
lossy incremental 32-fold — becomes the reused checkpoint. Landed in pin `aedf465` as **CPU-validated
primitives + a GATED staging seam** (`VLLM_QWEN3_5_FLARE_CANONICAL_PUBLISH`).

### Gate-1 — cache-on byte-match FRESH under multi-turn reuse: **BLOCKED-CANNOT-PASS (structural, not a math error)**
The staging seam is **landed but INERT**. Whole-repo grep: the two capture sinks that must feed the replay
window from the GDN forward — `capture_committed_gdn_inputs` (`qwen3_5_flare.py:1067`) and `seed_replay_window`
(`:1096`) — are **defined with ZERO callers** anywhere in `vllm/`. `_run_canonical_publish` only **stages** into
`GdnReplayState.published[layer_id]` (`:1184`); it **never applies** to the live align ssm row. The engine
documents this itself (`qwen3_5_flare.py:320-321`): *"Until (i) is wired the windows never fill, so even gate-on
is a no-op that touches no live cache."* ⇒ setting `VLLM_QWEN3_5_FLARE_CANONICAL_PUBLISH=1` changes **zero
serving bits** — replay windows never populate → `window.ready()` never true → publish never written. Gate-1
would fail identically with the env gate ON or OFF, so cache-on output is byte-**identical to today's lossy
gate-OFF path**, which diverges from fresh on the {gt20,gt21,gt60}/gt130 census class.

**Did NOT burn the ~8–15 GPU-h census/battery:** the divergence is code-guaranteed and already captured in
`runs/p2_engine_nevertrain/`. Re-deriving it yields zero new signal.

**Wiring remaining (the two blockers to a real lossless arm):**
- **W1** — call `capture_committed_gdn_inputs` + `seed_replay_window` from the GDN commit/prefill forward
  (`qwen_gdn_linear_attn.py`) with the exact per-request post-l2norm q/k/v/g/beta sliced to committed tokens,
  seeded from the prior canonical checkpoint.
- **W2** — apply the staged `published[layer_id]` into the align checkpoint ssm row at the 1024 crossing
  (currently staged only, never applied).

### Cache-on certificate status + gates 2–5 (all DEPENDENT, not run)
- **Gate-2 (full-63 cache-on parity certificate):** NOT-RUN-DEPENDENT. **The parity certificate CANNOT be
  upgraded to cache-on while the fix is inert** — it stays **anchored fresh-context**. No lossless cache-on
  certificate exists yet.
- **Gate-3 (20-turn never-train spot):** NOT-RUN-DEPENDENT.
- **Gate-4 (no-regression set):** NOT-RUN-DEPENDENT. With the gate OFF the seam is fully inert, so the shipped
  byte-parity / read-only-denoise / determinism / cudagraph certificates are **preserved by construction**; the
  no-regression set becomes meaningful only once W1+W2 land and must be re-run with the gate ON.
- **Gate-5 (refold overhead):** NOT-RUN-DEPENDENT — **un-measured**. Per-checkpoint refold is one
  `chunk_gated_delta_rule` call over ≤1024 committed tokens; it cannot be measured on the live path until W1
  feeds real windows. Envelope estimate below.

### END-GOAL agentic serving bench — the prefill-reuse SPEED ENVELOPE a lossless publish would inherit
Because **no functional lossless mode exists to time**, this benches the deployed align-APC reuse (byte-lossy on
a quality-neutral near-tie class: engine-on **52/57 byte-parity vs cold**; `exact_args` is APC-invariant) to
bound the savings a lossless publish inherits. Same vLLM build both backends (pin `aedf465`), PIECEWISE
cudagraph, batch=1, greedy, seed 20260701; nevertrain **ep0-9 = 10 episodes / 57 turns**, growing context
**1175→2640 tok**; exact per-turn `prompt_ids` replayed token-identically through the diffusion engine and stock
`Qwen3.5-9B` AR. Cold = `reset_prefix_cache()`/turn (the hybrid rejects `enable_prefix_caching=False` with
`mamba_block_size` set — the resetapc-cert protocol). Four arms `{engine,ar}×{cold,on}`.

1. **APC speedup (per-turn wall):** ENGINE **1.23×** (mean-ratio 1.28×; **1.26× on within-episode-reuse turns**,
   n=47), AR **1.24×**. Engine wall **0.515 s → 0.417 s**; **prefill 0.164 s → 0.064 s = 2.58× (−0.101 s/turn)**,
   decode unchanged. Turn-0 gain small (**1.10×**, no same-episode prefix). Per-episode range **1.03×–1.35×**
   (longer episodes → bigger gain).
2. **Prefill vs decode dominance (the honest ceiling):** engine cold split = **34% prefill / 66% decode** (AR
   31/69). APC only touches prefill ⇒ **theoretical ceiling ~1.47×**, realized ~1.23–1.26×. Agentic tool-call
   turns emit **short structured outputs (~34 tok)** over 1–2.6k context ⇒ **decode-bound by construction here**.
   **But prefill-saved scales with context** (0.059 s @<1400 tok → 0.153 s @>2300 tok) — so the SWE-class
   long-context end goal is where lossless-APC actually pays off; this short-turn bench is the **conservative
   floor, not the ceiling.**
3. **Engine vs AR at matched caching** (same build, cudagraph, batch=1, **un-guided** greedy): within **~5%** on
   per-turn wall (AR/ENG on-wall **0.95×**), **per-token parity** (ENG 10.44 vs AR 10.23 ms/tok), near-identical
   total tokens (1912 vs 1923). Engine block-parallelism (1.70 tok/forward @ 17.7 ms) cancels its higher
   per-forward + FLARE prefill cost. Note: the earlier never-train "engine beats AR" used the heavier
   **grammar-guided** AR server; stripping guidance (which only slows AR — favorable to AR) makes them
   **neck-and-neck**. The engine's speed edge lives in guided-AR comparisons, longer outputs, or batch — not
   this workload.
4. **Lossless refold overhead (gate-5 envelope, UN-MEASURED):** ≤2 checkpoint crossings/episode at these lengths
   ⇒ ~1–2 refolds/episode (~0.02–0.04 s/turn amortized), trimming engine APC **~1.23× → ~1.13–1.18×**. Net
   positive; the fraction shrinks as context grows. **Gate-5 blocked on W1+W2.**

### Bottom line
Lossless-APC's value on **this** agentic workload is real but **bounded (~1.2×/turn, decode-dominated)**; its
payoff is **context-length-gated** and points at the long-context SWE end goal. The wiring gap stands: **no
lossless serving arm exists** until W1 + W2 land — only then can gate-2 (cache-on byte-parity certificate) and
gate-5 (measured refold overhead) actually run. Route A's primitives are proven bit-exact; the fix is a wiring
task, not a math task.

### Artifacts — `runs/lossless_apc/` (bench commit `b6586f0`, pushed origin/main)
`gates/gate_results.jsonl` (the ordered ledger) · `rootcause.md` · `flare_refold_parity_probe.py` /
`flare_ssm_fold_probe.py` · `bench/{engine,ar}_{cold,on}.jsonl` (57 turns each, per-turn `wall_s` +
`prefill_s`/`decode_s`) · `bench/bench_aggregate.json` (all cuts + per-episode + per-turn rows) ·
`bench/run_engine_bench.py` (v3b harness + first-denoise-step prefill timer; the `.item()` sync makes the split
accurate) / `run_ar_bench.py` / `aggregate.py` / `env_engine.sh` / `runcage_*.sh`.

---

## 1. What was built (paths + local commits)

### Repo A — vLLM pin `/home/mark/shared/vllm_p2_pr42406`
Editable-installed into `/home/mark/qwen_diffusion/.venv-vllm-p2-main`. Upstream base pinned at
`2665ed7` (PR #46838, i.e. at/after the MRV2 align-APC PR #42406 merge). **Not pushed** (no upstream
on the branch). Branch layout:
- `main` → `2665ed7` (clean upstream pin)
- `hybrid-clean-decode-policy` → `397fc98`
- `qwen3_5-flare-modelstate` → `6482e1d` (current HEAD; contains everything below)

Local commits on the flare branch:
| commit | what |
|---|---|
| `397fc98` | `[v1][sample]` Add hybrid-clean tool-call decode policy |
| `edb4d05` | `[v1][diffusion]` Qwen3_5FlareModelState: GDN-hybrid block-diffusion serving |
| `3ff71a8` | `[v1][diffusion]` Fix FLARE read_calls double-count + mamba-block-size in denoise-row derivation |
| `6482e1d` | `[v1][sample]` hybrid_clean: suppress [MASK] sentinel on value/structural tokens |

Files:
- `vllm/v1/worker/gpu/model_states/qwen3_5_flare_ops.py` — pure torch-only state-machine primitives
  (import-light, CPU-testable): `right_shift_block_logits`/`capture_shift_logit`,
  `per_seq_causal_flags`, `flare_step_and_phase`/`flare_commit_num_sampled`, `commit_num_accepted`,
  `snapshot_readonly_rows`/`restore_readonly_rows`, `FlareBoundarySnapshot` + `assert_fp32_boundary`
  + `tail_after_append`.
- `vllm/v1/worker/gpu/model_states/qwen3_5_flare.py` — `Qwen3_5FlareModelState(MambaHybridModelState)`
  + `Qwen3_5FlareRequestStates` + `Qwen3_5FlareSampler` (canvas denoise/commit). Subclasses
  MambaHybrid to inherit align-APC pre/postcopy + `num_accepted_tokens` scatter + GDN attn metadata;
  grafts the DiffusionGemma canvas/commit path.
- `vllm/model_executor/models/qwen3_5.py` — registration: `get_model_state_cls()` returns the FLARE
  state under env `VLLM_QWEN3_5_FLARE=1`, else `MambaHybridModelState`.
- `vllm/v1/sample/hybrid_clean.py` — the standalone tool-call decode-policy REFERENCE (FSM,
  forced-token bulk-commit, value/structural split, audit counters), ported verbatim from
  `scripts/eval_fastdllm_toolcall_cases.py` + `diagnose_toolcall_json_completability.py` +
  `sample_hybrid_clean`. **See §2: this is NOT on the serving path.**
- Tests: `tests/v1/worker/gpu/test_qwen3_5_flare_state_machine.py` (17 pure-CPU ops tests),
  `tests/v1/sample/test_hybrid_clean.py` (20 pure-CPU tests incl. the [MASK]-sentinel regression).

### Repo B — qwen_diffusion `/home/mark/qwen_diffusion`
Remote `origin` = `MaCoredroid/qwen-diffusion-agentic`. On `main`, **2 commits ahead of origin/main**
(the harness base `ddfa895` is pushed; the audit-counter fixes are not):
| commit | pushed? | what |
|---|---|---|
| `ddfa895` | yes | P2 parity+audit harness: HF hybrid-clean reference vs new FLARE engine |
| `782b441` | **no** | P2 parity harness: close audit-counter tautology + gate FSM value-projection leak channel |
| `1e73790` | **no** | Add REPRODUCE_V2 (dual-mode AR+diffusion recipe; adjacent) |

Files:
- `scripts/parity_audit_flare_engine.py` — the M1/M2 promotion-gate instrument. Four modes:
  `selftest` (15/15, tokenizer-only), `ops-parity` (15/15, imports the REAL engine ops and proves
  byte/numeric equivalence to `flare_hf_cache`), `state-parity` (tiny route_i model, 4 gates), `turn`
  (GPU; one matched-20 turn, HF reference vs vLLM engine, 6 gates).
- `scripts/p2_vllm_smoke.py` — **untracked** smoke driver with three cases (`diffusiongemma`,
  `qwen-default`, `qwen-align-apc`); drives the §3 steps 2-3. Will be committed alongside this doc.

### Repo C — flywheel fork `/home/mark/shared/lumoFlyWheel_codex_fork`
Remote `fork` = `MaCoredroid/Lumo_FlyWheel-qwen-diffusion`. HEAD `b91184d0`. **Not pushed.**
| commit | what |
|---|---|
| `00b72352` | Wire P2 hybrid-diffusion serving surface for Qwen3.5-9B |
| `b91184d0` | flare-hybrid launcher: set the real engine gate `VLLM_QWEN3_5_FLARE` |

Files:
- `scripts/qwen35_9b_flare_hybrid_serve.sh` — host vLLM launcher pointed at the pin venv
  `.venv-vllm-p2-main/bin/vllm`, serving `models/qwen3.5-9b-fastdllm-b1000-vllm-bf16` as
  `qwen3.5-9b-flare-hybrid-clean`. Now derives+exports `VLLM_QWEN3_5_FLARE` from `DECODE_POLICY`.
- `model_registry.yaml` — entry `qwen3.5-9b-flare-hybrid-clean` (validated through `load_registry`).
- `docs/qwen3.5-9b-flare-hybrid-serving-note.md` — MTP coexistence + shared align prefix cache note.

---

## 2. Review verdicts + open issues

**All four reviews: `fix-needed`.** What was fixed in-loop, then the open issues that remain — the
GPU-only-unvalidatable crux, the architectural gaps, and the perf/util debt.

### Fixed in-loop (already committed)
- **read_calls double-count** + **mamba-block-size in denoise-row derivation** (`3ff71a8`): non-align
  branch was indexing the MAMBA block table with the attention `block_size`, pointing snapshot/restore
  at the wrong state rows. Now uses `mamba_spec.block_size`. read_calls no longer counted in the
  sampler's `_apply_shift`.
- **[MASK]-sentinel suppression** in hybrid_clean (`6482e1d`): reference decoder unconditionally does
  `logits[:, mask_id] = -inf` every forward; the port had dropped it, so a parameter VALUE token could
  argmax to the mask id and diverge. Restored + regression test (20/20).
- **Gate-#3 tautology** in the parity harness (`782b441`): `two_wave_wave1_projected_tokens` /
  `parallel_commit_forced_tokens` were hard-coded to 0, so the "ZERO projected values" gate could
  never fail on a projecting engine. Real counters now plumbed through; added `grammar_value_projection`
  gate + 5 selftest regression guards (15/15), incl. one that would have wrongly passed before the fix.
- **Launcher silent-AR gate mismatch** (`b91184d0`): launcher exported only `FLARE_DECODE_POLICY`
  (zero consumers in the pin) → it silently served the AR `MambaHybridModelState` under the diffusion
  model name. Now derives+exports `VLLM_QWEN3_5_FLARE`.

### Open — THE CRUX (GPU-only, unvalidatable off-GPU; this IS the M1 go/no-go)
- **GDN read-only-denoise state discipline** (`qwen3_5_flare.py` `_denoise_state_rows`/`_gdn_caches`,
  ~L389-481). Flagged by **all four** reviews. Correctness of the read-only-denoise restore hinges on
  this reconstructing EXACTLY the physical conv+ssm rows the GDN kernel writes:
  - It reconstructs rows independently (`_mamba_state_idx_gpu[denoise_slots]`) rather than reading the
    gdn_attn backend's actual `non_spec/spec_state_indices_tensor`. If the running-row vs
    block-table-slot or spec-decode index layout diverges, the snapshot protects the WRONG rows and
    denoise permanently corrupts the boundary `S_t`/conv state = the fatal, silent failure.
  - `_gdn_caches()` classifies ANY module whose `.kv_cache` is a 2-tuple of nonempty tensors as GDN.
    A full-attention layer whose kv_cache is also 2-length would be misclassified and its K/V
    snapshot/restored as conv/ssm (silent). No assertion pins layer count/identity/shape.
  - **Silent-fatal path:** `preprocess_state` does `if not caches: return` (no snapshot) and
    postprocess then skips restore, so an in-place kernel write-back of tentative denoise tokens leaks
    into `S_t`/conv with NO error. Should hard-fail when readonly is enabled AND denoise rows exist AND
    caches is empty.
  - **Postprocess ordering:** restore runs FIRST, then `super().postprocess_state` scatters
    `num_accepted=max(num_sampled,1)=1` and runs the align block-row publish over ALL rows including
    denoise. "Denoise advances GDN state by 0" relies on `num_computed` not advancing so the align copy
    is a no-op; if a denoise row is ever non-block-aligned the align kernel re-copies and clobbers the
    just-restored boundary.
  - NOT proven by the 17 CPU tests (which cover only the unused pure ops). This is exactly the M1
    day-1-2 read-only-denoise probe (§3 step 4).

### Open — architectural / semantic-drift (turn-parity risk)
- **Orphaned FSM** (reviews 2 & 4): `vllm/v1/sample/hybrid_clean.py` is imported by NOTHING in `vllm/`
  (grep-confirmed) and referenced nowhere in the FLARE ModelState/Sampler. The served path is
  `Qwen3_5FlareModelState.custom_sampler → Qwen3_5FlareSampler`, a plain canvas denoiser with zero
  grammar/tool/FSM/value logic. `decode_policy=hybrid_clean` is a name collision between two unrelated
  mechanisms (hybrid_clean = HF masked-diffusion with a `[MASK]` token; FLARE = canvas/random-renoise,
  no mask token — cannot be trivially merged). Every value-projection / FSM / "zero value projection"
  guarantee the serving note advertises is **off the actual serving path**. Wiring
  `parse_hybrid_clean_request`/`HybridCleanDecodePolicy` into a real decode scheduler is unimplemented.
- **Shifted-logit capture uses the wrong stream** (reviews 1, 2, 3): engine `capture_shift_logit`
  (`qwen3_5_flare.py` ~L762-773) captures `block_logits[:,-1:]` from the converging DENOISE step (a
  forward over the pre-freeze/renoised canvas), whereas the reference `flare_hf_cache.advance()`
  re-runs `cached_noisy_block_logits` over the COMMITTED (argmax) clean block. The carried
  position-0 logit can drift. The +1 right-shift DIRECTION itself is correct. state-parity/ops-parity
  cannot detect this off-GPU — only `--mode turn` on the real engine can. This is R5 semantic drift:
  regression to the 0/41 corruption regime is silent without the gate.
- **Bidirectional-denoise-on-GDN is the unproven premise** (review 1): per-seq `causal=False` only
  affects full-attention layers; GDN linear-attn ignores causal entirely — its "bidirectional block
  read" is realized ONLY by the snapshot/restore recurrence. Whether that reproduces training's
  bidirectional-block GDN semantics is the novel unvalidated claim. `rswa_prefix_lens` passed together
  with per-seq `causal=False` is also unvalidated.
- **Harness self-contradictions vs the real engine** (review 3): `compare_snapshot_sequences` pass
  condition requires `not only_ref and not only_eng`, but the reference records prefill boundaries
  (fresh FlarePrefixCache each turn) while the real engine exposes commit-only boundaries, so
  `state_snapshot_equality` will ALWAYS fail via `only_ref`. Gate #2's `reported_model_value_tokens`
  clause: the engine has no such counter, so a byte-identical engine yields 0 vs N and gate #2
  spuriously FAILS. Both are harness design decisions to resolve on-GPU (see §3 step 5 kill note).

### Open — audit theater (counters that can't catch a regression)
- `force_projected_value_tokens` (`# must stay 0`) and `residual_full_context_model_calls`: initialized
  to 0, NEVER incremented, NEVER asserted → dead.
- `advance_calls` increments on EVERY tensor postprocess (denoise AND commit), not commit-only. Since
  denoise dominates, read_calls ≈ advance_calls, so the read/advance ratio — **the ~13x-fewer-forwards
  ⇒ wall-clock-win thesis metric and the M2 gate** — collapses to ~1. Left unfixed: the commit signal
  is only cheaply available in the sampler, not `postprocess_state`; needs a design decision.
- `hybrid_clean.py` `value_projection_events` is never incremented on the live path, so
  `verify_invariants()`'s `assert value_projection_events==0` is tautological. The other two invariants
  are guaranteed by the loop structure. Reporting "0 value projections" as evidence is theater.
- **Dead ops vs commit-message claims:** `commit_num_accepted`, `FlareBoundarySnapshot`,
  `assert_fp32_boundary`, `tail_after_append` are defined + unit-tested but called NOWHERE in the live
  path (0 uses each). The fp32-boundary + raw-conv-tail publish is delegated wholesale to inherited
  align `postprocess_state` + `--mamba-ssm-cache-dtype float32`; the load-bearing integration is
  untested and the "17/17" covers only the unused pure ops.
- `route_i` guard is declarative-only: stock Qwen3.5 never exposes `flare_gdn_route`, so `_assert_route`
  always falls to the env default and the GROUND invariant is never checked against served semantics.

### Open — perf / GPU-utilization debt (violates the standing util rule)
- `Qwen3_5FlareSampler._gather_block_logits` (L639) and `_apply_shift` (L656) run per-decode-row Python
  loops with `.tolist()`/`bool(...)` host syncs + per-call `async_copy_to_gpu` allocations on the hot
  path every step. The proven DiffusionGemma path vectorizes the identical gather/pad sync-free
  (`diffusion_gemma.py` L1269-1274). Caps GPU util under batching; a CUDA-graph blocker. Not
  incorrectness — deferred to P2.2+/M3 (§4).
- If hybrid_clean were ever wired: `HybridCleanGrammar._keeps_prefix` does a full `tokenizer.decode` of
  the ENTIRE prefix for EVERY candidate EVERY step (O(prefix × candidates)) — not viable at serving
  latency.

### Open — config / docs hazards
- **Block/chunk misalignment:** engine default `_DEFAULT_BLOCK=32` is HALF a GDN chunk (FLA_CHUNK 64).
  A stock export has no `diffusion_config`, so 32 is what runs; commit boundaries land mid-chunk on
  every other block, and the fp32 boundary snapshot `chunk_states[:,-1]` mid-chunk is not a clean
  recurrent checkpoint. Trained `canvas_length` must be a multiple of 64; set
  `VLLM_QWEN3_5_FLARE_BLOCK` accordingly.
- **Per-request mode switching not wired:** registration is process-global via `VLLM_QWEN3_5_FLARE=1`
  in `get_model_state_cls`, not per-request `extra_args["decode_mode"]` (fr10). AR and block-diffusion
  cannot coexist in one server; every request is forced through the diffusion ModelState.
- **Stale docs:** serving note says `FLARE_DECODE_POLICY` selects the sampler; the real selector is
  `VLLM_QWEN3_5_FLARE`. The note also conflates KV `block_size` (~16) with `mamba_block_size` (1024) in
  the `max-num-batched-tokens` reasoning.

### Verified-clean (for the record)
- The **+1 right-shift is mathematically correct** (`right_shift_block_logits`/`capture_shift_logit`
  mirror the reference `shifted_active_logits`/`advance`; block_start==0 self-prepends noisy[:,:1]).
  Confirmed by 17 CPU state-machine tests + 20 hybrid_clean tests (all pass).
- All vLLM-pin CLI/env used by the launcher are valid in the pinned build (`--attention-backend
  TRITON_ATTN`, `--gdn-prefill-backend triton`, `--mamba-cache-mode align`, `--mamba-block-size`,
  `--mamba-ssm-cache-dtype float32`, `VLLM_USE_V2_MODEL_RUNNER`, `SamplingParams.extra_args`,
  `get_model_state_cls`). No API misuse found.
- **Caveat on test coverage:** torch was NOT installed in the review environment, so the 17 FLARE
  state-machine tests and ALL GPU-path behavior could not be executed there — only the 20 pure-Python
  hybrid_clean tests ran. FLARE ops correctness currently rests on code reading. Re-run under the pin
  venv on the GPU box (§3 step 1).

---

## 3. THE GPU SMOKE CHECKLIST (run in order, the moment the GPU frees)

This is the M1 day-1-2 fail-fast gauntlet, front-loaded exactly as `p2_serving_reuse_plan.md` §4
prescribes, extended through the M1 turn gate (step 5) and the M2 matched-20 battery (step 6). Each
step lists its **pass criterion** and its **kill criterion** (from plan §5). Do not proceed to the
next step until the current one passes.

Environment for every step: `VLLM_USE_V2_MODEL_RUNNER=1`, `VLLM_ATTENTION_BACKEND=TRITON_ATTN`,
python/vllm from `/home/mark/qwen_diffusion/.venv-vllm-p2-main`.

### Step 1 — pin-venv sanity (precursor)
- **Do:** confirm `.venv-vllm-p2-main/bin/vllm` imports vLLM from `/home/mark/shared/vllm_p2_pr42406`
  (editable), the flare branch is checked out (HEAD `6482e1d`), `torch.cuda.get_device_capability()`
  reports **sm_120 = [12, 0]**, and `VLLM_USE_V2_MODEL_RUNNER` is honored. Then re-run the CPU test
  suites under this venv (they could not run in the review env — no torch there):
  `pytest tests/v1/worker/gpu/test_qwen3_5_flare_state_machine.py` (17) and
  `pytest tests/v1/sample/test_hybrid_clean.py` (20), plus
  `scripts/parity_audit_flare_engine.py --mode selftest` (15), `--mode ops-parity` (15),
  `--mode state-parity` (4 gates).
- **Pass:** editable install confirmed, sm_120 detected, all 17+20 unit tests + selftest/ops-parity/
  state-parity green under the real torch.
- **Kill:** none — this is setup. If it fails, fix before proceeding; do NOT burn the fail-fast budget
  on a broken venv.

### Step 2 — DiffusionGemma smoke on sm_120  (plan §4 M1 (a))
- **Do:** `python scripts/p2_vllm_smoke.py diffusiongemma --out logs/smoke_diffusiongemma.json`
  (NVFP4 DiffusionGemma-26B, `quantization=modelopt`, `attention_config.backend=TRITON_ATTN`,
  `diffusion_config` canvas 32 / 4 steps).
- **Pass:** loads + generates a coherent short sentence, `status=PASS`. Proves the first-party dLLM
  decode path (canvas draft tokens, per-seq causal Triton kernel, commit machinery) runs on THIS card.
- **Kill K2 (R2 sm_120 attention):** if the DiffusionGemma smoke fails after **2 days of backend
  fallbacks** → drop to **A2 (0.23 V1-runner hook injection)**; its bidirectional mask would be our
  Triton kernel either way. Carry the ModelState design as our own seam spec.

### Step 3 — our export under MRV2, default then align+APC  (plan §4 M1 (b))
- **Do:** `python scripts/p2_vllm_smoke.py qwen-default --out logs/smoke_qwen_default.json` then
  `python scripts/p2_vllm_smoke.py qwen-align-apc --out logs/smoke_qwen_align_apc.json`
  (`models/qwen3.5-9b-fastdllm-b1000-vllm-bf16`; align case adds `enable_prefix_caching`,
  `mamba_cache_mode=align`, `mamba_block_size=1024`, `mamba_ssm_cache_dtype=float32`). Tests whether
  #38041 (MRV2×GDN broken) is stale and whether #42406 (align-APC for mamba) holds.
- **Pass:** Qwen3.5-9B forwards under MRV2 in BOTH configs, coherent generation, `status=PASS` both;
  align+APC config loads without the mamba-cache assertion tripping.
- **Kill K1 (R1 MRV2×GDN broken):** if Qwen3.5 cannot forward under MRV2 within **5 working days** of
  fixes/upstream triage → drop to **A2 (0.23 V1 injection)**, carrying the ModelState design as our
  seam spec.

### Step 4 — read-only-denoise probe  (plan §4 M1 (c) — the go/no-go artifact)
- **Do:** the ~20-line probe: forward the SAME block twice from a fixed `initial_state` with the
  denoise-phase flag on (`inplace_final_state=False` + `_denoise_state_rows` snapshot/restore + conv
  write suppression), diffing the conv/ssm slots and the logits between the two forwards. This is the
  live-fire test of the §2 GDN-state-discipline crux — it validates `_gdn_caches` classification (no
  full-attn layer misclassified), that the snapshot protects the exact physical conv+ssm rows the fla
  kernel writes, and that denoise leaks NOTHING into the boundary `S_t`/conv.
- **Pass (go):** for denoise rows, `conv_state` and `ssm_state` slots are **bit-identical** before vs
  after the denoise forward, and the two forwards produce **identical logits** — i.e. denoise advances
  GDN state by exactly 0. Emit the go/no-go artifact (diff report).
- **Kill:** this probe is the M1 go/no-go and a **precondition of the K3 thesis-level gate** ("with
  read-only O(block) denoise verified"). If denoise corrupts `S_t`/conv and it cannot be fixed within
  the R1 window → the MRV2×GDN read-only-denoise premise is unworkable on this substrate → drop to
  **A2 (K1)**. Also gate against **R5**: if the snapshot protects the wrong rows, the fix must be
  proven here, not deferred — this failure is silent downstream.

### Step 5 — parity harness turn: engine vs HF byte-match  (plan §4 M1 gate)
- **Do:** `python scripts/parity_audit_flare_engine.py --mode turn --engine vllm` on one matched-eval
  turn (temp 0, native chat_template, greedy per-call waves, FSM stub = leftmost-forced only). Boots
  the vLLM V2 runner with `VLLM_QWEN3_5_FLARE=1` and drives (a) the HF hybrid-clean reference and
  (b) the FLARE engine over the same turn.
- **Pass:** all six turn-mode gates green — `byte_identical` (token AND byte exact vs the HF
  reference), `value_token_counts_equal`, `reference_zero_projected_values`,
  `engine_zero_projected_values`, `no_grammar_value_projection`, `state_snapshot_equality`. Plus the
  plan M1 targets: forwards/turn ≈ **6-9** and s/turn already **< HF's 1.442**. This is where the
  §2 shifted-logit-capture divergence surfaces if real — off-GPU modes cannot detect it.
- **Harness caveats to resolve here (review 3, design decisions, not silent relaxations):**
  `state_snapshot_equality` will fail via `only_ref` because the reference records prefill boundaries
  while the real engine exposes commit-only boundaries — decide whether to compare the shared set only.
  Gate #2's `reported_model_value_tokens` clause spuriously fails a byte-identical engine that lacks
  the counter — either emit `model_value_tokens` from the engine adapter or drop that redundant clause
  (the XML-derived `value_token_count` already covers it).
- **Kill (R5 semantic drift):** if turn output cannot be made byte-identical to the HF reference, the
  port has drifted (shifted-logit capture or conv_tail seam) — equivalence-gate every stage against
  `flare_hf_cache.py`, assert value force-counters 0. Regression to the 0/41 corruption regime is
  silent without this gate. If unfixable → re-scope; do not advance to step 6.

### Step 6 — matched-20 battery on the engine path  (plan §4 M2 gate)
- **Do:** rerun the hashed slice (`baf90863`, 20 ep / 63 turns) on the FLARE engine path with the full
  offline FSM + wave-1/wave-2 wiring + cross-turn APC counters, **against guided-AR re-baselined on the
  SAME pinned build** (identical align-APC flags — R6 fairness). Trigger-test the align-APC
  pathologies (#40696 / #45238 / #43587) on our multi-turn prompt shapes; apply the chunked lm_head cap.
- **Pass (M2 engine-promotion gate, all of):** the **engine quality gate is PARITY with the HF
  hybrid-clean row** — the engine runs the same weights + same algorithm, and Step-5 byte-parity
  *implies* the same score, so the target is **ENGINE == HF row (47/63 exact-args, 13/20 episodes,
  63/63 exact_seq, 63/63 valid_xml, value force-counters == 0)**, NOT a higher model-quality number.
  Plus the speed target **< 1.120 s/turn** and **force-counters == 0 on values**. (Prerequisite: fix
  the `advance_calls` counter so the read/advance ratio is real before quoting the forwards-saved
  metric.) **Note:** the **55/63 / 15/20** figure is the *K3 thesis aspiration* for the diffusion
  model's raw quality — a stronger, model-training target that gates the overall thesis, **not** the
  engine's parity gate. Promoting the engine only requires reproducing the promoted HF row byte-for-byte
  at `< 1.120 s/turn`; lifting the model to 55/63 is a separate, training-side milestone.
- **Kill K3 (thesis-level):** if at M2 — with read-only O(block) denoise verified (step 4) and, after
  M3, graphs on — diffusion still **misses 1.120 s/turn by > 20% at healthy GPU util**, the
  ~13x-fewer-forwards ⇒ wall-clock-win thesis fails on this hardware. **Stop, publish the profile,
  re-scope** (kernel-level work, or accept the quality-only win). No sunk-cost continuation past K3.

---

## 4. What remains for P2.2+ (after M1 passes)

### Wall-clock measurement vs re-baselined guided-AR (R6 fairness — do this before any speed claim)
Moving engines (0.23 → post-0.24 main pin) **invalidates the existing 1.120 s/turn number**. Before
claiming a diffusion win, re-baseline **guided-AR on the pinned build** with the identical align-APC
flag set, same hashed slice, same engine. The quality caveat stands (N=20, single seed, synthetic tool
results). Only a same-engine A/B is admissible under the promotion discipline.

### Batching + engine-grade per-forward cost (plan §4 M3)
- **Remove the host-bound hot path** (standing GPU-util rule): vectorize `_gather_block_logits` /
  `_apply_shift` sync-free like `diffusion_gemma.py` L1269-1274 — eliminate the per-row `.tolist()` /
  `bool(...)` syncs and per-call `async_copy_to_gpu` allocations. These are CUDA-graph blockers.
- **CUDA-graph capture:** we currently run `--enforce-eager` — this is where the remaining headroom
  lives. dInfer recipe: control-flow-free wave logic (no `.item()`/`tolist()`), shape-bucketed graph
  capture bs=1 first, then multi-seq waves via UNIFORM_BATCH-style padding.
- **Gate:** GPU util healthy per the standing rule (profile, no host-bound stalls); target the honest
  ~2-3x blended band vs AR at held quality.

### Engineering debt to clear at/before P2.2 (from the reviews)
1. **Wire the FSM onto the serving path** (or explicitly drop the "hybrid_clean on the engine" framing):
   today the value-projection / FSM / "zero value projection" guarantees live only in the standalone
   `hybrid_clean.py` reference, not in `Qwen3_5FlareSampler`. Either integrate
   `HybridCleanDecodePolicy` into the FLARE custom_sampler or reconcile the two diffusion paradigms
   (masked-diffusion vs canvas/renoise).
2. **Fix `advance_calls`** to increment commit-only (needs the commit signal surfaced from the sampler)
   so the forwards-saved metric — the whole thesis KPI and the M2 gate — is real.
3. **Wire or delete the dead audit machinery:** `force_projected_value_tokens`,
   `residual_full_context_model_calls`, `commit_num_accepted`, `FlareBoundarySnapshot`,
   `assert_fp32_boundary`, `tail_after_append`. If the fp32-boundary/conv-tail publish is truly
   delegated to inherited align postprocess + `--mamba-ssm-cache-dtype float32`, prove that integration
   (it is currently untested) or route it through the primitives.
4. **Hard-fail the silent-fatal GDN paths:** assert layer count/identity/shape in `_gdn_caches`; error
   (not `return`) when readonly is enabled with denoise rows but no caches found.
5. **Enforce block/chunk alignment:** engine default block 32 → multiple of FLA_CHUNK 64; set/validate
   `VLLM_QWEN3_5_FLARE_BLOCK`; ensure trained `canvas_length` is a multiple of 64 so boundary snapshots
   land on clean recurrent checkpoints.
6. **Per-request mode switching:** honor `extra_args["decode_mode"]` so AR and block-diffusion coexist
   in one server, instead of the process-global `VLLM_QWEN3_5_FLARE=1` switch.
7. **Fix stale docs** in `docs/qwen3.5-9b-flare-hybrid-serving-note.md` (`FLARE_DECODE_POLICY` →
   `VLLM_QWEN3_5_FLARE`; the KV-block vs mamba-block conflation).

### Push discipline
Nothing is pushed to the shared forks yet. Per the standing commit workflow, once step 5 (M1 turn
gate) passes on-GPU, push the flare branch to the vLLM pin fork and the serving surface to the flywheel
fork, each with narrated reasoning. This status doc is committed+pushed to `qwen-diffusion-agentic`
now as the pre-GPU checkpoint.
