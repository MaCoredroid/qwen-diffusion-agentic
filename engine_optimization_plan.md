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

## Bars and the current budget

| row (matched-20 reference, `runs/endgame_scoreboard`, NOT the engine) | s/turn | fwd-or-tok/turn |
|---|---|---|
| OUR HF hybrid-clean (v2) | 3.904 | 56.83 denoise fwd/turn |
| stock-bf16-AR-guided | 1.213 | 82.24 tok/turn |
| **stock-AR aggregate** | **0.741** | 49.06 tok/turn |

- **M2 / K3 target:** **< 1.120 s/turn** (just under guided-AR). — the P0 bar.
- **Beyond-AR / thesis KPI:** **< 0.741 s/turn** at equal quality via
  forwards-saved. — the P2 bar.
- There is **no honest engine wall-clock yet**: the engine path is blocked at
  byte-parity (GAP 5A), so no valid matched-20 turn can be driven. The only
  diffusion number is the HF stack's 3.904 s/turn. **GAP 5A is therefore both a
  correctness gate and this plan's single largest forward-compute waste** — the
  reports and the build-status doc converge on the same root cause (OPT-3).

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

| Rank | Opt | Expected gain (per turn) | Effort | Payoff/effort | Tier |
|---|---|---|---|---|---|
| 1 | **OPT-1** GPU-native sampling | **~1.3–2.3 s** (host, MEASURED-scaled) | M | **Highest** | P0 |
| 2 | **OPT-2** graph guard + `cg_mode` log | avoids contingent **+0.2–0.6 s** eager fallback; enables honest measurement | **S** | Very high | P0 |
| 3 | **OPT-3** variable single-[MASK] width | correctness gate (unblocks M2 at all) + **~2–6×** forward-compute cut | M–L | High | P0 |
| 4 | **OPT-5** incremental detok/FSM | **~15–75 ms**, O(n²)→O(n) (grows w/ turn len) | L | Medium | P1 |
| 5 | **OPT-4** incremental KV/GDN 1-wide decode | closes residual GPU gap to AR (recurrent kernel + full graph) | L | Medium | P1 |
| 6 | **OPT-6** multi-token bulk commits | fewer forwards than AR → sub-0.741 s | M–L | Beyond-AR | P2 |

Numbers are UNVERIFIED for GPU components; the host components of OPT-1/OPT-5 are
MEASURED-CPU and re-scaled. The ranking is robust to the constants: OPT-1's host
cost alone exceeds the entire 1.120 s budget, and OPT-3 is a hard prerequisite
for *any* engine number, so those two dominate regardless.

---

# P0 — Needed to hit < 1.120 s/turn (M2)

Without all three of these there is either (a) no valid turn to measure (OPT-3),
(b) a host cost that alone blows the budget (OPT-1), or (c) an unguarded eager
fallback that silently blows it (OPT-2).

## P0-A · OPT-1 — Move full-vocab sampling onto the GPU

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

## P0-C · OPT-3 — Variable single-[MASK] forward width (also the GAP-5A fix)

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

## P1-A · OPT-4 — Incremental KV+GDN commit → true 1-token decode kernel

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

## P1-B · OPT-5 — Incremental detokenization + resumable FSM cursor

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

1. **OPT-2 (P0-B, S)** — land the `cg_mode` counter + fail-closed config assert
   *first*, so every subsequent GPU-verification number is trustworthy and a
   silent eager fallback cannot masquerade as a real bottleneck.
2. **OPT-3 (P0-C, M–L)** — variable single-[MASK] width. It is the correctness
   gate (unblocks the entire M2 battery), the largest forward-compute cut, and a
   hard prerequisite for OPT-4. Nothing downstream is measurable at quality
   without it.
3. **OPT-1 (P0-A, M)** — GPU-native batched sampling (incl. OPT-1b). The single
   biggest number (~1.3–2.3 s/turn) and the item that actually puts the turn
   under 1.120 s. Develop CPU-side in parallel with steps 1–2; GPU-verify on the
   first valid turn that OPT-3 produces. **Expected checkpoint: < 1.120 s/turn
   (M2 met).**
4. **OPT-5 (P1-B, L)** — incremental detok/FSM: the next host bottleneck once
   OPT-1 is gone; removes the O(n²) grammar cost.
5. **OPT-4 (P1-A, L)** — incremental KV/GDN 1-token decode: takes the forward
   from 1-wide-prefill-classed to AR-identical (recurrent kernel + FULL graph).
   **Expected checkpoint: ~0.741 s/turn (AR parity).**
6. **OPT-6 (P2-A, M–L)** — confidence-based multi-token bulk commits: amortize
   forwards below AR. **Expected checkpoint: < 0.741 s/turn (beyond-AR).**

**One-line rationale:** instrument (2) → unblock correctness + cut the forward
width (3) → kill the dominant host cost (1) to clear M2 → remove residual host
O(n²) (5) and residual GPU shape (4) to reach AR parity → amortize forwards (6)
to beat AR.
