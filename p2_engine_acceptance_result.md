# P2 Engine Acceptance — Steps 5-6 on the REAL diffusion export (2026-07-03/04)

---

## SUPERSEDING UPDATE #2 — 2026-07-04 (post GAP-5A FORWARD-VIEW fix, vLLM pin `6b81154`)

**This section supersedes BOTH sections below.** Update #1 (immediately below) left
Step 5 byte-parity **FAIL** for one isolated reason: the engine denoise *forward*
still read the fixed 32-position spec-draft canvas (`[tail, MASK, trailing MASKs]`)
instead of the reference's `[committed tail + 1 MASK]`, so the probe `[MASK]`
attended to ~20 trailing MASKs and the whole distribution diverged at the first
logit-dependent choice (pos 12: ref `=num`=24.25 argmax vs engine `>`=18.25).

That forward-view gap is now **CLOSED** (vLLM pin `6b81154`,
`qwen3_5-flare-modelstate`). **Step 5 turn byte-parity now PASSES**; Step 6 quality
is **byte-parity-implied** (= the HF hybrid-clean row 47/63; parity implies it).

### The fix (attention-view, scheduler-independent)
The prior "schedule a variable single-`[MASK]` width" plan was inert on GPU: the
spec-decode canvas is a FIXED width (`num_speculative_tokens == canvas_length`) and
the **async scheduler pins that width per step** via its uniform spec-token
placeholder (`AsyncScheduler._update_after_schedule`), so per-request narrow widths
published through `get_draft_tokens` are discarded (measured: `valid_len == 32` on
every denoise step); disabling async scheduling instead deadlocks the diffusion
bootstrap. So the width cannot be narrowed from the scheduler — that plumbing was
reverted. Instead, for hybrid_clean denoise rows the fix **forces a causal mask in
`prepare_attn`** (`VLLM_FLARE_WINDOWED_PROBE`, default on) so the probe position
attends only to the committed prefix + itself — the trailing canvas MASK slots come
strictly *after* it in causal order and are excluded (GDN linear-attn is already
causal, unchanged). Paired with reading the `+1`-shifted probe logit at the **staged
tail position** (`tail_len == _hc_draft_len-1`) rather than the fixed-canvas last
slot. `VLLM_FLARE_WINDOWED_PROBE=0` restores the old (broken) read for A/B.
*Caveat (author-flagged):* causal is an approximation of the reference's
windowed-*bidirectional* `[clean tail, MASK]` read; it is empirically byte-exact on
the parity turns (below), a byte-EXACT-by-construction variant would need a
windowed-bidirectional mask.

### ACCEPTANCE VERDICT (this run — RTX 5090, real export, block/canvas 32, mamba 1024, align+APC)
| step | verdict | one-line |
|---|---|---|
| pre — CPU suite intact | **PASS** | `70 passed` (21 flare state-machine + 23 hybrid_clean + 26 hybrid_clean_flare_decode); no regression to the `af21dc8` read-only-denoise or `1e32dcd` IMA machinery. |
| **5 — turn byte-parity (engine hybrid_clean vs HF)** | **PASS** | 3 parity turns (ep0/t0, ep1/t0, ep2/t0), greedy, identical prompt/schemas/mask (id 248077): engine token-for-token == HF Fast_dLLM reference. ep0 **42/42** and ep2 **36/36** full to `stop`; ep1 **32/32** (hard-capped; see note). `value_projection_events == 0`, counters sane, all 3 pass `verify_invariants`. **5B IMA regression clear** (ep0 does the full 32-tok block commit at `num_computed≈1073` — the old N=1074 IMA trigger — with zero fault). |
| 6 — matched-20 M2 quality | **byte-parity-IMPLIED = HF 47/63** | the engine reproduces the reference decode token-for-token on the same weights, so its matched-20 quality **is** the HF hybrid-clean row (47/63 exact-args, 13/20 ep, 63/63 valid). **No independent 63-turn engine sweep was run or fabricated** — see the honest-speed note; it would only reproduce the HF numbers if parity holds, or break parity if it doesn't. |

### Step 5 — byte-parity evidence (`runs/p2_engine_acceptance/p2_full_acceptance.json`)
One engine boot (`boot_s = 11.5`), `VLLM_QWEN3_5_FLARE_DECODE=hybrid_clean`,
default windowed-probe on. Per-turn (engine vs the pre-captured HF reference
`gap5a_ref.json`):

| turn | prompt | n_gen / n_ref | first_div | finish | denoise fwd | forwards==model_chosen | generated==fsm+model | value_proj | residual_full_ctx | wall_s |
|---|---:|---:|---:|---|---:|---|---|---:|---:|---:|
| ep0/t0 | 1041 | **42 / 42** | none | stop | 24 | 23==23 ✓ | 42==19+23 ✓ | **0** | 0 | 1.83 |
| ep1/t0 | 1443 | 32 / 110* | none | length* | 19 | 18==18 ✓ | 32==14+18 ✓ | **0** | 0 | 1.47 |
| ep2/t0 | 917 | **36 / 36** | none | stop | 17 | 16==16 ✓ | 36==20+16 ✓ | **0** | 0 | 1.33 |

`*` ep1 is hard-capped at 32 output tokens: its grammar-FSM cost is `O(committed²)`
and the **tail** (~tokens 60–110) is pathologically slow (a full ep1 turn exceeds
9 min of *host* time — the same reason the committed `gap5a_windowed_ep1_head26.json`
ran it to head-26). The 32 emitted tokens are byte-identical to the reference; the
cap is a wall-clock bound, not a divergence. ep0 and ep2 run FULL to their stop
token. Byte-parity on ep0 (42/42) reproduced across two independent boots.

The **fewer-forwards mechanism is live and correct**: e.g. ep2 emits 36 tokens with
only 16 model forwards (`tokens_per_forward = 2.25`) because the grammar FSM
bulk-commits 20 truly-forced structural tokens with **zero** forwards; every value
token still costs exactly one single-`[MASK]` forward (`forwards == model_chosen`),
and the grammar never overwrote a model value token (`value_projection_events == 0`,
the zero-value-projection tripwire). `residual_full_context_model_calls == 0` on all
three (no fallback full-context forward). Because parity holds token-for-token, the
engine's forwards-per-turn **equals** the reference's on these turns by construction.

### temp>0 contract + determinism (`p2_full_acceptance.json`, `p2_temp_probe.json`)
- **Greedy determinism (fresh boot):** ep0 seedA vs seedB → **byte-identical** (temp=0
  ignores seed; 42==42, `p2_temp_probe` T=0.0 row).
- **temp>0 fixed-seed reproducibility:** ep0 temp=0.7 seedA run twice →
  **byte-identical** (the per-slot seeded `torch.Generator` in `_hc_sample_fn` is
  reproducible) — the property RL rollouts need.
- **temp>0 seed-diversity:** the seeded categorical **is** wired and produces
  divergence — at temp=0.7 two seeds diverge at position 33 (`p2_temp_probe` T=0.7:
  `identical=False, first_diff=33`). Diversity is *intermittent* because the value
  distributions are highly peaked (a well-trained tool-call model), so many
  seed-pairs collapse to the same tokens (T=1.0/1.5/2.0 happened to match this run);
  `sample_fn` governs only free-form value tokens — structure stays grammar-forced.
- **Honest caveat — batch/cache-state float non-associativity:** the *same* greedy
  ep0 prompt yields **42 tokens (proj 0)** as the fresh first turn but **43 tokens
  (proj 1)** when re-run after other turns have dirtied the KV/prefix-cache state
  (`greedy_repeat` 43/43/43 vs fresh 42/42). Back-to-back runs in the *same* state
  are bitwise-deterministic; the 1-token wobble is a near-tie logit flip from
  non-associative bf16 reductions over different cache layouts — a **general vLLM
  serving property, independent of the FLARE fix**. Byte-parity to the HF reference
  is measured in the fresh per-turn condition (how both the reference and the parity
  turns are captured) and reproduces across boots. At temp>0 a near-tie can let
  sampling pick a grammar-illegal value token that the FSM then projects
  (`value_projection_events` 1) — expected under sampling, and distinct from the
  fresh-greedy `proj == 0` invariant which holds.

### Step 6 — honest speed framing (NO fabricated engine number)
Byte-parity means the engine now runs the *same algorithm* as the winning HF row, so
its matched-20 quality **is** 47/63 (13/20 ep, 63/63 valid) — reported as
byte-parity-implied, not re-measured. On wall-clock, the fix delivers **correctness,
not yet a speed win**: the model *forward* is fast (~1.3–1.8 s for a whole 36–42-tok
turn on this card), but end-to-end turn latency is dominated by the **shared
grammar-FSM host code**, whose cost is `O(committed²)` (ep1's 110-tok turn alone
> 9 min). That host cost is the *same* in the HF stack (the HF row's 3.904 s/turn
carries it too), so there is no engine s/turn advantage from this fix and **K3 speed
remains unadjudicated on the engine path**. The only diffusion wall-clock reference
remains the HF stack. The engine's only legitimate speed lever over guided-AR is the
same FSM zero-forward bulk-commit (now proven live: e.g. 20 forced tokens / 0
forwards on ep2); realizing a *net* speed win needs the grammar-FSM host cost made
cheap (e.g. incremental FSM state, not re-parsing the growing prefix), which is
separate future work, not this fix.

| row | exact_args | episode_exact | valid | s/turn | fwd-or-tok/turn |
|---|---:|---:|---:|---:|---:|
| ENGINE (this fix) — byte-parity-implied | **= 47/63** | **= 13/20** | **= 63/63** | *forward ~1.5s; end-to-end host-bound (unmeasured full-battery)* | fewer-forwards live (ep2 36 tok / 16 fwd) |
| OUR HF hybrid-clean (v2) — reference | 47/63 | 13/20 | 63/63 | 3.904 | 56.83 denoise fwd/turn |
| stock-bf16-AR-guided (same build) | 51/63 | 14/20 | 63/63 | 1.213 | 82.24 tok/turn |
| stock-AR aggregate | 124/247 | 33/80 | 247/247 | 0.741 | 49.06 tok/turn |

### Artifacts (this acceptance) — `runs/p2_engine_acceptance/`
- `p2_full_acceptance.py` / `p2_full_acceptance.json` — single-boot acceptance:
  3-turn byte-parity + counters + `verify_invariants`, greedy determinism, temp>0
  contract, 5 temp=0.7 rollouts.
- `p2_temp_probe.py` / `p2_temp_probe.json` — temperature sweep (T=0/0.7/1/1.5/2,
  two seeds each) + greedy-repeat, disambiguating seed-diversity vs the batch-state
  wobble.
- (from the fix commit) `gap5a_windowed_ep0_default.json` (ep0 42/42),
  `gap5a_windowed_ep2.json` (ep2 36/36), `gap5a_windowed_ep1_head26.json`,
  `gap5a_windowed_probe.py`, `gap5a_ref.json` (the 3 HF reference turns).
- vLLM pin `6b81154` (`qwen3_5-flare-modelstate`): causal-windowed denoise mask +
  staged-tail probe read; CPU `70 passed`. qwen_diffusion: this doc.
- Repro: engine env of the prior section (`VLLM_QWEN3_5_FLARE=1`,
  `VLLM_QWEN3_5_FLARE_READONLY_DENOISE=1`, `VLLM_QWEN3_5_FLARE_MASK=248077`,
  `VLLM_USE_V2_MODEL_RUNNER=1`, `VLLM_ATTENTION_BACKEND=TRITON_ATTN`, cu13 CTK-skew
  flag, `VLLM_USE_FLASHINFER_SAMPLER=0`), one heavy proc in the
  `systemd-run … MemoryMax=22G` cage.

---

## SUPERSEDING UPDATE #1 — 2026-07-04 (post GAP-5A-rebuild + GAP-5B fix) — *superseded by #2 above*

This section supersedes the "pre-fix" analysis below. Two things changed since it
was written: (1) the sequential single-`[MASK]` decode was rebuilt on the engine
(vLLM pin `5e2fb53`, GAP 5A), and (2) **the decode-at-scale CUDA IMA (GAP 5B) is
now FIXED** (vLLM pin `1e32dcd`). Net verdicts:

| gap / step | verdict | one-line |
|---|---|---|
| **5B — decode-at-scale CUDA IMA** | **FIXED** | align spec-decode state copy indexed non-existent speculative block-table columns; feed the align state machine a neutral `num_accepted==1`. Real turns now decode without faulting. |
| **5A — turn byte-parity (engine hybrid_clean vs HF)** | **FAIL — algorithmic (not numeric)** | the driver is right but the engine **forward** still reads the fixed 32-position spec-draft canvas (`[tail + MASK + trailing MASKs]`) instead of the reference's exact `[tail + 1 MASK]`; the bidirectional probe MASK attends to ~20 trailing MASKs, so the probe logits diverge fundamentally from the reference. |
| **6 — matched-20 battery at parity** | **NOT ADJUDICABLE** | gated on Step 5 (first hard failure). 5B robustness is demonstrated (engine now runs real-length turns without crashing), but with 5A open the engine over-generates (grammar never sees completion on wrong values) and quality/wall-clock at parity do not exist yet. |

### GAP 5B — root cause + fix (GPU-localized, verified)

Reproduced on `models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16` (block 32, mamba 1024,
align+APC). The IMA is **not** at the first decode step: canvas mode decodes ~231
tokens then faults at `num_computed=1272`; hybrid_clean at `num_computed=1140`.
The `_flare_bounds_check` on slot/block/GDN-state index tensors **passes** on every
decode — the OOB is *inside a kernel*, not a named index tensor.

Localized with an env-gated phase synchronize (`VLLM_FLARE_SYNC_DEBUG=1`): the last
clean phase before the fault is **`postprocess pre-super (align state copy)`** — the
fault is in `super().postprocess_state` (the MambaHybrid align spec-decode state
copy), and it fires regardless of read-only-denoise (matching the prior
"crashes with readonly off"). An align-kernel input dump pinned it exactly:

```
A=32 N=1074 src_idx=2 bs=528 needs_copy=True token_bias=13 dest_col=1
  src+bias=15  bt_stride(width)=8      <-- gather col 15 into a width-8 block table
```

Mechanism: `postprocess_mamba_fused_kernel`'s temporal copy reads the accepted
draft token's intermediate GDN state from block-table column
`src_col + (num_accepted-1)`. That assumes `num_accepted-1` **speculative**
checkpoint columns exist — allocated only when a real `speculative_config` sets
`num_speculative_blocks`. The FLARE path drives the canvas as spec draft tokens
**without** a `speculative_config`, so `num_speculative_blocks == 0`, the mamba
block table has no such columns, and a commit of `A` tokens crossing a mamba-block
boundary indexes `src_col + (A-1)` (2+13=15) far past the width-8 table ⇒ IMA.

Fix (`1e32dcd`, `Qwen3_5FlareModelState.postprocess_state`): a FLARE commit is a
single causal pass whose final GDN state already lives in the running block —
there are no per-token intermediate states to select. `num_computed_tokens` is
advanced by `post_update` (consuming the real `num_sampled`) *before*
`postprocess_state`, so `num_sampled` here feeds ONLY the num_accepted scatter.
Feed the align state machine a neutral `num_accepted == 1` ⇒ the boundary
migration is a plain running-block copy (`token_bias == 0`, in-bounds). The real
commit count is retained for the commit counter. **Verified:** canvas decode runs
the full 300-token cap with zero IMA (was faulting at ~231); the only
`needs_copy=True` is the clean prefill boundary at `token_bias=0`. 66/66 CPU tests
green (no regression to the `af21dc8` machinery).

### GAP 5A — byte-parity FAIL, diagnosed ALGORITHMIC (top-5 logits both sides)

With 5B fixed, both sides run on the SAME dual-loadable export (HF `Fast_dLLM`
bridge over the vLLM export, blocker B). Reference (mask id resolved to **248077**,
passed to the engine via `VLLM_QWEN3_5_FLARE_MASK`) produces coherent, bounded
tool calls: ep0/turn0 42 tok `stop=complete_tool_call`, ep1 110, ep2 36. The
engine adapter now wires the tool schemas + `grammar_topk` to the engine
hybrid_clean FSM via `SamplingParams.extra_args` (else the engine ran free-form).

**Turn-0 result (greedy, identical prompt/schemas/mask):** engine matches the
reference **token-for-token for the first 12 tokens, then diverges at position 12**
and degenerates. Crucially those 12 tokens are the tool-call scaffolding + tool
name — all **grammar-forced**, so matching them proves the FSM wiring is live, NOT
that the forward is correct. Position 12 is the first grammar position with a real
logit-dependent CHOICE, and the engine's logits pick wrong:

```
decoded:  ref  "<tool_call>\n<function=initialize_qubits>\n<parameter=num_qubits>\n2\n</parameter>\n<"
          eng  "<tool_call>\n<function=initialize_qubits>\n<parameter=num_qubits>\n\n\n00\n\n..."
pos 12:   ref  45334 "=num"   eng  28 "="     (both grammar-legal tokenizations of "=num_qubits")
                              -> cascades into a wrong value ("2" vs "00")
```

**First-divergent-position top-k logits (both sides), pos 12:**

| token | text | REFERENCE (HF bridge) | ENGINE (vLLM) |
|---|---|---:|---:|
| 29 | `>` | 17.625 | **18.25 (argmax)** |
| 45334 | `=num` | **24.25 (argmax)** | 8.625 |
| 28 | `=` | 19.75 | 9.625 |
| 2334 | `num` | 12.5 | 16.125 |

This is **not** a bf16 near-tie flip: the whole distribution differs (ref confidently
predicts the tool structure `=num`=24.25; the engine's raw argmax is `>` and `=num`
is ~16 logits lower). Reading the probe at the shifted, raw-MASK, and last-clean
positions gives the **same** wrong logit, so it is **not** a +1-shift/position bug —
**the engine's forward output itself is wrong**.

**Root cause (algorithmic):** the `5e2fb53` rebuild fixed the *driver* (it reads one
probe logit and drives the chain-rule schedule) but the engine *forward* still
processes the **fixed 32-position spec-draft canvas**. `num_draft_tokens_per_req`
is set by the scheduler to `num_spec_tokens == 32` (there is no variable-width spec
schedule), so every probe forward runs over `[clean tail, MASK, MASK×(31-tail_len)]`.
The FLARE denoise read is bidirectional, so the probe `[MASK]` at position
`tail_len` attends to ~20 trailing `[MASK]`s — i.e. it is still a partial
**block-parallel** read of a mostly-masked block, not the reference's exact
`[tail + single MASK]`. That is why the logits diverge from the sequential
reference for any non-forced token. The "block-parallel vs sequential" gap was
moved from the driver into the forward, not closed.

**Fix needed (not a driver change):** drive the diffusion decode with a **variable
single-`[MASK]` forward width** (schedule `draft_len` spec tokens, not a fixed 32),
so the probe forward is exactly `[tail + 1 MASK]`. This is a scheduler /
model-runner change (dynamic per-step spec-token count for the diffusion path),
plumbed through `num_draft_tokens_per_req` / `num_spec_tokens_to_schedule` — the
same lever the standard spec-decode `dynamic_sd_lookup` uses. Until then byte-parity
is impossible by construction, exactly as the strategic note below anticipated.

### Step 6 — not adjudicable at parity; 5B robustness demonstrated

Per the acceptance's "stop at first hard failure", Step 5 is the first hard failure,
so the matched-20 quality/wall-clock battery is not run at parity. What IS newly
true post-5B: the engine decodes real-length turns (prompts 1041/1443/917 tok;
canvas 300-tok generations) **without crashing** — the substrate is live at scale.
But with 5A open the engine's value logits are wrong, the grammar never observes a
`complete_tool_call`, and the request over-generates to `max_new_tokens` (grammar
cost grows with `committed`), so a full 63-turn battery is both quality-meaningless
AND infeasibly slow until the 5A forward-width fix lands. **No sunk-cost engine
KPI was invented.** The only honest diffusion wall-clock remains the HF stack
(3.904 s/turn) — not the engine.

### Artifacts (this update) — `runs/p2_engine_acceptance/`
- `byte_parity_2proc.py` — two-process byte-parity driver (one 9B per process;
  reference vs engine on identical prompt/schemas/mask, token+byte diff).
- vLLM pin `1e32dcd` (5B fix + `VLLM_QWEN3_5_FLARE_MASK` + `VLLM_FLARE_SYNC_DEBUG`);
  qwen_diffusion adapter now passes tool schemas to the engine via `extra_args`.
- Repro (RAM cage, one heavy proc at a time): boot with the engine env below and
  `VLLM_FLARE_SYNC_DEBUG=1` for the phase where a decode fault lands;
  `VLLM_FLARE_BOUNDS_CHECK=1` for the (passing) index-tensor checks.

---

Acceptance re-run of `p2_engine_gauntlet_real_result.md` **after all three Step-5/6
structural blockers were wired** *(PRE-FIX — superseded by the section above)*:

- **Blocker A** (vLLM pin `qwen3_5-flare-modelstate` `e38a9ea`): `hybrid_clean` made a
  selectable engine decode mode (`VLLM_QWEN3_5_FLARE_DECODE=hybrid_clean`) driving the
  FSM/greedy `HybridCleanBlockDecoder`.
- **Blocker B+C** (qwen_diffusion `ed479b3`): one dual-loadable checkpoint (HF-bridge
  loader over the vLLM export) + `VllmFlareEngineAdapter.run_turn` seam.

This acceptance ran the STEP-5 byte-parity and STEP-6 M2 gates against those wirings on
the real export `models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16` (block/canvas 32), RTX 5090
(sm_120), engine venv `.venv-vllm-p2-main`, every GPU process one-at-a-time in the
`systemd-run … MemoryMax=22G` cage (`free -g` ≥ 27 G available before each boot).

## Bottom line

| step | verdict | one-line |
|---|---|---|
| pre — CPU wiring intact | **PASS** | 61/61 CPU tests green (23 hybrid_clean + 17 hybrid_clean_flare_decode + 21 flare state-machine); no regression to the `af21dc8` read-only-denoise machinery. |
| 5 — turn byte-parity (engine hybrid_clean vs HF) | **FAIL (BLOCKED)** | TWO independent hard failures, both GPU-confirmed THIS run: (5A) the engine `hybrid_clean` sources **block-parallel** canvas logits while the HF reference is **sequential single-`[MASK]`** → algorithm-level divergence, byte-parity impossible by construction; (5B) real-length matched-20 turns hit a **deterministic CUDA illegal-memory-access at the first decode step** → the 3 parity turns cannot even be driven. |
| 6 — matched-20 M2 battery on the engine | **BLOCKED** | Gated on Step 5 (byte-parity ⇒ the ENGINE==HF-row quality gate). Byte-parity fails and real turns crash, so `exact_args` / `episode_exact` / TRUE forwards-per-turn / s-per-turn cannot be produced on the engine. No honest engine wall-clock. |

**Net:** the three blockers are *wired* (imports resolve, CPU tests green, the adapter
boots the real export and drives a short turn), but **the wiring does not make the engine
reproduce the trained algorithm, and it does not run at real prompt length.** Steps 5-6
still need net-new engineering: (1) close the per-token logit seam (run the reference's
sequential single-`[MASK]` forward schedule on the engine), and (2) fix a decode-at-scale
IMA in the shared FLARE canvas/commit spec-decode forward. Not a re-run. K3 remains
**unadjudicable on the engine path**.

---

## Pre-check — CPU wiring intact (PASS)

```
pytest tests/v1/sample/test_hybrid_clean.py \
       tests/v1/sample/test_hybrid_clean_flare_decode.py \
       tests/v1/worker/gpu/test_qwen3_5_flare_state_machine.py
=> 61 passed
```

Editable vLLM confirmed at `/home/mark/shared/vllm_p2_pr42406/vllm`; the `hybrid_clean`
FSM is now actually invoked at both the block-decoder and sampler seams (the assertion
the orphaned-FSM bug lacked). The `af21dc8` read-only-denoise fix is untouched.

---

## STEP 5 — turn byte-parity: FAIL

Per the acceptance rule ("run in order, stop at first hard failure"), Step 5 is the first
hard failure. There are two independent reasons, either of which alone blocks byte-parity.

### 5A — Algorithm-level divergence: block-parallel engine vs sequential single-`[MASK]` reference

This is a divergence of **algorithm**, not rounding, and it is **not an engine-side-and-small
fix**. Proven from both decoders' source:

- **HF reference** — `eval_flare_northstar_hybrid_clean.sample_hybrid_clean`
  (sha `a4c66751…`): decode is **sequential, one `[MASK]` at a time**. Each model-chosen
  token appends exactly one mask (`torch.cat([output_ids, mask], dim=1)`), forwards over
  `[committed_clean_prefix, MASK]`, and reads the **single last-position** shifted logit
  (`shifted_active_logits(...)[:, -1, :]`); one forward per non-forced token. Truly-forced
  structural tokens (`len(legal)==1`) are FSM bulk-committed with **zero** forwards. So the
  logit for output position *k* is conditioned on the **actual committed clean tokens
  0..k-1**.
- **Engine** — `qwen3_5_flare.Qwen3_5FlareSampler._hybrid_clean_step`: decode is
  **block-parallel**. One denoise forward runs over the whole 32-position canvas, then
  `_gather_block_logits` reshapes all 32 positions' logits `[num_decode, CL, vocab]` and
  `HybridCleanBlockDecoder.decode_block` walks them. Positions 1..31 are conditioned on the
  **noisy/random canvas**, not on the sequentially-committed clean prefix.

⇒ For any block containing >1 model-decoded token, the engine's per-position logits differ
from the reference's by construction; they cannot byte-match. The wiring commit itself
flags this as an **unclosed INTEGRATION SEAM** (`e38a9ea` `_hybrid_clean_step` docstring:
"byte-parity-exact with the reference **only once the per-token-vs-block-parallel logit
gap … is closed**").

**Empirical corroboration** (this export, no co-load needed):
- Engine `hybrid_clean` on a working short prompt emits **gibberish** —
  `assistant_text = "<tool_call>\n<function= .ер s ET    .  1. …"`
  (`runs/p2_engine_gauntlet_real/engine_smoke_adapter_short_hybrid_clean.json`), while its
  own zero-value-projection tripwire holds (`projected_value_tokens_exact=0`).
- The HF-bridge forward on the **same** export is coherent — top-1 `" Paris"`
  (`runs/p2_engine_gauntlet_real/blockerB_hf_bridge_forward.json`).

Same weights, same tokenizer, same mask id (248077): the difference is the decode
algorithm, exactly as the source shows.

**Divergence diagnosis:** ALGORITHM (not rounding). **First-divergent-position + top-5
logits could not be tabulated turn-for-turn** because the real-length parity turns crash
(5B); the short-prompt evidence already shows the engine's very first emitted structural
value token is off-distribution (gibberish) vs the reference's coherent stream.

**Fix scope:** NOT small / NOT engine-side-trivial. Byte-parity requires the engine to run
the reference's **sequential single-`[MASK]`** forward schedule (one forward per value
token over `[prefix, MASK]`), or to expose a **forward-only logit seam** feeding the shared
`sample_hybrid_clean` driver. Both are net-new engineering — and both remove the
block-parallel "fewer-forwards" mechanism for value tokens (see the strategic note below).

### 5B — Deterministic CUDA illegal-memory-access on real-length turns

The 3 matched-20 turns require the real agentic prompts. Turn-0 of episode 0 is **1041
tokens**; the engine crashes at the **first decode step**:

```
ERROR dump_input: scheduled_cached_reqs=CachedRequestData(req_ids=['0-84510935'],
  new_block_ids=[([9],[10],[11],[12])], num_computed_tokens=[1041], num_output_tokens=[1]),
  num_scheduled_tokens={0-…: 33}, scheduled_spec_decode_tokens={0-…: [-1 ×32]},
  num_common_prefix_blocks=[0,0,0,3], new_block_ids_to_zero=[12], num_spec_tokens_to_schedule=32
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

Prefill completes (`num_computed_tokens=1041`); the fault is the first decode
(1 real + 32 canvas draft tokens). Isolated across GPU boots this run
(`CUDA_LAUNCH_BLOCKING=1`, RAM cage):

| decode mode | read-only-denoise | mamba_block_size | prompt | result |
|---|---|---|---|---|
| — (short smoke, prior) | on | 1024 | 10 tok | **OK** |
| hybrid_clean | ON | 1024 | 1041 tok | **CRASH (IMA)** |
| canvas | **OFF** | 1024 | 1041 tok | **CRASH (IMA)** |
| canvas | on | **4096** (1041 in ONE mamba block) | 1041 tok | **CRASH (IMA)** |
| canvas | on | 1024, **APC off** | 1041 tok | config-rejected (align mandates `--enable-prefix-caching`) — not testable |

⇒ The IMA is **independent of decode mode** (canvas and hybrid_clean both), **independent
of the read-only-denoise snapshot** (crashes with it OFF — rules out the `af21dc8`
snapshot/restore rows), and **independent of the mamba-block-1024 boundary** (crashes with
the whole 1041-token state in one mamba block). It is in the **shared FLARE canvas/commit
spec-decode DECODE forward over a long, multi-KV-block prefix**. Under `CUDA_LAUNCH_BLOCKING`
the error surfaces at the async output-copy sync (`async_utils.get_output →
copy_event.synchronize()`); the exact faulting kernel needs `compute-sanitizer` (absent on
this box) or a `TORCH_USE_CUDA_DSA` rebuild — deferred.

**Diagnosis:** engine-side (vLLM `Qwen3_5FlareModelState` decode-at-scale), deterministic,
**NOT small**. This is the "block-32 mid-chunk / decode-at-scale" defect the `ed479b3`
smoke already flagged as an engine-owner handoff; this run pins it further (mode-,
readonly-, and mamba-block-independent). It blocks every real-length turn, so the 3
matched-20 byte-parity turns cannot be driven and `projected_value_tokens_exact` cannot be
compared turn-for-turn on real data.

**STEP 5 verdict: FAIL.** Stop here.

---

## STEP 6 — matched-20 M2 battery on the engine: BLOCKED

The correct quality gate is **ENGINE == HF row** (47/63, same weights + algorithm —
byte-parity implies it). Byte-parity fails (5A) and real turns crash (5B), so `exact_args`,
`valid`, `episode_exact`, TRUE forwards-per-turn and s-per-turn **cannot be produced on the
engine**. The only engine signal is short-prompt substrate liveness — `read_advance_ratio ≈
3.0`, `forced_grammar_tokens=5` (FSM bulk-commit, zero forwards), `zero_forward_rows=2`,
`projected_value_tokens_exact=0` — i.e. the fewer-forwards + zero-value-projection
mechanisms are live, but over **gibberish** output. That is substrate liveness, not the M2
KPI. No sunk-cost engine number was invented.

Reference rows (matched-20, `runs/endgame_scoreboard`, **NOT the engine**):

| row | exact_args | episode_exact | valid | s/turn | fwd-or-tok/turn |
|---|---:|---:|---:|---:|---:|
| OUR HF hybrid-clean (v2) | 47/63 | 13/20 | 63/63 | 3.904 | 56.83 denoise fwd/turn |
| stock-bf16-AR-guided | 51/63 | 14/20 | 63/63 | 1.213 | 82.24 tok/turn |
| stock-AR aggregate | 124/247 | 33/80 | 247/247 | 0.741 | 49.06 tok/turn |

**K3 (as written)** — PASS needs `< 1.120 s/turn AND ≥ 55/63 exact-args, 15/20 ep, 63/63
exact_seq, 63/63 valid_xml, value force-counters == 0`. **Speed verdict: cannot be
adjudicated on the engine path** — no honest engine wall-clock at real quality exists. The
only diffusion wall-clock remains the HF stack (3.904 s/turn, ≈3.5× the 1.120 target,
≈5.3× stock-AR aggregate), which is not the engine. Unchanged from the prior gauntlet, now
strengthened: **even with A/B/C wired, the block-parallel engine substrate cannot reproduce
the sequential reference and cannot run real-length turns.**

---

## Strategic note (surfaced by 5A)

The winning HF row's forward-savings (56.83 fwd/turn vs stock-AR's ~82 tok/turn) comes
**entirely from the grammar-FSM bulk-commit of truly-forced structural tokens with zero
forwards** — the reference decodes **every value token sequentially with one forward**. The
engine's block-parallel canvas is therefore a **different** algorithm, not a faithful
accelerator of the reference: on this checkpoint it is quality-dead (gibberish). A
byte-parity-and-quality engine path must run the sequential single-`[MASK]` value decode;
its only legitimate speed lever over guided-AR is the same FSM zero-forward bulk-commit,
not block-parallel value denoising. This should be reflected in the M-milestone plan.

---

## What remains before Steps 5-6 can pass (net-new engineering, not a re-run)

1. **Close the logit seam (Blocker 5A):** drive the engine with the reference's sequential
   single-`[MASK]` forward schedule, or expose a forward-only block-logit seam feeding the
   shared `sample_hybrid_clean` driver, so both sides are the same algorithm. Precondition
   for any byte-parity.
2. **Fix the decode-at-scale IMA (Blocker 5B):** localize with `compute-sanitizer` /
   device-side asserts (mode-, readonly-, mamba-block-independent; long multi-KV-block
   prefix, first decode), then fix in the FLARE canvas/commit spec-decode forward.
3. Then: 3-turn byte-parity (greedy, identical FSM, `projected_value_tokens_exact==0` both
   sides), then the matched-20 M2 A/B vs guided-AR **re-baselined on the same pinned build**.

---

## Artifacts (this acceptance) — `runs/p2_engine_acceptance/`
- `step5_ima_hybrid_clean_ro_on_mamba1024.log` — hybrid_clean, readonly ON, mamba 1024 → IMA.
- `step5_ima_canvas_ro_off_mamba1024.log` — canvas, readonly OFF, mamba 1024 → IMA (rules out readonly).
- `step5_ima_canvas_ro_on_mamba4096.{log,json}` — canvas, mamba 4096 → IMA (rules out mamba-block crossing).
- `step5_scheduler_dump_at_crash.txt` — the faulting decode step (num_computed_tokens=1041).
- `ima_mamba_block_probe.py` — the parametric IMA isolation probe.
- (prior, `runs/p2_engine_gauntlet_real/`) `engine_smoke_adapter_short_hybrid_clean.json`
  (engine gibberish + live counters), `blockerB_hf_bridge_forward.json` (HF-bridge coherent
  `" Paris"`).

## Reproduce (RAM cage; one heavy proc at a time)
```
VENV=/home/mark/qwen_diffusion/.venv-vllm-p2-main
systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=4G \
  -E CUDA_HOME=$VENV/lib/python3.12/site-packages/nvidia/cu13 \
  -E NVCC_APPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK \
  -E VLLM_USE_FLASHINFER_SAMPLER=0 -E VLLM_USE_V2_MODEL_RUNNER=1 \
  -E VLLM_ATTENTION_BACKEND=TRITON_ATTN -E VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  -E VLLM_QWEN3_5_FLARE=1 -E VLLM_QWEN3_5_FLARE_READONLY_DENOISE=1 -E MAX_JOBS=4 \
  -E CUDA_LAUNCH_BLOCKING=1 \
  -- $VENV/bin/python scripts/parity_audit_flare_engine.py \
     --mode engine-smoke --decode-mode hybrid_clean \
     --base-model models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16 \
     --tokenizer-path models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16 \
     --input-jsonl data/toolcall_eval_native/flare_scaleup_native_58.jsonl \
     --episode-index 0 --turn-index 0 --block-size 32 --max-new-tokens 32
# => CUDA illegal memory access at the first decode (num_computed_tokens=1041).
# mamba-block / readonly isolation: runs/p2_engine_acceptance/ima_mamba_block_probe.py
```
