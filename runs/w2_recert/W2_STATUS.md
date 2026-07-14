# RUNG W-2 — causal block-commit verify redesign + drafter boundary-trim — VERDICT: PROCEED

Object = twin@plain twinK1, FLARE hybrid_clean, mask 248077, maxlen 32768, gmu 0.74,
seqs 4, `VLLM_FLARE_BIDIR_PROBE=1` (C46-iter2 envelope). Gate
`VLLM_FASTDLLM_W1_DRAFT_VERIFY` OFF(ref)/ON(W-2). Two bounded boots (gate-OFF control,
gate-ON W-2); server DOWN + GPU idle (385 MiB, 0%) at exit. Engine changes are LOCAL on
`qwen3_5-flare-modelstate` (never pushed). Raw jsonl/logs/harnesses here.

## THE REDESIGN (byte-faithful, W-1d objection resolved)

W-1d root cause: the copy-verify was a SEPARATE, variable-width, BIDIRECTIONAL read
(`_apply_bidir_key_window`) — a full-reveal reconstruction (leak) whose novel (width,
state) config forked the live GPU numeric trajectory, so a downstream temp-0 argmax
flipped run-to-run (idx4 4/5). Three changes:

1. **CAUSAL verify** — the staged verify row's per-seq causal flag is forced ON in
   `prepare_attn` (it stays `is_encoder_phase=False` → read-only-denoise, zero GDN
   advance; `_denoise_state_rows_from_md` selects read-only rows by PHASE, not by this
   flag). Each drafted position `tail_len+j` now attends only to `[tail + draft[:j]]` —
   the authoritative causal K=1 chain. The bidirectional leak is gone, so the longest
   matching prefix is the causal-K=1 prefix → **prefix-commit is strict-subset-safe**.
2. **FIXED block-commit width** — the verify forward is scheduled at this block's commit
   width (`block_target`, MASK-padded), so it REUSES the block-commit's captured cudagraph
   (deterministic replay) instead of a novel per-width shape. This is what rides "the same
   width / same kernel path the K=1 decode runs" and removes the variable-width fork.
3. **PREFIX-COMMIT** — accept `draft[:m]`, `m = LCP(causal-argmax, draft)`; the divergent
   position + rejected suffix decode K=1 (one-step cooldown). Hard byte-assert retained
   (fail-closed). Plus **DRAFTER BOUNDARY-TRIM** (the 83% lever): the drafter's span is
   trimmed at the first content/scaffold boundary (fence / chat-template special) via the
   guard's G5 `grammar_clip` hook BEFORE staging, so it stops over-copying into scaffold.

### Architectural finding (verified in code, changes the handed spec)
The FLARE commit folds the WHOLE block into ONE end-state — `postprocess_state` feeds a
neutral `num_accepted=1` to the align state machine; there are **no per-position GDN
states to select**. So the handed "ride the ordinary block-commit forward, ZERO extra
forward, AND prefix-commit" is not jointly realizable: prefix-commit needs a partial-block
fold that the end-state-only commit cannot supply without an extra forward. The realized
design keeps a separate read-only verify forward, but at the FIXED commit width reusing the
commit's captured graph (deterministic) — the reproducibility win comes from causal +
fixed-width, not from eliminating the forward.

## CPU CERT (all green)
- boundary-trim 9/9; W-2 causal-verify (prefix-commit direct + 2 byte-parity) 3/3;
  W-1 draft-verify + engine seam 15/15; hybrid_clean regression 58/58 (**85 total**).
- Ideal-oracle divergence sweep (real `_hybrid_clean_step`, gate-ON vs gate-OFF):
  **0/56 configs diverge** (`cpu_repro.json`). Prefix-commit parity on a divergent copy
  source: byte-exact, `assert_rejects=0`.

## LIVE RECERT (decisive)
| metric | gate-OFF (control) | gate-ON (W-2) |
|---|---|---|
| exact (6×5) | **30/30** | **30/30** |
| bit-reproducible | **6/6** | **6/6** (idx4 W-1d 4/5 → **5/5**) |
| arej (byte-assert) | — | **0** / 34 reqs |

- **BAR MET by construction**: gate-ON is now 6/6 bit-reproducible, matching gate-OFF. The
  W-1d residual (downstream numeric flip) is gone — the causal fixed-width read-only verify
  reuses the commit graph, so gate-ON introduces no new deterministic-replay surface.
- **FA battery** (12 near-dup pointer-slip cases): **12/12 correct resolution, 0 false
  accepts**, arej=0 (`fa_battery.json`).
- **Throughput** (34-req copy-heavy corpus): forwards **2519 → 759 = 3.32× blended**
  (2999 tok; blended tok/fwd 1.19 → 3.95). Copy efficiency **19.6 tok/verify-fwd**.
  Wall-clock **23.96 → 9.41 ms/tok = 2.55×**. Beats W-1b's 2.35×/1.95×.
- **Reject share**: full-rejects **0/90 (0%)**, partial-diverged **10/90 (11.1%)** — vs
  W-1d **94.8%**. Boundary-trim + prefix-commit cut the reject tax.
- **Unfired byte-parity**: gate-ON committed stream == gate-OFF K=1 across the full corpus
  (30/30 exact each vs the same gold; `det_on.json` == `det_off.json`).

## VERDICT = PROCEED
Byte-exact (30/30), bit-reproducible (6/6, W-1d bar met), 0 false accepts, arej=0, at
3.32× blended forwards-speedup. Re-owe: the 6-ep C46 A/B then the C46-new-envelope
certification. Gate stays default-OFF; engine pin stays local.
