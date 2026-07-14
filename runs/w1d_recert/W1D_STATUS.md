# RUNG W-1d — root-cause + fix + recert of the W-1c live seam corruption — VERDICT: STOP

Object = twin@plain twinK1, FLARE hybrid_clean, mask 248077, maxlen 32768, gmu 0.74, seqs 4,
`VLLM_FLARE_BIDIR_PROBE=1` (the C46-iter2 envelope). Gate `VLLM_FASTDLLM_W1_DRAFT_VERIFY`
OFF(ref)/ON. Three bounded boots (root-cause trace, gate-ON recert, gate-OFF control); server DOWN
+ GPU idle (385 MiB, 0%) at exit; ~0.6 GPU-h of the ~3 budget. Pin `abb2f65`+W1d fixes (LOCAL,
never pushed). Raw jsonl/logs/harnesses here; consolidated `results.json`.

## ROOT CAUSE — the W-1c "truncated commit" hypothesis is DISPROVEN

Three independent lines kill the commit-slice / truncated-commit hypothesis:
1. **CPU ideal-oracle repro of the REAL `_hybrid_clean_step`** (`repro_seam2.py`): gate-ON ==
   gate-OFF committed stream for ALL `cl in {8,16,32}` × every `align_off` → **0 divergences**.
   The phase-machine accounting (accept → zero-forward commit → block-commit fold → emit slice)
   is byte-faithful.
2. **Live byte-assert counter `arej=0`** across all 34 gate-ON requests → no truncated/altered
   commit ever occurred; every accept commits the candidate byte-for-byte.
3. **GPU span trace** (`VLLM_W1_TRACE`): the accepted spans are byte-EXACT copies of the true
   content (`draft==pred==gold`). The idx4 corruption occurs **downstream of a byte-correct
   accept** (rep0 truncated at `replicas: 4`; the accept `apps/v1…spec:\n ` was identical to the
   3 non-corrupt reps).

**Actual root cause.** Firing the W-1 fast path replaces `k` uniform-width K=1 denoise forwards
with ONE variable-width bidirectional verify forward + a zero-forward span commit. This shifts the
live GPU numeric trajectory (cudagraph capture per width, GDN recurrent snapshot/restore, argmax
near-ties) off the deterministic gate-OFF K=1 path, so a **downstream temp-0 argmax flips
run-to-run** — corruption gate-OFF K=1 never exhibits. It is genuinely non-reproducible: idx5 was
0/5 in the W-1c boot but **5/5 in both W-1d boots**; idx4 flipped **1/5 WITHIN one boot** (same
slot, same input, sequential). So the W-1c "idx5 100%-deterministic + idx3/4 non-det ⇒ TWO
defects" framing is not borne out — it is **one class** (fast-path numeric non-reproducibility vs
the deterministic K=1 schedule); a boot's fixed cudagraph capture makes it look within-boot-
deterministic in a given boot.

**Secondary (structural, latent).** Under `BIDIR_PROBE=1` the verify read is a full-reveal
bidirectional denoise (`_apply_bidir_key_window`), NOT a causal K=1 check — each draft position
attends to future draft positions, so its argmax is a bidirectional reconstruction, not the serial
K=1 argmax. Empirically it still rejected the scaffold over-copies in-trace, so this is a latent
risk, not the active corruption.

## FIXES (committed as permanent hardening; fast path stays gate-default-OFF)

- **HARD BYTE-ASSERT (fail-closed, mandated).** `_hc_verify_read`: BEFORE commit, refuse (fall
  back to K=1, `w1_assert_rejects++`) if the candidate would be truncated at `max_new_tokens`;
  AFTER commit, assert `committed span == candidate ids` (logs on mismatch). Makes the
  truncated/altered-commit class structurally impossible under future regressions.
- **Block-limited accept.** `apply_verified_draft(ids, block_limit=…)`: the accepted span's
  trailing forced run now respects the chunk-aligned block boundary exactly as `decode_probe`
  does (was UNBOUNDED — a real K=1-unfaithfulness, though not the active corruption).
- **Counters/observability.** `w1_assert_rejects` → `stats()` + done-line (`arej=`); gated
  `VLLM_W1_TRACE` span-level candidate/pred/committed instrumentation.
- **CPU tests:** 15 W-1 tests pass (3 new: byte-assert fail-closed on cap truncation, no spurious
  assert on clean run, accept respects block-limit) + **58 hybrid_clean regression tests green**
  (gate-OFF parity intact).

## DECISIVE CONTROL RECERT — 6 snippets × 5 reps

| # | snippet | gate-OFF K=1 | gate-ON W1 (+W1d fixes) |
|---|---|---|---|
| 0 | utils/retry.py | det, 5/5 | det, 5/5 |
| 1 | models/user.py | det, 5/5 | det, 5/5 |
| 2 | handlers/webhook.py | det, 5/5 | det, 5/5 |
| 3 | migrations/0042.sql | det, 5/5 | det, 5/5 |
| 4 | k8s/deployment.yaml | det, 5/5 | **NON-det, 4/5** |
| 5 | lib/parse.py | det, 5/5 | det, 5/5 |
| | **total** | **30/30, 6/6 bit-repro** | **29/30, 5/6 bit-repro** |

`arej=0` on every gate-ON request → the residual corruption is NOT the commit-slice class the
byte-assert guards; it is the downstream numeric non-reproducibility, which no seam logic fix
removes. **BAR (gate-ON 30/30 bit-reproducible) NOT MET.**

## ON-POLICY REJECT ANATOMY (174 verify reads, idx3/4/5 ×3)

accepts 9 / rejects 165 (**94.8% reject**). Of rejects:
- **83% span-boundary over-copy** (LCP≥2, mean **8.8** recoverable prefix tokens): the drafter
  greedily extends the copy PAST the content boundary into the prompt's trailing scaffold (```` ``` ````
  fence, chat template); whole-span-or-nothing verify rejects the ENTIRE span.
- **8.5% context-divergence** (LCP=0): drafter mines irrelevant prompt text (e.g. the instruction
  line).
- **8.5% single-position** (LCP=1).

**Prefix-commit lever = NOT strict-subset-safe → NOT implemented.** The verify is a full-reveal
bidirectional read, so the longest-accepted-prefix (LCP of draft vs the LEAKED argmax) is NOT the
causal-K=1-matching prefix — it can commit DIFFERENT tokens than K=1, not merely fewer; and it does
not fix the downstream numeric flip. The byte-faithful redesign (DESIGN-ONLY, W-2): **verify at the
block commit using the authoritative causal block-commit logits** (which ARE the K=1 next-token
logits) — accept only the prefix whose block-commit argmax == draft. That removes both the leaked
verify read AND the extra perturbing forward.

## VERDICT = STOP
Not dispatch-ready. The gate-ON fast path is not bit-reproducible vs the certified deterministic
gate-OFF K=1 (29/30) even with the byte-assert + block-limit fixes, because the defect is downstream
GPU numeric non-reproducibility, not a commit-slice bug. Fixes committed as permanent hardening;
fast path stays gate-default-OFF; K=1 gate-OFF path remains byte-exact + deterministic (30/30,
unaffected). Re-owe the 6-ep C46 A/B + C46-new-envelope only after the block-commit-verify redesign.
