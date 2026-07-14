# RUNG W-2b — seeded-sampler-faithful copy verify at temp>0 — VERDICT: PROCEED

Fixes the dc585bf dormancy blocker: W-2's copy draft-verify was gated to
`temperature==0` (`_hc_try_stage_verify` returned before the drafter whenever
temp>0), so under the frozen C46 **temp-0.6** envelope the gate booted ON but sat
**DORMANT** (spans=0 across every turn) — a byte-identical placebo of the banked
gate-OFF twin.

## THE FIX (engine pin `qwen3_5-flare-modelstate @ 41fd28e`, LOCAL, never pushed)
- **Removed** the `temp>0 -> return False` short-circuit in `_hc_try_stage_verify`.
- **Rewrote** `_hc_verify_read` to decode each drafted position through the SAME
  `decode_model_token` the K=1 schedule runs — greedy grammar-steer for a
  structural/value token, and for a free-form token the request's own seeded
  categorical drawn from the slot's `torch.Generator`. The draw uses the same warp
  order, the same RNG stream object, and the same number of draws in the same order
  a serial K=1 run would, so after an accepted prefix the generator state EQUALS the
  state after that many K=1 draws (**RNG-STREAM CONSERVATION**). Accept while
  `draw==draft`, stop at the first mismatch; the mismatched draw IS the committed K=1
  token for that position (never redrawn) — the verify forward doubles as that
  position's K=1 forward. The committed span is exactly the K=1 (sampled) trajectory
  in bulk: **distribution-exact by identity, not approximation.** Draws are bounded to
  `max_new_tokens` room (draws==commits) so a cap-limited span never advances the RNG
  past what it commits. Byte-assert + boundary-trim + prefix-commit unchanged.

## CPU TESTS (88 green)
- `test_w2_temp_sampling.py` (NEW): RNG-stream conservation (accept-k leaves the slot
  generator identical to serial-k) + the ideal-oracle byte-parity sweep extended to
  **temp 0.6, 0 divergences** (18 configs × seeds, gate-ON == gate-OFF + identical
  generator state).
- `test_w2_causal_verify` + `test_w1b_engine_seam` updated for the divergent-draw
  commit + cap-room semantics; hybrid_clean + w1 + boundary-trim regressions green.

## LIVE RECERT @ temp 0.6 SAME SEED (`results06.json`, ~1 GPU-h)
6 snippets × 5 reps (copy-heavy write_file) + 12 near-dup pointer-slip FA, gate-OFF
vs gate-ON, one server at a time, same seed 20260714.

| bar | result |
|---|---|
| **gate-ON == gate-OFF BIT-IDENTICAL** | recert **30/30**, FA **12/12** (`VERDICT=BIT-IDENTICAL`) |
| bit-reproducible (5 reps) | 6/6 both arms |
| w1 counters LIVE (dormancy fixed) | spans **138**, toks 2905, vfwd 138, rej 10, **arej 0** |
| FA battery | **12/12** correct resolution, **0** false accepts |
| tok/fwd @0.6 | gate-OFF 1.17 -> gate-ON **4.51**; forward-speedup **3.855×** (fwd 3936->1021) |

Bit-identity holds by construction: value content is grammar-greedy (temp-independent)
and any free-form token is drawn from the identical seeded stream. The forward-speedup
exceeds the temp-0 3.32× on this copy-heavy corpus.

## RELAUNCH
C46-new-envelope runner (`runs/k_gate_c46_newenv/run_gate.sh`) relaunched VERBATIM with
a PERMANENT dormancy preflight in `run_arm_twin.sh`: after episode 1, assert cumulative
w1 spans>0 ELSE abort `[state] DORMANT_GATE` (never burn a placebo run again). Live:
GATE-ON assert PASS, proxy read-clamp smoke PASS, 4 shards fanned out, w1 spans>0 with
arej=0 from the first turns. Engine pin stays LOCAL, gate default OFF.
