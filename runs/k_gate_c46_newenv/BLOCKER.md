# C46 NEW-ENVELOPE — LAUNCH BLOCKED (gate dormant under the frozen envelope)

**Status: launched, verified, then STOPPED at eps=3/48 (wall ~600 s). Server torn down,
GPU settled (385 MiB), `[gate] EXIT rc=0`. No GPU-hours spent on a structurally-null run.**

## What passed
- Runner detached + alive (pid 962628, own session); W-2 engine-source assert PASS;
  48/48 images present; GPU-idle preflight clear.
- Server booted gate-ON: `FLARE W-1b copy draft-and-verify gate: True` — the runner's
  hard `GATE-ON ASSERT PASS` fired before episodes; proxy read-clamp smoke PASS.
- 4 shards fanned out (correct seeds); 74 turns completed; **arej=0** throughout
  (byte-assert never fired); resolve pipeline healthy.

## The blocker (caught by the "w1 span commits firing" verify sub-item)
Across **all 74 turns — including 7 turns with 300+ free "value" tokens** (write/edit
regions where the recert fired heavily) — the w1 counters stayed
**`spans=0 vfwd=0 rej=0 arej=0`**. The drafter proposed nothing; the verify never staged.

Root cause (structural, not transient): the W-2 draft-verify is a **greedy (temp==0)**
speculative mechanism. In `qwen3_5_flare.py:2460`, `_hc_maybe_stage_verify` does
`if self._hc_temperature(slot) > 0.0: return False` — it returns before the drafter is
even consulted whenever the request temperature is > 0.

But the **frozen C46 envelope forces `temperature=0.6`** (`run_arm_twin.sh` exports
`LUMO_PROXY_FORCE_TEMPERATURE=0.6`, inherited unchanged from the banked comparators).
**Confirmed empirically:** all 97 proxy-forwarded requests carried `"temperature": 0.6`.
So the gate is ON but **structurally dormant** — every turn decodes byte-identically to
gate-OFF K=1 sampling. The W-2 recert's 30/30 + 3.32× was run at **temp 0** (`W2_STATUS.md`:
"posts the fixed corpus at temp 0").

## Why this defeats the run's purpose
Under temp 0.6 the gate never fires, so:
- **speed deliverables are null by construction** — live tok/fwd ≈ 1.0, wall/episode ≈ the
  banked gate-OFF twin; no draft-verify speedup can be measured;
- **resolve@1 would be byte-identical to the banked gate-OFF twin (1/48)** — the McNemar
  reads reduce to "twin gate-ON == twin gate-OFF", telling us nothing new.
The paired temp-0.6 envelope (required to reuse the banked AR 12/48 + gate-OFF 1/48) and
the W-2 gate (temp-0 only) are **mutually exclusive under the current engine pin**.

## Options for the parent (decision needed — not taken unilaterally)
1. **All-greedy re-run (temp 0).** Re-run AR + twin gate-OFF at temp 0 as new banked
   comparators, then twin gate-ON greedy fires the gate. Exercises W-2 as built + keeps a
   valid paired read — but breaks "do not re-run the comparators" and changes the frozen
   envelope (greedy vs sampled resolve rates differ; empty-patch re-drive value differs).
2. **Extend W-2 to temperature>0 (rung W-3).** Speculative *sampling* with a proper
   accept/resample correction so the gate fires under the frozen 0.6 envelope. Preserves
   the banked temp-0.6 pairing (McNemar vs AR 12/48 stays valid). Engine work, not a launch
   tweak. This is the principled path if the paired read is the priority.
3. **Accept the dormant-gate null.** Let it run to confirm byte-safety-at-scale (arej=0
   over 48 real episodes) + gate-ON==gate-OFF inertness under sampling. Valid but yields
   **zero speed signal** — does not advance the speed goal or serve as the "convergent gate."

The infra in this directory is correct and reusable for whichever option is chosen (only
the temperature / comparator set changes). Nothing here was pushed to the engine; the pin
stays local, gate default OFF.
