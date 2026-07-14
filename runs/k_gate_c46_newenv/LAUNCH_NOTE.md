# C46 UNDER THE NEW ENVELOPE — LAUNCH NOTE (#141)

**Launched 2026-07-14, detached (setsid + cage), single twin arm.** This is the
convergent gate for BOTH campaign goals: the twin@K1 diffusion arm served **gate-ON**
with the **W-2 causal draft-verify** path, decoded on the frozen Tier1-C46 48-instance
slice, read against the two BANKED comparators. It mirrors the iteration-2 gate infra
(`runs/k_gate_c46_iter2/run_gate.sh` + `run_arm_twin.sh` + `score_twin.sh` +
`build_report.py`), reused with only the pre-registered new-envelope changes below.

## Changes vs iteration-2 (ONLY these)
- **(a) TWIN ARM ONLY.** The AR comparator (**12/48**) and the twin gate-OFF arm
  (**1/48**) are BANKED in `runs/k_gate_c46_iter2/ar_paired_report.json` and are NOT
  re-run. The report reads McNemar against BOTH, per-instance, over the same 48 ids.
- **(b) gate-ON boot.** The twin server launches with
  `VLLM_FASTDLLM_W1_DRAFT_VERIFY=1` — the **W-2 causal fixed-width block-commit
  draft-verify** path (engine pin `qwen3_5-flare-modelstate @ b92af2d`, LOCAL, never
  pushed; W-2 commit `786ed3d`). `VLLM_FLARE_BIDIR_PROBE` stays the serve-script default
  (`=1`, the C46-iter2 envelope). Two asserts:
    - engine-source assert (orchestrator preflight): the pinned `qwen3_5_flare.py`
      carries the `RUNG W-2 (byte-faithful redesign)` causal-verify marker;
    - server-log assert (`run_arm_twin.sh`, fail-closed before any episode fans out):
      `FLARE W-1b copy draft-and-verify gate: True`. A gate-OFF boot aborts (exit 5) so
      it can never silently duplicate the banked gate-OFF twin.
  The live `w1[on=True spans=.. toks=.. vfwd=.. rej=.. arej=..]` per-turn counters are
  the runtime proof the causal verify forward actually fires.
- **(c) CERTIFIED read-clamp proxy active** (`runs/k_gate_c46/proxy_readclamp.py`, cert
  `7ae55d4`, `LUMO_PROXY_READCLAMP_LIMIT=100`) — identical wiring to iteration-2.
- **(d) output = `runs/k_gate_c46_newenv/`.**
- **(e) w1 telemetry recorded.** After teardown + scoring, `parse_w1_telemetry.py` parses
  the gate-ON server log (per-turn `model_forwards`/`generated_tokens` + the cumulative
  `w1[..]` block) and the banked gate-OFF server log, writing `w1_telemetry.json`:
  live **blended tok/fwd** gate-ON vs gate-OFF, **wall/episode**, the final cumulative w1
  counters, and the **arej total (must be 0)**.

## Envelope (frozen IDENTICAL to the banked gate-OFF twin)
mask 248077, max_model_len 32768, **gmu 0.74 / max_num_seqs 4**, temp 0.6 / top_p 0.95 /
top_k 20 (NO presence_penalty), per-shard base seeds {1234,101234,201234,301234}, turn
cap 75, empty-patch re-drive 1, c=4. Same frozen pool `runs/k_gate_c46/shard_plan.json`
(pool_sha256 `49d8f46dc202bf50…`), same official swebench-harness scoring, same
**>=12/46** entry bar. MEMORY-BUDGET RULE honored: diffusion gmu 0.74 (never the AR arm's
0.85; the GDN align-cache lives outside the KV pool).

## Runner (self-bounded, one server)
`run_gate.sh` (pidfile `gate.pid`; detached via `setsid`; server caged via
`systemd-run --user --scope --unit=c46ne_diff_server`): W-2 engine-source assert +
verify 48 images + GPU-idle preflight -> twin gate-ON+clamp arm to completion + teardown
+ GPU-settle -> OFFICIAL docker scoring (server DOWN) -> `parse_w1_telemetry.py` ->
`build_report.py` (resolve@1 vs **>=12/46**; McNemar vs BOTH banked comparators;
ctx_overflow buckets; **arej-must-be-0** -> `VERDICT=INVALID-AREJ-NONZERO` if it fires;
live tok/fwd + wall/episode covariates). `[state]` lines (`eps done / 48`, wall) emit to
`logs/run_gate.log` every 60 s. **STOP-file** `runs/k_gate_c46_newenv/STOP` aborts
gracefully (server torn down, exit 9). Docker via the docker group (plain `docker`).

## Expected
Wall on the order of the banked gate-OFF twin (~4.15 h for 48 episodes at c=4) or somewhat
less given the W-2 ~2.55× decode wall-clock speedup; self-bounded by the 25-min agent wall
per episode. The report writes `report.md`/`report.json` + `w1_telemetry.json`. Verdict:
`ENTRY-PASS` (>=12/46) vs `INCONCLUSIVE-BY-POWER` (<12/46, a principled stop adjudicated
against the banked paired reads), gated on arej==0.

## OUTCOME — LAUNCH BLOCKED (see `BLOCKER.md`)
Launched + verified (runner alive, gate-ON asserted, arej=0), then **STOPPED at eps=3/48**
(server torn down, GPU settled). The verify sub-item "w1 span commits firing" **failed
structurally**: the W-2 draft-verify only engages at **temperature==0**
(`qwen3_5_flare.py:2460`), but the frozen envelope forces **temp 0.6** (confirmed: 97
proxy-forwarded requests carried `"temperature": 0.6`). The gate is ON but **dormant**
(`spans=0 vfwd=0 rej=0` across all 74 turns, incl. 300+-value-token write turns) — the run
would produce a null speed result and a resolve@1 byte-identical to the banked gate-OFF
twin. The paired temp-0.6 envelope and the temp-0 W-2 gate are mutually exclusive under the
current engine pin; parent decision required (options in `BLOCKER.md`). Infra is correct and
reusable for whichever option is chosen.
