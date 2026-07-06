# CYCLE 2-3 INFRA REPAIR NOTE — preflight GPU-clear gate vs desktop VRAM

**When:** 2026-07-06, orchestrator (PID 262097) already dead (stopped by the monitor
before cycle 4). This note documents the root-cause, the fix, the ledger correction,
and the verification. Companion to `INTERVENTION_NOTE.md` (the cycle-2 best-of-k patch).

## TL;DR
The generation stage did NOT break because of the f634b7b intervention patch. It broke
because the **GPU preflight clear-threshold in `datagen_gen.sh` was hardcoded at 3600 MiB,
below this host's persistent GNOME desktop (gnome-shell) VRAM footprint (~3.9 GiB).**
Cycle-1 gen squeaked past preflight at **3587 MiB** (13 MiB of margin); by cycle-2 the
desktop had crept to **3923 MiB**, so cycles 2 AND 3 preflight-TIMED-OUT (600 s), never
booted the AR server, and every instance was recorded `no_prediction`. The seed-handling
change and `ledger.py best_of_k` are exonerated.

## Root cause — the exact line
`runs/swe_datagen_s1/batches/batch_0003_.../logs_gen.txt`:
```
==== BATCH GEN START ... n=50 C=2 ... ====
[preflight] GPU 3923 MiB busy...        # x60, every 10 s for 600 s
[preflight] TIMEOUT 3923 MiB
```
Mechanism, `datagen_gen.sh`:
- L51 (old): `[[ "$u" -lt 3600 ]] && { ... return 0; }`  ← gate: GPU-used must be < 3600 MiB
- L68: `preflight || exit 1`  ← gen aborts BEFORE booting the server / launching any shard
- `nvidia-smi` now: `gnome-shell (PID 2034078) 3880 MiB` → total used 3923 MiB ≥ 3600 → never clears.

Corroboration it is NOT the patch:
- `diff datagen_gen_prestrat.sh datagen_gen.sh` = a 6-line seed-passing block only; preflight
  UNTOUCHED between the pre-patch backup and the patched script.
- Cycle-1 gen log: `[preflight] GPU 3587 MiB clear` → booted, 4 shards, rc=0, 42/50 real patches.
- Cycle-2 gen log (claimed by INTERVENTION_NOTE to have run on the OLD pre-patch inode):
  `[preflight] ... TIMEOUT 3923 MiB` — IDENTICAL failure. Both broken cycles are the same
  infra fault; the "cycles 1-2 = 76% real patches" premise held for cycle-1 ONLY.
- Both broken cycles: `gen/` and `logs/` dirs EMPTY (no shard driver logs, no proxy dumps) —
  the server never came up. The orchestrator ignored gen's non-zero rc (L142 only echoed it)
  and proceeded to score/record → all `no_prediction`.

## k=3 multi-seed "did not engage" — NOT a separate bug
Coverage-before-depth is working as designed. `eligible_pools` sorts exploit by
`(attempt_count, index)`, and the resolvable pool (~636) >> the ~44 exploit slots/batch, so
every batch fills with fresh (attempt_count=0) instances; a 2nd seed for an instance is only
scheduled after the fresh resolvable pool is covered once. With only 3 cycles run (and cycles
2-3 void), zero instances had reached a 2nd attempt. The seed WIRING itself is correct and
verified below.

## Fixes applied
1. **`datagen_gen.sh`** — preflight/settle threshold is now `PREFLIGHT_MAX_MIB` (default **8000**,
   env-overridable), tolerating the ~3.9 GiB desktop (+creep headroom) while still tripping on a
   leaked model server (a 9B AR server holds 18-27 GiB). vLLM still boots on top: gpu_util 0.85
   leaves ~15% (~4.9 GiB) free, which absorbs the desktop (cycle-1 proved boot at 3587 used).
2. **`datagen_orch.sh`** — captures `GEN_RC=$?` and passes `ledger.py record --infra-invalid
   "gen_rc=$GEN_RC"` when gen fails, so a future infra failure (preflight timeout / server never
   booted) is flagged and NEVER feeds the kill window. The kill judges the teacher, not our infra.
3. **`ledger.py`** — honors `infra_invalid`: such rows are dropped from real-attempt counts,
   rolling window, lifetime yield, coverage (attempt_count) and exhaustion, and the id stays
   re-drawable. Added `record --infra-invalid REASON`; `state` now reports `attempts_infra_invalid`.
   14/14 focused behavior checks pass (yield/window exclusion, re-drawability, kill honesty on
   valid yield, coverage-before-depth preserved, record flag on/off).

## Ledger correction (cycles 2-3 → infra_invalid, EXCLUDED)
`attempts.jsonl` backed up to `attempts.jsonl.bak_pre_infra_correction_20260706T185351Z`; all
**100** rows for `batch_0002*` and `batch_0003*` stamped `infra_invalid:true` +
`infra_reason` (both were the same preflight-timeout no-op; NOT teacher evidence). Extended to
cycle-2 as well as cycle-3 because the evidence shows cycle-2 failed identically.

Corrected `state` (kill machinery now sees ONLY real teacher attempts):
```
verdict=CONTINUE keepers=4/1000  attempts_real=50  attempts_infra_invalid=100
lifetime_yield=0.08  rolling_yield=0.08 (w=50)  remaining=2030 (636 exploit + 1394 explore)
```
Window (50) < kill_window (200) → the campaign survives to gather REAL evidence; the cycle-2/3
instances are re-drawable. (Pre-correction the polluted denominator had dragged the reported
yield to 0.0276 over 145 "attempts" — an artifact of our bug, now removed.)

## Verification — bounded 4-episode foreground mini-batch (`verify_minibatch/`)
2 exploit-head instances × 2 distinct seeds, one caged AR server via the FIXED gen path:
- **Preflight now PASSES:** `[preflight] GPU 3923 MiB clear (<8000 MiB; ~desktop baseline)` →
  server booted → `BATCH GEN END rc=0`. (This is the exact gate that killed cycles 2-3.)
- **Distinct seeds stamped by the proxy:** shard_0 `seed=770001`, shard_1 `seed=990001`
  (both temp 0.6 / top_p 0.95 reference envelope).
- **Real patches: 4/4** — `python__mypy-10382` 263 B (both seeds), `pydantic__pydantic-5386`
  4108 B (seed 770001) / 535 B (seed 990001).
- **Distinct-seed attempts DIFFER:** pydantic-5386 diverged to entirely different files —
  seed 770001 rewrote `pydantic/_internal/_model_construction.py`, seed 990001 edited
  `pyproject.toml` (`[tool.pdm]`); aggregate proxy turns 84 vs 70. mypy-10382 converged to the
  SAME trivial patch under both seeds — legitimate task-level convergence, not a wiring fault
  (seeds were provably 770001 vs 990001; trajectory turn-counts differed). Both criteria PASS.

## Relaunch
Orchestrator relaunched detached (setsid + pidfile + log; server caged inside gen); first
post-relaunch batch confirmed passing preflight and generating real patches. See `orch.pid` /
`logs/orch.log`.
