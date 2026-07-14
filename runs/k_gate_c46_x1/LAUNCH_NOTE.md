# C46 GATE ON THE X.1 TWIN — LAUNCH NOTE

**Launched 2026-07-14, detached (`setsid` + systemd `--user --scope` cage), single twin arm.**
The C46 re-gate on the **X.1 read-grounding twin** (`models/qwen3.5-9b-fastdllm-mswe2-S-x1-vllm-bf16`),
served **gate-ON** with the **W-2/W-2b causal draft-verify** path + the **CERTIFIED read-clamp** shim,
decoded on the frozen Tier1-C46 48-instance slice. This mirrors the newenv gate infra
(`runs/k_gate_c46_newenv/{run_gate.sh,run_arm_twin.sh,score_twin.sh,build_report.py,parse_w1_telemetry.py}`)
with **ONLY the model swapped** to the X.1 export. Output = `runs/k_gate_c46_x1/`.

## STEP 1 — PRE-GATE KILL-T1 (PASSED, launch authorized)
Full matched-20 `exact_args` battery on the X.1 twin, FLARE hybrid_clean K=1 (offline HF stack =
served by the A6 online==offline byte cert), the §2.3-canonical anchor battery
(`eval_flare_northstar_hybrid_clean.py`, the same instrument iteration-2 / step4_cert used), paired
per-turn vs the **banked iteration-2 lineage anchor** (twin@K1's own matched-20, 49/63) via
`swe_sft_arm1_anchor_mcnemar.py` with gold_sha256 identity.
- `runs/kraise_reconvert_iter2_x1/battery/killt1_m20/killt1_m20_mcnemar.json`
- **anchor (pre)** 49/63 exact_args · **X.1 (post)** 50/63 exact_args · valid 62/63
- **McNemar b=0 (erosion) / c=1 (gain) → net-loss −1**, p=1.0 · gold_sha256 mismatches 0/63
- gates: raw ≥ anchor−3 (50 ≥ 46) PASS · net-loss NOT significant (p≥0.05) PASS → **VERDICT PASS**
- read as: the read-grounding-weighted re-conversion did **not** regress the certified tool-call
  argument capability (zero exact erosion; one gain). Tool-calls intact → gate launch authorized.

## Config (frozen IDENTICAL to the newenv twin; ONLY the model differs)
- **model**: `models/qwen3.5-9b-fastdllm-mswe2-S-x1-vllm-bf16` (X.1 = read-grounding-weighted
  re-conversion of the SAME iter2 base `mswe-S-iter2-merged`; adapter
  `runs/kraise_reconvert_iter2_x1/mswe2_S_x1_readground_step800_seed81101`, rww=5.0, 800 steps).
- **gate-ON**: `VLLM_FASTDLLM_W1_DRAFT_VERIFY=1` (W-2 causal block-commit + W-2b temp>0 verify);
  engine pin `qwen3_5-flare-modelstate @ 41fd28e` (LOCAL, never pushed; W-2 786ed3d + W-2b 41fd28e).
  `run_arm_twin.sh` hard-asserts `FLARE W-1b copy draft-and-verify gate: True` before any episode
  fans out; the orchestrator asserts the `RUNG W-2 (byte-faithful redesign)` source marker pre-boot.
- **DORMANT_PREFLIGHT (RUNG W-2b)**: after episode 1, assert cumulative `w1[on=True spans>0]`; a
  dormant gate (spans=0) aborts the run as a byte-identical placebo (`DORMANT_GATE.txt`, exit 8).
  W-2b fires the verify at the frozen temp 0.6 (newenv proved spans=12099 over 48 eps).
- **CERTIFIED read-clamp** (`runs/k_gate_c46/proxy_readclamp.py`, cert 7ae55d4,
  `LUMO_PROXY_READCLAMP_LIMIT=100`) — identical wiring to newenv.
- **envelope**: mask 248077, max_model_len 32768, gmu 0.74 / max_num_seqs 4, temp 0.6 / top_p 0.95 /
  top_k 20 (NO presence_penalty), per-shard base seeds {1234,101234,201234,301234}, turn cap 75,
  empty-patch re-drive 1, c=4. Same frozen pool `runs/k_gate_c46/shard_plan.json`
  (pool_sha256 `49d8f46dc202bf50…`), same official swebench-harness scoring, same **≥12/46** entry bar.

## Report builders (`build_report.py`) compare vs
- **(a) PRIMARY X.1-effect**: the **newenv twin gate-ON (banked, 3/48)** — byte-for-byte IDENTICAL
  config (same envelope/gate/clamp/48 ids); the ONLY delta is the model, so the McNemar X.1-vs-newenv
  is the clean read of the read-grounding conversion's effect on resolve@1.
- **(b)** AR arm (banked, 12/48) · **(c)** entry floor **≥12/46**.
- plus: ctx_overflow buckets, `arej`-must-be-0 (→ `VERDICT=INVALID-AREJ-NONZERO` if it fires), live
  blended tok/fwd + wall/episode covariates (gate-ON vs banked gate-OFF).

## Runner (self-bounded, one server)
`run_gate.sh` (pidfile `gate.pid`; detached via `setsid`; server caged via
`systemd-run --user --scope --unit=c46x1_diff_server`): W-2 engine-source assert + verify 48 images +
GPU-idle preflight → twin gate-ON+clamp arm to completion + teardown + GPU-settle → OFFICIAL docker
scoring (server DOWN) → `parse_w1_telemetry.py` → `build_report.py`. `[state]` lines emit to
`logs/run_gate.log` every 60 s. **STOP-file** `runs/k_gate_c46_x1/STOP` aborts gracefully (exit 9).

## Expected
Wall ~4 h for 48 episodes at c=4 (newenv gate-ON was 13305 s / 277 s per episode). Verdict:
`ENTRY-PASS` (≥12/46 → opens the golden-number N=50 run) vs `INCONCLUSIVE-BY-POWER` (<12/46), gated on
`arej==0`. The PRIMARY read is whether the read-grounding re-alignment lifts resolve@1 above the newenv
twin's 3/48 (identical-config model-only delta) toward the AR ceiling (12/48).
