# Stage 0 — Phase 2 PROBE (SWE-Gym env + generation + official-filter pricing)

Prices the SWE-Gym data-gen path the campaign (`swe_tuning_campaign_design.md`) flagged as
its dominant risk. 20 SWE-Gym instances, stock-AR generator @ concurrency 4, official
resolve-filter. See `report.json` / `report_table.txt` for the measured numbers.

## Toolchain (what this run stands up)

- **Instance pool** — `artifacts/subset_probe20.json`: 20 SWE-Gym instances, 2× each of the
  10 repos the SWE-Bench-Fork spec map covers (MONAI excluded — absent from the fork map).
  All are `source=SWE-Gym` rows in `data/swe_sft_pool/pool_manifest.json` (KILL-D1 firewall
  clean: disjoint from verified_500 ∪ Tier0 ∪ Tier1 by construction).
- **Env acquisition** — `pull_and_tag.sh`: **docker pull of the official prebuilt
  `xingyaoww/sweb.eval.x86_64.<id: __→_s_>` image** (NOT a from-scratch env build), re-tagged
  to the two local keys the toolchain needs: the qwen-code driver key
  `swebench/sweb.eval.x86_64.<id: __→_1776_>:latest` and the fork scorer key
  `sweb.eval.x86_64.<id>:latest`.
- **Generation** — `probe_gen.sh`: ONE stock-AR vLLM server at `--max-num-seqs 4` (RAM cage),
  4 qwen-code driver shards run concurrently (episode-in-official-container, `--eval-mode
  skip` → predictions only), GPU util sampled to `gen/gpu_util.csv`.
- **Official filter** — `probe_score.sh`: **SWE-Gym/SWE-Bench-Fork** harness (`@242429c`),
  one-line patch (`artifacts/fork_reuse_prebuilt.patch`) so `build_instance_image` reuses the
  pre-pulled instance image instead of requiring a from-scratch env-image build. Keeps only
  `resolved=true` on the ground-truth FAIL_TO_PASS + PASS_TO_PASS gate.
  - **Net-fetch hardening** (`artifacts/fork_netfetch_hardening.patch`, 2026-07-09): the fork
    eagerly fetches each instance's `environment.yml`/`requirements.txt` from
    raw.githubusercontent.com with a bare, **unbounded** `requests.get()`. A single read
    timeout there once propagated out of `make_test_spec` and **killed a whole 50-instance
    scoring run** (batch_0002 -> 0 sub-reports, 49 no_prediction). The patch wraps those
    fetches (`swebench/harness/utils.py::fetch_raw_file`) with an explicit 30s timeout, 3
    backoff retries, and an immutable on-disk cache (`<fork>/.raw_fetch_cache`), and on final
    failure raises `EnvFetchError` which `safe_make_test_specs` catches **per-instance** —
    skipping just that instance (it lands as an error/unevaluated id) instead of aborting the
    run. Smoke: `tests/test_raw_fetch_hardening.py` (localhost only, no net/docker).
- **Orchestrator** — `probe_orch.sh`: serial A(pull) → B(gen, server up) → C(score) →
  D(report); each stage self-bounded; docker-heavy pull/score never overlap the GPU server.

## Reproduce

```
bash runs/stage0_swegym_probe/probe_orch.sh    # (needs SUDO_ASKPASS exported)
```

## Validation (de-risked before the 20-run)

- scoring toolchain: gold patch on `facebookresearch__hydra-1006` → `resolved=true`
  (patched fork reuses the pulled image; env-image build correctly skipped).
- generation path: 1-instance smoke → episode ran in the seeded SWE-Gym container, produced a
  patch; identical exit-code/wall profile to the certified `stage_c_n5v2` stock-AR 4/5 run.
