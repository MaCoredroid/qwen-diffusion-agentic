# SWE-Gym rescue — ledger correction (2026-07-08)

**What:** re-scored the paid-for SWE-Gym episodes that the fork-harness crash
(`environment_setup_commit: None` -> `os.path.join` TypeError, KILL_AUTOPSY §4B)
silently discarded across the 2026-07-07 relaunch batches, then flipped their rows
in `attempts.jsonl` from `infra_invalid` to their TRUE harness verdicts and extracted
keepers from the newly-resolved episodes.

**How (pull-free, docker-only, orch-safe):** `rescue_score_local.sh` re-runs the
SWE-Bench-Fork harness over the EXISTING on-disk gym predictions using the corrected
`dataset_gym.json` (env_setup_commit dropped), but ONLY for gym ids whose instance
image is already present locally (retag driver/xingyaoww -> fork key; never pull),
capped at 6 workers so the live orchestrator's own score phase is never starved. No
new generation; no GPU.

## Scored this rescue (local-image, corrected-dataset re-score)

| original batch | family mix | scored | resolved | unresolved | empty |
|---|---|--:|--:|--:|--:|
| batch_0001_…T024659Z | modin | 6 | 3 | 2 | 1 |
| batch_0011_…T101832Z | conan/dvc/pydantic/mypy | 9 | 1 | 8 | 0 |
| batch_0012_…T110240Z | pydantic/mypy/dvc/conan | 35 | 10 | 23 | 2 |
| **TOTAL** | | **50** | **14** | **33** | **3** |

## MEASURED gym-tail family yields (resolved / attempted, scorer-running)

| family | resolved | attempted | yield |
|---|--:|--:|--:|
| modin-project | 3 | 6 | 0.500 |
| iterative (dvc) | 5 | 14 | 0.357 |
| conan-io | 1 | 4 | 0.250 |
| python (mypy) | 3 | 14 | 0.214 |
| pydantic | 2 | 12 | 0.167 |
| **blended** | **14** | **50** | **0.280** |

Blended 0.28 is **2.5× the autopsy's pessimistic scored_y (0.111)** — the gym tail is
materially more resolvable than the collapse-slice (all-`no_prediction`) suggested,
confirming the L2 lever thesis with real official-harness verdicts.

## Corrected ledger (post-flip, post-extract)

| metric | pre-rescue | post-rescue |
|---|--:|--:|
| keepers | 218 | **232** (+14) |
| attempts_real | 728 | 778 |
| attempts_infra_invalid | 509 | 459 |
| lifetime_yield | 0.2995 | 0.2982 |
| rolling_yield (w=200) | 0.415 | 0.400 |
| verdict | CONTINUE | CONTINUE |

New keepers (14, all `kept_new`, 0 `already_had` → orch had not re-drawn any):
modin-6298/6333/6369, pydantic-9066/9287, conan-9431, dvc-3726/3727/3797/3836/3891,
mypy-11567/11632/11680.

Batches 0002/0003/0006–0010 (all-modin, 0 local images) and the non-local remainder
of 0011–0014 are LEFT re-drawable `infra_invalid` — the live orch, with the now-fixed
scorer, will re-draw and score them natively. Batches 0004/0005 had no predictions
(gen never booted) and are genuinely re-drawable.

## Race-safe flip
`apply_rescue.py --apply`: re-reads `attempts.jsonl` immediately before an atomic
temp+rename, touches ONLY rows flagged `infra_invalid` with reason
`fork_harness_no_report_env_setup_commit_none` that a rescue report scored, verifies
row-count monotonicity (no lost orch appends), and stamps each flipped row
`rescued=true` + `rescored_at`.

## Keepers
`extract_keepers.py` over each rescued batch's resolved ids (trajectories from the
ORIGINAL 07-07 gen dirs), idempotent instance_id dedup vs the existing 218 and vs any
id the live orch re-drew meanwhile.
