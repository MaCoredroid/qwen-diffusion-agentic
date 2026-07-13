# SWE-SFT windowed dataset (iteration-2, shape-corrected) — report

Built 2026-07-13T15:21:59Z by `runs/swe_datagen_s1/build_windowed_dataset.py` (block 12288, cap 6/episode, seed 71101) from `runs/swe_datagen_s1/keepers/keepers.jsonl` (383 keepers).

## Label retention (the headline vs iteration-1)

- iteration-1 front-truncation @12288: **69.88%** (911,531/1,304,354)
- iteration-2 windowing: **100.0%** (1,454,824/1,454,824); pre-cap 100.0%
- **delta: +30.12 pp** (beats 69.9%: True)

## Window counts

- total windows: **987** across 383 episodes (mean 2.577/episode, max 4)
- windows/episode (pre-cap) histogram: {1: 48, 2: 77, 3: 247, 4: 11}
- episodes hitting the cap: 0 (labels dropped by cap: 0)

## Window-position histogram (early/mid/late coverage)

- windows by position: {'early': 176, 'late': 462, 'mid': 301, 'full': 48}
- **emitted** label tokens by episode-third (iter2): {'early': 292290, 'late': 775381, 'mid': 387153}
- retained label tokens by third — iter1 FRONT-TRUNC (late-skewed): {'mid': 278213, 'late': 756413, 'early': 453}
- all label tokens by third (full episode, denominator): {'early': 195411, 'mid': 502779, 'late': 756634}

## Serve-exact spot-audit

- audited 10 random windows; ALL-PASS = **True** (exact-slice + turn-boundary + decoded-byte-identical + spans-wrap-assistant + block-fit)

## Block-fit

- windows over block 12288: **0** (max window len 12286) — guaranteed <= block by construction

## Leakage (no-op check)

- holdout 113 sha==pin; keeper∩holdout **0**; windows from keeper episodes only (external text: NONE)

## System/task coverage recovered

- episodes whose earliest window still carries system+task (loss-masked): **333/383** (86.9%) — iteration-1 dropped it on all 328 truncated episodes.

## Outputs

- dataset: `data/swe_sft_pool/train_swe_sft_windowed.tokenized.jsonl` (sha256 909c92eae5d9e576..)
- audit: `data/swe_sft_pool/windowed_dataset_audit.json`
- report: `data/swe_sft_pool/windowed_dataset_report.md`

## Final rebuild (pre-registered)

- MECHANICAL re-run of this exact script (same seed/block/cap) on the post-promotion keepers.jsonl — no design changes.
