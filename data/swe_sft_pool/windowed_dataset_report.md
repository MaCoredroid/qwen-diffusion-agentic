# SWE-SFT windowed dataset (iteration-2, shape-corrected) — report

Built 2026-07-13T12:39:11Z by `runs/swe_datagen_s1/build_windowed_dataset.py` (block 12288, cap 6/episode, seed 71101) from `runs/swe_datagen_s1/keepers/keepers.jsonl` (334 keepers).

## Label retention (the headline vs iteration-1)

- iteration-1 front-truncation @12288: **69.88%** (911,531/1,304,354)
- iteration-2 windowing: **100.0%** (1,304,354/1,304,354); pre-cap 100.0%
- **delta: +30.12 pp** (beats 69.9%: True)

## Window counts

- total windows: **889** across 334 episodes (mean 2.662/episode, max 4)
- windows/episode (pre-cap) histogram: {1: 26, 2: 70, 3: 229, 4: 9}
- episodes hitting the cap: 0 (labels dropped by cap: 0)

## Window-position histogram (early/mid/late coverage)

- windows by position: {'early': 166, 'late': 424, 'mid': 273, 'full': 26}
- **emitted** label tokens by episode-third (iter2): {'early': 279674, 'late': 676671, 'mid': 348009}
- retained label tokens by third — iter1 FRONT-TRUNC (late-skewed): {'mid': 263145, 'late': 647933, 'early': 453}
- all label tokens by third (full episode, denominator): {'early': 189297, 'mid': 467019, 'late': 648038}

## Serve-exact spot-audit

- audited 10 random windows; ALL-PASS = **True** (exact-slice + turn-boundary + decoded-byte-identical + spans-wrap-assistant + block-fit)

## Block-fit

- windows over block 12288: **0** (max window len 12286) — guaranteed <= block by construction

## Leakage (no-op check)

- holdout 113 sha==pin; keeper∩holdout **0**; windows from keeper episodes only (external text: NONE)

## System/task coverage recovered

- episodes whose earliest window still carries system+task (loss-masked): **312/334** (93.4%) — iteration-1 dropped it on all 328 truncated episodes.

## Outputs

- dataset: `data/swe_sft_pool/train_swe_sft_windowed.tokenized.jsonl` (sha256 30ff60456f07ba6d..)
- audit: `data/swe_sft_pool/windowed_dataset_audit.json`
- report: `data/swe_sft_pool/windowed_dataset_report.md`

## Final rebuild (pre-registered)

- MECHANICAL re-run of this exact script (same seed/block/cap) on the post-promotion keepers.jsonl — no design changes.
