# Per-Call Wave Schedule Report

Run-1 checkpoint, raw value lane. Main rows are the requested per-call waves at value tau 0.80 and 0.95; tau 0.99 is an extra conservative diagnostic for the remaining public miss.

## Per-Row Diff

- Old A lost public rows 1 and 9 versus the plain-careful 8/12 anchor. Row 1 is a cross-call separator/stop issue and is fixed by per-call waves. Row 9 is value corruption and remains the single public miss after per-call waves.
- Old B lost public rows 4, 5, 6, and 7. Rows 4-6 are tool-name scaffold corruptions; row 7 is a value/close-boundary corruption.

## Sweep

| Tau | Split | Quality | Blended TPF | Scaffold TPF | Value TPF | sec/rec | Wave1 projected | Value force pass |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 0.80 | heldout | 5/12 args, 11/12 seq, 11/12 valid | 1.852 | 6.39 | 1.056 | 9.37 | 1201 | yes |
| 0.80 | public | 6/12 args, 9/12 seq, 9/12 valid | 2.547 | 8.268 | 1.084 | 5.45 | 1112 | yes |
| 0.95 | heldout | 5/12 args, 11/12 seq, 11/12 valid | 1.795 | 6.39 | 1.018 | 9.58 | 1201 | yes |
| 0.95 | public | 7/12 args, 10/12 seq, 10/12 valid | 2.465 | 8.322 | 1.036 | 5.55 | 1112 | yes |
| 0.99 | heldout | 5/12 args, 11/12 seq, 11/12 valid | 1.783 | 6.39 | 1.009 | 9.63 | 1201 | yes |
| 0.99 | public | 7/12 args, 10/12 seq, 10/12 valid | 2.421 | 8.322 | 1.013 | 5.66 | 1112 | yes |

## Verdict

Per-call waves are a real improvement over whole-block A: tau 0.95 moves public exact_args from 6/12 to 7/12 and keeps heldout at 5/12, with blended TPF 1.80 heldout and 2.47 public.

The target is not met: public remains below the 8/12 anchor. The remaining row 9 miss persists even at tau 0.99, so it is not explained solely by whole-block right-context infill or residual value parallelism at tau 0.95.

Values remained raw: value force counters are zero in all per-call rows.
