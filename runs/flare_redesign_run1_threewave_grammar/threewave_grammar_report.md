# Three-Wave Grammar Wave-1 Comparison

Run-1 checkpoint, raw value lane, native heldout/public slices. Wave 2 is raw `argument_value` parallel commit at value tau 0.80 and 0.95. Condition C is the previously measured confidence-bulk wave-1 contrast.

## Headline

- Condition B (careful wave 1) does not reproduce the native exact-args anchor: best B is tau 0.95 with heldout 2/12 and public 4/12 exact args, below the requested 3/12 heldout and 8/12 public anchor.
- Condition A (grammar-projected wave 1) materially lifts blended TPF while holding or improving B's quality, but still misses the public exactness anchor: best A is tau 0.95 with heldout 4/12 and public 6/12 exact args.
- Condition C (confidence-bulk wave 1) remains the negative contrast: 0/12 exact args on both slices at tau 0.80 and tau 0.95, with degraded validity.
- Values were not forced or projected in A/B/C: all value force counters are zero in every row.

## Results

| Cond | Mode | Value tau | Split | Quality | Blended TPF | Scaffold TPF | Value TPF | sec/rec | Wave1 projected | Value force pass |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| A | grammar_projected | 0.80 | heldout | 3/12 args, 8/12 seq, 11/12 valid | 1.741 | 15.275 | 1.037 | 8.69 | 984 | yes |
| A | grammar_projected | 0.80 | public | 6/12 args, 9/12 seq, 10/12 valid | 1.932 | 10.096 | 1.045 | 6.33 | 852 | yes |
| A | grammar_projected | 0.95 | heldout | 4/12 args, 8/12 seq, 11/12 valid | 1.656 | 10.038 | 1.002 | 8.99 | 948 | yes |
| A | grammar_projected | 0.95 | public | 6/12 args, 9/12 seq, 10/12 valid | 1.887 | 10.096 | 1.003 | 6.39 | 852 | yes |
| B | careful | 0.80 | heldout | 0/12 args, 7/12 seq, 9/12 valid | 1.031 | 1.029 | 1.047 | 16.19 | 0 | yes |
| B | careful | 0.80 | public | 1/12 args, 7/12 seq, 11/12 valid | 1.039 | 1.03 | 1.067 | 11.56 | 0 | yes |
| B | careful | 0.95 | heldout | 2/12 args, 8/12 seq, 10/12 valid | 1.02 | 1.036 | 1.003 | 16.05 | 0 | yes |
| B | careful | 0.95 | public | 4/12 args, 6/12 seq, 11/12 valid | 1.025 | 1.037 | 1.01 | 11.58 | 0 | yes |
| C | confidence_bulk | 0.80 | heldout | 0/12 args, 1/12 seq, 7/12 valid | 1.017 | 1.014 | 1.046 | 13.1 | 0 | yes |
| C | confidence_bulk | 0.80 | public | 0/12 args, 3/12 seq, 8/12 valid | 1.024 | 1.024 | 1.067 | 8.75 | 0 | yes |
| C | confidence_bulk | 0.95 | heldout | 0/12 args, 1/12 seq, 7/12 valid | 1.005 | 1.016 | 1.013 | 12.07 | 0 | yes |
| C | confidence_bulk | 0.95 | public | 0/12 args, 3/12 seq, 8/12 valid | 1.02 | 1.025 | 1.029 | 8.3 | 0 | yes |

## Interpretation

A answers the narrow red-team: confidence-bulk scaffold was the wrong wave-1 implementation. Grammar-projected scaffold recovers validity and tool sequence quality while lifting blended TPF to 1.66-1.93 on the native slices.

The decisive negative is different: the careful left-context-first rescue from the copy-span slice does not generalize to native exact arguments. Even with conservative value tau 0.95, B reaches only 2/12 heldout and 4/12 public exact args. A is faster and better than B, but public exact args remain 6/12, below the 8/12 public anchor.

The honest speed ceiling at held exactness on both slices is therefore not established. There is a material speed signal for grammar-projected wave 1, but no operating point here preserves the requested public exactness quality.

Machine-readable report: `runs/flare_redesign_run1_threewave_grammar/threewave_grammar_report.json`
