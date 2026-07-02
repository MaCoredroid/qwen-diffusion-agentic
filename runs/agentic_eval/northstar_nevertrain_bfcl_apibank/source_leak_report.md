# Never-Train Source and Leak Report

Slice: 60 episodes / 184 generated turns.

## Sources

| Source | Episodes | Turns | Notes |
|---|---:|---:|---|
| BFCL-multi_turn | 27 | 147 | BFCL V4 non-live multi-turn eval categories: base, miss_func, miss_param, long_context. |
| BFCL-AST | 8 | 12 | BFCL V4 non-live AST categories: multiple, parallel, parallel_multiple, simple_python. |
| API-Bank-Lv1 | 13 | 13 | Raw DAMO `lv1-lv2-samples/level-1-given-desc`; Level-1 files only. |
| API-Bank-Lv2 | 12 | 12 | Raw DAMO `lv1-lv2-samples/level-1-given-desc`; Level-2 files only. |

BFCL came from `ShishirPatil/gorilla` eval-designated non-live BFCL V4 files, upstream repo license Apache-2.0. API-Bank came from raw `AlibabaResearch/DAMO-ConvAI` API-Bank files, upstream repo license MIT. API-Bank Level-3/toolsearcher files were excluded. xLAM stayed excluded because the local probe remains gated.

## Explicit Overlap Check

Compared against 92 local train files, 24,505 train rows, and 14,276 indexed train tool calls.

| Check | Count |
|---|---:|
| Canonical text overlap | 0 |
| User prompt overlap | 0 |
| Full tool-signature overlap | 0 |
| Same-tool/all-arg-value overlap | 0 |
| Descriptive tool-name overlap | 0 |
| Episodes with any hard overlap | 0 |

Verdict: no hard overlap found. The manifest records the method and per-row overlap results in `data/toolcall_eval_native/flare_nevertrain_bfcl_apibank.manifest.json` and `data/toolcall_eval_native/flare_nevertrain_bfcl_apibank.overlap_rows.jsonl`.

## Difficulty

| Source | Calls/turn | Args/call | Avg value length | Copy args | Derived/constant args |
|---|---:|---:|---:|---:|---:|
| BFCL-multi_turn | 1.000 | 1.544 | 14.629 | 150 | 81 |
| BFCL-AST | 1.000 | 2.167 | 2.033 | 7 | 20 |
| API-Bank-Lv1 | 1.000 | 2.769 | 11.103 | 26 | 10 |
| API-Bank-Lv2 | 1.000 | 2.750 | 13.438 | 17 | 16 |

## Matched Eval Headline

Diffusion per-call waves beats AR-guided on exact arguments by +104/184 turns and +38/60 episodes, with no AR-guided-only exact-argument flips. It is slower end-to-end on this mixed slice: 160.983s diffusion vs 108.684s AR-guided, or 1.481x diffusion/AR-guided wall ratio.
