# SWE-SFT dataset manifest

Built 2026-07-09T18:45:33Z by `runs/swe_datagen_s1/build_swe_sft_dataset.py` from `runs/swe_datagen_s1/keepers/keepers.jsonl` (323 keepers).

## Firewall (KILL-D1)

- eval holdout: **113 ids**, sha256 `c56f473ad31e52bee0f85151562f4e2122e4815dfa3f1b776b15fe121e8d168e` == pin **OK**
- keeper x holdout overlap: **0** / 323
- quarantined excluded: **['pydata__xarray-6461']** (0 in train)

## Render / native format

- authoritative template: the 9B **serving** `chat_template.jinja` (native qwen3_xml).
- double-templating: **NONE** (zero_double=True).
- native-format finding: trainer preset `fast_dllm_v2_native` diverges from serve on **7023/10038** assistant turns (extra tool_call/tool_response whitespace); **2/323** rows identical. **Train from the serve-exact `train_swe_sft.tokenized.jsonl`**, not the preset.

## Split
- single `train` split (no keeper-level dev per design).

## Teacher mix

- stock-9b-ar: 293 (90.7%)
- qwen3.6-27b-nvfp4-mtp: 17 (5.3%)
- opus-4.8: 13 (4.0%)

## Source mix

- SWE-bench_Verified: 192 (59.4%)
- SWE-Gym: 131 (40.6%)

## Family distribution (20 repos)

- django/django: 98
- sympy/sympy: 33
- conan-io/conan: 26
- python/mypy: 22
- iterative/dvc: 18
- scikit-learn/scikit-learn: 18
- pydantic/pydantic: 16
- getmoto/moto: 16
- pandas-dev/pandas: 15
- dask/dask: 13
- matplotlib/matplotlib: 12
- pydata/xarray: 10
- pytest-dev/pytest: 10
- psf/requests: 5
- astropy/astropy: 3
- modin-project/modin: 3
- pylint-dev/pylint: 2
- bokeh/bokeh: 1
- facebookresearch/hydra: 1
- pallets/flask: 1

## Tool-schema fidelity

- with tools block: 161/323; without: 162 (loss-masked context skew only).

## Length audit (serving template, block_size=32768, truncation=left)

- token length: {'min': 9815, 'p10': 18958, 'p25': 23340, 'p50': 24047, 'p75': 26956, 'p90': 29494, 'p95': 29945, 'p99': 30965, 'max': 40950, 'mean': 24413.9}
- assistant-target tokens: {'min': 1, 'p10': 1659, 'p25': 2559, 'p50': 4080, 'p75': 5175, 'p90': 6008, 'p95': 6544, 'p99': 7876, 'max': 8860, 'mean': 3975.9}
- total tokens: 7,885,679; total assistant-target tokens: 1,284,222
- over block_size (32768): **2** (0.62%); partial-after-trunc 0; zero-after-trunc 0
- labels lost when over-block: {'min': 0, 'p10': 0, 'p25': 0, 'p50': 0, 'p75': 0, 'p90': 0, 'p95': 0, 'p99': 0, 'max': 0, 'mean': 0.0}

## Thin-trajectory outliers (< 64 assistant-target tokens)

- count: **1** (flagged, NOT dropped -- owner curates)
  - pandas-dev__pandas-47714 (teacher qwen3.6-27b-nvfp4-mtp, 1 target toks, 1 assistant msgs)
