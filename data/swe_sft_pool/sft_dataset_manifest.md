# SWE-SFT dataset manifest

Built 2026-07-09T19:32:01Z by `runs/swe_datagen_s1/build_swe_sft_dataset.py` from `runs/swe_datagen_s1/keepers/keepers.jsonl` (334 keepers).

## Firewall (KILL-D1)

- eval holdout: **113 ids**, sha256 `c56f473ad31e52bee0f85151562f4e2122e4815dfa3f1b776b15fe121e8d168e` == pin **OK**
- keeper x holdout overlap: **0** / 334
- quarantined excluded: **['pydata__xarray-6461']** (0 in train)

## Render / native format

- authoritative template: the 9B **serving** `chat_template.jinja` (native qwen3_xml).
- double-templating: **NONE** (zero_double=True).
- native-format finding: trainer preset `fast_dllm_v2_native` diverges from serve on **7127/10153** assistant turns (extra tool_call/tool_response whitespace); **2/334** rows identical. **Train from the serve-exact `train_swe_sft.tokenized.jsonl`**, not the preset.

## Split
- single `train` split (no keeper-level dev per design).

## Teacher mix

- stock-9b-ar: 293 (87.7%)
- qwen3.6-27b-nvfp4-mtp: 28 (8.4%)
- opus-4.8: 13 (3.9%)

## Source mix

- SWE-bench_Verified: 192 (57.5%)
- SWE-Gym: 142 (42.5%)

## Family distribution (20 repos)

- django/django: 98
- sympy/sympy: 33
- conan-io/conan: 26
- python/mypy: 22
- pandas-dev/pandas: 20
- iterative/dvc: 18
- getmoto/moto: 18
- scikit-learn/scikit-learn: 18
- pydantic/pydantic: 16
- dask/dask: 14
- matplotlib/matplotlib: 12
- pydata/xarray: 10
- pytest-dev/pytest: 10
- modin-project/modin: 6
- psf/requests: 5
- astropy/astropy: 3
- pylint-dev/pylint: 2
- bokeh/bokeh: 1
- facebookresearch/hydra: 1
- pallets/flask: 1

## Tool-schema fidelity

- with tools block: 166/334; without: 168 (loss-masked context skew only).

## Length audit (serving template, block_size=32768, truncation=left)

- token length: {'min': 9815, 'p10': 18742, 'p25': 23336, 'p50': 24042, 'p75': 26855, 'p90': 29474, 'p95': 29871, 'p99': 30965, 'max': 40950, 'mean': 24289.4}
- assistant-target tokens: {'min': 1, 'p10': 1635, 'p25': 2388, 'p50': 4002, 'p75': 5167, 'p90': 6008, 'p95': 6509, 'p99': 7876, 'max': 8860, 'mean': 3905.3}
- total tokens: 8,112,663; total assistant-target tokens: 1,304,354
- over block_size (32768): **2** (0.6%); partial-after-trunc 0; zero-after-trunc 0
- labels lost when over-block: {'min': 0, 'p10': 0, 'p25': 0, 'p50': 0, 'p75': 0, 'p90': 0, 'p95': 0, 'p99': 0, 'max': 0, 'mean': 0.0}

## Thin-trajectory outliers (< 64 assistant-target tokens)

- count: **1** (flagged, NOT dropped -- owner curates)
  - pandas-dev__pandas-47714 (teacher qwen3.6-27b-nvfp4-mtp, 1 target toks, 1 assistant msgs)
