# Cross-instance answer-leakage audit — swe_datagen_s1
_generated 2026-07-09T17:15:18Z · CPU-only · 2.08s · text-sim backend `tfidf_cosine_sklearn`_
## Holdout / snapshot integrity
- eval holdout: **113 distinct ids** (inner5 ∪ tier0_20 ∪ tier1_100)
- holdout sha256 `c56f473ad31e52bee0f85151562f4e2122e4815dfa3f1b776b15fe121e8d168e` — pin-assert **PASS**
- keepers scanned: **311** (193 Verified, 118 SWE-Gym); snapshot sha256 `2aeecce15b7a9b85…`
- SWE-Gym disjoint-repo assert: **PASS** (collisions: none)

## Topline
- same-repo keeper×holdout pairs: **5816**; flagged: **368**
- keepers with a same-repo holdout neighbor: **192**
- keepers with ANY file overlap vs holdout gold+test: **65**
- keepers overlapping a holdout **GOLD** file: **65**
- keepers with function-level overlap: **12**
- keepers with issue-text sim ≥ p95 (0.119632): **116**

## Top 40 flagged pairs (by severity)
| # | keeper | holdout | repo | gold-file∩ | func∩ | text-sim | sev |
|--:|---|---|---|---|--:|--:|--:|
| 1 | `django__django-16485` | `django__django-15863` | django/django | django/template/defaultfilters.py | django/template/defaultfilters.py:floatformat | 0.443308 | 8.886616 |
| 2 | `sympy__sympy-24213` | `sympy__sympy-24066` | sympy/sympy | sympy/physics/units/unitsystem.py | sympy/physics/units/unitsystem.py:_collect_factor_and_dimension | 0.422357 | 8.844714 |
| 3 | `django__django-13128` | `django__django-13121` | django/django | django/db/models/expressions.py | django/db/models/expressions.py:set_source_expressions | 0.249132 | 8.498264 |
| 4 | `pydata__xarray-6461` | `pydata__xarray-4687` | pydata/xarray | xarray/core/computation.py | xarray/core/computation.py:where | 0.227859 | 8.455718 |
| 5 | `django__django-16139` | `django__django-13741` | django/django | django/contrib/auth/forms.py | django/contrib/auth/forms.py:__init__ | 0.215677 | 8.431354 |
| 6 | `django__django-16145` | `django__django-13809` | django/django | django/core/management/commands/runserver.py | django/core/management/commands/runserver.py:inner_run | 0.205008 | 8.410016 |
| 7 | `sympy__sympy-12481` | `sympy__sympy-12489` | sympy/sympy | sympy/combinatorics/permutations.py | sympy/combinatorics/permutations.py:__new__ | 0.167403 | 8.334806 |
| 8 | `django__django-13012` | `django__django-13121` | django/django | django/db/models/expressions.py | django/db/models/expressions.py:set_source_expressions | 0.115225 | 8.23045 |
| 9 | `django__django-14017` | `django__django-13121` | django/django | django/db/models/expressions.py | django/db/models/expressions.py:Combinable | 0.096094 | 8.192188 |
| 10 | `django__django-11790` | `django__django-13741` | django/django | django/contrib/auth/forms.py | django/contrib/auth/forms.py:__init__ | 0.086526 | 8.173052 |
| 11 | `django__django-11999` | `django__django-15561` | django/django | django/db/models/fields/__init__.py | django/db/models/fields/__init__.py:Field | 0.057553 | 8.115106 |
| 12 | `django__django-15315` | `django__django-15561` | django/django | django/db/models/fields/__init__.py | django/db/models/fields/__init__.py:Field | 0.014069 | 8.028138 |
| 13 | `django__django-15380` | `django__django-15104` | django/django | django/db/migrations/autodetector.py | — | 0.46368 | 5.92736 |
| 14 | `pydata__xarray-6599` | `pydata__xarray-4687` | pydata/xarray | xarray/core/computation.py | — | 0.40567 | 5.81134 |
| 15 | `sympy__sympy-22714` | `sympy__sympy-17655` | sympy/sympy | sympy/geometry/point.py | — | 0.323073 | 5.646146 |
| 16 | `matplotlib__matplotlib-26113` | `matplotlib__matplotlib-13989` | matplotlib/matplotlib | lib/matplotlib/axes/_axes.py | — | 0.315241 | 5.630482 |
| 17 | `matplotlib__matplotlib-25311` | `matplotlib__matplotlib-20859` | matplotlib/matplotlib | lib/matplotlib/legend.py | — | 0.277101 | 5.554202 |
| 18 | `django__django-13670` | `django__django-14373` | django/django | django/utils/dateformat.py | — | 0.259894 | 5.519788 |
| 19 | `pytest-dev__pytest-10081` | `pytest-dev__pytest-8399` | pytest-dev/pytest | src/_pytest/unittest.py | — | 0.220017 | 5.440034 |
| 20 | `django__django-15814` | `django__django-13028` | django/django | django/db/models/sql/query.py | — | 0.210005 | 5.42001 |
| 21 | `django__django-11999` | `django__django-11211` | django/django | django/db/models/fields/__init__.py | — | 0.200301 | 5.400602 |
| 22 | `django__django-12209` | `django__django-12273` | django/django | django/db/models/base.py | — | 0.196359 | 5.392718 |
| 23 | `django__django-13449` | `django__django-13121` | django/django | django/db/models/expressions.py | — | 0.1893 | 5.3786 |
| 24 | `django__django-14017` | `django__django-14140` | django/django | django/db/models/query_utils.py | — | 0.183999 | 5.367998 |
| 25 | `django__django-16333` | `django__django-13741` | django/django | django/contrib/auth/forms.py | — | 0.1703 | 5.3406 |
| 26 | `django__django-13925` | `django__django-12273` | django/django | django/db/models/base.py | — | 0.160406 | 5.320812 |
| 27 | `django__django-11299` | `django__django-13028` | django/django | django/db/models/sql/query.py | — | 0.15456 | 5.30912 |
| 28 | `pytest-dev__pytest-7236` | `pytest-dev__pytest-8399` | pytest-dev/pytest | src/_pytest/unittest.py | — | 0.153004 | 5.306008 |
| 29 | `matplotlib__matplotlib-24570` | `matplotlib__matplotlib-24637` | matplotlib/matplotlib | lib/matplotlib/offsetbox.py | — | 0.146506 | 5.293012 |
| 30 | `django__django-11292` | `django__django-16454` | django/django | django/core/management/base.py | — | 0.14467 | 5.28934 |
| 31 | `scikit-learn__scikit-learn-11578` | `scikit-learn__scikit-learn-14087` | scikit-learn/scikit-learn | sklearn/linear_model/logistic.py | — | 0.141922 | 5.283844 |
| 32 | `psf__requests-1142` | `psf__requests-5414` | psf/requests | requests/models.py | — | 0.139413 | 5.278826 |
| 33 | `django__django-15380` | `django__django-12754` | django/django | django/db/migrations/autodetector.py | — | 0.131672 | 5.263344 |
| 34 | `django__django-16569` | `django__django-14608` | django/django | django/forms/formsets.py | — | 0.128641 | 5.257282 |
| 35 | `psf__requests-1724` | `psf__requests-5414` | psf/requests | requests/models.py | — | 0.126225 | 5.25245 |
| 36 | `django__django-13658` | `django__django-16454` | django/django | django/core/management/base.py | — | 0.12559 | 5.25118 |
| 37 | `psf__requests-1921` | `psf__requests-5414` | psf/requests | requests/models.py | — | 0.125234 | 5.250468 |
| 38 | `django__django-11999` | `django__django-11734` | django/django | django/db/models/fields/__init__.py | — | 0.119696 | 5.239392 |
| 39 | `django__django-12276` | `django__django-12193` | django/django | django/forms/widgets.py | — | 0.119611 | 5.239222 |
| 40 | `pytest-dev__pytest-6202` | `pytest-dev__pytest-8399` | pytest-dev/pytest | src/_pytest/python.py | — | 0.114607 | 5.229214 |

## Frontier pre-screen (not-yet-collected Verified-adjacent ids)
- remaining same-repo ids scanned: **0**
- ids with a gold-file overlap vs a holdout task: **0**

| # | frontier id | holdout | repo | gold-file∩ | func∩ | sev |
|--:|---|---|---|---|--:|--:|
