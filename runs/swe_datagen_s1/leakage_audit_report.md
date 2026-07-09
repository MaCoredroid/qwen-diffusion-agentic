# Cross-instance answer-leakage audit — swe_datagen_s1
_generated 2026-07-09T18:02:45Z · CPU-only · 2.06s · text-sim backend `tfidf_cosine_sklearn`_
## Holdout / snapshot integrity
- eval holdout: **113 distinct ids** (inner5 ∪ tier0_20 ∪ tier1_100)
- holdout sha256 `c56f473ad31e52bee0f85151562f4e2122e4815dfa3f1b776b15fe121e8d168e` — pin-assert **PASS**
- keepers scanned: **323** (192 Verified, 131 SWE-Gym); snapshot sha256 `434f598ce35f4015…`
- SWE-Gym disjoint-repo assert: **PASS** (collisions: none)

## Topline
- same-repo keeper×holdout pairs: **5811**; flagged: **367**
- keepers with a same-repo holdout neighbor: **191**
- keepers with ANY file overlap vs holdout gold+test: **64**
- keepers overlapping a holdout **GOLD** file: **64**
- keepers with function-level overlap: **11**
- keepers with issue-text sim ≥ p95 (0.119325): **116**

## Top 40 flagged pairs (by severity)
| # | keeper | holdout | repo | gold-file∩ | func∩ | text-sim | sev |
|--:|---|---|---|---|--:|--:|--:|
| 1 | `django__django-16485` | `django__django-15863` | django/django | django/template/defaultfilters.py | django/template/defaultfilters.py:floatformat | 0.443003 | 8.886006 |
| 2 | `sympy__sympy-24213` | `sympy__sympy-24066` | sympy/sympy | sympy/physics/units/unitsystem.py | sympy/physics/units/unitsystem.py:_collect_factor_and_dimension | 0.423054 | 8.846108 |
| 3 | `django__django-13128` | `django__django-13121` | django/django | django/db/models/expressions.py | django/db/models/expressions.py:set_source_expressions | 0.249049 | 8.498098 |
| 4 | `django__django-16139` | `django__django-13741` | django/django | django/contrib/auth/forms.py | django/contrib/auth/forms.py:__init__ | 0.215598 | 8.431196 |
| 5 | `django__django-16145` | `django__django-13809` | django/django | django/core/management/commands/runserver.py | django/core/management/commands/runserver.py:inner_run | 0.20659 | 8.41318 |
| 6 | `sympy__sympy-12481` | `sympy__sympy-12489` | sympy/sympy | sympy/combinatorics/permutations.py | sympy/combinatorics/permutations.py:__new__ | 0.167337 | 8.334674 |
| 7 | `django__django-13012` | `django__django-13121` | django/django | django/db/models/expressions.py | django/db/models/expressions.py:set_source_expressions | 0.115127 | 8.230254 |
| 8 | `django__django-14017` | `django__django-13121` | django/django | django/db/models/expressions.py | django/db/models/expressions.py:Combinable | 0.095971 | 8.191942 |
| 9 | `django__django-11790` | `django__django-13741` | django/django | django/contrib/auth/forms.py | django/contrib/auth/forms.py:__init__ | 0.086427 | 8.172854 |
| 10 | `django__django-11999` | `django__django-15561` | django/django | django/db/models/fields/__init__.py | django/db/models/fields/__init__.py:Field | 0.057533 | 8.115066 |
| 11 | `django__django-15315` | `django__django-15561` | django/django | django/db/models/fields/__init__.py | django/db/models/fields/__init__.py:Field | 0.014055 | 8.02811 |
| 12 | `django__django-15380` | `django__django-15104` | django/django | django/db/migrations/autodetector.py | — | 0.463664 | 5.927328 |
| 13 | `pydata__xarray-6599` | `pydata__xarray-4687` | pydata/xarray | xarray/core/computation.py | — | 0.406708 | 5.813416 |
| 14 | `sympy__sympy-22714` | `sympy__sympy-17655` | sympy/sympy | sympy/geometry/point.py | — | 0.322989 | 5.645978 |
| 15 | `matplotlib__matplotlib-26113` | `matplotlib__matplotlib-13989` | matplotlib/matplotlib | lib/matplotlib/axes/_axes.py | — | 0.315101 | 5.630202 |
| 16 | `matplotlib__matplotlib-25311` | `matplotlib__matplotlib-20859` | matplotlib/matplotlib | lib/matplotlib/legend.py | — | 0.276795 | 5.55359 |
| 17 | `django__django-13670` | `django__django-14373` | django/django | django/utils/dateformat.py | — | 0.259641 | 5.519282 |
| 18 | `pytest-dev__pytest-10081` | `pytest-dev__pytest-8399` | pytest-dev/pytest | src/_pytest/unittest.py | — | 0.220259 | 5.440518 |
| 19 | `django__django-15814` | `django__django-13028` | django/django | django/db/models/sql/query.py | — | 0.209796 | 5.419592 |
| 20 | `django__django-11999` | `django__django-11211` | django/django | django/db/models/fields/__init__.py | — | 0.200192 | 5.400384 |
| 21 | `django__django-12209` | `django__django-12273` | django/django | django/db/models/base.py | — | 0.19641 | 5.39282 |
| 22 | `django__django-13449` | `django__django-13121` | django/django | django/db/models/expressions.py | — | 0.189288 | 5.378576 |
| 23 | `django__django-14017` | `django__django-14140` | django/django | django/db/models/query_utils.py | — | 0.183877 | 5.367754 |
| 24 | `django__django-16333` | `django__django-13741` | django/django | django/contrib/auth/forms.py | — | 0.170181 | 5.340362 |
| 25 | `django__django-13925` | `django__django-12273` | django/django | django/db/models/base.py | — | 0.160428 | 5.320856 |
| 26 | `django__django-11299` | `django__django-13028` | django/django | django/db/models/sql/query.py | — | 0.154648 | 5.309296 |
| 27 | `pytest-dev__pytest-7236` | `pytest-dev__pytest-8399` | pytest-dev/pytest | src/_pytest/unittest.py | — | 0.153242 | 5.306484 |
| 28 | `matplotlib__matplotlib-24570` | `matplotlib__matplotlib-24637` | matplotlib/matplotlib | lib/matplotlib/offsetbox.py | — | 0.146346 | 5.292692 |
| 29 | `django__django-11292` | `django__django-16454` | django/django | django/core/management/base.py | — | 0.144483 | 5.288966 |
| 30 | `scikit-learn__scikit-learn-11578` | `scikit-learn__scikit-learn-14087` | scikit-learn/scikit-learn | sklearn/linear_model/logistic.py | — | 0.141957 | 5.283914 |
| 31 | `psf__requests-1142` | `psf__requests-5414` | psf/requests | requests/models.py | — | 0.139438 | 5.278876 |
| 32 | `django__django-15380` | `django__django-12754` | django/django | django/db/migrations/autodetector.py | — | 0.13154 | 5.26308 |
| 33 | `django__django-16569` | `django__django-14608` | django/django | django/forms/formsets.py | — | 0.12845 | 5.2569 |
| 34 | `psf__requests-1724` | `psf__requests-5414` | psf/requests | requests/models.py | — | 0.126231 | 5.252462 |
| 35 | `django__django-13658` | `django__django-16454` | django/django | django/core/management/base.py | — | 0.125546 | 5.251092 |
| 36 | `psf__requests-1921` | `psf__requests-5414` | psf/requests | requests/models.py | — | 0.125185 | 5.25037 |
| 37 | `django__django-11999` | `django__django-11734` | django/django | django/db/models/fields/__init__.py | — | 0.119595 | 5.23919 |
| 38 | `django__django-12276` | `django__django-12193` | django/django | django/forms/widgets.py | — | 0.11941 | 5.23882 |
| 39 | `pytest-dev__pytest-6202` | `pytest-dev__pytest-8399` | pytest-dev/pytest | src/_pytest/python.py | — | 0.114629 | 5.229258 |
| 40 | `django__django-15315` | `django__django-11211` | django/django | django/db/models/fields/__init__.py | — | 0.113268 | 5.226536 |

## Frontier pre-screen (not-yet-collected Verified-adjacent ids)
- remaining same-repo ids scanned: **0**
- ids with a gold-file overlap vs a holdout task: **0**

| # | frontier id | holdout | repo | gold-file∩ | func∩ | sev |
|--:|---|---|---|---|--:|--:|
