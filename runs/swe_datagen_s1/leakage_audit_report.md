# Cross-instance answer-leakage audit — swe_datagen_s1
_generated 2026-07-13T15:20:20Z · CPU-only · 2.01s · text-sim backend `tfidf_cosine_sklearn`_
## Holdout / snapshot integrity
- eval holdout: **113 distinct ids** (inner5 ∪ tier0_20 ∪ tier1_100)
- holdout sha256 `c56f473ad31e52bee0f85151562f4e2122e4815dfa3f1b776b15fe121e8d168e` — pin-assert **PASS**
- keepers scanned: **383** (199 Verified, 184 SWE-Gym); snapshot sha256 `b5a628a160034cc0…`
- SWE-Gym disjoint-repo assert: **PASS** (collisions: none)

## Topline
- same-repo keeper×holdout pairs: **5850**; flagged: **371**
- keepers with a same-repo holdout neighbor: **198**
- keepers with ANY file overlap vs holdout gold+test: **65**
- keepers overlapping a holdout **GOLD** file: **65**
- keepers with function-level overlap: **11**
- keepers with issue-text sim ≥ p95 (0.11982): **116**

## Top 40 flagged pairs (by severity)
| # | keeper | holdout | repo | gold-file∩ | func∩ | text-sim | sev |
|--:|---|---|---|---|--:|--:|--:|
| 1 | `django__django-16485` | `django__django-15863` | django/django | django/template/defaultfilters.py | django/template/defaultfilters.py:floatformat | 0.443989 | 8.887978 |
| 2 | `sympy__sympy-24213` | `sympy__sympy-24066` | sympy/sympy | sympy/physics/units/unitsystem.py | sympy/physics/units/unitsystem.py:_collect_factor_and_dimension | 0.4234 | 8.8468 |
| 3 | `django__django-13128` | `django__django-13121` | django/django | django/db/models/expressions.py | django/db/models/expressions.py:set_source_expressions | 0.249715 | 8.49943 |
| 4 | `django__django-16139` | `django__django-13741` | django/django | django/contrib/auth/forms.py | django/contrib/auth/forms.py:__init__ | 0.21478 | 8.42956 |
| 5 | `django__django-16145` | `django__django-13809` | django/django | django/core/management/commands/runserver.py | django/core/management/commands/runserver.py:inner_run | 0.207753 | 8.415506 |
| 6 | `sympy__sympy-12481` | `sympy__sympy-12489` | sympy/sympy | sympy/combinatorics/permutations.py | sympy/combinatorics/permutations.py:__new__ | 0.167707 | 8.335414 |
| 7 | `django__django-13012` | `django__django-13121` | django/django | django/db/models/expressions.py | django/db/models/expressions.py:set_source_expressions | 0.115369 | 8.230738 |
| 8 | `django__django-14017` | `django__django-13121` | django/django | django/db/models/expressions.py | django/db/models/expressions.py:Combinable | 0.096664 | 8.193328 |
| 9 | `django__django-11790` | `django__django-13741` | django/django | django/contrib/auth/forms.py | django/contrib/auth/forms.py:__init__ | 0.086249 | 8.172498 |
| 10 | `django__django-11999` | `django__django-15561` | django/django | django/db/models/fields/__init__.py | django/db/models/fields/__init__.py:Field | 0.057629 | 8.115258 |
| 11 | `django__django-15315` | `django__django-15561` | django/django | django/db/models/fields/__init__.py | django/db/models/fields/__init__.py:Field | 0.013936 | 8.027872 |
| 12 | `django__django-15380` | `django__django-15104` | django/django | django/db/migrations/autodetector.py | — | 0.464052 | 5.928104 |
| 13 | `pydata__xarray-6599` | `pydata__xarray-4687` | pydata/xarray | xarray/core/computation.py | — | 0.406329 | 5.812658 |
| 14 | `sympy__sympy-22714` | `sympy__sympy-17655` | sympy/sympy | sympy/geometry/point.py | — | 0.324469 | 5.648938 |
| 15 | `matplotlib__matplotlib-26113` | `matplotlib__matplotlib-13989` | matplotlib/matplotlib | lib/matplotlib/axes/_axes.py | — | 0.315348 | 5.630696 |
| 16 | `matplotlib__matplotlib-25311` | `matplotlib__matplotlib-20859` | matplotlib/matplotlib | lib/matplotlib/legend.py | — | 0.277153 | 5.554306 |
| 17 | `django__django-13670` | `django__django-14373` | django/django | django/utils/dateformat.py | — | 0.260483 | 5.520966 |
| 18 | `pytest-dev__pytest-10081` | `pytest-dev__pytest-8399` | pytest-dev/pytest | src/_pytest/unittest.py | — | 0.219202 | 5.438404 |
| 19 | `django__django-15814` | `django__django-13028` | django/django | django/db/models/sql/query.py | — | 0.211026 | 5.422052 |
| 20 | `django__django-11999` | `django__django-11211` | django/django | django/db/models/fields/__init__.py | — | 0.201147 | 5.402294 |
| 21 | `django__django-12209` | `django__django-12273` | django/django | django/db/models/base.py | — | 0.19667 | 5.39334 |
| 22 | `django__django-13449` | `django__django-13121` | django/django | django/db/models/expressions.py | — | 0.189776 | 5.379552 |
| 23 | `django__django-14017` | `django__django-14140` | django/django | django/db/models/query_utils.py | — | 0.184698 | 5.369396 |
| 24 | `django__django-16333` | `django__django-13741` | django/django | django/contrib/auth/forms.py | — | 0.169623 | 5.339246 |
| 25 | `django__django-13925` | `django__django-12273` | django/django | django/db/models/base.py | — | 0.160821 | 5.321642 |
| 26 | `django__django-11299` | `django__django-13028` | django/django | django/db/models/sql/query.py | — | 0.154892 | 5.309784 |
| 27 | `pytest-dev__pytest-7236` | `pytest-dev__pytest-8399` | pytest-dev/pytest | src/_pytest/unittest.py | — | 0.152798 | 5.305596 |
| 28 | `django__django-11292` | `django__django-16454` | django/django | django/core/management/base.py | — | 0.144626 | 5.289252 |
| 29 | `matplotlib__matplotlib-24570` | `matplotlib__matplotlib-24637` | matplotlib/matplotlib | lib/matplotlib/offsetbox.py | — | 0.144045 | 5.28809 |
| 30 | `scikit-learn__scikit-learn-11578` | `scikit-learn__scikit-learn-14087` | scikit-learn/scikit-learn | sklearn/linear_model/logistic.py | — | 0.141788 | 5.283576 |
| 31 | `psf__requests-1142` | `psf__requests-5414` | psf/requests | requests/models.py | — | 0.139218 | 5.278436 |
| 32 | `django__django-15380` | `django__django-12754` | django/django | django/db/migrations/autodetector.py | — | 0.132052 | 5.264104 |
| 33 | `django__django-16569` | `django__django-14608` | django/django | django/forms/formsets.py | — | 0.129064 | 5.258128 |
| 34 | `psf__requests-1724` | `psf__requests-5414` | psf/requests | requests/models.py | — | 0.126403 | 5.252806 |
| 35 | `psf__requests-1921` | `psf__requests-5414` | psf/requests | requests/models.py | — | 0.125312 | 5.250624 |
| 36 | `django__django-13658` | `django__django-16454` | django/django | django/core/management/base.py | — | 0.125039 | 5.250078 |
| 37 | `django__django-11999` | `django__django-11734` | django/django | django/db/models/fields/__init__.py | — | 0.119846 | 5.239692 |
| 38 | `django__django-12276` | `django__django-12193` | django/django | django/forms/widgets.py | — | 0.119441 | 5.238882 |
| 39 | `django__django-15315` | `django__django-11211` | django/django | django/db/models/fields/__init__.py | — | 0.114219 | 5.228438 |
| 40 | `pytest-dev__pytest-6202` | `pytest-dev__pytest-8399` | pytest-dev/pytest | src/_pytest/python.py | — | 0.113814 | 5.227628 |

## Frontier pre-screen (not-yet-collected Verified-adjacent ids)
- remaining same-repo ids scanned: **0**
- ids with a gold-file overlap vs a holdout task: **0**

| # | frontier id | holdout | repo | gold-file∩ | func∩ | sev |
|--:|---|---|---|---|--:|--:|
