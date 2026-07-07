# Cross-instance answer-leakage audit — swe_datagen_s1
_generated 2026-07-07T06:15:08Z · CPU-only · 0.79s · text-sim backend `tfidf_cosine_sklearn`_
## Holdout / snapshot integrity
- eval holdout: **113 distinct ids** (inner5 ∪ tier0_20 ∪ tier1_100)
- holdout sha256 `c56f473ad31e52bee0f85151562f4e2122e4815dfa3f1b776b15fe121e8d168e` — pin-assert **PASS**
- keepers scanned: **114** (54 Verified, 60 SWE-Gym); snapshot sha256 `61e1f89092b34e6f…`
- SWE-Gym disjoint-repo assert: **PASS** (collisions: none)

## Topline
- same-repo keeper×holdout pairs: **720**; flagged: **49**
- keepers with a same-repo holdout neighbor: **53**
- keepers with ANY file overlap vs holdout gold+test: **15**
- keepers overlapping a holdout **GOLD** file: **15**
- keepers with function-level overlap: **1**
- keepers with issue-text sim ≥ p95 (0.16608): **18**

## Top 40 flagged pairs (by severity)
| # | keeper | holdout | repo | gold-file∩ | func∩ | text-sim | sev |
|--:|---|---|---|---|--:|--:|--:|
| 1 | `sympy__sympy-12481` | `sympy__sympy-12489` | sympy/sympy | sympy/combinatorics/permutations.py | sympy/combinatorics/permutations.py:__new__ | 0.166197 | 8.332394 |
| 2 | `pydata__xarray-6599` | `pydata__xarray-4687` | pydata/xarray | xarray/core/computation.py | — | 0.391935 | 5.78387 |
| 3 | `pytest-dev__pytest-10081` | `pytest-dev__pytest-8399` | pytest-dev/pytest | src/_pytest/unittest.py | — | 0.220793 | 5.441586 |
| 4 | `pytest-dev__pytest-7236` | `pytest-dev__pytest-8399` | pytest-dev/pytest | src/_pytest/unittest.py | — | 0.152346 | 5.304692 |
| 5 | `scikit-learn__scikit-learn-11578` | `scikit-learn__scikit-learn-14087` | scikit-learn/scikit-learn | sklearn/linear_model/logistic.py | — | 0.141905 | 5.28381 |
| 6 | `psf__requests-1142` | `psf__requests-5414` | psf/requests | requests/models.py | — | 0.139025 | 5.27805 |
| 7 | `psf__requests-1724` | `psf__requests-5414` | psf/requests | requests/models.py | — | 0.123444 | 5.246888 |
| 8 | `psf__requests-1921` | `psf__requests-5414` | psf/requests | requests/models.py | — | 0.118348 | 5.236696 |
| 9 | `pytest-dev__pytest-6202` | `pytest-dev__pytest-8399` | pytest-dev/pytest | src/_pytest/python.py | — | 0.112026 | 5.224052 |
| 10 | `matplotlib__matplotlib-24149` | `matplotlib__matplotlib-13989` | matplotlib/matplotlib | lib/matplotlib/axes/_axes.py | — | 0.104322 | 5.208644 |
| 11 | `sympy__sympy-12419` | `sympy__sympy-17630` | sympy/sympy | sympy/matrices/expressions/matexpr.py | — | 0.081409 | 5.162818 |
| 12 | `sympy__sympy-11618` | `sympy__sympy-17655` | sympy/sympy | sympy/geometry/point.py | — | 0.079368 | 5.158736 |
| 13 | `django__django-11095` | `django__django-12713` | django/django | django/contrib/admin/options.py | — | 0.077809 | 5.155618 |
| 14 | `django__django-11133` | `django__django-13195` | django/django | django/http/response.py | — | 0.073951 | 5.147902 |
| 15 | `pytest-dev__pytest-7982` | `pytest-dev__pytest-5840` | pytest-dev/pytest | src/_pytest/pathlib.py | — | 0.060428 | 5.120856 |
| 16 | `django__django-11095` | `django__django-16100` | django/django | django/contrib/admin/options.py | — | 0.032265 | 5.06453 |
| 17 | `pydata__xarray-3305` | `pydata__xarray-4687` | pydata/xarray | — | — | 0.648004 | 1.296008 |
| 18 | `pydata__xarray-4629` | `pydata__xarray-4687` | pydata/xarray | — | — | 0.581169 | 1.162338 |
| 19 | `pydata__xarray-3305` | `pydata__xarray-4075` | pydata/xarray | — | — | 0.542414 | 1.084828 |
| 20 | `pydata__xarray-3305` | `pydata__xarray-4094` | pydata/xarray | — | — | 0.462946 | 0.925892 |
| 21 | `pydata__xarray-3305` | `pydata__xarray-3151` | pydata/xarray | — | — | 0.433273 | 0.866546 |
| 22 | `matplotlib__matplotlib-23314` | `matplotlib__matplotlib-25332` | matplotlib/matplotlib | — | — | 0.376141 | 0.752282 |
| 23 | `pydata__xarray-4629` | `pydata__xarray-4075` | pydata/xarray | — | — | 0.37118 | 0.74236 |
| 24 | `pydata__xarray-4629` | `pydata__xarray-4094` | pydata/xarray | — | — | 0.339719 | 0.679438 |
| 25 | `pydata__xarray-6599` | `pydata__xarray-4075` | pydata/xarray | — | — | 0.329526 | 0.659052 |
| 26 | `pydata__xarray-4629` | `pydata__xarray-3151` | pydata/xarray | — | — | 0.307586 | 0.615172 |
| 27 | `pydata__xarray-6599` | `pydata__xarray-4094` | pydata/xarray | — | — | 0.282481 | 0.564962 |
| 28 | `scikit-learn__scikit-learn-10844` | `scikit-learn__scikit-learn-14053` | scikit-learn/scikit-learn | — | — | 0.270432 | 0.540864 |
| 29 | `pydata__xarray-6599` | `pydata__xarray-3151` | pydata/xarray | — | — | 0.261848 | 0.523696 |
| 30 | `matplotlib__matplotlib-23314` | `matplotlib__matplotlib-25122` | matplotlib/matplotlib | — | — | 0.253113 | 0.506226 |
| 31 | `pytest-dev__pytest-7205` | `pytest-dev__pytest-5631` | pytest-dev/pytest | — | — | 0.241022 | 0.482044 |
| 32 | `scikit-learn__scikit-learn-13439` | `scikit-learn__scikit-learn-14087` | scikit-learn/scikit-learn | — | — | 0.233766 | 0.467532 |
| 33 | `pytest-dev__pytest-7205` | `pytest-dev__pytest-8399` | pytest-dev/pytest | — | — | 0.230793 | 0.461586 |
| 34 | `matplotlib__matplotlib-24149` | `matplotlib__matplotlib-26291` | matplotlib/matplotlib | — | — | 0.229249 | 0.458498 |
| 35 | `pylint-dev__pylint-6903` | `pylint-dev__pylint-8898` | pylint-dev/pylint | — | — | 0.224925 | 0.44985 |
| 36 | `scikit-learn__scikit-learn-13328` | `scikit-learn__scikit-learn-14087` | scikit-learn/scikit-learn | — | — | 0.215965 | 0.43193 |
| 37 | `pydata__xarray-6599` | `pydata__xarray-6721` | pydata/xarray | — | — | 0.215777 | 0.431554 |
| 38 | `matplotlib__matplotlib-23412` | `matplotlib__matplotlib-25332` | matplotlib/matplotlib | — | — | 0.20417 | 0.40834 |
| 39 | `scikit-learn__scikit-learn-10297` | `scikit-learn__scikit-learn-14087` | scikit-learn/scikit-learn | — | — | 0.196239 | 0.392478 |
| 40 | `matplotlib__matplotlib-24149` | `matplotlib__matplotlib-24970` | matplotlib/matplotlib | — | — | 0.19593 | 0.39186 |

## Frontier pre-screen (not-yet-collected Verified-adjacent ids)
- remaining same-repo ids scanned: **333**
- ids with a gold-file overlap vs a holdout task: **114**

| # | frontier id | holdout | repo | gold-file∩ | func∩ | sev |
|--:|---|---|---|---|--:|--:|
| 1 | `django__django-11138` | `django__django-13121` | django/django | django/db/backends/mysql/operations.py, django/db/backends/sqlite3/operations.py | — | 10.0 |
| 2 | `django__django-16032` | `django__django-11734` | django/django | django/db/models/fields/related_lookups.py, django/db/models/sql/query.py | — | 10.0 |
| 3 | `django__django-11265` | `django__django-11734` | django/django | django/db/models/sql/query.py | django/db/models/sql/query.py:split_exclude | 8.0 |
| 4 | `django__django-11265` | `django__django-13158` | django/django | django/db/models/sql/query.py | django/db/models/sql/query.py:split_exclude | 8.0 |
| 5 | `django__django-11790` | `django__django-13741` | django/django | django/contrib/auth/forms.py | django/contrib/auth/forms.py:__init__ | 8.0 |
| 6 | `django__django-13012` | `django__django-13121` | django/django | django/db/models/expressions.py | django/db/models/expressions.py:set_source_expressions | 8.0 |
| 7 | `django__django-13128` | `django__django-13121` | django/django | django/db/models/expressions.py | django/db/models/expressions.py:set_source_expressions | 8.0 |
| 8 | `django__django-13343` | `django__django-16493` | django/django | django/db/models/fields/files.py | django/db/models/fields/files.py:deconstruct | 8.0 |
| 9 | `django__django-14311` | `django__django-13837` | django/django | django/utils/autoreload.py | django/utils/autoreload.py:get_child_arguments | 8.0 |
| 10 | `django__django-14771` | `django__django-13837` | django/django | django/utils/autoreload.py | django/utils/autoreload.py:get_child_arguments | 8.0 |
| 11 | `django__django-14999` | `django__django-15695` | django/django | django/db/migrations/operations/models.py | django/db/migrations/operations/models.py:database_forwards | 8.0 |
| 12 | `django__django-15161` | `django__django-13121` | django/django | django/db/models/expressions.py | django/db/models/expressions.py:get_group_by_cols | 8.0 |
| 13 | `django__django-15732` | `django__django-14434` | django/django | django/db/backends/base/schema.py | django/db/backends/base/schema.py:create_unique_name | 8.0 |
| 14 | `django__django-15957` | `django__django-16256` | django/django | django/db/models/fields/related_descriptors.py | django/db/models/fields/related_descriptors.py:Child | 8.0 |
| 15 | `django__django-16139` | `django__django-13741` | django/django | django/contrib/auth/forms.py | django/contrib/auth/forms.py:__init__ | 8.0 |
| 16 | `django__django-16145` | `django__django-13809` | django/django | django/core/management/commands/runserver.py | django/core/management/commands/runserver.py:inner_run | 8.0 |
| 17 | `django__django-16263` | `django__django-13121` | django/django | django/db/models/expressions.py | django/db/models/expressions.py:resolve_expression | 8.0 |
| 18 | `django__django-16263` | `django__django-17084` | django/django | django/db/models/sql/query.py | django/db/models/sql/query.py:get_aggregation | 8.0 |
| 19 | `django__django-16485` | `django__django-15863` | django/django | django/template/defaultfilters.py | django/template/defaultfilters.py:floatformat | 8.0 |
| 20 | `psf__requests-2931` | `psf__requests-5414` | psf/requests | requests/models.py | requests/models.py:prepare_url | 8.0 |
| 21 | `pydata__xarray-6461` | `pydata__xarray-4687` | pydata/xarray | xarray/core/computation.py | xarray/core/computation.py:where | 8.0 |
| 22 | `pydata__xarray-7229` | `pydata__xarray-4687` | pydata/xarray | xarray/core/computation.py | xarray/core/computation.py:where | 8.0 |
| 23 | `sphinx-doc__sphinx-7462` | `sphinx-doc__sphinx-9602` | sphinx-doc/sphinx | sphinx/domains/python.py | sphinx/domains/python.py:unparse | 8.0 |
| 24 | `sphinx-doc__sphinx-7985` | `sphinx-doc__sphinx-8269` | sphinx-doc/sphinx | sphinx/builders/linkcheck.py | sphinx/builders/linkcheck.py:check_uri | 8.0 |
| 25 | `sphinx-doc__sphinx-8475` | `sphinx-doc__sphinx-8269` | sphinx-doc/sphinx | sphinx/builders/linkcheck.py | sphinx/builders/linkcheck.py:check_uri | 8.0 |
| 26 | `sphinx-doc__sphinx-8551` | `sphinx-doc__sphinx-9230` | sphinx-doc/sphinx | sphinx/util/docfields.py | sphinx/util/docfields.py:transform | 8.0 |
| 27 | `sympy__sympy-24213` | `sympy__sympy-24066` | sympy/sympy | sympy/physics/units/unitsystem.py | sympy/physics/units/unitsystem.py:_collect_factor_and_dimension | 8.0 |
| 28 | `astropy__astropy-14598` | `astropy__astropy-14508` | astropy/astropy | astropy/io/fits/card.py | — | 5.0 |
| 29 | `astropy__astropy-8707` | `astropy__astropy-14508` | astropy/astropy | astropy/io/fits/card.py | — | 5.0 |
| 30 | `django__django-10554` | `django__django-11734` | django/django | django/db/models/sql/query.py | — | 5.0 |
