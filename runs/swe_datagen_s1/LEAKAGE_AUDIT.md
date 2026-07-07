# Cross-instance answer-leakage audit — swe_datagen_s1

**Question answered (verbatim from the ask):** *how similar are the trained
Verified-adjacent instances to the held-out 113?* — measured (scan) + adjudicated
(eyeball) with quoted evidence, in **measure-and-report mode**: the DROP-LIST below
is *listed, not executed*. The drop decision runs at pool freeze.

Generated `2026-07-07T06:15:08Z` · CPU-only · `.venv` python · 0.79 s ·
text-sim backend `tfidf_cosine_sklearn` (TF-IDF cosine).
Re-runnable verbatim at pool freeze: `runs/swe_datagen_s1/leakage_audit.py`.

---

## Standing rule (read this before quoting any number here)

- **The absolute numbers against the 113-ring are `train-adjacent`, not a
  promotion signal.** This audit measures *whether a trained keeper hands a
  held-out task its fix* (a firewall question). Similarity/overlap counts against
  the eval ring describe the training pool's shape; they are **not** evidence that
  the diffusion arm is better or worse.
- **Paired arm-vs-arm reads are the promotion currency.** A capability is promoted
  only on a raw/constrained *model-only* gain measured **arm vs arm on the same
  held-out slice** — never on an absolute 113-ring score. This audit exists to keep
  that slice clean, not to score on it.

The two audit layers are independent:
- **Instance-level holdout** is already airtight upstream — `expand_frontier.py`
  hash-asserts that no eval-ring id ever enters training.
- **This audit** is the *cross-instance* layer: a trained keeper whose patch (or
  whose issue text) is close enough to a held-out instance that solving the keeper
  could reveal the held-out fix.

---

## Integrity pins (all asserted PASS)

| pin | value | status |
|---|---|---|
| eval holdout | **113 distinct ids** (`inner5 ∪ tier0_20 ∪ tier1_100`) | rebuilt byte-identical to `expand_frontier.py` |
| holdout sha256 | `c56f473ad31e52bee0f85151562f4e2122e4815dfa3f1b776b15fe121e8d168e` | **pin-assert PASS** vs `.eval_holdout_sha256` |
| keepers snapshot | **114 rows** — 54 Verified-source + 60 SWE-Gym | sha256 `61e1f89092b34e6f…` |
| SWE-Gym disjoint-repo | 0 repo collisions with any holdout repo | **assert PASS** |

Because the 60 SWE-Gym keepers share **no repo** with any holdout id, the *entire*
cross-instance leakage surface is the **54 Verified-source keepers**.

---

## Topline (measured)

| metric | value |
|---|--:|
| same-repo keeper×holdout pairs | **720** |
| flagged pairs (severity-ranked) | **49** |
| keepers with a same-repo holdout neighbor | **53** |
| keepers with ANY file overlap vs holdout gold+test | **15** |
| keepers overlapping a holdout **GOLD** file | **15** |
| keepers with **function-level** symbol overlap | **1** |
| keepers with issue-text sim ≥ p95 (p95 = 0.16608, p99 = 0.365) | **18** |

**Reading of the topline:** all 15 file co-locations are *same-file / different-
function*; exactly **1** keeper (`sympy-12481`) shares a function-level symbol
(`Permutation.__new__`) with any holdout, and even that one is a co-location, not a
shared fix (adjudicated below). Highest text-sim pair (`xarray-3305 → 4687`, 0.648)
has **zero** file overlap — it is boilerplate MCVE bug-report similarity.

---

## Eyeball verdicts — the 16 highest-signal pairs (all with GOLD-file overlap)

Every keeper that touches a held-out **gold** file was read by hand against the
held-out gold+test patch. **Verdict legend:** `ANSWER_LEAK` (keeper reveals the
held-out fix — would be dropped) · `SUBSYSTEM_ADJACENT` (same file/subsystem, but
disjoint functions & logic — solving the keeper hands over nothing) · `BENIGN`
(repo-only, no shared gold file).

**Result: 16 of 16 = `SUBSYSTEM_ADJACENT`. Zero `ANSWER_LEAK`.**

| # | keeper | holdout | shared gold file | func∩ | text-sim | verdict |
|--:|---|---|---|---|--:|---|
| 1 | `sympy__sympy-12481` | `sympy__sympy-12489` | `combinatorics/permutations.py` | `Permutation.__new__` | 0.166 | SUBSYSTEM_ADJACENT |
| 2 | `pydata__xarray-6599` | `pydata__xarray-4687` | `xarray/core/computation.py` | — | 0.392 | SUBSYSTEM_ADJACENT |
| 3 | `pytest-dev__pytest-10081` | `pytest-dev__pytest-8399` | `src/_pytest/unittest.py` | — | 0.221 | SUBSYSTEM_ADJACENT |
| 4 | `pytest-dev__pytest-7236` | `pytest-dev__pytest-8399` | `src/_pytest/unittest.py` | — | 0.152 | SUBSYSTEM_ADJACENT |
| 5 | `scikit-learn__scikit-learn-11578` | `scikit-learn__scikit-learn-14087` | `sklearn/linear_model/logistic.py` | — | 0.142 | SUBSYSTEM_ADJACENT |
| 6 | `psf__requests-1142` | `psf__requests-5414` | `requests/models.py` | — | 0.139 | SUBSYSTEM_ADJACENT |
| 7 | `psf__requests-1724` | `psf__requests-5414` | `requests/models.py` | — | 0.123 | SUBSYSTEM_ADJACENT |
| 8 | `psf__requests-1921` | `psf__requests-5414` | `requests/models.py` | — | 0.118 | SUBSYSTEM_ADJACENT |
| 9 | `pytest-dev__pytest-6202` | `pytest-dev__pytest-8399` | `src/_pytest/python.py` | — | 0.112 | SUBSYSTEM_ADJACENT |
| 10 | `matplotlib__matplotlib-24149` | `matplotlib__matplotlib-13989` | `lib/matplotlib/axes/_axes.py` | — | 0.104 | SUBSYSTEM_ADJACENT |
| 11 | `sympy__sympy-12419` | `sympy__sympy-17630` | `matrices/expressions/matexpr.py` | — | 0.081 | SUBSYSTEM_ADJACENT |
| 12 | `sympy__sympy-11618` | `sympy__sympy-17655` | `sympy/geometry/point.py` | — | 0.079 | SUBSYSTEM_ADJACENT |
| 13 | `django__django-11095` | `django__django-12713` | `contrib/admin/options.py` | — | 0.078 | SUBSYSTEM_ADJACENT |
| 14 | `django__django-11133` | `django__django-13195` | `django/http/response.py` | — | 0.074 | SUBSYSTEM_ADJACENT |
| 15 | `pytest-dev__pytest-7982` | `pytest-dev__pytest-5840` | `src/_pytest/pathlib.py` | — | 0.060 | SUBSYSTEM_ADJACENT |
| 16 | `django__django-11095` | `django__django-16100` | `contrib/admin/options.py` | — | 0.032 | SUBSYSTEM_ADJACENT |

### Quoted evidence (load-bearing diffs)

**#1 `sympy-12481 → 12489` — the single function-level hit, and it is co-location only.**
Both patches touch `permutations.py::Permutation.__new__`, but in disjoint regions
solving unrelated problems. Keeper (12481, "constructor fails with non-disjoint
cycles") adds exactly one line — `if not is_cycle:` — deleting the `ValueError`
branch in the `has_dups` block so non-disjoint cycles apply left-to-right. Holdout
(12489, "Permutation can't be subclassed") is a subclassing refactor:
`@staticmethod def _af_new(perm)` → `@classmethod def _af_new(cls, perm)`,
`Basic.__new__(Perm, perm)` → `Basic.__new__(cls, perm)`, tail
`obj = Basic.__new__(cls, aform)` → `return cls._af_new(aform)`. **0 of the 34
distinct added lines in the holdout gold patch appear anywhere in the keeper row**
(patch, prompt, or 69-message trajectory); the keeper's added line does not appear
in the gold patch. `"subclass"`/`"classmethod"` appear 0 times in the trajectory.

**#2 `xarray-6599 → 4687` — same gold file, disjoint functions; hazard checked.**
Keeper edits only `_ensure_numeric` (polyval dtype): `if x.dtype.kind in "mM":` →
`== "M"` plus an `elif x.dtype.kind == "m": … astype("float64")`. Holdout edits only
`where()`: adds `keep_attrs=None`, `_get_keep_attrs(default=False)`, the decisive
`keep_attrs = lambda attrs, context: attrs[1]`. The 6599 base already contains the
merged 4687 fix, so all 61 trajectory messages were grepped: the trajectory **never
reveals** the `where()` signature change, the `_get_keep_attrs` defaulting, or the
`attrs[1]` line the `test_where_attrs` FAIL_TO_PASS requires. Elevated text-sim
(0.392) is boilerplate `xr.show_versions()` dumps + adjacent test insertion.

**#3/#4/#9 `pytest-10081, -7236, -6202 → 8399`.** All three share a pytest gold file
but touch disjoint functions. 10081/7236 edit `TestCaseFunction.runtest`/`.teardown`
(`--pdb` + class-level `unittest.skip` teardown timing); 6202 edits
`PyobjMixin.getmodpath` (`return s.replace(".[","[")` → `return s`). Held-out 8399
is a fixture-visibility rename in `_make_xunit_fixture` /
`_inject_setup_*_fixture` — `name=f"unittest_…"` → `f"_unittest_…"` (leading
underscore) plus four analogous renames in `src/_pytest/python.py`. None of the three
keepers contains any fixture-naming logic or the leading-underscore convention.

**#5 `sklearn-11578 → 14087`.** Keeper: one kwarg in `_log_reg_scoring_path` —
`LogisticRegression(fit_intercept=fit_intercept)` → `…, multi_class=multi_class)`.
Holdout: inside `LogisticRegressionCV.fit` — `if self.multi_class=='ovr':` →
`if multi_class=='ovr':` plus elasticnet-only `l1_ratio_` bookkeeping. Different
functions, different bug classes; 11578's 0.20-era base predates the elasticnet code
14087 fixes.

**#6/#7/#8 `requests-1142, -1724, -1921 → 5414`.** All three edit
`requests/models.py::PreparedRequest` but disjoint methods: 1142 `prepare_content_length`
(body/`Content-Length`), 1724 `prepare_method` (`to_native_string(...)`), 1921
`prepare_headers` (None-filtering). Held-out 5414 edits `prepare_url`:
`host.startswith(u'*')` → `host.startswith((u'*', u'.'))` guarding
`raise InvalidURL('URL has an invalid label.')`. No keeper mentions `InvalidURL`,
idna, hosts, or `startswith`; the idna host-label path did not exist in the keepers'
years-older base trees.

**#10 `matplotlib-24149 → 13989`.** Keeper edits only `cbook/__init__.py::_safe_first_finite`
(wraps `next(...)` in `try/except StopIteration`). Held-out edits one line in
`_axes.py::hist` (`hist_kwargs = dict(density=density)` → `hist_kwargs['density'] =
density`). Grep of the 82 KB trajectory: **0** occurrences of `hist_kwargs`,
`def hist`, `density`, `range=`; 68 of `_safe_first_finite`. The flag is pure
overlap on the ~8000-line `_axes.py` hub (keeper only appears there via a traceback).

**#11 `sympy-12419 → 17630`** (`matexpr.py`): keeper edits `Identity._entry`
(`KroneckerDelta(i, j)`); holdout edits module-level `_postprocessor` (`MatAdd`
dispatch). **#12 `sympy-11618 → 17655`** (`geometry/point.py`): keeper edits
`Point.distance` (zero-pad mismatched dims); holdout adds `Point.__rmul__`.
**#13/#16 `django-11095 → 12713 / 16100`** (`admin/options.py`): keeper adds
`get_inlines` hook; holdouts edit `formfield_for_manytomany` (widget guard) and
`changelist_view` (`transaction.atomic`) respectively — ~1400 lines apart, `func∩=0`.
**#14 `django-11133 → 13195`** (`http/response.py`): keeper adds `memoryview`
coercion to `make_bytes`; holdout rewrites `delete_cookie` (`samesite`). **#15
`pytest-7982 → 5840`** (`pathlib.py`): keeper flips `follow_symlinks=False→True` in
`visit()`; holdout deletes `unique_path()` and re-keys the conftest cache in
`config/__init__.py`. In every case the trained diff and its problem statement never
mention the identifiers the held-out fix and its FAIL_TO_PASS tests require.

*(Full per-pair evidence with exact diff hunks is preserved in the campaign's eyeball
record and `leakage_audit_report.json`.)*

---

## DROP-LIST (measure-and-report — LISTED, not executed)

> **`[]` — empty. No keeper was adjudicated `ANSWER_LEAK`.**

No trained keeper's patch or trajectory hands a held-out task its fix. The 15
gold-file co-locations are same-file / different-function and warrant only spot
review (done above), **not exclusion**. If the drop decision is executed at pool
freeze, **zero keepers** are removed on leakage grounds. Should any of the 54
Verified keepers be re-derived or the pool grow, re-run `leakage_audit.py` verbatim
and re-adjudicate any *new* GOLD-file-overlap pair before promotion.

---

## Frontier pre-screen (not-yet-collected Verified-adjacent ids)

A forward guard over the **333** not-yet-collected same-repo frontier ids vs holdout
gold files (gold-patch files only): **221** gold-file-overlap pairs across **114**
frontier ids — django-heavy, driven by shared hub files (`db/models/sql/query.py`,
`db/models/expressions.py`, `utils/autoreload.py`). Two django ids reach sev=10 (two
shared gold files each): `django-11138 → 13121` and `django-16032 → 11734`. These are
**not** keepers today — they are a re-examine list **if** they later become keepers,
not a current leak.

---

## Method & re-run

For every keeper × same-repo held-out id the audit computes: **(a)** file overlap of
`{keeper.final_patch ∪ keeper gold patch}` vs `{holdout gold ∪ holdout test}`;
**(b)** issue-text TF-IDF cosine (sklearn; token-Jaccard fallback); **(c)** enclosing
def/class overlap from unified-diff hunk headers, restricted to shared files.
Deterministic, CPU-only, no network/GPU. Rebuilds the 113-id holdout byte-identically
and asserts its sha256 before doing anything.

```
.venv/bin/python runs/swe_datagen_s1/leakage_audit.py
```

Artifacts: `leakage_audit_report.json` · `leakage_audit_report.md` ·
`leakage_audit_keepers_snapshot.jsonl` · this file.
